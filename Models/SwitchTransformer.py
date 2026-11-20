from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50


@dataclass
class SwitchRouterStats:
    router_probs: torch.Tensor
    expert_counts: torch.Tensor
    sample_expert_counts: torch.Tensor
    selected_experts: torch.Tensor
    aux_loss: torch.Tensor


class SimpleCNNTokenizer(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        stage1_channels = max(hidden_dim // 4, 32)
        stage2_channels = max(hidden_dim // 2, 64)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, stage1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stage1_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stage1_channels, stage1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stage1_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(stage1_channels, stage2_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stage2_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stage2_channels, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)


class PatchTokenizer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class SwitchFeedForward(nn.Module):
    """Top-1 routed feed-forward layer used by Switch Transformer blocks."""

    def __init__(self, hidden_dim, ffn_dim, num_experts, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts)
        self.router.server_temperature = nn.Parameter(torch.ones(()), requires_grad=False)
        self.router.server_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)
        self.router.server_temperature_min = 0.5
        self.router.server_temperature_max = 2.0
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x, server_adapt_config=None):
        batch_size, seq_len, hidden_dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, hidden_dim)

        router_logits = self.router(flat_x)
        if hasattr(self.router, "server_temperature") and self.router.server_temperature is not None:
            temp_min = float(
                getattr(
                    server_adapt_config,
                    "server_adapt_router_temp_min",
                    getattr(self.router, "server_temperature_min", 0.5),
                )
            )
            temp_max = float(
                getattr(
                    server_adapt_config,
                    "server_adapt_router_temp_max",
                    getattr(self.router, "server_temperature_max", 2.0),
                )
            )
            router_logits = router_logits / self.router.server_temperature.clamp(temp_min, temp_max)
        if hasattr(self.router, "server_bias") and self.router.server_bias is not None:
            router_logits = router_logits + self.router.server_bias
        router_probs = F.softmax(router_logits, dim=-1)
        top1_probs, top1_experts = torch.max(router_probs, dim=-1)

        flat_output = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_mask = top1_experts == expert_id
            if token_mask.any():
                flat_output[token_mask] = expert(flat_x[token_mask]) * top1_probs[token_mask].unsqueeze(-1)

        expert_mask = F.one_hot(top1_experts, num_classes=self.num_experts).float()
        sample_expert_counts = expert_mask.view(batch_size, seq_len, self.num_experts).sum(dim=1)
        density = expert_mask.mean(dim=0)
        density_proxy = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * density_proxy)

        stats = SwitchRouterStats(
            router_probs=router_probs.detach(),
            expert_counts=expert_mask.sum(dim=0).detach(),
            sample_expert_counts=sample_expert_counts.detach(),
            selected_experts=top1_experts.view(batch_size, seq_len).detach(),
            aux_loss=aux_loss,
        )
        stats.router_probs_raw = router_probs
        stats.router_logits_raw = router_logits
        stats.router_probs_tokens_raw = router_probs.view(batch_size, seq_len, self.num_experts)
        stats.router_logits_tokens_raw = router_logits.view(batch_size, seq_len, self.num_experts)
        return flat_output.reshape(batch_size, seq_len, hidden_dim), stats


class SwitchTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, num_experts, dropout=0.1, attention_dropout=0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.switch_ffn = SwitchFeedForward(hidden_dim, ffn_dim, num_experts, dropout)

    def forward(self, x, server_adapt_config=None, return_moe_inputs=False):
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attn_dropout(attn_output)

        ffn_residual_input = x
        ffn_input = self.ffn_norm(x)
        ffn_output, stats = self.switch_ffn(ffn_input, server_adapt_config=server_adapt_config)
        x = x + ffn_output
        if return_moe_inputs:
            stats.moe_input_raw = ffn_input
            stats.block_residual_input_raw = ffn_residual_input
            stats.block_output_raw = x
        return x, stats


class SwitchTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        image_size=32,
        patch_size=4,
        in_channels=3,
        backbone_name="patch_embed",
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
        num_experts=4,
        dropout=0.1,
        resnet_stem_mode="auto",
        vit_real_embed_dim=192,
        vit_real_depth=6,
        vit_real_num_heads=3,
        vit_real_mlp_ratio=4.0,
        vit_real_dropout=0.1,
        vit_real_attention_dropout=0.0,
        vit_real_use_cls_token=True,
        vit_real_use_learnable_pos_embed=True,
    ):
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.switch_num_experts = int(num_experts)
        self.resnet_stem_mode = str(resnet_stem_mode)
        self.backbone_family = self._canonical_backbone_name(self.backbone_name)
        self.backbone_config = self._resolve_backbone_config(
            backbone_name=self.backbone_name,
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            vit_real_embed_dim=vit_real_embed_dim,
            vit_real_depth=vit_real_depth,
            vit_real_num_heads=vit_real_num_heads,
            vit_real_mlp_ratio=vit_real_mlp_ratio,
            vit_real_dropout=vit_real_dropout,
            vit_real_attention_dropout=vit_real_attention_dropout,
            vit_real_use_cls_token=vit_real_use_cls_token,
            vit_real_use_learnable_pos_embed=vit_real_use_learnable_pos_embed,
        )
        self.hidden_dim = int(self.backbone_config["hidden_dim"])
        self.switch_layers = int(self.backbone_config["num_layers"])
        self.num_heads = int(self.backbone_config["num_heads"])
        self.ffn_dim = int(self.backbone_config["ffn_dim"])
        self.dropout_rate = float(self.backbone_config["dropout"])
        self.attention_dropout = float(self.backbone_config["attention_dropout"])
        self.patch_size = int(self.backbone_config["patch_size"])
        self.use_cls_token = bool(self.backbone_config["use_cls_token"])
        self.use_learnable_pos_embed = bool(self.backbone_config["use_learnable_pos_embed"])
        self.pretrained = False

        self.tokenizer = self._build_tokenizer(
            backbone_name=self.backbone_family,
            image_size=image_size,
            patch_size=self.patch_size,
            in_channels=in_channels,
            hidden_dim=self.hidden_dim,
            resnet_stem_mode=resnet_stem_mode,
        )
        num_patches = self._infer_num_patches(image_size=image_size, in_channels=in_channels)
        token_count = num_patches + (1 if self.use_cls_token else 0)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        else:
            self.register_parameter("cls_token", None)
        if self.use_learnable_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, token_count, self.hidden_dim))
        else:
            self.register_buffer("pos_embed", self._build_sincos_pos_embed(token_count, self.hidden_dim), persistent=False)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.blocks = nn.ModuleList(
            [
                SwitchTransformerBlock(
                    self.hidden_dim,
                    self.num_heads,
                    self.ffn_dim,
                    num_experts,
                    self.dropout_rate,
                    self.attention_dropout,
                )
                for _ in range(self.switch_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

        self._init_parameters()

    def _canonical_backbone_name(self, backbone_name: str) -> str:
        if backbone_name in {"patch_embed", "vit_tiny", "vit_small", "patch_embed_switch", "patch_tokenizer_switch"}:
            return "patch_embed_switch"
        return backbone_name

    def _resolve_backbone_config(
        self,
        backbone_name,
        image_size,
        patch_size,
        hidden_dim,
        num_layers,
        num_heads,
        ffn_dim,
        dropout,
        vit_real_embed_dim,
        vit_real_depth,
        vit_real_num_heads,
        vit_real_mlp_ratio,
        vit_real_dropout,
        vit_real_attention_dropout,
        vit_real_use_cls_token,
        vit_real_use_learnable_pos_embed,
    ):
        if backbone_name == "vit_tiny_real":
            embed_dim = int(vit_real_embed_dim)
            mlp_ratio = float(vit_real_mlp_ratio)
            real_ffn_dim = max(int(round(embed_dim * mlp_ratio)), embed_dim)
            return {
                "family": "vit_tiny_real",
                "patch_size": int(patch_size),
                "hidden_dim": embed_dim,
                "num_layers": int(vit_real_depth),
                "num_heads": int(vit_real_num_heads),
                "ffn_dim": real_ffn_dim,
                "dropout": float(vit_real_dropout),
                "attention_dropout": float(vit_real_attention_dropout),
                "use_cls_token": bool(vit_real_use_cls_token),
                "use_learnable_pos_embed": bool(vit_real_use_learnable_pos_embed),
                "mlp_ratio": mlp_ratio,
            }
        return {
            "family": self._canonical_backbone_name(backbone_name),
            "patch_size": int(patch_size),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "ffn_dim": int(ffn_dim),
            "dropout": float(dropout),
            "attention_dropout": float(dropout),
            "use_cls_token": True,
            "use_learnable_pos_embed": True,
            "mlp_ratio": float(ffn_dim) / float(max(int(hidden_dim), 1)),
        }

    def get_backbone_summary(self):
        return {
            "backbone_name": self.backbone_name,
            "backbone_family": self.backbone_family,
            "image_size": None,
            "patch_size": self.patch_size,
            "embed_dim": self.hidden_dim,
            "depth": self.switch_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": float(self.backbone_config["mlp_ratio"]),
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
            "use_cls_token": self.use_cls_token,
            "use_learnable_pos_embed": self.use_learnable_pos_embed,
            "pretrained": self.pretrained,
            "num_experts": self.switch_num_experts,
        }

    def _resolve_resnet_stem_mode(self, image_size, resnet_stem_mode):
        normalized_mode = str(resnet_stem_mode).lower()
        if normalized_mode == "auto":
            return "imagenet" if int(image_size) > 32 else "cifar"
        return normalized_mode

    def _build_resnet_tokenizer(self, backbone_builder, feature_dim, in_channels, hidden_dim, image_size, resnet_stem_mode):
        try:
            backbone = backbone_builder(weights=None)
        except TypeError:
            backbone = backbone_builder(pretrained=False)

        stem_mode = self._resolve_resnet_stem_mode(image_size=image_size, resnet_stem_mode=resnet_stem_mode)
        if stem_mode == "cifar":
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
        elif stem_mode == "imagenet":
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise ValueError(f"Unsupported resnet_stem_mode: {resnet_stem_mode}")

        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def _build_tokenizer(self, backbone_name, image_size, patch_size, in_channels, hidden_dim, resnet_stem_mode):
        if backbone_name in {"patch_embed_switch", "vit_tiny_real"}:
            if image_size % patch_size != 0:
                raise ValueError("image_size must be divisible by patch_size")
            return PatchTokenizer(in_channels=in_channels, embed_dim=hidden_dim, patch_size=patch_size)

        if backbone_name == "simplecnn":
            return SimpleCNNTokenizer(in_channels=in_channels, hidden_dim=hidden_dim)

        resnet_builders = {
            "resnet18": (resnet18, 512),
            "resnet34": (resnet34, 512),
            "resnet50": (resnet50, 2048),
        }
        if backbone_name in resnet_builders:
            backbone_builder, feature_dim = resnet_builders[backbone_name]
            return self._build_resnet_tokenizer(
                backbone_builder=backbone_builder,
                feature_dim=feature_dim,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                image_size=image_size,
                resnet_stem_mode=resnet_stem_mode,
            )

        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    def _infer_num_patches(self, image_size, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            token_map = self.tokenizer(dummy)
        return token_map.shape[-2] * token_map.shape[-1]

    def _build_sincos_pos_embed(self, token_count: int, dim: int) -> torch.Tensor:
        positions = torch.arange(token_count, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 1)))
        pe = torch.zeros(1, token_count, dim, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(positions * div_term)
        if dim % 2 == 0:
            pe[0, :, 1::2] = torch.cos(positions * div_term)
        else:
            pe[0, :, 1::2] = torch.cos(positions * div_term[:-1])
        return pe

    def _init_parameters(self):
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, return_aux=False, server_adapt_config=None, return_moe_inputs=False):
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x + self.pos_embed)

        aux_losses = []
        expert_counts = []
        sample_expert_counts = []
        selected_experts_by_layer = []
        router_probs_by_layer = []
        router_logits_by_layer = []
        router_probs_tokens_by_layer = []
        router_logits_tokens_by_layer = []
        moe_inputs_by_layer = []
        block_residual_inputs_by_layer = []
        for block in self.blocks:
            x, stats = block(x, server_adapt_config=server_adapt_config, return_moe_inputs=return_moe_inputs)
            aux_losses.append(stats.aux_loss)
            expert_counts.append(stats.expert_counts)
            sample_expert_counts.append(stats.sample_expert_counts)
            selected_experts_by_layer.append(stats.selected_experts)
            router_probs_by_layer.append(stats.router_probs_raw)
            router_logits_by_layer.append(stats.router_logits_raw)
            router_probs_tokens_by_layer.append(stats.router_probs_tokens_raw)
            router_logits_tokens_by_layer.append(stats.router_logits_tokens_raw)
            if return_moe_inputs:
                moe_inputs_by_layer.append(stats.moe_input_raw)
                block_residual_inputs_by_layer.append(stats.block_residual_input_raw)

        x = self.norm(x)
        pooled = x[:, 0] if self.use_cls_token else x.mean(dim=1)
        logits = self.classifier(pooled)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_tensor(0.0)
        expert_counts_by_layer = torch.stack(expert_counts) if expert_counts else None
        sample_expert_counts_by_layer = torch.stack(sample_expert_counts) if sample_expert_counts else None
        selected_experts = torch.stack(selected_experts_by_layer) if selected_experts_by_layer else None
        router_probs_flat = torch.cat(router_probs_by_layer, dim=0) if router_probs_by_layer else None
        router_stats = {
            "aux_loss": aux_loss,
            "expert_counts": expert_counts_by_layer.sum(dim=0) if expert_counts else None,
            "expert_counts_by_layer": expert_counts_by_layer,
            "sample_expert_counts_by_layer": sample_expert_counts_by_layer,
            "selected_experts": selected_experts.reshape(-1) if selected_experts is not None else None,
            "selected_experts_by_layer": selected_experts,
            "router_probs": router_probs_flat,
            "router_probs_by_layer": torch.stack(router_probs_by_layer) if router_probs_by_layer else None,
            "router_logits_by_layer": torch.stack(router_logits_by_layer) if router_logits_by_layer else None,
            "router_probs_tokens_by_layer": torch.stack(router_probs_tokens_by_layer) if router_probs_by_layer else None,
            "router_logits_tokens_by_layer": torch.stack(router_logits_tokens_by_layer) if router_logits_by_layer else None,
        }
        if return_moe_inputs:
            router_stats["moe_inputs_by_layer"] = torch.stack(moe_inputs_by_layer) if moe_inputs_by_layer else None
            router_stats["block_residual_inputs_by_layer"] = torch.stack(block_residual_inputs_by_layer) if block_residual_inputs_by_layer else None
        return logits, router_stats

    def forward_from_block_output(self, hidden, start_block_index=0, server_adapt_config=None):
        x = hidden
        for block in self.blocks[start_block_index:]:
            x, _ = block(x, server_adapt_config=server_adapt_config, return_moe_inputs=False)
        x = self.norm(x)
        pooled = x[:, 0] if self.use_cls_token else x.mean(dim=1)
        return self.classifier(pooled)
