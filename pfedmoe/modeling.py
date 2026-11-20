from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50

from Models.SwitchTransformer import SwitchTransformerBlock


class SwitchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_var", torch.ones(self.num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("SwitchNorm1d expects [batch, features] input")

        layer_mean = x.mean(dim=1, keepdim=True)
        layer_var = x.var(dim=1, keepdim=True, unbiased=False)
        if self.training and x.size(0) > 1:
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_var = x.var(dim=0, keepdim=True, unbiased=False)
            self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * batch_mean.squeeze(0).detach())
            self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * batch_var.squeeze(0).detach())
        else:
            batch_mean = self.running_mean.view(1, -1)
            batch_var = self.running_var.view(1, -1)

        mean_weight = torch.softmax(self.mean_weight, dim=0)
        var_weight = torch.softmax(self.var_weight, dim=0)
        mean = mean_weight[0] * batch_mean + mean_weight[1] * layer_mean
        var = var_weight[0] * batch_var + var_weight[1] * layer_var
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.weight + self.bias


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.dim() == 2 and x.size(0) < 2:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(x)


class PFedMoEGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = SwitchNorm1d(self.input_dim)
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.act = nn.Sigmoid()
        self.bn1 = SafeBatchNorm1d(self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 2)
        self.bn2 = SafeBatchNorm1d(2)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=-1)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels),
            ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class TinyResNetTokenizer(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, stage_channels, block_repeats):
        super().__init__()
        self.stem = ConvBNAct(in_channels, stage_channels[0], kernel_size=3, stride=1)
        layers = []
        current_channels = stage_channels[0]
        for stage_idx, (out_channels, repeats) in enumerate(zip(stage_channels, block_repeats)):
            for block_idx in range(repeats):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                layers.append(TinyResidualBlock(current_channels, out_channels, stride=stride))
                current_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Conv2d(current_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        return self.proj(x)


class TinyMobileNetTokenizer(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, stride=1),
            DepthwiseSeparableConv(32, 48, stride=1),
            DepthwiseSeparableConv(48, 64, stride=2),
            DepthwiseSeparableConv(64, 96, stride=1),
            DepthwiseSeparableConv(96, hidden_dim, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwitchTransformerFeatureExtractor(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        backbone_name: str,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        dropout: float,
        in_channels: int = 3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.hidden_dim = int(hidden_dim)
        self.tokenizer = self._build_tokenizer(
            backbone_name=backbone_name,
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )
        num_patches = self._infer_num_patches(image_size=image_size, in_channels=in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                SwitchTransformerBlock(hidden_dim, num_heads, ffn_dim, num_experts, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self._init_parameters()

    def _build_tokenizer(self, backbone_name, image_size, patch_size, in_channels, hidden_dim):
        if backbone_name == "patch_embed":
            if image_size % patch_size != 0:
                raise ValueError("image_size must be divisible by patch_size")
            return nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        if backbone_name == "cnn_small":
            return nn.Sequential(
                ConvBNAct(in_channels, 32, kernel_size=3, stride=1),
                ConvBNAct(32, 64, kernel_size=3, stride=2),
                ConvBNAct(64, 96, kernel_size=3, stride=1),
                ConvBNAct(96, hidden_dim, kernel_size=1, stride=1),
            )
        if backbone_name == "resnet8":
            return TinyResNetTokenizer(in_channels, hidden_dim, stage_channels=(32, 64, 96), block_repeats=(1, 1, 1))
        if backbone_name == "resnet10":
            return TinyResNetTokenizer(in_channels, hidden_dim, stage_channels=(32, 64, 96, 128), block_repeats=(1, 1, 1, 1))
        if backbone_name == "mobilenet_tiny":
            return TinyMobileNetTokenizer(in_channels, hidden_dim)

        resnet_builders = {
            "resnet18": (resnet18, 512),
            "resnet34": (resnet34, 512),
            "resnet50": (resnet50, 2048),
        }
        if backbone_name in resnet_builders:
            backbone_builder, feature_dim = resnet_builders[backbone_name]
            try:
                backbone = backbone_builder(weights=None)
            except TypeError:
                backbone = backbone_builder(pretrained=False)
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            backbone.maxpool = nn.Identity()
            return nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                nn.Conv2d(feature_dim, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    def _infer_num_patches(self, image_size, in_channels):
        was_training = self.tokenizer.training
        self.tokenizer.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            token_map = self.tokenizer(dummy)
        self.tokenizer.train(was_training)
        return token_map.shape[-2] * token_map.shape[-1]

    def _init_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x + self.pos_embed)
        for block in self.blocks:
            x, _ = block(x, server_adapt_config=None, return_moe_inputs=False)
        x = self.norm(x)
        return x[:, 0]


class PFedMoEPersonalModel(nn.Module):
    def __init__(
        self,
        global_extractor: nn.Module,
        local_extractor: nn.Module,
        gate: nn.Module,
        prediction_head: nn.Module,
        global_aux_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.global_extractor = global_extractor
        self.local_extractor = local_extractor
        self.gate = gate
        self.prediction_head = prediction_head
        self.global_aux_head = global_aux_head

    def forward(self, x: torch.Tensor, gate_override: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        global_features = self.global_extractor(x)
        local_features = self.local_extractor(x)
        gate_input = torch.cat([global_features, local_features], dim=-1)
        gate_weights = self.gate(gate_input) if gate_override is None else gate_override
        if gate_weights.dim() == 1:
            gate_weights = gate_weights.view(1, -1).expand(global_features.size(0), -1)
        gate_weights = gate_weights.to(device=global_features.device, dtype=global_features.dtype)
        mixed_features = (
            gate_weights[:, 0:1] * global_features
            + gate_weights[:, 1:2] * local_features
        )
        logits = self.prediction_head(mixed_features)
        global_logits = None if self.global_aux_head is None else self.global_aux_head(global_features)
        return {
            "logits": logits,
            "global_logits": global_logits,
            "global_features": global_features,
            "local_features": local_features,
            "gate_weights": gate_weights,
            "mixed_features": mixed_features,
        }


def freeze_module(module: nn.Module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def clone_state_dict(state_dict):
    return {
        name: tensor.detach().clone().cpu() if torch.is_tensor(tensor) else tensor
        for name, tensor in state_dict.items()
    }


def count_parameters(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))
