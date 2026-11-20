from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


class CommonExpertWrapper(nn.Module):
    def __init__(self, model: nn.Module, device: torch.device, cache_embeddings: bool = True):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.cache_embeddings = bool(cache_embeddings)
        self._embedding_cache: Dict[Tuple[int, str], torch.Tensor] = {}
        self._freeze_model()

    @property
    def embedding_dim(self) -> int:
        classifier = getattr(self.model, "classifier", None)
        if classifier is not None and hasattr(classifier, "in_features"):
            return int(classifier.in_features)
        raise AttributeError("Unable to infer common expert embedding dimension")

    def _freeze_model(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def clear_cache(self):
        self._embedding_cache.clear()

    def _extract_switch_transformer_features(self, x: torch.Tensor) -> torch.Tensor:
        token_map = self.model.tokenizer(x)
        hidden = token_map.flatten(2).transpose(1, 2)
        cls_tokens = self.model.cls_token.expand(hidden.size(0), -1, -1)
        hidden = torch.cat((cls_tokens, hidden), dim=1)
        hidden = self.model.dropout(hidden + self.model.pos_embed)
        for block in self.model.blocks:
            hidden, _ = block(hidden, server_adapt_config=None, return_moe_inputs=False)
        hidden = self.model.norm(hidden)
        return hidden[:, 0]

    def _extract_generic_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(x)
            if isinstance(features, (tuple, list)):
                features = features[0]
            if features.dim() > 2:
                features = torch.flatten(features, start_dim=1)
            return features

        captured = {}

        def hook(_, inputs, __):
            if inputs:
                captured["features"] = inputs[0].detach()

        classifier = getattr(self.model, "classifier", None)
        if classifier is None:
            raise AttributeError("CommonExpertWrapper could not find a feature extraction path")
        handle = classifier.register_forward_hook(hook)
        try:
            outputs = self.model(x)
            if isinstance(outputs, (tuple, list)):
                _ = outputs[0]
        finally:
            handle.remove()
        features = captured.get("features")
        if features is None:
            raise RuntimeError("Failed to capture classifier input features from the common expert")
        return features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            if all(hasattr(self.model, attr) for attr in ("tokenizer", "cls_token", "blocks", "norm")):
                features = self._extract_switch_transformer_features(x)
            else:
                features = self._extract_generic_features(x)
        return features.detach()

    def get_cached_client_embeddings(
        self,
        client_id: int,
        split: str,
        dataset,
        batch_size: int,
    ) -> torch.Tensor:
        cache_key = (int(client_id), str(split))
        if self.cache_embeddings and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        loader = DataLoader(dataset, batch_size=max(int(batch_size), 1), shuffle=False)
        outputs = []
        for inputs, _ in loader:
            outputs.append(self.extract_features(inputs).cpu())
        embeddings = torch.cat(outputs, dim=0) if outputs else torch.empty(0, self.embedding_dim)
        if self.cache_embeddings:
            self._embedding_cache[cache_key] = embeddings
        return embeddings
