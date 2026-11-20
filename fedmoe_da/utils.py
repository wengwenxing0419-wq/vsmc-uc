from __future__ import annotations

import collections
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


TensorState = Dict[str, torch.Tensor]
_EPS = 1e-12


def ordered_union(items: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def cosine_similarity_matrix(embeddings: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a rank-2 tensor")
    if embeddings.size(0) == 0:
        return embeddings.new_zeros((0, 0))
    normalized = F.normalize(embeddings, p=2, dim=1, eps=eps)
    return normalized @ normalized.transpose(0, 1)


def top_p_neighbors(
    similarity_row: torch.Tensor,
    top_p: int,
    min_similarity: float,
    keep_index: Optional[int] = None,
) -> torch.Tensor:
    if similarity_row.ndim != 1:
        raise ValueError("similarity_row must be rank-1")
    if similarity_row.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=similarity_row.device)

    top_k = max(1, min(int(top_p), int(similarity_row.numel())))
    values, indices = torch.topk(similarity_row, k=top_k, largest=True, sorted=False)
    selected = indices[values >= float(min_similarity)]
    if selected.numel() == 0:
        fallback_index = int(torch.argmax(similarity_row).item()) if keep_index is None else int(keep_index)
        selected = torch.tensor([fallback_index], dtype=torch.long, device=similarity_row.device)
    elif keep_index is not None and int(keep_index) not in selected.tolist():
        selected = torch.cat(
            [
                selected,
                torch.tensor([int(keep_index)], dtype=torch.long, device=similarity_row.device),
            ],
            dim=0,
        )
    return torch.unique(selected, sorted=False)


def softmax_over_indices(
    similarity_row: torch.Tensor,
    selected_indices: torch.Tensor,
    temperature: float,
    eps: float = _EPS,
) -> torch.Tensor:
    if selected_indices.numel() == 0:
        raise ValueError("selected_indices must not be empty")
    logits = similarity_row.index_select(0, selected_indices).to(dtype=torch.float32)
    weights = torch.softmax(logits / max(float(temperature), eps), dim=0)
    if not torch.isfinite(weights).all():
        weights = torch.full_like(weights, 1.0 / float(weights.numel()))
    return weights.to(dtype=torch.float64)


def blend_state_dicts(primary_state: TensorState, secondary_state: TensorState, mix: float) -> TensorState:
    alpha = min(max(float(mix), 0.0), 1.0)
    blended: TensorState = collections.OrderedDict()
    for key in ordered_union(list(primary_state.keys()) + list(secondary_state.keys())):
        primary_value = primary_state.get(key)
        secondary_value = secondary_state.get(key)
        if primary_value is None:
            blended[key] = secondary_value
            continue
        if secondary_value is None:
            blended[key] = primary_value
            continue
        if (
            torch.is_tensor(primary_value)
            and torch.is_tensor(secondary_value)
            and torch.is_floating_point(primary_value)
            and torch.is_floating_point(secondary_value)
            and primary_value.shape == secondary_value.shape
        ):
            secondary_aligned = secondary_value.detach().to(device=primary_value.device, dtype=primary_value.dtype)
            blended[key] = (1.0 - alpha) * primary_value.detach() + alpha * secondary_aligned
        else:
            blended[key] = primary_value
    return blended
