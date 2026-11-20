from __future__ import annotations

import collections
import copy
from typing import Any, Dict, MutableMapping, Optional, Sequence, Tuple

import torch


TensorState = Dict[str, torch.Tensor]


def clone_state_dict(state_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    cloned = collections.OrderedDict()
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def is_float_tensor(value: Any) -> bool:
    return torch.is_tensor(value) and torch.is_floating_point(value)


def compute_client_weights(sample_counts: Sequence[int], mode: str) -> Sequence[float]:
    normalized_mode = str(mode).lower()
    if not sample_counts:
        raise ValueError("sample_counts must not be empty")
    if normalized_mode == "uniform":
        return [1.0 / float(len(sample_counts))] * len(sample_counts)
    if normalized_mode != "sample_count":
        raise ValueError(f"Unsupported client weighting mode: {mode}")
    clipped_counts = [max(float(count), 0.0) for count in sample_counts]
    total = sum(clipped_counts)
    if total <= 0.0:
        raise ValueError("Cannot compute sample-count weights from all-zero sample counts")
    return [count / total for count in clipped_counts]


def weighted_average_state_dicts(
    state_dicts: Sequence[MutableMapping[str, Any]],
    weights: Sequence[float],
    fallback_state: Optional[MutableMapping[str, Any]] = None,
) -> MutableMapping[str, Any]:
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")
    if not state_dicts:
        raise ValueError("Cannot average an empty list of state_dicts")

    weights_tensor = torch.tensor([float(weight) for weight in weights], dtype=torch.float32)
    if not torch.isfinite(weights_tensor).all():
        raise ValueError("weights contain NaN or Inf")
    if weights_tensor.sum().item() <= 0.0:
        raise ValueError("weights must sum to a positive value")
    weights_tensor = weights_tensor / weights_tensor.sum()

    reference_keys = list(state_dicts[0].keys())
    reference_key_set = set(reference_keys)
    for state_idx, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict.keys()) != reference_key_set:
            raise ValueError(f"State dict keys do not match for weighted averaging at index {state_idx}")
    if fallback_state is not None and set(fallback_state.keys()) != reference_key_set:
        raise ValueError("Fallback state keys do not match the client state keys")

    averaged = collections.OrderedDict()
    for key in reference_keys:
        values = [state_dict[key] for state_dict in state_dicts]
        template = fallback_state[key] if fallback_state is not None else values[0]
        if is_float_tensor(template):
            accumulator = torch.zeros_like(template.detach(), dtype=torch.float32)
            for value, weight in zip(values, weights_tensor):
                if not is_float_tensor(value):
                    raise TypeError(f"Expected floating tensor for key '{key}'")
                accumulator = accumulator + value.detach().to(dtype=torch.float32) * float(weight.item())
            if not torch.isfinite(accumulator).all():
                raise ValueError(f"NaN/Inf detected while averaging key '{key}'")
            averaged[key] = accumulator.to(dtype=template.dtype)
        elif torch.is_tensor(template):
            averaged[key] = template.detach().clone()
        else:
            averaged[key] = copy.deepcopy(template)
    return averaged


def split_expert_and_non_expert_state(
    state_dict: MutableMapping[str, Any]
) -> Tuple[MutableMapping[str, Any], MutableMapping[str, Any]]:
    expert_state = collections.OrderedDict()
    non_expert_state = collections.OrderedDict()
    for key, value in state_dict.items():
        if ".switch_ffn.experts." in str(key):
            expert_state[key] = value
        else:
            non_expert_state[key] = value
    return expert_state, non_expert_state


def merge_state_dicts(
    base_state: MutableMapping[str, Any],
    expert_state: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    merged = clone_state_dict(base_state)
    for key, value in expert_state.items():
        merged[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
    return merged


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float().reshape(pred.size(0), -1)
    target = target.float().reshape(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()
