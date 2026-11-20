from __future__ import annotations

import collections
import copy
import re
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import torch


TensorState = Dict[str, torch.Tensor]
_EXPERT_CONTAINER_TOKENS = {"expert", "experts", "ffn_expert", "ffn_experts"}
_EXPERT_LIKE_KEYWORDS = ("expert", "experts", "moe", "switch", "ffn_expert")
_LAYER_INDEX_PATTERNS = (
    re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)encoder\.blocks\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)switch_layers\.(\d+)(?:\.|$)"),
)


def clone_state_dict(state_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep copy a PyTorch-like state_dict."""
    cloned = collections.OrderedDict()
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def is_float_tensor(tensor: Any) -> bool:
    """Return True when the value is a floating-point tensor."""
    return torch.is_tensor(tensor) and torch.is_floating_point(tensor)


def ordered_union(state_dicts: Iterable[MutableMapping[str, Any]]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for state_dict in state_dicts:
        for key in state_dict.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def compute_client_weights(sample_counts: Sequence[int], mode: str) -> List[float]:
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


def normalize_signature(signature: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if signature.numel() == 0:
        return signature.detach().clone().float()
    normalized = signature.detach().clone().float()
    norm = normalized.norm(p=2)
    if torch.isnan(norm) or torch.isinf(norm):
        raise ValueError("Encountered NaN/Inf while normalizing expert signature")
    if norm.item() <= eps:
        return normalized
    return normalized / norm.clamp_min(eps)


def _validate_matching_key_sets(
    state_dicts: Sequence[MutableMapping[str, Any]],
    fallback_state: Optional[MutableMapping[str, Any]] = None,
) -> List[str]:
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    key_order = ordered_union([fallback_state] + list(state_dicts) if fallback_state is not None else state_dicts)
    first_keys = list(state_dicts[0].keys())
    first_key_set = set(first_keys)
    for state_idx, state_dict in enumerate(state_dicts[1:], start=1):
        current_keys = set(state_dict.keys())
        if current_keys != first_key_set:
            missing = sorted(first_key_set - current_keys)
            extra = sorted(current_keys - first_key_set)
            raise ValueError(
                "State dict key mismatch during weighted averaging: "
                f"state 0 vs state {state_idx}; missing={missing[:8]} extra={extra[:8]}"
            )
    if fallback_state is not None:
        fallback_keys = set(fallback_state.keys())
        if fallback_keys != first_key_set:
            missing = sorted(first_key_set - fallback_keys)
            extra = sorted(fallback_keys - first_key_set)
            raise ValueError(
                "Fallback state key mismatch during weighted averaging: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
    return key_order


def weighted_average_state_dicts(
    state_dicts: Sequence[MutableMapping[str, Any]],
    weights: Sequence[float],
    fallback_state: Optional[MutableMapping[str, Any]] = None,
) -> MutableMapping[str, Any]:
    """
    Weighted FedAvg over aligned state_dicts.

    Floating tensors are averaged.
    Non-floating tensors keep the fallback value when provided, otherwise the first client value.
    """
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")
    if not state_dicts:
        raise ValueError("Cannot average an empty list of state_dicts")

    weights_tensor = torch.tensor([float(weight) for weight in weights], dtype=torch.float32)
    if weights_tensor.numel() == 0:
        raise ValueError("weights must not be empty")
    if not torch.isfinite(weights_tensor).all():
        raise ValueError("weights contain NaN or Inf")
    if weights_tensor.sum().item() <= 0.0:
        raise ValueError("weights must sum to a positive value")
    weights_tensor = weights_tensor / weights_tensor.sum()

    key_order = _validate_matching_key_sets(state_dicts, fallback_state=fallback_state)
    averaged = collections.OrderedDict()

    for key in key_order:
        values = [state_dict[key] for state_dict in state_dicts]
        template_value = fallback_state[key] if fallback_state is not None else values[0]

        if is_float_tensor(template_value):
            accumulator = torch.zeros_like(template_value.detach(), dtype=torch.float32)
            reference_shape = tuple(template_value.shape)
            for value, weight in zip(values, weights_tensor):
                if not torch.is_tensor(value):
                    raise TypeError(f"Expected tensor value for key '{key}', got {type(value)!r}")
                if tuple(value.shape) != reference_shape:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': expected {reference_shape}, got {tuple(value.shape)}"
                    )
                if not torch.is_floating_point(value):
                    raise TypeError(f"Expected floating tensor for key '{key}', got dtype {value.dtype}")
                accumulator = accumulator + value.detach().to(dtype=torch.float32) * float(weight.item())
            if not torch.isfinite(accumulator).all():
                raise ValueError(f"NaN/Inf detected while averaging key '{key}'")
            averaged[key] = accumulator.to(dtype=template_value.dtype)
            continue

        if torch.is_tensor(template_value):
            averaged[key] = template_value.detach().clone()
        else:
            averaged[key] = copy.deepcopy(template_value)

    return averaged


def _is_router_or_gate_key(key: str) -> bool:
    lower_key = str(key).lower()
    return "router" in lower_key or "gate" in lower_key


def _looks_like_expert_key(key: str) -> bool:
    lower_key = str(key).lower()
    if _is_router_or_gate_key(lower_key):
        return False
    return any(keyword in lower_key for keyword in _EXPERT_LIKE_KEYWORDS)


def parse_expert_key(key: str) -> Optional[Dict[str, Any]]:
    tokens = str(key).split(".")
    for index, token in enumerate(tokens[:-1]):
        normalized = token.lower()
        if normalized not in _EXPERT_CONTAINER_TOKENS:
            continue
        if index + 1 >= len(tokens) or not tokens[index + 1].isdigit():
            continue
        layer_scope = ".".join(tokens[:index])
        return {
            "layer_scope": layer_scope,
            "expert_id": int(tokens[index + 1]),
            "container_token": token,
            "expert_token_index": index + 1,
            "tokens": tokens,
        }
    return None


def infer_layer_index_from_scope(layer_scope: str) -> int:
    for pattern in _LAYER_INDEX_PATTERNS:
        match = pattern.search(str(layer_scope))
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"Unable to infer MoE layer index from scope '{layer_scope}'")


def get_expert_param_keys(state_dict: MutableMapping[str, Any]) -> List[str]:
    """
    Identify expert parameter keys using robust expert-id parsing.
    Router and gate parameters are excluded.
    """
    expert_keys: List[str] = []
    suspicious_keys: List[str] = []
    for key in state_dict.keys():
        parsed = parse_expert_key(key)
        if parsed is not None:
            expert_keys.append(key)
            continue
        if _looks_like_expert_key(key):
            suspicious_keys.append(key)
    if suspicious_keys:
        raise ValueError(
            "Unable to recognize expert ids for the following expert-like keys: "
            + ", ".join(sorted(suspicious_keys))
        )
    return expert_keys


def group_expert_keys_by_moe_layer(state_dict: MutableMapping[str, Any]) -> "collections.OrderedDict[str, Dict[str, Any]]":
    layer_groups: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()
    suspicious_keys: List[str] = []

    for key in state_dict.keys():
        parsed = parse_expert_key(key)
        if parsed is None:
            if _looks_like_expert_key(key):
                suspicious_keys.append(key)
            continue

        layer_scope = str(parsed["layer_scope"])
        expert_id = int(parsed["expert_id"])
        group = layer_groups.setdefault(
            layer_scope,
            {
                "layer_scope": layer_scope,
                "layer_index": None,
                "container_token": parsed["container_token"],
                "key_groups": collections.OrderedDict(),
            },
        )
        if group["layer_index"] is None:
            try:
                group["layer_index"] = infer_layer_index_from_scope(layer_scope)
            except ValueError:
                group["layer_index"] = len(layer_groups) - 1
        group["key_groups"].setdefault(expert_id, [])
        group["key_groups"][expert_id].append(key)

    if suspicious_keys:
        raise ValueError(
            "Found expert-like parameter names that could not be grouped by MoE layer/expert id: "
            + ", ".join(sorted(suspicious_keys))
        )

    normalized_groups: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()
    for layer_scope, layer_group in layer_groups.items():
        sorted_key_groups = collections.OrderedDict()
        for expert_id in sorted(layer_group["key_groups"].keys()):
            sorted_key_groups[expert_id] = sorted(layer_group["key_groups"][expert_id])
        normalized_group = dict(layer_group)
        normalized_group["key_groups"] = sorted_key_groups
        normalized_groups[layer_scope] = normalized_group
    return normalized_groups


def group_expert_keys_by_expert_id(state_dict: MutableMapping[str, Any]) -> Dict[int, List[str]]:
    """
    Group expert keys by expert id for single-layer MoE models.

    When multiple MoE layers are present, this function raises to avoid mixing experts
    from different layers in one mapping.
    """
    layer_groups = group_expert_keys_by_moe_layer(state_dict)
    if not layer_groups:
        return {}
    if len(layer_groups) > 1:
        raise ValueError(
            "Multiple MoE layers detected; use group_expert_keys_by_moe_layer to avoid cross-layer expert mixing"
        )
    only_group = next(iter(layer_groups.values()))
    return copy.deepcopy(only_group["key_groups"])


def split_expert_and_non_expert_state(
    state_dict: MutableMapping[str, Any]
) -> Tuple[MutableMapping[str, Any], MutableMapping[str, Any]]:
    """
    Split a full model state into expert and non-expert sub-states.
    """
    expert_keys = set(get_expert_param_keys(state_dict))
    expert_state = collections.OrderedDict()
    non_expert_state = collections.OrderedDict()
    for key, value in state_dict.items():
        if key in expert_keys:
            expert_state[key] = value
        else:
            non_expert_state[key] = value
    return expert_state, non_expert_state


def _replace_expert_id_in_key(key: str, expected_from_expert_id: int, to_expert_id: int) -> str:
    tokens = str(key).split(".")
    for index, token in enumerate(tokens[:-1]):
        if token.lower() not in _EXPERT_CONTAINER_TOKENS:
            continue
        if index + 1 >= len(tokens) or not tokens[index + 1].isdigit():
            continue
        current_expert_id = int(tokens[index + 1])
        if current_expert_id != int(expected_from_expert_id):
            raise ValueError(
                f"Expert-id mismatch while remapping key '{key}': expected {expected_from_expert_id}, got {current_expert_id}"
            )
        remapped_tokens = list(tokens)
        remapped_tokens[index + 1] = str(int(to_expert_id))
        return ".".join(remapped_tokens)
    raise ValueError(f"Unable to replace expert id in key '{key}'")


def remap_expert_params(
    client_state: MutableMapping[str, Any],
    from_expert_id: int,
    to_expert_id: int,
    expert_key_groups: Dict[int, List[str]],
) -> MutableMapping[str, Any]:
    """
    Extract one local expert and rename it to the target global expert id.
    """
    if from_expert_id not in expert_key_groups:
        raise KeyError(f"Expert id {from_expert_id} is missing from expert_key_groups")
    remapped = collections.OrderedDict()
    for key in expert_key_groups[from_expert_id]:
        if key not in client_state:
            raise KeyError(f"Key '{key}' is missing from the provided client_state")
        remapped_key = _replace_expert_id_in_key(key, expected_from_expert_id=from_expert_id, to_expert_id=to_expert_id)
        remapped[remapped_key] = clone_state_dict({key: client_state[key]})[key]
    return remapped


def load_partial_state_dict(model, partial_state: MutableMapping[str, Any]) -> None:
    """
    Load only the provided keys into the model.
    """
    current_state = model.state_dict()
    missing = [key for key in partial_state.keys() if key not in current_state]
    if missing:
        raise KeyError(f"Cannot load unknown state_dict keys: {missing[:8]}")
    current_state.update(clone_state_dict(partial_state))
    model.load_state_dict(current_state, strict=True)
