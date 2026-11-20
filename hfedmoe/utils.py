from __future__ import annotations

import collections
import json
from typing import Dict, Iterable, List, Mapping, Sequence

import torch


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> collections.OrderedDict:
    return collections.OrderedDict(
        (name, tensor.detach().clone().cpu() if torch.is_tensor(tensor) else tensor)
        for name, tensor in state_dict.items()
    )


def normalize_weights(weights: Sequence[float]) -> List[float]:
    cleaned = [max(float(weight), 0.0) for weight in weights]
    total = sum(cleaned)
    if total <= 0.0:
        if not cleaned:
            return []
        return [1.0 / float(len(cleaned))] * len(cleaned)
    return [weight / total for weight in cleaned]


def average_state_dicts(state_dicts: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]):
    if not state_dicts:
        raise ValueError("Cannot average empty state dicts")
    normalized = normalize_weights(weights)
    result = collections.OrderedDict()
    for state_dict, weight in zip(state_dicts, normalized):
        for name, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                continue
            tensor_cpu = tensor.detach().to(dtype=torch.float32, device="cpu")
            if name not in result:
                result[name] = tensor_cpu * float(weight)
            else:
                result[name] += tensor_cpu * float(weight)
    return result


def merge_histograms(histograms: Iterable[Mapping[int, int]]) -> Dict[int, int]:
    merged: Dict[int, int] = {}
    for histogram in histograms:
        for expert_id, count in histogram.items():
            merged[int(expert_id)] = merged.get(int(expert_id), 0) + int(count)
    return merged


def state_dict_l2_norm(state_dict: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in state_dict.values():
        if torch.is_tensor(tensor):
            total += float(torch.sum(tensor.detach().float().pow(2)).item())
    return total ** 0.5


def histogram_to_json(histogram: Mapping[int, int]) -> str:
    return json.dumps({str(int(key)): int(value) for key, value in sorted(histogram.items())}, sort_keys=True)
