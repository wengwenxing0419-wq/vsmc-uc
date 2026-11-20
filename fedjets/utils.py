from __future__ import annotations

import collections
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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


def average_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
) -> collections.OrderedDict:
    if not state_dicts:
        raise ValueError("Cannot average an empty state dict list")
    normalized = normalize_weights(weights)
    result = collections.OrderedDict()
    for state, weight in zip(state_dicts, normalized):
        for name, tensor in state.items():
            if not torch.is_tensor(tensor):
                continue
            tensor_cpu = tensor.detach().to(dtype=torch.float32, device="cpu")
            if name not in result:
                result[name] = tensor_cpu * float(weight)
            else:
                result[name] += tensor_cpu * float(weight)
    return result


def compute_client_weights(sample_counts: Sequence[int], mode: str) -> List[float]:
    normalized_mode = str(mode).lower()
    if not sample_counts:
        return []
    if normalized_mode == "uniform":
        return [1.0 / float(len(sample_counts))] * len(sample_counts)
    return normalize_weights(sample_counts)


def select_topk_experts(gate, embeddings: torch.Tensor, k: int, device: torch.device | None = None) -> Tuple[List[int], torch.Tensor]:
    if embeddings.numel() == 0:
        raise ValueError("Embeddings must be non-empty for expert selection")
    target_device = device or next(gate.parameters()).device
    gate_was_training = gate.training
    gate.eval()
    with torch.no_grad():
        probs = gate(embeddings.to(target_device))
        client_score = probs.sum(dim=0)
        topk = torch.topk(client_score, k=min(max(int(k), 1), client_score.numel()))
    if gate_was_training:
        gate.train()
    return [int(index) for index in topk.indices.detach().cpu().tolist()], client_score.detach().cpu()


def select_anchor_clients(
    client_ids: Sequence[int],
    num_anchor_clients: int,
    num_experts: int,
) -> Tuple[List[int], Dict[int, int]]:
    ordered = sorted(int(client_id) for client_id in client_ids)
    anchors = ordered[: min(max(int(num_anchor_clients), 0), len(ordered))]
    assignment = {
        int(client_id): int(idx % max(int(num_experts), 1))
        for idx, client_id in enumerate(anchors)
    }
    return anchors, assignment


def sample_client_subset(
    client_ids: Sequence[int],
    sample_size: int,
    seed: int,
) -> List[int]:
    ordered = sorted(int(client_id) for client_id in client_ids)
    if sample_size <= 0 or sample_size >= len(ordered):
        return ordered
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(ordered), generator=generator)[:sample_size].tolist()
    return sorted(ordered[idx] for idx in indices)


def merge_histograms(histograms: Iterable[Mapping[int, int]]) -> Dict[int, int]:
    merged: Dict[int, int] = {}
    for histogram in histograms:
        for expert_id, count in histogram.items():
            merged[int(expert_id)] = merged.get(int(expert_id), 0) + int(count)
    return merged
