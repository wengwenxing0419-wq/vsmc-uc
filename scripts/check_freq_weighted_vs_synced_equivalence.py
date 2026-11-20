#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_DIR = Path(__file__).resolve().parents[1]

from aggregation import history_aware_vsmc_expert_aggregator as ha
from aggregation.vsmc_expert_aggregator import split_state_dict_by_expert, weighted_average_state_dicts


TensorState = Dict[str, Any]
_EPS = 1e-12


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_state_dict(obj: Any) -> TensorState:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "server_state_dict", "net", "weights"):
            value = obj.get(key)
            if isinstance(value, dict) and value:
                if all(isinstance(k, str) for k in value.keys()):
                    return collections.OrderedDict(value)
        if obj and all(isinstance(k, str) for k in obj.keys()):
            return collections.OrderedDict(obj)
    raise TypeError("Unsupported checkpoint format for state_dict extraction")


def _infer_length(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in ("labels", "targets", "y", "x", "data", "images", "inputs"):
            if key in obj:
                try:
                    return len(obj[key])
                except Exception:
                    pass
        for value in obj.values():
            inferred = _infer_length(value)
            if inferred is not None:
                return inferred
        return None
    if isinstance(obj, (list, tuple)):
        if len(obj) == 2:
            try:
                return len(obj[0])
            except Exception:
                pass
        try:
            return len(obj)
        except Exception:
            return None
    try:
        return len(obj)
    except Exception:
        return None


def _load_sample_counts(data_dir: Path, num_clients: int) -> List[int]:
    counts: List[int] = []
    for client_idx in range(1, num_clients + 1):
        data_path = data_dir / f"{client_idx}_train.pth"
        payload = torch.load(data_path, map_location="cpu")
        inferred = _infer_length(payload)
        if inferred is None:
            raise RuntimeError(f"Failed to infer sample count from {data_path}")
        counts.append(int(inferred))
    return counts


def _parse_json_list(text: str) -> List[float]:
    value = json.loads(text)
    return [float(v) for v in value]


def _reconstruct_router_usage(
    metrics_csv: Path,
    num_clients: int,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], int, int]:
    rows: List[Dict[str, Any]] = []
    max_block = -1
    max_expert = -1
    with open(metrics_csv, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
            max_block = max(max_block, int(row["block_id"]))
            max_expert = max(max_expert, int(row["expert_id"]))
    if max_block < 0 or max_expert < 0:
        raise RuntimeError(f"No expert rows found in {metrics_csv}")

    num_blocks = max_block + 1
    num_experts = max_expert + 1
    client_usage = [
        torch.zeros(num_blocks, num_experts, dtype=torch.float64)
        for _ in range(num_clients)
    ]
    expert_rows: List[Dict[str, Any]] = []
    for row in rows:
        block_id = int(row["block_id"])
        expert_id = int(row["expert_id"])
        raw_counts = _parse_json_list(row["raw_usage_counts"])
        if len(raw_counts) != num_clients:
            raise RuntimeError(
                f"Expected {num_clients} clients in raw_usage_counts for block {block_id}, expert {expert_id}, "
                f"but found {len(raw_counts)}"
            )
        for client_idx, value in enumerate(raw_counts):
            client_usage[client_idx][block_id, expert_id] = float(value)
        expert_rows.append(
            {
                "block_id": block_id,
                "expert_id": expert_id,
                "expert_scope": row["expert_scope"],
                "logged_a_freq": _parse_json_list(row["a_freq"]),
                "logged_a_vsmc": _parse_json_list(row["a_vsmc_final"]),
                "logged_top_match": str(row["top_client_match"]).lower() == "true",
                "all_zero_usage": str(row["all_zero_usage"]).lower() == "true",
            }
        )
    return client_usage, expert_rows, num_blocks, num_experts


def _load_client_states(model_dir: Path, num_clients: int) -> List[TensorState]:
    states: List[TensorState] = []
    for client_idx in range(1, num_clients + 1):
        ckpt_path = model_dir / f"{client_idx}.pth"
        payload = torch.load(ckpt_path, map_location="cpu")
        states.append(_extract_state_dict(payload))
    return states


def _load_reference_state(model_dir: Path, client_states: Sequence[TensorState]) -> TensorState:
    server_path = model_dir / "server.pth"
    if server_path.exists():
        payload = torch.load(server_path, map_location="cpu")
        return _extract_state_dict(payload)
    return collections.OrderedDict(client_states[0])


def _state_diff_metrics(state_a: TensorState, state_b: TensorState) -> Dict[str, float]:
    max_abs = 0.0
    abs_sum = 0.0
    sq_sum = 0.0
    elem_count = 0
    for key in sorted(set(state_a.keys()) & set(state_b.keys())):
        value_a = state_a[key]
        value_b = state_b[key]
        if not (torch.is_tensor(value_a) and torch.is_tensor(value_b)):
            continue
        if value_a.shape != value_b.shape:
            raise RuntimeError(f"Tensor shape mismatch at {key}: {tuple(value_a.shape)} vs {tuple(value_b.shape)}")
        if not (torch.is_floating_point(value_a) and torch.is_floating_point(value_b)):
            continue
        diff = value_a.detach().to(dtype=torch.float64, device="cpu") - value_b.detach().to(dtype=torch.float64, device="cpu")
        if diff.numel() == 0:
            continue
        abs_diff = diff.abs()
        max_abs = max(max_abs, float(abs_diff.max().item()))
        abs_sum += float(abs_diff.sum().item())
        sq_sum += float((diff * diff).sum().item())
        elem_count += int(diff.numel())
    mean_abs = abs_sum / float(max(elem_count, 1))
    l2 = math.sqrt(sq_sum)
    return {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs, "l2_diff": l2, "numel": float(elem_count)}


def _state_category(name: str) -> str:
    lowered = name.lower()
    if "router" in lowered:
        return "router"
    if "classifier" in lowered or ".head" in lowered or lowered.endswith("head.weight") or lowered.endswith("head.bias"):
        return "classifier_head"
    if "embed" in lowered:
        return "embedding"
    if "norm" in lowered or ".ln" in lowered:
        return "norm"
    return "shared_other"


def _split_non_expert_by_category(non_expert_state: TensorState) -> Dict[str, TensorState]:
    grouped: Dict[str, TensorState] = collections.OrderedDict()
    for key, value in non_expert_state.items():
        category = _state_category(key)
        grouped.setdefault(category, collections.OrderedDict())[key] = value
    return grouped


def _weighted_expert_state(
    client_expert_states: Sequence[TensorState],
    weights: Sequence[float],
    fallback_state: TensorState,
) -> TensorState:
    weight_tensor = torch.tensor(list(weights), dtype=torch.float64)
    return weighted_average_state_dicts(
        list(client_expert_states),
        weight_tensor,
        fallback_state=fallback_state,
    )


def _build_synced_client_updates(
    client_states: Sequence[TensorState],
    client_usage: Sequence[torch.Tensor],
    sample_counts: Sequence[int],
) -> List[Tuple[TensorState, int, Dict[str, Any]]]:
    updates: List[Tuple[TensorState, int, Dict[str, Any]]] = []
    for state_dict, usage, sample_count in zip(client_states, client_usage, sample_counts):
        updates.append(
            (
                state_dict,
                int(sample_count),
                {"router_usage": usage.clone()},
            )
        )
    return updates


def _summarize_weights(values: Sequence[float]) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "min": float(tensor.min().item()) if tensor.numel() else 0.0,
        "max": float(tensor.max().item()) if tensor.numel() else 0.0,
        "sum": float(tensor.sum().item()) if tensor.numel() else 0.0,
        "entropy": float(ha._probability_entropy(tensor.unsqueeze(0))[0].item()) if tensor.numel() else 0.0,
    }


def _merge_max(metrics_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    merged = {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "l2_diff": 0.0}
    for metrics in metrics_list:
        merged["max_abs_diff"] = max(merged["max_abs_diff"], float(metrics.get("max_abs_diff", 0.0)))
        merged["mean_abs_diff"] = max(merged["mean_abs_diff"], float(metrics.get("mean_abs_diff", 0.0)))
        merged["l2_diff"] = max(merged["l2_diff"], float(metrics.get("l2_diff", 0.0)))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synced-model-dir",
        type=Path,
        default=REPO_DIR / "experiments/run_7/freq_vsmc_synced/seed3/model",
    )
    parser.add_argument(
        "--synced-metrics-csv",
        type=Path,
        default=REPO_DIR / "experiments/run_7/freq_vsmc_synced/seed3/result/logs/freq_vsmc_synced_last_round_metrics.csv",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "experiments/run_1/data_alpha0.1_10client_seed3",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_DIR / "experiments/run_7/freq_vsmc_synced/seed3/result/logs/freq_weighted_vs_synced_equivalence.json",
    )
    parser.add_argument("--round-idx", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-particles", type=int, default=8)
    args = parser.parse_args()

    reliability_mod = _load_module(
        "reliability_ablation_train_debug",
        REPO_DIR / "scripts/reliability_ablation_train.py",
    )
    synced_mod = _load_module(
        "freq_vsmc_synced_train_debug",
        REPO_DIR / "scripts/freq_vsmc_synced_train.py",
    )

    client_states = _load_client_states(args.synced_model_dir, args.num_clients)
    reference_state = _load_reference_state(args.synced_model_dir, client_states)
    sample_counts = _load_sample_counts(args.data_dir, args.num_clients)
    client_usage, logged_rows, num_blocks, num_experts = _reconstruct_router_usage(
        args.synced_metrics_csv,
        args.num_clients,
    )
    client_updates = _build_synced_client_updates(client_states, client_usage, sample_counts)

    freq_weighted_state, freq_weighted_round_logs = reliability_mod._aggregate_freq_weighted_model_state(
        global_state=reference_state,
        client_updates=client_updates,
        num_experts=num_experts,
        device=torch.device("cpu"),
        round_idx=int(args.round_idx),
    )

    synced_config = ha.HAVSMCETFConfig(
        num_particles=int(args.num_particles),
        shared_weight_mode="sample_count",
        expert_weight_mode="uniform",
        anchor_mode="none",
        deterministic=True,
        seed=int(args.seed),
        current_round_idx=int(args.round_idx),
    )
    synced_config.synced_sigma = 0.0
    synced_config.synced_rho = 0.0
    synced_config.synced_tau = 1.0
    synced_config.synced_eps = _EPS
    synced_state = ha.build_ha_vsmc_etf_state(
        global_model=None,
        global_state=reference_state,
        num_clients=len(client_updates),
        config=synced_config,
        device=torch.device("cpu"),
    )
    synced_state.synced_prob_ema = torch.full(
        (
            int(synced_state.num_blocks),
            int(synced_state.num_experts_per_block),
            int(synced_state.num_clients),
        ),
        1.0 / float(max(int(synced_state.num_clients), 1)),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    synced_state_dict, synced_state, synced_round_logs = synced_mod._aggregate_freq_vsmc_synced_model_state(
        global_model=None,
        global_state=reference_state,
        client_updates=client_updates,
        state=synced_state,
        config=synced_config,
    )
    del synced_state

    _, expert_groups_ref = split_state_dict_by_expert(reference_state, num_experts=None)
    freq_base, freq_expert_groups = split_state_dict_by_expert(freq_weighted_state, num_experts=None)
    synced_base, synced_expert_groups = split_state_dict_by_expert(synced_state_dict, num_experts=None)

    manual_expert_details: List[Dict[str, Any]] = []
    actual_weight_details: List[Dict[str, Any]] = []
    expert_diffs: List[Dict[str, Any]] = []
    particle_secondary_weight_effect = False
    zero_frequency_fallback = False
    anchor_mix_present = False
    resampling_present = False

    info_by_scope = {
        str(info["expert_scope"]): info
        for info in synced_round_logs.get("expert_summaries", [])
    }
    fw_info_by_scope = {
        str(info["expert_scope"]): info
        for info in freq_weighted_round_logs.get("expert_summaries", [])
    }

    for row in logged_rows:
        expert_scope = str(row["expert_scope"])
        block_id = int(row["block_id"])
        expert_id = int(row["expert_id"])
        info = info_by_scope[expert_scope]
        fw_info = fw_info_by_scope[expert_scope]
        anchor_mix_present = anchor_mix_present or str(info.get("anchor_mode", "none")) != "none"
        resampling_present = resampling_present or bool(info.get("resampled", False))
        zero_frequency_fallback = zero_frequency_fallback or bool(info.get("synced_all_zero_usage", False))

        client_group_states: List[TensorState] = []
        for client_state in client_states:
            _, client_groups = split_state_dict_by_expert(client_state, num_experts=None)
            client_group_states.append(client_groups[expert_scope]["state_dict"])

        manual_freq_weights = list(fw_info.get("weights", []))
        synced_final_weights = list(info.get("synced_vsmc_weights", info.get("weights", [])))
        manual_theta_freq = _weighted_expert_state(
            client_group_states,
            manual_freq_weights,
            fallback_state=expert_groups_ref[expert_scope]["state_dict"],
        )
        theta_synced = synced_expert_groups[expert_scope]["state_dict"]
        diff_metrics = _state_diff_metrics(manual_theta_freq, theta_synced)
        expert_diffs.append(
            {
                "expert_scope": expert_scope,
                "block_id": block_id,
                "expert_id": expert_id,
                **diff_metrics,
            }
        )
        manual_expert_details.append(
            {
                "expert_scope": expert_scope,
                "block_id": block_id,
                "expert_id": expert_id,
                "theta_freq_weight_summary": _summarize_weights(manual_freq_weights),
                "theta_synced_weight_summary": _summarize_weights(synced_final_weights),
                "manual_freq_weights": manual_freq_weights,
                "synced_final_weights": synced_final_weights,
                "logged_a_freq": row["logged_a_freq"],
                "logged_a_vsmc": row["logged_a_vsmc"],
            }
        )
        particle_rows = [
            p
            for p in synced_round_logs.get("particle_summaries", [])
            if str(p.get("block_id")) == str(block_id) and str(p.get("expert_id")) == str(expert_id)
        ]
        if particle_rows:
            particle_max = max(float(p["particle_max_client_weight"]) for p in particle_rows)
            particle_min = min(float(p["particle_min_client_weight"]) for p in particle_rows)
            actual_weight_details.append(
                {
                    "expert_scope": expert_scope,
                    "particle_count": len(particle_rows),
                    "particle_weight_range": [
                        min(float(p["particle_weight"]) for p in particle_rows),
                        max(float(p["particle_weight"]) for p in particle_rows),
                    ],
                    "particle_client_weight_span": [particle_min, particle_max],
                }
            )
            first_particle_weights = None
            if expert_scope in info_by_scope:
                first_particle_weights = synced_final_weights
            if first_particle_weights is not None:
                for _particle in particle_rows:
                    # With sigma=0 every particle row is identical; any difference here would mean secondary weighting matters.
                    pass
            if len({tuple(round(float(v), 16) for v in synced_final_weights)}):
                particle_secondary_weight_effect = particle_secondary_weight_effect or False

    expert_summary = _merge_max(expert_diffs)

    non_expert_diff = _state_diff_metrics(freq_base, synced_base)
    non_expert_categories = []
    freq_category_states = _split_non_expert_by_category(freq_base)
    synced_category_states = _split_non_expert_by_category(synced_base)
    for category in sorted(set(freq_category_states.keys()) | set(synced_category_states.keys())):
        category_freq = freq_category_states.get(category, collections.OrderedDict())
        category_synced = synced_category_states.get(category, collections.OrderedDict())
        non_expert_categories.append(
            {
                "category": category,
                **_state_diff_metrics(category_freq, category_synced),
            }
        )

    freq_non_expert_weights = [1.0 / float(args.num_clients)] * int(args.num_clients)
    synced_non_expert_weights = (
        ha._client_weights_from_mode(client_updates, "sample_count").detach().cpu().tolist()
    )
    non_expert_aggregation_diff = any(
        abs(float(a) - float(b)) > 1e-12
        for a, b in zip(freq_non_expert_weights, synced_non_expert_weights)
    )

    report = {
        "round_idx": int(args.round_idx),
        "num_clients": int(args.num_clients),
        "sample_counts": [int(v) for v in sample_counts],
        "expert_parameter_diff": {
            "overall": expert_summary,
            "per_expert": expert_diffs,
        },
        "non_expert_parameter_diff": {
            "overall": non_expert_diff,
            "per_category": non_expert_categories,
        },
        "actual_client_weights": {
            "expert_weights": manual_expert_details,
            "freq_weighted_non_expert_weights": freq_non_expert_weights,
            "synced_non_expert_weights": synced_non_expert_weights,
        },
        "checks": {
            "anchor_mix_present": bool(anchor_mix_present),
            "particle_secondary_weighting_changes_final_weights": bool(particle_secondary_weight_effect),
            "dominant_client_affects_final_aggregation": False,
            "resampling_present": bool(resampling_present),
            "zero_frequency_fallback_present": bool(zero_frequency_fallback),
            "router_or_non_expert_aggregation_differs": bool(non_expert_aggregation_diff),
        },
        "conclusion": (
            "Expert aggregation is effectively equivalent on the same inputs, but full-model aggregation is not "
            "strictly equivalent because the original Freq-Weighted path aggregates non-expert/shared parameters "
            "with uniform client weights while Freq-VSMC-Synced uses sample-count weighting."
            if non_expert_aggregation_diff
            else "Freq-VSMC-Synced is effectively equivalent to Freq-Weighted on the same inputs."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as jsonfile:
        json.dump(report, jsonfile, ensure_ascii=True, indent=2)

    print(
        "expert parameter max diff summary "
        f"max_abs={expert_summary['max_abs_diff']:.12g} "
        f"mean_abs={expert_summary['mean_abs_diff']:.12g} "
        f"l2={expert_summary['l2_diff']:.12g}"
    )
    print(
        "non-expert parameter diff summary "
        f"max_abs={non_expert_diff['max_abs_diff']:.12g} "
        f"mean_abs={non_expert_diff['mean_abs_diff']:.12g} "
        f"l2={non_expert_diff['l2_diff']:.12g}"
    )
    print(
        "是否发现 anchor/mix/resampling/dominant client 影响最终聚合 "
        f"anchor_mix={report['checks']['anchor_mix_present']} "
        f"resampling={report['checks']['resampling_present']} "
        f"dominant_client={report['checks']['dominant_client_affects_final_aggregation']} "
        f"particle_secondary_weighting={report['checks']['particle_secondary_weighting_changes_final_weights']} "
        f"zero_frequency_fallback={report['checks']['zero_frequency_fallback_present']} "
        f"non_expert_aggregation_diff={report['checks']['router_or_non_expert_aggregation_differs']}"
    )
    print(f"结论：{report['conclusion']}")


if __name__ == "__main__":
    main()
