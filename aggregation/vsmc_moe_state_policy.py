from __future__ import annotations

import collections
import copy
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call

from aggregation.vsmc_expert_aggregator import (
    TensorState,
    _merge_with_reference_state,
    _ordered_union_keys,
    _sample_count_weights,
    clone_detached_state_dict,
    effective_sample_size,
    group_expert_groups_by_block,
    merge_expert_states_into_global_state,
    split_state_dict_by_expert,
    weighted_average_state_dicts,
)

_SAFE_LOG_MIN = -50.0
_SAFE_LOG_MAX = 50.0
_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class MoEStatePolicyConfig:
    enabled: bool = False
    num_particles: int = 8
    elite_top_k: int = 4
    ess_threshold: float = 0.5
    jitter_std: float = 0.03
    logq_coef: float = 0.1
    reward_temperature: float = 10.0
    transition_coef: float = 0.05
    proposal_lr: float = 1e-4
    proposal_hidden_dim: int = 128
    residual_latent_dim: int = 8
    max_residual_ratio: float = 0.15
    aggregation_modes: Tuple[str, ...] = ("fedavg", "shrink075", "shrink05", "momentum", "clipped_mean", "latent_residual")
    momentum_beta: float = 0.9
    clip_norm_quantile: float = 0.8
    block_scale_min: float = 0.25
    block_scale_max: float = 1.25
    val_loss_coef: float = 1.0
    val_loss_improve_coef: float = 0.5
    val_acc_improve_coef: float = 0.2
    router_balance_coef: float = 0.1
    expert_conflict_coef: float = 0.05
    update_norm_coef: float = 0.02
    class_balance_coef: float = 0.05
    val_batches: int = 1
    full_eval_interval: int = 5
    train_proposal: bool = True
    detach_resampling: bool = True
    log_diagnostics: bool = True
    selector_mode: str = "vsmc"
    total_rounds: int = 1
    grad_clip: float = 1.0
    log_dir: str = "logs"


@dataclass
class MoEStateParticle:
    block_scale_logits: torch.Tensor
    mode_logits: torch.Tensor
    expert_client_factor_u: torch.Tensor
    client_factor_v: torch.Tensor
    router_mix_logit: torch.Tensor
    residual_latent: torch.Tensor
    prev_log_weight: torch.Tensor
    ancestor_id: int
    particle_id: int


@dataclass
class MoEStateAggregatorState:
    particles: List[MoEStateParticle] = field(default_factory=list)
    block_momentum: Dict[str, TensorState] = field(default_factory=dict)
    prev_eval_loss: float = 0.0
    prev_eval_acc: float = 0.0
    eval_loss_ema: float = 0.0
    eval_acc_ema: float = 0.0
    previous_ess: float = 1.0
    previous_reward_std: float = 0.0
    previous_selected_particle_entropy: float = 0.0
    previous_fedavg_weight_proxy: float = 1.0
    best_acc_so_far: float = float("-inf")
    round_idx: int = 0
    particle_counter: int = 0
    last_round_summary: Dict[str, Any] = field(default_factory=dict)


class MoEStateProposalNet(nn.Module):
    def __init__(self, diagnostics_dim: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.diagnostics_dim = diagnostics_dim
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(diagnostics_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, diagnostics: torch.Tensor, prev_state_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([diagnostics.reshape(-1), prev_state_flat.reshape(-1)], dim=0)
        raw = self.net(x)
        mean_delta, raw_log_std = raw.chunk(2, dim=-1)
        log_std = -3.0 + 2.0 * torch.tanh(raw_log_std)
        return mean_delta, log_std


class MoEStateDiagnostics:
    ROUND_FIELDS = [
        "round", "seed", "method", "num_particles", "elite_top_k", "normalized_ess", "resampled",
        "proposal_loss", "proposal_grad_norm", "proposal_param_delta_norm", "logq_coef", "reward_temperature",
        "logf_mean", "logf_std", "logg_mean", "logg_std", "logq_mean", "logq_std", "log_weight_mean",
        "log_weight_std", "particle_weight_entropy", "top_particle_idx", "top_particle_weight", "top_particle_logg",
        "top_particle_logq", "top_particle_logf", "oracle_particle_idx", "oracle_particle_logg", "reward_std",
        "val_loss_mean_particles", "val_loss_best_particle", "val_acc_mean_particles", "val_acc_best_particle",
        "fedavg_reference_val_loss", "fedavg_reference_val_acc", "best_minus_fedavg_val_loss", "best_minus_fedavg_val_acc",
        "server_test_acc", "server_test_loss", "best_acc_so_far", "final_acc", "min_eval_loss_so_far",
    ]
    PARTICLE_FIELDS = [
        "round", "particle_id", "ancestor_id", "weight", "log_weight", "log_g", "log_q", "log_f", "val_loss", "val_acc",
        "router_load_cv", "router_entropy", "expert_conflict", "update_norm", "class_loss_std", "mean_block_scale",
        "min_block_scale", "max_block_scale", "mean_fedavg_mode_weight", "mean_shrink_mode_weight",
        "mean_momentum_mode_weight", "mean_clipped_mean_mode_weight", "mean_latent_residual_mode_weight",
        "residual_norm_mean", "residual_norm_max",
    ]
    BLOCK_FIELDS = [
        "round", "particle_id", "block_id", "block_scale", "fedavg_mode_weight", "shrink075_mode_weight",
        "shrink05_mode_weight", "momentum_mode_weight", "clipped_mean_mode_weight", "latent_residual_mode_weight",
        "fedavg_delta_norm", "momentum_delta_norm", "clipped_mean_delta_norm", "latent_residual_norm", "final_delta_norm",
    ]

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.round_csv = os.path.join(log_dir, "moe_state_vsmc_round_metrics.csv")
        self.particle_csv = os.path.join(log_dir, "moe_state_vsmc_particle_metrics.csv")
        self.block_csv = os.path.join(log_dir, "moe_state_vsmc_block_metrics.csv")
        self._init_csv(self.round_csv, self.ROUND_FIELDS)
        self._init_csv(self.particle_csv, self.PARTICLE_FIELDS)
        self._init_csv(self.block_csv, self.BLOCK_FIELDS)

    def _init_csv(self, path: str, fieldnames: Sequence[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as csvfile:
            csv.DictWriter(csvfile, fieldnames=fieldnames).writeheader()

    def write_round(self, row: Dict[str, Any]):
        with open(self.round_csv, "a", newline="") as csvfile:
            csv.DictWriter(csvfile, fieldnames=self.ROUND_FIELDS).writerow(row)

    def write_particles(self, rows: Sequence[Dict[str, Any]]):
        if not rows:
            return
        with open(self.particle_csv, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.PARTICLE_FIELDS)
            for row in rows:
                writer.writerow(row)

    def write_blocks(self, rows: Sequence[Dict[str, Any]]):
        if not rows:
            return
        with open(self.block_csv, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.BLOCK_FIELDS)
            for row in rows:
                writer.writerow(row)


def _sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _state_to_device(state: TensorState, device: torch.device, detach: bool = False) -> TensorState:
    out: TensorState = collections.OrderedDict()
    for key, value in state.items():
        if torch.is_tensor(value):
            tensor = value.detach() if detach else value
            out[key] = tensor.to(device)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _float_state_keys(state: TensorState) -> List[str]:
    return [key for key, value in state.items() if torch.is_tensor(value) and torch.is_floating_point(value)]


def _subtract_float_states(a: TensorState, b: TensorState, device: torch.device, detach: bool = False) -> TensorState:
    out: TensorState = collections.OrderedDict()
    for key in sorted(set(a.keys()) & set(b.keys())):
        va = a[key]
        vb = b[key]
        if not (torch.is_tensor(va) and torch.is_tensor(vb)):
            continue
        if not (torch.is_floating_point(va) and torch.is_floating_point(vb)):
            continue
        if va.shape != vb.shape:
            continue
        lhs = va.detach() if detach else va
        rhs = vb.detach() if detach else vb
        out[key] = _sanitize_tensor(lhs.to(device=device, dtype=torch.float32) - rhs.to(device=device, dtype=torch.float32))
    return out


def _add_float_states(reference: TensorState, delta: TensorState) -> TensorState:
    out: TensorState = collections.OrderedDict()
    for key in _ordered_union_keys([reference, delta]):
        ref = reference.get(key)
        if not torch.is_tensor(ref):
            if key in reference:
                out[key] = reference[key]
            continue
        if not torch.is_floating_point(ref):
            out[key] = ref
            continue
        dv = delta.get(key, torch.zeros_like(ref))
        out[key] = _sanitize_tensor(ref + dv.to(device=ref.device, dtype=ref.dtype))
    return out


def _scale_float_state(delta: TensorState, scale: torch.Tensor | float) -> TensorState:
    out: TensorState = collections.OrderedDict()
    scale_t = scale if torch.is_tensor(scale) else None
    for key, value in delta.items():
        if not (torch.is_tensor(value) and torch.is_floating_point(value)):
            continue
        factor = scale_t.to(device=value.device, dtype=value.dtype) if scale_t is not None else float(scale)
        out[key] = _sanitize_tensor(value * factor)
    return out


def _combine_weighted_states(state_weight_pairs: Sequence[Tuple[TensorState, torch.Tensor]]) -> TensorState:
    out: TensorState = collections.OrderedDict()
    if not state_weight_pairs:
        return out
    for key in _ordered_union_keys([state for state, _ in state_weight_pairs]):
        acc = None
        for state, weight in state_weight_pairs:
            value = state.get(key)
            if not (torch.is_tensor(value) and torch.is_floating_point(value)):
                continue
            w = weight.to(device=value.device, dtype=value.dtype)
            term = value * w
            acc = term if acc is None else acc + term
        if acc is not None:
            out[key] = _sanitize_tensor(acc)
    return out


def _state_sq_sum_and_count(state: TensorState) -> Tuple[float, int]:
    sq_sum = 0.0
    count = 0
    for value in state.values():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            tensor = _sanitize_tensor(value.detach().float())
            sq_sum += float(tensor.pow(2).sum().item())
            count += tensor.numel()
    return sq_sum, count


def _state_l2_norm(state: TensorState) -> float:
    sq_sum, _ = _state_sq_sum_and_count(state)
    return math.sqrt(max(sq_sum, 0.0))


def _flatten_float_state(state: TensorState, keys: Sequence[str], device: torch.device) -> torch.Tensor:
    parts = []
    for key in keys:
        value = state.get(key)
        if torch.is_tensor(value) and torch.is_floating_point(value):
            parts.append(_sanitize_tensor(value.to(device=device, dtype=torch.float32)).reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.cat(parts, dim=0)


def _delta_state_from_vector(reference_state: TensorState, keys: Sequence[str], vector: torch.Tensor) -> TensorState:
    out: TensorState = collections.OrderedDict()
    offset = 0
    for key in keys:
        ref = reference_state.get(key)
        if not (torch.is_tensor(ref) and torch.is_floating_point(ref)):
            continue
        numel = ref.numel()
        piece = vector[offset: offset + numel].reshape(ref.shape)
        out[key] = piece.to(device=ref.device, dtype=ref.dtype)
        offset += numel
    return out


def _weighted_average_float_states(states: Sequence[TensorState], weights: torch.Tensor, device: torch.device) -> TensorState:
    out: TensorState = collections.OrderedDict()
    weights = _sanitize_tensor(weights.to(device=device, dtype=torch.float32)).reshape(-1)
    if weights.numel() == 0:
        return out
    if weights.sum().item() <= 0.0:
        weights = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
    weights = weights / weights.sum().clamp_min(1e-12)
    for key in _ordered_union_keys(states):
        tensors = []
        local_weights = []
        for idx, state in enumerate(states):
            value = state.get(key)
            if not (torch.is_tensor(value) and torch.is_floating_point(value)):
                continue
            tensors.append(_sanitize_tensor(value.to(device=device, dtype=torch.float32)))
            local_weights.append(weights[idx])
        if not tensors:
            continue
        stacked = torch.stack(tensors, dim=0)
        w = torch.stack(local_weights, dim=0)
        view_shape = [w.shape[0]] + [1] * (stacked.dim() - 1)
        out[key] = _sanitize_tensor((stacked * w.view(*view_shape)).sum(dim=0))
    return out


def _normalize_client_weights(scores: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(scores, dim=-1)
    weights = weights * sample_weights.reshape(1, -1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights


def _state_cosine_conflict(delta_states: Sequence[TensorState], keys: Sequence[str], device: torch.device) -> float:
    if len(delta_states) < 2:
        return 0.0
    flat = [_flatten_float_state(state, keys, device=device) for state in delta_states]
    flat = [vec for vec in flat if vec.numel() > 0]
    if len(flat) < 2:
        return 0.0
    norms = [vec.norm().clamp_min(1e-12) for vec in flat]
    total = 0.0
    count = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            cosine = torch.dot(flat[i] / norms[i], flat[j] / norms[j]).item()
            total += cosine
            count += 1
    mean_cosine = total / max(count, 1)
    return float(1.0 - mean_cosine)


def _compute_class_loss_std(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sample_losses = nn.functional.cross_entropy(logits, labels, reduction="none")
    per_class = []
    for cls in labels.unique():
        mask = labels == cls
        if mask.any():
            per_class.append(sample_losses[mask].mean())
    if len(per_class) <= 1:
        return logits.new_tensor(0.0)
    return torch.stack(per_class).std(unbiased=False)


def collect_moe_fl_diagnostics(
    global_state: TensorState,
    client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
    block_client_delta_states: Dict[str, List[TensorState]],
    grouped_global: Dict[str, Dict[str, Any]],
    prev_eval_metrics: Dict[str, float],
    round_idx: int,
    total_rounds: int,
    device: torch.device,
) -> Dict[str, float]:
    client_update_norms = []
    block_norms = []
    conflicts = []
    router_entropies = []
    router_load_cvs = []
    for state_dict, _, metrics in client_updates:
        delta_state = _subtract_float_states(state_dict, global_state, device=device, detach=True)
        client_update_norms.append(_state_l2_norm(delta_state))
        router_entropies.append(float(metrics.get("router_entropy", 0.0) or 0.0))
        layer_counts = metrics.get("layer_route_counts", {}) or {}
        layer_values = []
        for layer_map in layer_counts.values():
            counts = torch.tensor([float(v) for v in layer_map.values()], dtype=torch.float32)
            if counts.numel() > 0 and counts.mean().item() > 0.0:
                layer_values.append(float((counts.std(unbiased=False) / counts.mean().clamp_min(1e-12)).item()))
        router_load_cvs.append(sum(layer_values) / len(layer_values) if layer_values else 0.0)
    for block_scope, delta_states in block_client_delta_states.items():
        keys = _float_state_keys(grouped_global[block_scope]["state_dict"])
        conflicts.append(_state_cosine_conflict(delta_states, keys, device=device))
        for delta_state in delta_states:
            block_norms.append(_state_l2_norm(delta_state))
    def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        tensor = torch.tensor(values, dtype=torch.float32)
        return float(tensor.mean().item()), float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
    client_mean, client_std = _mean_std(client_update_norms)
    block_mean, block_std = _mean_std(block_norms)
    conflict_mean, _ = _mean_std(conflicts)
    router_entropy_mean, _ = _mean_std(router_entropies)
    router_load_cv_mean, _ = _mean_std(router_load_cvs)
    return {
        "round_progress": 0.0 if total_rounds <= 1 else float(round_idx - 1) / float(total_rounds - 1),
        "prev_eval_loss": float(prev_eval_metrics.get("prev_eval_loss", 0.0)),
        "prev_eval_acc": float(prev_eval_metrics.get("prev_eval_acc", 0.0)),
        "eval_loss_ema": float(prev_eval_metrics.get("eval_loss_ema", 0.0)),
        "eval_acc_ema": float(prev_eval_metrics.get("eval_acc_ema", 0.0)),
        "last_eval_loss_improvement": float(prev_eval_metrics.get("eval_loss_improvement", 0.0)),
        "last_eval_acc_improvement": float(prev_eval_metrics.get("eval_acc_improvement", 0.0)),
        "mean_client_update_norm": float(client_mean),
        "std_client_update_norm": float(client_std),
        "mean_expert_update_norm": float(block_mean),
        "std_expert_update_norm": float(block_std),
        "mean_expert_conflict": float(conflict_mean),
        "max_expert_conflict": float(max(conflicts) if conflicts else 0.0),
        "router_entropy": float(router_entropy_mean),
        "router_load_cv": float(router_load_cv_mean),
        "previous_ess": float(prev_eval_metrics.get("previous_ess", 1.0)),
        "previous_reward_std": float(prev_eval_metrics.get("previous_reward_std", 0.0)),
        "previous_selected_particle_entropy": float(prev_eval_metrics.get("previous_selected_particle_entropy", 0.0)),
        "previous_fedavg_weight_proxy": float(prev_eval_metrics.get("previous_fedavg_weight_proxy", 1.0)),
    }


class MoEStateVSMCAggregator:
    def __init__(self, config: MoEStatePolicyConfig, num_experts: int, logger: Optional[Any] = None):
        self.config = config
        self.num_experts = num_experts
        self.logger = logger
        self.state = MoEStateAggregatorState()
        self.proposal_net: Optional[MoEStateProposalNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.metrics = MoEStateDiagnostics(log_dir=config.log_dir) if config.log_diagnostics else None
        self._structure = None

    def _materialize_val_batches(self, val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]], device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batches = []
        if val_loader is None:
            return batches
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max(int(self.config.val_batches), 1):
                break
            inputs, labels = batch
            batches.append((inputs.to(device), labels.to(device)))
        return batches

    def _build_structure(self, grouped_global: Dict[str, Dict[str, Any]], num_clients: int) -> Dict[str, Any]:
        block_scopes = list(grouped_global.keys())
        num_blocks = len(block_scopes)
        rank = max(1, min(4, int(self.config.residual_latent_dim)))
        mode_names = tuple(str(v).strip() for v in self.config.aggregation_modes if str(v).strip())
        if not mode_names:
            mode_names = ("fedavg",)
        expert_ids = []
        for group in grouped_global.values():
            expert_ids.extend([int(v) for v in group.get("expert_ids", [])])
        expert_ids = sorted(set(expert_ids))
        expert_index = {expert_id: idx for idx, expert_id in enumerate(expert_ids)}
        state_dim = (
            num_blocks
            + num_blocks * len(mode_names)
            + len(expert_ids) * rank
            + num_clients * rank
            + 1
            + num_blocks * int(self.config.residual_latent_dim)
        )
        diagnostics_dim = 15
        return {
            "block_scopes": block_scopes,
            "num_blocks": num_blocks,
            "mode_names": mode_names,
            "num_modes": len(mode_names),
            "rank": rank,
            "expert_ids": expert_ids,
            "expert_index": expert_index,
            "state_dim": state_dim,
            "diagnostics_dim": diagnostics_dim,
            "num_clients": num_clients,
        }

    def _init_network_if_needed(self, structure: Dict[str, Any]):
        if self.proposal_net is not None and self._structure is not None and self._structure["state_dim"] == structure["state_dim"] and self._structure["diagnostics_dim"] == structure["diagnostics_dim"]:
            return
        self._structure = structure
        self.proposal_net = MoEStateProposalNet(
            diagnostics_dim=structure["diagnostics_dim"],
            state_dim=structure["state_dim"],
            hidden_dim=int(self.config.proposal_hidden_dim),
        )
        self.optimizer = torch.optim.Adam(self.proposal_net.parameters(), lr=float(self.config.proposal_lr))
        self.state.particles = []
        self.state.particle_counter = 0

    def _particle_to_flat(self, particle: MoEStateParticle) -> torch.Tensor:
        return torch.cat([
            particle.block_scale_logits.reshape(-1),
            particle.mode_logits.reshape(-1),
            particle.expert_client_factor_u.reshape(-1),
            particle.client_factor_v.reshape(-1),
            particle.router_mix_logit.reshape(-1),
            particle.residual_latent.reshape(-1),
        ], dim=0)

    def _flat_to_particle(self, flat: torch.Tensor, structure: Dict[str, Any], particle_id: int, ancestor_id: int, prev_log_weight: float) -> MoEStateParticle:
        nb = structure["num_blocks"]
        nm = structure["num_modes"]
        ne = len(structure["expert_ids"])
        rank = structure["rank"]
        nc = structure["num_clients"]
        latent = int(self.config.residual_latent_dim)
        offset = 0
        block_scale_logits = flat[offset: offset + nb]
        offset += nb
        mode_logits = flat[offset: offset + nb * nm].reshape(nb, nm)
        offset += nb * nm
        expert_u = flat[offset: offset + ne * rank].reshape(ne, rank)
        offset += ne * rank
        client_v = flat[offset: offset + nc * rank].reshape(nc, rank)
        offset += nc * rank
        router_mix = flat[offset: offset + 1]
        offset += 1
        residual = flat[offset: offset + nb * latent].reshape(nb, latent)
        return MoEStateParticle(
            block_scale_logits=block_scale_logits,
            mode_logits=mode_logits,
            expert_client_factor_u=expert_u,
            client_factor_v=client_v,
            router_mix_logit=router_mix,
            residual_latent=residual,
            prev_log_weight=torch.tensor(float(prev_log_weight), device=flat.device, dtype=flat.dtype),
            ancestor_id=int(ancestor_id),
            particle_id=int(particle_id),
        )

    def _clone_particle(self, particle: MoEStateParticle, detach: bool = True) -> MoEStateParticle:
        def maybe(t: torch.Tensor):
            out = t.detach().clone() if detach else t.clone()
            return out
        return MoEStateParticle(
            block_scale_logits=maybe(particle.block_scale_logits),
            mode_logits=maybe(particle.mode_logits),
            expert_client_factor_u=maybe(particle.expert_client_factor_u),
            client_factor_v=maybe(particle.client_factor_v),
            router_mix_logit=maybe(particle.router_mix_logit),
            residual_latent=maybe(particle.residual_latent),
            prev_log_weight=maybe(particle.prev_log_weight),
            ancestor_id=particle.ancestor_id,
            particle_id=particle.particle_id,
        )

    def _initial_particle(self, structure: Dict[str, Any], device: torch.device) -> MoEStateParticle:
        nb = structure["num_blocks"]
        nm = structure["num_modes"]
        ne = len(structure["expert_ids"])
        nc = structure["num_clients"]
        rank = structure["rank"]
        latent = int(self.config.residual_latent_dim)
        block_scale_logits = torch.zeros(nb, device=device)
        mode_logits = torch.full((nb, nm), -2.0, device=device)
        fedavg_idx = structure["mode_names"].index("fedavg") if "fedavg" in structure["mode_names"] else 0
        mode_logits[:, fedavg_idx] = 2.0
        expert_u = torch.zeros(ne, rank, device=device)
        client_v = torch.zeros(nc, rank, device=device)
        router_mix = torch.zeros(1, device=device)
        residual = torch.zeros(nb, latent, device=device)
        particle = MoEStateParticle(
            block_scale_logits=block_scale_logits,
            mode_logits=mode_logits,
            expert_client_factor_u=expert_u,
            client_factor_v=client_v,
            router_mix_logit=router_mix,
            residual_latent=residual,
            prev_log_weight=torch.zeros(1, device=device).reshape(()),
            ancestor_id=-1,
            particle_id=self.state.particle_counter,
        )
        self.state.particle_counter += 1
        return particle

    def _ensure_particles(self, structure: Dict[str, Any], device: torch.device):
        if self.state.particles:
            return
        particles = []
        for idx in range(int(self.config.num_particles)):
            particle = self._initial_particle(structure, device)
            if idx > 0:
                noise = torch.randn_like(self._particle_to_flat(particle)) * float(self.config.jitter_std)
                particle = self._flat_to_particle(self._particle_to_flat(particle) + noise, structure, particle.particle_id, -1, 0.0)
            particles.append(particle)
        self.state.particles = particles

    def _diagnostics_to_tensor(self, diagnostics: Dict[str, float], device: torch.device) -> torch.Tensor:
        values = [
            diagnostics.get("round_progress", 0.0),
            diagnostics.get("prev_eval_loss", 0.0),
            diagnostics.get("prev_eval_acc", 0.0),
            diagnostics.get("eval_loss_ema", 0.0),
            diagnostics.get("eval_acc_ema", 0.0),
            diagnostics.get("last_eval_loss_improvement", 0.0),
            diagnostics.get("last_eval_acc_improvement", 0.0),
            diagnostics.get("mean_client_update_norm", 0.0),
            diagnostics.get("std_client_update_norm", 0.0),
            diagnostics.get("mean_expert_update_norm", 0.0),
            diagnostics.get("std_expert_update_norm", 0.0),
            diagnostics.get("mean_expert_conflict", 0.0),
            diagnostics.get("max_expert_conflict", 0.0),
            diagnostics.get("router_entropy", 0.0),
            diagnostics.get("router_load_cv", 0.0),
        ]
        tensor = torch.tensor(values, dtype=torch.float32, device=device)
        return torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)

    def _build_reliability_scores(self, particle: MoEStateParticle, structure: Dict[str, Any], sample_weight_tensor: torch.Tensor) -> torch.Tensor:
        scores = particle.expert_client_factor_u @ particle.client_factor_v.transpose(0, 1)
        return _normalize_client_weights(scores, sample_weight_tensor)

    def _compute_block_client_deltas(
        self,
        previous_block_state: TensorState,
        client_block_states: Sequence[TensorState],
        device: torch.device,
    ) -> List[TensorState]:
        return [_subtract_float_states(state, previous_block_state, device=device, detach=False) for state in client_block_states]

    def _compute_clipped_delta(self, delta_states: Sequence[TensorState], client_weights: torch.Tensor, device: torch.device) -> TensorState:
        norms = torch.tensor([_state_l2_norm(delta_state) for delta_state in delta_states], dtype=torch.float32, device=device)
        if norms.numel() == 0:
            return collections.OrderedDict()
        threshold = torch.quantile(norms, float(self.config.clip_norm_quantile)) if norms.numel() > 1 else norms[0]
        clipped_states = []
        for idx, delta_state in enumerate(delta_states):
            norm = norms[idx]
            scale = min(1.0, float((threshold / (norm + 1e-12)).item())) if norm.item() > 0 else 1.0
            clipped_states.append(_scale_float_state(delta_state, scale))
        return _weighted_average_float_states(clipped_states, client_weights, device=device)

    def _build_block_basis(self, delta_states: Sequence[TensorState], reference_state: TensorState, device: torch.device) -> Tuple[List[str], torch.Tensor]:
        keys = _float_state_keys(reference_state)
        if not keys:
            return keys, torch.empty(int(self.config.residual_latent_dim), 0, device=device)
        matrix = torch.stack([_flatten_float_state(state, keys, device=device) for state in delta_states], dim=0)
        if matrix.numel() == 0 or matrix.shape[1] == 0:
            return keys, torch.empty(int(self.config.residual_latent_dim), 0, device=device)
        _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(int(self.config.residual_latent_dim), int(vh.shape[0]))
        basis = vh[:rank]
        if rank < int(self.config.residual_latent_dim):
            basis = torch.cat([basis, torch.zeros(int(self.config.residual_latent_dim) - rank, basis.shape[1], device=device)], dim=0)
        return keys, basis

    def _decode_residual(self, latent: torch.Tensor, basis: torch.Tensor, reference_state: TensorState, keys: Sequence[str], fedavg_norm: float) -> Tuple[TensorState, float]:
        if basis.numel() == 0:
            return collections.OrderedDict(), 0.0
        vector = torch.matmul(latent.to(dtype=basis.dtype), basis)
        delta_state = _delta_state_from_vector(reference_state, keys, vector)
        residual_norm = _state_l2_norm(delta_state)
        max_norm = float(self.config.max_residual_ratio) * max(float(fedavg_norm), 1e-12)
        if residual_norm > max_norm > 0.0:
            delta_state = _scale_float_state(delta_state, max_norm / max(residual_norm, 1e-12))
            residual_norm = _state_l2_norm(delta_state)
        return delta_state, residual_norm

    def _apply_particle_policy(
        self,
        particle: MoEStateParticle,
        grouped_previous: Dict[str, Dict[str, Any]],
        grouped_clients: List[Dict[str, Dict[str, Any]]],
        sample_weight_tensor: torch.Tensor,
        structure: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[Dict[str, TensorState], Dict[str, Any], List[Dict[str, Any]]]:
        candidate_blocks: Dict[str, TensorState] = collections.OrderedDict()
        block_metrics: List[Dict[str, Any]] = []
        reliability = self._build_reliability_scores(particle, structure, sample_weight_tensor)
        block_scales = torch.sigmoid(particle.block_scale_logits) * (float(self.config.block_scale_max) - float(self.config.block_scale_min)) + float(self.config.block_scale_min)
        mode_weights = torch.softmax(particle.mode_logits, dim=-1)
        mean_conflicts = []
        update_norms = []
        residual_norms = []
        for block_idx, block_scope in enumerate(structure["block_scopes"]):
            previous_block_state = _state_to_device(grouped_previous[block_scope]["state_dict"], device=device, detach=False)
            client_block_states = [_state_to_device(group_map[block_scope]["state_dict"], device=device, detach=False) for group_map in grouped_clients if block_scope in group_map]
            if not client_block_states:
                candidate_blocks[block_scope] = previous_block_state
                continue
            delta_states = self._compute_block_client_deltas(previous_block_state, client_block_states, device=device)
            fedavg_delta = _weighted_average_float_states(delta_states, sample_weight_tensor[:len(delta_states)], device=device)
            fedavg_norm = _state_l2_norm(fedavg_delta)
            block_expert_ids = grouped_previous[block_scope].get("expert_ids", [])
            if block_expert_ids:
                rel_rows = [reliability[structure["expert_index"][int(expert_id)]] for expert_id in block_expert_ids if int(expert_id) in structure["expert_index"]]
                client_weights = torch.stack(rel_rows, dim=0).mean(dim=0)[:len(delta_states)] if rel_rows else sample_weight_tensor[:len(delta_states)]
            else:
                client_weights = sample_weight_tensor[:len(delta_states)]
            client_weights = client_weights / client_weights.sum().clamp_min(1e-12)
            reliable_delta = _weighted_average_float_states(delta_states, client_weights, device=device)
            clipped_delta = self._compute_clipped_delta(delta_states, client_weights, device=device)
            momentum_prev = self.state.block_momentum.get(block_scope)
            if momentum_prev is None:
                momentum_delta = clone_detached_state_dict(fedavg_delta)
            else:
                momentum_delta = collections.OrderedDict()
                for key in _ordered_union_keys([momentum_prev, fedavg_delta]):
                    mv = momentum_prev.get(key)
                    fv = fedavg_delta.get(key)
                    if torch.is_tensor(mv) and torch.is_tensor(fv) and torch.is_floating_point(mv) and torch.is_floating_point(fv):
                        momentum_delta[key] = _sanitize_tensor(float(self.config.momentum_beta) * mv.to(device=device, dtype=torch.float32) + fv.to(device=device, dtype=torch.float32))
            keys, basis = self._build_block_basis(delta_states, previous_block_state, device=device)
            residual_delta, residual_norm = self._decode_residual(particle.residual_latent[block_idx], basis, previous_block_state, keys, fedavg_norm)
            latent_residual_delta = _combine_weighted_states([
                (reliable_delta, torch.tensor(1.0, device=device)),
                (residual_delta, torch.tensor(1.0, device=device)),
            ])
            mode_state_map = {
                "fedavg": fedavg_delta,
                "shrink075": _scale_float_state(fedavg_delta, 0.75),
                "shrink05": _scale_float_state(fedavg_delta, 0.50),
                "momentum": momentum_delta,
                "clipped_mean": clipped_delta,
                "latent_residual": latent_residual_delta,
            }
            mixed_delta = _combine_weighted_states([
                (mode_state_map[mode_name], mode_weights[block_idx, mode_idx])
                for mode_idx, mode_name in enumerate(structure["mode_names"])
                if mode_name in mode_state_map
            ])
            final_delta = _scale_float_state(mixed_delta, block_scales[block_idx])
            candidate_blocks[block_scope] = _add_float_states(previous_block_state, final_delta)
            conflict = _state_cosine_conflict(delta_states, keys, device=device)
            final_delta_norm = _state_l2_norm(final_delta)
            mean_conflicts.append(conflict)
            update_norms.append(final_delta_norm)
            residual_norms.append(residual_norm)
            mode_weight_map = {mode_name: float(mode_weights[block_idx, mode_idx].detach().item()) for mode_idx, mode_name in enumerate(structure["mode_names"])}
            block_metrics.append({
                "block_id": block_scope,
                "block_scale": float(block_scales[block_idx].detach().item()),
                "fedavg_mode_weight": mode_weight_map.get("fedavg", 0.0),
                "shrink075_mode_weight": mode_weight_map.get("shrink075", 0.0),
                "shrink05_mode_weight": mode_weight_map.get("shrink05", 0.0),
                "momentum_mode_weight": mode_weight_map.get("momentum", 0.0),
                "clipped_mean_mode_weight": mode_weight_map.get("clipped_mean", 0.0),
                "latent_residual_mode_weight": mode_weight_map.get("latent_residual", 0.0),
                "fedavg_delta_norm": float(fedavg_norm),
                "momentum_delta_norm": float(_state_l2_norm(momentum_delta)),
                "clipped_mean_delta_norm": float(_state_l2_norm(clipped_delta)),
                "latent_residual_norm": float(residual_norm),
                "final_delta_norm": float(final_delta_norm),
            })
        particle_stats = {
            "mean_block_scale": float(block_scales.detach().mean().item()) if block_scales.numel() else 0.0,
            "min_block_scale": float(block_scales.detach().min().item()) if block_scales.numel() else 0.0,
            "max_block_scale": float(block_scales.detach().max().item()) if block_scales.numel() else 0.0,
            "mean_fedavg_mode_weight": float(mode_weights[:, structure["mode_names"].index("fedavg")].mean().item()) if "fedavg" in structure["mode_names"] else 0.0,
            "mean_shrink_mode_weight": float(mode_weights[:, structure["mode_names"].index("shrink075")].mean().item()) if "shrink075" in structure["mode_names"] else 0.0,
            "mean_momentum_mode_weight": float(mode_weights[:, structure["mode_names"].index("momentum")].mean().item()) if "momentum" in structure["mode_names"] else 0.0,
            "mean_clipped_mean_mode_weight": float(mode_weights[:, structure["mode_names"].index("clipped_mean")].mean().item()) if "clipped_mean" in structure["mode_names"] else 0.0,
            "mean_latent_residual_mode_weight": float(mode_weights[:, structure["mode_names"].index("latent_residual")].mean().item()) if "latent_residual" in structure["mode_names"] else 0.0,
            "residual_norm_mean": float(sum(residual_norms) / max(len(residual_norms), 1)) if residual_norms else 0.0,
            "residual_norm_max": float(max(residual_norms) if residual_norms else 0.0),
            "expert_conflict": float(sum(mean_conflicts) / max(len(mean_conflicts), 1)) if mean_conflicts else 0.0,
            "update_norm": float(sum(update_norms) / max(len(update_norms), 1)) if update_norms else 0.0,
        }
        return candidate_blocks, particle_stats, block_metrics

    def _build_candidate_global_state(
        self,
        fedavg_state: TensorState,
        fedavg_base_state: TensorState,
        candidate_block_states: Dict[str, TensorState],
        grouped_previous: Dict[str, Dict[str, Any]],
        num_experts: int,
        device: torch.device,
    ) -> Tuple[TensorState, Dict[str, Dict[str, Any]]]:
        final_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        for block_scope, block_state in candidate_block_states.items():
            split_groups = split_state_dict_by_expert(block_state, num_experts=num_experts)[1]
            for expert_scope in grouped_previous[block_scope].get("expert_scopes", []):
                inner = split_groups.get(expert_scope)
                if inner is None:
                    continue
                final_expert_groups[expert_scope] = {
                    "expert_id": int(inner["expert_id"]),
                    "expert_scope": expert_scope,
                    "state_dict": inner["state_dict"],
                }
        merged_state = merge_expert_states_into_global_state(fedavg_base_state, final_expert_groups)
        merged_state = _merge_with_reference_state(fedavg_state, merged_state, device=device)
        return merged_state, final_expert_groups

    def _evaluate_candidate(
        self,
        global_model: nn.Module,
        candidate_state: TensorState,
        val_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module,
        device: torch.device,
        aux_loss_weight: float,
    ) -> Dict[str, torch.Tensor]:
        functional_state: Dict[str, torch.Tensor] = {}
        model_state = global_model.state_dict()
        for name, base_value in model_state.items():
            value = candidate_state.get(name, base_value)
            if torch.is_tensor(value):
                functional_state[name] = value.to(device=device, dtype=base_value.dtype)
            else:
                functional_state[name] = value
        total_loss = None
        total_correct = None
        total_samples = 0
        total_router_entropy = None
        total_router_load_cv = None
        total_class_loss_std = None
        for inputs, labels in val_batches:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, router_stats = functional_call(global_model, functional_state, (inputs,))
            safe_outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)
            batch_loss = loss_fn(safe_outputs, labels) + float(aux_loss_weight) * router_stats.get("aux_loss", safe_outputs.new_tensor(0.0))
            preds = safe_outputs.argmax(dim=-1)
            acc = (preds == labels).float().mean()
            router_counts = router_stats.get("expert_counts")
            if router_counts is None:
                router_load_cv = safe_outputs.new_tensor(0.0)
            else:
                router_counts = router_counts.float()
                router_load_cv = router_counts.std(unbiased=False) / router_counts.mean().clamp_min(1e-12)
            router_probs_by_layer = router_stats.get("router_probs_by_layer")
            if router_probs_by_layer is None:
                router_entropy = safe_outputs.new_tensor(0.0)
            else:
                router_entropy = -(router_probs_by_layer.clamp_min(1e-12) * torch.log(router_probs_by_layer.clamp_min(1e-12))).sum(dim=-1).mean()
            class_loss_std = _compute_class_loss_std(safe_outputs, labels)
            batch_size = int(inputs.size(0))
            total_loss = batch_loss * batch_size if total_loss is None else total_loss + batch_loss * batch_size
            total_correct = acc * batch_size if total_correct is None else total_correct + acc * batch_size
            total_router_entropy = router_entropy * batch_size if total_router_entropy is None else total_router_entropy + router_entropy * batch_size
            total_router_load_cv = router_load_cv * batch_size if total_router_load_cv is None else total_router_load_cv + router_load_cv * batch_size
            total_class_loss_std = class_loss_std * batch_size if total_class_loss_std is None else total_class_loss_std + class_loss_std * batch_size
            total_samples += batch_size
        if total_samples == 0:
            zero = torch.zeros((), device=device)
            return {"val_loss": zero + 10.0, "val_acc": zero, "router_entropy": zero, "router_load_cv": zero, "class_loss_std": zero}
        return {
            "val_loss": torch.nan_to_num(total_loss / total_samples, nan=10.0, posinf=10.0, neginf=10.0),
            "val_acc": torch.nan_to_num(total_correct / total_samples, nan=0.0, posinf=1.0, neginf=0.0),
            "router_entropy": torch.nan_to_num(total_router_entropy / total_samples, nan=0.0, posinf=0.0, neginf=0.0),
            "router_load_cv": torch.nan_to_num(total_router_load_cv / total_samples, nan=0.0, posinf=10.0, neginf=0.0),
            "class_loss_std": torch.nan_to_num(total_class_loss_std / total_samples, nan=0.0, posinf=10.0, neginf=0.0),
        }

    def _compute_log_f(self, particle: MoEStateParticle, ancestor_particle: MoEStateParticle) -> torch.Tensor:
        current = self._particle_to_flat(particle)
        previous = self._particle_to_flat(ancestor_particle).to(device=current.device, dtype=current.dtype)
        mse = (current - previous).pow(2).mean()
        return -float(self.config.transition_coef) * mse

    def _compute_log_q(self, sample: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_std)
        log_prob = -0.5 * (((sample - mean) ** 2) / var + 2.0 * log_std + _LOG_2PI)
        return torch.clamp(torch.nan_to_num(log_prob.sum(), nan=0.0, posinf=_SAFE_LOG_MAX, neginf=_SAFE_LOG_MIN), _SAFE_LOG_MIN, _SAFE_LOG_MAX)

    def _compute_log_g(
        self,
        metrics: Dict[str, torch.Tensor],
        eval_ema: Dict[str, float],
        particle_stats: Dict[str, Any],
        fedavg_reference: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        val_loss = metrics["val_loss"]
        val_acc = metrics["val_acc"]
        router_load_cv = metrics["router_load_cv"]
        expert_conflict = val_loss.new_tensor(float(particle_stats.get("expert_conflict", 0.0)))
        update_norm = val_loss.new_tensor(float(particle_stats.get("update_norm", 0.0)))
        class_loss_std = metrics["class_loss_std"]
        log_g = (
            - float(self.config.val_loss_coef) * val_loss
            + float(self.config.val_loss_improve_coef) * (float(eval_ema.get("eval_loss_ema", 0.0)) - val_loss)
            + float(self.config.val_acc_improve_coef) * (val_acc - float(eval_ema.get("eval_acc_ema", 0.0)))
            - float(self.config.router_balance_coef) * router_load_cv
            - float(self.config.expert_conflict_coef) * expert_conflict
            - float(self.config.update_norm_coef) * update_norm / max(float(eval_ema.get("mean_expert_update_norm", 1.0)), 1e-12)
            - float(self.config.class_balance_coef) * class_loss_std
        )
        return float(self.config.reward_temperature) * log_g

    def _normalize_weights(self, scores: torch.Tensor) -> torch.Tensor:
        if self.config.selector_mode == "oracle":
            weights = torch.zeros_like(scores)
            if weights.numel() > 0:
                weights[int(torch.argmax(scores).item())] = 1.0
            return weights
        if self.config.selector_mode == "reward_softmax":
            return torch.softmax(scores, dim=0)
        if self.config.selector_mode == "uniform":
            return torch.full_like(scores, 1.0 / max(scores.numel(), 1))
        return torch.softmax(scores, dim=0)

    def _resample_particles(self, weights: torch.Tensor, structure: Dict[str, Any], device: torch.device) -> bool:
        normalized_ess = effective_sample_size(weights.detach().cpu().tolist()) / max(float(weights.numel()), 1.0)
        if normalized_ess >= float(self.config.ess_threshold):
            return False
        indices = torch.multinomial(weights.detach(), num_samples=int(weights.numel()), replacement=True)
        new_particles = []
        for idx in indices.tolist():
            ancestor = self.state.particles[idx]
            base_flat = self._particle_to_flat(ancestor).detach().clone()
            jitter = torch.randn_like(base_flat) * float(self.config.jitter_std)
            new_flat = base_flat + jitter
            if bool(self.config.detach_resampling):
                new_flat = new_flat.detach()
            particle = self._flat_to_particle(
                new_flat,
                structure,
                particle_id=self.state.particle_counter,
                ancestor_id=ancestor.particle_id,
                prev_log_weight=0.0,
            )
            self.state.particle_counter += 1
            new_particles.append(particle)
        self.state.particles = new_particles
        return True

    def _compute_grad_norm(self) -> float:
        if self.proposal_net is None:
            return 0.0
        sq_sum = 0.0
        for param in self.proposal_net.parameters():
            if param.grad is None:
                continue
            grad = _sanitize_tensor(param.grad.detach().float())
            sq_sum += float(grad.pow(2).sum().item())
        return math.sqrt(max(sq_sum, 0.0))

    def _compute_param_delta_norm(self, before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> float:
        sq_sum = 0.0
        for key, before_tensor in before.items():
            after_tensor = after[key]
            if torch.is_tensor(before_tensor) and torch.is_tensor(after_tensor):
                sq_sum += float((after_tensor.detach().float() - before_tensor.detach().float()).pow(2).sum().item())
        return math.sqrt(max(sq_sum, 0.0))

    def aggregate_model_state(
        self,
        global_model: nn.Module,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        num_experts: int,
        device: torch.device,
        val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
        loss_fn: nn.Module,
        round_idx: int,
        aux_loss_weight: float = 0.0,
    ) -> Tuple[TensorState, List[Dict[str, Any]]]:
        self.state.round_idx = int(round_idx)
        sample_counts = [int(num_samples) for _, num_samples, _ in client_updates]
        sample_weights = _sample_count_weights(sample_counts).to(device=device, dtype=torch.float32)
        fedavg_state = weighted_average_state_dicts([state_dict for state_dict, _, _ in client_updates], sample_weights, fallback_state=global_state)
        fedavg_state = _merge_with_reference_state(global_state, fedavg_state, device=device)
        fedavg_base_state, fedavg_expert_groups = split_state_dict_by_expert(fedavg_state, num_experts=num_experts)
        _, global_expert_groups = split_state_dict_by_expert(global_state, num_experts=num_experts)
        client_expert_groups = [split_state_dict_by_expert(state_dict, num_experts=num_experts)[1] for state_dict, _, _ in client_updates]
        grouped_previous = group_expert_groups_by_block(global_expert_groups)
        grouped_fedavg = group_expert_groups_by_block(fedavg_expert_groups)
        grouped_clients = [group_expert_groups_by_block(group_map) for group_map in client_expert_groups]
        structure = self._build_structure(grouped_previous, num_clients=len(client_updates))
        self._init_network_if_needed(structure)
        assert self.proposal_net is not None and self.optimizer is not None
        self.proposal_net.to(device)
        val_batches = self._materialize_val_batches(val_loader, device=device)
        self._ensure_particles(structure, device)

        block_client_delta_states = collections.OrderedDict()
        for block_scope in structure["block_scopes"]:
            previous_state = grouped_previous[block_scope]["state_dict"]
            client_states = [group_map[block_scope]["state_dict"] for group_map in grouped_clients if block_scope in group_map]
            block_client_delta_states[block_scope] = self._compute_block_client_deltas(previous_state, client_states, device=device)

        prev_eval = {
            "prev_eval_loss": self.state.prev_eval_loss,
            "prev_eval_acc": self.state.prev_eval_acc,
            "eval_loss_ema": self.state.eval_loss_ema,
            "eval_acc_ema": self.state.eval_acc_ema,
            "eval_loss_improvement": self.state.prev_eval_loss - self.state.eval_loss_ema,
            "eval_acc_improvement": self.state.prev_eval_acc - self.state.eval_acc_ema,
            "previous_ess": self.state.previous_ess,
            "previous_reward_std": self.state.previous_reward_std,
            "previous_selected_particle_entropy": self.state.previous_selected_particle_entropy,
            "previous_fedavg_weight_proxy": self.state.previous_fedavg_weight_proxy,
        }
        diagnostics = collect_moe_fl_diagnostics(
            global_state=global_state,
            client_updates=client_updates,
            block_client_delta_states=block_client_delta_states,
            grouped_global=grouped_previous,
            prev_eval_metrics=prev_eval,
            round_idx=round_idx,
            total_rounds=int(self.config.total_rounds),
            device=device,
        )
        diagnostics_tensor = self._diagnostics_to_tensor(diagnostics, device=device)

        fedavg_reference_metrics = self._evaluate_candidate(global_model, fedavg_state, val_batches, loss_fn, device, aux_loss_weight)
        oracle_logg = []
        particle_rows = []
        block_rows = []
        candidate_block_states_per_particle = []
        candidate_global_states = []
        log_f_values = []
        log_g_values = []
        log_q_values = []
        sampled_particles = []
        for particle in self.state.particles:
            prev_flat = self._particle_to_flat(particle).to(device=device)
            mean_delta, log_std = self.proposal_net(diagnostics_tensor, prev_flat)
            eps = torch.randn_like(mean_delta)
            sampled_flat = prev_flat + mean_delta + torch.exp(log_std) * eps
            sampled_particle = self._flat_to_particle(
                sampled_flat,
                structure,
                particle_id=particle.particle_id,
                ancestor_id=particle.particle_id,
                prev_log_weight=float(particle.prev_log_weight.detach().item()),
            )
            candidate_blocks, particle_stats, per_block_metrics = self._apply_particle_policy(
                sampled_particle,
                grouped_previous=grouped_previous,
                grouped_clients=grouped_clients,
                sample_weight_tensor=sample_weights,
                structure=structure,
                device=device,
            )
            candidate_state, _ = self._build_candidate_global_state(
                fedavg_state=fedavg_state,
                fedavg_base_state=fedavg_base_state,
                candidate_block_states=candidate_blocks,
                grouped_previous=grouped_previous,
                num_experts=num_experts,
                device=device,
            )
            metrics = self._evaluate_candidate(global_model, candidate_state, val_batches, loss_fn, device, aux_loss_weight)
            log_g = self._compute_log_g(metrics, {**diagnostics, **prev_eval}, particle_stats, fedavg_reference_metrics)
            log_f = self._compute_log_f(sampled_particle, particle)
            log_q = self._compute_log_q(sampled_flat, prev_flat + mean_delta, log_std)
            sampled_particles.append(sampled_particle)
            candidate_block_states_per_particle.append(candidate_blocks)
            candidate_global_states.append(candidate_state)
            log_f_values.append(log_f)
            log_g_values.append(log_g)
            log_q_values.append(log_q)
            oracle_logg.append(float(log_g.detach().item()))
            particle_rows.append({
                "round": int(round_idx),
                "particle_id": int(sampled_particle.particle_id),
                "ancestor_id": int(sampled_particle.ancestor_id),
                "weight": 0.0,
                "log_weight": 0.0,
                "log_g": float(log_g.detach().item()),
                "log_q": float(log_q.detach().item()),
                "log_f": float(log_f.detach().item()),
                "val_loss": float(metrics["val_loss"].detach().item()),
                "val_acc": float(metrics["val_acc"].detach().item()),
                "router_load_cv": float(metrics["router_load_cv"].detach().item()),
                "router_entropy": float(metrics["router_entropy"].detach().item()),
                "expert_conflict": float(particle_stats["expert_conflict"]),
                "update_norm": float(particle_stats["update_norm"]),
                "class_loss_std": float(metrics["class_loss_std"].detach().item()),
                **particle_stats,
            })
            for block_metric in per_block_metrics:
                block_rows.append({
                    "round": int(round_idx),
                    "particle_id": int(sampled_particle.particle_id),
                    **block_metric,
                })

        log_f_tensor = torch.stack(log_f_values, dim=0)
        log_g_tensor = torch.stack(log_g_values, dim=0)
        log_q_tensor = torch.stack(log_q_values, dim=0)
        log_weight_tensor = torch.clamp(
            torch.nan_to_num(log_f_tensor + log_g_tensor - float(self.config.logq_coef) * log_q_tensor, nan=_SAFE_LOG_MIN, posinf=_SAFE_LOG_MAX, neginf=_SAFE_LOG_MIN),
            _SAFE_LOG_MIN,
            _SAFE_LOG_MAX,
        )
        selector_scores = log_weight_tensor if self.config.selector_mode == "vsmc" else log_g_tensor
        weights = self._normalize_weights(selector_scores)
        elite_top_k = min(max(int(self.config.elite_top_k), 1), int(weights.numel()))
        elite_idx = torch.topk(weights, k=elite_top_k).indices
        elite_weights = weights[elite_idx] / weights[elite_idx].sum().clamp_min(1e-12)

        final_block_states: Dict[str, TensorState] = collections.OrderedDict()
        for block_scope in structure["block_scopes"]:
            elite_pairs = []
            for local_idx, particle_idx in enumerate(elite_idx.tolist()):
                elite_pairs.append((candidate_block_states_per_particle[particle_idx][block_scope], elite_weights[local_idx]))
            final_block_states[block_scope] = _combine_weighted_states(elite_pairs)
        aggregated_state, final_expert_groups = self._build_candidate_global_state(
            fedavg_state=fedavg_state,
            fedavg_base_state=fedavg_base_state,
            candidate_block_states=final_block_states,
            grouped_previous=grouped_previous,
            num_experts=num_experts,
            device=device,
        )

        proposal_loss = None
        proposal_grad_norm = 0.0
        proposal_param_delta_norm = 0.0
        if bool(self.config.train_proposal):
            proposal_loss = -(torch.logsumexp(log_weight_tensor, dim=0) - math.log(max(int(log_weight_tensor.numel()), 1)))
            if torch.isfinite(proposal_loss).item():
                before = {name: tensor.detach().clone() for name, tensor in self.proposal_net.state_dict().items()}
                self.optimizer.zero_grad()
                proposal_loss.backward()
                proposal_grad_norm = self._compute_grad_norm()
                clip_grad_norm_(self.proposal_net.parameters(), max_norm=float(self.config.grad_clip))
                self.optimizer.step()
                proposal_param_delta_norm = self._compute_param_delta_norm(before, self.proposal_net.state_dict())
        normalized_ess = effective_sample_size(weights.detach().cpu().tolist()) / max(float(weights.numel()), 1.0)
        weight_entropy = float((-(weights * torch.log(weights.clamp_min(1e-12))).sum()).detach().item())
        resampled = self._resample_particles(weights, structure, device=device)
        if not resampled:
            self.state.particles = [self._clone_particle(p, detach=True) for p in sampled_particles]
        for block_scope in structure["block_scopes"]:
            prev_m = self.state.block_momentum.get(block_scope)
            fedavg_delta = _subtract_float_states(grouped_fedavg[block_scope]["state_dict"], grouped_previous[block_scope]["state_dict"], device=device, detach=True)
            if prev_m is None:
                self.state.block_momentum[block_scope] = clone_detached_state_dict(fedavg_delta)
            else:
                updated = collections.OrderedDict()
                for key in _ordered_union_keys([prev_m, fedavg_delta]):
                    pv = prev_m.get(key)
                    dv = fedavg_delta.get(key)
                    if torch.is_tensor(pv) and torch.is_tensor(dv) and torch.is_floating_point(pv) and torch.is_floating_point(dv):
                        updated[key] = _sanitize_tensor(float(self.config.momentum_beta) * pv.to(device=device, dtype=torch.float32) + dv.to(device=device, dtype=torch.float32))
                self.state.block_momentum[block_scope] = updated
        reward_std = float(log_g_tensor.detach().std(unbiased=False).item()) if log_g_tensor.numel() > 1 else 0.0
        self.state.previous_ess = normalized_ess
        self.state.previous_reward_std = reward_std
        self.state.previous_selected_particle_entropy = weight_entropy
        self.state.previous_fedavg_weight_proxy = float(weights[0].item()) if weights.numel() else 1.0
        top_particle_idx = int(torch.argmax(weights).item()) if weights.numel() else -1
        self.state.last_round_summary = {
            "round": int(round_idx),
            "seed": None,
            "method": "learned_vsmc_moe_state_policy_k8",
            "num_particles": int(self.config.num_particles),
            "elite_top_k": int(elite_top_k),
            "normalized_ess": float(normalized_ess),
            "resampled": bool(resampled),
            "proposal_loss": float(proposal_loss.detach().item()) if proposal_loss is not None and torch.isfinite(proposal_loss).item() else float("nan"),
            "proposal_grad_norm": float(proposal_grad_norm),
            "proposal_param_delta_norm": float(proposal_param_delta_norm),
            "logq_coef": float(self.config.logq_coef),
            "reward_temperature": float(self.config.reward_temperature),
            "logf_mean": float(log_f_tensor.detach().mean().item()) if log_f_tensor.numel() else 0.0,
            "logf_std": float(log_f_tensor.detach().std(unbiased=False).item()) if log_f_tensor.numel() > 1 else 0.0,
            "logg_mean": float(log_g_tensor.detach().mean().item()) if log_g_tensor.numel() else 0.0,
            "logg_std": float(log_g_tensor.detach().std(unbiased=False).item()) if log_g_tensor.numel() > 1 else 0.0,
            "logq_mean": float(log_q_tensor.detach().mean().item()) if log_q_tensor.numel() else 0.0,
            "logq_std": float(log_q_tensor.detach().std(unbiased=False).item()) if log_q_tensor.numel() > 1 else 0.0,
            "log_weight_mean": float(log_weight_tensor.detach().mean().item()) if log_weight_tensor.numel() else 0.0,
            "log_weight_std": float(log_weight_tensor.detach().std(unbiased=False).item()) if log_weight_tensor.numel() > 1 else 0.0,
            "particle_weight_entropy": float(weight_entropy),
            "top_particle_idx": int(sampled_particles[top_particle_idx].particle_id) if top_particle_idx >= 0 else -1,
            "top_particle_weight": float(weights[top_particle_idx].item()) if top_particle_idx >= 0 else 0.0,
            "top_particle_logg": float(log_g_tensor[top_particle_idx].detach().item()) if top_particle_idx >= 0 else 0.0,
            "top_particle_logq": float(log_q_tensor[top_particle_idx].detach().item()) if top_particle_idx >= 0 else 0.0,
            "top_particle_logf": float(log_f_tensor[top_particle_idx].detach().item()) if top_particle_idx >= 0 else 0.0,
            "oracle_particle_idx": int(sampled_particles[int(torch.argmax(log_g_tensor).item())].particle_id) if log_g_tensor.numel() else -1,
            "oracle_particle_logg": float(log_g_tensor.max().detach().item()) if log_g_tensor.numel() else 0.0,
            "reward_std": float(reward_std),
            "val_loss_mean_particles": float(torch.stack([row["val_loss"] if torch.is_tensor(row["val_loss"]) else torch.tensor(row["val_loss"]) for row in []]).mean().item()) if False else float(torch.tensor([row["val_loss"] for row in particle_rows], dtype=torch.float32).mean().item()) if particle_rows else 0.0,
            "val_loss_best_particle": float(particle_rows[top_particle_idx]["val_loss"]) if top_particle_idx >= 0 else 0.0,
            "val_acc_mean_particles": float(torch.tensor([row["val_acc"] for row in particle_rows], dtype=torch.float32).mean().item()) if particle_rows else 0.0,
            "val_acc_best_particle": float(particle_rows[top_particle_idx]["val_acc"]) if top_particle_idx >= 0 else 0.0,
            "fedavg_reference_val_loss": float(fedavg_reference_metrics["val_loss"].detach().item()),
            "fedavg_reference_val_acc": float(fedavg_reference_metrics["val_acc"].detach().item()),
            "best_minus_fedavg_val_loss": float(fedavg_reference_metrics["val_loss"].detach().item() - particle_rows[top_particle_idx]["val_loss"]) if top_particle_idx >= 0 else 0.0,
            "best_minus_fedavg_val_acc": float(particle_rows[top_particle_idx]["val_acc"] - fedavg_reference_metrics["val_acc"].detach().item()) if top_particle_idx >= 0 else 0.0,
        }
        for row, weight, log_weight, log_f, log_g, log_q in zip(particle_rows, weights.detach().cpu().tolist(), log_weight_tensor.detach().cpu().tolist(), log_f_tensor.detach().cpu().tolist(), log_g_tensor.detach().cpu().tolist(), log_q_tensor.detach().cpu().tolist()):
            row["weight"] = float(weight)
            row["log_weight"] = float(log_weight)
            row["log_f"] = float(log_f)
            row["log_g"] = float(log_g)
            row["log_q"] = float(log_q)
        if self.metrics is not None:
            self.metrics.write_particles(particle_rows)
            self.metrics.write_blocks(block_rows)
        round_infos = [{
            "expert_scope": scope,
            "expert_id": idx,
            "normalized_ess": float(normalized_ess),
        } for idx, scope in enumerate(structure["block_scopes"])]
        return aggregated_state, round_infos

    def record_post_eval_metrics(self, round_idx: int, server_test_loss: float, server_test_acc: float):
        self.state.prev_eval_loss = float(server_test_loss)
        self.state.prev_eval_acc = float(server_test_acc)
        if self.state.round_idx <= 1:
            self.state.eval_loss_ema = float(server_test_loss)
            self.state.eval_acc_ema = float(server_test_acc)
        else:
            self.state.eval_loss_ema = 0.8 * self.state.eval_loss_ema + 0.2 * float(server_test_loss)
            self.state.eval_acc_ema = 0.8 * self.state.eval_acc_ema + 0.2 * float(server_test_acc)
        self.state.best_acc_so_far = max(self.state.best_acc_so_far, float(server_test_acc))
        if self.metrics is None:
            return
        row = dict(self.state.last_round_summary)
        row.update({
            "server_test_acc": float(server_test_acc),
            "server_test_loss": float(server_test_loss),
            "best_acc_so_far": float(self.state.best_acc_so_far),
            "final_acc": float(server_test_acc),
            "min_eval_loss_so_far": float(min(self.state.eval_loss_ema, float(server_test_loss))),
        })
        self.metrics.write_round(row)
