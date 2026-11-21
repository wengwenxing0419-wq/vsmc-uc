"""Dynamic Beta-precision FedTRA aggregation.

This module adds a separate aggregation path. It keeps the original FedTRA
observation semantics (routing support and leave-one-out update consistency),
uses a bootstrap particle filter for the scalar latent reliability state, and
updates the two Beta concentration shares with negative-entropy OMD.

The OMD objective is the Jensen upper bound of the one-step predictive
negative log likelihood. The original VSMC aggregation path is not modified.
"""
from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from aggregation.vsmc_expert_aggregator import (
    merge_expert_states_into_global_state,
    split_state_dict_by_expert,
    weighted_average_state_dicts,
)

TensorState = Dict[str, Any]


@dataclass
class BetaPrecisionFedTRAConfig:
    num_particles: int = 32
    kappa_u: float = 10.0
    kappa_c: float = 10.0
    omd_lr: float = 0.05
    gamma_min: float = 1e-3
    transition_rho: float = 0.9
    transition_std: float = 0.15
    initial_std: float = 1.0
    mean_slope_u: float = 1.0
    mean_slope_c: float = 1.0
    mean_intercept_c: float = 0.0
    eps: float = 1e-6
    seed: int = 42
    log_dir: str = "logs"
    log_csv_name: str = "beta_precision_fedtra_round_metrics.csv"
    pair_log_csv_name: str = "beta_precision_fedtra_pair_metrics.csv"

    def validate(self) -> None:
        if self.num_particles < 2:
            raise ValueError("num_particles must be at least 2")
        if self.kappa_u <= 0.0 or self.kappa_c <= 0.0:
            raise ValueError("kappa_u and kappa_c must be positive")
        if self.omd_lr <= 0.0:
            raise ValueError("omd_lr must be positive")
        if not 0.0 < self.gamma_min < 0.5:
            raise ValueError("gamma_min must be in (0, 0.5)")
        if not 0.0 < self.transition_rho < 1.0:
            raise ValueError("transition_rho must be in (0, 1)")
        if self.transition_std <= 0.0 or self.initial_std <= 0.0:
            raise ValueError("transition_std and initial_std must be positive")
        if self.mean_slope_u <= 0.0 or self.mean_slope_c <= 0.0:
            raise ValueError("Beta mean slopes must be positive")
        if not 0.0 < self.eps < 0.5:
            raise ValueError("eps must be in (0, 0.5)")


class BetaPrecisionFedTRAggregator:
    """FedTRA-style expert aggregation with adaptive Beta precision shares."""

    def __init__(self, config: BetaPrecisionFedTRAConfig, logger: Optional[Any] = None):
        config.validate()
        self.config = config
        self.logger = logger
        self.total_concentration = float(config.kappa_u + config.kappa_c)
        gamma0 = torch.tensor(
            [config.kappa_u / self.total_concentration,
             config.kappa_c / self.total_concentration],
            dtype=torch.float64,
        )
        if float(gamma0.min().item()) <= config.gamma_min:
            raise ValueError(
                "gamma_min must be smaller than both initial concentration shares"
            )
        self.gamma = gamma0
        self.latent_memory: Dict[str, Dict[str, torch.Tensor]] = {}
        self.round_idx = 0
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(config.seed))

    @staticmethod
    def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            return float(value.detach().reshape(-1)[0].item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _indexed_value(container: Any, index: int) -> Optional[float]:
        if isinstance(container, dict):
            value = container.get(index, container.get(str(index)))
            return BetaPrecisionFedTRAggregator._as_float(value)
        if torch.is_tensor(container):
            flat = container.detach().reshape(-1)
            if 0 <= index < flat.numel():
                return float(flat[index].item())
            return None
        if isinstance(container, (list, tuple)) and 0 <= index < len(container):
            return BetaPrecisionFedTRAggregator._as_float(container[index])
        return None

    @staticmethod
    def _sum_indexed_values(container: Any) -> float:
        if isinstance(container, dict):
            values = container.values()
        elif torch.is_tensor(container):
            values = container.detach().reshape(-1).tolist()
        elif isinstance(container, (list, tuple)):
            values = container
        else:
            return 0.0
        return sum(
            max(BetaPrecisionFedTRAggregator._as_float(value, 0.0) or 0.0, 0.0)
            for value in values
        )

    @staticmethod
    def _layer_index(expert_scope: str) -> Optional[int]:
        match = re.search(
            r"(?:blocks|layers|layer|transformer\.h)\.(\d+)",
            str(expert_scope),
        )
        return int(match.group(1)) if match else None

    def _extract_usage(
        self,
        metrics: Dict[str, Any],
        expert_scope: str,
        expert_id: int,
        num_experts: int,
    ) -> Tuple[float, float, bool]:
        """Return (routing fraction, route count, active) for one client-expert."""
        eps = self.config.eps
        layer_idx = self._layer_index(expert_scope)

        # Prefer training-time routing statistics: FedTRA's u is based on
        # tokens routed during local training, not validation-time routing.
        for prefix in ("train_", ""):
            layer_counts = metrics.get(prefix + "layer_route_counts")
            if layer_idx is not None and isinstance(layer_counts, dict):
                layer_values = layer_counts.get(layer_idx, layer_counts.get(str(layer_idx)))
                if layer_values is not None:
                    route_count = self._indexed_value(layer_values, expert_id)
                    if route_count is not None:
                        total = self._sum_indexed_values(layer_values)
                        usage = route_count / max(total, eps) if total > 0.0 else 0.0
                        return (
                            min(max(usage, 0.0), 1.0),
                            max(route_count, 0.0),
                            route_count > 0.0,
                        )

        for prefix in ("train_", ""):
            route_count = self._indexed_value(
                metrics.get(prefix + "route_counts"), expert_id
            )
            usage = self._indexed_value(
                metrics.get(prefix + "expert_usage"), expert_id
            )
            if usage is None and route_count is not None:
                total_tokens = self._as_float(
                    metrics.get(prefix + "num_tokens"), 0.0
                ) or 0.0
                usage = (
                    route_count / max(total_tokens, eps)
                    if total_tokens > 0.0
                    else 0.0
                )
            if usage is not None:
                active = route_count > 0.0 if route_count is not None else usage > 0.0
                return (
                    min(max(float(usage), 0.0), 1.0),
                    max(float(route_count or 0.0), 0.0),
                    bool(active),
                )

        # Missing metrics are treated as neutral evidence rather than silently
        # dropping a client update. Normal training runs provide these metrics.
        return 1.0 / max(int(num_experts), 1), 1.0, True

    @staticmethod
    def _flatten_delta(current: TensorState, previous: TensorState) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for key in sorted(set(current).intersection(previous)):
            current_value = current[key]
            previous_value = previous[key]
            if not torch.is_tensor(current_value) or not torch.is_tensor(previous_value):
                continue
            if not torch.is_floating_point(current_value) or not torch.is_floating_point(previous_value):
                continue
            if current_value.shape != previous_value.shape:
                continue
            delta = (
                current_value.detach().to(device="cpu", dtype=torch.float64)
                - previous_value.detach().to(device="cpu", dtype=torch.float64)
            ).reshape(-1)
            pieces.append(delta)
        if not pieces:
            return torch.empty(0, dtype=torch.float64)
        return torch.cat(pieces, dim=0)

    def _leave_one_out_consistency(
        self,
        client_states: Sequence[TensorState],
        previous_state: TensorState,
    ) -> List[float]:
        """Compute c=(1+cos(local direction, LOO consensus))/2."""
        if len(client_states) <= 1:
            return [0.5 for _ in client_states]

        directions: List[torch.Tensor] = []
        for state in client_states:
            delta = self._flatten_delta(state, previous_state)
            norm = torch.linalg.vector_norm(delta) if delta.numel() else torch.tensor(0.0)
            if delta.numel() == 0 or float(norm.item()) <= self.config.eps:
                directions.append(torch.zeros_like(delta))
            else:
                directions.append(delta / norm)

        scores: List[float] = []
        for idx, local_direction in enumerate(directions):
            if local_direction.numel() == 0 or float(
                torch.linalg.vector_norm(local_direction).item()
            ) <= self.config.eps:
                cosine = 0.0
            else:
                others = [
                    direction
                    for other_idx, direction in enumerate(directions)
                    if other_idx != idx and direction.numel() == local_direction.numel()
                ]
                if not others:
                    cosine = 0.0
                else:
                    consensus = torch.stack(others, dim=0).sum(dim=0)
                    consensus_norm = torch.linalg.vector_norm(consensus)
                    if float(consensus_norm.item()) <= self.config.eps:
                        cosine = 0.0
                    else:
                        consensus = consensus / consensus_norm
                        cosine = float(torch.dot(local_direction, consensus).item())
            cosine = min(max(cosine, -1.0), 1.0)
            scores.append(min(max(0.5 * (1.0 + cosine), 0.0), 1.0))
        return scores

    def _predict_particles(self, state_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        count = int(self.config.num_particles)
        previous = self.latent_memory.get(state_key)
        if previous is None:
            particles = (
                torch.randn(count, generator=self.generator, dtype=torch.float64)
                * float(self.config.initial_std)
            )
        else:
            previous_particles = previous["particles"].to(dtype=torch.float64, device="cpu")
            previous_weights = previous["weights"].to(dtype=torch.float64, device="cpu")
            previous_weights = previous_weights.clamp_min(0.0)
            if (
                previous_weights.numel() != previous_particles.numel()
                or not torch.isfinite(previous_weights).all()
                or float(previous_weights.sum().item()) <= self.config.eps
            ):
                previous_weights = torch.full_like(
                    previous_particles, 1.0 / max(previous_particles.numel(), 1)
                )
            else:
                previous_weights = previous_weights / previous_weights.sum()
            ancestors = torch.multinomial(
                previous_weights,
                num_samples=count,
                replacement=True,
                generator=self.generator,
            )
            noise = torch.randn(count, generator=self.generator, dtype=torch.float64)
            particles = (
                float(self.config.transition_rho) * previous_particles[ancestors]
                + float(self.config.transition_std) * noise
            )
        predictive_weights = torch.full(
            (count,), 1.0 / count, dtype=torch.float64
        )
        return particles, predictive_weights

    def _mean_functions(
        self,
        particles: torch.Tensor,
        num_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_usage = min(
            max(1.0 / max(int(num_experts), 1), self.config.eps),
            1.0 - self.config.eps,
        )
        intercept_u = math.log(target_usage / (1.0 - target_usage))
        mu_u = torch.sigmoid(
            float(self.config.mean_slope_u) * particles + intercept_u
        )
        mu_c = torch.sigmoid(
            float(self.config.mean_slope_c) * particles
            + float(self.config.mean_intercept_c)
        )
        return mu_u.clamp(self.config.eps, 1.0 - self.config.eps), mu_c.clamp(
            self.config.eps, 1.0 - self.config.eps
        )

    @staticmethod
    def _beta_log_prob(
        observation: float,
        concentration: float,
        mean: torch.Tensor,
    ) -> torch.Tensor:
        x = min(max(float(observation), 1e-12), 1.0 - 1e-12)
        alpha = float(concentration) * mean
        beta = float(concentration) * (1.0 - mean)
        log_normalizer = (
            torch.lgamma(alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
        )
        return (
            log_normalizer
            + (alpha - 1.0) * math.log(x)
            + (beta - 1.0) * math.log1p(-x)
        )

    def _channel_concentration_gradient(
        self,
        observation: float,
        concentration: float,
        mean: torch.Tensor,
        predictive_weights: torch.Tensor,
    ) -> torch.Tensor:
        x = min(max(float(observation), self.config.eps), 1.0 - self.config.eps)
        alpha = float(concentration) * mean
        beta = float(concentration) * (1.0 - mean)
        derivative = (
            mean * torch.digamma(alpha)
            + (1.0 - mean) * torch.digamma(beta)
            - torch.digamma(torch.as_tensor(concentration, dtype=torch.float64))
            - mean * math.log(x)
            - (1.0 - mean) * math.log1p(-x)
        )
        return float(self.total_concentration) * torch.sum(
            predictive_weights * derivative
        )

    def _score_pair(
        self,
        u_value: float,
        c_value: float,
        particles: torch.Tensor,
        predictive_weights: torch.Tensor,
        gamma: torch.Tensor,
        num_experts: int,
    ) -> Dict[str, Any]:
        mu_u, mu_c = self._mean_functions(particles, num_experts)
        concentration_u = self.total_concentration * float(gamma[0].item())
        concentration_c = self.total_concentration * float(gamma[1].item())

        logp_u = self._beta_log_prob(u_value, concentration_u, mu_u)
        logp_c = self._beta_log_prob(c_value, concentration_c, mu_c)
        logp = logp_u + logp_c

        log_predictive_weights = torch.log(
            predictive_weights.clamp_min(self.config.eps)
        )
        predictive_nll = -torch.logsumexp(
            log_predictive_weights + logp, dim=0
        )
        surrogate_nll = -torch.sum(predictive_weights * logp)

        grad_u = self._channel_concentration_gradient(
            u_value, concentration_u, mu_u, predictive_weights
        )
        grad_c = self._channel_concentration_gradient(
            c_value, concentration_c, mu_c, predictive_weights
        )

        posterior_weights = torch.softmax(
            log_predictive_weights + logp, dim=0
        )
        posterior_weights = posterior_weights / posterior_weights.sum().clamp_min(
            self.config.eps
        )
        reliability = torch.sum(
            posterior_weights * torch.sigmoid(particles)
        )
        ess = 1.0 / torch.sum(posterior_weights.square()).clamp_min(
            self.config.eps
        )

        return {
            "predictive_nll": float(predictive_nll.item()),
            "surrogate_nll": float(surrogate_nll.item()),
            "grad_u": float(grad_u.item()),
            "grad_c": float(grad_c.item()),
            "posterior_weights": posterior_weights.detach(),
            "reliability": float(reliability.item()),
            "ess": float(ess.item()),
        }

    def _update_gamma(self, gradient: torch.Tensor) -> torch.Tensor:
        eps = float(self.config.gamma_min)
        gamma = self.gamma.clone()
        pi = (gamma - eps) / (1.0 - 2.0 * eps)
        pi = pi.clamp_min(1e-12)
        pi = pi / pi.sum()
        transformed_gradient = (1.0 - 2.0 * eps) * gradient
        logits = torch.log(pi) - float(self.config.omd_lr) * transformed_gradient
        next_pi = torch.softmax(logits, dim=0)
        next_gamma = eps + (1.0 - 2.0 * eps) * next_pi
        return next_gamma / next_gamma.sum()

    def _append_metrics(
        self,
        round_info: Dict[str, Any],
        pair_records: Sequence[Dict[str, Any]],
    ) -> None:
        os.makedirs(self.config.log_dir, exist_ok=True)
        round_path = os.path.join(self.config.log_dir, self.config.log_csv_name)
        round_fields = [
            "round",
            "num_pairs",
            "gamma_u_before",
            "gamma_c_before",
            "gamma_u_after",
            "gamma_c_after",
            "predictive_nll",
            "surrogate_nll",
            "gradient_u",
            "gradient_c",
            "mean_ess",
            "mean_reliability",
        ]
        write_header = not os.path.exists(round_path)
        with open(round_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=round_fields)
            if write_header:
                writer.writeheader()
            writer.writerow({key: round_info.get(key, 0.0) for key in round_fields})

        pair_path = os.path.join(self.config.log_dir, self.config.pair_log_csv_name)
        pair_fields = [
            "round",
            "expert_scope",
            "expert_id",
            "client_id",
            "u",
            "c",
            "reliability",
            "ess",
            "predictive_nll",
            "surrogate_nll",
        ]
        write_header = not os.path.exists(pair_path)
        with open(pair_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=pair_fields)
            if write_header:
                writer.writeheader()
            for record in pair_records:
                writer.writerow({key: record.get(key, "") for key in pair_fields})

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        num_experts: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[TensorState, Dict[str, Any]]:
        if not client_updates:
            raise ValueError("client_updates must not be empty")
        if int(num_experts) <= 0:
            raise ValueError("num_experts must be positive")

        client_states = [state for state, _, _ in client_updates]
        sample_counts = [max(int(count), 0) for _, count, _ in client_updates]
        client_metrics = [dict(metrics or {}) for _, _, metrics in client_updates]

        sample_tensor = torch.as_tensor(sample_counts, dtype=torch.float64)
        if float(sample_tensor.sum().item()) <= self.config.eps:
            fedavg_weights = torch.full_like(
                sample_tensor, 1.0 / max(sample_tensor.numel(), 1)
            )
        else:
            fedavg_weights = sample_tensor / sample_tensor.sum()
        fedavg_state = weighted_average_state_dicts(
            client_states, fedavg_weights, fallback_state=global_state
        )

        base_state, fedavg_groups = split_state_dict_by_expert(
            fedavg_state, num_experts=num_experts
        )
        _, previous_groups = split_state_dict_by_expert(
            global_state, num_experts=num_experts
        )
        client_groups = [
            split_state_dict_by_expert(state, num_experts=num_experts)[1]
            for state in client_states
        ]

        scopes = list(
            dict.fromkeys(
                list(previous_groups.keys())
                + list(fedavg_groups.keys())
                + [
                    scope
                    for group_map in client_groups
                    for scope in group_map.keys()
                ]
            )
        )

        gamma_before = self.gamma.clone()
        pair_records: List[Dict[str, Any]] = []
        per_expert_info: List[Dict[str, Any]] = []
        gradient_sum = torch.zeros(2, dtype=torch.float64)
        predictive_loss_sum = 0.0
        surrogate_loss_sum = 0.0
        pair_count = 0
        ess_values: List[float] = []
        reliability_values: List[float] = []

        merged_expert_groups: Dict[str, Dict[str, Any]] = {}
        for expert_scope in scopes:
            previous_group = previous_groups.get(expert_scope)
            if previous_group is None:
                previous_group = fedavg_groups.get(expert_scope)
            if previous_group is None:
                continue
            previous_expert_state = previous_group["state_dict"]
            expert_id = int(previous_group.get("expert_id", 0))

            active_indices: List[int] = []
            active_states: List[TensorState] = []
            active_metrics: List[Dict[str, Any]] = []
            active_client_ids: List[Any] = []
            usages: List[float] = []
            route_counts: List[float] = []

            for client_idx, (group_map, metrics) in enumerate(
                zip(client_groups, client_metrics)
            ):
                if expert_scope not in group_map:
                    continue
                usage, route_count, is_active = self._extract_usage(
                    metrics, expert_scope, expert_id, int(num_experts)
                )
                if not is_active:
                    continue
                active_indices.append(client_idx)
                active_states.append(group_map[expert_scope]["state_dict"])
                active_metrics.append(metrics)
                active_client_ids.append(
                    metrics.get("client_id", metrics.get("_client_id", client_idx))
                )
                usages.append(usage)
                route_counts.append(route_count)

            if not active_states:
                merged_expert_groups[expert_scope] = {
                    "expert_id": expert_id,
                    "expert_scope": expert_scope,
                    "state_dict": previous_expert_state,
                }
                per_expert_info.append(
                    {
                        "expert_scope": expert_scope,
                        "expert_id": expert_id,
                        "num_active_clients": 0,
                        "weights": [],
                        "fallback_used": True,
                        "fallback_reason": "no_active_client_updates",
                    }
                )
                continue

            consistency_values = self._leave_one_out_consistency(
                active_states, previous_expert_state
            )
            reliability_scores: List[float] = []
            expert_ess: List[float] = []
            expert_predictive_losses: List[float] = []
            expert_surrogate_losses: List[float] = []

            for local_idx, (client_id, usage, consistency) in enumerate(
                zip(active_client_ids, usages, consistency_values)
            ):
                u_observation = min(
                    max(float(usage), self.config.eps), 1.0 - self.config.eps
                )
                c_observation = min(
                    max(float(consistency), self.config.eps),
                    1.0 - self.config.eps,
                )
                state_key = f"{expert_scope}::{client_id}"
                particles, predictive_weights = self._predict_particles(state_key)
                score = self._score_pair(
                    u_observation,
                    c_observation,
                    particles,
                    predictive_weights,
                    gamma_before,
                    int(num_experts),
                )
                self.latent_memory[state_key] = {
                    "particles": particles.detach().clone(),
                    "weights": score["posterior_weights"].detach().clone(),
                }

                reliability_scores.append(score["reliability"])
                expert_ess.append(score["ess"])
                expert_predictive_losses.append(score["predictive_nll"])
                expert_surrogate_losses.append(score["surrogate_nll"])
                gradient_sum[0] += score["grad_u"]
                gradient_sum[1] += score["grad_c"]
                predictive_loss_sum += score["predictive_nll"]
                surrogate_loss_sum += score["surrogate_nll"]
                pair_count += 1
                ess_values.append(score["ess"])
                reliability_values.append(score["reliability"])

                pair_records.append(
                    {
                        "round": self.round_idx + 1,
                        "expert_scope": expert_scope,
                        "expert_id": expert_id,
                        "client_id": client_id,
                        "u": u_observation,
                        "c": c_observation,
                        "reliability": score["reliability"],
                        "ess": score["ess"],
                        "predictive_nll": score["predictive_nll"],
                        "surrogate_nll": score["surrogate_nll"],
                    }
                )

            reliability_tensor = torch.as_tensor(
                reliability_scores, dtype=torch.float64
            ).clamp_min(self.config.eps)
            alpha = reliability_tensor / reliability_tensor.sum()
            aggregated_expert_state = weighted_average_state_dicts(
                active_states, alpha, fallback_state=previous_expert_state
            )
            merged_expert_groups[expert_scope] = {
                "expert_id": expert_id,
                "expert_scope": expert_scope,
                "state_dict": aggregated_expert_state,
            }
            per_expert_info.append(
                {
                    "expert_scope": expert_scope,
                    "expert_id": expert_id,
                    "num_active_clients": len(active_states),
                    "client_ids": active_client_ids,
                    "u_values": [float(value) for value in usages],
                    "c_values": [float(value) for value in consistency_values],
                    "reliability_values": reliability_scores,
                    "weights": alpha.tolist(),
                    "mean_ess": sum(expert_ess) / max(len(expert_ess), 1),
                    "predictive_nll_mean": sum(expert_predictive_losses)
                    / max(len(expert_predictive_losses), 1),
                    "surrogate_nll_mean": sum(expert_surrogate_losses)
                    / max(len(expert_surrogate_losses), 1),
                    "fallback_used": False,
                    "fallback_reason": "",
                }
            )

        if pair_count > 0:
            mean_gradient = gradient_sum / float(pair_count)
            self.gamma = self._update_gamma(mean_gradient)
        else:
            mean_gradient = torch.zeros(2, dtype=torch.float64)

        new_global_state = merge_expert_states_into_global_state(
            base_state, merged_expert_groups
        )
        for key, value in global_state.items():
            if key not in new_global_state:
                new_global_state[key] = value
        if device is not None:
            new_global_state = type(new_global_state)(
                (
                    key,
                    value.to(device) if torch.is_tensor(value) else value,
                )
                for key, value in new_global_state.items()
            )

        self.round_idx += 1
        round_info = {
            "round_idx": self.round_idx,
            "num_pairs": pair_count,
            "gamma_u_before": float(gamma_before[0].item()),
            "gamma_c_before": float(gamma_before[1].item()),
            "gamma_u_after": float(self.gamma[0].item()),
            "gamma_c_after": float(self.gamma[1].item()),
            "predictive_nll": predictive_loss_sum / max(pair_count, 1),
            "surrogate_nll": surrogate_loss_sum / max(pair_count, 1),
            "gradient_u": float(mean_gradient[0].item()),
            "gradient_c": float(mean_gradient[1].item()),
            "mean_ess": sum(ess_values) / max(len(ess_values), 1),
            "mean_reliability": sum(reliability_values)
            / max(len(reliability_values), 1),
            "per_expert_info": per_expert_info,
        }
        self._append_metrics(round_info, pair_records)
        if self.logger is not None:
            self.logger.info(
                "--beta_precision_fedtra "
                f"--round:{self.round_idx} "
                f"--pairs:{pair_count} "
                f"--gamma_u:{round_info['gamma_u_after']:.6f} "
                f"--gamma_c:{round_info['gamma_c_after']:.6f} "
                f"--predictive_nll:{round_info['predictive_nll']:.6f} "
                f"--surrogate_nll:{round_info['surrogate_nll']:.6f}"
            )
        return new_global_state, round_info
