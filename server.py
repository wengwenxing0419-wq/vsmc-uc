import argparse
import collections
import csv
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from args import apply_plain_fedavg_baseline_overrides, parse
from client import Client
from dataset_catalog import dataset_metadata_dict, get_dataset_spec
from full_vsmc_expert_aggregator import FullVSMCAggregatorConfig, FullVSMCExpertAggregator
from Models.SwitchTransformer import SwitchTransformerClassifier
from temporal_latent_expert_aggregator import (
    TemporalLatentAggregatorConfig,
    TemporalLatentExpertAggregator,
)
from temporal_vsmc_expert_aggregator import (
    TemporalVSMCAggregatorConfig,
    TemporalVSMCExpertAggregator,
)
from utils import (
    append_vsmc_group_detail_log,
    init_ha_vsmc_etf_metrics_csv,
    init_result_csv,
    init_server_adapt_metrics_csv,
    init_vsmc_round_metrics_csv,
    record_ha_vsmc_etf_expert_summary,
    record_ha_vsmc_etf_particle_summary,
    record_ha_vsmc_etf_round_summary,
    record_server_adapt_metric,
    record_vsmc_round_metric,
    record_vsmc_round_summary,
    write_vsmc_run_summary,
)
from aggregation.history_aware_vsmc_expert_aggregator import (
    HAVSMCETFConfig,
    HAVSMCETFExpertAggregator,
)
from aggregation.beta_precision_fedtra_aggregator import (
    BetaPrecisionFedTRAConfig,
    BetaPrecisionFedTRAggregator,
)
from aggregation.vsmc_expert_aggregator import (
    VSMCExpertAggregationConfig,
    VSMCExpertAggregator,
    _merge_with_reference_state,
    compute_server_validation_g_for_expert,
    compute_server_validation_strong_signals_for_expert,
    group_expert_groups_by_block,
    merge_expert_states_into_global_state,
    split_state_dict_by_expert,
)
from aggregation.joint_vsmc_aggregator import (
    JointVSMCAggregationConfig,
    JointVSMCExpertAggregator,
    split_model_state_by_expert_and_router,
)
from fedmoe_da import FedMoEDAExpertAggregator, resolve_fedmoe_da_config
from fedjets import FedJETSRuntime
from pfedmoe import PFedMoERuntime
from hfedmoe import HFedMoERuntime


@dataclass
class AdaptMetrics:
    loss: float
    acc: float
    margin: float
    router_load_entropy: Optional[float]


@dataclass
class SelectiveAdaptState:
    best_proxy: float = float("-inf")
    best_accept_loss: float = float("inf")
    cooldown_remaining: int = 0
    accept_threshold_multiplier: float = 1.0
    near_best: bool = False


@dataclass
class TeacherRouterMetrics:
    teacher_router_loss: Optional[float]
    teacher_label_entropy_mean: Optional[float]
    teacher_conf_mean: Optional[float]
    teacher_conf_std: Optional[float]
    teacher_valid_ratio: Optional[float]
    router_teacher_acc: Optional[float]
    teacher_proxy_mode: bool
    counterfactual_success: bool
    counterfactual_fallback_reason: str


def strict_adapt_accept_rule(
    before: AdaptMetrics,
    after: AdaptMetrics,
    progress: float,
    config,
    accept_threshold_multiplier: float = 1.0,
):
    late_phase = bool(progress >= float(config.server_adapt_late_start_frac))
    base_min_delta = (
        float(config.server_adapt_late_min_delta)
        if late_phase
        else float(config.server_adapt_min_delta)
    )
    base_margin_delta = (
        float(config.server_adapt_late_margin_delta)
        if late_phase
        else float(config.server_adapt_margin_delta)
    )
    effective_min_delta = base_min_delta * float(accept_threshold_multiplier)
    effective_margin_delta = base_margin_delta * float(accept_threshold_multiplier)
    loss_delta = float(before.loss - after.loss)
    margin_delta = float(after.margin - before.margin)
    acc_delta = float(after.acc - before.acc)
    loss_evidence = bool(loss_delta > effective_min_delta)
    margin_evidence = bool(margin_delta > effective_margin_delta)
    acc_evidence = bool(acc_delta > 0.0)
    strong_accept = (
        after.loss < before.loss - effective_min_delta
        and after.margin > before.margin + effective_margin_delta
    )
    acc_accept = (
        after.acc > before.acc
        and after.loss < before.loss - float(config.server_adapt_weak_loss_delta)
    )
    late_strict_accept = False
    late_insufficient_improvement = False
    improvement_score = loss_delta + float(config.server_adapt_margin_weight) * margin_delta
    if late_phase:
        late_strict_accept = (
            after.loss < before.loss - effective_min_delta
            and after.margin > before.margin + effective_margin_delta
            and after.acc >= before.acc
        )
        accept = late_strict_accept
        reason = "late_strict_accept" if accept else "late_reject"
        if improvement_score < float(config.server_adapt_late_min_delta):
            accept = False
            reason = "late_insufficient_improvement"
            late_insufficient_improvement = True
    else:
        accept = strong_accept or acc_accept
        if strong_accept:
            reason = "strong_loss_margin_accept"
        elif acc_accept:
            reason = "acc_loss_accept"
        else:
            reason = "strict_reject"
    debug = {
        "adapt_phase": "late" if late_phase else "early",
        "loss_delta": loss_delta,
        "margin_delta": margin_delta,
        "acc_delta": acc_delta,
        "router_entropy_delta": 0.0,
        "loss_improved": bool(after.loss < before.loss),
        "acc_improved": bool(after.acc > before.acc),
        "margin_improved": bool(after.margin > before.margin),
        "margin_not_worse": bool(after.margin >= before.margin),
        "loss_not_worse": bool(after.loss <= before.loss),
        "loss_evidence": loss_evidence,
        "margin_evidence": margin_evidence,
        "acc_evidence": acc_evidence,
        "router_evidence": True,
        "evidence_count": int(loss_evidence) + int(margin_evidence) + int(acc_evidence) + 1,
        "evidence_required": 2 if not late_phase else 3,
        "strong_accept": bool(strong_accept),
        "acc_accept": bool(acc_accept),
        "late_strict_accept": bool(late_strict_accept),
        "late_phase": bool(late_phase),
        "late_insufficient_improvement": bool(late_insufficient_improvement),
        "effective_min_delta": effective_min_delta,
        "effective_margin_delta": effective_margin_delta,
        "improvement_score": improvement_score,
        "near_best": False,
        "near_best_reject": False,
        "cooldown_active": False,
        "cooldown_remaining": 0,
        "cooldown_skip": False,
        "rescue_adapt": False,
    }
    return bool(accept), reason, debug


def selective_adapt_accept_rule(
    before: AdaptMetrics,
    after: AdaptMetrics,
    progress: float,
    config,
    state: SelectiveAdaptState,
):
    multiplier = float(getattr(state, "accept_threshold_multiplier", 1.0))
    if progress >= float(getattr(config, "server_adapt_late_start_frac", 0.8)):
        phase = "late"
        base_min_delta = float(getattr(config, "server_adapt_late_min_delta", 1e-3))
        base_margin_delta = float(getattr(config, "server_adapt_late_margin_delta", 1e-3))
        evidence_required = int(getattr(config, "server_adapt_evidence_required_late", 3))
    elif progress >= float(getattr(config, "server_adapt_mid_start_frac", 0.7)):
        phase = "mid"
        base_min_delta = float(getattr(config, "server_adapt_mid_min_delta", 7e-4))
        base_margin_delta = float(getattr(config, "server_adapt_mid_margin_delta", 8e-4))
        evidence_required = int(getattr(config, "server_adapt_evidence_required_mid", 2))
    else:
        phase = "early"
        base_min_delta = float(getattr(config, "server_adapt_min_delta", 3e-4))
        base_margin_delta = float(getattr(config, "server_adapt_margin_delta", 5e-4))
        evidence_required = 2

    effective_min_delta = base_min_delta * multiplier
    effective_margin_delta = base_margin_delta * multiplier
    loss_delta = float(before.loss - after.loss)
    margin_delta = float(after.margin - before.margin)
    acc_delta = float(after.acc - before.acc)
    if before.router_load_entropy is not None and after.router_load_entropy is not None:
        router_entropy_delta = float(after.router_load_entropy - before.router_load_entropy)
    else:
        router_entropy_delta = 0.0

    loss_evidence = bool(loss_delta > effective_min_delta)
    margin_evidence = bool(margin_delta > effective_margin_delta)
    acc_evidence = bool(acc_delta > 0.0)
    router_evidence = bool(
        router_entropy_delta >= -float(getattr(config, "server_adapt_router_entropy_tolerance", 0.03))
    )
    evidence_count = int(loss_evidence) + int(margin_evidence) + int(acc_evidence) + int(router_evidence)

    evidence_gate = bool(getattr(config, "server_adapt_evidence_gate", True))
    if evidence_gate:
        accept = bool(evidence_count >= evidence_required)
    else:
        accept = bool(loss_evidence and margin_evidence)
    reason = f"{phase}_evidence_accept" if accept else f"{phase}_evidence_reject"

    if phase == "late":
        accept = bool(accept and acc_delta >= 0.0)
        if not accept and reason == "late_evidence_accept":
            reason = "late_reject"

    best_proxy = float(getattr(state, "best_proxy", float("-inf")))
    near_best_tolerance = float(getattr(config, "server_adapt_near_best_tolerance", 0.002))
    near_best = bool(best_proxy != float("-inf") and before.acc >= best_proxy - near_best_tolerance)
    state.near_best = near_best
    near_best_reject = False
    if bool(getattr(config, "server_adapt_near_best_protection", True)) and near_best:
        if bool(getattr(config, "server_adapt_near_best_require_late_rule", True)):
            accept = bool(
                loss_delta > float(getattr(config, "server_adapt_late_min_delta", 1e-3))
                and margin_delta > float(getattr(config, "server_adapt_late_margin_delta", 1e-3))
                and acc_delta >= 0.0
            )
            if accept:
                reason = "near_best_strict_accept"
            else:
                reason = "near_best_reject"
                near_best_reject = True

    cooldown_active = bool(
        getattr(config, "server_adapt_late_cooldown", True)
        and progress >= float(getattr(config, "server_adapt_cooldown_start_frac", 0.75))
        and int(getattr(state, "cooldown_remaining", 0)) > 0
    )
    rescue_adapt = False
    cooldown_skip = False
    if cooldown_active:
        if bool(getattr(config, "server_adapt_rescue_if_loss_worse", True)) and before.loss > (
            float(getattr(state, "best_accept_loss", float("inf")))
            + float(getattr(config, "server_adapt_rescue_loss_threshold", 0.01))
        ):
            rescue_adapt = True
        else:
            accept = False
            reason = "cooldown_skip"
            cooldown_skip = True

    if bool(accept) and progress >= float(getattr(config, "server_adapt_cooldown_start_frac", 0.75)):
        state.cooldown_remaining = max(int(getattr(config, "server_adapt_cooldown_after_accept", 1)), 0)
    else:
        state.cooldown_remaining = max(0, int(getattr(state, "cooldown_remaining", 0)) - 1)

    debug = {
        "adapt_phase": phase,
        "loss_delta": loss_delta,
        "margin_delta": margin_delta,
        "acc_delta": acc_delta,
        "router_entropy_delta": router_entropy_delta,
        "loss_improved": bool(after.loss < before.loss),
        "acc_improved": bool(after.acc > before.acc),
        "margin_improved": bool(after.margin > before.margin),
        "margin_not_worse": bool(after.margin >= before.margin),
        "loss_not_worse": bool(after.loss <= before.loss),
        "loss_evidence": loss_evidence,
        "margin_evidence": margin_evidence,
        "acc_evidence": acc_evidence,
        "router_evidence": router_evidence,
        "evidence_count": evidence_count,
        "evidence_required": evidence_required,
        "strong_accept": bool(loss_evidence and margin_evidence),
        "acc_accept": acc_evidence,
        "late_strict_accept": bool(phase == "late" and accept and acc_delta >= 0.0),
        "late_phase": bool(phase == "late"),
        "late_insufficient_improvement": False,
        "effective_min_delta": effective_min_delta,
        "effective_margin_delta": effective_margin_delta,
        "improvement_score": loss_delta + float(getattr(config, "server_adapt_margin_weight", 0.2)) * margin_delta,
        "near_best": near_best,
        "near_best_reject": near_best_reject,
        "cooldown_active": cooldown_active,
        "cooldown_remaining": int(getattr(state, "cooldown_remaining", 0)),
        "cooldown_skip": cooldown_skip,
        "rescue_adapt": rescue_adapt,
    }
    return bool(accept), reason, debug


def teacher_router_accept_rule(
    before: AdaptMetrics,
    after: AdaptMetrics,
    before_teacher: Optional[TeacherRouterMetrics],
    after_teacher: Optional[TeacherRouterMetrics],
    progress: float,
    config,
):
    del progress
    loss_delta = float(before.loss - after.loss)
    margin_delta = float(after.margin - before.margin)
    acc_delta = float(after.acc - before.acc)
    weak_loss_delta = float(getattr(config, "server_adapt_weak_loss_delta", 1e-4))
    teacher_tolerance = weak_loss_delta
    teacher_loss_before = None if before_teacher is None else before_teacher.teacher_router_loss
    teacher_loss_after = None if after_teacher is None else after_teacher.teacher_router_loss
    teacher_ok = (
        teacher_loss_before is None
        or teacher_loss_after is None
        or teacher_loss_after <= teacher_loss_before + teacher_tolerance
    )
    loss_accept = bool(loss_delta > weak_loss_delta and margin_delta >= 0.0 and teacher_ok)
    acc_accept = bool(acc_delta > 0.0 and after.loss <= before.loss + weak_loss_delta)
    accept = bool(loss_accept or acc_accept)
    if loss_accept:
        reason = "teacher_loss_margin_accept"
    elif acc_accept:
        reason = "acc_loss_accept"
    else:
        reason = "teacher_router_reject"
    debug = {
        "adapt_phase": "teacher_router",
        "loss_delta": loss_delta,
        "margin_delta": margin_delta,
        "acc_delta": acc_delta,
        "router_entropy_delta": 0.0,
        "loss_improved": bool(after.loss < before.loss),
        "acc_improved": bool(after.acc > before.acc),
        "margin_improved": bool(after.margin > before.margin),
        "margin_not_worse": bool(after.margin >= before.margin),
        "loss_not_worse": bool(after.loss <= before.loss),
        "loss_evidence": bool(loss_delta > weak_loss_delta),
        "margin_evidence": bool(margin_delta >= 0.0),
        "acc_evidence": bool(acc_delta > 0.0),
        "router_evidence": bool(teacher_ok),
        "evidence_count": (
            int(loss_delta > weak_loss_delta)
            + int(margin_delta >= 0.0)
            + int(acc_delta > 0.0)
            + int(teacher_ok)
        ),
        "evidence_required": 3,
        "strong_accept": bool(loss_accept),
        "acc_accept": bool(acc_accept),
        "late_strict_accept": False,
        "late_phase": False,
        "late_insufficient_improvement": False,
        "effective_min_delta": weak_loss_delta,
        "effective_margin_delta": 0.0,
        "improvement_score": loss_delta + float(getattr(config, "server_adapt_margin_weight", 0.2)) * margin_delta,
        "near_best": False,
        "near_best_reject": False,
        "cooldown_active": False,
        "cooldown_remaining": 0,
        "cooldown_skip": False,
        "rescue_adapt": False,
        "teacher_router_ok": bool(teacher_ok),
        "teacher_router_loss_before": teacher_loss_before,
        "teacher_router_loss_after": teacher_loss_after,
    }
    return bool(accept), reason, debug


class Server:
    """Federated Averaging server for a Switch Transformer global model."""

    def __init__(self, parse: argparse.ArgumentParser, logger):
        self.args = apply_plain_fedavg_baseline_overrides(parse.parse_args())
        if (
            not bool(getattr(self.args, "use_plain_fedavg_baseline", False))
            and getattr(self.args, "method", "")
            and self.args.method == "learned_vsmc_moe_state_policy_k8"
        ):
            self.args.aggregation_method = "learned_vsmc_moe_state_policy_k8"
            self.args.learned_vsmc_moe_state_policy = True
        if self.args.aggregation_method == "learned_vsmc_moe_v11_server_adapt":
            self.args.server_adapt_enabled = True
        if bool(getattr(self.args, "fedjets_enabled", False)) or str(getattr(self.args, "method", "")).lower() == "fedjets":
            self.args.aggregation_method = "fedjets"
        if bool(getattr(self.args, "pfedmoe_enabled", False)) or str(getattr(self.args, "method", "")).lower() == "pfedmoe":
            self.args.aggregation_method = "pfedmoe"
        if bool(getattr(self.args, "hfedmoe_enabled", False)) or str(getattr(self.args, "method", "")).lower() == "hfedmoe":
            self.args.aggregation_method = "hfedmoe"
        self.base_aggregation_method = self._resolve_base_aggregation_method(self.args.aggregation_method)
        self.dataset_spec = get_dataset_spec(self.args.data_name)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        self.logger = logger
        self.device = torch.device(self.args.device)
        self.num_clients = self.args.num_clients
        self.server_epochs = self.args.server_epochs
        self.clients_id_list = list(range(1, self.num_clients + 1))
        self.model_path = os.path.join(self.args.model_save_path, "server.pth")

        os.makedirs(self.args.model_save_path, exist_ok=True)
        init_result_csv()

        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.server_test_loader = self.build_server_test_loader()
        self.server_val_loader = self.build_server_validation_loader()
        self.ha_vsmc_client_class_stats = self._load_ha_vsmc_client_class_stats()
        self.ha_vsmc_client_coverage_scores = [
            float(item.get("coverage_score", 0.0))
            for item in self.ha_vsmc_client_class_stats
        ]
        self.ha_vsmc_proxy_loader = self.build_ha_vsmc_proxy_loader()
        self._log_dataset_and_backbone_config()
        self.vsmc_aggregator = None
        self.beta_precision_fedtra_aggregator = None
        self.ha_vsmc_etf_aggregator = None
        self.temporal_vsmc_aggregator = None
        self.temporal_vsmc_memory = {}
        self.temporal_latent_aggregator = None
        self.temporal_latent_memory = {}
        self.joint_vsmc_aggregator = None
        self.learned_vsmc_aggregator = None
        self.moe_state_vsmc_aggregator = None
        self.fedmoe_da_aggregator = None
        self.fedjets_runtime = None
        self.pfedmoe_runtime = None
        self.hfedmoe_runtime = None
        self.latest_vsmc_round_infos = []
        self.latest_ha_vsmc_etf_round_data = None
        self.best_server_test_acc = float("-inf")
        self.best_server_test_round = 0
        self.server_adapt_accept_history: List[int] = []
        self.server_adapt_accept_threshold_multiplier = 1.0
        self.server_adapt_ema_delta: Dict[str, torch.Tensor] = {}
        self.server_adapt_best_proxy = float("-inf")
        self.server_adapt_best_accept_loss = float("inf")
        self.server_adapt_cooldown_remaining = 0
        if bool(getattr(self.args, "use_plain_fedavg_baseline", False)):
            self.logger.info(
                "Plain FedAvg baseline enabled: sample-count aggregation only; "
                "aux/router/prox/diversity/server-adapt extras are disabled."
            )
        if self._server_adapt_enabled():
            init_server_adapt_metrics_csv()
        if bool(getattr(self.args, "server_adapt_update_experts", False)):
            self.logger.warning("server_adapt_update_experts is ignored; server adaptation never updates expert parameters.")
        if self.base_aggregation_method == "fedavg_expert_only":
            self.logger.info(
                "FedAvg expert-only baseline selected: only Switch-FFN expert parameters are averaged; "
                "shared, router, backbone, and head parameters stay on the previous server state."
            )
        elif self.base_aggregation_method == "fedavg_shared_only":
            self.logger.info(
                "FedAvg shared-only baseline selected: backbone, router, shared, and head parameters are averaged; "
                "Switch-FFN expert parameters stay on the previous server state."
            )
        elif self.base_aggregation_method == "fedmoe_da":
            self.fedmoe_da_aggregator = FedMoEDAExpertAggregator(
                resolve_fedmoe_da_config(self.args),
                logger=self.logger,
            )
            self.logger.info(
                "FedMoE-DA comparison baseline selected: common experiment settings are inherited from args.py; "
                "shared/router/non-expert parameters remain on the shared FedAvg path while experts use "
                "router-similarity top-P aggregation."
            )
        elif self.base_aggregation_method == "fedjets":
            self.fedjets_runtime = FedJETSRuntime(
                args=self.args,
                logger=self.logger,
                model_factory=self.build_model,
                device=self.device,
            )
            self.logger.info(
                "FedJETs comparison baseline selected: common experiment settings are inherited from args.py; "
                "a frozen common expert produces embeddings, a two-layer gate selects top-K server experts, "
                "anchor clients specialize one expert each, and gate plus updated experts aggregate with FedAvg."
            )
        elif self.base_aggregation_method == "pfedmoe":
            self.pfedmoe_runtime = PFedMoERuntime(
                args=self.args,
                logger=self.logger,
                device=self.device,
            )
            self.logger.info(
                "pFedMoE comparison baseline selected: common experiment settings are inherited from args.py; "
                "clients keep personalized local extractors, gates, and heads while the server only aggregates the shared small extractor."
            )
        elif self.base_aggregation_method == "hfedmoe":
            self.hfedmoe_runtime = HFedMoERuntime(
                args=self.args,
                logger=self.logger,
                device=self.device,
            )
            self.logger.info(
                "HFedMoE comparison baseline selected: common experiment settings are inherited from args.py; "
                "a frozen shared backbone feeds resource-aware expert subset selection, local clients fine-tune only their selected experts plus gate, and the server sparsely aggregates active experts with importance-weighted contributions."
            )
        elif self.base_aggregation_method == "beta_precision_fedtra":
            self.beta_precision_fedtra_aggregator = BetaPrecisionFedTRAggregator(
                self.build_beta_precision_fedtra_config(), logger=self.logger
            )
            self.logger.info(
                "Dynamic Beta-precision FedTRA selected: training routing support and "
                "leave-one-out update consistency form separate Beta observations; "
                "Jensen-surrogate predictive loss drives OMD precision allocation."
            )
        elif self.base_aggregation_method == "vsmc_expert":
            self.vsmc_aggregator = self.build_vsmc_aggregator()
            init_vsmc_round_metrics_csv()
            if self.args.vsmc_variant == "full_vsmc":
                self.logger.info(
                    "Full-VSMC expert aggregation skeleton selected: expert parameters use a latent state-space "
                    "aggregation scaffold with transition/proposal-style memory; router and non-expert parameters "
                    "remain on sample-count FedAvg."
                )
            else:
                self.logger.info(
                    "VSMC expert aggregation selected: expert parameters use VSMC-style particle weights; "
                    "router and all non-expert parameters remain on sample-count FedAvg."
                )
        elif self.base_aggregation_method == "ha_vsmc_etf":
            self.ha_vsmc_etf_aggregator = HAVSMCETFExpertAggregator(
                self.build_ha_vsmc_etf_config(),
                logger=self.logger,
            )
            init_ha_vsmc_etf_metrics_csv()
            self.logger.info(
                "History-aware VSMC expert trust filtering selected: expert parameters use HA-VSMC-ETF, "
                "while router and non-expert parameters remain on their original FedAvg path."
            )
        elif self.base_aggregation_method == "joint_vsmc_expert":
            self.joint_vsmc_aggregator = JointVSMCExpertAggregator(
                self.build_joint_vsmc_config(),
                num_experts=self.args.switch_num_experts,
                logger=self.logger,
            )
            init_vsmc_round_metrics_csv()
            self.logger.info(
                "Joint VSMC expert aggregation selected: expert and router parameters use a router-aware, "
                "validation-guided, client-specific proposal aggregation path; shared parameters remain on FedAvg."
            )
        elif self.base_aggregation_method == "temporal_vsmc_expert":
            self.temporal_vsmc_aggregator = TemporalVSMCExpertAggregator(
                self.build_temporal_vsmc_config(),
                logger=self.logger,
            )
            init_vsmc_round_metrics_csv()
            self.logger.info(
                "Temporal VSMC expert aggregation selected: expert parameters use cross-round temporal "
                "consistency-aware weights; router and non-expert parameters remain on sample-count FedAvg."
            )
        elif self.base_aggregation_method == "temporal_latent_vsmc_expert":
            self.temporal_latent_aggregator = TemporalLatentExpertAggregator(
                self.build_temporal_latent_config(),
                logger=self.logger,
            )
            init_vsmc_round_metrics_csv()
            self.logger.info(
                "Temporal latent VSMC expert aggregation selected: expert parameters use a low-dimensional "
                "latent Gaussian update model with cross-round memory; router and non-expert parameters "
                "remain on sample-count FedAvg."
            )
        elif self.base_aggregation_method == "learned_vsmc":
            _, LearnedVSMCExpertAggregator = self._load_learned_vsmc_components()
            self.learned_vsmc_aggregator = LearnedVSMCExpertAggregator(
                self.build_learned_vsmc_config(),
                num_experts=self.args.switch_num_experts,
                logger=self.logger,
            )
            init_vsmc_round_metrics_csv()
            if self.args.aggregation_method == "learned_vsmc_moe_v11_server_adapt":
                self.logger.info(
                    "Learned VSMC expert aggregation with post-aggregation server-side adaptation selected: "
                    "experts stay on learned VSMC, while head/norm/router-temperature calibration runs server-side with rollback."
                )
            else:
                self.logger.info(
                    "Learned proposal VSMC aggregation selected: expert parameters use learned low-dimensional proposal "
                    "particles with server-validation-guided VSMC weights; router and non-expert parameters remain on FedAvg."
                )
        elif self.base_aggregation_method == "learned_vsmc_moe_state_policy_k8" or bool(getattr(self.args, "learned_vsmc_moe_state_policy", False)):
            _, MoEStateVSMCAggregator = self._load_moe_state_policy_components()
            self.moe_state_vsmc_aggregator = MoEStateVSMCAggregator(
                self.build_moe_state_policy_config(),
                num_experts=self.args.switch_num_experts,
                logger=self.logger,
            )
            self.logger.info(
                "MoE-state sequential VSMC policy aggregation selected: cross-round policy particles govern "
                "MoE block aggregation while non-expert parameters remain on FedAvg."
            )
        if self._server_adapt_enabled():
            self.logger.info(
                f"Server adaptation enabled: version={self.args.server_adapt_version} "
                f"start_frac={float(self.args.server_adapt_start_frac):.2f} "
                f"strict_accept={bool(self.args.server_adapt_strict_accept)} "
                f"split_accept={bool(self.args.server_adapt_split_accept)}"
            )
        if self.base_aggregation_method == "fedjets" and self.fedjets_runtime is not None:
            self.fedjets_runtime.save_checkpoint()
        elif self.base_aggregation_method == "pfedmoe" and self.pfedmoe_runtime is not None:
            self.pfedmoe_runtime.save_checkpoint()
        elif self.base_aggregation_method == "hfedmoe" and self.hfedmoe_runtime is not None:
            self.hfedmoe_runtime.save_checkpoint()
        else:
            self.save_server_model()

    def get_num_classes(self):
        return int(self.dataset_spec.num_classes)

    def get_image_size(self):
        configured_size = int(getattr(self.args, "image_size", 32))
        if configured_size != 32:
            return configured_size
        return int(self.dataset_spec.image_size)

    def _dataset_metadata_path(self):
        return os.path.join(self.args.data_save_path, "dataset_metadata.json")

    def _load_dataset_metadata(self):
        metadata_path = self._dataset_metadata_path()
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return dataset_metadata_dict(
            self.args.data_name,
            num_clients=self.args.num_clients,
            alpha=self.args.alpha,
            seed=self.args.seed,
            artifact_format="unknown",
        )

    def _resolve_resnet_stem_mode(self):
        requested_mode = str(getattr(self.args, "resnet_stem_mode", "auto")).lower()
        if requested_mode == "auto":
            return "imagenet" if self.get_image_size() > 32 else "cifar"
        return requested_mode

    def _backbone_note(self):
        backbone_name = str(self.args.backbone_name)
        if backbone_name == "simplecnn":
            return (
                "SimpleCNN/CNN-4 tokenizer feeds the existing Switch blocks; "
                "MoE remains in the downstream transformer FFN blocks after the CNN stem."
            )
        if backbone_name in {"patch_embed", "vit_tiny", "vit_small", "patch_embed_switch", "patch_tokenizer_switch"}:
            return (
                "Legacy patch tokenizer Switch path: a single patchify Conv2d feeds a shallow Switch stack. "
                "This is a lightweight patch baseline, not a standard ViT-Tiny/ViT-Small encoder."
            )
        if backbone_name == "vit_tiny_real":
            return (
                "Real ViT-Tiny-style path: patch embedding plus a deeper Transformer encoder with MoE inserted in the FFN blocks. "
                "This is the backbone intended for fair patch-based comparisons."
            )
        if backbone_name.startswith("resnet"):
            return (
                f"ResNet tokenizer feeds the existing Switch blocks; resolved stem mode={self._resolve_resnet_stem_mode()}."
            )
        return "Backbone uses the standard Switch Transformer tokenization path."

    def _log_dataset_and_backbone_config(self):
        metadata = self._load_dataset_metadata()
        self.logger.info(
            f"Dataset config: name={metadata.get('display_name', self.args.data_name)} "
            f"num_classes={metadata.get('num_classes', self.get_num_classes())} "
            f"image_size={self.get_image_size()} in_channels={metadata.get('in_channels', 3)}"
        )
        self.logger.info(
            f"Dataset normalization: mean={metadata.get('normalization_mean')} "
            f"std={metadata.get('normalization_std')} train_split={metadata.get('train_split')} "
            f"val_split={metadata.get('val_split')} test_split={metadata.get('test_split')}"
        )
        self.logger.info(
            f"Client augmentation: use_client_aug={bool(getattr(self.args, 'use_client_aug', False))} "
            f"crop_pad={int(getattr(self.args, 'aug_random_crop_pad', 0))} "
            f"horizontal_flip={float(getattr(self.args, 'aug_horizontal_flip', 0.0)):.2f} "
            f"random_erasing={float(getattr(self.args, 'aug_random_erasing', 0.0)):.2f} "
            f"mixup_alpha={float(getattr(self.args, 'mixup_alpha', 0.0)):.2f}"
        )
        backbone_summary = getattr(self.model, "get_backbone_summary", lambda: {})()
        parameter_count = sum(param.numel() for param in self.model.parameters())
        self.logger.info(
            f"Backbone config: backbone={self.args.backbone_name} family={backbone_summary.get('backbone_family')} "
            f"image_size={self.get_image_size()} patch_size={backbone_summary.get('patch_size', self.args.patch_size)} "
            f"embed_dim={backbone_summary.get('embed_dim', self.args.hidden_dim)} depth={backbone_summary.get('depth', self.args.switch_layers)} "
            f"num_heads={backbone_summary.get('num_heads', self.args.switch_heads)} mlp_ratio={backbone_summary.get('mlp_ratio')} "
            f"switch_ffn_dim={backbone_summary.get('ffn_dim', self.args.switch_ffn_dim)} num_experts={backbone_summary.get('num_experts', self.args.switch_num_experts)} "
            f"dropout={backbone_summary.get('dropout', self.args.dropout)} attention_dropout={backbone_summary.get('attention_dropout')} "
            f"use_cls_token={backbone_summary.get('use_cls_token')} use_learnable_pos_embed={backbone_summary.get('use_learnable_pos_embed')} "
            f"pretrained={backbone_summary.get('pretrained', False)} resnet_stem_mode={self._resolve_resnet_stem_mode() if str(self.args.backbone_name).startswith('resnet') else 'n/a'} "
            f"num_classes={self.get_num_classes()} parameter_count={parameter_count}"
        )
        self.logger.info(
            f"Optimizer config: client_optimizer={self.args.client_optimizer} learning_rate={self.args.learning_rate} "
            f"client_weight_decay={self.args.client_weight_decay} lr_schedule={self.args.client_lr_schedule} "
            f"lr_warmup_rounds={getattr(self.args, 'client_lr_warmup_rounds', 0)} batch_size={self.args.batch_size}"
        )
        self.logger.info(f"Backbone note: {self._backbone_note()}")
        for note in metadata.get("notes", []):
            self.logger.info(f"Dataset note: {note}")

    def _load_learned_vsmc_components(self):
        try:
            from aggregation.learned_vsmc_aggregator import (
                LearnedVSMCAggregationConfig,
                LearnedVSMCExpertAggregator,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The learned_vsmc aggregation path requires a newer PyTorch build "
                "that provides torch.func.functional_call or "
                "torch.nn.utils.stateless.functional_call."
            ) from exc
        return LearnedVSMCAggregationConfig, LearnedVSMCExpertAggregator

    def _load_moe_state_policy_components(self):
        try:
            from aggregation.vsmc_moe_state_policy import (
                MoEStatePolicyConfig,
                MoEStateVSMCAggregator,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The learned_vsmc_moe_state_policy_k8 aggregation path requires a newer "
                "PyTorch build that provides torch.func.functional_call or "
                "torch.nn.utils.stateless.functional_call."
            ) from exc
        return MoEStatePolicyConfig, MoEStateVSMCAggregator

    def build_model(self):
        model = SwitchTransformerClassifier(
            num_classes=self.get_num_classes(),
            image_size=self.get_image_size(),
            patch_size=self.args.patch_size,
            backbone_name=self.args.backbone_name,
            hidden_dim=self.args.hidden_dim,
            num_layers=self.args.switch_layers,
            num_heads=self.args.switch_heads,
            ffn_dim=self.args.switch_ffn_dim,
            num_experts=self.args.switch_num_experts,
            dropout=self.args.dropout,
            resnet_stem_mode=getattr(self.args, "resnet_stem_mode", "auto"),
            vit_real_embed_dim=getattr(self.args, "vit_real_embed_dim", 192),
            vit_real_depth=getattr(self.args, "vit_real_depth", 6),
            vit_real_num_heads=getattr(self.args, "vit_real_num_heads", 3),
            vit_real_mlp_ratio=getattr(self.args, "vit_real_mlp_ratio", 4.0),
            vit_real_dropout=getattr(self.args, "vit_real_dropout", self.args.dropout),
            vit_real_attention_dropout=getattr(self.args, "vit_real_attention_dropout", 0.0),
            vit_real_use_cls_token=getattr(self.args, "vit_real_use_cls_token", True),
            vit_real_use_learnable_pos_embed=getattr(self.args, "vit_real_use_learnable_pos_embed", True),
        )
        for module in model.modules():
            router = getattr(module, "router", None)
            if router is not None:
                router.server_temperature_min = float(getattr(self.args, "server_adapt_router_temp_min", 0.5))
                router.server_temperature_max = float(getattr(self.args, "server_adapt_router_temp_max", 2.0))
        return model

    def save_server_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_server_state(self):
        return torch.load(self.model_path, map_location=self.device)

    def build_server_test_loader(self):
        data_path = os.path.join(self.args.data_save_path, "server_test.pth")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Missing server test dataset at {data_path}. "
                "Please regenerate the dataset splits before training."
            )
        test_dataset = torch.load(data_path, weights_only=False)
        return DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

    def build_server_validation_loader(self):
        data_path = os.path.join(self.args.data_save_path, "server_val.pth")
        if not os.path.exists(data_path):
            return self.build_server_test_loader()
        val_dataset = torch.load(data_path, weights_only=False)
        return DataLoader(
            val_dataset,
            batch_size=self.args.vsmc_server_val_batch_size,
            shuffle=False,
        )

    def _load_ha_vsmc_client_class_stats(self):
        stats = []
        if not os.path.isdir(self.args.data_save_path):
            return stats
        for client_id in range(1, int(self.num_clients) + 1):
            data_path = os.path.join(self.args.data_save_path, f"{client_id}_train.pth")
            if not os.path.exists(data_path):
                stats.append({"client_id": client_id, "coverage_count": 0, "coverage_score": 0.0, "labels": []})
                continue
            dataset = torch.load(data_path, weights_only=False)
            labels = []
            for item in dataset:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    labels.append(int(item[1]))
            unique_labels = sorted(set(labels))
            coverage_count = len(unique_labels)
            stats.append(
                {
                    "client_id": client_id,
                    "coverage_count": coverage_count,
                    "coverage_score": coverage_count / float(max(self.get_num_classes(), 1)),
                    "labels": unique_labels,
                    "sample_count": len(labels),
                }
            )
        return stats

    def build_ha_vsmc_proxy_loader(self):
        if float(getattr(self.args, "ha_vsmc_proxy_gain_gamma", 0.0)) <= 0.0:
            return None
        samples_per_class = max(int(getattr(self.args, "ha_vsmc_proxy_samples_per_class", 2)), 1)
        by_class = collections.defaultdict(list)
        for client_id in range(1, int(self.num_clients) + 1):
            data_path = os.path.join(self.args.data_save_path, f"{client_id}_train.pth")
            if not os.path.exists(data_path):
                continue
            dataset = torch.load(data_path, weights_only=False)
            for item in dataset:
                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                x, y = item[0], int(item[1])
                if len(by_class[y]) < samples_per_class:
                    by_class[y].append((x, y))
        proxy_dataset = []
        for class_id in sorted(by_class.keys()):
            proxy_dataset.extend(by_class[class_id])
        if not proxy_dataset:
            return None
        return DataLoader(
            proxy_dataset,
            batch_size=min(max(int(getattr(self.args, "ha_vsmc_proxy_batch_size", 128)), 1), len(proxy_dataset)),
            shuffle=False,
        )

    def _get_ha_vsmc_probe_batch(self):
        preferred_loader = self.ha_vsmc_proxy_loader if self.ha_vsmc_proxy_loader is not None else self.server_val_loader
        if preferred_loader is None:
            return None
        max_batch_size = max(int(getattr(self.args, "ha_vsmc_probe_batch_size", 128)), 1)
        for batch in preferred_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                return None
            inputs, labels = batch[0], batch[1]
            if not torch.is_tensor(inputs) or not torch.is_tensor(labels):
                return None
            return inputs[:max_batch_size].to(self.device), labels[:max_batch_size].to(self.device)
        return None

    @staticmethod
    def _resolve_base_aggregation_method(method: str) -> str:
        if str(method) == "learned_vsmc_moe_v11_server_adapt":
            return "learned_vsmc"
        if str(method) == "history_aware_vsmc_expert_trust_filter":
            return "ha_vsmc_etf"
        return str(method)

    def _server_adapt_enabled(self) -> bool:
        return bool(getattr(self.args, "server_adapt_enabled", False))

    @staticmethod
    def _server_adapt_version(config) -> str:
        return str(getattr(config, "server_adapt_version", "v11c_strict_accept"))

    def _teacher_router_enabled(self, config, progress: Optional[float] = None) -> bool:
        enabled = self._server_adapt_version(config) == "v12_teacher_router" and bool(
            getattr(config, "server_adapt_teacher_router", True)
        )
        if progress is not None:
            enabled = enabled and progress >= float(
                getattr(config, "server_adapt_teacher_start_frac", getattr(config, "server_adapt_start_frac", 0.5))
            )
        return enabled

    def _server_adapt_router_temperature_enabled(self, config) -> bool:
        if self._teacher_router_enabled(config):
            return bool(getattr(config, "server_adapt_teacher_update_router_temperature", True))
        return bool(getattr(config, "server_adapt_update_router_temperature", False))

    def _server_adapt_router_bias_enabled(self, config) -> bool:
        if self._teacher_router_enabled(config):
            return bool(getattr(config, "server_adapt_teacher_update_router_bias", True))
        return bool(getattr(config, "server_adapt_update_router_bias", False))

    def _server_adapt_full_router_enabled(self, config) -> bool:
        if self._teacher_router_enabled(config):
            return bool(getattr(config, "server_adapt_teacher_update_full_router", False))
        return bool(getattr(config, "server_adapt_update_full_router", False))

    def _uses_vsmc_round_metrics(self) -> bool:
        return self.base_aggregation_method in {
            "vsmc_expert",
            "temporal_vsmc_expert",
            "temporal_latent_vsmc_expert",
            "learned_vsmc",
            "learned_vsmc_moe_state_policy_k8",
        }

    @staticmethod
    def _is_expert_param_name(name: str) -> bool:
        return ".switch_ffn.experts." in str(name)

    @staticmethod
    def _is_router_param_name(name: str) -> bool:
        return ".switch_ffn.router." in str(name)

    @staticmethod
    def _is_server_router_temperature_name(name: str) -> bool:
        return str(name).endswith(".server_temperature")

    @staticmethod
    def _is_server_router_bias_name(name: str) -> bool:
        return str(name).endswith(".server_bias")

    @staticmethod
    def _safe_float(value, default=float("nan")):
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _ha_vsmc_logs_dir(self):
        return os.path.join(self.args.save_result, "logs")

    def _compute_weight_diagnostics(self, weights):
        weights = [max(float(w), 0.0) for w in list(weights)]
        total = sum(weights)
        if total <= 0.0:
            return {
                "N_eff": 0.0,
                "entropy": 0.0,
                "max_client_weight": 0.0,
                "active_client_count": 0,
                "top1_client": -1,
                "top1_weight": 0.0,
                "top3_clients": "",
                "top3_weight_sum": 0.0,
                "cum80_clients": "",
                "cum80_weight_sum": 0.0,
            }
        weights = [w / total for w in weights]
        indexed = list(enumerate(weights, start=1))
        indexed.sort(key=lambda item: item[1], reverse=True)
        entropy = -sum(w * math.log(max(w, 1e-12)) for _, w in indexed if w > 0.0)
        neff = 1.0 / max(sum(w * w for _, w in indexed), 1e-12)
        active = sum(1 for _, w in indexed if w > 0.0)
        top3 = indexed[:3]
        cum_clients = []
        cum_weight = 0.0
        for client_id, weight in indexed:
            if cum_weight < 0.8:
                cum_clients.append(client_id)
                cum_weight += weight
        return {
            "N_eff": neff,
            "entropy": entropy,
            "max_client_weight": indexed[0][1],
            "active_client_count": active,
            "top1_client": indexed[0][0],
            "top1_weight": indexed[0][1],
            "top3_clients": ";".join(str(client_id) for client_id, _ in top3),
            "top3_weight_sum": sum(weight for _, weight in top3),
            "cum80_clients": ";".join(str(client_id) for client_id in cum_clients),
            "cum80_weight_sum": cum_weight,
        }

    def _compute_class_coverage_diagnostics(self, client_ids):
        covered = set()
        for client_id in client_ids:
            if int(client_id) - 1 < 0 or int(client_id) - 1 >= len(self.ha_vsmc_client_class_stats):
                continue
            covered.update(self.ha_vsmc_client_class_stats[int(client_id) - 1].get("labels", []))
        return len(covered)

    def _write_ha_vsmc_round_weight_exports(self, round_idx, infos):
        if not bool(getattr(self.args, "ha_vsmc_save_client_weights", True)):
            return
        logs_dir = self._ha_vsmc_logs_dir()
        os.makedirs(logs_dir, exist_ok=True)
        matrix_path = os.path.join(logs_dir, f"expert_client_weights_round_{int(round_idx)}.csv")
        diag_path = os.path.join(logs_dir, f"expert_client_weight_diagnostics_round_{int(round_idx)}.csv")
        matrix_rows = []
        diag_rows = []
        for info in infos:
            weights = list(info.get("weights", []) or [])
            matrix_row = {
                "round": int(round_idx),
                "block_id": int(info.get("block_id", -1)),
                "expert_id": int(info.get("expert_id", -1)),
                "expert_scope": str(info.get("expert_scope", "")),
            }
            for idx, weight in enumerate(weights, start=1):
                matrix_row[f"client_{idx}"] = float(weight)
            matrix_rows.append(matrix_row)
            diag = self._compute_weight_diagnostics(weights)
            top3_clients = [int(x) for x in diag["top3_clients"].split(';') if x]
            cum80_clients = [int(x) for x in diag["cum80_clients"].split(';') if x]
            diag_rows.append(
                {
                    "round": int(round_idx),
                    "block_id": int(info.get("block_id", -1)),
                    "expert_id": int(info.get("expert_id", -1)),
                    "expert_scope": str(info.get("expert_scope", "")),
                    **diag,
                    "class_coverage_top1": self._compute_class_coverage_diagnostics([diag["top1_client"]]) if diag["top1_client"] > 0 else 0,
                    "class_coverage_top3": self._compute_class_coverage_diagnostics(top3_clients),
                    "class_coverage_cum80": self._compute_class_coverage_diagnostics(cum80_clients),
                    "class_coverage_active": self._compute_class_coverage_diagnostics(list(range(1, len(weights) + 1))),
                    "reliability_score_mean": float(info.get("reliability_score_mean", 0.0)),
                    "reliability_score_std": float(info.get("reliability_score_std", 0.0)),
                    "score_u_mean": float(info.get("usage_signal_mean", 0.0)),
                    "score_u_std": float(info.get("usage_signal_std", 0.0)),
                    "score_h_mean": float(info.get("history_signal_mean", 0.0)),
                    "score_h_std": float(info.get("history_signal_std", 0.0)),
                    "score_c_mean": float(info.get("consensus_signal_mean", 0.0)),
                    "score_c_std": float(info.get("consensus_signal_std", 0.0)),
                    "score_d_mean": float(info.get("drift_signal_mean", 0.0)),
                    "score_d_std": float(info.get("drift_signal_std", 0.0)),
                    "score_f_mean": float(info.get("function_signal_mean", 0.0)),
                    "score_f_std": float(info.get("function_signal_std", 0.0)),
                    "proxy_val_loss_delta": float(info.get("proxy_val_loss_delta", 0.0)),
                    "proxy_val_acc_delta": float(info.get("proxy_val_acc_delta", 0.0)),
                }
            )
        if matrix_rows:
            fieldnames = list(matrix_rows[0].keys())
            with open(matrix_path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(matrix_rows)
        if diag_rows:
            fieldnames = list(diag_rows[0].keys())
            with open(diag_path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(diag_rows)

    def _compute_client_weights(
        self,
        client_updates,
        mode: str = "sample_count",
    ) -> List[float]:
        if not client_updates:
            raise ValueError("Cannot aggregate empty client updates")
        normalized_mode = str(mode).lower()
        num_clients = len(client_updates)
        if normalized_mode == "uniform":
            return [1.0 / float(num_clients)] * num_clients

        sample_counts = [max(float(num_samples), 0.0) for _, num_samples, _ in client_updates]
        total_samples = sum(sample_counts)
        if total_samples <= 0.0:
            raise ValueError("Cannot aggregate client updates with zero samples")
        return [count / total_samples for count in sample_counts]

    def select_server_adapt_params(self, model, config):
        selected_params: List[nn.Parameter] = []
        selected_param_names: List[str] = []
        named_params = dict(model.named_parameters())
        seen_ids = set()

        def add_param(name: str):
            param = named_params.get(name)
            if param is None or id(param) in seen_ids or self._is_expert_param_name(name):
                return
            if not torch.is_floating_point(param.data):
                return
            seen_ids.add(id(param))
            selected_params.append(param)
            selected_param_names.append(name)

        def add_module_params(module: Optional[nn.Module], prefix: str):
            if module is None:
                return
            for local_name, _ in module.named_parameters(prefix=prefix):
                add_param(local_name)

        if bool(getattr(config, "server_adapt_update_head", False)):
            explicit_head_found = False
            for module_name in ("classifier", "head", "fc", "out_proj"):
                module = getattr(model, module_name, None)
                if isinstance(module, nn.Module):
                    add_module_params(module, module_name)
                    explicit_head_found = True
                    if module_name in {"classifier", "head"}:
                        break
            if not explicit_head_found:
                for name in named_params:
                    lower_name = name.lower()
                    if any(token in lower_name for token in ("classifier", "head", "out_proj")) and ".attn." not in lower_name:
                        add_param(name)

        if bool(getattr(config, "server_adapt_update_final_norm", False)):
            for module_name in ("norm", "final_norm", "layernorm", "ln"):
                module = getattr(model, module_name, None)
                if isinstance(module, nn.Module):
                    add_module_params(module, module_name)
                    break

        if bool(getattr(config, "server_adapt_update_moe_layernorm", False)):
            for block_idx, block in enumerate(getattr(model, "blocks", [])):
                add_module_params(getattr(block, "ffn_norm", None), f"blocks.{block_idx}.ffn_norm")
                add_module_params(getattr(block, "attn_norm", None), f"blocks.{block_idx}.attn_norm")

        if self._server_adapt_router_temperature_enabled(config):
            for name in named_params:
                if self._is_server_router_temperature_name(name):
                    add_param(name)

        if self._server_adapt_router_bias_enabled(config):
            for name in named_params:
                if self._is_server_router_bias_name(name):
                    add_param(name)

        if self._server_adapt_full_router_enabled(config):
            for name in named_params:
                if self._is_router_param_name(name) and not self._is_server_router_temperature_name(name) and not self._is_server_router_bias_name(name):
                    add_param(name)

        if bool(getattr(config, "server_adapt_update_full_dense", False)):
            for name in named_params:
                if self._is_expert_param_name(name):
                    continue
                if self._is_router_param_name(name):
                    if self._is_server_router_temperature_name(name) and self._server_adapt_router_temperature_enabled(config):
                        add_param(name)
                    elif self._is_server_router_bias_name(name) and self._server_adapt_router_bias_enabled(config):
                        add_param(name)
                    elif self._server_adapt_full_router_enabled(config):
                        add_param(name)
                    continue
                add_param(name)

        if bool(getattr(config, "server_adapt_update_experts", False)):
            self.logger.warning("Ignoring server_adapt_update_experts=True because expert adaptation is explicitly disabled.")

        return selected_params, selected_param_names

    def _collect_server_adapt_batches(self, config):
        train_target = max(int(getattr(config, "server_adapt_train_batches", 0)), 0)
        accept_target = max(int(getattr(config, "server_adapt_accept_batches", 0)), 0)
        split_accept = bool(getattr(config, "server_adapt_split_accept", True))
        total_target = max(int(getattr(config, "server_adapt_batches", 0)), train_target + accept_target)
        batches = []
        for batch_idx, batch in enumerate(self.server_val_loader):
            if batch_idx >= total_target:
                break
            batches.append(batch)
        if split_accept:
            train_batches = batches[:train_target]
            accept_batches = batches[train_target:train_target + accept_target]
        else:
            train_batches = batches[: max(train_target, 1 if batches else 0)]
            accept_batches = (
                batches[:accept_target]
                if accept_target > 0
                else list(train_batches)
            )
        return train_batches, accept_batches

    def _move_batch_to_device(self, batch):
        inputs, labels = batch
        return inputs.to(self.device), labels.to(self.device)

    @staticmethod
    def _compute_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        masked_logits = logits.clone()
        masked_logits.scatter_(1, labels.unsqueeze(1), float("-inf"))
        max_wrong_logits = masked_logits.max(dim=-1).values
        return (true_logits - max_wrong_logits).mean()

    @staticmethod
    def _compute_margin_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        masked_logits = logits.clone()
        masked_logits.scatter_(1, labels.unsqueeze(1), float("-inf"))
        max_wrong_logits = masked_logits.max(dim=-1).values
        return true_logits - max_wrong_logits

    def _compute_router_entropy_penalty(self, router_stats) -> Tuple[torch.Tensor, bool]:
        router_probs_by_layer = None if router_stats is None else router_stats.get("router_probs_by_layer")
        if router_probs_by_layer is None:
            return torch.zeros((), device=self.device), False
        penalty = torch.zeros((), device=router_probs_by_layer.device)
        for layer_probs in router_probs_by_layer:
            load = layer_probs.mean(dim=0)
            num_experts = max(int(load.numel()), 2)
            entropy = -(
                load.clamp_min(1e-12) * torch.log(load.clamp_min(1e-12))
            ).sum() / torch.log(load.new_tensor(float(num_experts)))
            if float(entropy.detach().item()) < 0.65:
                penalty = penalty + (load.new_tensor(0.65) - entropy) ** 2
        return penalty, True

    def _compute_router_load_entropy_metric(self, router_stats) -> Tuple[Optional[float], bool]:
        counts_by_layer = None if router_stats is None else router_stats.get("sample_expert_counts_by_layer")
        if counts_by_layer is None:
            return None, False
        entropies: List[float] = []
        for layer_counts in counts_by_layer:
            counts = layer_counts.detach().to(dtype=torch.float32).sum(dim=0)
            total = float(counts.sum().item())
            if total <= 0.0:
                continue
            probs = counts / counts.sum().clamp_min(1e-12)
            num_experts = max(int(probs.numel()), 2)
            entropy = -(
                probs.clamp_min(1e-12) * torch.log(probs.clamp_min(1e-12))
            ).sum() / torch.log(probs.new_tensor(float(num_experts)))
            entropies.append(float(entropy.item()))
        if not entropies:
            return None, False
        return sum(entropies) / len(entropies), True

    def compute_router_teacher_labels(self, model, batch, config):
        teacher_labels_by_block: Dict[int, torch.Tensor] = {}
        teacher_confidence_by_block: Dict[int, torch.Tensor] = {}
        diagnostics: Dict[str, Any] = {
            "teacher_proxy_mode": False,
            "counterfactual_success": False,
            "counterfactual_fallback_reason": "",
            "teacher_label_entropy_mean": None,
            "teacher_conf_mean": None,
            "teacher_conf_std": None,
            "teacher_valid_ratio": None,
        }
        if not self._teacher_router_enabled(config):
            diagnostics["counterfactual_fallback_reason"] = "teacher_router_disabled"
            return teacher_labels_by_block, teacher_confidence_by_block, diagnostics

        inputs, labels = self._move_batch_to_device(batch)
        was_training = bool(model.training)
        stopgrad = bool(getattr(config, "server_adapt_teacher_loss_stopgrad", True))
        context = torch.no_grad() if stopgrad else nullcontext()
        try:
            model.eval()
            with context:
                _, aux = model(
                    inputs,
                    return_aux=True,
                    server_adapt_config=config,
                    return_moe_inputs=True,
                )
                moe_inputs_by_layer = aux.get("moe_inputs_by_layer")
                residual_inputs_by_layer = aux.get("block_residual_inputs_by_layer")
                router_logits_tokens_by_layer = aux.get("router_logits_tokens_by_layer")
                if (
                    moe_inputs_by_layer is None
                    or residual_inputs_by_layer is None
                    or router_logits_tokens_by_layer is None
                ):
                    diagnostics["counterfactual_fallback_reason"] = "missing_moe_intermediates"
                    return teacher_labels_by_block, teacher_confidence_by_block, diagnostics

                num_layers = int(moe_inputs_by_layer.size(0))
                num_experts = int(router_logits_tokens_by_layer.size(-1))
                topk_setting = str(getattr(config, "server_adapt_teacher_topk_experts", "all")).lower()
                teacher_metric = str(getattr(config, "server_adapt_teacher_metric", "margin")).lower()
                label_entropy_values: List[float] = []
                conf_chunks: List[torch.Tensor] = []
                valid_total = 0
                total_possible = 0

                for block_idx in range(num_layers):
                    block = model.blocks[block_idx]
                    moe_input = moe_inputs_by_layer[block_idx]
                    residual_input = residual_inputs_by_layer[block_idx]
                    router_logits_tokens = router_logits_tokens_by_layer[block_idx][:, 0, :]
                    batch_size = int(router_logits_tokens.size(0))
                    candidate_mask = torch.ones(
                        batch_size,
                        num_experts,
                        dtype=torch.bool,
                        device=router_logits_tokens.device,
                    )
                    if topk_setting != "all":
                        try:
                            topk = max(1, min(int(topk_setting), num_experts))
                        except ValueError:
                            topk = num_experts
                        candidate_mask = torch.zeros_like(candidate_mask)
                        topk_indices = torch.topk(router_logits_tokens, k=topk, dim=-1).indices
                        candidate_mask.scatter_(1, topk_indices, True)

                    scores = router_logits_tokens.new_full((batch_size, num_experts), float("-inf"))
                    for expert_idx, expert in enumerate(block.switch_ffn.experts):
                        sample_mask = candidate_mask[:, expert_idx]
                        if not bool(sample_mask.any()):
                            continue
                        expert_output = expert(moe_input[sample_mask])
                        block_output = residual_input[sample_mask] + expert_output
                        logits_k = model.forward_from_block_output(
                            block_output,
                            start_block_index=block_idx + 1,
                            server_adapt_config=config,
                        )
                        if teacher_metric == "margin":
                            score = self._compute_margin_per_sample(logits_k, labels[sample_mask])
                        else:
                            true_logits = logits_k.gather(1, labels[sample_mask].unsqueeze(1)).squeeze(1)
                            score = true_logits
                        scores[sample_mask, expert_idx] = score

                    if not torch.isfinite(scores).any():
                        continue
                    teacher_labels = torch.argmax(scores, dim=-1)
                    topk_scores = torch.topk(scores, k=min(2, num_experts), dim=-1).values
                    if topk_scores.size(1) > 1:
                        teacher_conf = topk_scores[:, 0] - topk_scores[:, 1]
                    else:
                        teacher_conf = topk_scores[:, 0]
                    teacher_labels_by_block[block_idx] = teacher_labels.detach()
                    teacher_confidence_by_block[block_idx] = teacher_conf.detach()

                    valid_mask = torch.isfinite(teacher_conf)
                    threshold = float(getattr(config, "server_adapt_teacher_conf_threshold", 0.0))
                    if threshold > 0.0:
                        valid_mask = valid_mask & (teacher_conf > threshold)
                    total_possible += int(valid_mask.numel())
                    valid_count = int(valid_mask.sum().item())
                    valid_total += valid_count
                    if valid_count > 0:
                        valid_labels = teacher_labels[valid_mask]
                        counts = torch.bincount(valid_labels, minlength=num_experts).to(dtype=torch.float32)
                        probs = counts / counts.sum().clamp_min(1e-12)
                        entropy = -(
                            probs.clamp_min(1e-12) * torch.log(probs.clamp_min(1e-12))
                        ).sum() / torch.log(probs.new_tensor(float(max(num_experts, 2))))
                        label_entropy_values.append(float(entropy.item()))
                        conf_chunks.append(teacher_conf[valid_mask].detach())

                diagnostics["counterfactual_success"] = bool(teacher_labels_by_block)
                if diagnostics["counterfactual_success"]:
                    diagnostics["counterfactual_fallback_reason"] = ""
                    diagnostics["teacher_label_entropy_mean"] = (
                        sum(label_entropy_values) / len(label_entropy_values)
                        if label_entropy_values
                        else None
                    )
                    if conf_chunks:
                        all_conf = torch.cat(conf_chunks, dim=0)
                        diagnostics["teacher_conf_mean"] = float(all_conf.mean().item())
                        diagnostics["teacher_conf_std"] = float(all_conf.std(unbiased=False).item())
                    diagnostics["teacher_valid_ratio"] = (
                        float(valid_total) / float(max(total_possible, 1))
                    )
                else:
                    diagnostics["counterfactual_fallback_reason"] = "no_valid_teacher_labels"
        except Exception as exc:
            diagnostics["teacher_proxy_mode"] = False
            diagnostics["counterfactual_success"] = False
            diagnostics["counterfactual_fallback_reason"] = str(exc)
            teacher_labels_by_block = {}
            teacher_confidence_by_block = {}
        finally:
            model.train(was_training)
        return teacher_labels_by_block, teacher_confidence_by_block, diagnostics

    def _compute_router_teacher_loss(
        self,
        router_stats,
        teacher_labels_by_block: Dict[int, torch.Tensor],
        teacher_confidence_by_block: Dict[int, torch.Tensor],
        diagnostics: Dict[str, Any],
        config,
    ):
        router_logits_tokens_by_layer = None if router_stats is None else router_stats.get("router_logits_tokens_by_layer")
        if router_logits_tokens_by_layer is None or not teacher_labels_by_block:
            return torch.zeros((), device=self.device), TeacherRouterMetrics(
                teacher_router_loss=None,
                teacher_label_entropy_mean=diagnostics.get("teacher_label_entropy_mean"),
                teacher_conf_mean=diagnostics.get("teacher_conf_mean"),
                teacher_conf_std=diagnostics.get("teacher_conf_std"),
                teacher_valid_ratio=diagnostics.get("teacher_valid_ratio"),
                router_teacher_acc=None,
                teacher_proxy_mode=bool(diagnostics.get("teacher_proxy_mode", False)),
                counterfactual_success=bool(diagnostics.get("counterfactual_success", False)),
                counterfactual_fallback_reason=str(diagnostics.get("counterfactual_fallback_reason", "")),
            )

        loss_terms: List[torch.Tensor] = []
        total_correct = 0
        total_valid = 0
        threshold = float(getattr(config, "server_adapt_teacher_conf_threshold", 0.0))
        for block_idx, teacher_labels in teacher_labels_by_block.items():
            router_logits = router_logits_tokens_by_layer[block_idx][:, 0, :]
            teacher_conf = teacher_confidence_by_block.get(block_idx)
            valid_mask = torch.ones_like(teacher_labels, dtype=torch.bool)
            if teacher_conf is not None:
                valid_mask = valid_mask & torch.isfinite(teacher_conf)
                if threshold > 0.0:
                    valid_mask = valid_mask & (teacher_conf > threshold)
            if not bool(valid_mask.any()):
                continue
            loss_terms.append(F.cross_entropy(router_logits[valid_mask], teacher_labels[valid_mask], reduction="mean"))
            preds = router_logits.argmax(dim=-1)
            total_correct += int((preds[valid_mask] == teacher_labels[valid_mask]).sum().item())
            total_valid += int(valid_mask.sum().item())

        if loss_terms:
            teacher_loss = torch.stack(loss_terms).mean()
            teacher_loss_value = float(teacher_loss.detach().item())
        else:
            teacher_loss = torch.zeros((), device=self.device)
            teacher_loss_value = None
        teacher_metrics = TeacherRouterMetrics(
            teacher_router_loss=teacher_loss_value,
            teacher_label_entropy_mean=diagnostics.get("teacher_label_entropy_mean"),
            teacher_conf_mean=diagnostics.get("teacher_conf_mean"),
            teacher_conf_std=diagnostics.get("teacher_conf_std"),
            teacher_valid_ratio=diagnostics.get("teacher_valid_ratio"),
            router_teacher_acc=(total_correct / total_valid) if total_valid > 0 else None,
            teacher_proxy_mode=bool(diagnostics.get("teacher_proxy_mode", False)),
            counterfactual_success=bool(diagnostics.get("counterfactual_success", False)),
            counterfactual_fallback_reason=str(diagnostics.get("counterfactual_fallback_reason", "")),
        )
        return teacher_loss, teacher_metrics

    def compute_server_adapt_loss(self, model, batch, config, progress: Optional[float] = None, use_teacher_router: Optional[bool] = None):
        inputs, labels = self._move_batch_to_device(batch)
        logits, router_stats = model(inputs, return_aux=True, server_adapt_config=config)
        ce_loss = self.criterion(logits, labels)
        margin = self._compute_margin(logits, labels)
        margin_loss = -margin
        router_entropy_penalty, router_aux_available = self._compute_router_entropy_penalty(router_stats)
        router_load_entropy, _ = self._compute_router_load_entropy_metric(router_stats)
        teacher_router_loss = torch.zeros((), device=ce_loss.device)
        teacher_metrics = TeacherRouterMetrics(
            teacher_router_loss=None,
            teacher_label_entropy_mean=None,
            teacher_conf_mean=None,
            teacher_conf_std=None,
            teacher_valid_ratio=None,
            router_teacher_acc=None,
            teacher_proxy_mode=False,
            counterfactual_success=False,
            counterfactual_fallback_reason="teacher_router_disabled",
        )
        teacher_enabled = self._teacher_router_enabled(config, progress=progress)
        if use_teacher_router is not None:
            teacher_enabled = bool(teacher_enabled and use_teacher_router)
        if teacher_enabled:
            teacher_labels_by_block, teacher_confidence_by_block, diagnostics = self.compute_router_teacher_labels(model, batch, config)
            teacher_router_loss, teacher_metrics = self._compute_router_teacher_loss(
                router_stats,
                teacher_labels_by_block,
                teacher_confidence_by_block,
                diagnostics,
                config,
            )
        adapt_loss = (
            ce_loss
            + float(config.server_adapt_margin_weight) * margin_loss
            + float(config.server_adapt_entropy_weight) * router_entropy_penalty
            + float(getattr(config, "server_adapt_teacher_router_weight", 0.15)) * teacher_router_loss
        )
        metrics = {
            "ce": float(ce_loss.detach().item()),
            "margin": float(margin.detach().item()),
            "router_entropy_penalty": float(router_entropy_penalty.detach().item()),
            "router_aux_available": bool(router_aux_available),
            "router_load_entropy": router_load_entropy,
            "teacher_router_enabled": bool(teacher_enabled),
            "teacher_router_loss": teacher_metrics.teacher_router_loss,
            "teacher_label_entropy_mean": teacher_metrics.teacher_label_entropy_mean,
            "teacher_conf_mean": teacher_metrics.teacher_conf_mean,
            "teacher_conf_std": teacher_metrics.teacher_conf_std,
            "teacher_valid_ratio": teacher_metrics.teacher_valid_ratio,
            "router_teacher_acc": teacher_metrics.router_teacher_acc,
            "teacher_proxy_mode": teacher_metrics.teacher_proxy_mode,
            "counterfactual_success": teacher_metrics.counterfactual_success,
            "counterfactual_fallback_reason": teacher_metrics.counterfactual_fallback_reason,
        }
        return adapt_loss, metrics

    def evaluate_teacher_router_metrics(self, model, batches, config, progress: Optional[float] = None):
        if not batches or not self._teacher_router_enabled(config, progress=progress):
            return TeacherRouterMetrics(
                teacher_router_loss=None,
                teacher_label_entropy_mean=None,
                teacher_conf_mean=None,
                teacher_conf_std=None,
                teacher_valid_ratio=None,
                router_teacher_acc=None,
                teacher_proxy_mode=False,
                counterfactual_success=False,
                counterfactual_fallback_reason="teacher_router_disabled",
            )
        total_loss = 0.0
        loss_weight = 0
        entropy_values: List[float] = []
        conf_means: List[float] = []
        conf_stds: List[float] = []
        valid_ratios: List[float] = []
        acc_numer = 0.0
        acc_denom = 0
        proxy_mode = False
        counterfactual_success = False
        fallback_reason = ""
        for batch in batches[: max(int(getattr(config, "server_adapt_teacher_batches", 2)), 1)]:
            inputs, _ = self._move_batch_to_device(batch)
            with torch.no_grad():
                _, router_stats = model(inputs, return_aux=True, server_adapt_config=config)
            teacher_labels_by_block, teacher_confidence_by_block, diagnostics = self.compute_router_teacher_labels(model, batch, config)
            teacher_loss, teacher_metrics = self._compute_router_teacher_loss(
                router_stats,
                teacher_labels_by_block,
                teacher_confidence_by_block,
                diagnostics,
                config,
            )
            if teacher_metrics.teacher_router_loss is not None:
                batch_size = int(inputs.size(0))
                total_loss += float(teacher_loss.detach().item()) * batch_size
                loss_weight += batch_size
            if teacher_metrics.teacher_label_entropy_mean is not None:
                entropy_values.append(float(teacher_metrics.teacher_label_entropy_mean))
            if teacher_metrics.teacher_conf_mean is not None:
                conf_means.append(float(teacher_metrics.teacher_conf_mean))
            if teacher_metrics.teacher_conf_std is not None:
                conf_stds.append(float(teacher_metrics.teacher_conf_std))
            if teacher_metrics.teacher_valid_ratio is not None:
                valid_ratios.append(float(teacher_metrics.teacher_valid_ratio))
            if teacher_metrics.router_teacher_acc is not None:
                batch_size = int(inputs.size(0))
                acc_numer += float(teacher_metrics.router_teacher_acc) * batch_size
                acc_denom += batch_size
            proxy_mode = proxy_mode or bool(teacher_metrics.teacher_proxy_mode)
            counterfactual_success = counterfactual_success or bool(teacher_metrics.counterfactual_success)
            if teacher_metrics.counterfactual_fallback_reason:
                fallback_reason = teacher_metrics.counterfactual_fallback_reason

        return TeacherRouterMetrics(
            teacher_router_loss=(total_loss / loss_weight) if loss_weight > 0 else None,
            teacher_label_entropy_mean=(sum(entropy_values) / len(entropy_values)) if entropy_values else None,
            teacher_conf_mean=(sum(conf_means) / len(conf_means)) if conf_means else None,
            teacher_conf_std=(sum(conf_stds) / len(conf_stds)) if conf_stds else None,
            teacher_valid_ratio=(sum(valid_ratios) / len(valid_ratios)) if valid_ratios else None,
            router_teacher_acc=(acc_numer / acc_denom) if acc_denom > 0 else None,
            teacher_proxy_mode=proxy_mode,
            counterfactual_success=counterfactual_success,
            counterfactual_fallback_reason=fallback_reason,
        )

    def evaluate_adapt_metrics(self, model, batches, config):
        if not batches:
            return AdaptMetrics(loss=float("inf"), acc=0.0, margin=0.0, router_load_entropy=None)
        model.eval()
        total_samples = 0
        total_loss = 0.0
        total_correct = 0
        total_margin = 0.0
        entropy_sum = 0.0
        entropy_weight = 0
        with torch.no_grad():
            for batch in batches:
                inputs, labels = self._move_batch_to_device(batch)
                logits, router_stats = model(inputs, return_aux=True, server_adapt_config=config)
                ce_loss = self.criterion(logits, labels)
                margin = self._compute_margin(logits, labels)
                router_entropy_penalty, _ = self._compute_router_entropy_penalty(router_stats)
                adapt_loss = (
                    ce_loss
                    + float(config.server_adapt_margin_weight) * (-margin)
                    + float(config.server_adapt_entropy_weight) * router_entropy_penalty
                )
                batch_size = int(labels.size(0))
                preds = logits.argmax(dim=1)
                total_samples += batch_size
                total_loss += float(adapt_loss.item()) * batch_size
                total_margin += float(margin.item()) * batch_size
                total_correct += int((preds == labels).sum().item())
                router_load_entropy, router_aux_available = self._compute_router_load_entropy_metric(router_stats)
                if router_aux_available and router_load_entropy is not None:
                    entropy_sum += float(router_load_entropy) * batch_size
                    entropy_weight += batch_size
        if total_samples == 0:
            return AdaptMetrics(loss=float("inf"), acc=0.0, margin=0.0, router_load_entropy=None)
        return AdaptMetrics(
            loss=total_loss / total_samples,
            acc=total_correct / total_samples,
            margin=total_margin / total_samples,
            router_load_entropy=(entropy_sum / entropy_weight) if entropy_weight > 0 else None,
        )

    @staticmethod
    def _clone_selected_params(selected_params, selected_names):
        return {
            name: param.detach().clone()
            for name, param in zip(selected_names, selected_params)
        }

    def _restore_selected_params(self, model, before_state):
        named_params = dict(model.named_parameters())
        for name, tensor in before_state.items():
            if name in named_params:
                named_params[name].data.copy_(tensor)

    def _freeze_all_params_except(self, selected_params):
        selected_ids = {id(param) for param in selected_params}
        original_trainability = {}
        for name, param in self.model.named_parameters():
            original_trainability[name] = bool(param.requires_grad)
            param.requires_grad = id(param) in selected_ids
            if not param.requires_grad:
                param.grad = None
        return original_trainability

    def _restore_param_trainability(self, original_trainability):
        for name, param in self.model.named_parameters():
            if name in original_trainability:
                param.requires_grad = bool(original_trainability[name])
            if not param.requires_grad:
                param.grad = None

    def _assert_no_expert_grads(self):
        for name, param in self.model.named_parameters():
            if self._is_expert_param_name(name) and param.grad is not None:
                raise RuntimeError(f"Expert parameter received gradient during server adaptation: {name}")

    @staticmethod
    def _selected_delta_norm(selected_params, selected_names, before_state):
        total = 0.0
        for name, param in zip(selected_names, selected_params):
            reference = before_state.get(name)
            if reference is None:
                continue
            delta = param.detach() - reference
            total += float(delta.pow(2).sum().item())
        return total ** 0.5

    def _apply_server_adapt_delta_ema(self, selected_params, selected_names, before_state, config):
        raw_norm_sq = 0.0
        ema_norm_sq = 0.0
        decay = float(getattr(config, "server_adapt_ema_decay", 0.8))
        for name, param in zip(selected_names, selected_params):
            before_tensor = before_state[name]
            after_tensor = param.detach().clone()
            raw_delta = after_tensor - before_tensor
            raw_norm_sq += float(raw_delta.pow(2).sum().item())
            if name not in self.server_adapt_ema_delta:
                self.server_adapt_ema_delta[name] = torch.zeros_like(raw_delta)
            ema_delta = (
                decay * self.server_adapt_ema_delta[name].to(raw_delta.device)
                + (1.0 - decay) * raw_delta
            )
            self.server_adapt_ema_delta[name] = ema_delta.detach().clone()
            param.data.copy_(before_tensor + ema_delta)
            ema_norm_sq += float(ema_delta.pow(2).sum().item())
        return raw_norm_sq ** 0.5, ema_norm_sq ** 0.5

    def _decay_server_adapt_ema_buffers(self, selected_names, config):
        decay = float(getattr(config, "server_adapt_ema_decay", 0.8))
        for name in selected_names:
            if name in self.server_adapt_ema_delta:
                self.server_adapt_ema_delta[name].mul_(decay)

    def _update_server_adapt_accept_rate(self, accepted, config):
        self.server_adapt_accept_history.append(int(bool(accepted)))
        window = max(int(getattr(config, "server_adapt_accept_rate_window", 10)), 1)
        recent = self.server_adapt_accept_history[-window:]
        window_rate = sum(recent) / len(recent) if recent else 0.0
        if bool(getattr(config, "server_adapt_accept_rate_control", True)):
            if window_rate > float(getattr(config, "server_adapt_accept_rate_high", 0.9)):
                self.server_adapt_accept_threshold_multiplier *= 1.5
            elif window_rate < float(getattr(config, "server_adapt_accept_rate_low", 0.1)):
                self.server_adapt_accept_threshold_multiplier *= 0.8
            self.server_adapt_accept_threshold_multiplier = max(
                float(getattr(config, "server_adapt_accept_multiplier_min", 0.5)),
                min(
                    float(getattr(config, "server_adapt_accept_multiplier_max", 4.0)),
                    float(self.server_adapt_accept_threshold_multiplier),
                ),
            )
        return window_rate

    def _record_server_adapt_metrics(self, record: Dict[str, Any]):
        record_server_adapt_metric(record)
        self.logger.info(
            f"--server_adapt --round:{record.get('round')} "
            f"--attempted:{record.get('adapt_attempted')} "
            f"--accepted:{record.get('adapt_accepted')} "
            f"--reason:{record.get('adapt_accept_reason')} "
            f"--phase:{record.get('adapt_phase', '')} "
            f"--accept_loss_delta:{self._safe_float(record.get('loss_delta_accept'), 0.0):.6f} "
            f"--accept_margin_delta:{self._safe_float(record.get('margin_delta_accept'), 0.0):.6f} "
            f"--threshold_multiplier:{self._safe_float(record.get('accept_threshold_multiplier'), 1.0):.4f}"
        )

    def server_side_adapt_with_rollback(self, model, server_val_loader, round_id, total_rounds, config):
        del server_val_loader
        progress = float(round_id) / float(max(total_rounds, 1))
        nan = float("nan")
        base_record: Dict[str, Any] = {
            "round": int(round_id),
            "server_adapt_version": str(getattr(config, "server_adapt_version", "v11c_strict_accept")),
            "adapt_attempted": False,
            "adapt_accepted": False,
            "adapt_accept_reason": "",
            "accept_reason": "",
            "adapt_phase": "",
            "split_accept": bool(getattr(config, "server_adapt_split_accept", True)),
            "train_batches": 0,
            "accept_batches": 0,
            "strict_accept": bool(getattr(config, "server_adapt_strict_accept", True)),
            "selected_param_count": 0,
            "selected_param_names": "[]",
            "progress": progress,
            "accept_threshold_multiplier": float(self.server_adapt_accept_threshold_multiplier),
            "accept_rate_window": nan,
            "accept_rate_window_size": 0,
            "router_bias_enabled": self._server_adapt_router_bias_enabled(config),
            "router_temperature_enabled": self._server_adapt_router_temperature_enabled(config),
            "full_router_enabled": self._server_adapt_full_router_enabled(config),
            "ema_applied": False,
            "rollback_reason": "",
            "error_info": "",
            "router_entropy_delta_accept": nan,
            "loss_evidence": False,
            "margin_evidence": False,
            "acc_evidence": False,
            "router_evidence": False,
            "evidence_count": 0,
            "evidence_required": 0,
            "near_best": False,
            "near_best_reject": False,
            "cooldown_active": False,
            "cooldown_remaining": int(self.server_adapt_cooldown_remaining),
            "cooldown_skip": False,
            "rescue_adapt": False,
            "teacher_router_enabled": self._teacher_router_enabled(config, progress=progress),
            "teacher_router_loss_before": nan,
            "teacher_router_loss_after": nan,
            "teacher_router_weight": float(getattr(config, "server_adapt_teacher_router_weight", 0.15)),
            "teacher_label_entropy_mean": nan,
            "teacher_conf_mean": nan,
            "teacher_conf_std": nan,
            "teacher_valid_ratio": nan,
            "router_teacher_acc_before": nan,
            "router_teacher_acc_after": nan,
            "teacher_proxy_mode": False,
            "counterfactual_success": False,
            "counterfactual_fallback_reason": "",
        }
        if progress < float(getattr(config, "server_adapt_start_frac", 0.5)):
            base_record["adapt_accept_reason"] = "before_start_frac"
            base_record["rollback_reason"] = "before_start_frac"
            self._record_server_adapt_metrics(base_record)
            return model

        train_batches, accept_batches = self._collect_server_adapt_batches(config)
        base_record["train_batches"] = len(train_batches)
        base_record["accept_batches"] = len(accept_batches)
        if not train_batches or not accept_batches:
            base_record["adapt_accept_reason"] = "insufficient_split_batches"
            base_record["rollback_reason"] = "insufficient_split_batches"
            self._record_server_adapt_metrics(base_record)
            return model

        selected_params, selected_names = self.select_server_adapt_params(model, config)
        base_record["selected_param_count"] = len(selected_names)
        base_record["selected_param_names"] = json.dumps(selected_names)
        if not selected_params:
            base_record["adapt_accept_reason"] = "no_selected_params"
            base_record["rollback_reason"] = "no_selected_params"
            self._record_server_adapt_metrics(base_record)
            return model

        before_state = self._clone_selected_params(selected_params, selected_names)
        before_train_metrics = self.evaluate_adapt_metrics(model, train_batches, config)
        before_accept_metrics = self.evaluate_adapt_metrics(model, accept_batches, config)
        before_accept_teacher_metrics = self.evaluate_teacher_router_metrics(
            model,
            accept_batches,
            config,
            progress=progress,
        )
        base_record.update(
            {
                "adapt_attempted": True,
                "train_loss_before": before_train_metrics.loss,
                "train_acc_before": before_train_metrics.acc,
                "train_margin_before": before_train_metrics.margin,
                "accept_loss_before": before_accept_metrics.loss,
                "accept_acc_before": before_accept_metrics.acc,
                "accept_margin_before": before_accept_metrics.margin,
                "router_load_entropy_before": self._safe_float(before_accept_metrics.router_load_entropy),
                "router_aux_available": before_accept_metrics.router_load_entropy is not None,
                "adapt_loss_before": before_accept_metrics.loss,
                "adapt_margin_before": before_accept_metrics.margin,
                "teacher_router_loss_before": self._safe_float(before_accept_teacher_metrics.teacher_router_loss),
                "teacher_label_entropy_mean": self._safe_float(before_accept_teacher_metrics.teacher_label_entropy_mean),
                "teacher_conf_mean": self._safe_float(before_accept_teacher_metrics.teacher_conf_mean),
                "teacher_conf_std": self._safe_float(before_accept_teacher_metrics.teacher_conf_std),
                "teacher_valid_ratio": self._safe_float(before_accept_teacher_metrics.teacher_valid_ratio),
                "router_teacher_acc_before": self._safe_float(before_accept_teacher_metrics.router_teacher_acc),
                "teacher_proxy_mode": bool(before_accept_teacher_metrics.teacher_proxy_mode),
                "counterfactual_success": bool(before_accept_teacher_metrics.counterfactual_success),
                "counterfactual_fallback_reason": str(before_accept_teacher_metrics.counterfactual_fallback_reason),
            }
        )

        original_trainability = None
        was_training = bool(model.training)
        try:
            optimizer = torch.optim.AdamW(
                selected_params,
                lr=float(getattr(config, "server_adapt_lr", 1e-4)),
                weight_decay=float(getattr(config, "server_adapt_weight_decay", 0.0)),
            )
            original_trainability = self._freeze_all_params_except(selected_params)
            model.train()
            teacher_batch_limit = max(int(getattr(config, "server_adapt_teacher_batches", 2)), 0)
            for _ in range(max(int(getattr(config, "server_adapt_steps", 1)), 1)):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for batch_idx, batch in enumerate(train_batches):
                    use_teacher_router = batch_idx < teacher_batch_limit
                    loss, _ = self.compute_server_adapt_loss(
                        model,
                        batch,
                        config,
                        progress=progress,
                        use_teacher_router=use_teacher_router,
                    )
                    losses.append(loss)
                if not losses:
                    break
                loss = torch.stack(losses).mean()
                loss.backward()
                self._assert_no_expert_grads()
                torch.nn.utils.clip_grad_norm_(selected_params, max_norm=1.0)
                optimizer.step()

            after_train_metrics = self.evaluate_adapt_metrics(model, train_batches, config)
            after_accept_metrics = self.evaluate_adapt_metrics(model, accept_batches, config)
            after_accept_teacher_metrics = self.evaluate_teacher_router_metrics(
                model,
                accept_batches,
                config,
                progress=progress,
            )
            if bool(getattr(config, "server_adapt_strict_accept", True)):
                version = str(getattr(config, "server_adapt_version", "v11c_strict_accept"))
                if version == "v12_teacher_router":
                    accept, reason, debug = teacher_router_accept_rule(
                        before=before_accept_metrics,
                        after=after_accept_metrics,
                        before_teacher=before_accept_teacher_metrics,
                        after_teacher=after_accept_teacher_metrics,
                        progress=progress,
                        config=config,
                    )
                elif version == "v11d_selective_adapt":
                    selective_state = SelectiveAdaptState(
                        best_proxy=float(self.server_adapt_best_proxy),
                        best_accept_loss=float(self.server_adapt_best_accept_loss),
                        cooldown_remaining=int(self.server_adapt_cooldown_remaining),
                        accept_threshold_multiplier=float(self.server_adapt_accept_threshold_multiplier),
                    )
                    accept, reason, debug = selective_adapt_accept_rule(
                        before=before_accept_metrics,
                        after=after_accept_metrics,
                        progress=progress,
                        config=config,
                        state=selective_state,
                    )
                    self.server_adapt_cooldown_remaining = int(selective_state.cooldown_remaining)
                else:
                    accept, reason, debug = strict_adapt_accept_rule(
                        before=before_accept_metrics,
                        after=after_accept_metrics,
                        progress=progress,
                        config=config,
                        accept_threshold_multiplier=float(self.server_adapt_accept_threshold_multiplier),
                    )
            else:
                loss_delta = float(before_accept_metrics.loss - after_accept_metrics.loss)
                margin_delta = float(after_accept_metrics.margin - before_accept_metrics.margin)
                acc_delta = float(after_accept_metrics.acc - before_accept_metrics.acc)
                accept = bool(after_accept_metrics.loss <= before_accept_metrics.loss and after_accept_metrics.acc >= before_accept_metrics.acc)
                reason = "non_strict_accept" if accept else "non_strict_reject"
                debug = {
                    "adapt_phase": "late" if progress >= float(getattr(config, "server_adapt_late_start_frac", 0.8)) else "early",
                    "loss_delta": loss_delta,
                    "margin_delta": margin_delta,
                    "acc_delta": acc_delta,
                    "router_entropy_delta": 0.0,
                    "loss_improved": bool(after_accept_metrics.loss < before_accept_metrics.loss),
                    "acc_improved": bool(after_accept_metrics.acc > before_accept_metrics.acc),
                    "margin_improved": bool(after_accept_metrics.margin > before_accept_metrics.margin),
                    "margin_not_worse": bool(after_accept_metrics.margin >= before_accept_metrics.margin),
                    "loss_not_worse": bool(after_accept_metrics.loss <= before_accept_metrics.loss),
                    "loss_evidence": bool(after_accept_metrics.loss < before_accept_metrics.loss),
                    "margin_evidence": bool(after_accept_metrics.margin > before_accept_metrics.margin),
                    "acc_evidence": bool(after_accept_metrics.acc > before_accept_metrics.acc),
                    "router_evidence": True,
                    "evidence_count": (
                        int(after_accept_metrics.loss < before_accept_metrics.loss)
                        + int(after_accept_metrics.margin > before_accept_metrics.margin)
                        + int(after_accept_metrics.acc > before_accept_metrics.acc)
                        + 1
                    ),
                    "evidence_required": 0,
                    "strong_accept": bool(accept),
                    "acc_accept": bool(after_accept_metrics.acc >= before_accept_metrics.acc),
                    "late_strict_accept": False,
                    "late_phase": bool(progress >= float(getattr(config, "server_adapt_late_start_frac", 0.8))),
                    "late_insufficient_improvement": False,
                    "effective_min_delta": 0.0,
                    "effective_margin_delta": 0.0,
                    "improvement_score": loss_delta + float(getattr(config, "server_adapt_margin_weight", 0.2)) * margin_delta,
                    "near_best": False,
                    "near_best_reject": False,
                    "cooldown_active": False,
                    "cooldown_remaining": int(self.server_adapt_cooldown_remaining),
                    "cooldown_skip": False,
                    "rescue_adapt": False,
                }
            adapt_delta_norm_raw = self._selected_delta_norm(selected_params, selected_names, before_state)
            adapt_delta_norm_ema = 0.0
            ema_applied = False
            if accept:
                if bool(getattr(config, "server_adapt_delta_ema", True)):
                    adapt_delta_norm_raw, adapt_delta_norm_ema = self._apply_server_adapt_delta_ema(
                        selected_params,
                        selected_names,
                        before_state,
                        config,
                    )
                    ema_applied = True
                else:
                    adapt_delta_norm_ema = adapt_delta_norm_raw
                adapt_param_delta_norm = self._selected_delta_norm(selected_params, selected_names, before_state)
                rollback_reason = ""
            else:
                self._restore_selected_params(model, before_state)
                self._decay_server_adapt_ema_buffers(selected_names, config)
                adapt_param_delta_norm = 0.0
                rollback_reason = reason
            live_proxy = after_accept_metrics.acc if accept else before_accept_metrics.acc
            live_accept_loss = after_accept_metrics.loss if accept else before_accept_metrics.loss
            self.server_adapt_best_proxy = max(float(self.server_adapt_best_proxy), float(live_proxy))
            self.server_adapt_best_accept_loss = min(float(self.server_adapt_best_accept_loss), float(live_accept_loss))
            window_rate = self._update_server_adapt_accept_rate(accept, config)
            base_record.update(
                {
                    "adapt_accepted": bool(accept),
                    "adapt_accept_reason": reason,
                    "accept_reason": reason,
                    "adapt_phase": debug["adapt_phase"],
                    "train_loss_after": after_train_metrics.loss,
                    "train_acc_after": after_train_metrics.acc,
                    "train_margin_after": after_train_metrics.margin,
                    "accept_loss_after": after_accept_metrics.loss,
                    "accept_acc_after": after_accept_metrics.acc,
                    "accept_margin_after": after_accept_metrics.margin,
                    "teacher_router_loss_after": self._safe_float(after_accept_teacher_metrics.teacher_router_loss),
                    "teacher_label_entropy_mean": self._safe_float(
                        after_accept_teacher_metrics.teacher_label_entropy_mean,
                        self._safe_float(before_accept_teacher_metrics.teacher_label_entropy_mean),
                    ),
                    "teacher_conf_mean": self._safe_float(
                        after_accept_teacher_metrics.teacher_conf_mean,
                        self._safe_float(before_accept_teacher_metrics.teacher_conf_mean),
                    ),
                    "teacher_conf_std": self._safe_float(
                        after_accept_teacher_metrics.teacher_conf_std,
                        self._safe_float(before_accept_teacher_metrics.teacher_conf_std),
                    ),
                    "teacher_valid_ratio": self._safe_float(
                        after_accept_teacher_metrics.teacher_valid_ratio,
                        self._safe_float(before_accept_teacher_metrics.teacher_valid_ratio),
                    ),
                    "router_teacher_acc_after": self._safe_float(after_accept_teacher_metrics.router_teacher_acc),
                    "teacher_proxy_mode": bool(
                        before_accept_teacher_metrics.teacher_proxy_mode
                        or after_accept_teacher_metrics.teacher_proxy_mode
                    ),
                    "counterfactual_success": bool(
                        before_accept_teacher_metrics.counterfactual_success
                        or after_accept_teacher_metrics.counterfactual_success
                    ),
                    "counterfactual_fallback_reason": (
                        str(after_accept_teacher_metrics.counterfactual_fallback_reason)
                        or str(before_accept_teacher_metrics.counterfactual_fallback_reason)
                    ),
                    "loss_delta_accept": debug["loss_delta"],
                    "margin_delta_accept": debug["margin_delta"],
                    "acc_delta_accept": debug["acc_delta"],
                    "router_entropy_delta_accept": debug["router_entropy_delta"],
                    "loss_improved": debug["loss_improved"],
                    "acc_improved": debug["acc_improved"],
                    "margin_improved": debug["margin_improved"],
                    "margin_not_worse": debug["margin_not_worse"],
                    "loss_not_worse": debug["loss_not_worse"],
                    "loss_evidence": debug["loss_evidence"],
                    "margin_evidence": debug["margin_evidence"],
                    "acc_evidence": debug["acc_evidence"],
                    "router_evidence": debug["router_evidence"],
                    "evidence_count": debug["evidence_count"],
                    "evidence_required": debug["evidence_required"],
                    "strong_accept": debug["strong_accept"],
                    "acc_accept": debug["acc_accept"],
                    "late_strict_accept": debug["late_strict_accept"],
                    "late_phase": debug["late_phase"],
                    "late_insufficient_improvement": debug["late_insufficient_improvement"],
                    "effective_min_delta": debug["effective_min_delta"],
                    "effective_margin_delta": debug["effective_margin_delta"],
                    "accept_rate_window": window_rate,
                    "accept_rate_window_size": min(
                        len(self.server_adapt_accept_history),
                        max(int(getattr(config, "server_adapt_accept_rate_window", 10)), 1),
                    ),
                    "router_load_entropy_after": self._safe_float(after_accept_metrics.router_load_entropy),
                    "router_aux_available": (
                        before_accept_metrics.router_load_entropy is not None
                        or after_accept_metrics.router_load_entropy is not None
                    ),
                    "near_best": debug["near_best"],
                    "near_best_reject": debug["near_best_reject"],
                    "cooldown_active": debug["cooldown_active"],
                    "cooldown_remaining": debug["cooldown_remaining"],
                    "cooldown_skip": debug["cooldown_skip"],
                    "rescue_adapt": debug["rescue_adapt"],
                    "adapt_delta_norm_raw": adapt_delta_norm_raw,
                    "adapt_delta_norm_ema": adapt_delta_norm_ema,
                    "adapt_param_delta_norm": adapt_param_delta_norm,
                    "ema_applied": bool(ema_applied),
                    "rollback_reason": rollback_reason,
                    "adapt_loss_after": after_accept_metrics.loss,
                    "adapt_margin_after": after_accept_metrics.margin,
                }
            )
        except Exception as exc:
            self._restore_selected_params(model, before_state)
            self._decay_server_adapt_ema_buffers(selected_names, config)
            base_record.update(
                {
                    "adapt_attempted": True,
                    "adapt_accepted": False,
                    "adapt_accept_reason": "adapt_exception",
                    "rollback_reason": "adapt_exception",
                    "error_info": str(exc),
                }
            )
        finally:
            if original_trainability is not None:
                self._restore_param_trainability(original_trainability)
            model.train(was_training)
        self._record_server_adapt_metrics(base_record)
        return model

    def evaluate_server_test(self):
        if self.base_aggregation_method == "fedjets" and self.fedjets_runtime is not None:
            metrics = self.fedjets_runtime.evaluate_server_test()
            return metrics["server_test_loss"], metrics["server_test_acc"]
        if self.base_aggregation_method == "pfedmoe" and self.pfedmoe_runtime is not None:
            metrics = self.pfedmoe_runtime.evaluate_server_test()
            return metrics["server_test_loss"], metrics["server_test_acc"]
        if self.base_aggregation_method == "hfedmoe" and self.hfedmoe_runtime is not None:
            metrics = self.hfedmoe_runtime.evaluate_server_test()
            return metrics["server_test_loss"], metrics["server_test_acc"]
        if len(self.server_test_loader.dataset) == 0:
            return float("inf"), 0.0

        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in self.server_test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, router_stats = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss + self.args.aux_loss_weight * router_stats["aux_loss"]

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        test_loss = running_loss / len(self.server_test_loader.dataset)
        test_acc = (running_corrects.double() / len(self.server_test_loader.dataset)).item()
        return test_loss, test_acc

    def aggregate(self, client_updates, round_idx):
        self.latest_vsmc_round_infos = []
        self.latest_ha_vsmc_etf_round_data = None
        method = self.base_aggregation_method
        if method == "fedavg_expert_only":
            self.aggregate_fedavg_expert_only(client_updates)
        elif method == "fedavg_shared_only":
            self.aggregate_fedavg_shared_only(client_updates)
        elif method == "beta_precision_fedtra":
            self.aggregate_beta_precision_fedtra(client_updates, round_idx)
        elif method == "vsmc_expert":
            self.aggregate_vsmc_expert(client_updates, round_idx)
        elif method == "ha_vsmc_etf":
            self.aggregate_ha_vsmc_etf(client_updates, round_idx)
        elif method == "joint_vsmc_expert":
            self.aggregate_joint_vsmc_expert(client_updates, round_idx)
        elif method == "temporal_vsmc_expert":
            self.aggregate_temporal_vsmc_expert(client_updates, round_idx)
        elif method == "temporal_latent_vsmc_expert":
            self.aggregate_temporal_latent_vsmc_expert(client_updates, round_idx)
        elif method == "learned_vsmc":
            self.aggregate_learned_vsmc(client_updates, round_idx)
        elif method == "learned_vsmc_moe_state_policy_k8" or bool(getattr(self.args, "learned_vsmc_moe_state_policy", False)):
            self.aggregate_moe_state_policy(client_updates, round_idx)
        elif method == "fedmoe_da":
            self.aggregate_fedmoe_da(client_updates, round_idx)
        else:
            self.aggregate_fedavg(client_updates)
        if self._server_adapt_enabled():
            self.server_side_adapt_with_rollback(
                model=self.model,
                server_val_loader=self.server_val_loader,
                round_id=round_idx,
                total_rounds=self.args.server_epochs,
                config=self.args,
            )

    def build_ha_vsmc_etf_config(self):
        return HAVSMCETFConfig(
            enabled=bool(self.args.ha_vsmc_enabled),
            num_particles=self.args.ha_vsmc_num_particles,
            state_level=self.args.ha_vsmc_state_level,
            expert_velocity_beta=self.args.ha_vsmc_expert_velocity_beta,
            second_moment_beta=self.args.ha_vsmc_second_moment_beta,
            client_reliability_beta=self.args.ha_vsmc_client_reliability_beta,
            usage_beta=self.args.ha_vsmc_usage_beta,
            temperature_init=self.args.ha_vsmc_temperature_init,
            temperature_min=self.args.ha_vsmc_temperature_min,
            temperature_max=self.args.ha_vsmc_temperature_max,
            lambda_max=self.args.ha_vsmc_lambda_max,
            lambda_warmup_rounds=self.args.ha_vsmc_lambda_warmup_rounds,
            anchor_mode=self.args.ha_vsmc_anchor_mode,
            pure_vsmc=self.args.ha_vsmc_pure_vsmc,
            score_transform=self.args.ha_vsmc_score_transform,
            score_clip_low=self.args.ha_vsmc_score_clip_low,
            score_clip_high=self.args.ha_vsmc_score_clip_high,
            temperature_scale=self.args.ha_vsmc_temperature_scale,
            prob_ema_rho=self.args.ha_vsmc_prob_ema_rho,
            reliability_floor_eps=self.args.ha_vsmc_reliability_floor_eps,
            reliability_floor_source=self.args.ha_vsmc_reliability_floor_source,
            top_k_clients=self.args.ha_vsmc_top_k_clients,
            adaptive_temperature=self.args.ha_vsmc_adaptive_temperature,
            entropy_target=self.args.ha_vsmc_entropy_target,
            entropy_target_min_max_weight=self.args.ha_vsmc_entropy_target_min_max_weight,
            entropy_target_max_max_weight=self.args.ha_vsmc_entropy_target_max_max_weight,
            save_client_weights=self.args.ha_vsmc_save_client_weights,
            anti_collapse_enabled=self.args.ha_vsmc_anti_collapse_enabled,
            anti_collapse_max_weight_thresh=self.args.ha_vsmc_anti_collapse_max_weight_thresh,
            anti_collapse_neff_thresh=self.args.ha_vsmc_anti_collapse_neff_thresh,
            anti_collapse_entropy_thresh=self.args.ha_vsmc_anti_collapse_entropy_thresh,
            anti_collapse_eps=self.args.ha_vsmc_anti_collapse_eps,
            anti_collapse_prior=self.args.ha_vsmc_anti_collapse_prior,
            coverage_bonus_beta=self.args.ha_vsmc_coverage_bonus_beta,
            client_coverage_scores=getattr(self, "ha_vsmc_client_coverage_scores", None),
            proxy_gain_gamma=self.args.ha_vsmc_proxy_gain_gamma,
            ess_threshold_ratio=self.args.ha_vsmc_ess_threshold_ratio,
            trust_update_lr=self.args.ha_vsmc_trust_update_lr,
            reliability_momentum_lr=self.args.ha_vsmc_reliability_momentum_lr,
            prediction_noise_std=self.args.ha_vsmc_prediction_noise_std,
            alpha_usage=self.args.ha_vsmc_alpha_usage,
            alpha_history_direction=self.args.ha_vsmc_alpha_history_direction,
            alpha_consensus=self.args.ha_vsmc_alpha_consensus,
            alpha_drift=self.args.ha_vsmc_alpha_drift,
            alpha_function_drift=self.args.ha_vsmc_alpha_function_drift,
            collapse_penalty=self.args.ha_vsmc_collapse_penalty,
            probe_batch_size=self.args.ha_vsmc_probe_batch_size,
            use_function_drift=self.args.ha_vsmc_use_function_drift,
            function_drift_metric=self.args.ha_vsmc_function_drift_metric,
            function_drift_interval=self.args.ha_vsmc_function_drift_interval,
            require_function_drift=self.args.ha_vsmc_require_function_drift,
            use_validation_loss=self.args.ha_vsmc_use_validation_loss,
            normalize_observation_components=self.args.ha_vsmc_normalize_observation_components,
            min_router_usage=self.args.ha_vsmc_min_router_usage,
            eps=self.args.ha_vsmc_eps,
            log_every_round=self.args.ha_vsmc_log_every_round,
            deterministic=self.args.ha_vsmc_deterministic,
            shared_weight_mode=self.args.fedavg_weight_mode,
            expert_weight_mode=self.args.expert_weight_mode,
            small_entropy_bonus=self.args.ha_vsmc_small_entropy_bonus,
            reliability_weighted_update=self.args.ha_vsmc_reliability_weighted_update,
            seed=self.args.seed,
        )

    def build_beta_precision_fedtra_config(self):
        return BetaPrecisionFedTRAConfig(
            num_particles=self.args.beta_precision_num_particles,
            kappa_u=self.args.beta_precision_kappa_u,
            kappa_c=self.args.beta_precision_kappa_c,
            omd_lr=self.args.beta_precision_omd_lr,
            gamma_min=self.args.beta_precision_gamma_min,
            transition_rho=self.args.beta_precision_transition_rho,
            transition_std=self.args.beta_precision_transition_std,
            initial_std=self.args.beta_precision_initial_std,
            mean_slope_u=self.args.beta_precision_mean_slope_u,
            mean_slope_c=self.args.beta_precision_mean_slope_c,
            mean_intercept_c=self.args.beta_precision_mean_intercept_c,
            eps=self.args.beta_precision_eps,
            seed=self.args.seed,
            log_dir=os.path.join(self.args.save_result, "logs"),
        )

    def build_vsmc_config(self):
        return VSMCExpertAggregationConfig(
            enabled=True,
            vsmc_mode=self.args.vsmc_mode,
            temperature=self.args.vsmc_temperature,
            sigma_f=self.args.vsmc_sigma_f,
            sigma_r=self.args.vsmc_sigma_r,
            obs_temp=self.args.vsmc_obs_temp,
            loss_beta=self.args.vsmc_loss_beta,
            route_beta=self.args.vsmc_route_beta,
            usage_beta=self.args.vsmc_usage_beta,
            sample_beta=self.args.vsmc_sample_beta,
            server_val_beta=self.args.vsmc_server_val_beta if self.args.vsmc_use_server_validation else 0.0,
            aggregate_delta=self.args.vsmc_aggregate_delta,
            expert_server_lr=self.args.vsmc_expert_server_lr,
            use_server_validation_g=self.args.vsmc_use_server_validation_g,
            server_val_batches=self.args.vsmc_server_val_batches,
            server_val_batch_size=self.args.vsmc_server_val_batch_size,
            server_val_every_round=self.args.vsmc_server_val_every_round,
            server_g_beta=self.args.vsmc_server_g_beta,
            server_g_clip=self.args.vsmc_server_g_clip,
            decouple_mu=self.args.vsmc_decouple_mu,
            base_alpha_source=self.args.vsmc_base_alpha_source,
            lambda_r=self.args.vsmc_lambda_r,
            r_bonus_clip=self.args.vsmc_r_bonus_clip,
            lambda_g=self.args.vsmc_lambda_g,
            use_local_g_if_no_server_val=self.args.vsmc_use_local_g_if_no_server_val,
            use_ema_f_smoothing=self.args.vsmc_use_ema_f_smoothing,
            expert_ema_rho=self.args.vsmc_expert_ema_rho,
            mix_with_fedavg=self.args.vsmc_mix_with_fedavg,
            alpha_max=self.args.vsmc_alpha_max,
            normalized_ess_min=self.args.vsmc_normalized_ess_min,
            fallback_to_fedavg_on_bad_alpha=self.args.vsmc_fallback_to_fedavg_on_bad_alpha,
            use_router_usage_calibration=self.args.vsmc_use_router_usage_calibration,
            router_bias_lr=self.args.vsmc_router_bias_lr,
            target_usage=self.args.vsmc_target_usage_override,
            replace_reward_beta=self.args.vsmc_replace_reward_beta,
            delta_reward_beta=self.args.vsmc_delta_reward_beta,
            stability_beta=self.args.vsmc_stability_beta,
            novelty_beta=self.args.vsmc_novelty_beta,
            credit_beta=self.args.vsmc_credit_beta,
            signal_temperature=self.args.vsmc_signal_temperature,
            reward_ema_rho=self.args.vsmc_reward_ema_rho,
            credit_momentum=self.args.vsmc_credit_momentum,
            uniform_mix=self.args.vsmc_uniform_mix,
            novelty_clip=self.args.vsmc_novelty_clip,
            stability_clip=self.args.vsmc_stability_clip,
            signal_reward_clip=self.args.vsmc_signal_reward_clip,
            clamp_log_weight=self.args.vsmc_clamp_log_weight,
            eps=self.args.vsmc_eps,
            normalize_loss=self.args.vsmc_normalize_loss,
            normalize_distance=self.args.vsmc_normalize_distance,
            use_resampling=self.args.vsmc_use_resampling,
            ess_threshold=self.args.vsmc_ess_threshold,
            resampling_mode=self.args.vsmc_resampling_mode,
            keep_particle_bank=self.args.vsmc_keep_particle_bank,
            missing_client_decay=self.args.vsmc_missing_client_decay,
            fallback_to_fedavg_on_nan=self.args.vsmc_fallback_to_fedavg_on_nan,
            fallback_on_low_ess=self.args.vsmc_fallback_on_low_ess,
            low_ess_threshold=self.args.vsmc_low_ess_threshold,
            fallback_when_all_route_zero=self.args.vsmc_fallback_when_all_route_zero,
            debug=self.args.vsmc_debug,
            log_dir=os.path.join(self.args.save_result, "logs"),
            log_csv_name="vsmc_aggregation_round_metrics.csv",
        )

    def build_full_vsmc_config(self):
        return FullVSMCAggregatorConfig(
            temperature=self.args.vsmc_temperature,
            loss_beta=self.args.vsmc_loss_beta,
            usage_beta=self.args.vsmc_usage_beta,
            drift_beta=self.args.vsmc_drift_beta,
            route_beta=self.args.vsmc_route_beta,
            target_usage=self.args.vsmc_target_usage,
            min_route_count=self.args.vsmc_min_route_count,
            ess_threshold=self.args.vsmc_ess_threshold,
            use_resampling=self.args.vsmc_use_resampling,
            resampling_noise_std=self.args.vsmc_resampling_noise_std,
            fallback_to_fedavg_on_nan=self.args.vsmc_fallback_to_fedavg_on_nan,
            fallback_to_fedavg_on_degenerate=self.args.vsmc_fallback_to_fedavg_on_degenerate,
            clamp_log_weight=self.args.vsmc_clamp_log_weight,
            degeneration_ess_threshold=self.args.vsmc_degeneration_ess_threshold,
            degeneration_temperature_scale=self.args.vsmc_degeneration_temperature_scale,
            latent_dim=self.args.latent_vsmc_dim,
            transition_rho=self.args.latent_transition_rho,
            process_noise=self.args.latent_process_noise,
            obs_noise_base=self.args.latent_obs_noise_base,
            loss_noise_beta=self.args.latent_loss_noise_beta,
            route_noise_beta=self.args.latent_route_noise_beta,
            sample_noise_beta=self.args.latent_sample_noise_beta,
            proposal_mode=self.args.full_vsmc_proposal_mode,
            proposal_blend=self.args.full_vsmc_proposal_blend,
            proposal_var_scale=self.args.full_vsmc_proposal_var_scale,
            use_proposal_particles=self.args.full_vsmc_use_proposal_particles,
            num_particles=self.args.full_vsmc_num_particles,
            responsibility_temperature=self.args.full_vsmc_responsibility_temperature,
            particle_temperature=self.args.full_vsmc_particle_temperature,
            metric_ema_beta=self.args.full_vsmc_metric_ema_beta,
            responsibility_floor=self.args.full_vsmc_responsibility_floor,
            prev_weight_decay=self.args.full_vsmc_prev_weight_decay,
            warmup_rounds=self.args.latent_warmup_rounds,
            projection_seed=self.args.latent_projection_seed,
            fedavg_mix=self.args.latent_fedavg_mix,
            min_weight=self.args.latent_min_weight,
            debug=self.args.vsmc_debug,
        )

    def build_vsmc_aggregator(self):
        if self.args.vsmc_variant == "full_vsmc":
            return FullVSMCExpertAggregator(self.build_full_vsmc_config(), logger=self.logger)
        return VSMCExpertAggregator(
            config=self.build_vsmc_config(),
            num_experts=self.args.switch_num_experts,
            logger=self.logger,
        )

    def build_joint_vsmc_config(self):
        return JointVSMCAggregationConfig(
            enabled=True,
            temperature=self.args.vsmc_temperature,
            sigma_f=self.args.vsmc_sigma_f,
            sigma_r=self.args.vsmc_sigma_r,
            obs_temp=self.args.vsmc_obs_temp,
            loss_beta=self.args.vsmc_loss_beta,
            route_beta=self.args.vsmc_route_beta,
            usage_beta=self.args.vsmc_usage_beta,
            sample_beta=self.args.vsmc_sample_beta,
            server_val_beta=self.args.vsmc_server_val_beta if self.args.vsmc_use_server_validation else 0.0,
            clamp_log_weight=self.args.vsmc_clamp_log_weight,
            eps=self.args.vsmc_eps,
            normalize_loss=self.args.vsmc_normalize_loss,
            normalize_distance=self.args.vsmc_normalize_distance,
            use_resampling=self.args.vsmc_use_resampling,
            ess_threshold=self.args.vsmc_ess_threshold,
            resampling_mode=self.args.vsmc_resampling_mode,
            keep_particle_bank=self.args.vsmc_keep_particle_bank,
            missing_client_decay=self.args.vsmc_missing_client_decay,
            fallback_to_fedavg_on_nan=self.args.vsmc_fallback_to_fedavg_on_nan,
            fallback_on_low_ess=self.args.vsmc_fallback_on_low_ess,
            low_ess_threshold=self.args.vsmc_low_ess_threshold,
            fallback_when_all_route_zero=self.args.vsmc_fallback_when_all_route_zero,
            debug=self.args.vsmc_debug,
            log_dir=os.path.join(self.args.save_result, "logs"),
            log_csv_name="vsmc_aggregation_round_metrics.csv",
            expert_transition_beta=self.args.joint_expert_transition_beta,
            local_obs_beta=self.args.joint_local_obs_beta,
            global_obs_beta=self.args.joint_global_obs_beta,
            proposal_beta=self.args.joint_proposal_beta,
            router_consistency_beta=self.args.joint_router_consistency_beta,
            router_validation_beta=self.args.joint_router_validation_beta,
            router_entropy_beta=self.args.joint_router_entropy_beta,
            router_aux_beta=self.args.joint_router_aux_beta,
            router_load_balance_beta=self.args.joint_router_load_balance_beta,
            proposal_ema_decay=self.args.joint_proposal_ema_decay,
            use_client_specific_proposal=self.args.joint_use_client_specific_proposal,
            use_router_validation=self.args.joint_use_router_validation,
            route_weight_temperature=self.args.joint_route_weight_temperature,
        )

    def build_moe_state_policy_config(self):
        MoEStatePolicyConfig, _ = self._load_moe_state_policy_components()
        modes = tuple(
            token.strip()
            for token in str(getattr(self.args, "moe_state_aggregation_modes", "") or "").split(",")
            if token.strip()
        )
        if not modes:
            modes = ("fedavg", "shrink075", "shrink05", "momentum", "clipped_mean", "latent_residual")
        return MoEStatePolicyConfig(
            enabled=bool(getattr(self.args, "learned_vsmc_moe_state_policy", False) or self.args.aggregation_method == "learned_vsmc_moe_state_policy_k8"),
            num_particles=self.args.moe_state_num_particles,
            elite_top_k=self.args.moe_state_elite_top_k,
            ess_threshold=self.args.moe_state_ess_threshold,
            jitter_std=self.args.moe_state_jitter_std,
            logq_coef=self.args.moe_state_logq_coef,
            reward_temperature=self.args.moe_state_reward_temperature,
            transition_coef=self.args.moe_state_transition_coef,
            proposal_lr=self.args.moe_state_proposal_lr,
            proposal_hidden_dim=self.args.moe_state_proposal_hidden_dim,
            residual_latent_dim=self.args.moe_state_residual_latent_dim,
            max_residual_ratio=self.args.moe_state_max_residual_ratio,
            aggregation_modes=modes,
            momentum_beta=self.args.moe_state_momentum_beta,
            clip_norm_quantile=self.args.moe_state_clip_norm_quantile,
            block_scale_min=self.args.moe_state_block_scale_min,
            block_scale_max=self.args.moe_state_block_scale_max,
            val_loss_coef=self.args.moe_state_val_loss_coef,
            val_loss_improve_coef=self.args.moe_state_val_loss_improve_coef,
            val_acc_improve_coef=self.args.moe_state_val_acc_improve_coef,
            router_balance_coef=self.args.moe_state_router_balance_coef,
            expert_conflict_coef=self.args.moe_state_expert_conflict_coef,
            update_norm_coef=self.args.moe_state_update_norm_coef,
            class_balance_coef=self.args.moe_state_class_balance_coef,
            val_batches=self.args.moe_state_val_batches,
            full_eval_interval=self.args.moe_state_full_eval_interval,
            train_proposal=self.args.moe_state_train_proposal,
            detach_resampling=self.args.moe_state_detach_resampling,
            log_diagnostics=self.args.moe_state_log_diagnostics,
            selector_mode=self.args.moe_state_selector_mode,
            total_rounds=self.args.server_epochs,
            grad_clip=1.0,
            log_dir=os.path.join(self.args.save_result, "logs"),
        )

    def build_learned_vsmc_config(self):
        LearnedVSMCAggregationConfig, _ = self._load_learned_vsmc_components()
        return LearnedVSMCAggregationConfig(
            num_particles=self.args.learned_vsmc_num_particles,
            proposal_lr=self.args.learned_vsmc_proposal_lr,
            proposal_train_steps=self.args.learned_vsmc_train_steps,
            sigma_f=self.args.learned_vsmc_sigma_f,
            server_g_beta=self.args.learned_vsmc_server_g_beta,
            use_reward_g=self.args.learned_vsmc_use_reward_g,
            val_batches=self.args.learned_vsmc_val_batches,
            grad_clip=self.args.learned_vsmc_grad_clip,
            use_reinforce_if_not_differentiable=self.args.learned_vsmc_use_reinforce_if_not_differentiable,
            require_differentiable_g=self.args.learned_vsmc_require_differentiable_g,
            reward_type=self.args.learned_vsmc_reward_type,
            confidence_reward_weight=self.args.learned_vsmc_confidence_reward_weight,
            loss_reward_weight=self.args.learned_vsmc_loss_reward_weight,
            total_rounds=self.args.server_epochs,
            latent_dim=self.args.learned_vsmc_latent_dim,
            block_latent_dim=self.args.learned_vsmc_block_latent_dim,
            use_block_joint=self.args.learned_vsmc_use_block_joint,
            latent_prior_sigma=self.args.learned_vsmc_latent_prior_sigma,
            latent_mean_scale=self.args.learned_vsmc_latent_mean_scale,
            latent_residual_max=self.args.learned_vsmc_latent_residual_max,
            final_residual_max=self.args.learned_vsmc_final_residual_max,
            anneal_start_ratio=self.args.learned_vsmc_anneal_start_ratio,
            include_zero_particle=self.args.learned_vsmc_include_zero_particle,
            mean_a_scale=self.args.learned_vsmc_mean_a_scale,
            mean_b_scale=self.args.learned_vsmc_mean_b_scale,
            init_logvar=self.args.learned_vsmc_init_logvar,
            logvar_scale=self.args.learned_vsmc_logvar_scale,
            a_min=self.args.learned_vsmc_a_min,
            a_max=self.args.learned_vsmc_a_max,
            b_min=self.args.learned_vsmc_b_min,
            b_max=self.args.learned_vsmc_b_max,
            reward_clip=self.args.learned_vsmc_reward_clip,
            large_loss=self.args.learned_vsmc_large_loss,
            feature_clip=self.args.learned_vsmc_feature_clip,
            logq_coef=self.args.learned_vsmc_logq_coef,
            normalize_reward=self.args.learned_vsmc_normalize_reward,
            reward_norm_eps=self.args.learned_vsmc_reward_norm_eps,
            include_fedavg_particle=self.args.learned_vsmc_include_fedavg_particle,
            mix_fedavg_candidate=self.args.learned_vsmc_mix_fedavg_candidate,
            final_mix_fedavg_candidate=self.args.learned_vsmc_final_mix_fedavg_candidate,
            mix_uniform=self.args.learned_vsmc_mix_uniform,
            alpha_max=self.args.learned_vsmc_alpha_max,
            min_normalized_ess=self.args.learned_vsmc_min_normalized_ess,
            mode=self.args.learned_vsmc_mode,
            debug=self.args.vsmc_debug,
        )

    def build_temporal_vsmc_config(self):
        return TemporalVSMCAggregatorConfig(
            temperature=self.args.vsmc_temperature,
            loss_beta=self.args.vsmc_loss_beta,
            usage_beta=self.args.vsmc_usage_beta,
            drift_beta=self.args.vsmc_drift_beta,
            sample_beta=self.args.vsmc_sample_beta,
            route_beta=self.args.vsmc_route_beta,
            target_usage=self.args.vsmc_target_usage,
            min_route_count=self.args.vsmc_min_route_count,
            ess_threshold=self.args.vsmc_ess_threshold,
            use_resampling=self.args.vsmc_use_resampling,
            resampling_noise_std=self.args.vsmc_resampling_noise_std,
            fallback_to_fedavg_on_nan=self.args.vsmc_fallback_to_fedavg_on_nan,
            clamp_log_weight=self.args.vsmc_clamp_log_weight,
            debug=self.args.temporal_debug or self.args.vsmc_debug,
            momentum=self.args.temporal_momentum,
            align_beta=self.args.temporal_align_beta,
            jump_beta=self.args.temporal_jump_beta,
            warmup_rounds=self.args.temporal_warmup_rounds,
            score_mode=self.args.temporal_score_mode,
            history_size=self.args.temporal_history_size,
        )

    def build_temporal_latent_config(self):
        return TemporalLatentAggregatorConfig(
            temperature=self.args.vsmc_temperature,
            loss_beta=self.args.vsmc_loss_beta,
            usage_beta=self.args.vsmc_usage_beta,
            drift_beta=self.args.vsmc_drift_beta,
            route_beta=self.args.vsmc_route_beta,
            target_usage=self.args.vsmc_target_usage,
            min_route_count=self.args.vsmc_min_route_count,
            fallback_to_fedavg_on_nan=self.args.vsmc_fallback_to_fedavg_on_nan,
            clamp_log_weight=self.args.vsmc_clamp_log_weight,
            latent_dim=self.args.latent_vsmc_dim,
            transition_rho=self.args.latent_transition_rho,
            process_noise=self.args.latent_process_noise,
            obs_noise_base=self.args.latent_obs_noise_base,
            loss_noise_beta=self.args.latent_loss_noise_beta,
            route_noise_beta=self.args.latent_route_noise_beta,
            sample_noise_beta=self.args.latent_sample_noise_beta,
            warmup_rounds=self.args.latent_warmup_rounds,
            projection_seed=self.args.latent_projection_seed,
            fedavg_mix=self.args.latent_fedavg_mix,
            min_weight=self.args.latent_min_weight,
            debug=self.args.latent_debug,
        )

    def _compute_fedavg_state(self, client_updates):
        shared_mode = str(getattr(self.args, "fedavg_weight_mode", "uniform")).lower()
        sample_count_weights = self._compute_client_weights(client_updates, mode=shared_mode)
        expert_mode = str(getattr(self.args, "expert_weight_mode", "uniform")).lower()
        expert_weights = self._compute_client_weights(client_updates, mode=expert_mode)

        global_state = collections.OrderedDict()
        for client_idx, (state_dict, _, _) in enumerate(client_updates):
            shared_weight = float(sample_count_weights[client_idx])
            expert_weight = float(expert_weights[client_idx])
            for name, tensor in state_dict.items():
                tensor = tensor.to(self.device)
                weight = expert_weight if self._is_expert_param_name(name) else shared_weight
                if name not in global_state:
                    global_state[name] = tensor * weight
                else:
                    global_state[name] += tensor * weight
        return global_state

    def aggregate_fedavg(self, client_updates):
        global_state = self._compute_fedavg_state(client_updates)
        self.model.load_state_dict(global_state)
        self.logger.info(
            f"--fedavg --shared_weight_mode:{getattr(self.args, 'fedavg_weight_mode', 'uniform')} "
            f"--expert_weight_mode:{getattr(self.args, 'expert_weight_mode', 'uniform')}"
        )

    def aggregate_fedavg_expert_only(self, client_updates):
        expert_mode = str(getattr(self.args, "expert_weight_mode", "uniform")).lower()
        expert_weights = self._compute_client_weights(client_updates, mode=expert_mode)

        global_state = collections.OrderedDict(
            (name, tensor.detach().clone().to(self.device))
            for name, tensor in self.model.state_dict().items()
        )
        updated_expert_names = set()
        expert_state = collections.OrderedDict()

        for client_idx, (state_dict, _, _) in enumerate(client_updates):
            expert_weight = float(expert_weights[client_idx])
            for name, tensor in state_dict.items():
                if not self._is_expert_param_name(name):
                    continue
                updated_expert_names.add(name)
                weighted_tensor = tensor.to(self.device) * expert_weight
                if name not in expert_state:
                    expert_state[name] = weighted_tensor
                else:
                    expert_state[name] += weighted_tensor

        for name, tensor in expert_state.items():
            global_state[name] = tensor

        self.model.load_state_dict(global_state)
        self.logger.info(
            f"--fedavg_expert_only --shared_router:server_previous "
            f"--experts:{getattr(self.args, 'expert_weight_mode', 'uniform')} "
            f"--updated_expert_tensors:{len(updated_expert_names)}"
        )

    def aggregate_fedavg_shared_only(self, client_updates):
        shared_mode = str(getattr(self.args, "fedavg_weight_mode", "uniform")).lower()
        shared_weights = self._compute_client_weights(client_updates, mode=shared_mode)

        global_state = collections.OrderedDict(
            (name, tensor.detach().clone().to(self.device))
            for name, tensor in self.model.state_dict().items()
        )
        shared_state = collections.OrderedDict()

        for client_idx, (state_dict, _, _) in enumerate(client_updates):
            shared_weight = float(shared_weights[client_idx])
            for name, tensor in state_dict.items():
                if self._is_expert_param_name(name):
                    continue
                weighted_tensor = tensor.to(self.device) * shared_weight
                if name not in shared_state:
                    shared_state[name] = weighted_tensor
                else:
                    shared_state[name] += weighted_tensor

        for name, tensor in shared_state.items():
            global_state[name] = tensor

        self.model.load_state_dict(global_state)
        self.logger.info(
            f"--fedavg_shared_only --shared_router:{getattr(self.args, 'fedavg_weight_mode', 'uniform')}_fedavg "
            f"--experts:server_previous --updated_shared_tensors:{len(shared_state)}"
        )

    def aggregate_fedmoe_da(self, client_updates, round_idx):
        if self.fedmoe_da_aggregator is None:
            self.fedmoe_da_aggregator = FedMoEDAExpertAggregator(
                resolve_fedmoe_da_config(self.args),
                logger=self.logger,
            )

        global_state = self.load_server_state()
        aggregated_state, round_data = self.fedmoe_da_aggregator.aggregate_model_state(
            global_state=global_state,
            client_updates=client_updates,
            device=self.device,
            round_idx=round_idx,
        )
        self.model.load_state_dict(aggregated_state)
        self.logger.info(
            f"--fedmoe_da --round:{round_idx} --num_expert_groups:{int(round_data.get('num_expert_groups', 0))} "
            f"--mean_pairwise_similarity:{float(round_data.get('mean_pairwise_similarity', 0.0)):.4f} "
            f"--mean_selected_similarity:{float(round_data.get('mean_selected_similarity', 0.0)):.4f} "
            f"--top_p:{int(round_data.get('top_p', self.args.fedmoe_da_top_p))}"
        )
        for info in round_data.get("per_expert_info", []):
            self.logger.info(
                f"--fedmoe_da_group --scope:{info['expert_scope']} --expert_id:{info['expert_id']} "
                f"--active_clients:{info.get('num_active_clients', 0)} "
                f"--mean_pairwise_similarity:{info.get('mean_pairwise_similarity', 0.0):.4f} "
                f"--mean_selected_similarity:{info.get('mean_selected_similarity', 0.0):.4f} "
                f"--mean_neighbor_count:{info.get('mean_neighbor_count', 0.0):.2f}"
            )

    def aggregate_ha_vsmc_etf(self, client_updates, round_idx):
        if self.ha_vsmc_etf_aggregator is None:
            self.ha_vsmc_etf_aggregator = HAVSMCETFExpertAggregator(
                self.build_ha_vsmc_etf_config(),
                logger=self.logger,
            )

        global_state = self.load_server_state()
        client_updates = self.attach_ha_vsmc_proxy_gains(client_updates, global_state)
        aggregated_state, round_data = self.ha_vsmc_etf_aggregator.aggregate_model_state(
            global_model=self.model,
            global_state=global_state,
            client_updates=client_updates,
            device=self.device,
            round_idx=round_idx,
            probe_batch=self._get_ha_vsmc_probe_batch(),
        )
        aggregated_state = collections.OrderedDict(
            (name, tensor.to(self.device) if torch.is_tensor(tensor) else tensor)
            for name, tensor in aggregated_state.items()
        )
        self.latest_ha_vsmc_etf_round_data = round_data
        self.model.load_state_dict(aggregated_state)
        expert_infos = round_data.get("expert_summaries", [])
        self.logger.info(
            f"--ha_vsmc_etf --round:{round_idx} --num_expert_groups:{len(expert_infos)} "
            f"--mean_ess_ratio:{float(round_data.get('mean_ess_ratio', 0.0)):.4f} "
            f"--fallback_groups:{int(round_data.get('fallback_count', 0))} "
            f"--pure_vsmc:{bool(round_data.get('pure_vsmc', False))} "
            f"--anchor_mode:{str(round_data.get('anchor_mode', 'fedavg'))} "
            f"--fedavg_blend_enabled:{bool(round_data.get('fedavg_blend_enabled', True))} "
            f"--expert_output:{str(round_data.get('expert_output', 'unknown'))}"
        )

    def aggregate_beta_precision_fedtra(self, client_updates, round_idx):
        if self.beta_precision_fedtra_aggregator is None:
            self.beta_precision_fedtra_aggregator = BetaPrecisionFedTRAggregator(
                self.build_beta_precision_fedtra_config(), logger=self.logger
            )
        global_state = self.load_server_state()
        aggregated_state, round_info = self.beta_precision_fedtra_aggregator.aggregate_model_state(
            global_state=global_state,
            client_updates=client_updates,
            num_experts=self.args.switch_num_experts,
            device=self.device,
        )
        self.model.load_state_dict(aggregated_state)
        self.logger.info(
            f"--beta_precision_fedtra --round:{round_idx} "
            f"--gamma_u:{round_info['gamma_u_after']:.6f} "
            f"--gamma_c:{round_info['gamma_c_after']:.6f} "
            f"--predictive_nll:{round_info['predictive_nll']:.6f} "
            f"--surrogate_nll:{round_info['surrogate_nll']:.6f} "
            f"--pairs:{round_info['num_pairs']}"
        )

    def aggregate_vsmc_expert(self, client_updates, round_idx):
        if self.vsmc_aggregator is None:
            self.vsmc_aggregator = self.build_vsmc_aggregator()

        global_state = self.load_server_state()
        if self.args.vsmc_mode in {"strong_signal_joint", "block_particle_gain_filter"}:
            if max(int(self.args.vsmc_server_val_every_round), 1) > 0 and round_idx % max(int(self.args.vsmc_server_val_every_round), 1) == 0:
                client_updates = self.attach_server_validation_strong_signals(client_updates, global_state)
        elif self.args.vsmc_mode == "server_validated_delta_decoupled":
            if (
                self.args.vsmc_use_server_validation_g
                and max(int(self.args.vsmc_server_val_every_round), 1) > 0
                and round_idx % max(int(self.args.vsmc_server_val_every_round), 1) == 0
            ):
                client_updates = self.attach_server_validation_rewards(client_updates, global_state)
        elif self.args.vsmc_use_server_validation and self.args.vsmc_variant != "full_vsmc":
            client_updates = self.attach_server_validation_scores(client_updates, global_state)
        if self.args.vsmc_variant == "full_vsmc":
            aggregated_state, round_infos = self.vsmc_aggregator.aggregate_model_state(
                global_state=global_state,
                client_updates=client_updates,
                num_experts=self.args.switch_num_experts,
                device=self.device,
            )
            full_info = {
                "per_expert_info": round_infos,
                "mean_normalized_ess": (
                    sum(float(info.get("normalized_ess", 0.0)) for info in round_infos) / max(len(round_infos), 1)
                ),
                "mean_logZ_increment": (
                    sum(float(info.get("logZ_increment", 0.0)) for info in round_infos) / max(len(round_infos), 1)
                ),
                "fallback_expert_count": sum(1 for info in round_infos if info.get("fallback_used")),
            }
        else:
            aggregated_state, full_info = self.vsmc_aggregator.aggregate_model(
                previous_global_state_dict=global_state,
                client_state_dicts=[state_dict for state_dict, _, _ in client_updates],
                client_metrics=[metrics for _, _, metrics in client_updates],
                sample_counts=[num_samples for _, num_samples, _ in client_updates],
                num_experts=self.args.switch_num_experts,
            )
            aggregated_state = collections.OrderedDict(
                (name, tensor.to(self.device) if torch.is_tensor(tensor) else tensor)
                for name, tensor in aggregated_state.items()
            )
            round_infos = full_info["per_expert_info"]
        self.latest_vsmc_round_infos = round_infos
        self.model.load_state_dict(aggregated_state)
        if self.args.vsmc_mode in {"server_validated_delta_decoupled", "strong_signal_joint"}:
            self.apply_router_usage_calibration()

        self.logger.info(
            f"--vsmc_expert --round:{round_idx} --num_expert_groups:{len(round_infos)} "
            f"--mean_normalized_ess:{full_info['mean_normalized_ess']:.4f} "
            f"--mean_logZ_increment:{full_info['mean_logZ_increment']:.4f} "
            f"--fallback_groups:{full_info['fallback_expert_count']}"
        )
        for info in round_infos:
            weights = info.get("weights", [])
            self.logger.info(
                f"--vsmc_expert_group --scope:{info['expert_scope']} --expert_id:{info['expert_id']} "
                f"--num_particles:{info.get('num_particles', 0)} "
                f"--ess:{info.get('ess', 0.0):.4f} --normalized_ess:{info.get('normalized_ess', 0.0):.4f} "
                f"--min_alpha:{min(weights) if weights else 0.0:.4f} "
                f"--max_alpha:{max(weights) if weights else 0.0:.4f} "
                f"--log_f_mean:{info.get('log_f_mean', 0.0):.4f} "
                f"--log_g_mean:{info.get('log_g_mean', 0.0):.4f} "
                f"--log_r_mean:{info.get('log_r_mean', 0.0):.4f} "
                f"--server_val_loss_mean:{info.get('server_val_loss_mean', 0.0):.4f} "
                f"--server_val_reward_mean:{info.get('server_val_reward_mean', 0.0):.4f} "
                f"--delta_reward_mean:{info.get('delta_reward_mean', 0.0):.4f} "
                f"--credit_mean:{info.get('credit_mean', 0.0):.4f} "
                f"--novelty_mean:{info.get('novelty_mean', 0.0):.4f} "
                f"--stability_mean:{info.get('stability_mean', 0.0):.4f} "
                f"--alpha_g_server_ess:{info.get('alpha_g_server_ess', 0.0):.4f} "
                f"--alpha_r_ess:{info.get('alpha_r_ess', 0.0):.4f} "
                f"--alpha_final_ess:{info.get('alpha_final_ess', 0.0):.4f} "
                f"--delta_norm_mean:{info.get('delta_norm_mean', 0.0):.4f} "
                f"--fedavg_delta_norm:{info.get('fedavg_delta_norm', 0.0):.4f} "
                f"--logZ_increment:{info.get('logZ_increment', 0.0):.4f} "
                f"--logZ_total:{info.get('logZ_total', 0.0):.4f} "
                f"--fallback:{bool(info.get('fallback_used'))} "
                f"--resampled:{bool(info.get('resampled'))}"
            )

    def _iter_server_validation_batches(self, max_batches):
        for batch_idx, batch in enumerate(self.server_val_loader):
            if batch_idx >= max_batches:
                break
            yield batch

    def evaluate_state_on_server_validation(self, state_dict, max_batches):
        model = self.build_model().to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        running_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in self._iter_server_validation_batches(max_batches):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, router_stats = model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss + self.args.aux_loss_weight * router_stats["aux_loss"]
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        if total_samples == 0:
            return float("inf")
        return running_loss / total_samples

    def evaluate_state_on_loader(self, state_dict, loader):
        if loader is None:
            return float("inf"), 0.0
        model = self.build_model().to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, router_stats = model(inputs)
                loss = self.criterion(outputs, labels) + self.args.aux_loss_weight * router_stats["aux_loss"]
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(outputs.argmax(dim=1) == labels).item()
                total_samples += inputs.size(0)
        if total_samples <= 0:
            return float("inf"), 0.0
        return running_loss / total_samples, running_corrects / float(total_samples)

    def attach_ha_vsmc_proxy_gains(self, client_updates, global_state):
        if self.ha_vsmc_proxy_loader is None or float(getattr(self.args, "ha_vsmc_proxy_gain_gamma", 0.0)) <= 0.0:
            return client_updates
        base_loss, base_acc = self.evaluate_state_on_loader(global_state, self.ha_vsmc_proxy_loader)
        updated = []
        for state_dict, num_samples, metrics in client_updates:
            metrics = dict(metrics or {})
            client_loss, client_acc = self.evaluate_state_on_loader(state_dict, self.ha_vsmc_proxy_loader)
            metrics["proxy_gain"] = float(base_loss - client_loss)
            metrics["proxy_val_loss_delta"] = float(base_loss - client_loss)
            metrics["proxy_val_acc_delta"] = float(client_acc - base_acc)
            updated.append((state_dict, num_samples, metrics))
        return updated

    def attach_server_validation_scores(self, client_updates, global_state, include_router=False):
        num_batches = max(int(self.args.vsmc_server_val_batches), 1)
        base_state, global_expert_groups = split_state_dict_by_expert(
            global_state,
            num_experts=self.args.switch_num_experts,
        )
        _, _, global_router_groups = split_model_state_by_expert_and_router(
            global_state,
            num_experts=self.args.switch_num_experts,
        )
        client_expert_groups = [
            split_state_dict_by_expert(state_dict, num_experts=self.args.switch_num_experts)[1]
            for state_dict, _, _ in client_updates
        ]
        client_router_groups = [
            split_model_state_by_expert_and_router(state_dict, num_experts=self.args.switch_num_experts)[2]
            for state_dict, _, _ in client_updates
        ]
        expert_scopes = sorted(global_expert_groups.keys())
        updated_client_updates = []
        for state_dict, num_samples, metrics in client_updates:
            metrics = dict(metrics or {})
            metrics["server_val_expert_losses"] = {}
            metrics["server_val_router_losses"] = {}
            updated_client_updates.append((state_dict, num_samples, metrics))

        for expert_scope in expert_scopes:
            previous_group = global_expert_groups.get(expert_scope)
            if previous_group is None:
                continue
            previous_expert_state = previous_group["state_dict"]
            for client_idx, (state_dict, _, metrics) in enumerate(updated_client_updates):
                client_group = client_expert_groups[client_idx]
                if expert_scope not in client_group:
                    continue
                candidate_expert_state = client_group[expert_scope]["state_dict"]
                merged_expert_groups = dict(global_expert_groups)
                merged_expert_groups[expert_scope] = {
                    "expert_id": previous_group["expert_id"],
                    "expert_scope": expert_scope,
                    "state_dict": _merge_with_reference_state(previous_expert_state, candidate_expert_state),
                }
                candidate_state = _merge_with_reference_state(
                    global_state,
                    merge_expert_states_into_global_state(base_state, merged_expert_groups),
                    device=self.device,
                )
                metrics["server_val_expert_losses"][expert_scope] = self.evaluate_state_on_server_validation(
                    candidate_state,
                    max_batches=num_batches,
                )
        if include_router:
            for router_scope, router_group in global_router_groups.items():
                for client_idx, (_, _, metrics) in enumerate(updated_client_updates):
                    client_router_map = client_router_groups[client_idx]
                    if router_scope not in client_router_map:
                        continue
                    candidate_router_state = client_router_map[router_scope]["state_dict"]
                    _, expert_groups_for_merge, router_groups_for_merge = split_model_state_by_expert_and_router(
                        global_state,
                        num_experts=self.args.switch_num_experts,
                    )
                    router_groups_for_merge = dict(router_groups_for_merge)
                    router_groups_for_merge[router_scope] = {
                        "router_scope": router_scope,
                        "state_dict": _merge_with_reference_state(router_group["state_dict"], candidate_router_state),
                    }
                    candidate_state = _merge_with_reference_state(
                        global_state,
                        merge_expert_states_into_global_state(
                            merge_expert_states_into_global_state(base_state, expert_groups_for_merge),
                            router_groups_for_merge,
                        ),
                        device=self.device,
                    )
                    metrics["server_val_router_losses"][router_scope] = self.evaluate_state_on_server_validation(
                        candidate_state,
                        max_batches=num_batches,
                    )
        return updated_client_updates

    def attach_server_validation_rewards(self, client_updates, global_state):
        num_batches = max(int(self.args.vsmc_server_val_batches), 1)
        base_state, global_expert_groups = split_state_dict_by_expert(
            global_state,
            num_experts=self.args.switch_num_experts,
        )
        del base_state
        client_expert_groups = [
            split_state_dict_by_expert(state_dict, num_experts=self.args.switch_num_experts)[1]
            for state_dict, _, _ in client_updates
        ]
        updated_client_updates = []
        for state_dict, num_samples, metrics in client_updates:
            metrics = dict(metrics or {})
            metrics.setdefault("server_val_expert_rewards", {})
            metrics.setdefault("server_val_base_losses", {})
            metrics.setdefault("server_val_expert_losses", {})
            updated_client_updates.append((state_dict, num_samples, metrics))

        for expert_scope, previous_group in global_expert_groups.items():
            current_states = []
            valid_client_indices = []
            for client_idx, client_group_map in enumerate(client_expert_groups):
                if expert_scope not in client_group_map:
                    continue
                valid_client_indices.append(client_idx)
                current_states.append(client_group_map[expert_scope]["state_dict"])
            if not current_states:
                continue
            log_g_server, reward_info = compute_server_validation_g_for_expert(
                global_model=self.model,
                previous_global_state_dict=global_state,
                expert_id=int(previous_group["expert_id"]),
                client_expert_states=current_states,
                val_loader=self._iter_server_validation_batches(num_batches),
                loss_fn=self.criterion,
                device=self.device,
                config=self.build_vsmc_config(),
                expert_scope=expert_scope,
                aux_loss_weight=self.args.aux_loss_weight,
            )
            rewards = reward_info.get("rewards", [])
            base_loss = reward_info.get("base_loss", float("nan"))
            replaced_losses = reward_info.get("replaced_losses", [])
            for local_idx, client_idx in enumerate(valid_client_indices):
                metrics = updated_client_updates[client_idx][2]
                metrics["server_val_expert_rewards"][expert_scope] = float(rewards[local_idx]) if local_idx < len(rewards) else 0.0
                metrics["server_val_base_losses"][expert_scope] = float(base_loss)
                metrics["server_val_expert_losses"][expert_scope] = (
                    float(replaced_losses[local_idx]) if local_idx < len(replaced_losses) else float("nan")
                )
                metrics.setdefault("server_val_expert_logits", {})[expert_scope] = float(log_g_server[local_idx].item())
        return updated_client_updates

    def attach_server_validation_strong_signals(self, client_updates, global_state):
        num_batches = max(int(self.args.vsmc_server_val_batches), 1)
        _, global_expert_groups = split_state_dict_by_expert(
            global_state,
            num_experts=self.args.switch_num_experts,
        )
        client_expert_groups = [
            split_state_dict_by_expert(state_dict, num_experts=self.args.switch_num_experts)[1]
            for state_dict, _, _ in client_updates
        ]
        updated_client_updates = []
        for client_idx, (state_dict, num_samples, metrics) in enumerate(client_updates):
            metrics = dict(metrics or {})
            metrics["_client_id"] = self.clients_id_list[client_idx] if client_idx < len(self.clients_id_list) else client_idx
            metrics.setdefault("server_val_expert_rewards", {})
            metrics.setdefault("server_val_delta_rewards", {})
            metrics.setdefault("server_val_best_delta_steps", {})
            metrics.setdefault("server_val_base_losses", {})
            metrics.setdefault("server_val_expert_losses", {})
            metrics.setdefault("server_val_candidate_signatures", {})
            metrics.setdefault("server_val_base_signatures", {})
            updated_client_updates.append((state_dict, num_samples, metrics))

        use_block_mode = self.args.vsmc_mode == "block_particle_gain_filter"
        if use_block_mode:
            grouped_global = group_expert_groups_by_block(global_expert_groups)
            grouped_clients = [group_expert_groups_by_block(group_map) for group_map in client_expert_groups]
            loop_items = grouped_global.items()
        else:
            grouped_global = global_expert_groups
            grouped_clients = client_expert_groups
            loop_items = grouped_global.items()

        for scope_key, previous_group in loop_items:
            current_states = []
            valid_client_indices = []
            for client_idx, client_group_map in enumerate(grouped_clients):
                if scope_key not in client_group_map:
                    continue
                valid_client_indices.append(client_idx)
                current_states.append(client_group_map[scope_key]["state_dict"])
            if not current_states:
                continue

            reward_info = compute_server_validation_strong_signals_for_expert(
                global_model=self.model,
                previous_global_state_dict=global_state,
                previous_global_expert_state=previous_group["state_dict"],
                expert_id=int(previous_group.get("expert_id", previous_group.get("expert_ids", [0])[0])),
                client_expert_states=current_states,
                val_loader=self._iter_server_validation_batches(num_batches),
                loss_fn=self.criterion,
                device=self.device,
                config=self.build_vsmc_config(),
                expert_scope=scope_key,
                aux_loss_weight=self.args.aux_loss_weight,
                sample_counts=[updated_client_updates[idx][1] for idx in valid_client_indices],
            )
            replacement_rewards = reward_info.get("replacement_rewards", [])
            delta_rewards = reward_info.get("delta_rewards", [])
            replaced_losses = reward_info.get("replaced_losses", [])
            best_delta_steps = reward_info.get("best_delta_steps", [])
            candidate_signatures = reward_info.get("candidate_signatures", [])
            base_signature = reward_info.get("base_signature", torch.empty(0)).tolist()
            base_loss = float(reward_info.get("base_loss", float("nan")))
            for local_idx, client_idx in enumerate(valid_client_indices):
                metrics = updated_client_updates[client_idx][2]
                metrics["server_val_expert_rewards"][scope_key] = (
                    float(replacement_rewards[local_idx]) if local_idx < len(replacement_rewards) else 0.0
                )
                metrics["server_val_delta_rewards"][scope_key] = (
                    float(delta_rewards[local_idx]) if local_idx < len(delta_rewards) else 0.0
                )
                metrics["server_val_best_delta_steps"][scope_key] = (
                    float(best_delta_steps[local_idx]) if local_idx < len(best_delta_steps) else 0.0
                )
                metrics["server_val_base_losses"][scope_key] = base_loss
                metrics["server_val_expert_losses"][scope_key] = (
                    float(replaced_losses[local_idx]) if local_idx < len(replaced_losses) else float("nan")
                )
                metrics["server_val_candidate_signatures"][scope_key] = (
                    candidate_signatures[local_idx].tolist() if local_idx < len(candidate_signatures) else []
                )
                metrics["server_val_base_signatures"][scope_key] = list(base_signature)
        return updated_client_updates

    def apply_router_usage_calibration(self):
        if not self.args.vsmc_use_router_usage_calibration:
            return
        usage_by_expert = collections.defaultdict(list)
        for info in self.latest_vsmc_round_infos:
            expert_id = int(info.get("expert_id", -1))
            if expert_id < 0:
                continue
            usage_value = float(info.get("usage_mean", 0.0))
            usage_by_expert[expert_id].append(usage_value)
        if not usage_by_expert:
            return
        target_usage = self.args.vsmc_target_usage_override
        if target_usage is None:
            target_usage = 1.0 / max(int(self.args.switch_num_experts), 1)
        mean_usage = {
            expert_id: sum(values) / max(len(values), 1)
            for expert_id, values in usage_by_expert.items()
        }
        usage_values = list(mean_usage.values())
        router_usage_entropy = 0.0
        if usage_values:
            probs = torch.as_tensor(usage_values, dtype=torch.float64)
            probs = probs / probs.sum().clamp_min(1e-12)
            router_usage_entropy = float((-probs * torch.log(probs.clamp_min(1e-12))).sum().item())

        state_dict = self.model.state_dict()
        updated = False
        for name, tensor in state_dict.items():
            lower_name = name.lower()
            if "bias" not in lower_name or ("router" not in lower_name and "gate" not in lower_name):
                continue
            if not (torch.is_tensor(tensor) and torch.is_floating_point(tensor)):
                continue
            if tensor.ndim != 1 or tensor.numel() < self.args.switch_num_experts:
                continue
            updated_tensor = tensor.detach().clone()
            for expert_id in range(min(updated_tensor.numel(), self.args.switch_num_experts)):
                usage_error = float(target_usage) - float(mean_usage.get(expert_id, target_usage))
                updated_tensor[expert_id] = updated_tensor[expert_id] + float(self.args.vsmc_router_bias_lr) * usage_error
            state_dict[name] = updated_tensor
            updated = True
        if updated:
            self.model.load_state_dict(state_dict)
        for info in self.latest_vsmc_round_infos:
            info["router_usage_entropy"] = float(router_usage_entropy)
            info["router_usage_min"] = float(min(usage_values) if usage_values else 0.0)
            info["router_usage_max"] = float(max(usage_values) if usage_values else 0.0)
        self.logger.info(
            f"--router_usage_calibration --enabled:{updated} "
            f"--router_usage_entropy:{router_usage_entropy:.4f} "
            f"--router_usage_min:{min(usage_values) if usage_values else 0.0:.4f} "
            f"--router_usage_max:{max(usage_values) if usage_values else 0.0:.4f}"
        )

    def apply_server_router_finetune(self):
        if not self.args.use_server_router_finetune:
            return
        named_router_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if ("router" in name.lower() or "gate" in name.lower()) and param.requires_grad
        ]
        if not named_router_params:
            self.logger.warning("--router_finetune --enabled:false --reason:no_router_params")
            return
        original_requires_grad = {name: param.requires_grad for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.requires_grad = ("router" in name.lower() or "gate" in name.lower())
        router_params = [param for _, param in named_router_params]
        optimizer = torch.optim.Adam(router_params, lr=float(self.args.router_finetune_lr))
        losses = []
        original_training = self.model.training
        self.model.eval()
        for _ in range(max(int(self.args.router_finetune_steps), 0)):
            optimizer.zero_grad()
            total_loss = None
            batch_count = 0
            for inputs, labels in self._iter_server_validation_batches(max(int(self.args.router_finetune_batches), 1)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, router_stats = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if isinstance(router_stats, dict) and router_stats.get("aux_loss") is not None:
                    loss = loss + self.args.aux_loss_weight * router_stats["aux_loss"]
                total_loss = loss if total_loss is None else (total_loss + loss)
                batch_count += 1
            if total_loss is None or batch_count == 0:
                continue
            total_loss = total_loss / batch_count
            if not torch.isfinite(total_loss).all().item():
                self.logger.warning("--router_finetune --enabled:false --reason:non_finite_loss")
                break
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(router_params, max_norm=1.0)
            optimizer.step()
            losses.append(float(total_loss.detach().item()))
        for name, param in self.model.named_parameters():
            param.requires_grad = original_requires_grad[name]
        if original_training:
            self.model.train()
        self.logger.info(
            f"--router_finetune --enabled:true --steps:{max(int(self.args.router_finetune_steps), 0)} "
            f"--batches:{max(int(self.args.router_finetune_batches), 1)} "
            f"--lr:{float(self.args.router_finetune_lr):.6f} "
            f"--mean_loss:{(sum(losses) / len(losses)) if losses else 0.0:.4f}"
        )

    def apply_expert_alignment(self, round_idx):
        if not self.args.use_expert_alignment:
            return
        update_every = max(int(self.args.alignment_update_every), 1)
        if round_idx % update_every != 0:
            self.logger.info(
                f"--expert_alignment --enabled:false --round:{round_idx} --reason:update_every --update_every:{update_every}"
            )
            return
        global_state = self.model.state_dict()
        base_state, expert_groups = split_state_dict_by_expert(global_state, num_experts=self.args.switch_num_experts)
        if not expert_groups:
            self.logger.warning("--expert_alignment --enabled:false --reason:no_expert_groups")
            return
        grouped_blocks = group_expert_groups_by_block(expert_groups)
        updated_expert_groups = collections.OrderedDict()
        for expert_scope, group in expert_groups.items():
            updated_expert_groups[expert_scope] = {
                "expert_id": int(group["expert_id"]),
                "expert_scope": expert_scope,
                "state_dict": collections.OrderedDict(
                    (key, value.detach().clone() if torch.is_tensor(value) else value)
                    for key, value in group["state_dict"].items()
                ),
            }
        aligned_blocks = 0
        total_experts = 0
        strength = float(max(0.0, min(1.0, self.args.alignment_strength)))
        for block_scope, block_group in grouped_blocks.items():
            expert_scopes = list(block_group.get("expert_scopes", []))
            if len(expert_scopes) < 2:
                continue
            total_experts += len(expert_scopes)
            key_union = []
            seen = set()
            for expert_scope in expert_scopes:
                for key in expert_groups[expert_scope]["state_dict"].keys():
                    if key not in seen:
                        seen.add(key)
                        key_union.append(key)
            mean_state = collections.OrderedDict()
            for key in key_union:
                tensors = []
                for expert_scope in expert_scopes:
                    value = expert_groups[expert_scope]["state_dict"].get(key)
                    if torch.is_tensor(value) and torch.is_floating_point(value):
                        tensors.append(value.detach().to(device=self.device, dtype=torch.float32))
                if not tensors:
                    continue
                mean_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
            if not mean_state:
                continue
            aligned_blocks += 1
            for expert_scope in expert_scopes:
                state_dict = updated_expert_groups[expert_scope]["state_dict"]
                for key, center_value in mean_state.items():
                    if key not in state_dict:
                        continue
                    current_value = state_dict[key]
                    if not (torch.is_tensor(current_value) and torch.is_floating_point(current_value)):
                        continue
                    center = center_value.to(device=current_value.device, dtype=current_value.dtype)
                    state_dict[key] = (1.0 - strength) * current_value + strength * center
        merged_state = merge_expert_states_into_global_state(base_state, updated_expert_groups)
        merged_state = _merge_with_reference_state(global_state, merged_state, device=self.device)
        self.model.load_state_dict(merged_state)
        self.logger.info(
            f"--expert_alignment --enabled:true --round:{round_idx} --metric:{self.args.alignment_metric} "
            f"--strength:{strength:.4f} --aligned_blocks:{aligned_blocks} --aligned_experts:{total_experts}"
        )

    def aggregate_temporal_vsmc_expert(self, client_updates, round_idx):
        if self.temporal_vsmc_aggregator is None:
            self.temporal_vsmc_aggregator = TemporalVSMCExpertAggregator(
                self.build_temporal_vsmc_config(),
                logger=self.logger,
            )

        global_state = self.load_server_state()
        aggregated_state, round_infos, updated_memory = self.temporal_vsmc_aggregator.aggregate_model_state(
            global_state=global_state,
            client_updates=client_updates,
            num_experts=self.args.switch_num_experts,
            round_idx=round_idx,
            temporal_memory=self.temporal_vsmc_memory,
            device=self.device,
        )
        self.temporal_vsmc_memory = updated_memory
        self.latest_vsmc_round_infos = round_infos
        self.model.load_state_dict(aggregated_state)
        if self.args.vsmc_use_router_usage_calibration:
            self.apply_router_usage_calibration()
        if self.args.use_server_router_finetune:
            self.apply_server_router_finetune()
        if self.args.use_expert_alignment:
            self.apply_expert_alignment(round_idx)

        summary_ess = [
            info["normalized_ess"]
            for info in round_infos
            if isinstance(info.get("normalized_ess"), (int, float))
        ]
        summary_fallbacks = sum(1 for info in round_infos if info.get("fallback_used"))
        self.logger.info(
            f"--temporal_vsmc_expert --round:{round_idx} --num_expert_groups:{len(round_infos)} "
            f"--mean_normalized_ess:{(sum(summary_ess) / len(summary_ess)) if summary_ess else 0.0:.4f} "
            f"--fallback_groups:{summary_fallbacks}"
        )
        for info in round_infos:
            weights = info.get("weights", [])
            mean_drift = (
                sum(info.get("drift_values", [])) / len(info.get("drift_values", []))
                if info.get("drift_values")
                else 0.0
            )
            mean_align = (
                sum(info.get("alignment_values", [])) / len(info.get("alignment_values", []))
                if info.get("alignment_values")
                else 0.0
            )
            mean_jump = (
                sum(info.get("jump_values", [])) / len(info.get("jump_values", []))
                if info.get("jump_values")
                else 0.0
            )
            self.logger.info(
                f"--temporal_vsmc_expert_group --scope:{info['expert_scope']} --expert_id:{info['expert_id']} "
                f"--ess:{info.get('ess', 0.0):.4f} --normalized_ess:{info.get('normalized_ess', 0.0):.4f} "
                f"--min_weight:{min(weights) if weights else 0.0:.4f} "
                f"--max_weight:{max(weights) if weights else 0.0:.4f} "
                f"--mean_drift:{mean_drift:.4f} --mean_alignment:{mean_align:.4f} "
                f"--mean_jump:{mean_jump:.4f} --temporal_enabled:{bool(info.get('temporal_enabled'))} "
                f"--warmup:{bool(info.get('warmup_used'))} --fallback:{bool(info.get('fallback_used'))}"
            )

    def aggregate_joint_vsmc_expert(self, client_updates, round_idx):
        if self.joint_vsmc_aggregator is None:
            self.joint_vsmc_aggregator = JointVSMCExpertAggregator(
                self.build_joint_vsmc_config(),
                num_experts=self.args.switch_num_experts,
                logger=self.logger,
            )
        global_state = self.load_server_state()
        if self.args.vsmc_use_server_validation:
            client_updates = self.attach_server_validation_scores(client_updates, global_state, include_router=True)
        aggregated_state, round_infos = self.joint_vsmc_aggregator.aggregate_model_state(
            global_state=global_state,
            client_updates=client_updates,
            num_experts=self.args.switch_num_experts,
            device=self.device,
        )
        self.latest_vsmc_round_infos = round_infos
        self.model.load_state_dict(aggregated_state)
        summary_ess = [
            info["normalized_ess"]
            for info in round_infos
            if isinstance(info.get("normalized_ess"), (int, float))
        ]
        summary_fallbacks = sum(1 for info in round_infos if info.get("fallback_used"))
        self.logger.info(
            f"--joint_vsmc_expert --round:{round_idx} --num_groups:{len(round_infos)} "
            f"--mean_normalized_ess:{(sum(summary_ess) / len(summary_ess)) if summary_ess else 0.0:.4f} "
            f"--fallback_groups:{summary_fallbacks}"
        )
        for info in round_infos:
            weights = info.get("weights", [])
            self.logger.info(
                f"--joint_group --scope:{info.get('expert_scope', '')} "
                f"--expert_id:{info.get('expert_id', -1)} "
                f"--ess:{info.get('ess', 0.0):.4f} "
                f"--normalized_ess:{info.get('normalized_ess', 0.0):.4f} "
                f"--min_alpha:{min(weights) if weights else 0.0:.4f} "
                f"--max_alpha:{max(weights) if weights else 0.0:.4f} "
                f"--log_f_mean:{info.get('log_f_mean', 0.0):.4f} "
                f"--log_g_mean:{info.get('log_g_mean', 0.0):.4f} "
                f"--log_r_mean:{info.get('log_r_mean', 0.0):.4f} "
                f"--server_val_loss_mean:{info.get('server_val_loss_mean', 0.0):.4f} "
                f"--router_consistency_mean:{info.get('router_consistency_mean', 0.0):.4f} "
                f"--fallback:{bool(info.get('fallback_used'))}"
            )

    def aggregate_moe_state_policy(self, client_updates, round_idx):
        if self.moe_state_vsmc_aggregator is None:
            _, MoEStateVSMCAggregator = self._load_moe_state_policy_components()
            self.moe_state_vsmc_aggregator = MoEStateVSMCAggregator(
                self.build_moe_state_policy_config(),
                num_experts=self.args.switch_num_experts,
                logger=self.logger,
            )
        global_state = self.load_server_state()
        aggregated_state, round_infos = self.moe_state_vsmc_aggregator.aggregate_model_state(
            global_model=self.model,
            global_state=global_state,
            client_updates=client_updates,
            num_experts=self.args.switch_num_experts,
            device=self.device,
            val_loader=self._iter_server_validation_batches(max(int(self.args.moe_state_val_batches), 1)),
            loss_fn=self.criterion,
            round_idx=round_idx,
            aux_loss_weight=self.args.aux_loss_weight,
        )
        self.latest_vsmc_round_infos = round_infos
        self.model.load_state_dict(aggregated_state)
        summary = dict(getattr(self.moe_state_vsmc_aggregator.state, "last_round_summary", {}) or {})
        if self.moe_state_vsmc_aggregator.state.last_round_summary is not None:
            self.moe_state_vsmc_aggregator.state.last_round_summary["seed"] = int(self.args.seed)
            self.moe_state_vsmc_aggregator.state.last_round_summary["method"] = str(self.args.aggregation_method)
        self.logger.info(
            f"--moe_state_vsmc --round:{round_idx} --selector_mode:{self.args.moe_state_selector_mode} "
            f"--num_particles:{summary.get('num_particles', self.args.moe_state_num_particles)} "
            f"--elite_top_k:{summary.get('elite_top_k', self.args.moe_state_elite_top_k)} "
            f"--normalized_ess:{float(summary.get('normalized_ess', 0.0) or 0.0):.4f} "
            f"--proposal_loss:{float(summary.get('proposal_loss', 0.0) or 0.0):.4f} "
            f"--logg_mean:{float(summary.get('logg_mean', 0.0) or 0.0):.4f} "
            f"--logq_mean:{float(summary.get('logq_mean', 0.0) or 0.0):.4f} "
            f"--resampled:{bool(summary.get('resampled', False))}"
        )

    def aggregate_learned_vsmc(self, client_updates, round_idx):
        if self.learned_vsmc_aggregator is None:
            _, LearnedVSMCExpertAggregator = self._load_learned_vsmc_components()
            self.learned_vsmc_aggregator = LearnedVSMCExpertAggregator(
                self.build_learned_vsmc_config(),
                num_experts=self.args.switch_num_experts,
                logger=self.logger,
            )

        global_state = self.load_server_state()
        aggregated_state, round_infos = self.learned_vsmc_aggregator.aggregate_model_state(
            global_model=self.model,
            global_state=global_state,
            client_updates=client_updates,
            num_experts=self.args.switch_num_experts,
            device=self.device,
            val_loader=self._iter_server_validation_batches(max(int(self.args.learned_vsmc_val_batches), 1)),
            loss_fn=self.criterion,
            round_idx=round_idx,
            aux_loss_weight=self.args.aux_loss_weight,
        )
        self.latest_vsmc_round_infos = round_infos
        self.model.load_state_dict(aggregated_state)

        summary_ess = [
            info["normalized_ess"]
            for info in round_infos
            if isinstance(info.get("normalized_ess"), (int, float))
        ]
        differentiable_g_count = sum(1 for info in round_infos if bool(info.get("differentiable_g", False)))
        reward_zero_std_count = sum(1 for info in round_infos if bool(info.get("reward_zero_std", False)))
        effective_log_g_count = sum(1 for info in round_infos if float(info.get("log_g_std", 0.0) or 0.0) > 1e-3)
        total_g_groups = max(len(round_infos), 1)
        self.logger.info(
            f"--learned_vsmc --round:{round_idx} --mode:{self.args.learned_vsmc_mode} --num_expert_groups:{len(round_infos)} "
            f"--mean_normalized_ess:{(sum(summary_ess) / len(summary_ess)) if summary_ess else 0.0:.4f} "
            f"--differentiable_g_group_count:{differentiable_g_count} --total_g_group_count:{len(round_infos)} "
            f"--differentiable_g_group_ratio:{differentiable_g_count / total_g_groups:.4f} "
            f"--reward_zero_std_count:{reward_zero_std_count} --log_g_effective_group_ratio:{effective_log_g_count / total_g_groups:.4f}"
        )
        for info in round_infos:
            weights = info.get("weights", [])
            append_vsmc_group_detail_log(
                f"--learned_vsmc_group --round:{round_idx} --scope:{info['expert_scope']} --expert_id:{info['expert_id']} "
                f"--num_particles:{info.get('num_particles', 0)} "
                f"--ess:{info.get('ess', 0.0):.4f} --normalized_ess:{info.get('normalized_ess', 0.0):.4f} "
                f"--min_alpha:{min(weights) if weights else 0.0:.4f} "
                f"--max_alpha:{max(weights) if weights else 0.0:.4f} "
                f"--learned_vsmc_logZ:{info.get('logZ_increment', 0.0):.4f} "
                f"--proposal_loss:{info.get('proposal_loss', 0.0):.4f} "
                f"--latent_coeff0_mean:{info.get('latent_coeff0_mean', info.get('mean_a', 0.0)):.4f} --latent_coeff0_std:{info.get('latent_coeff0_std', info.get('std_a', 0.0)):.4f} "
                f"--latent_coeff1_mean:{info.get('latent_coeff1_mean', info.get('mean_b', 0.0)):.4f} --latent_coeff1_std:{info.get('latent_coeff1_std', info.get('std_b', 0.0)):.4f} "
                f"--log_f_mean:{info.get('log_f_mean', 0.0):.4f} "
                f"--log_g_mean:{info.get('log_g_mean', 0.0):.4f} "
                f"--log_q_mean:{info.get('log_q_mean', 0.0):.4f} "
                f"--log_w_mean:{info.get('log_w_mean', 0.0):.4f} "
                f"--val_reward_mean:{info.get('val_reward_mean', 0.0):.4f} "
                f"--val_reward_max:{info.get('val_reward_max', 0.0):.4f} "
                f"--server_val_loss_mean:{info.get('server_val_loss_mean', 0.0):.4f} "
                f"--fedavg_delta_norm:{info.get('fedavg_delta_norm', 0.0):.4f} "
                f"--particle_delta_norm_mean:{info.get('particle_delta_norm_mean', 0.0):.4f} "
                f"--raw_reward_std:{info.get('raw_reward_std', 0.0)} "
                f"--log_g_std:{info.get('log_g_std', 0.0)} "
                f"--differentiable_g:{bool(info.get('differentiable_g', False))} "
                f"--particle_requires_grad:{bool(info.get('particle_requires_grad', False))} "
                f"--base_correct_confidence:{info.get('base_correct_confidence', 0.0):.4f} "
                f"--particle_correct_confidence_mean:{info.get('particle_correct_confidence_mean', 0.0):.4f} "
                f"--log_q_coef:{info.get('learned_vsmc_logq_coef', 0.0):.4f} "
                f"--alpha_max_before_clip:{info.get('alpha_max_before_clip', 0.0)} "
                f"--alpha_max_after_clip:{info.get('alpha_max_after_clip', 0.0)} "
                f"--proposal_grad_norm:{info.get('proposal_grad_norm', 'NA')} "
                f"--proposal_param_delta_norm:{info.get('proposal_param_delta_norm', 'NA')} "
                f"--best_particle_is_fedavg:{info.get('best_particle_is_fedavg', False)} "
                f"--fallback_reason:{info.get('fallback_reason', '')} "
                f"--used_reinforce:{bool(info.get('used_reinforce'))}"
            )

    def aggregate_temporal_latent_vsmc_expert(self, client_updates, round_idx):
        if self.temporal_latent_aggregator is None:
            self.temporal_latent_aggregator = TemporalLatentExpertAggregator(
                self.build_temporal_latent_config(),
                logger=self.logger,
            )

        global_state = self.load_server_state()
        aggregated_state, round_infos, updated_memory = self.temporal_latent_aggregator.aggregate_model_state(
            global_state=global_state,
            client_updates=client_updates,
            num_experts=self.args.switch_num_experts,
            round_idx=round_idx,
            latent_memory=self.temporal_latent_memory,
            device=self.device,
        )
        self.temporal_latent_memory = updated_memory
        self.latest_vsmc_round_infos = round_infos
        self.model.load_state_dict(aggregated_state)

        summary_ess = [
            info["normalized_ess"]
            for info in round_infos
            if isinstance(info.get("normalized_ess"), (int, float))
        ]
        summary_fallbacks = sum(1 for info in round_infos if info.get("fallback_used"))
        self.logger.info(
            f"--temporal_latent_vsmc_expert --round:{round_idx} --num_expert_groups:{len(round_infos)} "
            f"--mean_normalized_ess:{(sum(summary_ess) / len(summary_ess)) if summary_ess else 0.0:.4f} "
            f"--fallback_groups:{summary_fallbacks}"
        )
        for info in round_infos:
            weights = info.get("weights", [])
            mean_drift = (
                sum(info.get("drift_values", [])) / len(info.get("drift_values", []))
                if info.get("drift_values")
                else 0.0
            )
            mean_latent_distance = (
                sum(info.get("latent_distance_values", [])) / len(info.get("latent_distance_values", []))
                if info.get("latent_distance_values")
                else 0.0
            )
            self.logger.info(
                f"--temporal_latent_group --scope:{info['expert_scope']} --expert_id:{info['expert_id']} "
                f"--ess:{info.get('ess', 0.0):.4f} --normalized_ess:{info.get('normalized_ess', 0.0):.4f} "
                f"--min_weight:{min(weights) if weights else 0.0:.4f} "
                f"--max_weight:{max(weights) if weights else 0.0:.4f} "
                f"--mean_drift:{mean_drift:.4f} --mean_latent_distance:{mean_latent_distance:.4f} "
                f"--posterior_cov_trace:{info.get('posterior_cov_trace', 0.0):.4f} "
                f"--latent_enabled:{bool(info.get('latent_enabled'))} "
                f"--fallback:{bool(info.get('fallback_used'))}"
            )

    @staticmethod
    def _serialize_list(values):
        return json.dumps(values)

    @staticmethod
    def _metric_value(info, key, default=0.0):
        value = info.get(key, default)
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    def _safe_float_list_mean(self, values, default=0.0):
        vals = [float(v) for v in values if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else float(default)

    def _round_phase(self, round_idx: int) -> str:
        ratio = 0.0 if self.server_epochs <= 1 else float(round_idx - 1) / float(self.server_epochs - 1)
        anneal_start = float(getattr(self.args, "learned_vsmc_anneal_start_ratio", 0.6))
        if ratio < 0.2:
            return "warmup"
        if ratio < anneal_start:
            return "exploration"
        return "exploitation"

    def record_ha_vsmc_etf_round_metrics(
        self,
        round_idx,
        avg_client_train_loss,
        avg_client_train_acc,
        server_test_loss,
        server_test_acc,
    ):
        del avg_client_train_acc
        if float(server_test_acc) > float(self.best_server_test_acc):
            self.best_server_test_acc = float(server_test_acc)
            self.best_server_test_round = int(round_idx)
        round_data = self.latest_ha_vsmc_etf_round_data or {}
        infos = round_data.get("expert_summaries", [])
        particles = round_data.get("particle_summaries", [])
        self._write_ha_vsmc_round_weight_exports(round_idx, infos)
        for info in infos:
            record_ha_vsmc_etf_expert_summary(
                {
                    "round": int(round_idx),
                    "block_id": int(info.get("block_id", -1)),
                    "expert_id": int(info.get("expert_id", -1)),
                    "expert_scope": str(info.get("expert_scope", "")),
                    "lambda_expert": float(info.get("lambda_expert", 0.0)),
                    "anchor_mode": str(info.get("anchor_mode", "fedavg")),
                    "pure_vsmc": bool(info.get("pure_vsmc", False)),
                    "fedavg_blend_enabled": bool(info.get("fedavg_blend_enabled", True)),
                    "expert_output": str(info.get("expert_output", "")),
                    "function_drift_enabled": bool(info.get("function_drift_enabled", False)),
                    "temperature": float(info.get("temperature", 0.0)),
                    "ESS": float(info.get("ESS", 0.0)),
                    "ESS_ratio": float(info.get("ESS_ratio", 0.0)),
                    "resampled": bool(info.get("resampled", False)),
                    "particle_weight_entropy": float(info.get("particle_weight_entropy", 0.0)),
                    "particle_max_weight": float(info.get("particle_max_weight", 0.0)),
                    "max_particle_weight": float(info.get("max_particle_weight", info.get("particle_max_weight", 0.0))),
                    "client_weight_entropy": float(info.get("client_weight_entropy", 0.0)),
                    "expert_weight_entropy": float(info.get("expert_weight_entropy", info.get("client_weight_entropy", 0.0))),
                    "client_max_weight": float(info.get("client_max_weight", 0.0)),
                    "client_min_weight": float(info.get("client_min_weight", 0.0)),
                    "max_client_weight_mean": float(info.get("max_client_weight_mean", info.get("client_max_weight", 0.0))),
                    "min_client_weight_mean": float(info.get("min_client_weight_mean", info.get("client_min_weight", 0.0))),
                    "active_client_count_per_expert": int(info.get("active_client_count_per_expert", info.get("active_client_count", 0))),
                    "usage_score_mean": float(info.get("usage_score_mean", 0.0)),
                    "usage_score_std": float(info.get("usage_score_std", 0.0)),
                    "history_cos_mean": float(info.get("history_cos_mean", 0.0)),
                    "history_cos_std": float(info.get("history_cos_std", 0.0)),
                    "consensus_cos_mean": float(info.get("consensus_cos_mean", 0.0)),
                    "consensus_cos_std": float(info.get("consensus_cos_std", 0.0)),
                    "normalized_drift_mean": float(info.get("normalized_drift_mean", 0.0)),
                    "normalized_drift_std": float(info.get("normalized_drift_std", 0.0)),
                    "function_drift_mean": float(info.get("function_drift_mean", 0.0)),
                    "function_drift_std": float(info.get("function_drift_std", 0.0)),
                    "usage_signal_mean": float(info.get("usage_signal_mean", 0.0)),
                    "usage_signal_std": float(info.get("usage_signal_std", 0.0)),
                    "history_signal_mean": float(info.get("history_signal_mean", 0.0)),
                    "history_signal_std": float(info.get("history_signal_std", 0.0)),
                    "consensus_signal_mean": float(info.get("consensus_signal_mean", 0.0)),
                    "consensus_signal_std": float(info.get("consensus_signal_std", 0.0)),
                    "drift_signal_mean": float(info.get("drift_signal_mean", 0.0)),
                    "drift_signal_std": float(info.get("drift_signal_std", 0.0)),
                    "function_signal_mean": float(info.get("function_signal_mean", 0.0)),
                    "function_signal_std": float(info.get("function_signal_std", 0.0)),
                    "observation_score_mean": float(info.get("observation_score_mean", 0.0)),
                    "observation_score_std": float(info.get("observation_score_std", 0.0)),
                    "reliability_score_mean": float(info.get("reliability_score_mean", info.get("observation_score_mean", 0.0))),
                    "reliability_score_std": float(info.get("reliability_score_std", info.get("observation_score_std", 0.0))),
                    "particle_score_mean": float(info.get("particle_score_mean", 0.0)),
                    "particle_score_std": float(info.get("particle_score_std", 0.0)),
                    "vsmc_fedavg_l2_distance": float(info.get("vsmc_fedavg_l2_distance", 0.0)),
                    "final_fedavg_l2_distance": float(info.get("final_fedavg_l2_distance", 0.0)),
                    "final_residual_ratio": float(info.get("final_residual_ratio", 0.0)),
                    "velocity_norm": float(info.get("velocity_norm", 0.0)),
                    "second_moment_norm": float(info.get("second_moment_norm", 0.0)),
                    "function_drift_available": bool(info.get("function_drift_available", False)),
                    "fallback_used": bool(info.get("fallback_used", False)),
                    "fallback_reason": str(info.get("fallback_reason", "")),
                    "fallback_type": str(info.get("fallback_type", "")),
                    "confidence": float(info.get("confidence", 0.0)),
                    "score_transform": str(info.get("score_transform", "")),
                }
            )
        for row in particles:
            record_ha_vsmc_etf_particle_summary(
                {
                    "round": int(round_idx),
                    "block_id": int(row.get("block_id", -1)),
                    "expert_id": int(row.get("expert_id", -1)),
                    "particle_id": int(row.get("particle_id", -1)),
                    "particle_weight": float(row.get("particle_weight", 0.0)),
                    "particle_score": float(row.get("particle_score", 0.0)),
                    "particle_entropy": float(row.get("particle_entropy", 0.0)),
                    "particle_max_client_weight": float(row.get("particle_max_client_weight", 0.0)),
                    "particle_min_client_weight": float(row.get("particle_min_client_weight", 0.0)),
                    "dominant_client_idx": int(row.get("dominant_client_idx", -1)),
                    "resampled": bool(row.get("resampled", False)),
                }
            )
        record_ha_vsmc_etf_round_summary(
            {
                "round": int(round_idx),
                "anchor_mode": str(round_data.get("anchor_mode", "fedavg")),
                "pure_vsmc": bool(round_data.get("pure_vsmc", False)),
                "fedavg_blend_enabled": bool(round_data.get("fedavg_blend_enabled", True)),
                "expert_output": str(round_data.get("expert_output", "")),
                "function_drift_enabled": any(bool(info.get("function_drift_enabled", False)) for info in infos),
                "train_loss": float(avg_client_train_loss),
                "eval_loss": float(server_test_loss),
                "eval_acc": float(server_test_acc),
                "best_acc": float(self.best_server_test_acc),
                "final_acc": float(server_test_acc),
                "num_experts_logged": int(len(infos)),
                "fallback_count": int(sum(1 for info in infos if bool(info.get("fallback_used", False)))),
                "mean_lambda_expert": self._safe_float_list_mean([info.get("lambda_expert") for info in infos]),
                "mean_ESS_ratio": self._safe_float_list_mean([info.get("ESS_ratio") for info in infos]),
                "mean_particle_weight_entropy": self._safe_float_list_mean([info.get("particle_weight_entropy") for info in infos]),
                "mean_client_weight_entropy": self._safe_float_list_mean([info.get("client_weight_entropy") for info in infos]),
                "mean_expert_weight_entropy": self._safe_float_list_mean([info.get("expert_weight_entropy") for info in infos]),
                "mean_vsmc_fedavg_l2_distance": self._safe_float_list_mean([info.get("vsmc_fedavg_l2_distance") for info in infos]),
                "mean_final_residual_ratio": self._safe_float_list_mean([info.get("final_residual_ratio") for info in infos]),
                "mean_max_particle_weight": self._safe_float_list_mean([info.get("max_particle_weight", info.get("particle_max_weight")) for info in infos]),
                "mean_max_client_weight": self._safe_float_list_mean([info.get("max_client_weight_mean", info.get("client_max_weight")) for info in infos]),
                "mean_min_client_weight": self._safe_float_list_mean([info.get("min_client_weight_mean", info.get("client_min_weight")) for info in infos]),
                "mean_active_client_count_per_expert": self._safe_float_list_mean([info.get("active_client_count_per_expert", info.get("active_client_count")) for info in infos]),
            }
        )

    def record_vsmc_round_metrics(
        self,
        round_idx,
        avg_client_train_loss,
        avg_client_train_acc,
        server_test_loss,
        server_test_acc,
    ):
        if float(server_test_acc) > float(self.best_server_test_acc):
            self.best_server_test_acc = float(server_test_acc)
            self.best_server_test_round = int(round_idx)
        for info in self.latest_vsmc_round_infos:
            weights = info.get("weights", [])
            loss_values = info.get("loss_values", [])
            usage_values = info.get("usage_values", [])
            log_f_values = info.get("log_f_values", [])
            log_g_values = info.get("log_g_values", [])
            log_r_values = info.get("log_r_values", [])
            log_w_values = info.get("log_w_values", [])
            record_vsmc_round_metric(
                {
                    "round": int(round_idx),
                    "expert_id": int(info["expert_id"]),
                    "expert_scope": info["expert_scope"],
                    "num_particles": int(info.get("num_particles", 0)),
                    "log_f_mean": float(sum(log_f_values) / len(log_f_values)) if log_f_values else 0.0,
                    "log_f_min": float(min(log_f_values)) if log_f_values else 0.0,
                    "log_f_max": float(max(log_f_values)) if log_f_values else 0.0,
                    "log_g_mean": float(sum(log_g_values) / len(log_g_values)) if log_g_values else 0.0,
                    "log_g_min": float(min(log_g_values)) if log_g_values else 0.0,
                    "log_g_max": float(max(log_g_values)) if log_g_values else 0.0,
                    "log_r_mean": float(sum(log_r_values) / len(log_r_values)) if log_r_values else 0.0,
                    "log_r_min": float(min(log_r_values)) if log_r_values else 0.0,
                    "log_r_max": float(max(log_r_values)) if log_r_values else 0.0,
                    "log_q_mean": float(info.get("log_q_mean", 0.0)),
                    "proposal_loss": float(info.get("proposal_loss", 0.0)),
                    "mean_a": float(info.get("mean_a", 0.0)),
                    "std_a": float(info.get("std_a", 0.0)),
                    "mean_b": float(info.get("mean_b", 0.0)),
                    "std_b": float(info.get("std_b", 0.0)),
                    "latent_coeff0_mean": float(info.get("latent_coeff0_mean", info.get("mean_a", 0.0))),
                    "latent_coeff0_std": float(info.get("latent_coeff0_std", info.get("std_a", 0.0))),
                    "latent_coeff1_mean": float(info.get("latent_coeff1_mean", info.get("mean_b", 0.0))),
                    "latent_coeff1_std": float(info.get("latent_coeff1_std", info.get("std_b", 0.0))),
                    "val_reward_mean": float(info.get("val_reward_mean", 0.0)),
                    "val_reward_max": float(info.get("val_reward_max", 0.0)),
                    "log_w_min": float(min(log_w_values)) if log_w_values else 0.0,
                    "log_w_max": float(max(log_w_values)) if log_w_values else 0.0,
                    "log_w_mean": float(sum(log_w_values) / len(log_w_values)) if log_w_values else 0.0,
                    "min_alpha": float(info.get("min_alpha", 0.0)),
                    "max_alpha": float(info.get("max_alpha", 0.0)),
                    "mean_alpha": float(info.get("mean_alpha", 0.0)),
                    "ess": float(info.get("ess", 0.0)),
                    "normalized_ess": float(info.get("normalized_ess", 0.0)),
                    "logZ_increment": float(info.get("logZ_increment", 0.0)),
                    "logZ_total": float(info.get("logZ_total", 0.0)),
                    "loss_mean": float(sum(loss_values) / len(loss_values)) if loss_values else 0.0,
                    "loss_min": float(min(loss_values)) if loss_values else 0.0,
                    "loss_max": float(max(loss_values)) if loss_values else 0.0,
                    "server_val_base_loss": float(info.get("server_val_base_loss", 0.0)),
                    "server_val_replaced_loss_mean": float(info.get("server_val_replaced_loss_mean", 0.0)),
                    "server_val_reward_mean": float(info.get("server_val_reward_mean", 0.0)),
                    "server_val_reward_std": float(info.get("server_val_reward_std", 0.0)),
                    "delta_reward_mean": float(info.get("delta_reward_mean", 0.0)),
                    "delta_reward_std": float(info.get("delta_reward_std", 0.0)),
                    "credit_mean": float(info.get("credit_mean", 0.0)),
                    "credit_std": float(info.get("credit_std", 0.0)),
                    "novelty_mean": float(info.get("novelty_mean", 0.0)),
                    "stability_mean": float(info.get("stability_mean", 0.0)),
                    "server_val_loss_mean": float(info.get("server_val_loss_mean", 0.0)),
                    "usage_mean": float(sum(usage_values) / len(usage_values)) if usage_values else 0.0,
                    "route_count_sum": float(sum(info.get("route_counts", []))) if info.get("route_counts") else 0.0,
                    "alpha_g_server_ess": self._metric_value(info, "alpha_g_server_ess", "NA"),
                    "alpha_r_ess": self._metric_value(info, "alpha_r_ess", "NA"),
                    "alpha_final_ess": self._metric_value(info, "alpha_final_ess", info.get("normalized_ess", 0.0)),
                    "delta_norm_mean": float(info.get("delta_norm_mean", 0.0)),
                    "delta_norm_max": float(info.get("delta_norm_max", 0.0)),
                    "fedavg_delta_norm": float(info.get("fedavg_delta_norm", 0.0)),
                    "expert_server_lr": float(info.get("expert_server_lr", self.args.vsmc_expert_server_lr)),
                    "expert_ema_rho": float(info.get("expert_ema_rho", self.args.vsmc_expert_ema_rho)),
                    "router_usage_entropy": float(info.get("router_usage_entropy", 0.0)),
                    "router_usage_min": float(info.get("router_usage_min", 0.0)),
                    "router_usage_max": float(info.get("router_usage_max", 0.0)),
                    "fallback_used": bool(info.get("fallback_used")),
                    "fallback_reason": str(info.get("fallback_reason", "")),
                    "resampled": bool(info.get("resampled")),
                    "feature_min": float(info.get("feature_min", 0.0)),
                    "feature_max": float(info.get("feature_max", 0.0)),
                    "feature_mean": float(info.get("feature_mean", 0.0)),
                    "feature_has_nan": bool(info.get("feature_has_nan", False)),
                    "particle_delta_norm_mean": self._metric_value(info, "particle_delta_norm_mean", 0.0),
                    "particle_delta_mse_mean": self._metric_value(info, "particle_delta_mse_mean", 0.0),
                    "fedavg_delta_mse": self._metric_value(info, "fedavg_delta_mse", 0.0),
                    "particle_delta_norm_min": self._metric_value(info, "particle_delta_norm_min", 0.0),
                    "particle_delta_norm_max": self._metric_value(info, "particle_delta_norm_max", 0.0),
                    "particle_to_prev_mse_mean": self._metric_value(info, "particle_to_prev_mse_mean", 0.0),
                    "particle_to_fedavg_mse_mean": self._metric_value(info, "particle_to_fedavg_mse_mean", 0.0),
                    "particle_to_fedavg_mse_min": self._metric_value(info, "particle_to_fedavg_mse_min", 0.0),
                    "particle_to_fedavg_mse_max": self._metric_value(info, "particle_to_fedavg_mse_max", 0.0),
                    "proposal_grad_norm": self._metric_value(info, "proposal_grad_norm", "NA"),
                    "proposal_param_delta_norm": self._metric_value(info, "proposal_param_delta_norm", "NA"),
                    "logZ_requires_grad": self._metric_value(info, "logZ_requires_grad", False),
                    "log_f_requires_grad": self._metric_value(info, "log_f_requires_grad", False),
                    "log_g_requires_grad": self._metric_value(info, "log_g_requires_grad", False),
                    "log_q_requires_grad": self._metric_value(info, "log_q_requires_grad", False),
                    "raw_reward_mean": self._metric_value(info, "raw_reward_mean", 0.0),
                    "raw_reward_std": self._metric_value(info, "raw_reward_std", 0.0),
                    "raw_reward_min": self._metric_value(info, "raw_reward_min", 0.0),
                    "raw_reward_max": self._metric_value(info, "raw_reward_max", 0.0),
                    "normalized_reward_mean": self._metric_value(info, "normalized_reward_mean", 0.0),
                    "normalized_reward_std": self._metric_value(info, "normalized_reward_std", 0.0),
                    "log_g_std": self._metric_value(info, "log_g_std", 0.0),
                    "log_g_min": self._metric_value(info, "log_g_min", 0.0),
                    "log_g_max": self._metric_value(info, "log_g_max", 0.0),
                    "server_val_particle_loss_mean": self._metric_value(info, "server_val_particle_loss_mean", 0.0),
                    "server_val_particle_loss_min": self._metric_value(info, "server_val_particle_loss_min", 0.0),
                    "server_val_particle_loss_max": self._metric_value(info, "server_val_particle_loss_max", 0.0),
                    "particle_requires_grad": self._metric_value(info, "particle_requires_grad", False),
                    "differentiable_g": self._metric_value(info, "differentiable_g", False),
                    "normalized_reward_min": self._metric_value(info, "normalized_reward_min", 0.0),
                    "normalized_reward_max": self._metric_value(info, "normalized_reward_max", 0.0),
                    "base_correct_confidence": self._metric_value(info, "base_correct_confidence", 0.0),
                    "particle_correct_confidence_mean": self._metric_value(info, "particle_correct_confidence_mean", 0.0),
                    "particle_correct_confidence_min": self._metric_value(info, "particle_correct_confidence_min", 0.0),
                    "particle_correct_confidence_max": self._metric_value(info, "particle_correct_confidence_max", 0.0),
                    "confidence_reward_mean": self._metric_value(info, "confidence_reward_mean", 0.0),
                    "confidence_reward_std": self._metric_value(info, "confidence_reward_std", 0.0),
                    "reward_zero_std": self._metric_value(info, "reward_zero_std", False),
                    "log_w_std": self._metric_value(info, "log_w_std", 0.0),
                    "alpha_entropy": self._metric_value(info, "alpha_entropy", 0.0),
                    "alpha_uniform_l1": self._metric_value(info, "alpha_uniform_l1", 0.0),
                    "alpha_max_before_clip": self._metric_value(info, "alpha_max_before_clip", "NA"),
                    "alpha_max_after_clip": self._metric_value(info, "alpha_max_after_clip", "NA"),
                    "alpha_mixed_with_uniform": self._metric_value(info, "alpha_mixed_with_uniform", False),
                    "bad_alpha_fallback": self._metric_value(info, "bad_alpha_fallback", False),
                    "any_particle_nan": self._metric_value(info, "any_particle_nan", False),
                    "any_particle_inf": self._metric_value(info, "any_particle_inf", False),
                    "any_aggregated_expert_nan": self._metric_value(info, "any_aggregated_expert_nan", False),
                    "any_global_nan": self._metric_value(info, "any_global_nan", False),
                    "delta_div_norm": self._metric_value(info, "delta_div_norm", 0.0),
                    "delta_div_enabled": self._metric_value(info, "delta_div_enabled", False),
                    "learned_vsmc_logq_coef": self._metric_value(info, "learned_vsmc_logq_coef", 0.0),
                    "log_f_contribution_mean": self._metric_value(info, "log_f_contribution_mean", 0.0),
                    "log_g_contribution_mean": self._metric_value(info, "log_g_contribution_mean", 0.0),
                    "log_q_contribution_mean": self._metric_value(info, "log_q_contribution_mean", 0.0),
                    "log_q_contribution_abs_mean": self._metric_value(info, "log_q_contribution_abs_mean", 0.0),
                    "log_q_dominates": self._metric_value(info, "log_q_dominates", False),
                    "fedavg_particle_alpha": self._metric_value(info, "fedavg_particle_alpha", "NA"),
                    "fedavg_particle_val_reward": self._metric_value(info, "fedavg_particle_val_reward", "NA"),
                    "fedavg_particle_raw_reward": self._metric_value(info, "fedavg_particle_raw_reward", "NA"),
                    "fedavg_particle_normalized_reward": self._metric_value(info, "fedavg_particle_normalized_reward", "NA"),
                    "fedavg_particle_log_g": self._metric_value(info, "fedavg_particle_log_g", "NA"),
                    "best_particle_index": self._metric_value(info, "best_particle_index", -1),
                    "best_particle_is_fedavg": self._metric_value(info, "best_particle_is_fedavg", False),
                }
            )
        infos = self.latest_vsmc_round_infos
        num_groups = len(infos)
        differentiable_ratio = sum(1 for info in infos if bool(info.get("differentiable_g", False))) / max(num_groups, 1)
        reward_zero_std_count = sum(1 for info in infos if bool(info.get("reward_zero_std", False)))
        effective_log_g_ratio = sum(1 for info in infos if float(info.get("log_g_std", 0.0) or 0.0) > 1e-3) / max(num_groups, 1)
        log_q_dominates_ratio = sum(1 for info in infos if bool(info.get("log_q_dominates", False))) / max(num_groups, 1)
        best_particle_is_fedavg_rate = sum(1 for info in infos if bool(info.get("best_particle_is_fedavg", False))) / max(num_groups, 1)
        fallback_count = sum(1 for info in infos if bool(info.get("fallback_used", False)))
        used_reinforce_count = sum(1 for info in infos if bool(info.get("used_reinforce", False)))
        record_vsmc_round_summary(
            {
                "round": int(round_idx),
                "method": str(self.args.aggregation_method),
                "mode": str(getattr(self.args, "learned_vsmc_mode", getattr(self.args, "vsmc_mode", "NA"))),
                "num_groups": int(num_groups),
                "group_level": "block" if (infos and ".experts." not in str(infos[0].get("expert_scope", ""))) else "expert",
                "avg_client_train_loss": float(avg_client_train_loss),
                "avg_client_train_acc": float(avg_client_train_acc),
                "server_test_loss": float(server_test_loss),
                "server_test_acc": float(server_test_acc),
                "best_acc_so_far": float(self.best_server_test_acc),
                "best_acc_round": int(self.best_server_test_round),
                "mean_normalized_ess": self._safe_float_list_mean([info.get("normalized_ess") for info in infos]),
                "differentiable_g_group_ratio": float(differentiable_ratio),
                "reward_zero_std_count": int(reward_zero_std_count),
                "reward_std_mean": self._safe_float_list_mean([info.get("raw_reward_std") for info in infos]),
                "log_g_std_mean": self._safe_float_list_mean([info.get("log_g_std") for info in infos]),
                "log_g_effective_group_ratio": float(effective_log_g_ratio),
                "log_q_dominates_ratio": float(log_q_dominates_ratio),
                "best_particle_is_fedavg_rate": float(best_particle_is_fedavg_rate),
                "fedavg_particle_alpha_mean": self._safe_float_list_mean([info.get("fedavg_particle_alpha") for info in infos]),
                "proposal_grad_norm_mean": self._safe_float_list_mean([info.get("proposal_grad_norm") for info in infos]),
                "proposal_param_delta_norm_mean": self._safe_float_list_mean([info.get("proposal_param_delta_norm") for info in infos]),
                "fallback_count": int(fallback_count),
                "used_reinforce_count": int(used_reinforce_count),
                "phase": self._round_phase(int(round_idx)),
            }
        )

    def _write_run_summary(self, final_round: int, final_server_test_loss: float, final_server_test_acc: float):
        method = str(self.args.aggregation_method)
        text = (
            f"Method: {method}\n"
            f"Final round: {int(final_round)}\n"
            f"Final acc: {float(final_server_test_acc):.4f}\n"
            f"Final loss: {float(final_server_test_loss):.4f}\n"
            f"Best acc: {float(self.best_server_test_acc):.4f}\n"
            f"Best acc round: {int(self.best_server_test_round)}\n"
        )
        write_vsmc_run_summary(text)

    def train(self):
        if self.base_aggregation_method == "fedjets" and self.fedjets_runtime is not None:
            self.fedjets_runtime.train()
            return
        if self.base_aggregation_method == "pfedmoe" and self.pfedmoe_runtime is not None:
            self.pfedmoe_runtime.train()
            return
        if self.base_aggregation_method == "hfedmoe" and self.hfedmoe_runtime is not None:
            self.hfedmoe_runtime.train()
            return
        for round_idx in range(self.server_epochs):
            self.logger.info(
                f"============================== Round {round_idx + 1}/{self.server_epochs} start =============================="
            )
            global_state = self.load_server_state()
            client_updates = []
            client_train_loss_list = []
            client_train_acc_list = []

            for client_id in self.clients_id_list:
                client = Client(
                    parse=parse,
                    client_id=client_id,
                    logger=self.logger,
                    c_T=round_idx,
                    model_factory=self.build_model,
                    global_state=global_state,
                )
                client_state, num_samples, metrics = client.train()
                client_updates.append((client_state, num_samples, metrics))
                client_train_loss_list.append(metrics["train_loss"])
                client_train_acc_list.append(metrics["train_acc"])

            self.aggregate(client_updates, round_idx + 1)
            self.save_server_model()
            server_test_loss, server_test_acc = self.evaluate_server_test()
            if self.base_aggregation_method == "learned_vsmc_moe_state_policy_k8" and self.moe_state_vsmc_aggregator is not None:
                self.moe_state_vsmc_aggregator.record_post_eval_metrics(round_idx + 1, server_test_loss, server_test_acc)

            avg_client_train_acc = sum(client_train_acc_list) / len(client_train_acc_list)
            avg_client_train_loss = sum(client_train_loss_list) / len(client_train_loss_list)
            if self.base_aggregation_method in {"vsmc_expert", "temporal_vsmc_expert", "temporal_latent_vsmc_expert", "learned_vsmc"}:
                self.record_vsmc_round_metrics(
                    round_idx=round_idx + 1,
                    avg_client_train_loss=avg_client_train_loss,
                    avg_client_train_acc=avg_client_train_acc,
                    server_test_loss=server_test_loss,
                    server_test_acc=server_test_acc,
                )
            elif self.base_aggregation_method == "ha_vsmc_etf":
                self.record_ha_vsmc_etf_round_metrics(
                    round_idx=round_idx + 1,
                    avg_client_train_loss=avg_client_train_loss,
                    avg_client_train_acc=avg_client_train_acc,
                    server_test_loss=server_test_loss,
                    server_test_acc=server_test_acc,
                )
            if self.base_aggregation_method == "learned_vsmc" and round_idx + 1 >= 3 and server_test_acc <= 0.11:
                self.logger.warning("learned_vsmc global model not learning; likely reward/delta/log_q issue")
                if round_idx + 1 >= 5:
                    self.logger.warning("Suggested actions: increase learned_vsmc_mix_fedavg_candidate to 0.5; set learned_vsmc_logq_coef to 0.1 or 0; enable reward normalization; check fedavg_delta_norm")

            self.logger.info(
                f"--round:{round_idx + 1} --avg_client_train_loss:{avg_client_train_loss:.4f} "
                f"--avg_client_train_acc:{avg_client_train_acc:.4f}\n"
                f"--server_test_loss:{server_test_loss:.4f} --server_test_acc:{server_test_acc:.4f}\n"
                f"--global_train_loss:{avg_client_train_loss:.4f} "
                f"--global_eval_loss:{server_test_loss:.4f} "
                f"--global_eval_accuracy:{server_test_acc:.4f}\n"
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if self._uses_vsmc_round_metrics():
            self._write_run_summary(self.server_epochs, server_test_loss, server_test_acc)
