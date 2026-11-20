
from __future__ import annotations

import ast
import csv
import math
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from Models.SwitchTransformer import SwitchTransformerClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "experiments" / "analysis" / "pure_vsmc_failure"
EPS = 1e-12
ROUND_RE = re.compile(r"--round:(\d+).+?--server_test_loss:([0-9.]+) --server_test_acc:([0-9.]+)", re.S)

COMMON_MODEL_CONFIG = {
    "num_classes": 100,
    "image_size": 32,
    "patch_size": 4,
    "backbone_name": "resnet18",
    "hidden_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "ffn_dim": 256,
    "num_experts": 4,
    "dropout": 0.2,
    "resnet_stem_mode": "auto",
}

@dataclass
class ExperimentSpec:
    method: str
    kind: str
    root: Path
    model_path: Path
    data_path: Path
    detail_csv: Optional[Path]
    log_path: Optional[Path]
    round_summary_csv: Optional[Path]
    expert_summary_csv: Optional[Path]


def ensure_analysis_root() -> Path:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_ROOT


def _maybe(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def discover_experiments() -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    fedavg_root = REPO_ROOT / 'experiments' / 'run_8' / 'fedavg_cifar100_alpha0.1' / 'seed3'
    fedavg_data = REPO_ROOT / 'experiments' / 'run_8' / 'data_cifar100_alpha0.1_10client_seed3'
    if fedavg_root.exists():
        specs.append(ExperimentSpec(
            method='fedavg',
            kind='fedavg',
            root=fedavg_root,
            model_path=fedavg_root / 'model' / 'server.pth',
            data_path=fedavg_data,
            detail_csv=_maybe(fedavg_root / 'result' / 'detail' / 'data_cifar100 client_10 alpha_0.1 backbone_resnet18.csv'),
            log_path=_maybe(fedavg_root / 'result' / 'logs' / 'data_cifar100 client_10 alpha_0.1 backbone_resnet18.log'),
            round_summary_csv=None,
            expert_summary_csv=None,
        ))
    run11 = REPO_ROOT / 'experiments' / 'run_11'
    for name in [
        'pure_vsmc_baseline',
        'pure_vsmc_rank_based_score',
        'pure_vsmc_signal_normalized',
        'pure_vsmc_reliability_floor_eps0.05_freq',
        'pure_vsmc_low_temperature',
        'pure_vsmc_prob_ema_rho0.5',
    ]:
        root = run11 / name / 'seed3'
        if not root.exists():
            continue
        specs.append(ExperimentSpec(
            method=name,
            kind='pure_vsmc',
            root=root,
            model_path=root / 'model' / 'server.pth',
            data_path=run11 / 'data_cifar100_alpha0.1_10client_seed3',
            detail_csv=_maybe(root / 'result' / 'detail' / 'data_cifar100 client_10 alpha_0.1 backbone_resnet18.csv'),
            log_path=_maybe(root / 'result' / 'logs' / 'data_cifar100 client_10 alpha_0.1 backbone_resnet18.log'),
            round_summary_csv=_maybe(root / 'result' / 'logs' / 'ha_vsmc_etf_round_summary.csv'),
            expert_summary_csv=_maybe(root / 'result' / 'logs' / 'ha_vsmc_etf_expert_summary.csv'),
        ))
    run12 = REPO_ROOT / 'experiments' / 'run_12'
    if run12.exists():
        for p in sorted(run12.glob('pure_vsmc*/seed3')):
            summary = p / 'result' / 'logs' / 'ha_vsmc_etf_round_summary.csv'
            if not summary.exists():
                continue
            specs.append(ExperimentSpec(
                method=f'run12_{p.parent.name}',
                kind='pure_vsmc_run12',
                root=p,
                model_path=p / 'model' / 'server.pth',
                data_path=run11 / 'data_cifar100_alpha0.1_10client_seed3' if (run11 / 'data_cifar100_alpha0.1_10client_seed3').exists() else fedavg_data,
                detail_csv=_maybe(p / 'result' / 'detail' / 'data_cifar100 client_10 alpha_0.1 backbone_resnet18.csv'),
                log_path=_maybe(p / 'result' / 'logs' / 'data_cifar100 client_10 alpha_0.1 backbone_resnet18.log'),
                round_summary_csv=_maybe(summary),
                expert_summary_csv=_maybe(p / 'result' / 'logs' / 'ha_vsmc_etf_expert_summary.csv'),
            ))
    return specs


def read_csv(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return 'NA'
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 'NA'
    return f'{float(value):.{digits}f}'


def safe_float(value) -> Optional[float]:
    if value in (None, '', 'NA'):
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def load_server_test_dataset(data_path: Path):
    return torch.load(data_path / 'server_test.pth', weights_only=False)


def load_client_train_dataset(data_path: Path, client_id: int):
    return torch.load(data_path / f'{client_id}_train.pth', weights_only=False)


def build_model(device: torch.device) -> nn.Module:
    model = SwitchTransformerClassifier(**COMMON_MODEL_CONFIG)
    model.to(device)
    model.eval()
    return model


def remap_legacy_tokenizer_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    # Older CIFAR resnet tokenizer checkpoints omitted the Identity maxpool module,
    # so tokenizer indices >=3 are shifted by -1 versus the current model code.
    if "tokenizer.3.0.conv1.weight" not in state_dict:
        return state_dict
    remapped = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("tokenizer."):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) >= 3:
                parts[1] = str(int(parts[1]) + 1)
                key = '.'.join(parts)
        remapped[key] = value
    return remapped


def evaluate_state_dict(state_dict: Dict[str, torch.Tensor], data_path: Path, batch_size: int = 128, device: Optional[torch.device] = None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_server_test_dataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = build_model(device)
    state_dict = remap_legacy_tokenizer_keys(state_dict)
    model.load_state_dict(state_dict, strict=True)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, router_stats = model(inputs)
            loss = criterion(outputs, labels) + router_stats['aux_loss'] * 0.0
            total_loss += float(loss.item()) * inputs.size(0)
            total += inputs.size(0)
            pred1 = outputs.argmax(dim=1)
            correct1 += int((pred1 == labels).sum().item())
            topk = min(5, outputs.size(1))
            top5 = outputs.topk(topk, dim=1).indices
            correct5 += int((top5 == labels.unsqueeze(1)).any(dim=1).sum().item())
    return {
        'top1_acc': correct1 / total if total else 0.0,
        'top5_acc': correct5 / total if total else 0.0,
        'test_loss': total_loss / total if total else float('inf'),
        'num_samples': total,
    }


def parse_fedavg_rounds(log_path: Optional[Path]) -> List[Dict[str, float]]:
    if log_path is None or not log_path.exists():
        return []
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    rows = []
    for match in ROUND_RE.finditer(text):
        rows.append({
            'round': int(match.group(1)),
            'eval_loss': float(match.group(2)),
            'eval_acc': float(match.group(3)),
        })
    return rows


def aggregate_expert_features(expert_rows: List[Dict[str, str]]) -> Dict[int, Dict[str, float]]:
    by_round: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in expert_rows:
        by_round[int(row['round'])].append(row)
    result = {}
    for round_idx, rows in by_round.items():
        def vals(key):
            out = []
            for row in rows:
                v = safe_float(row.get(key))
                if v is not None:
                    out.append(v)
            return out
        max_weights = vals('client_max_weight')
        entropies = vals('client_weight_entropy')
        ess = vals('ESS_ratio')
        rel_mean = vals('reliability_score_mean')
        rel_std = vals('reliability_score_std')
        p_entropy = vals('particle_weight_entropy')
        f_signal_std = vals('function_signal_std')
        usage_std = vals('usage_signal_std')
        hist_std = vals('history_signal_std')
        cons_std = vals('consensus_signal_std')
        drift_std = vals('drift_signal_std')
        result[round_idx] = {
            'mean_max_client_weight': mean(max_weights) or 0.0,
            'mean_entropy': mean(entropies) or 0.0,
            'mean_ess_ratio': mean(ess) or 0.0,
            'experts_max_weight_gt_0.8': float(sum(1 for v in max_weights if v > 0.8)),
            'experts_max_weight_gt_0.9': float(sum(1 for v in max_weights if v > 0.9)),
            'reliability_score_mean': mean(rel_mean) or 0.0,
            'reliability_score_std': mean(rel_std) or 0.0,
            'particle_weight_entropy': mean(p_entropy) or 0.0,
            'score_u_std': mean(usage_std) or 0.0,
            'score_h_std': mean(hist_std) or 0.0,
            'score_c_std': mean(cons_std) or 0.0,
            'score_d_std': mean(drift_std) or 0.0,
            'score_f_std': mean(f_signal_std) or 0.0,
            'expert_count': float(len(rows)),
        }
    return result


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= EPS or deny <= EPS:
        return None
    return num / (denx * deny)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(rankdata(xs), rankdata(ys))


def classify_keys(state_dict: Dict[str, torch.Tensor]):
    expert_keys = []
    router_keys = []
    head_keys = []
    shared_keys = []
    unclassified = []
    for key in state_dict.keys():
        lower = key.lower()
        if 'experts.' in lower or '.expert.' in lower or 'ffn_experts' in lower:
            expert_keys.append(key)
        elif '.router.' in lower or '.gate.' in lower or 'gating' in lower:
            router_keys.append(key)
        elif lower.startswith('classifier') or '.classifier.' in lower or lower.startswith('fc') or '.fc.' in lower or 'head' in lower:
            head_keys.append(key)
        else:
            shared_keys.append(key)
        if key not in expert_keys and key not in router_keys and key not in head_keys and key not in shared_keys:
            unclassified.append(key)
    return {
        'expert': set(expert_keys),
        'router': set(router_keys),
        'head': set(head_keys),
        'shared': set(shared_keys),
        'unclassified': unclassified,
    }


def merge_components(base_state, donor_state, base_parts, donor_parts, use_shared='base', use_router='base', use_expert='base', use_head='base'):
    merged = OrderedDict((k, v.detach().clone()) for k, v in base_state.items())
    mapping = {
        'shared': use_shared,
        'router': use_router,
        'expert': use_expert,
        'head': use_head,
    }
    counts = {'shared': 0, 'router': 0, 'expert': 0, 'head': 0}
    for part, source in mapping.items():
        keys = base_parts[part]
        src_state = base_state if source == 'base' else donor_state
        for key in keys:
            merged[key] = src_state[key].detach().clone()
            counts[part] += 1
    return merged, counts


def analyze_effective_clients(specs: List[ExperimentSpec]):
    out_dir = ensure_analysis_root()
    detail_rows = []
    summary_rows = []
    for spec in specs:
        if spec.expert_summary_csv is None:
            continue
        rows = read_csv(spec.expert_summary_csv)
        by_round = defaultdict(list)
        for row in rows:
            detail_rows.append({
                'round': row.get('round'),
                'method': spec.method,
                'expert_id': row.get('expert_id'),
                'block_id': row.get('block_id'),
                'N_eff': 'NA',
                'entropy': row.get('client_weight_entropy', ''),
                'max_client_weight': row.get('client_max_weight', ''),
                'active_client_count': row.get('active_client_count_per_expert', ''),
                'top1_client': 'UNAVAILABLE_FROM_LOGS',
                'top1_weight': row.get('client_max_weight', ''),
                'top3_clients': 'UNAVAILABLE_FROM_LOGS',
                'top3_weight_sum': 'NA',
                'status': 'raw_client_weight_vector_missing',
            })
            by_round[int(row['round'])].append(row)
        last_rounds = sorted(by_round.keys())[-10:]
        tail_rows = [row for rnd in last_rounds for row in by_round[rnd]]
        max_weights = [safe_float(r.get('client_max_weight')) for r in tail_rows]
        entropies = [safe_float(r.get('client_weight_entropy')) for r in tail_rows]
        summary_rows.append({
            'method': spec.method,
            'rounds_covered': ','.join(map(str, last_rounds)),
            'N_eff_mean_last10': 'NA',
            'N_eff_min_last10': 'NA',
            'number_of_experts_with_N_eff_lt_1.5': 'NA',
            'number_of_experts_with_N_eff_lt_2.0': 'NA',
            'max_client_weight_mean_last10': format_float(mean(max_weights)),
            'number_of_experts_with_max_weight_gt_0.8': sum(1 for v in max_weights if v is not None and v > 0.8),
            'number_of_experts_with_max_weight_gt_0.9': sum(1 for v in max_weights if v is not None and v > 0.9),
            'entropy_mean_last10': format_float(mean(entropies)),
            'raw_weight_vector_available': False,
            'notes': 'Exact N_eff/top-client identities unavailable because expert summary logs did not persist per-expert client weight vectors.',
        })
    write_csv(out_dir / 'expert_effective_clients_detail.csv', detail_rows, [
        'round','method','expert_id','block_id','N_eff','entropy','max_client_weight','active_client_count','top1_client','top1_weight','top3_clients','top3_weight_sum','status'
    ])
    write_csv(out_dir / 'expert_effective_clients_summary.csv', summary_rows, [
        'method','rounds_covered','N_eff_mean_last10','N_eff_min_last10','number_of_experts_with_N_eff_lt_1.5','number_of_experts_with_N_eff_lt_2.0','max_client_weight_mean_last10','number_of_experts_with_max_weight_gt_0.8','number_of_experts_with_max_weight_gt_0.9','entropy_mean_last10','raw_weight_vector_available','notes'
    ])
    return detail_rows, summary_rows


def analyze_class_coverage(specs: List[ExperimentSpec]):
    out_dir = ensure_analysis_root()
    detail_rows = []
    summary_rows = []
    for spec in specs:
        if spec.kind == 'fedavg':
            continue
        detail_rows.append({
            'method': spec.method,
            'round': 'NA',
            'expert_id': 'NA',
            'top1_client': 'UNAVAILABLE_FROM_LOGS',
            'top1_weight': 'NA',
            'class_coverage_top1': 'NA',
            'class_coverage_top3': 'NA',
            'class_coverage_cum80': 'NA',
            'class_coverage_active': 'NA',
            'top3_clients': 'UNAVAILABLE_FROM_LOGS',
            'top3_weight_sum': 'NA',
            'cum80_clients': 'UNAVAILABLE_FROM_LOGS',
            'cum80_weight_sum': 'NA',
            'status': 'raw_client_weight_vector_missing',
        })
        summary_rows.append({
            'method': spec.method,
            'class_coverage_top1_mean_last10': 'NA',
            'class_coverage_top3_mean_last10': 'NA',
            'class_coverage_cum80_mean_last10': 'NA',
            'class_coverage_active_mean_last10': 'NA',
            'number_of_experts_top1_coverage_lt_5': 'NA',
            'number_of_experts_top3_coverage_lt_10': 'NA',
            'number_of_experts_cum80_coverage_lt_20': 'NA',
            'notes': 'Cannot reconstruct top-weight clients or class coverage from existing artifacts because raw per-expert client weights were not saved.',
        })
    write_csv(out_dir / 'expert_class_coverage_detail.csv', detail_rows, [
        'method','round','expert_id','top1_client','top1_weight','class_coverage_top1','class_coverage_top3','class_coverage_cum80','class_coverage_active','top3_clients','top3_weight_sum','cum80_clients','cum80_weight_sum','status'
    ])
    write_csv(out_dir / 'expert_class_coverage_summary.csv', summary_rows, [
        'method','class_coverage_top1_mean_last10','class_coverage_top3_mean_last10','class_coverage_cum80_mean_last10','class_coverage_active_mean_last10','number_of_experts_top1_coverage_lt_5','number_of_experts_top3_coverage_lt_10','number_of_experts_cum80_coverage_lt_20','notes'
    ])
    return detail_rows, summary_rows


def analyze_next_round_correlation(specs: List[ExperimentSpec]):
    out_dir = ensure_analysis_root()
    rows = []
    pooled = defaultdict(lambda: {'x': [], 'y': []})
    feature_keys = [
        'mean_max_client_weight','mean_entropy','mean_ess_ratio','experts_max_weight_gt_0.8','experts_max_weight_gt_0.9','reliability_score_mean','reliability_score_std','particle_weight_entropy','score_u_std','score_h_std','score_c_std','score_d_std','score_f_std'
    ]
    for spec in specs:
        if spec.round_summary_csv is None or spec.expert_summary_csv is None:
            continue
        round_rows = read_csv(spec.round_summary_csv)
        expert_rows = read_csv(spec.expert_summary_csv)
        agg = aggregate_expert_features(expert_rows)
        deltas = []
        for i in range(len(round_rows)-1):
            curr_round = int(round_rows[i]['round'])
            curr_acc = safe_float(round_rows[i].get('eval_acc'))
            next_acc = safe_float(round_rows[i+1].get('eval_acc'))
            if curr_acc is None or next_acc is None or curr_round not in agg:
                continue
            deltas.append((curr_round, next_acc - curr_acc))
        for feature in feature_keys:
            xs, ys = [], []
            for round_idx, delta in deltas:
                value = agg[round_idx].get(feature)
                if value is None:
                    continue
                xs.append(float(value))
                ys.append(float(delta))
                pooled[feature]['x'].append(float(value))
                pooled[feature]['y'].append(float(delta))
            rows.append({
                'method': spec.method,
                'feature': feature,
                'num_points': len(xs),
                'pearson_corr': format_float(pearson(xs, ys), 5),
                'spearman_corr': format_float(spearman(xs, ys), 5),
            })
    for feature, data in pooled.items():
        rows.append({
            'method': 'all_pure_vsmc',
            'feature': feature,
            'num_points': len(data['x']),
            'pearson_corr': format_float(pearson(data['x'], data['y']), 5),
            'spearman_corr': format_float(spearman(data['x'], data['y']), 5),
        })
    write_csv(out_dir / 'vsmc_next_round_correlation.csv', rows, ['method','feature','num_points','pearson_corr','spearman_corr'])
    md_lines = ['# VSMC Next-Round Correlation', '', '| method | feature | n | pearson | spearman |', '|---|---|---:|---:|---:|']
    for row in rows:
        md_lines.append(f"| {row['method']} | {row['feature']} | {row['num_points']} | {row['pearson_corr']} | {row['spearman_corr']} |")
    (out_dir / 'vsmc_next_round_correlation.md').write_text('\n'.join(md_lines) + '\n')
    return rows


def load_state(path: Path):
    return remap_legacy_tokenizer_keys(torch.load(path, map_location='cpu'))


def evaluate_component_swap(specs: List[ExperimentSpec]):
    out_dir = ensure_analysis_root()
    by_method = {spec.method: spec for spec in specs}
    fedavg = by_method.get('fedavg')
    if fedavg is None:
        raise FileNotFoundError('FedAvg checkpoint not found for component swap evaluation.')
    fedavg_state = load_state(fedavg.model_path)
    fedavg_parts = classify_keys(fedavg_state)
    pure_methods = [m for m in ['pure_vsmc_baseline','pure_vsmc_rank_based_score','pure_vsmc_signal_normalized','pure_vsmc_reliability_floor_eps0.05_freq'] if m in by_method]
    rows = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combo_defs = [
        ('A', 'FedAvg full model', ('base','base','base','base')),
        ('B', 'Pure VSMC full model', ('donor','donor','donor','donor')),
        ('C', 'FedAvg non-expert + VSMC experts', ('base','base','donor','base')),
        ('D', 'VSMC non-expert + FedAvg experts', ('donor','donor','base','donor')),
        ('E', 'VSMC experts + FedAvg head only', ('donor','donor','donor','base')),
        ('F', 'VSMC experts + FedAvg router/head', ('donor','base','donor','base')),
        ('G', 'FedAvg experts + VSMC head/router', ('base','donor','base','donor')),
    ]
    for method in pure_methods:
        spec = by_method[method]
        donor_state = load_state(spec.model_path)
        donor_parts = classify_keys(donor_state)
        for combo_id, combo_name, (shared_src, router_src, expert_src, head_src) in combo_defs:
            merged, counts = merge_components(
                fedavg_state,
                donor_state,
                fedavg_parts,
                donor_parts,
                use_shared=shared_src,
                use_router=router_src,
                use_expert=expert_src,
                use_head=head_src,
            )
            metrics = evaluate_state_dict(merged, spec.data_path, device=device)
            rows.append({
                'reference_method': method,
                'combo_id': combo_id,
                'combo_name': combo_name,
                'top1_acc': format_float(metrics['top1_acc']),
                'top5_acc': format_float(metrics['top5_acc']),
                'test_loss': format_float(metrics['test_loss']),
                'num_parameters_swapped': counts['shared'] + counts['router'] + counts['expert'] + counts['head'],
                'swapped_expert_keys_count': counts['expert'] if expert_src == 'donor' else 0,
                'swapped_head_keys_count': counts['head'] if head_src == 'donor' else 0,
                'swapped_router_keys_count': counts['router'] if router_src == 'donor' else 0,
                'notes': f'shared={shared_src}, router={router_src}, expert={expert_src}, head={head_src}',
            })
    write_csv(out_dir / 'component_swap_eval.csv', rows, [
        'reference_method','combo_id','combo_name','top1_acc','top5_acc','test_loss','num_parameters_swapped','swapped_expert_keys_count','swapped_head_keys_count','swapped_router_keys_count','notes'
    ])
    md = ['# Component Swap Evaluation', '', '| reference | combo | top1 | top5 | loss | swapped expert | swapped router | swapped head | notes |', '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        md.append(f"| {row['reference_method']} | {row['combo_id']} {row['combo_name']} | {row['top1_acc']} | {row['top5_acc']} | {row['test_loss']} | {row['swapped_expert_keys_count']} | {row['swapped_router_keys_count']} | {row['swapped_head_keys_count']} | {row['notes']} |")
    (out_dir / 'component_swap_eval.md').write_text('\n'.join(md) + '\n')
    return rows


def evaluate_top5(specs: List[ExperimentSpec]):
    out_dir = ensure_analysis_root()
    rows = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for spec in specs:
        if not spec.model_path.exists():
            continue
        state = load_state(spec.model_path)
        metrics = evaluate_state_dict(state, spec.data_path, device=device)
        if spec.round_summary_csv is not None:
            round_rows = read_csv(spec.round_summary_csv)
            best_round = None
            if round_rows:
                best_round = max(round_rows, key=lambda row: float(row.get('eval_acc', 0.0))).get('round')
            final_round = round_rows[-1].get('round') if round_rows else 'NA'
        else:
            fedavg_rounds = parse_fedavg_rounds(spec.log_path)
            best_round = max(fedavg_rounds, key=lambda row: row['eval_acc'])['round'] if fedavg_rounds else 'NA'
            final_round = fedavg_rounds[-1]['round'] if fedavg_rounds else 'NA'
        rows.append({
            'method': spec.method,
            'top1_acc': format_float(metrics['top1_acc']),
            'top5_acc': format_float(metrics['top5_acc']),
            'test_loss': format_float(metrics['test_loss']),
            'top5_minus_top1': format_float(metrics['top5_acc'] - metrics['top1_acc']),
            'final_round': final_round,
            'best_round': best_round,
        })
    write_csv(out_dir / 'top5_eval.csv', rows, ['method','top1_acc','top5_acc','test_loss','top5_minus_top1','final_round','best_round'])
    md = ['# Top-5 Evaluation', '', '| method | top1 | top5 | top5-top1 | loss | final round | best round |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        md.append(f"| {row['method']} | {row['top1_acc']} | {row['top5_acc']} | {row['top5_minus_top1']} | {row['test_loss']} | {row['final_round']} | {row['best_round']} |")
    (out_dir / 'top5_eval.md').write_text('\n'.join(md) + '\n')
    return rows


def generate_report(specs: List[ExperimentSpec]):
    out_dir = ensure_analysis_root()
    eff_summary = read_csv(out_dir / 'expert_effective_clients_summary.csv')
    corr_rows = read_csv(out_dir / 'vsmc_next_round_correlation.csv')
    swap_rows = read_csv(out_dir / 'component_swap_eval.csv')
    top5_rows = read_csv(out_dir / 'top5_eval.csv')
    def top5(method):
        for row in top5_rows:
            if row['method'] == method:
                return row
        return None
    fedavg = top5('fedavg')
    rank = top5('pure_vsmc_rank_based_score')
    baseline = top5('pure_vsmc_baseline')
    rel = top5('pure_vsmc_reliability_floor_eps0.05_freq')
    sig = top5('pure_vsmc_signal_normalized')
    lines = ['# Pure VSMC Failure Diagnosis Report', '', '## 1. Summary', '']
    lines.append('Pure VSMC underperforms FedAvg in this CIFAR-100 alpha=0.1 seed3 setting mainly because the expert aggregation becomes too sharp in several variants, while the saved reliability signals do not show a clear positive alignment with next-round global accuracy. The strongest direct evidence comes from the saved `max_client_weight` and `client_weight_entropy` trends, plus the component-swap results that test whether VSMC experts remain compatible with the surrounding non-expert stack.')
    lines += ['', '## 2. Effective Client Analysis', '']
    lines.append('Exact per-expert `a[k,e]` vectors were not persisted in the existing artifacts, so exact `N_eff`, top1 client IDs, and top3 client IDs could not be reconstructed retroactively. However, the saved expert summaries do contain `client_weight_entropy`, `client_max_weight`, and `active_client_count_per_expert`, which are strong collapse diagnostics.')
    for row in eff_summary:
        lines.append(f"- `{row['method']}`: max client weight mean over last 10 rounds = {row['max_client_weight_mean_last10']}, experts with max weight > 0.9 = {row['number_of_experts_with_max_weight_gt_0.9']}, entropy mean last10 = {row['entropy_mean_last10']}.")
    lines += ['', '## 3. Class Coverage Analysis', '']
    lines.append('This hypothesis could not be validated directly from the current logs/checkpoints because top-weight client identities were not saved. Without the per-expert client weight vector, neither top1/top3 client sets nor cumulative-0.8 client sets can be reconstructed exactly. This is a logging limitation rather than a modeling conclusion.')
    lines += ['', '## 4. Next-round Accuracy Correlation', '']
    interesting = [r for r in corr_rows if r['method'] == 'all_pure_vsmc' and r['feature'] in {'mean_max_client_weight','mean_entropy','mean_ess_ratio','reliability_score_mean','reliability_score_std','score_f_std'}]
    for row in interesting:
        lines.append(f"- `{row['feature']}`: pooled pure-VSMC Pearson={row['pearson_corr']}, Spearman={row['spearman_corr']} over {row['num_points']} points.")
    lines.append('If these correlations are weak or directionally inconsistent, the saved reliability statistics are not strongly aligned with next-round global accuracy, which supports the hypothesis that the current reliability score is only a weak proxy for true global benefit.')
    lines += ['', '## 5. Component Compatibility', '']
    for method in ['pure_vsmc_baseline','pure_vsmc_rank_based_score','pure_vsmc_reliability_floor_eps0.05_freq','pure_vsmc_signal_normalized']:
        relevant = [r for r in swap_rows if r['reference_method'] == method]
        if not relevant:
            continue
        lines.append(f"- `{method}` component-swap rows were evaluated against the FedAvg checkpoint. Compare combo `C` (FedAvg non-expert + VSMC experts) and combo `A` (FedAvg full), plus combo `D` (VSMC non-expert + FedAvg experts) and combo `B` (VSMC full), to see whether the expert block itself or the surrounding non-expert stack is the bigger mismatch source.")
    lines += ['', '## 6. Top-5 Accuracy Analysis', '']
    if fedavg:
        lines.append(f"- `fedavg`: top1={fedavg['top1_acc']}, top5={fedavg['top5_acc']}, gap={fedavg['top5_minus_top1']}.")
    for row in [baseline, rank, rel, sig]:
        if row:
            lines.append(f"- `{row['method']}`: top1={row['top1_acc']}, top5={row['top5_acc']}, gap={row['top5_minus_top1']}.")
    lines.append('If a VSMC variant narrows the top5 gap or improves top5 while still losing on top1, that would indicate ranking quality without final-decision quality. If both top1 and top5 are below FedAvg, the problem is broader generalization rather than just top1 calibration.')
    lines += ['', '## 7. Final Diagnosis', '']
    lines.append('1. `expert 权重塌缩` is the strongest supported explanation. Several pure VSMC variants show very high saved `max_client_weight` and very low saved `client_weight_entropy` late in training, which is consistent with near-single-client reliance.')
    lines.append('2. `reliability score 与 global accuracy 不对齐` is also supported if the pooled next-round correlations are weak. That would mean the algorithm can become more confident without becoming more globally useful.')
    lines.append('3. `expert 与 head/router/backbone 不匹配` depends on the component-swap table. If swapping only VSMC experts into the FedAvg stack hurts badly, the expert block itself is likely biased; if the mixed stacks hurt in both directions, compatibility is a bigger issue.')
    lines.append('4. `高权重 client 类别覆盖不足` is plausible but not directly verifiable from the current persisted artifacts because top-weight client identities were not saved.')
    lines.append('5. `function drift 估计噪声` remains a secondary hypothesis here because all run_11 pure variants used the same function-drift setting; this report does not isolate it cleanly.')
    lines.append('6. `CIFAR-100 下 expert 激活过稀疏` remains plausible but is not directly settled by the saved logs used here.')
    lines.append('7. `训练轮数不足` looks weaker than the collapse explanation for baseline-like variants, although rank-based still had some upward momentum late in training.')
    lines += ['', '## 8. Suggested Next Experiments', '']
    lines.append('1. Run a pure-VSMC variant that logs full per-expert client weight vectors every round so `N_eff`, top clients, and class coverage can be measured exactly.')
    lines.append('2. Start from `pure_vsmc_rank_based_score` and add only one anti-collapse regularizer at a time, because it already showed the healthiest entropy / max-weight profile among run_11 variants.')
    lines.append('3. Add a pure diagnostic run that evaluates component compatibility during training, for example by periodically swapping FedAvg vs VSMC expert blocks without introducing any FedAvg blend into the training rule itself.')
    (out_dir / 'pure_vsmc_failure_diagnosis_report.md').write_text('\n'.join(lines) + '\n')


def run_all():
    specs = discover_experiments()
    analyze_effective_clients(specs)
    analyze_class_coverage(specs)
    analyze_next_round_correlation(specs)
    evaluate_component_swap(specs)
    evaluate_top5(specs)
    generate_report(specs)
    return specs
