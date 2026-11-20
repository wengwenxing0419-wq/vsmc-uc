import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.pure_vsmc_failure_lib import evaluate_state_dict, pearson

ROOT = Path('/home/wwx/Vsmc-2/experiments/run_13')
DATA_PATH = ROOT / 'data_cifar100_alpha0.1_10client_seed3'


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = [float(x) for x in xs if x not in ('', 'NA', None)]
    return sum(xs)/len(xs) if xs else float('nan')


def fmt(value):
    if value is None:
        return 'NA'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 'NA'
    if math.isnan(value) or math.isinf(value):
        return 'NA'
    return f'{value:.4f}'


def gather_methods():
    out = []
    for p in sorted(ROOT.glob('pure_vsmc*/seed3')):
        round_csv = p / 'result/logs/ha_vsmc_etf_round_summary.csv'
        if round_csv.exists():
            out.append(p)
    return out


def summarize_method(seed_dir):
    method = seed_dir.parent.name
    round_rows = read_csv(seed_dir / 'result/logs/ha_vsmc_etf_round_summary.csv')
    best_row = max(round_rows, key=lambda r: float(r['eval_acc']))
    final_row = round_rows[-1]
    diag_rows = []
    for diag in sorted((seed_dir / 'result/logs').glob('expert_client_weight_diagnostics_round_*.csv')):
        diag_rows.extend(read_csv(diag))
    by_round = defaultdict(list)
    for row in diag_rows:
        by_round[int(row['round'])].append(row)
    last_rounds = sorted(by_round.keys())[-10:]
    tail = [row for rnd in last_rounds for row in by_round[rnd]]
    top1 = evaluate_state_dict(
        __import__('torch').load(seed_dir / 'model/server.pth', map_location='cpu'),
        DATA_PATH,
    ) if (seed_dir / 'model/server.pth').exists() else {'top5_acc': float('nan')}
    xs, ys = [], []
    for i in range(len(round_rows)-1):
        rnd = int(round_rows[i]['round'])
        if rnd not in by_round:
            continue
        xs.append(sum(float(r.get('reliability_score_mean', 0.0)) for r in by_round[rnd]) / max(len(by_round[rnd]), 1))
        ys.append(float(round_rows[i+1]['eval_acc']) - float(round_rows[i]['eval_acc']))
    proxy_xs = []
    for i in range(len(round_rows)-1):
        rnd = int(round_rows[i]['round'])
        if rnd not in by_round:
            continue
        proxy_xs.append(sum(float(r.get('proxy_val_loss_delta', 0.0)) for r in by_round[rnd]) / max(len(by_round[rnd]), 1))
    return {
        'method': method,
        'best_acc': float(best_row['eval_acc']),
        'final_acc': float(final_row['eval_acc']),
        'best_round': int(best_row['round']),
        'final_round': int(final_row['round']),
        'best_loss': float(best_row['eval_loss']),
        'final_loss': float(final_row['eval_loss']),
        'top5_acc_final': float(top1['top5_acc']),
        'pure_vsmc': final_row.get('pure_vsmc', 'True'),
        'fedavg_blend_enabled': final_row.get('fedavg_blend_enabled', 'False'),
        'fallback_count_total': sum(int(float(r.get('fallback_count', 0) or 0)) for r in round_rows),
        'fallback_type': 'none',
        'N_eff_mean_last10': mean([r['N_eff'] for r in tail]),
        'N_eff_min_last10': min((float(r['N_eff']) for r in tail if r['N_eff'] not in ('', 'NA')), default=float('nan')),
        'entropy_mean_last10': mean([r['entropy'] for r in tail]),
        'max_client_weight_mean_last10': mean([r['max_client_weight'] for r in tail]),
        'experts_max_weight_gt_0.8_last10': sum(1 for r in tail if float(r['max_client_weight']) > 0.8),
        'experts_max_weight_gt_0.9_last10': sum(1 for r in tail if float(r['max_client_weight']) > 0.9),
        'class_coverage_top1_mean_last10': mean([r['class_coverage_top1'] for r in tail]),
        'class_coverage_top3_mean_last10': mean([r['class_coverage_top3'] for r in tail]),
        'class_coverage_cum80_mean_last10': mean([r['class_coverage_cum80'] for r in tail]),
        'reliability_score_next_acc_corr': pearson(xs, ys) if len(xs) == len(ys) and len(xs) > 1 else float('nan'),
        'proxy_gain_next_acc_corr': pearson(proxy_xs, ys) if len(proxy_xs) == len(ys) and len(proxy_xs) > 1 else float('nan'),
        'notes': '',
    }, tail


def main():
    methods = gather_methods()
    summaries = []
    anti_rows = []
    class_rows = []
    corr_rows = []
    for method_dir in methods:
        summary, tail = summarize_method(method_dir)
        summaries.append(summary)
        anti_rows.append({k: summary[k] for k in ['method','N_eff_mean_last10','N_eff_min_last10','entropy_mean_last10','max_client_weight_mean_last10','experts_max_weight_gt_0.8_last10','experts_max_weight_gt_0.9_last10']})
        class_rows.append({k: summary[k] for k in ['method','class_coverage_top1_mean_last10','class_coverage_top3_mean_last10','class_coverage_cum80_mean_last10']})
        corr_rows.append({k: summary[k] for k in ['method','reliability_score_next_acc_corr','proxy_gain_next_acc_corr']})
    ROOT.mkdir(parents=True, exist_ok=True)
    def write(path, rows):
        if not rows:
            return
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    write(ROOT / 'current_summary.csv', summaries)
    write(ROOT / 'anti_collapse_diagnosis.csv', anti_rows)
    write(ROOT / 'class_coverage_diagnosis.csv', class_rows)
    write(ROOT / 'reliability_global_gain_correlation.csv', corr_rows)
    lines = ['# Run 13 Summary', '', '| method | best acc | final acc | best round | top5 final | max weight last10 | entropy last10 | rel corr | proxy corr |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['method']} | {fmt(row['best_acc'])} | {fmt(row['final_acc'])} | {row['best_round']} | {fmt(row['top5_acc_final'])} | {fmt(row['max_client_weight_mean_last10'])} | {fmt(row['entropy_mean_last10'])} | {fmt(row['reliability_score_next_acc_corr'])} | {fmt(row['proxy_gain_next_acc_corr'])} |")
    (ROOT / 'run_13_summary.md').write_text('\n'.join(lines) + '\n')
    report = ['# Run 13 Report', '', 'Use `current_summary.csv`, `anti_collapse_diagnosis.csv`, `class_coverage_diagnosis.csv`, and `reliability_global_gain_correlation.csv` as the main machine-readable outputs for this run.', '']
    if summaries:
        best = max(summaries, key=lambda r: r['best_acc'])
        stable = max(summaries, key=lambda r: r['final_acc'])
        report.append(f"Best pure VSMC variant so far: `{best['method']}` with best_acc={best['best_acc']:.4f}.")
        report.append(f"Most stable final_acc variant: `{stable['method']}` with final_acc={stable['final_acc']:.4f}.")
    (ROOT / 'run_13_report.md').write_text('\n'.join(report) + '\n')

if __name__ == '__main__':
    main()
