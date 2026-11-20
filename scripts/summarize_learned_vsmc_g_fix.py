#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


ROUND_BLOCK_RE = re.compile(
    r"--round:(?P<round>\d+)\s+--avg_client_train_loss:(?P<train_loss>[-+0-9.eE]+)\s+"
    r"--avg_client_train_acc:(?P<train_acc>[-+0-9.eE]+)\s+"
    r"--server_test_loss:(?P<test_loss>[-+0-9.eE]+)\s+--server_test_acc:(?P<test_acc>[-+0-9.eE]+)",
    re.MULTILINE,
)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def try_float(value):
    if value in {None, "", "NA"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "NA"
    return sum(vals) / len(vals)


def parse_rounds(log_text: str):
    rounds = []
    for match in ROUND_BLOCK_RE.finditer(log_text):
        rounds.append(
            {
                "round": int(match.group("round")),
                "train_loss": float(match.group("train_loss")),
                "train_acc": float(match.group("train_acc")),
                "test_loss": float(match.group("test_loss")),
                "test_acc": float(match.group("test_acc")),
            }
        )
    return rounds


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def parse_run(experiment_dir: Path):
    result_dir = experiment_dir / "result"
    logs_dir = result_dir / "logs"
    train_log = next(iter(sorted(logs_dir.glob("*.log"))), None)
    log_text = read_text(train_log) + "\n" + read_text(experiment_dir / "auto_stage_stdout.log")
    rounds = parse_rounds(log_text)

    summary = {
        "experiment": experiment_dir.name,
        "seed": 3,
        "final_acc": "NA",
        "best_acc": "NA",
        "best_acc_round": "NA",
        "min_eval_loss": "NA",
        "min_eval_loss_round": "NA",
        "reward_std_mean": "NA",
        "log_g_std_mean": "NA",
        "log_q_dominates_count": 0,
        "log_g_not_differentiable_count": log_text.count("server validation log_g is not differentiable"),
        "differentiable_g_group_ratio": "NA",
        "best_particle_is_fedavg_rate": "NA",
        "fedavg_particle_alpha_mean": "NA",
    }

    if rounds:
        last = rounds[-1]
        best = max(rounds, key=lambda item: item["test_acc"])
        min_loss = min(rounds, key=lambda item: item["test_loss"])
        summary.update(
            {
                "final_acc": last["test_acc"],
                "best_acc": best["test_acc"],
                "best_acc_round": best["round"],
                "min_eval_loss": min_loss["test_loss"],
                "min_eval_loss_round": min_loss["round"],
            }
        )

    csv_path = logs_dir / "vsmc_aggregation_round_metrics.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
        summary["reward_std_mean"] = mean([try_float(row.get("raw_reward_std")) for row in rows])
        summary["log_g_std_mean"] = mean([try_float(row.get("log_g_std")) for row in rows])
        summary["log_q_dominates_count"] = sum(1 for row in rows if parse_bool(row.get("log_q_dominates")))
        differentiable = [1.0 if parse_bool(row.get("differentiable_g")) else 0.0 for row in rows]
        summary["differentiable_g_group_ratio"] = mean(differentiable)
        summary["best_particle_is_fedavg_rate"] = mean(
            [1.0 if parse_bool(row.get("best_particle_is_fedavg")) else 0.0 for row in rows]
        )
        summary["fedavg_particle_alpha_mean"] = mean([try_float(row.get("fedavg_particle_alpha")) for row in rows])

    return summary


def format_metric(value):
    if value == "NA":
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    experiment_dirs = [
        base_dir / "fedavg_seed3",
        base_dir / "default_old_seed3",
        base_dir / "default_gfix_seed3",
        base_dir / "g_only_selector_seed3",
        base_dir / "loss_zscore_seed3",
    ]
    rows = [parse_run(path) for path in experiment_dirs if path.exists()]

    csv_path = base_dir / "g_fix_summary.csv"
    txt_path = base_dir / "g_fix_summary.txt"

    fieldnames = [
        "experiment",
        "seed",
        "final_acc",
        "best_acc",
        "best_acc_round",
        "min_eval_loss",
        "min_eval_loss_round",
        "reward_std_mean",
        "log_g_std_mean",
        "log_q_dominates_count",
        "log_g_not_differentiable_count",
        "differentiable_g_group_ratio",
        "best_particle_is_fedavg_rate",
        "fedavg_particle_alpha_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = ["Learned VSMC g-fix summary", f"Base dir: {base_dir}", ""]
    for row in rows:
        lines.append(
            f"{row['experiment']}: best_acc={format_metric(row['best_acc'])} "
            f"final_acc={format_metric(row['final_acc'])} "
            f"reward_std_mean={format_metric(row['reward_std_mean'])} "
            f"log_g_std_mean={format_metric(row['log_g_std_mean'])} "
            f"differentiable_g_group_ratio={format_metric(row['differentiable_g_group_ratio'])} "
            f"log_q_dominates_count={format_metric(row['log_q_dominates_count'])}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
