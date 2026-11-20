#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROUND_RE = re.compile(
    r"--round:(?P<round>\d+)\s+--avg_client_train_loss:(?P<avg_train_loss>[0-9.]+)\s+"
    r"--avg_client_train_acc:(?P<avg_train_acc>[0-9.]+)\s+"
    r"--server_test_loss:(?P<server_test_loss>[0-9.]+)\s+"
    r"--server_test_acc:(?P<server_test_acc>[0-9.]+)",
    re.MULTILINE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three requested reliability ablation experiments.")
    repo_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo_dir", type=Path, default=repo_dir)
    parser.add_argument("--root_dir", type=Path, default=repo_dir / "runs" / "client50_reliability_ablation_resnet18")
    parser.add_argument("--data_name", type=str, default="cifar10")
    parser.add_argument("--num_clients", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--server_epochs", type=int, default=100)
    parser.add_argument("--client_epochs", type=int, default=1)
    parser.add_argument("--backbone_name", type=str, default="resnet18")
    parser.add_argument("--summary_csv", type=Path, default=None)
    return parser.parse_args()


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ensure_seed_data(args: argparse.Namespace, seed: int) -> Path:
    data_dir = args.root_dir / f"seed{seed}" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_files = list(data_dir.glob("*_train.pth"))
    if (data_dir / "server_test.pth").exists() and len(train_files) == int(args.num_clients):
        print(f"[skip] seed{seed} data already exists at {data_dir}", flush=True)
        return data_dir
    cmd = [
        sys.executable,
        str(args.repo_dir / "data.py"),
        "--data_name",
        args.data_name,
        "--num_clients",
        str(args.num_clients),
        "--alpha",
        str(args.alpha),
        "--seed",
        str(seed),
        "--data_save_path",
        str(data_dir),
    ]
    _run_cmd(cmd, cwd=args.repo_dir)
    return data_dir


def _run_single_experiment(
    args: argparse.Namespace,
    seed: int,
    label: str,
    ablation_method: str,
    data_dir: Path,
) -> Path:
    run_dir = args.root_dir / f"seed{seed}" / label
    model_dir = run_dir / "model"
    result_dir = run_dir / "result"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(args.repo_dir / "scripts" / "reliability_ablation_train.py"),
        "--reliability_ablation_method",
        ablation_method,
        "--data_name",
        args.data_name,
        "--num_clients",
        str(args.num_clients),
        "--alpha",
        str(args.alpha),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--server_epochs",
        str(args.server_epochs),
        "--client_epochs",
        str(args.client_epochs),
        "--backbone_name",
        args.backbone_name,
        "--aggregation_method",
        "ha_vsmc_etf",
        "--data_save_path",
        str(data_dir),
        "--model_save_path",
        str(model_dir),
        "--save_result",
        str(result_dir),
        "--ha_vsmc_num_particles",
        "8",
        "--ha_vsmc_lambda_max",
        "0.20",
        "--ha_vsmc_use_validation_loss",
        "false",
        "--ha_vsmc_use_function_drift",
        "true",
    ]
    _run_cmd(cmd, cwd=args.repo_dir)
    return run_dir


def _find_log(run_dir: Path) -> Path:
    log_dir = run_dir / "result" / "logs"
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        raise FileNotFoundError(f"No log files found under {log_dir}")
    return logs[0]


def _parse_final_test_acc(log_path: Path) -> float:
    content = log_path.read_text(encoding="utf-8")
    rows = list(ROUND_RE.finditer(content))
    if not rows:
        raise ValueError(f"No round rows found in {log_path}")
    return float(rows[-1].group("server_test_acc"))


def _mean(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list")
    return sum(values) / float(len(values))


def main() -> None:
    args = _parse_args()
    args.root_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("freq_weighted", "Freq-Weighted", "freq_weighted"),
        ("freq_vsmc", "Freq-VSMC", "freq_vsmc"),
        ("multi_signal_weighted", "Multi-signal Weighted", "multi_signal_weighted"),
    ]

    summary_rows: List[Dict[str, object]] = []
    final_scores: Dict[str, List[float]] = {display_name: [] for _, display_name, _ in experiments}

    for seed in args.seeds:
        data_dir = _ensure_seed_data(args, seed)
        for label, display_name, ablation_method in experiments:
            print(f"[run] seed{seed} / {label}", flush=True)
            run_dir = _run_single_experiment(
                args=args,
                seed=seed,
                label=label,
                ablation_method=ablation_method,
                data_dir=data_dir,
            )
            final_acc = _parse_final_test_acc(_find_log(run_dir))
            final_scores[display_name].append(final_acc)
            summary_rows.append(
                {
                    "seed": int(seed),
                    "label": label,
                    "display_name": display_name,
                    "final_test_acc": float(final_acc),
                    "run_dir": str(run_dir),
                }
            )

    summary_csv = args.summary_csv or (args.root_dir / "reliability_ablation_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "label", "display_name", "final_test_acc", "run_dir"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Freq-Weighted ACC={_mean(final_scores['Freq-Weighted']):.4f}")
    print(f"Freq-VSMC ACC={_mean(final_scores['Freq-VSMC']):.4f}")
    print(f"Multi-signal Weighted ACC={_mean(final_scores['Multi-signal Weighted']):.4f}")


if __name__ == "__main__":
    main()
