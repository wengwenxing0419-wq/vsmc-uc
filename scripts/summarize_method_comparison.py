#!/usr/bin/env python3
"""Summarize FL method logs into a compact comparison table."""

from __future__ import annotations

import argparse
import csv
import os
import re


ROUND_RE = re.compile(
    r"--round:(?P<round>\d+)\s+--avg_client_train_loss:(?P<avg_train_loss>[0-9.]+)\s+"
    r"--avg_client_train_acc:(?P<avg_train_acc>[0-9.]+)\s+"
    r"--server_test_loss:(?P<server_test_loss>[0-9.]+)\s+"
    r"--server_test_acc:(?P<server_test_acc>[0-9.]+)",
    re.MULTILINE,
)


def parse_log(path):
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    rows = []
    for match in ROUND_RE.finditer(content):
        rows.append(
            {
                "round": int(match.group("round")),
                "server_test_loss": float(match.group("server_test_loss")),
                "server_test_acc": float(match.group("server_test_acc")),
            }
        )
    if not rows:
        raise ValueError(f"No round rows found in {path}")
    return rows


def find_log(root, seed, method):
    log_dir = os.path.join(root, f"seed{seed}", method, "result", "logs")
    if not os.path.isdir(log_dir):
        return None
    for filename in sorted(os.listdir(log_dir)):
        if filename.endswith(".log"):
            return os.path.join(log_dir, filename)
    return None


def summarize(path):
    rows = parse_log(path)
    best_acc_row = max(rows, key=lambda row: row["server_test_acc"])
    best_loss_row = min(rows, key=lambda row: row["server_test_loss"])
    final_row = rows[-1]
    return {
        "best_test_acc": best_acc_row["server_test_acc"],
        "final_test_acc": final_row["server_test_acc"],
        "best_test_loss": best_loss_row["server_test_loss"],
        "final_test_loss": final_row["server_test_loss"],
        "best_acc_round": best_acc_row["round"],
        "best_loss_round": best_loss_row["round"],
        "final_round": final_row["round"],
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize FL ablation logs.")
    parser.add_argument("--root", default="runs/client50_method_ablation")
    parser.add_argument("--seeds", nargs="+", default=["1"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fedavg", "fedavg_expert_only"],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for method in args.methods:
            log_path = find_log(args.root, seed, method)
            if log_path is None:
                continue
            row = {"seed": seed, "method": method, "log_path": log_path}
            row.update(summarize(log_path))
            rows.append(row)

    if not rows:
        raise SystemExit("No logs found for the requested root/seeds/methods.")

    fieldnames = [
        "seed",
        "method",
        "best_test_acc",
        "final_test_acc",
        "best_test_loss",
        "final_test_loss",
        "best_acc_round",
        "best_loss_round",
        "final_round",
        "log_path",
    ]
    output = args.output or os.path.join(args.root, "method_comparison.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(",".join(["method", "best_test_acc", "final_test_acc", "best_test_loss", "final_test_loss"]))
    for row in rows:
        print(
            f"{row['method']},{row['best_test_acc']:.4f},{row['final_test_acc']:.4f},"
            f"{row['best_test_loss']:.4f},{row['final_test_loss']:.4f}"
        )
    print(f"Wrote comparison table to: {output}")


if __name__ == "__main__":
    main()
