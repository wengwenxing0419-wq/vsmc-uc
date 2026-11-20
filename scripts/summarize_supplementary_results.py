#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dataset_catalog import format_dataset_notes


ROUND_RE = re.compile(
    r"--round:(?P<round>\d+)\s+--avg_client_train_loss:(?P<avg_train_loss>[0-9.]+)\s+"
    r"--avg_client_train_acc:(?P<avg_train_acc>[0-9.]+)\s+"
    r"--server_test_loss:(?P<server_test_loss>[0-9.]+)\s+"
    r"--server_test_acc:(?P<server_test_acc>[0-9.]+)",
    re.MULTILINE,
)


def parse_standard_log(log_path: Path) -> List[Dict[str, float]]:
    content = log_path.read_text(encoding="utf-8")
    rows = []
    for match in ROUND_RE.finditer(content):
        rows.append(
            {
                "round": int(match.group("round")),
                "server_test_loss": float(match.group("server_test_loss")),
                "server_test_acc": float(match.group("server_test_acc")),
            }
        )
    return rows


def parse_csv_rows(csv_path: Path) -> List[Dict[str, float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "round": int(row["round"]),
                    "server_test_loss": float(row["server_test_loss"]),
                    "server_test_acc": float(row["server_test_acc"]),
                }
            )
    return rows


def load_rows_for_run(run_dir: Path, method: str) -> List[Dict[str, float]]:
    log_dir = run_dir / "result" / "logs"
    if method == "fedjets":
        csv_path = log_dir / "fedjets_round_metrics.csv"
        if csv_path.exists():
            return parse_csv_rows(csv_path)
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return []
    return parse_standard_log(logs[0])


def summarize_rows(rows: List[Dict[str, float]]) -> Optional[Dict[str, object]]:
    if not rows:
        return None
    final_row = rows[-1]
    best_acc_row = max(rows, key=lambda row: row["server_test_acc"])
    best_loss_row = min(rows, key=lambda row: row["server_test_loss"])
    return {
        "final_acc": final_row["server_test_acc"],
        "best_acc": best_acc_row["server_test_acc"],
        "best_round": best_acc_row["round"],
        "final_loss": final_row["server_test_loss"],
        "best_loss": best_loss_row["server_test_loss"],
    }


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_markdown_table(rows: Iterable[Dict[str, object]], columns: List[str]) -> str:
    rows = list(rows)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep] + body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize supplementary experiment outputs.")
    parser.add_argument("--manifest", type=Path, default=Path("runs/supplementary_experiments/supplementary_manifest.json"))
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    summary_rows: List[Dict[str, object]] = []
    for item in manifest:
        run_dir = Path(item["run_dir"])
        metrics = summarize_rows(load_rows_for_run(run_dir, str(item["method"])))
        if metrics is None:
            continue
        notes = "; ".join(format_dataset_notes(str(item["dataset"])))
        summary_rows.append(
            {
                "dataset": item["dataset"],
                "clients": item["clients"],
                "alpha": item["alpha"],
                "backbone": item["backbone"],
                "method": item["method_label"],
                "final_acc": metrics["final_acc"],
                "best_acc": metrics["best_acc"],
                "best_round": metrics["best_round"],
                "final_loss": metrics["final_loss"],
                "best_loss": metrics["best_loss"],
                "total_rounds": item["total_rounds"],
                "local_epochs": item["local_epochs"],
                "batch_size": item["batch_size"],
                "learning_rate": item["learning_rate"],
                "seed": item["seed"],
                "non_expert_aggregation": item["non_expert_aggregation"],
                "expert_aggregation": item["expert_aggregation"],
                "num_experts": item["num_experts"],
                "notes": notes,
                "suite": item["suite"],
                "table_name": item["table_name"],
                "run_dir": str(run_dir),
            }
        )

    output_csv = args.output_csv or args.manifest.with_name("supplementary_results.csv")
    output_md = args.output_md or args.manifest.with_name("supplementary_tables.md")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "clients",
        "alpha",
        "backbone",
        "method",
        "final_acc",
        "best_acc",
        "best_round",
        "final_loss",
        "best_loss",
        "total_rounds",
        "local_epochs",
        "batch_size",
        "learning_rate",
        "seed",
        "non_expert_aggregation",
        "expert_aggregation",
        "num_experts",
        "notes",
        "suite",
        "table_name",
        "run_dir",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["table_name"]), []).append(row)

    sections: List[str] = []
    for table_name, rows in grouped.items():
        sections.append(f"## {table_name}")
        sections.append(
            to_markdown_table(
                rows,
                ["dataset", "clients", "alpha", "backbone", "method", "final_acc", "best_acc", "best_round", "final_loss"],
            )
        )
        sections.append("")

    output_md.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")
    print(f"Wrote CSV summary to {output_csv}")
    print(f"Wrote Markdown tables to {output_md}")


if __name__ == "__main__":
    main()
