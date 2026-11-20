#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dataset_catalog import get_dataset_spec


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_planned_runs(config: Dict[str, object]):
    defaults = dict(config["defaults"])
    method_defaults = dict(config["method_defaults"])
    for suite_name, suite in dict(config["suites"]).items():
        for run in suite["runs"]:
            base = dict(defaults)
            base.update(run.get("overrides", {}))
            for alpha in run["alphas"]:
                for method_name in run["methods"]:
                    method_cfg = dict(method_defaults[method_name])
                    method_cfg.update(run.get("method_overrides", {}).get(method_name, {}))
                    yield {
                        "suite": suite_name,
                        "dataset": run["dataset"],
                        "clients": int(run["clients"]),
                        "alpha": float(alpha),
                        "backbone": run["backbone"],
                        "method": method_name,
                        "method_label": method_cfg["method_label"],
                        "non_expert_aggregation": base["fedavg_weight_mode"],
                        "expert_aggregation": method_cfg["expert_aggregation"],
                        "num_classes": get_dataset_spec(run["dataset"]).num_classes,
                        "image_size": get_dataset_spec(run["dataset"]).image_size,
                    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for supplementary FL experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/supplementary_experiments.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    rows: List[Dict[str, object]] = list(iter_planned_runs(config))

    issues: List[str] = []
    for row in rows:
        if str(row["non_expert_aggregation"]).lower() != "sample_count":
            issues.append(
                f"{row['suite']} / {row['dataset']} / {row['method']}: non-expert aggregation is not sample_count"
            )
        if int(row["num_classes"]) != int(get_dataset_spec(str(row["dataset"])).num_classes):
            issues.append(f"{row['suite']} / {row['dataset']} / {row['method']}: num_classes mismatch")

    print("Supplementary experiment preflight")
    print("=" * 40)
    for row in rows:
        print(
            f"[ok] suite={row['suite']} dataset={row['dataset']} clients={row['clients']} alpha={row['alpha']} "
            f"backbone={row['backbone']} method={row['method_label']} "
            f"non_expert={row['non_expert_aggregation']} expert={row['expert_aggregation']} "
            f"num_classes={row['num_classes']} image_size={row['image_size']}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump({"rows": rows, "issues": issues}, handle, indent=2, ensure_ascii=False)

    if issues:
        for issue in issues:
            print(f"[issue] {issue}")
        raise SystemExit(1)

    print("All configured supplementary experiments use non-expert=sample_count and valid dataset num_classes.")


if __name__ == "__main__":
    main()
