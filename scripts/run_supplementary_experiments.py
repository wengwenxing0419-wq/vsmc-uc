#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_run_specs(config: Dict[str, object], suites: Iterable[str]) -> Iterable[Dict[str, object]]:
    defaults = dict(config["defaults"])
    method_defaults = dict(config["method_defaults"])
    selected = set(suites) if suites else set(config["suites"].keys())
    for suite_name, suite in dict(config["suites"]).items():
        if suite_name not in selected:
            continue
        for run in suite["runs"]:
            base = dict(defaults)
            base.update(run.get("overrides", {}))
            for alpha in run["alphas"]:
                for method_name in run["methods"]:
                    method_cfg = dict(method_defaults[method_name])
                    method_cfg.update(run.get("method_overrides", {}).get(method_name, {}))
                    spec = dict(base)
                    spec.update(method_cfg)
                    spec.update(
                        {
                            "suite": suite_name,
                            "table_name": suite["table_name"],
                            "dataset": run["dataset"],
                            "clients": int(run["clients"]),
                            "alpha": float(alpha),
                            "backbone": run["backbone"],
                            "method": method_name,
                        }
                    )
                    yield spec


def ensure_seed_data(repo_dir: Path, root_dir: Path, spec: Dict[str, object], seed: int, dry_run: bool) -> Path:
    data_dir = root_dir / spec["suite"] / f"{spec['dataset']}_c{spec['clients']}_a{spec['alpha']}" / f"seed{seed}" / "data"
    train_files = list(data_dir.glob("*_train.pth"))
    needs_regen = not (data_dir / "server_test.pth").exists() or len(train_files) != int(spec["clients"])
    if not needs_regen:
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_dir / "data.py"),
        "--data_name",
        str(spec["dataset"]),
        "--data_path",
        str(spec["data_path"]),
        "--num_clients",
        str(spec["clients"]),
        "--alpha",
        str(spec["alpha"]),
        "--seed",
        str(seed),
        "--data_save_path",
        str(data_dir),
    ]
    run_cmd(cmd, repo_dir, dry_run=dry_run)
    return data_dir


def build_train_cmd(repo_dir: Path, spec: Dict[str, object], seed: int, data_dir: Path, run_dir: Path) -> List[str]:
    entry = repo_dir / str(spec["entry"])
    cmd = [
        sys.executable,
        str(entry),
        "--data_name",
        str(spec["dataset"]),
        "--data_path",
        str(spec["data_path"]),
        "--num_clients",
        str(spec["clients"]),
        "--alpha",
        str(spec["alpha"]),
        "--seed",
        str(seed),
        "--device",
        str(spec["device"]),
        "--server_epochs",
        str(spec["server_epochs"]),
        "--client_epochs",
        str(spec["client_epochs"]),
        "--batch_size",
        str(spec["batch_size"]),
        "--learning_rate",
        str(spec["learning_rate"]),
        "--client_optimizer",
        str(spec["client_optimizer"]),
        "--backbone_name",
        str(spec["backbone"]),
        "--patch_size",
        str(spec["patch_size"]),
        "--hidden_dim",
        str(spec["hidden_dim"]),
        "--switch_layers",
        str(spec["switch_layers"]),
        "--switch_heads",
        str(spec["switch_heads"]),
        "--switch_ffn_dim",
        str(spec["switch_ffn_dim"]),
        "--dropout",
        str(spec["dropout"]),
        "--num_experts",
        str(spec["num_experts"]),
        "--switch_num_experts",
        str(spec["switch_num_experts"]),
        "--fedavg_weight_mode",
        str(spec["fedavg_weight_mode"]),
        "--expert_weight_mode",
        str(spec["expert_weight_mode"]),
        "--aggregation_method",
        str(spec["aggregation_method"]),
        "--data_save_path",
        str(data_dir),
        "--model_save_path",
        str(run_dir / "model"),
        "--save_result",
        str(run_dir / "result"),
    ]
    optional_pairs: List[Tuple[str, str]] = [
        ("reliability_ablation_method", "--reliability_ablation_method"),
        ("fedjets_num_experts", "--fedjets_num_experts"),
        ("fedjets_top_k", "--fedjets_top_k"),
        ("fedjets_num_anchor_clients", "--fedjets_num_anchor_clients"),
        ("fedjets_anchor_clients_per_round", "--fedjets_anchor_clients_per_round"),
        ("fedjets_normal_clients_per_round", "--fedjets_normal_clients_per_round"),
    ]
    for key, flag in optional_pairs:
        if key in spec:
            cmd.extend([flag, str(spec[key])])
    return cmd


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch supplementary Switch-MoE / FedETF experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/supplementary_experiments.json"))
    parser.add_argument("--root_dir", type=Path, default=Path("runs/supplementary_experiments"))
    parser.add_argument("--suites", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--write_manifest", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    seeds = args.seeds or [int(config["defaults"]["seed"])]
    manifest_rows: List[Dict[str, object]] = []

    for spec in iter_run_specs(config, args.suites):
        for seed in seeds:
            data_dir = ensure_seed_data(repo_dir, args.root_dir, spec, seed, dry_run=args.dry_run)
            run_dir = (
                args.root_dir
                / spec["suite"]
                / f"{spec['dataset']}_c{spec['clients']}_a{spec['alpha']}"
                / f"backbone_{spec['backbone']}"
                / f"seed{seed}"
                / spec["method"]
            )
            (run_dir / "model").mkdir(parents=True, exist_ok=True)
            (run_dir / "result").mkdir(parents=True, exist_ok=True)
            cmd = build_train_cmd(repo_dir, spec, seed, data_dir, run_dir)
            run_cmd(cmd, repo_dir, dry_run=args.dry_run)
            manifest_rows.append(
                {
                    "suite": spec["suite"],
                    "table_name": spec["table_name"],
                    "dataset": spec["dataset"],
                    "clients": spec["clients"],
                    "alpha": spec["alpha"],
                    "backbone": spec["backbone"],
                    "method": spec["method"],
                    "method_label": spec["method_label"],
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "non_expert_aggregation": spec["fedavg_weight_mode"],
                    "expert_aggregation": spec["expert_aggregation"],
                    "num_experts": spec["switch_num_experts"],
                    "batch_size": spec["batch_size"],
                    "learning_rate": spec["learning_rate"],
                    "local_epochs": spec["client_epochs"],
                    "total_rounds": spec["server_epochs"],
                }
            )

    if args.write_manifest or args.dry_run:
        manifest_path = args.root_dir / "supplementary_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest_rows, handle, indent=2, ensure_ascii=False)
        print(f"[manifest] wrote {manifest_path}")


if __name__ == "__main__":
    main()
