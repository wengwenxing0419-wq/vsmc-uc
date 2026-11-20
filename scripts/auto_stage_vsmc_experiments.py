#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"
DATA_PY = ROOT / "data.py"
ARGS_PY = ROOT / "args.py"
POLL_SECONDS = 5.0
IMPROVEMENT_MARGIN = 0.001

WARNING_PATTERNS = {
    "nan_warning_count": re.compile(r"\bnan\b|\bNaN\b"),
    "global_not_learning_count": re.compile(r"global model not learning"),
    "log_q_dominates_count": re.compile(r"log_q still dominates log_w"),
    "log_g_not_differentiable_count": re.compile(r"server validation log_g is not differentiable"),
    "fallback_warning_count": re.compile(r"fallback", re.IGNORECASE),
}

ROUND_BLOCK_RE = re.compile(
    r"--round:(?P<round>\d+)\s+--avg_client_train_loss:(?P<train_loss>[-+0-9.eE]+)\s+"
    r"--avg_client_train_acc:(?P<train_acc>[-+0-9.eE]+)\s+"
    r"--server_test_loss:(?P<test_loss>[-+0-9.eE]+)\s+--server_test_acc:(?P<test_acc>[-+0-9.eE]+)",
    re.MULTILINE,
)


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def try_float(value):
    if value in {None, "", "NA"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def mean_or_na(values: Sequence[Optional[float]]):
    nums = [v for v in values if v is not None]
    if not nums:
        return "NA"
    return sum(nums) / len(nums)


def std_or_na(values: Sequence[Optional[float]]):
    nums = [v for v in values if v is not None]
    if not nums:
        return "NA"
    if len(nums) == 1:
        return 0.0
    return statistics.pstdev(nums)


def format_metric(value) -> str:
    if value == "NA" or value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def discover_arg_support(args_py_path: Path) -> Tuple[set, Dict[str, List[str]]]:
    text = read_text_if_exists(args_py_path)
    add_argument_blocks = re.findall(r"parse\.add_argument\((.*?)\)\s*", text, re.DOTALL)
    flags = set()
    choices: Dict[str, List[str]] = {}
    for block in add_argument_blocks:
        flag_match = re.search(r"['\"](?P<flag>--[^'\"]+)['\"]", block)
        if not flag_match:
            continue
        flag = flag_match.group("flag")
        flags.add(flag)
        choices_match = re.search(r"choices\s*=\s*\[(?P<choices>[^\]]*)\]", block, re.DOTALL)
        if choices_match:
            raw_choices = choices_match.group("choices")
            parsed = [item.strip().strip('"').strip("'") for item in raw_choices.split(',') if item.strip()]
            choices[flag] = parsed
    return flags, choices


@dataclass
class ExperimentSpec:
    stage: str
    experiment_name: str
    extra_args: List[str]
    required_flags: List[str] = field(default_factory=list)
    required_choices: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentTask:
    spec: ExperimentSpec
    seed: int
    output_dir: Path
    data_dir: Path
    model_dir: Path
    result_dir: Path
    command: List[str] = field(default_factory=list)
    status: str = "pending"
    skip_reason: str = ""
    return_code: Optional[int] = None
    gpu_id: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    stdout_log: Optional[Path] = None
    command_txt: Optional[Path] = None
    parsed_result: Optional[Dict[str, object]] = None


def base_train_args(cli_args, seed: int, data_dir: Path, model_dir: Path, result_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(TRAIN_PY),
        "--data_name",
        cli_args.data_name,
        "--num_clients",
        str(cli_args.num_clients),
        "--alpha",
        str(cli_args.alpha),
        "--seed",
        str(seed),
        "--device",
        cli_args.device,
        "--server_epochs",
        str(cli_args.num_rounds),
        "--client_epochs",
        str(cli_args.client_epochs),
        "--backbone_name",
        cli_args.backbone_name,
        "--data_save_path",
        str(data_dir),
        "--model_save_path",
        str(model_dir),
        "--save_result",
        str(result_dir),
    ]


def build_stage_specs() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            stage="stage1",
            experiment_name="fedavg",
            extra_args=["--aggregation_method", "fedavg"],
        ),
        ExperimentSpec(
            stage="stage1",
            experiment_name="learned_vsmc_default",
            extra_args=[
                "--aggregation_method",
                "learned_vsmc",
                "--learned_vsmc_mode",
                "default",
                "--learned_vsmc_logq_coef",
                "0.1",
                "--learned_vsmc_normalize_reward",
                "true",
                "--learned_vsmc_include_fedavg_particle",
                "true",
                "--learned_vsmc_mix_fedavg_candidate",
                "0.2",
                "--learned_vsmc_mix_uniform",
                "0.2",
                "--learned_vsmc_alpha_max",
                "0.7",
            ],
            required_flags=[
                "--aggregation_method",
                "--learned_vsmc_mode",
                "--learned_vsmc_logq_coef",
                "--learned_vsmc_normalize_reward",
                "--learned_vsmc_include_fedavg_particle",
                "--learned_vsmc_mix_fedavg_candidate",
                "--learned_vsmc_mix_uniform",
                "--learned_vsmc_alpha_max",
            ],
            required_choices={"--learned_vsmc_mode": "default"},
        ),
        ExperimentSpec(
            stage="stage1",
            experiment_name="server_reward_selector",
            extra_args=[
                "--aggregation_method",
                "learned_vsmc",
                "--learned_vsmc_mode",
                "server_reward_selector",
                "--learned_vsmc_logq_coef",
                "0.0",
                "--learned_vsmc_normalize_reward",
                "true",
                "--learned_vsmc_include_fedavg_particle",
                "true",
                "--learned_vsmc_mix_fedavg_candidate",
                "0.2",
                "--learned_vsmc_mix_uniform",
                "0.1",
                "--learned_vsmc_alpha_max",
                "0.7",
            ],
            required_flags=[
                "--aggregation_method",
                "--learned_vsmc_mode",
                "--learned_vsmc_logq_coef",
                "--learned_vsmc_normalize_reward",
                "--learned_vsmc_include_fedavg_particle",
                "--learned_vsmc_mix_fedavg_candidate",
                "--learned_vsmc_mix_uniform",
                "--learned_vsmc_alpha_max",
            ],
            required_choices={"--learned_vsmc_mode": "server_reward_selector"},
        ),
        ExperimentSpec(
            stage="stage2",
            experiment_name="reward_selector_router_calib",
            extra_args=[
                "--aggregation_method",
                "learned_vsmc",
                "--learned_vsmc_mode",
                "server_reward_selector",
                "--learned_vsmc_logq_coef",
                "0.0",
                "--learned_vsmc_normalize_reward",
                "true",
                "--learned_vsmc_include_fedavg_particle",
                "true",
                "--learned_vsmc_mix_fedavg_candidate",
                "0.2",
                "--learned_vsmc_mix_uniform",
                "0.1",
                "--learned_vsmc_alpha_max",
                "0.7",
                "--vsmc_use_router_usage_calibration",
                "true",
                "--vsmc_router_bias_lr",
                "0.01",
            ],
            required_flags=[
                "--aggregation_method",
                "--learned_vsmc_mode",
                "--vsmc_use_router_usage_calibration",
                "--vsmc_router_bias_lr",
            ],
            required_choices={"--learned_vsmc_mode": "server_reward_selector"},
        ),
        ExperimentSpec(
            stage="stage3",
            experiment_name="reward_selector_router_ft",
            extra_args=[
                "--aggregation_method",
                "learned_vsmc",
                "--learned_vsmc_mode",
                "server_reward_selector",
                "--learned_vsmc_logq_coef",
                "0.0",
                "--learned_vsmc_normalize_reward",
                "true",
                "--learned_vsmc_include_fedavg_particle",
                "true",
                "--learned_vsmc_mix_fedavg_candidate",
                "0.2",
                "--learned_vsmc_mix_uniform",
                "0.1",
                "--learned_vsmc_alpha_max",
                "0.7",
                "--use_server_router_finetune",
                "true",
                "--router_finetune_steps",
                "1",
                "--router_finetune_lr",
                "1e-4",
                "--router_finetune_batches",
                "1",
            ],
            required_flags=[
                "--aggregation_method",
                "--learned_vsmc_mode",
                "--use_server_router_finetune",
                "--router_finetune_steps",
                "--router_finetune_lr",
                "--router_finetune_batches",
            ],
            required_choices={"--learned_vsmc_mode": "server_reward_selector"},
        ),
        ExperimentSpec(
            stage="stage4",
            experiment_name="best_plus_alignment",
            extra_args=[],
            required_flags=[
                "--use_expert_alignment",
                "--alignment_metric",
                "--alignment_batch_size",
                "--alignment_update_every",
            ],
        ),
    ]


def build_alignment_extra_args() -> List[str]:
    return [
        "--use_expert_alignment",
        "true",
        "--alignment_metric",
        "output_cosine",
        "--alignment_batch_size",
        "128",
        "--alignment_update_every",
        "1",
    ]


def make_task(cli_args, spec: ExperimentSpec, seed: int, flags: set, choices: Dict[str, List[str]], inherited_extra_args: Optional[List[str]] = None) -> ExperimentTask:
    output_dir = Path(cli_args.base_output_dir).resolve() / f"{spec.stage}_{spec.experiment_name}_seed{seed}"
    data_dir = Path(cli_args.base_output_dir).resolve() / f"shared_data_seed{seed}"
    model_dir = output_dir / "model"
    result_dir = output_dir / "result"
    ensure_dir(output_dir)
    ensure_dir(model_dir)
    ensure_dir(result_dir)

    task = ExperimentTask(
        spec=spec,
        seed=seed,
        output_dir=output_dir,
        data_dir=data_dir,
        model_dir=model_dir,
        result_dir=result_dir,
    )

    missing_flags = [flag for flag in spec.required_flags if flag not in flags]
    if missing_flags:
        task.status = "skipped"
        task.skip_reason = f"required args not implemented: missing {', '.join(missing_flags)}"
        return task

    for flag, expected_choice in spec.required_choices.items():
        available_choices = choices.get(flag)
        if available_choices is None or expected_choice not in available_choices:
            task.status = "skipped"
            task.skip_reason = f"required args not implemented: {flag} lacks choice {expected_choice}"
            return task

    command = base_train_args(cli_args, seed, data_dir, model_dir, result_dir)
    if inherited_extra_args:
        command.extend(inherited_extra_args)
    else:
        command.extend(spec.extra_args)
    task.command = command
    task.stdout_log = output_dir / "auto_stage_stdout.log"
    task.command_txt = output_dir / "command.txt"
    return task


def prepare_data_for_seed(cli_args, seed: int, data_dir: Path, base_output_dir: Path) -> Dict[str, object]:
    ensure_dir(data_dir)
    log_path = base_output_dir / f"data_seed{seed}_stdout.log"
    command = [
        sys.executable,
        str(DATA_PY),
        "--data_name",
        cli_args.data_name,
        "--num_clients",
        str(cli_args.num_clients),
        "--alpha",
        str(cli_args.alpha),
        "--seed",
        str(seed),
        "--data_save_path",
        str(data_dir),
    ]
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return {
        "seed": seed,
        "status": "completed" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "log_path": str(log_path),
        "command": shlex.join(command),
    }


def launch_task(task: ExperimentTask, gpu_id: str) -> None:
    ensure_dir(task.output_dir)
    ensure_dir(task.model_dir)
    ensure_dir(task.result_dir)
    task.gpu_id = gpu_id
    task.status = "running"
    assert task.stdout_log is not None
    assert task.command_txt is not None
    task.command_txt.write_text(shlex.join(task.command) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    stdout_handle = open(task.stdout_log, "w", encoding="utf-8")
    stdout_handle.write(f"CUDA_VISIBLE_DEVICES={gpu_id} {shlex.join(task.command)}\n")
    stdout_handle.flush()
    task.process = subprocess.Popen(
        task.command,
        cwd=str(ROOT),
        env=env,
        stdout=stdout_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    task._stdout_handle = stdout_handle  # type: ignore[attr-defined]


def finalize_task(task: ExperimentTask) -> None:
    if task.process is None:
        return
    return_code = task.process.poll()
    if return_code is None:
        return
    task.return_code = int(return_code)
    task.status = "completed" if return_code == 0 else "failed"
    stdout_handle = getattr(task, "_stdout_handle", None)
    if stdout_handle is not None and not stdout_handle.closed:
        stdout_handle.flush()
        stdout_handle.close()
    task.process = None


def run_stage(tasks: List[ExperimentTask], gpu_ids: List[str], max_parallel_per_gpu: int, num_rounds: int, base_output_dir: Path, stage_name: str) -> None:
    pending = [task for task in tasks if task.status == "pending"]
    running: List[ExperimentTask] = []
    capacity = {gpu_id: max_parallel_per_gpu for gpu_id in gpu_ids}
    last_progress_print = 0.0
    _print_stage_progress(stage_name, tasks, num_rounds, base_output_dir)
    while pending or running:
        for task in list(running):
            finalize_task(task)
            if task.process is None:
                running.remove(task)
                if task.gpu_id is not None:
                    capacity[task.gpu_id] += 1

        launched_any = False
        for task in list(pending):
            gpu_choice = next((gpu for gpu, free_slots in capacity.items() if free_slots > 0), None)
            if gpu_choice is None:
                break
            launch_task(task, gpu_choice)
            capacity[gpu_choice] -= 1
            running.append(task)
            pending.remove(task)
            launched_any = True
            print(f"launch {stage_name}/{task.spec.experiment_name} seed={task.seed} on gpu {gpu_choice}", flush=True)

        now = time.time()
        if launched_any or now - last_progress_print >= POLL_SECONDS or not running:
            _print_stage_progress(stage_name, tasks, num_rounds, base_output_dir)
            last_progress_print = now

        if pending or running:
            time.sleep(1.0 if launched_any else POLL_SECONDS)

    for task in tasks:
        if task.status == "running":
            finalize_task(task)
    _print_stage_progress(stage_name, tasks, num_rounds, base_output_dir)


def _parse_live_round_from_text(log_text: str) -> Dict[str, object]:
    rounds = parse_round_metrics_from_log(log_text)
    if not rounds:
        return {"round": 0, "acc": "NA", "loss": "NA"}
    last = rounds[-1]
    return {"round": int(last["round"]), "acc": float(last["test_acc"]), "loss": float(last["test_loss"])}


def _task_live_progress(task: ExperimentTask, num_rounds: int) -> Dict[str, object]:
    info = {
        "stage": task.spec.stage,
        "experiment_name": task.spec.experiment_name,
        "seed": task.seed,
        "status": task.status,
        "gpu_id": task.gpu_id if task.gpu_id is not None else "NA",
        "round": 0,
        "num_rounds": num_rounds,
        "progress": 0.0,
        "latest_acc": "NA",
        "latest_loss": "NA",
        "skip_reason": task.skip_reason,
    }
    if task.stdout_log is None or not task.stdout_log.exists():
        return info
    log_text = read_text_if_exists(task.stdout_log)
    live = _parse_live_round_from_text(log_text)
    current_round = int(live["round"])
    info["round"] = current_round
    info["progress"] = min(max(current_round / max(num_rounds, 1), 0.0), 1.0)
    info["latest_acc"] = live["acc"]
    info["latest_loss"] = live["loss"]
    return info


def _print_stage_progress(stage_name: str, tasks: List[ExperimentTask], num_rounds: int, base_output_dir: Path) -> None:
    snapshot = {
        "stage": stage_name,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "tasks": [],
    }
    lines = [f"[{snapshot['updated_at']}] {stage_name} progress"]
    for task in tasks:
        info = _task_live_progress(task, num_rounds)
        snapshot["tasks"].append(info)
        lines.append(
            f"  - {task.spec.experiment_name} seed={task.seed} status={task.status} gpu={info['gpu_id']} "
            f"round={info['round']}/{num_rounds} progress={info['progress'] * 100:.1f}% "
            f"acc={format_metric(info['latest_acc'])} loss={format_metric(info['latest_loss'])}"
            + (f" skip_reason={task.skip_reason}" if task.status == 'skipped' and task.skip_reason else "")
        )
    print("\n".join(lines), flush=True)
    (base_output_dir / "live_status.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_primary_train_log(result_dir: Path) -> Optional[Path]:
    logs_dir = result_dir / "logs"
    if not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("*.log"))
    return candidates[0] if candidates else None


def parse_round_metrics_from_log(log_text: str) -> List[Dict[str, float]]:
    rounds = []
    for match in ROUND_BLOCK_RE.finditer(log_text):
        rounds.append(
            {
                "round": float(match.group("round")),
                "train_loss": float(match.group("train_loss")),
                "train_acc": float(match.group("train_acc")),
                "test_loss": float(match.group("test_loss")),
                "test_acc": float(match.group("test_acc")),
            }
        )
    return rounds


def parse_vsmc_csv(csv_path: Path) -> Dict[str, object]:
    metrics = {
        "mean_normalized_ess_mean": "NA",
        "mean_normalized_ess_last": "NA",
        "proposal_grad_norm_mean": "NA",
        "proposal_param_delta_norm_mean": "NA",
        "reward_std_mean": "NA",
        "fedavg_delta_norm_mean": "NA",
        "particle_delta_norm_mean": "NA",
        "best_particle_is_fedavg_rate": "NA",
        "fallback_count": "NA",
    }
    if not csv_path.exists():
        return metrics

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        return metrics

    normalized_ess = [try_float(row.get("normalized_ess")) for row in rows]
    proposal_grad_norm = [try_float(row.get("proposal_grad_norm")) for row in rows]
    proposal_param_delta_norm = [try_float(row.get("proposal_param_delta_norm")) for row in rows]
    reward_std = [try_float(row.get("raw_reward_std")) for row in rows]
    fedavg_delta_norm = [try_float(row.get("fedavg_delta_norm")) for row in rows]
    particle_delta_norm = [try_float(row.get("particle_delta_norm_mean")) for row in rows]
    best_particle_is_fedavg = []
    fallback_count = 0
    max_round = max(int(float(row.get("round", 0) or 0)) for row in rows)
    last_round_ess = []

    for row in rows:
        fedavg_flag = str(row.get("best_particle_is_fedavg", "")).strip().lower()
        if fedavg_flag in {"true", "1"}:
            best_particle_is_fedavg.append(1.0)
        elif fedavg_flag in {"false", "0"}:
            best_particle_is_fedavg.append(0.0)
        value = str(row.get("fallback_used", "")).strip().lower()
        if value in {"true", "1"}:
            fallback_count += 1
        round_id = int(float(row.get("round", 0) or 0))
        if round_id == max_round:
            last_round_ess.append(try_float(row.get("normalized_ess")))

    metrics.update(
        {
            "mean_normalized_ess_mean": mean_or_na(normalized_ess),
            "mean_normalized_ess_last": mean_or_na(last_round_ess),
            "proposal_grad_norm_mean": mean_or_na(proposal_grad_norm),
            "proposal_param_delta_norm_mean": mean_or_na(proposal_param_delta_norm),
            "reward_std_mean": mean_or_na(reward_std),
            "fedavg_delta_norm_mean": mean_or_na(fedavg_delta_norm),
            "particle_delta_norm_mean": mean_or_na(particle_delta_norm),
            "best_particle_is_fedavg_rate": mean_or_na(best_particle_is_fedavg),
            "fallback_count": fallback_count,
        }
    )
    return metrics


def parse_result(task: ExperimentTask) -> Dict[str, object]:
    result = {
        "stage": task.spec.stage,
        "experiment_name": task.spec.experiment_name,
        "seed": task.seed,
        "status": task.status,
        "return_code": task.return_code if task.return_code is not None else "NA",
        "output_dir": str(task.output_dir),
        "final_acc": "NA",
        "best_acc": "NA",
        "best_acc_round": "NA",
        "final_eval_loss": "NA",
        "min_eval_loss": "NA",
        "min_eval_loss_round": "NA",
        "acc_at_min_eval_loss": "NA",
        "final_train_loss": "NA",
        "final_train_acc": "NA",
        "log_q_dominates_count": 0,
        "log_g_not_differentiable_count": 0,
        "mean_normalized_ess_mean": "NA",
        "mean_normalized_ess_last": "NA",
        "proposal_grad_norm_mean": "NA",
        "proposal_param_delta_norm_mean": "NA",
        "reward_std_mean": "NA",
        "fedavg_delta_norm_mean": "NA",
        "particle_delta_norm_mean": "NA",
        "best_particle_is_fedavg_rate": "NA",
        "fallback_count": "NA",
        "nan_warning_count": 0,
        "global_not_learning_count": 0,
        "fallback_warning_count": 0,
        "warning_count_total": 0,
        "skip_reason": task.skip_reason or "",
    }

    log_text = ""
    if task.stdout_log is not None:
        log_text += read_text_if_exists(task.stdout_log)
    primary_log = find_primary_train_log(task.result_dir)
    if primary_log is not None:
        log_text += "\n" + read_text_if_exists(primary_log)

    rounds = parse_round_metrics_from_log(log_text)
    if rounds:
        best = max(rounds, key=lambda item: item["test_acc"])
        min_loss = min(rounds, key=lambda item: item["test_loss"])
        last = rounds[-1]
        result.update(
            {
                "final_acc": last["test_acc"],
                "best_acc": best["test_acc"],
                "best_acc_round": int(best["round"]),
                "final_eval_loss": last["test_loss"],
                "min_eval_loss": min_loss["test_loss"],
                "min_eval_loss_round": int(min_loss["round"]),
                "acc_at_min_eval_loss": min_loss["test_acc"],
                "final_train_loss": last["train_loss"],
                "final_train_acc": last["train_acc"],
            }
        )

    for key, pattern in WARNING_PATTERNS.items():
        result[key] = len(pattern.findall(log_text))
    result["warning_count_total"] = sum(int(result[key]) for key in WARNING_PATTERNS)

    vsmc_metrics = parse_vsmc_csv(task.result_dir / "logs" / "vsmc_aggregation_round_metrics.csv")
    result.update(vsmc_metrics)
    return result


def aggregate_experiment_results(results: List[Dict[str, object]], experiment_name: str) -> Dict[str, object]:
    subset = [item for item in results if item["experiment_name"] == experiment_name and item["status"] == "completed"]
    summary = {
        "experiment_name": experiment_name,
        "num_completed": len(subset),
        "mean_best_acc": "NA",
        "std_best_acc": "NA",
        "mean_final_acc": "NA",
        "std_final_acc": "NA",
        "mean_min_eval_loss": "NA",
        "std_min_eval_loss": "NA",
        "has_valid_result": bool(subset),
    }
    if not subset:
        return summary
    summary.update(
        {
            "mean_best_acc": mean_or_na([try_float(item.get("best_acc")) for item in subset]),
            "std_best_acc": std_or_na([try_float(item.get("best_acc")) for item in subset]),
            "mean_final_acc": mean_or_na([try_float(item.get("final_acc")) for item in subset]),
            "std_final_acc": std_or_na([try_float(item.get("final_acc")) for item in subset]),
            "mean_min_eval_loss": mean_or_na([try_float(item.get("min_eval_loss")) for item in subset]),
            "std_min_eval_loss": std_or_na([try_float(item.get("min_eval_loss")) for item in subset]),
        }
    )
    return summary


def compare_metric(lhs, rhs, margin: float) -> bool:
    lhs_val = try_float(lhs)
    rhs_val = try_float(rhs)
    if lhs_val is None or rhs_val is None:
        return False
    return lhs_val >= rhs_val + margin


def best_completed_result(results: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    completed = [item for item in results if item["status"] == "completed" and try_float(item.get("best_acc")) is not None]
    if not completed:
        return None
    return max(completed, key=lambda item: float(item["best_acc"]))


def stage_results(results: List[Dict[str, object]], stage: str) -> List[Dict[str, object]]:
    return [item for item in results if item["stage"] == stage]


def summarize_stage_line(result: Dict[str, object]) -> str:
    return (
        f"{result['experiment_name']} seed={result['seed']} status={result['status']} "
        f"best_acc={format_metric(result['best_acc'])} final_acc={format_metric(result['final_acc'])} "
        f"min_eval_loss={format_metric(result['min_eval_loss'])}"
    )


def build_txt_report(
    base_output_dir: Path,
    all_results: List[Dict[str, object]],
    grouped: Dict[str, Dict[str, object]],
    decision: Dict[str, object],
) -> str:
    lines: List[str] = []
    lines.append("Auto Stage VSMC Experiment Summary")
    lines.append(f"Base output dir: {base_output_dir}")
    lines.append("")

    for stage in ["stage1", "stage2", "stage3", "stage4"]:
        stage_items = stage_results(all_results, stage)
        if not stage_items:
            continue
        lines.append(f"{stage}:")
        for item in stage_items:
            lines.append(f"  - {summarize_stage_line(item)}")
            if item.get("skip_reason"):
                lines.append(f"    reason: {item['skip_reason']}")
        lines.append("")

    lines.append("Grouped metrics:")
    for experiment_name in sorted(grouped):
        group = grouped[experiment_name]
        lines.append(
            f"  - {experiment_name}: mean_best_acc={format_metric(group['mean_best_acc'])} "
            f"std_best_acc={format_metric(group['std_best_acc'])} "
            f"mean_final_acc={format_metric(group['mean_final_acc'])} "
            f"std_final_acc={format_metric(group['std_final_acc'])}"
        )
    lines.append("")

    lines.append("Stage decisions:")
    lines.append(f"  - stopped_after_stage: {decision.get('stopped_after_stage', 'NA')}")
    lines.append(f"  - stop_reason: {decision.get('stop_reason', 'NA')}")
    lines.append(f"  - current_best_experiment: {decision.get('current_best_experiment', 'NA')}")
    lines.append(f"  - current_best_seed: {decision.get('current_best_seed', 'NA')}")
    lines.append(f"  - current_best_best_acc: {format_metric(decision.get('current_best_best_acc'))}")
    lines.append(f"  - fedavg_best_acc: {format_metric(decision.get('fedavg_best_acc'))}")
    lines.append(f"  - improvement_over_fedavg_best: {format_metric(decision.get('improvement_over_fedavg_best'))}")
    lines.append(f"  - improvement_over_fedavg_final: {format_metric(decision.get('improvement_over_fedavg_final'))}")
    lines.append(f"  - reached_3_percent: {decision.get('reached_3_percent', False)}")
    lines.append("")

    suggestions: List[str] = []
    selector = grouped.get("server_reward_selector", {})
    default = grouped.get("learned_vsmc_default", {})
    router_calib = grouped.get("reward_selector_router_calib", {})
    router_ft = grouped.get("reward_selector_router_ft", {})

    if not compare_metric(selector.get("mean_best_acc"), default.get("mean_best_acc"), IMPROVEMENT_MARGIN):
        suggestions.append("server_reward_selector 不如 default，建议先检查 reward scale 和 server validation g。")
    if compare_metric(router_calib.get("mean_best_acc"), selector.get("mean_best_acc"), IMPROVEMENT_MARGIN):
        suggestions.append("router calibration 有提升，建议用多 seed 复验 calibration。")
    if compare_metric(router_ft.get("mean_best_acc"), router_calib.get("mean_best_acc"), IMPROVEMENT_MARGIN) or compare_metric(router_ft.get("mean_best_acc"), selector.get("mean_best_acc"), IMPROVEMENT_MARGIN):
        suggestions.append("router finetune 有提升，建议继续调 router_finetune_lr 和 steps。")

    total_logq = sum(int(item.get("log_q_dominates_count", 0) or 0) for item in all_results)
    total_logg = sum(int(item.get("log_g_not_differentiable_count", 0) or 0) for item in all_results)
    if total_logq > 0:
        suggestions.append("log_q_dominates_count 偏高，建议把 learned_vsmc_logq_coef 调到 0 或继续弱化。")
    if total_logg > 0:
        suggestions.append("log_g_not_differentiable_count 偏高，建议继续修 functional_call 的可微路径。")
    if not decision.get("reached_3_percent", False):
        suggestions.append("如果所有增强都不明显，建议进入 expert alignment 或 client 端 routing loss 改造。")

    lines.append("Next-step suggestions:")
    if suggestions:
        for suggestion in suggestions:
            lines.append(f"  - {suggestion}")
    else:
        lines.append("  - 当前没有额外建议。")
    return "\n".join(lines) + "\n"


def write_summary_csv(path: Path, results: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "stage",
        "experiment_name",
        "seed",
        "status",
        "return_code",
        "final_acc",
        "best_acc",
        "best_acc_round",
        "final_eval_loss",
        "min_eval_loss",
        "min_eval_loss_round",
        "acc_at_min_eval_loss",
        "final_train_loss",
        "final_train_acc",
        "log_q_dominates_count",
        "log_g_not_differentiable_count",
        "fallback_count",
        "mean_normalized_ess_mean",
        "mean_normalized_ess_last",
        "proposal_grad_norm_mean",
        "proposal_param_delta_norm_mean",
        "reward_std_mean",
        "fedavg_delta_norm_mean",
        "particle_delta_norm_mean",
        "best_particle_is_fedavg_rate",
        "output_dir",
        "skip_reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "NA") for field in fieldnames})


def decide_progression(all_results: List[Dict[str, object]], include_alignment: bool) -> Dict[str, object]:
    grouped = {name: aggregate_experiment_results(all_results, name) for name in {item["experiment_name"] for item in all_results}}
    decision = {
        "current_best_experiment": "NA",
        "current_best_seed": "NA",
        "current_best_best_acc": "NA",
        "fedavg_best_acc": grouped.get("fedavg", {}).get("mean_best_acc", "NA"),
        "improvement_over_fedavg_best": "NA",
        "improvement_over_fedavg_final": "NA",
        "reached_3_percent": False,
        "stopped_after_stage": "stage1",
        "stop_reason": "stage1_not_evaluated",
    }

    fedavg_group = grouped.get("fedavg", {})
    default_group = grouped.get("learned_vsmc_default", {})
    selector_group = grouped.get("server_reward_selector", {})

    current_best_name = "fedavg"
    current_best_group = fedavg_group

    for name in ["learned_vsmc_default", "server_reward_selector"]:
        candidate = grouped.get(name, {})
        if compare_metric(candidate.get("mean_best_acc"), current_best_group.get("mean_best_acc"), 0.0):
            current_best_name = name
            current_best_group = candidate

    stage1_selector_missing = not selector_group.get("has_valid_result", False)
    stage1_default_missing = not default_group.get("has_valid_result", False)
    if stage1_selector_missing or stage1_default_missing:
        decision["current_best_experiment"] = current_best_name
        decision["current_best_best_acc"] = current_best_group.get("mean_best_acc", "NA")
        decision["stop_reason"] = "key stage1 experiment failed or required args not implemented"
    elif compare_metric(selector_group.get("mean_best_acc"), default_group.get("mean_best_acc"), IMPROVEMENT_MARGIN):
        current_best_name = "server_reward_selector"
        current_best_group = selector_group
        decision["stopped_after_stage"] = "stage4" if include_alignment else "stage3"
        decision["stop_reason"] = "stage1_selector_beat_default"

        router_calib = grouped.get("reward_selector_router_calib", {})
        if router_calib.get("has_valid_result", False) and compare_metric(router_calib.get("mean_best_acc"), selector_group.get("mean_best_acc"), IMPROVEMENT_MARGIN):
            current_best_name = "reward_selector_router_calib"
            current_best_group = router_calib
            decision["stop_reason"] = "router_calibration_current_best"

        router_ft = grouped.get("reward_selector_router_ft", {})
        if router_ft.get("has_valid_result", False) and compare_metric(router_ft.get("mean_best_acc"), current_best_group.get("mean_best_acc"), IMPROVEMENT_MARGIN):
            current_best_name = "reward_selector_router_ft"
            current_best_group = router_ft
            decision["stop_reason"] = "router_finetune_current_best"

        alignment = grouped.get("best_plus_alignment", {})
        if include_alignment and alignment.get("has_valid_result", False) and compare_metric(alignment.get("mean_best_acc"), current_best_group.get("mean_best_acc"), IMPROVEMENT_MARGIN):
            current_best_name = "best_plus_alignment"
            current_best_group = alignment
            decision["stop_reason"] = "alignment_current_best"
    else:
        decision["stop_reason"] = "reward selector 未超过 default，先不要跑 router 增强"
        decision["stopped_after_stage"] = "stage1"

    decision["current_best_experiment"] = current_best_name
    decision["current_best_best_acc"] = current_best_group.get("mean_best_acc", "NA")
    best_row = best_completed_result([row for row in all_results if row["experiment_name"] == current_best_name])
    if best_row is not None:
        decision["current_best_seed"] = best_row["seed"]
    fedavg_mean_best = try_float(fedavg_group.get("mean_best_acc"))
    current_best_mean_best = try_float(current_best_group.get("mean_best_acc"))
    fedavg_mean_final = try_float(fedavg_group.get("mean_final_acc"))
    current_best_mean_final = try_float(current_best_group.get("mean_final_acc"))
    if current_best_mean_best is not None and fedavg_mean_best is not None:
        decision["improvement_over_fedavg_best"] = current_best_mean_best - fedavg_mean_best
        decision["reached_3_percent"] = (current_best_mean_best - fedavg_mean_best) >= 0.03
    if current_best_mean_final is not None and fedavg_mean_final is not None:
        decision["improvement_over_fedavg_final"] = current_best_mean_final - fedavg_mean_final
    return decision


def create_alignment_task_from_current_best(
    cli_args,
    current_best_name: str,
    seeds: List[int],
    flags: set,
    choices: Dict[str, List[str]],
) -> List[ExperimentTask]:
    specs = {spec.experiment_name: spec for spec in build_stage_specs()}
    parent_spec = specs.get(current_best_name)
    alignment_spec = specs["best_plus_alignment"]
    if parent_spec is None:
        return [make_task(cli_args, alignment_spec, seed, flags, choices) for seed in seeds]
    tasks = []
    for seed in seeds:
        inherited = list(parent_spec.extra_args) + build_alignment_extra_args()
        tasks.append(make_task(cli_args, alignment_spec, seed, flags, choices, inherited_extra_args=inherited))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Automatic staged scheduler for learned VSMC experiments.")
    parser.add_argument("--base_output_dir", type=str, default="run_stage_vsmc")
    parser.add_argument("--seeds", type=str, default="3")
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3")
    parser.add_argument("--max_parallel_per_gpu", type=int, default=1)
    parser.add_argument("--include_alignment", type=str2bool, default=False)
    parser.add_argument("--run_all_stages", type=str2bool, default=False)
    parser.add_argument("--data_name", type=str, default="cifar10")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--client_epochs", type=int, default=1)
    parser.add_argument("--backbone_name", type=str, default="resnet18")
    args = parser.parse_args()

    base_output_dir = Path(args.base_output_dir).resolve()
    ensure_dir(base_output_dir)
    ensure_dir(base_output_dir / "logs")

    seeds = [int(item) for item in parse_csv_list(args.seeds)]
    gpu_ids = parse_csv_list(args.gpu_ids)
    if not gpu_ids:
        raise SystemExit("No GPU ids provided.")

    supported_flags, supported_choices = discover_arg_support(ARGS_PY)

    data_prep_results = []
    for seed in seeds:
        data_prep_results.append(prepare_data_for_seed(args, seed, base_output_dir / f"shared_data_seed{seed}", base_output_dir))

    all_results: List[Dict[str, object]] = []
    specs = build_stage_specs()

    stage1_tasks = [make_task(args, spec, seed, supported_flags, supported_choices) for spec in specs if spec.stage == "stage1" for seed in seeds]
    run_stage(stage1_tasks, gpu_ids, args.max_parallel_per_gpu, args.num_rounds, base_output_dir, "stage1")
    for task in stage1_tasks:
        task.parsed_result = parse_result(task)
        all_results.append(task.parsed_result)

    grouped = {name: aggregate_experiment_results(all_results, name) for name in {item["experiment_name"] for item in all_results}}
    selector_group = grouped.get("server_reward_selector", {})
    default_group = grouped.get("learned_vsmc_default", {})
    continue_stage2 = bool(args.run_all_stages) or (selector_group.get("has_valid_result", False) and default_group.get("has_valid_result", False) and compare_metric(selector_group.get("mean_best_acc"), default_group.get("mean_best_acc"), IMPROVEMENT_MARGIN))

    if continue_stage2:
        stage2_tasks = [make_task(args, spec, seed, supported_flags, supported_choices) for spec in specs if spec.stage == "stage2" for seed in seeds]
        run_stage(stage2_tasks, gpu_ids, args.max_parallel_per_gpu, args.num_rounds, base_output_dir, "stage2")
        for task in stage2_tasks:
            task.parsed_result = parse_result(task)
            all_results.append(task.parsed_result)

        grouped = {name: aggregate_experiment_results(all_results, name) for name in {item["experiment_name"] for item in all_results}}
        current_best_name = "server_reward_selector"
        if compare_metric(grouped.get("reward_selector_router_calib", {}).get("mean_best_acc"), grouped.get("server_reward_selector", {}).get("mean_best_acc"), IMPROVEMENT_MARGIN):
            current_best_name = "reward_selector_router_calib"

        stage3_tasks = [make_task(args, spec, seed, supported_flags, supported_choices) for spec in specs if spec.stage == "stage3" for seed in seeds]
        run_stage(stage3_tasks, gpu_ids, args.max_parallel_per_gpu, args.num_rounds, base_output_dir, "stage3")
        for task in stage3_tasks:
            task.parsed_result = parse_result(task)
            all_results.append(task.parsed_result)

        grouped = {name: aggregate_experiment_results(all_results, name) for name in {item["experiment_name"] for item in all_results}}
        if compare_metric(grouped.get("reward_selector_router_ft", {}).get("mean_best_acc"), grouped.get(current_best_name, {}).get("mean_best_acc"), IMPROVEMENT_MARGIN):
            current_best_name = "reward_selector_router_ft"

        if args.include_alignment:
            stage4_tasks = create_alignment_task_from_current_best(args, current_best_name, seeds, supported_flags, supported_choices)
            run_stage(stage4_tasks, gpu_ids, args.max_parallel_per_gpu, args.num_rounds, base_output_dir, "stage4")
            for task in stage4_tasks:
                task.parsed_result = parse_result(task)
                all_results.append(task.parsed_result)

    grouped = {name: aggregate_experiment_results(all_results, name) for name in {item["experiment_name"] for item in all_results}}
    decision = decide_progression(all_results, include_alignment=args.include_alignment)
    decision["data_prep"] = data_prep_results
    decision["grouped"] = grouped

    write_summary_csv(base_output_dir / "auto_stage_summary.csv", all_results)
    (base_output_dir / "auto_stage_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (base_output_dir / "auto_stage_summary.txt").write_text(
        build_txt_report(base_output_dir, all_results, grouped, decision),
        encoding="utf-8",
    )

    print(f"Wrote summary CSV to {base_output_dir / 'auto_stage_summary.csv'}")
    print(f"Wrote summary TXT to {base_output_dir / 'auto_stage_summary.txt'}")
    print(f"Wrote decision JSON to {base_output_dir / 'auto_stage_decision.json'}")


if __name__ == "__main__":
    main()
