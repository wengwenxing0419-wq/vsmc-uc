import argparse
import csv
import os
import re
from collections import defaultdict


DEFAULT_LOG_A = (
    "save_k50_aux0.01_lr1e-4_ep100/result/logs/"
    "data_cifar10 client_50 alpha_0.1.log"
)
DEFAULT_LOG_B = (
    "save_k50_aux0.01_lr1e-4_ep100_base/result/logs/"
    "data_cifar10 client_50 alpha_0.1.log"
)

CLIENT_RE = re.compile(
    r"--client:(?P<client>\d+)\s+--epoch:(?P<epoch>\d+)/(?P<epochs>\d+)\s+"
    r"--client_train_loss:(?P<client_train_loss>[0-9.]+)\s+"
    r"--client_train_acc:(?P<client_train_acc>[0-9.]+)"
)
ROUND_START_RE = re.compile(r"Round\s+(?P<round>\d+)/(?P<total>\d+)\s+start")
ROUND_RE = re.compile(
    r"--round:(?P<round>\d+)\s+--avg_client_train_loss:(?P<avg_train_loss>[0-9.]+)\s+"
    r"--avg_client_train_acc:(?P<avg_train_acc>[0-9.]+)"
    r"(?:\s+--max_client_train_acc:(?P<max_train_acc>[0-9.]+)\s+"
    r"--min_client_train_acc:(?P<min_train_acc>[0-9.]+))?\s+"
    r"--server_test_loss:(?P<server_test_loss>[0-9.]+)\s+"
    r"--server_test_acc:(?P<server_test_acc>[0-9.]+)",
    re.MULTILINE,
)


def parse_log(path):
    current_round = None
    client_rows = []
    round_rows = []

    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()

    for line in content.splitlines():
        start_match = ROUND_START_RE.search(line)
        if start_match:
            current_round = int(start_match.group("round"))
            continue

        client_match = CLIENT_RE.search(line)
        if client_match:
            if current_round is None:
                raise ValueError(f"Found client row before round start in {path}")
            row = {
                "round": current_round,
                "client": int(client_match.group("client")),
                "epoch": int(client_match.group("epoch")),
                "client_train_loss": float(client_match.group("client_train_loss")),
                "client_train_acc": float(client_match.group("client_train_acc")),
            }
            client_rows.append(row)

    for round_match in ROUND_RE.finditer(content):
        row = {
            "round": int(round_match.group("round")),
            "avg_client_train_loss": float(round_match.group("avg_train_loss")),
            "avg_client_train_acc": float(round_match.group("avg_train_acc")),
            "server_test_loss": float(round_match.group("server_test_loss")),
            "server_test_acc": float(round_match.group("server_test_acc")),
        }
        if round_match.group("max_train_acc") is not None:
            row["max_client_train_acc"] = float(round_match.group("max_train_acc"))
        if round_match.group("min_train_acc") is not None:
            row["min_client_train_acc"] = float(round_match.group("min_train_acc"))
        round_rows.append(row)

    if not round_rows:
        raise ValueError(f"No round summary rows found in {path}")
    return {"path": path, "rounds": round_rows, "clients": client_rows}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def aggregate_client_metrics_by_round(parsed):
    if not parsed["clients"]:
        return {
            row["round"]: {
                "client_train_loss": row["avg_client_train_loss"],
                "client_train_acc": row["avg_client_train_acc"],
                "p10_client_train_acc": float("nan"),
            }
            for row in parsed["rounds"]
        }

    grouped = defaultdict(list)
    for row in parsed["clients"]:
        grouped[row["round"]].append(row)

    return {
        round_id: {
            "client_train_loss": mean(row["client_train_loss"] for row in rows),
            "client_train_acc": mean(row["client_train_acc"] for row in rows),
            "p10_client_train_acc": percentile((row["client_train_acc"] for row in rows), 10),
        }
        for round_id, rows in grouped.items()
    }


def summarize(parsed, last_n=10):
    rounds = parsed["rounds"]
    clients = parsed["clients"]
    round_client_metrics = aggregate_client_metrics_by_round(parsed)
    best_round = max(rounds, key=lambda row: row["server_test_acc"])
    final_round = rounds[-1]
    last_rounds = rounds[-last_n:]
    final_round_metrics = round_client_metrics[final_round["round"]]

    by_client = defaultdict(list)
    for row in clients:
        by_client[row["client"]].append(row)

    final_clients = {
        client_id: rows[-1]["client_train_acc"]
        for client_id, rows in by_client.items()
    }
    last_client_means = {
        client_id: mean(row["client_train_acc"] for row in rows[-last_n:])
        for client_id, rows in by_client.items()
    }
    has_client_rows = bool(final_clients)

    return {
        "path": parsed["path"],
        "round_count": len(rounds),
        "client_count": len(by_client) if has_client_rows else "n/a",
        "final_round": final_round["round"],
        "final_server_test_loss": final_round["server_test_loss"],
        "final_server_test_acc": final_round["server_test_acc"],
        "final_avg_client_train_acc": final_round["avg_client_train_acc"],
        "final_max_client_train_acc": max(final_clients.values()) if has_client_rows else float("nan"),
        "final_min_client_train_acc": min(final_clients.values()) if has_client_rows else float("nan"),
        "final_p10_client_train_acc": final_round_metrics["p10_client_train_acc"],
        "best_round": best_round["round"],
        "best_server_test_acc": best_round["server_test_acc"],
        f"last{last_n}_server_test_acc": mean(row["server_test_acc"] for row in last_rounds),
        f"last{last_n}_avg_client_train_acc": mean(row["avg_client_train_acc"] for row in last_rounds),
        f"last{last_n}_p10_client_train_acc": mean(
            round_client_metrics[row["round"]]["p10_client_train_acc"] for row in last_rounds
        ),
        "final_client_train_mean": (
            mean(final_clients.values()) if has_client_rows else final_round["avg_client_train_acc"]
        ),
        "final_client_train_min": min(final_clients.values()) if has_client_rows else float("nan"),
        "final_client_train_max": max(final_clients.values()) if has_client_rows else float("nan"),
        "final_clients_le_0.5": (
            sum(value <= 0.5 for value in final_clients.values()) if has_client_rows else "n/a"
        ),
        f"last{last_n}_client_train_mean": (
            mean(last_client_means.values())
            if has_client_rows
            else mean(row["avg_client_train_acc"] for row in last_rounds)
        ),
        f"last{last_n}_clients_le_0.5": (
            sum(value <= 0.5 for value in last_client_means.values()) if has_client_rows else "n/a"
        ),
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_round_comparison(label_a, parsed_a, label_b, parsed_b):
    by_round_a = {row["round"]: row for row in parsed_a["rounds"]}
    by_round_b = {row["round"]: row for row in parsed_b["rounds"]}
    client_metrics_a = aggregate_client_metrics_by_round(parsed_a)
    client_metrics_b = aggregate_client_metrics_by_round(parsed_b)
    rows = []
    for round_id in sorted(set(by_round_a) & set(by_round_b)):
        row_a = by_round_a[round_id]
        row_b = by_round_b[round_id]
        metrics_a = client_metrics_a[round_id]
        metrics_b = client_metrics_b[round_id]
        rows.append(
            {
                "round": round_id,
                f"{label_a}_server_test_acc": row_a["server_test_acc"],
                f"{label_b}_server_test_acc": row_b["server_test_acc"],
                "server_test_acc_diff": row_a["server_test_acc"] - row_b["server_test_acc"],
                f"{label_a}_server_test_loss": row_a["server_test_loss"],
                f"{label_b}_server_test_loss": row_b["server_test_loss"],
                "server_test_loss_diff": row_a["server_test_loss"] - row_b["server_test_loss"],
                f"{label_a}_avg_client_train_acc": row_a["avg_client_train_acc"],
                f"{label_b}_avg_client_train_acc": row_b["avg_client_train_acc"],
                "avg_client_train_acc_diff": row_a["avg_client_train_acc"] - row_b["avg_client_train_acc"],
                f"{label_a}_p10_client_train_acc": metrics_a["p10_client_train_acc"],
                f"{label_b}_p10_client_train_acc": metrics_b["p10_client_train_acc"],
                "p10_client_train_acc_diff": metrics_a["p10_client_train_acc"] - metrics_b["p10_client_train_acc"],
                f"{label_a}_avg_client_train_loss": metrics_a["client_train_loss"],
                f"{label_b}_avg_client_train_loss": metrics_b["client_train_loss"],
                "avg_client_train_loss_diff": metrics_a["client_train_loss"] - metrics_b["client_train_loss"],
            }
        )
    return rows


def build_client_final_comparison(label_a, parsed_a, label_b, parsed_b):
    if not parsed_a["clients"] or not parsed_b["clients"]:
        return []

    def final_by_client(parsed):
        values = {}
        for row in parsed["clients"]:
            values[row["client"]] = row
        return values

    final_a = final_by_client(parsed_a)
    final_b = final_by_client(parsed_b)
    rows = []
    for client_id in sorted(set(final_a) & set(final_b)):
        row_a = final_a[client_id]
        row_b = final_b[client_id]
        rows.append(
            {
                "client": client_id,
                f"{label_a}_final_client_train_acc": row_a["client_train_acc"],
                f"{label_b}_final_client_train_acc": row_b["client_train_acc"],
                "final_client_train_acc_diff": row_a["client_train_acc"] - row_b["client_train_acc"],
                f"{label_a}_final_client_train_loss": row_a["client_train_loss"],
                f"{label_b}_final_client_train_loss": row_b["client_train_loss"],
                "final_client_train_loss_diff": row_a["client_train_loss"] - row_b["client_train_loss"],
            }
        )
    return rows


def maybe_plot(output_dir, label_a, parsed_a, label_b, parsed_b):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    rounds_a = [row["round"] for row in parsed_a["rounds"]]
    server_test_acc_a = [row["server_test_acc"] for row in parsed_a["rounds"]]
    rounds_b = [row["round"] for row in parsed_b["rounds"]]
    server_test_acc_b = [row["server_test_acc"] for row in parsed_b["rounds"]]
    client_metrics_a = aggregate_client_metrics_by_round(parsed_a)
    client_metrics_b = aggregate_client_metrics_by_round(parsed_b)
    p10_a = [client_metrics_a[round_id]["p10_client_train_acc"] for round_id in rounds_a]
    p10_b = [client_metrics_b[round_id]["p10_client_train_acc"] for round_id in rounds_b]
    train_loss_a = [client_metrics_a[round_id]["client_train_loss"] for round_id in rounds_a]
    train_loss_b = [client_metrics_b[round_id]["client_train_loss"] for round_id in rounds_b]
    server_test_loss_a = [row["server_test_loss"] for row in parsed_a["rounds"]]
    server_test_loss_b = [row["server_test_loss"] for row in parsed_b["rounds"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    axes[0].plot(rounds_a, server_test_acc_a, label=label_a)
    axes[0].plot(rounds_b, server_test_acc_b, label=label_b)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Server Test Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(rounds_a, p10_a, label=label_a)
    axes[1].plot(rounds_b, p10_b, label=label_b)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("P10 Client Train Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(rounds_a, train_loss_a, label=label_a)
    axes[2].plot(rounds_b, train_loss_b, label=label_b)
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Average Training Loss")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(rounds_a, server_test_loss_a, label=label_a)
    axes[3].plot(rounds_b, server_test_loss_b, label=label_b)
    axes[3].set_xlabel("Round")
    axes[3].set_ylabel("Loss")
    axes[3].set_title("Server Test Loss")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.suptitle("VSMC vs FedAvg Training Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "combined_metrics.png"), dpi=200)
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    plt.plot(rounds_a, server_test_acc_a, label=label_a)
    plt.plot(rounds_b, server_test_acc_b, label=label_b)
    plt.xlabel("Round")
    plt.ylabel("Server test accuracy")
    plt.title("Round Server Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "round_server_test_acc.png"), dpi=200)
    plt.close()

    diff_rows = build_round_comparison(label_a, parsed_a, label_b, parsed_b)
    plt.figure(figsize=(10, 4))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(
        [row["round"] for row in diff_rows],
        [row["server_test_acc_diff"] for row in diff_rows],
        label=f"{label_a} - {label_b}",
    )
    plt.xlabel("Round")
    plt.ylabel("Server test accuracy diff")
    plt.title("Round Server Test Accuracy Difference")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "round_server_test_acc_diff.png"), dpi=200)
    plt.close()


def print_summary(label, summary):
    print(f"\n[{label}]")
    print(f"path: {summary['path']}")
    print(f"rounds: {summary['round_count']} | clients: {summary['client_count']}")
    print(
        "final server test loss/acc: "
        f"{summary['final_server_test_loss']:.4f} / "
        f"{summary['final_server_test_acc']:.4f}"
    )
    print(
        f"best server test acc: {summary['best_server_test_acc']:.4f} "
        f"at round {summary['best_round']}"
    )
    print(
        "final avg/max/min client train acc: "
        f"{summary['final_avg_client_train_acc']:.4f} / "
        f"{summary['final_max_client_train_acc']:.4f} / "
        f"{summary['final_min_client_train_acc']:.4f}"
    )
    print(f"last10 server test acc: {summary['last10_server_test_acc']:.4f}")
    print(
        "final p10 / last10 p10 client train acc: "
        f"{summary['final_p10_client_train_acc']:.4f} / "
        f"{summary['last10_p10_client_train_acc']:.4f}"
    )
    print(
        "final client train mean/min/max: "
        f"{summary['final_client_train_mean']:.4f} / "
        f"{summary['final_client_train_min']:.4f} / "
        f"{summary['final_client_train_max']:.4f}"
    )
    print(f"final clients <= 0.5: {summary['final_clients_le_0.5']}")


def main():
    parser = argparse.ArgumentParser(description="Compare two FL training log files.")
    parser.add_argument("--log-a", default=DEFAULT_LOG_A)
    parser.add_argument("--log-b", default=DEFAULT_LOG_B)
    parser.add_argument("--label-a", default="method_a")
    parser.add_argument("--label-b", default="fedavg")
    parser.add_argument("--output-dir", default="log_compare_output")
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    parsed_a = parse_log(args.log_a)
    parsed_b = parse_log(args.log_b)
    summary_a = summarize(parsed_a, args.last_n)
    summary_b = summarize(parsed_b, args.last_n)

    print_summary(args.label_a, summary_a)
    print_summary(args.label_b, summary_b)
    print("\n[delta]")
    print(
        f"final server test acc diff ({args.label_a} - {args.label_b}): "
        f"{summary_a['final_server_test_acc'] - summary_b['final_server_test_acc']:+.4f}"
    )
    print(
        f"best server test acc diff ({args.label_a} - {args.label_b}): "
        f"{summary_a['best_server_test_acc'] - summary_b['best_server_test_acc']:+.4f}"
    )
    print(
        f"last{args.last_n} server test acc diff ({args.label_a} - {args.label_b}): "
        f"{summary_a[f'last{args.last_n}_server_test_acc'] - summary_b[f'last{args.last_n}_server_test_acc']:+.4f}"
    )
    print(
        f"last{args.last_n} p10 client train acc diff ({args.label_a} - {args.label_b}): "
        f"{summary_a[f'last{args.last_n}_p10_client_train_acc'] - summary_b[f'last{args.last_n}_p10_client_train_acc']:+.4f}"
    )

    summary_rows = []
    for label, summary in [(args.label_a, summary_a), (args.label_b, summary_b)]:
        row = {"label": label}
        row.update(summary)
        summary_rows.append(row)

    summary_fields = ["label"] + list(summary_a.keys())
    write_csv(os.path.join(args.output_dir, "summary.csv"), summary_rows, summary_fields)

    round_rows = build_round_comparison(args.label_a, parsed_a, args.label_b, parsed_b)
    if round_rows:
        write_csv(
            os.path.join(args.output_dir, "round_comparison.csv"),
            round_rows,
            list(round_rows[0].keys()),
        )

    client_rows = build_client_final_comparison(args.label_a, parsed_a, args.label_b, parsed_b)
    if client_rows:
        write_csv(
            os.path.join(args.output_dir, "client_final_comparison.csv"),
            client_rows,
            list(client_rows[0].keys()),
        )

    if not args.no_plot:
        maybe_plot(args.output_dir, args.label_a, parsed_a, args.label_b, parsed_b)

    print(f"\nWrote comparison files to: {args.output_dir}")


if __name__ == "__main__":
    main()
