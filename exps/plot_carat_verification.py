#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from glob import glob
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_series(path: Path, max_rounds: int | None = None) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if max_rounds is not None:
        rows = rows[:max_rounds]

    rounds = np.asarray([int(row["epoch"]) + 1 for row in rows], dtype=np.int32)
    eval_acc = np.asarray([100.0 * float(row["eval_acc"]) for row in rows], dtype=np.float64)
    round_time = np.asarray([float(row["round_time_sec"]) for row in rows], dtype=np.float64)
    return {"round": rounds, "eval_acc": eval_acc, "round_time": round_time}


def resolve_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def stack_metric(paths: list[Path], metric: str, max_rounds: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for path in paths:
        series = load_series(path, max_rounds=max_rounds)
        if len(series["round"]) < max_rounds:
            raise ValueError(f"{path} has only {len(series['round'])} rounds; expected {max_rounds}")
        arrays.append(series[metric])
    stacked = np.vstack(arrays)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)


def mean_round_time(paths: list[Path], max_rounds: int) -> tuple[float, float]:
    values = []
    for path in paths:
        series = load_series(path, max_rounds=max_rounds)
        values.append(series["round_time"].mean())
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def plot_verification(
    clean: Path,
    mean_paths: list[Path],
    carat_paths: list[Path],
    output: Path,
    max_rounds: int,
) -> None:
    if not mean_paths:
        raise ValueError("No Mean seed files resolved")
    if not carat_paths:
        raise ValueError("No CARAT seed files resolved")

    clean_series = load_series(clean)
    rounds = np.arange(1, max_rounds + 1, dtype=np.int32)
    mean_avg, mean_std = stack_metric(mean_paths, "eval_acc", max_rounds)
    carat_avg, carat_std = stack_metric(carat_paths, "eval_acc", max_rounds)
    mean_time_avg, mean_time_std = mean_round_time(mean_paths, max_rounds)
    carat_time_avg, carat_time_std = mean_round_time(carat_paths, max_rounds)
    clean_time_avg = float(clean_series["round_time"].mean())

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        clean_series["round"],
        clean_series["eval_acc"],
        color="#555555",
        linewidth=2.0,
        linestyle="--",
        label="Clean Mean (1 seed)",
    )
    ax.plot(rounds, mean_avg, color="#dd8452", linewidth=2.2, label=f"ALIE + Mean ({len(mean_paths)} seeds)")
    ax.fill_between(rounds, mean_avg - mean_std, mean_avg + mean_std, color="#dd8452", alpha=0.18, linewidth=0)
    ax.plot(rounds, carat_avg, color="#4c72b0", linewidth=2.2, label=f"ALIE + CARAT ({len(carat_paths)} seeds)")
    ax.fill_between(rounds, carat_avg - carat_std, carat_avg + carat_std, color="#4c72b0", alpha=0.18, linewidth=0)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Mean Curve with Seed Variance")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    labels = ["Clean Mean", "ALIE + Mean", "ALIE + CARAT"]
    values = [clean_time_avg, mean_time_avg, carat_time_avg]
    errors = [0.0, mean_time_std, carat_time_std]
    colors = ["#555555", "#dd8452", "#4c72b0"]
    bars = ax.bar(labels, values, yerr=errors, color=colors, width=0.62, capsize=5, ecolor="#333333")
    ax.set_ylabel("Average Round Time (s)")
    ax.set_title("Runtime Overhead")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.18,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(output, dpi=300, bbox_inches="tight")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot verification-scale CARAT curves from local multi-seed CSVs."
    )
    parser.add_argument(
        "--clean",
        type=Path,
        default=repo_root
        / "logs/local_runs/FedAvg/CIFAR100_resnet18/iid/NoAttack__Mean"
        / "ep300_clients20_lr0.05_adv0_seed42_exp0_cfgFedAvg_CIFAR100_Resnet18_protocol/metrics_exp0.csv",
    )
    parser.add_argument(
        "--mean-pattern",
        action="append",
        default=[
            str(
                repo_root
                / "logs/manual_runs/FedAvg/CIFAR100_resnet18/iid/ALIE__Mean"
                / "ep170_clients20_lr0.05_adv0.2_seed4*_exp0_cfgFedAvg_CIFAR100_Resnet18_protocol_apr23/metrics_exp0.csv"
            )
        ],
    )
    parser.add_argument(
        "--carat-pattern",
        action="append",
        default=[
            str(
                repo_root
                / "logs/manual_runs/FedAvg/CIFAR100_resnet18/iid/ALIE__CARAT"
                / "ep170_clients20_lr0.05_adv0.2_seed4*_exp0_cfgFedAvg_CIFAR100_Resnet18_protocol_apr23/metrics_exp0.csv"
            ),
            str(
                repo_root
                / "logs/manual_runs/FedAvg/CIFAR100_resnet18/iid/ALIE__CARAT"
                / "ep300_clients20_lr0.05_adv0.2_seed42_exp0_cfgFedAvg_CIFAR100_Resnet18_protocol_apr23/metrics_exp0.csv"
            ),
        ],
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=170,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "CARAT/Formatting_Instructions_For_NeurIPS_2026/figures/carat_verification.pdf",
    )
    args = parser.parse_args()

    mean_paths = resolve_paths(args.mean_pattern)
    carat_paths = resolve_paths(args.carat_pattern)
    plot_verification(args.clean, mean_paths, carat_paths, args.output, args.max_rounds)


if __name__ == "__main__":
    main()
