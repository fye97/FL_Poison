#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+.*?Test Acc:\s*([0-9.]+)\s*Test loss:\s*([0-9.]+)"
)

ATTACK = "FangAttack"
ALPHAS = ("1", "0.5")
DEFENSES = (
    "Mean",
    "NormClipping",
    "MultiKrum",
    "FLTrust",
    "FLDetector",
    "TriGuardFL",
    "CARAT",
)

COLORS = {
    "NoAttack+Mean": "#222222",
    "Mean": "#7f7f7f",
    "NormClipping": "#c44e52",
    "MultiKrum": "#8172b2",
    "FLTrust": "#8c6d31",
    "FLDetector": "#dd8452",
    "TriGuardFL": "#55a868",
    "CARAT": "#4c72b0",
}

LINESTYLES = {
    "NoAttack+Mean": (0, (4, 2)),
    "Mean": "-",
    "NormClipping": "-",
    "MultiKrum": "-",
    "FLTrust": "-",
    "FLDetector": "-",
    "TriGuardFL": "-",
    "CARAT": "-",
}


def parse_series(path: Path, expected_rounds: int) -> np.ndarray | None:
    values: list[tuple[int, float]] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            match = EPOCH_RE.match(line)
            if not match:
                continue
            epoch = int(match.group(1))
            test_acc = 100.0 * float(match.group(2))
            values.append((epoch, test_acc))

    if not values:
        return None
    values.sort(key=lambda item: item[0])
    if values[-1][0] < expected_rounds - 1:
        return None

    arr = np.full(expected_rounds, np.nan, dtype=np.float64)
    for epoch, test_acc in values:
        if 0 <= epoch < expected_rounds:
            arr[epoch] = test_acc
    if np.isnan(arr).any():
        return None
    return arr


def collect(
    root: Path,
    alpha: str,
    attack: str,
    defense: str,
    expected_rounds: int,
) -> list[np.ndarray]:
    pattern = (
        f"CIFAR100_resnet18_non-iid_{attack}_{defense}_200_20_0.05_"
        f"FedAvg_adv0.2_seed*_alpha{alpha}_cfgFedAvg_CIFAR100_config_exp*.txt"
    )
    series: list[np.ndarray] = []
    for path in sorted(root.glob(pattern)):
        parsed = parse_series(path, expected_rounds)
        if parsed is not None:
            series.append(parsed)
    return series


def mean_and_std(series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(series)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)


def plot(output: Path, playground_root: Path, expected_rounds: int) -> None:
    root = playground_root / "FedAvg" / "CIFAR100_resnet18" / "non-iid"
    rounds = np.arange(expected_rounds)

    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(10.4, 3.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.65])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.set_axis_off()

    legend_handles = {}

    for ax, alpha in zip(axes, ALPHAS):
        plotted_values: list[np.ndarray] = []

        clean_series = collect(root, alpha, "NoAttack", "Mean", expected_rounds)
        if clean_series:
            clean_mean, _ = mean_and_std(clean_series)
            (line,) = ax.plot(
                rounds,
                clean_mean,
                color=COLORS["NoAttack+Mean"],
                linestyle=LINESTYLES["NoAttack+Mean"],
                linewidth=1.9,
                label=f"NoAttack+Mean (n={len(clean_series)})",
                zorder=1,
            )
            legend_handles.setdefault("NoAttack+Mean", line)
            plotted_values.append(clean_mean)

        for defense in DEFENSES:
            series = collect(root, alpha, ATTACK, defense, expected_rounds)
            if not series:
                continue
            avg, std = mean_and_std(series)
            linewidth = 2.5 if defense == "CARAT" else 1.55
            alpha_line = 1.0 if defense in {"CARAT", "TriGuardFL", "FLDetector"} else 0.78
            (line,) = ax.plot(
                rounds,
                avg,
                color=COLORS[defense],
                linestyle=LINESTYLES[defense],
                linewidth=linewidth,
                alpha=alpha_line,
                label=f"{defense} (n={len(series)})",
                zorder=4 if defense == "CARAT" else 2,
            )
            if defense == "CARAT":
                ax.fill_between(
                    rounds,
                    avg - std,
                    avg + std,
                    color=COLORS[defense],
                    alpha=0.14,
                    linewidth=0,
                    zorder=3,
                )
            legend_handles.setdefault(defense, line)
            plotted_values.append(avg)

        ax.set_title(rf"{ATTACK}, $\alpha={alpha}$")
        ax.set_xlabel("Round")
        ax.grid(alpha=0.22, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if plotted_values:
            stacked = np.vstack(plotted_values)
            ymax = min(60.0, float(np.nanmax(stacked)) + 3.0)
            ax.set_ylim(0.0, ymax)
        ax.set_xlim(0, expected_rounds - 1)

    axes[0].set_ylabel("Test accuracy (%)")

    ordered_labels = ["NoAttack+Mean", *DEFENSES]
    handles = [legend_handles[label] for label in ordered_labels if label in legend_handles]
    labels = [handle.get_label() for handle in handles]
    legend_ax.legend(
        handles,
        labels,
        loc="center left",
        frameon=False,
        handlelength=2.6,
        borderaxespad=0.0,
    )

    fig.savefig(output, bbox_inches="tight")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_playground = repo_root.parent / "Poisoning_Resilient_Federated_Learning_Playground"
    default_output = (
        repo_root
        / "CARAT"
        / "Formatting_Instructions_For_NeurIPS_2026"
        / "figures"
        / "cifar100_fangattack_curves.pdf"
    )

    parser = argparse.ArgumentParser(
        description="Plot CIFAR-100/ResNet18 FangAttack accuracy curves for the CARAT paper."
    )
    parser.add_argument("--playground-root", type=Path, default=default_playground)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--rounds", type=int, default=200)
    args = parser.parse_args()

    plot(args.output, args.playground_root, args.rounds)
    print(args.output)


if __name__ == "__main__":
    main()
