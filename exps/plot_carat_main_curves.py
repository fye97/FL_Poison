#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    "CARAT",
)

COLORS = {
    "NoAttack+Mean": "#272727",
    "Mean": "#8A8A8A",
    "NormClipping": "#B64342",
    "MultiKrum": "#9A4D8E",
    "FLTrust": "#42949E",
    "FLDetector": "#D9892B",
    "CARAT": "#0F4D92",
}

LINESTYLES = {
    "NoAttack+Mean": (0, (5, 2.4)),
    "Mean": "-",
    "NormClipping": "-",
    "MultiKrum": "-",
    "FLTrust": "-",
    "FLDetector": "-",
    "CARAT": "-",
}

FINAL_PROTOCOL_DEFENSES = {"FLTrust", "CARAT"}
FINAL_CFG = "FedAvg_CIFAR100_Resnet18_protocol"
LEGACY_CFG = "FedAvg_CIFAR100_config"

SHADE_ALPHA = {
    "NoAttack+Mean": 0.045,
    "Mean": 0.035,
    "NormClipping": 0.035,
    "MultiKrum": 0.04,
    "FLTrust": 0.04,
    "FLDetector": 0.045,
    "CARAT": 0.16,
}


def parse_legacy_series(path: Path, expected_rounds: int) -> np.ndarray | None:
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


def parse_csv_series(path: Path, expected_rounds: int) -> np.ndarray | None:
    values: list[tuple[int, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch_text = (row.get("epoch") or "").strip()
            acc_text = (row.get("eval_acc") or "").strip()
            if not epoch_text or not acc_text:
                continue
            values.append((int(float(epoch_text)), 100.0 * float(acc_text)))

    if not values:
        return None
    values.sort(key=lambda item: item[0])
    if len(values) < expected_rounds or values[-1][0] < expected_rounds - 1:
        return None

    arr = np.full(expected_rounds, np.nan, dtype=np.float64)
    for epoch, test_acc in values:
        if 0 <= epoch < expected_rounds:
            arr[epoch] = test_acc
    if np.isnan(arr).any():
        return None
    return arr


def collect_local_csv(
    result_root: Path,
    alpha: str,
    attack: str,
    defense: str,
    expected_rounds: int,
) -> list[np.ndarray]:
    pattern = (
        "FedAvg/CIFAR100_resnet18/non-iid/"
        f"{attack}__{defense}/"
        f"ep{expected_rounds}_clients20_lr0.05_adv0.2_seed*_exp*_alpha{alpha}_"
        "cfgFedAvg_CIFAR100_Resnet18_protocol/metrics_exp*.csv"
    )
    series: list[np.ndarray] = []
    for path in sorted(result_root.glob(pattern)):
        if not (path.parent / "task.complete").exists():
            continue
        parsed = parse_csv_series(path, expected_rounds)
        if parsed is not None:
            series.append(parsed)
    return series


def collect_protocol_txt(
    root: Path,
    alpha: str,
    attack: str,
    defense: str,
    expected_rounds: int,
) -> list[np.ndarray]:
    alpha_tokens = ("1", "1.0") if alpha == "1" else (alpha,)
    series: list[np.ndarray] = []
    for alpha_token in alpha_tokens:
        pattern = (
            f"CIFAR100_resnet18_non-iid_{attack}_{defense}_{expected_rounds}_20_0.05_"
            f"FedAvg_adv0.2_seed*_alpha{alpha_token}_cfg{FINAL_CFG}_exp*.txt"
        )
        for path in sorted(root.glob(pattern)):
            parsed = parse_legacy_series(path, expected_rounds)
            if parsed is not None:
                series.append(parsed)
    return series


def collect_legacy(
    root: Path,
    alpha: str,
    attack: str,
    defense: str,
    expected_rounds: int,
) -> list[np.ndarray]:
    alpha_tokens = ("1", "1.0") if alpha == "1" else (alpha,)
    series: list[np.ndarray] = []
    for alpha_token in alpha_tokens:
        pattern = (
            f"CIFAR100_resnet18_non-iid_{attack}_{defense}_{expected_rounds}_20_0.05_"
            f"FedAvg_adv0.2_seed*_alpha{alpha_token}_cfg{LEGACY_CFG}_exp*.txt"
        )
        for path in sorted(root.glob(pattern)):
            parsed = parse_legacy_series(path, expected_rounds)
            if parsed is not None:
                series.append(parsed)
    return series


def collect(
    playground_root: Path,
    result_root: Path,
    alpha: str,
    attack: str,
    defense: str,
    expected_rounds: int,
) -> list[np.ndarray]:
    root = playground_root / "FedAvg" / "CIFAR100_resnet18" / "non-iid"
    if defense in FINAL_PROTOCOL_DEFENSES:
        protocol = collect_protocol_txt(root, alpha, attack, defense, expected_rounds)
        if protocol:
            return protocol
    local = collect_local_csv(result_root, alpha, attack, defense, expected_rounds)
    if local:
        return local
    return collect_legacy(root, alpha, attack, defense, expected_rounds)


def mean_and_std(series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(series)
    ddof = 1 if len(series) > 1 else 0
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=ddof)


def plot(output: Path, playground_root: Path, result_root: Path, expected_rounds: int) -> None:
    rounds = np.arange(expected_rounds)

    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.7,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.9,
            "legend.fontsize": 6.9,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(6.9, 2.34), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.56], wspace=0.05)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.set_axis_off()

    legend_handles = {}

    for ax, alpha in zip(axes, ALPHAS):
        plotted_values: list[np.ndarray] = []

        clean_series = collect(playground_root, result_root, alpha, "NoAttack", "Mean", expected_rounds)
        if clean_series:
            clean_mean, clean_std = mean_and_std(clean_series)
            (line,) = ax.plot(
                rounds,
                clean_mean,
                color=COLORS["NoAttack+Mean"],
                linestyle=LINESTYLES["NoAttack+Mean"],
                linewidth=1.75,
                label=f"NoAttack+Mean (n={len(clean_series)})",
                zorder=2,
            )
            ax.fill_between(
                rounds,
                clean_mean - clean_std,
                clean_mean + clean_std,
                color=COLORS["NoAttack+Mean"],
                alpha=SHADE_ALPHA["NoAttack+Mean"],
                linewidth=0,
                zorder=0,
            )
            legend_handles.setdefault("NoAttack+Mean", line)
            plotted_values.append(clean_mean)

        for defense in DEFENSES:
            series = collect(playground_root, result_root, alpha, ATTACK, defense, expected_rounds)
            if not series:
                continue
            avg, std = mean_and_std(series)
            linewidth = 2.6 if defense == "CARAT" else 1.42
            alpha_line = 1.0 if defense == "CARAT" else 0.76
            (line,) = ax.plot(
                rounds,
                avg,
                color=COLORS[defense],
                linestyle=LINESTYLES[defense],
                linewidth=linewidth,
                alpha=alpha_line,
                label=f"{defense} (n={len(series)})",
                zorder=5 if defense == "CARAT" else 3,
            )
            band_zorder = 4 if defense == "CARAT" else 1
            ax.fill_between(
                rounds,
                avg - std,
                avg + std,
                color=COLORS[defense],
                alpha=SHADE_ALPHA[defense],
                linewidth=0,
                zorder=band_zorder,
            )
            legend_handles.setdefault(defense, line)
            plotted_values.append(avg)

        ax.set_title(rf"Dirichlet $\alpha={alpha}$", pad=3)
        ax.set_xlabel("Round")
        ax.grid(axis="y", color="#E4E4E4", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color("#3A3A3A")
        ax.spines["bottom"].set_color("#3A3A3A")
        if plotted_values:
            stacked = np.vstack(plotted_values)
            ymax = min(60.0, float(np.nanmax(stacked)) + 3.0)
            ax.set_ylim(0.0, ymax)
        ax.set_xlim(0, expected_rounds)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_yticks([0, 20, 40, 60])

    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].set_ylabel("")

    ordered_labels = ["CARAT", "NoAttack+Mean", "Mean", "NormClipping", "MultiKrum", "FLTrust", "FLDetector"]
    handles = [legend_handles[label] for label in ordered_labels if label in legend_handles]
    labels = [handle.get_label() for handle in handles]
    legend_ax.legend(
        handles,
        labels,
        loc="center left",
        title="Defense",
        title_fontsize=7.1,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.36,
        borderaxespad=0.0,
    )

    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)


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
    parser.add_argument("--result-root", type=Path, default=repo_root / "logs" / "local_runs")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--rounds", type=int, default=200)
    args = parser.parse_args()

    plot(args.output, args.playground_root, args.result_root, args.rounds)
    print(args.output)


if __name__ == "__main__":
    main()
