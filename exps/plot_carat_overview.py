#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#333333"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height - 0.13,
        title,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 0.06,
        body,
        ha="center",
        va="center",
        fontsize=8.8,
        linespacing=1.18,
    )
    return box


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color="#333333",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output = (
        repo_root
        / "CARAT"
        / "Formatting_Instructions_For_NeurIPS_2026"
        / "figures"
        / "carat_overview.pdf"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(11.2, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.27
    w = 0.145
    h = 0.48
    gap = 0.025
    xs = [0.025 + i * (w + gap) for i in range(6)]

    boxes = [
        add_box(
            ax,
            (xs[0], y),
            w,
            h,
            "Client updates",
            "Model deltas\nfrom selected\nclients",
            "#eeeeee",
        ),
        add_box(
            ax,
            (xs[1], y),
            w,
            h,
            "Stage 1",
            "MAD clipping\nand common-radius\nevaluation",
            "#d9e8f6",
        ),
        add_box(
            ax,
            (xs[2], y),
            w,
            h,
            "Stage 2a",
            "Hidden class-balanced\ntasks produce\nutility certificates",
            "#d7ecd9",
        ),
        add_box(
            ax,
            (xs[3], y),
            w,
            h,
            "Stage 2b",
            "Coordinate-rank\nsubsamples estimate\nstructural anomaly",
            "#f1e3c4",
        ),
        add_box(
            ax,
            (xs[4], y),
            w,
            h,
            "Stage 3",
            "Capped-simplex\noptimization with\ntemporal prior",
            "#e2ddf0",
        ),
        add_box(
            ax,
            (xs[5], y),
            w,
            h,
            "Aggregation",
            "Weighted clipped\nupdate for the\nnext global model",
            "#f4d8d8",
        ),
    ]

    for left, right in zip(boxes[:-1], boxes[1:]):
        add_arrow(
            ax,
            (left.get_x() + left.get_width(), left.get_y() + left.get_height() / 2),
            (right.get_x(), right.get_y() + right.get_height() / 2),
        )

    ax.text(
        xs[2] + w / 2,
        0.12,
        "Small labeled server reference set",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#2b5d34",
    )
    add_arrow(ax, (xs[2] + w / 2, 0.17), (xs[2] + w / 2, y))

    ax.text(
        xs[4] + w / 2,
        0.91,
        r"Output weights $\alpha^\star$: high hidden-task utility, low rank penalty, bounded per-client mass",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#3b3561",
    )
    add_arrow(ax, (xs[4] + w / 2, 0.86), (xs[4] + w / 2, y + h))

    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
