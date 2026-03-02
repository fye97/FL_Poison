#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

INITIAL_FIGSIZE = (24, 8)
FONT_SCALE = 2.0
BASE_FONT_SIZE = 10.0
AXES_TITLE_SIZE = 12.0
SUP_TITLE_SIZE = 11.0
LEGEND_SIZE = 9.0
TARGET_SUBPLOT_RATIO = 4.0 / 3.0
TARGET_SUBPLOT_HEIGHT_INCH = 3.6
LAYOUT_RECT = [0.0, 0.08, 1.0, 0.95]

plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE * FONT_SCALE,
        "axes.labelsize": BASE_FONT_SIZE * FONT_SCALE,
        "xtick.labelsize": BASE_FONT_SIZE * FONT_SCALE,
        "ytick.labelsize": BASE_FONT_SIZE * FONT_SCALE,
    }
)

FILENAME_RE = re.compile(
    r"^(?P<dataset>[^_]+)_(?P<model>[^_]+)_(?P<iid>iid|non-iid)_(?P<attack>[^_]+)_(?P<defense>[^_]+)"
    r"_(?P<epochs>\d+)_(?P<num_clients>\d+)_(?P<lr>[0-9.]+)_(?P<algo>[^_]+)_adv(?P<adv>[0-9.]+)"
    r"_seed(?P<seed>\d+)(?:_alpha(?P<alpha>[0-9.]+))?_cfg(?P<cfg>.+?)(?:_exp(?P<expid>\d+))?\.txt$"
)
EPOCH_ACC_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+.*?Test Acc:\s*(?P<acc>[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
)

DEFAULT_ATTACKS = ["ALIE", "FangAttack", "MinMax", "MinSum"]
DEFAULT_DEFENSE_ORDER = [
    "Mean",
    "NormClipping",
    "MultiKrum",
    "FLTrust",
    "FLDetector",
    "TriGuardFL",
]
EXCLUDED_ATTACK_DEFENSES = {"Mean"}
BASELINE_ATTACK = "NoAttack"
BASELINE_DEFENSE = "Mean"
BASELINE_LABEL = "NoAttack+Mean(Base)"
DEFENSE_COLORS = {
    "Mean": "#1f77b4",
    "NormClipping": "#ff7f0e",
    "MultiKrum": "#2ca02c",
    "FLTrust": "#d62728",
    "FLDetector": "#9467bd",
    "TriGuardFL": "#8c564b",
}
BASELINE_COLOR = "#111111"
ATTACK_COLORS = {
    "ALIE": "#1f77b4",
    "FangAttack": "#ff7f0e",
    "MinMax": "#2ca02c",
    "MinSum": "#d62728",
}
GROUP_FIELDS = [
    "dataset",
    "model",
    "num_clients",
    "lr",
    "algo",
    "adv",
    "cfg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot non-iid alpha sensitivity using test accuracy at target epoch. "
            "Each figure is 1x4 attacks, with defenses as alpha-accuracy lines."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/logs_from_slurm_split"),
        help="Root folder containing split *.txt logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/plots_alpha_non_iid_epoch200"),
        help="Output folder. All figures are written into this single directory.",
    )
    parser.add_argument(
        "--target-epoch",
        type=int,
        default=199,
        help=(
            "Strict target epoch for test accuracy. A run is used only when "
            "its max logged epoch >= target and it contains the exact target epoch."
        ),
    )
    parser.add_argument(
        "--attacks",
        type=str,
        default=",".join(DEFAULT_ATTACKS),
        help="Comma-separated list of attacks to place in 1x4 subplots.",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default="42,43,44,45,46",
        help=(
            "Comma-separated seeds used as repeated runs for mean/std. "
            "Set to empty string to include all seeds."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG output DPI.",
    )
    parser.add_argument(
        "--exclude-config-epochs",
        type=str,
        default="",
        help=(
            "Comma-separated configured epochs values to exclude by filename metadata, "
            "e.g. '300'."
        ),
    )
    parser.add_argument(
        "--filter-buggy-fltrust",
        action="store_true",
        help=(
            "Filter FLTrust runs where tail test accuracy is suspiciously around 0.1."
        ),
    )
    parser.add_argument(
        "--fltrust-tail-low",
        type=float,
        default=0.09,
        help="Lower bound of suspicious FLTrust tail mean test accuracy.",
    )
    parser.add_argument(
        "--fltrust-tail-high",
        type=float,
        default=0.11,
        help="Upper bound of suspicious FLTrust tail mean test accuracy.",
    )
    parser.add_argument(
        "--fltrust-tail-span-max",
        type=float,
        default=0.04,
        help="Max allowed tail span (max-min) for suspicious FLTrust run.",
    )
    return parser.parse_args()


def parse_log_name(path: Path):
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    meta = match.groupdict()
    if meta["alpha"] is None:
        meta["alpha"] = "NA"
    if meta["expid"] is None:
        meta["expid"] = "NA"
    meta["file"] = str(path)
    return meta


def parse_test_acc_curve(path: Path):
    curve = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EPOCH_ACC_RE.search(line)
            if not match:
                continue
            epoch = int(match.group("epoch"))
            curve[epoch] = float(match.group("acc"))
    return curve


def is_buggy_fltrust(curve: dict, low: float, high: float, span_max: float):
    if not curve:
        return False
    epochs = sorted(curve.keys())
    accs = [curve[e] for e in epochs]
    tail = accs[-20:] if len(accs) >= 20 else accs
    tail_mean = float(np.mean(tail))
    tail_span = float(np.max(tail) - np.min(tail))
    return low <= tail_mean <= high and tail_span <= span_max


def get_acc_at_target(curve: dict, target_epoch: int):
    if not curve:
        return None, "empty"
    max_epoch = max(curve.keys())
    if max_epoch < target_epoch:
        return None, "short"
    if target_epoch not in curve:
        return None, "missing_target"
    return curve[target_epoch], "ok"


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def defense_sort_key(name: str):
    if name in DEFAULT_DEFENSE_ORDER:
        return DEFAULT_DEFENSE_ORDER.index(name), name
    return len(DEFAULT_DEFENSE_ORDER), name


def group_key(meta: dict):
    return tuple((k, meta[k]) for k in GROUP_FIELDS)


def key_to_dict(key):
    return dict(key)


def make_output_name(info: dict, epochs_tag: str):
    base = (
        f"alpha_sensitivity_non-iid_{info['dataset']}_{info['model']}_{info['algo']}"
        f"_adv{info['adv']}_clients{info['num_clients']}_lr{info['lr']}_cfg{info['cfg']}"
        f"_epochs{sanitize(epochs_tag.replace(',', '-'))}"
    )
    return sanitize(base)


def tint(color: str, alpha: float = 0.05):
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)


def apply_adaptive_figsize(fig, axes, max_iter: int = 4):
    # Adapt figure size so each subplot is approximately 4:3 while preserving
    # room for suptitle and bottom legend.
    for _ in range(max_iter):
        fig.tight_layout(rect=LAYOUT_RECT)
        fig.canvas.draw()

        fw, fh = fig.get_size_inches()
        widths = []
        heights = []
        for ax in axes:
            pos = ax.get_position()
            widths.append(pos.width)
            heights.append(pos.height)

        w_frac = float(np.mean(widths)) if widths else 0.0
        h_frac = float(np.mean(heights)) if heights else 0.0
        if w_frac <= 0.0 or h_frac <= 0.0:
            break

        target_h = TARGET_SUBPLOT_HEIGHT_INCH / h_frac
        target_w = (TARGET_SUBPLOT_RATIO * TARGET_SUBPLOT_HEIGHT_INCH) / w_frac
        new_w = 0.5 * fw + 0.5 * target_w
        new_h = 0.5 * fh + 0.5 * target_h

        if (
            abs(new_w - fw) / max(fw, 1e-6) < 0.01
            and abs(new_h - fh) / max(fh, 1e-6) < 0.01
        ):
            fig.set_size_inches(target_w, target_h, forward=True)
            break
        fig.set_size_inches(new_w, new_h, forward=True)

    fig.tight_layout(rect=LAYOUT_RECT)


def main():
    args = parse_args()
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    if len(attacks) != 4:
        raise ValueError("For 1x4 plots, --attacks must contain exactly 4 attacks.")
    if not args.logs_root.exists():
        raise FileNotFoundError(f"Log directory not found: {args.logs_root}")

    if args.seed_list.strip():
        seed_filter = {s.strip() for s in args.seed_list.split(",") if s.strip()}
    else:
        seed_filter = set()
    if args.exclude_config_epochs.strip():
        excluded_config_epochs = {
            e.strip() for e in args.exclude_config_epochs.split(",") if e.strip()
        }
    else:
        excluded_config_epochs = set()

    all_txt = sorted(args.logs_root.rglob("*.txt"))
    parsed = []
    unparsed = []
    filtered_by_seed = 0
    filtered_by_epoch_cfg = 0
    filtered_buggy_fltrust = 0
    curve_cache = {}

    def load_curve(path_str: str):
        if path_str not in curve_cache:
            curve_cache[path_str] = parse_test_acc_curve(Path(path_str))
        return curve_cache[path_str]

    for path in all_txt:
        meta = parse_log_name(path)
        if meta is None:
            unparsed.append(str(path))
            continue
        if meta["iid"] != "non-iid":
            continue
        if meta["alpha"] == "NA":
            continue
        if seed_filter and meta["seed"] not in seed_filter:
            filtered_by_seed += 1
            continue
        if excluded_config_epochs and meta["epochs"] in excluded_config_epochs:
            filtered_by_epoch_cfg += 1
            continue
        if args.filter_buggy_fltrust and meta["defense"] == "FLTrust":
            curve = load_curve(str(path))
            if is_buggy_fltrust(
                curve,
                args.fltrust_tail_low,
                args.fltrust_tail_high,
                args.fltrust_tail_span_max,
            ):
                filtered_buggy_fltrust += 1
                continue
        parsed.append(meta)

    grouped = defaultdict(list)
    for meta in parsed:
        grouped[group_key(meta)].append(meta)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary_rows = []
    generated = 0
    skipped_no_points = 0
    discarded_short_runs = 0
    discarded_missing_target_runs = 0

    for gk, records in grouped.items():
        info = key_to_dict(gk)

        alpha_values = sorted(
            {float(r["alpha"]) for r in records if r["alpha"] not in {"NA", ""}}
        )
        if not alpha_values:
            skipped_no_points += 1
            continue
        alpha_to_str = {a: str(a).rstrip("0").rstrip(".") if "." in str(a) else str(a) for a in alpha_values}

        by_attack_def_alpha = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        baseline_by_alpha = defaultdict(list)
        epochs_seen = set()
        for rec in records:
            epochs_seen.add(rec["epochs"])
            alpha = float(rec["alpha"])
            if rec["attack"] == BASELINE_ATTACK and rec["defense"] == BASELINE_DEFENSE:
                baseline_by_alpha[alpha].append(rec)
            if rec["attack"] in attacks:
                by_attack_def_alpha[rec["attack"]][rec["defense"]][alpha].append(rec)

        fig, axes = plt.subplots(
            1, 4, figsize=INITIAL_FIGSIZE, sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes).flatten()
        plotted_any = False
        legend_handles = {}
        seeds_used = set()
        curve_ns = []

        for ax, attack in zip(axes, attacks):
            attack_color = ATTACK_COLORS.get(attack, "#444444")
            ax.set_facecolor(tint(attack_color, alpha=0.05))
            for spine in ax.spines.values():
                spine.set_color(attack_color)
                spine.set_alpha(0.35)
            ax.set_title(
                attack,
                color=attack_color,
                fontweight="bold",
                fontsize=AXES_TITLE_SIZE * FONT_SCALE,
            )
            ax.set_xlabel("Alpha (non-iid)", fontsize=BASE_FONT_SIZE * FONT_SCALE)
            ax.set_ylabel("Test Accuracy", fontsize=BASE_FONT_SIZE * FONT_SCALE)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=BASE_FONT_SIZE * FONT_SCALE)

            plotted_on_ax = False
            defense_to_alpha_stats = defaultdict(dict)
            for defense, alpha_map in by_attack_def_alpha.get(attack, {}).items():
                if defense in EXCLUDED_ATTACK_DEFENSES:
                    continue
                for alpha, recs in alpha_map.items():
                    by_seed = defaultdict(list)
                    for rec in recs:
                        by_seed[rec["seed"]].append(rec)

                    vals = []
                    for seed, same_seed_recs in by_seed.items():
                        chosen = sorted(same_seed_recs, key=lambda r: r["file"])[-1]
                        curve = load_curve(chosen["file"])
                        acc, status = get_acc_at_target(curve, args.target_epoch)
                        if status == "short":
                            discarded_short_runs += 1
                            continue
                        if status == "missing_target":
                            discarded_missing_target_runs += 1
                            continue
                        if acc is None:
                            continue
                        vals.append(acc)
                        seeds_used.add(seed)
                    if vals:
                        defense_to_alpha_stats[defense][alpha] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                            "n": len(vals),
                        }
                        curve_ns.append(len(vals))
                        plotted_on_ax = True
                        plotted_any = True

            for defense in sorted(defense_to_alpha_stats.keys(), key=defense_sort_key):
                points = sorted(defense_to_alpha_stats[defense].items(), key=lambda kv: kv[0])
                x = np.array([p[0] for p in points], dtype=float)
                y = np.array([p[1]["mean"] for p in points], dtype=float)
                yerr = np.array([p[1]["std"] for p in points], dtype=float)
                nmin = min(p[1]["n"] for p in points)
                nmax = max(p[1]["n"] for p in points)
                n_tag = f"{nmin}" if nmin == nmax else f"{nmin}-{nmax}"
                color = DEFENSE_COLORS.get(defense, None)
                line = ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.8,
                    markersize=4.5,
                    color=color,
                    label=f"{defense} (n={n_tag})",
                )[0]
                ax.fill_between(
                    x,
                    np.clip(y - yerr, 0.0, 1.0),
                    np.clip(y + yerr, 0.0, 1.0),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
                if defense not in legend_handles:
                    legend_handles[defense] = line

            baseline_points = []
            for alpha in alpha_values:
                recs = baseline_by_alpha.get(alpha, [])
                by_seed = defaultdict(list)
                for rec in recs:
                    by_seed[rec["seed"]].append(rec)
                vals = []
                for seed, same_seed_recs in by_seed.items():
                    chosen = sorted(same_seed_recs, key=lambda r: r["file"])[-1]
                    curve = load_curve(chosen["file"])
                    acc, status = get_acc_at_target(curve, args.target_epoch)
                    if status == "short":
                        discarded_short_runs += 1
                        continue
                    if status == "missing_target":
                        discarded_missing_target_runs += 1
                        continue
                    if acc is None:
                        continue
                    vals.append(acc)
                    seeds_used.add(seed)
                if vals:
                    baseline_points.append(
                        (alpha, float(np.mean(vals)), float(np.std(vals)), len(vals))
                    )
                    curve_ns.append(len(vals))
                    plotted_on_ax = True
                    plotted_any = True

            if baseline_points:
                baseline_points = sorted(baseline_points, key=lambda x: x[0])
                bx = np.array([p[0] for p in baseline_points], dtype=float)
                by = np.array([p[1] for p in baseline_points], dtype=float)
                bs = np.array([p[2] for p in baseline_points], dtype=float)
                bn_min = min(p[3] for p in baseline_points)
                bn_max = max(p[3] for p in baseline_points)
                bn_tag = f"{bn_min}" if bn_min == bn_max else f"{bn_min}-{bn_max}"
                bline = ax.plot(
                    bx,
                    by,
                    marker="s",
                    linewidth=2.0,
                    markersize=4.5,
                    linestyle="--",
                    color=BASELINE_COLOR,
                    label=f"{BASELINE_LABEL} (n={bn_tag})",
                )[0]
                ax.fill_between(
                    bx,
                    np.clip(by - bs, 0.0, 1.0),
                    np.clip(by + bs, 0.0, 1.0),
                    color=BASELINE_COLOR,
                    alpha=0.12,
                    linewidth=0,
                )
                if BASELINE_LABEL not in legend_handles:
                    legend_handles[BASELINE_LABEL] = bline

            ax.set_xticks(alpha_values)
            ax.set_xticklabels([alpha_to_str[a] for a in alpha_values])

            if not plotted_on_ax:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=BASE_FONT_SIZE * FONT_SCALE,
                    transform=ax.transAxes,
                )

        if not plotted_any:
            plt.close(fig)
            skipped_no_points += 1
            continue

        epochs_tag = ",".join(
            sorted(epochs_seen, key=lambda x: int(x) if str(x).isdigit() else str(x))
        )
        seeds_sorted = sorted(seeds_used, key=lambda x: int(x) if x.isdigit() else x)
        min_n = min(curve_ns) if curve_ns else 0
        max_n = max(curve_ns) if curve_ns else 0

        title = (
            f"non-iid alpha sensitivity | {info['dataset']}_{info['model']} | {info['algo']} | "
            f"adv={info['adv']} | target_epoch={args.target_epoch} | epochs_src={epochs_tag} | "
            f"cfg={info['cfg']} | mean±std over available seeds"
        )
        if seeds_sorted:
            title += f" ({','.join(seeds_sorted)}; curve n={min_n}-{max_n})"
        fig.suptitle(title, fontsize=SUP_TITLE_SIZE * FONT_SCALE)

        if legend_handles:
            legend_order = [
                d
                for d in DEFAULT_DEFENSE_ORDER
                if d not in EXCLUDED_ATTACK_DEFENSES and d in legend_handles
            ]
            if BASELINE_LABEL in legend_handles:
                legend_order.append(BASELINE_LABEL)
            extras = sorted(k for k in legend_handles if k not in legend_order)
            legend_order.extend(extras)
            fig.legend(
                [legend_handles[k] for k in legend_order],
                legend_order,
                loc="lower center",
                ncol=min(6, max(1, len(legend_order))),
                fontsize=LEGEND_SIZE * FONT_SCALE,
                frameon=True,
                bbox_to_anchor=(0.5, 0.01),
            )

        apply_adaptive_figsize(fig, axes)

        out_name = make_output_name(info, epochs_tag)
        png_path = args.out_dir / f"{out_name}.png"
        pdf_path = args.out_dir / f"{out_name}.pdf"
        fig.savefig(png_path, dpi=args.dpi)
        fig.savefig(pdf_path, format="pdf")
        plt.close(fig)

        rows.append(
            {
                "figure_png": str(png_path),
                "figure_pdf": str(pdf_path),
                **info,
                "epochs_tag": epochs_tag,
                "target_epoch": args.target_epoch,
                "seeds_used": ",".join(seeds_sorted),
                "n_seeds": len(seeds_sorted),
                "min_curve_n": min_n,
                "max_curve_n": max_n,
            }
        )
        generated += 1

    index_path = args.out_dir / "plot_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "figure_png",
            "figure_pdf",
            *GROUP_FIELDS,
            "epochs_tag",
            "target_epoch",
            "seeds_used",
            "n_seeds",
            "min_curve_n",
            "max_curve_n",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows.extend(
        [
            f"log_root={args.logs_root}",
            f"output_dir={args.out_dir}",
            f"total_txt={len(all_txt)}",
            f"parsed_non_iid_alpha_after_seed_filter={len(parsed)}",
            f"filtered_by_seed={filtered_by_seed}",
            f"filtered_by_epoch_cfg={filtered_by_epoch_cfg}",
            f"filtered_buggy_fltrust={filtered_buggy_fltrust}",
            f"unparsed_txt={len(unparsed)}",
            f"groups_total={len(grouped)}",
            f"generated_figures={generated}",
            f"skipped_no_points={skipped_no_points}",
            f"target_epoch={args.target_epoch}",
            f"discarded_short_runs_max_epoch_lt_target={discarded_short_runs}",
            f"discarded_missing_exact_target_epoch={discarded_missing_target_runs}",
            f"attacks={','.join(attacks)}",
            f"seed_filter={','.join(sorted(seed_filter)) if seed_filter else 'ALL'}",
            f"exclude_config_epochs={','.join(sorted(excluded_config_epochs)) if excluded_config_epochs else 'NONE'}",
            f"filter_buggy_fltrust={args.filter_buggy_fltrust}",
            f"subplot_target_ratio={TARGET_SUBPLOT_RATIO}",
            f"subplot_target_height_in={TARGET_SUBPLOT_HEIGHT_INCH}",
        ]
    )

    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_rows) + "\n", encoding="utf-8")

    print("\n".join(summary_rows))
    print(f"index_csv={index_path}")
    if unparsed:
        print("unparsed_samples:")
        for line in unparsed[:20]:
            print(line)


if __name__ == "__main__":
    main()
