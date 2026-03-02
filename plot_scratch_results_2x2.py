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
ATTACK_COLORS = {
    "ALIE": "#1f77b4",
    "FangAttack": "#ff7f0e",
    "MinMax": "#2ca02c",
    "MinSum": "#d62728",
}
DEFENSE_COLORS = {
    "Mean": "#1f77b4",
    "NormClipping": "#ff7f0e",
    "MultiKrum": "#2ca02c",
    "FLTrust": "#d62728",
    "FLDetector": "#9467bd",
    "TriGuardFL": "#8c564b",
}
BASELINE_COLOR = "#111111"

# Group key excludes seed/expid by default, so repeated runs can be averaged.
# By default epochs is not in group key (can merge 200/300 logs while plotting first 200 epochs).
BASE_GROUP_FIELDS = [
    "iid",
    "dataset",
    "model",
    "num_clients",
    "lr",
    "algo",
    "adv",
    "alpha",
    "cfg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch plot FL_Poison logs into 1x4 figures: one attack per subplot, "
            "defenses as curves, and repeated seeds aggregated as mean±std."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/logs"),
        help="Root folder containing *.txt logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/plots_2x2"),
        help="Output folder. All figures are written into this single directory.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=200,
        help="Only epochs < max-epoch are plotted.",
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
        "--min-defenses-per-attack",
        type=int,
        default=2,
        help="Require at least this many defenses for each attack in one figure.",
    )
    parser.add_argument(
        "--allow-partial-attacks",
        action="store_true",
        help="Allow figures when some requested attacks are missing.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--merge-iid-noniid",
        action="store_true",
        help=(
            "Deprecated and kept only for backward compatibility. "
            "The script always writes all figures into one folder."
        ),
    )
    parser.add_argument(
        "--group-by-epochs",
        action="store_true",
        help=(
            "If set, keep different configured epochs (e.g., 200 vs 300) in separate groups. "
            "Default is to merge them."
        ),
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


def parse_test_acc_curve(path: Path, max_epoch: int):
    # epoch -> acc (if duplicate epoch appears, keep the latest one)
    curve = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EPOCH_ACC_RE.search(line)
            if not match:
                continue
            epoch = int(match.group("epoch"))
            if epoch >= max_epoch:
                continue
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


def aggregate_mean_std(curves):
    per_epoch = defaultdict(list)
    for curve in curves:
        for epoch, acc in curve.items():
            per_epoch[epoch].append(acc)
    if not per_epoch:
        return None, None, None

    epochs = np.array(sorted(per_epoch.keys()), dtype=float)
    means = np.array([np.mean(per_epoch[int(e)]) for e in epochs], dtype=float)
    stds = np.array([np.std(per_epoch[int(e)]) for e in epochs], dtype=float)
    return epochs, means, stds


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def defense_sort_key(name: str):
    if name in DEFAULT_DEFENSE_ORDER:
        return DEFAULT_DEFENSE_ORDER.index(name), name
    return len(DEFAULT_DEFENSE_ORDER), name


def get_group_fields(args):
    fields = list(BASE_GROUP_FIELDS)
    if args.group_by_epochs:
        fields.append("epochs")
    return fields


def build_group_key(meta: dict, group_fields):
    return tuple((k, meta[k]) for k in group_fields)


def key_to_dict(key):
    return dict(key)


def make_output_name(info: dict, _group_fields, epochs_tag: str):
    epochs_token = sanitize(str(epochs_tag).replace(",", "-"))
    base = (
        f"{info['iid']}_{info['dataset']}_{info['model']}_{info['algo']}"
        f"_adv{info['adv']}_alpha{info['alpha']}"
        f"_clients{info['num_clients']}_lr{info['lr']}"
        f"_cfg{info['cfg']}_epochs{epochs_token}"
    )
    return sanitize(base)


def tint(color: str, alpha: float = 0.06):
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
    group_fields = get_group_fields(args)
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

    curve_cache = {}

    def load_curve(path_str: str):
        if path_str not in curve_cache:
            curve_cache[path_str] = parse_test_acc_curve(Path(path_str), args.max_epoch)
        return curve_cache[path_str]

    all_txt = sorted(args.logs_root.rglob("*.txt"))
    parsed = []
    unparsed = []
    filtered_by_seed = 0
    filtered_by_epoch_cfg = 0
    filtered_buggy_fltrust = 0
    for path in all_txt:
        meta = parse_log_name(path)
        if meta is None:
            unparsed.append(str(path))
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for meta in parsed:
        grouped[build_group_key(meta, group_fields)].append(meta)

    csv_rows = []
    skipped_missing_attacks = 0
    skipped_insufficient_defenses = 0
    skipped_empty_curves = 0
    generated = 0

    for group_key, records in grouped.items():
        by_attack = defaultdict(lambda: defaultdict(list))
        baseline_candidates = []
        for meta in records:
            if (
                meta["attack"] == BASELINE_ATTACK
                and meta["defense"] == BASELINE_DEFENSE
            ):
                baseline_candidates.append(meta)
            if meta["attack"] in attacks:
                by_attack[meta["attack"]][meta["defense"]].append(meta)

        if not args.allow_partial_attacks:
            if any(attack not in by_attack for attack in attacks):
                skipped_missing_attacks += 1
                continue

        if args.min_defenses_per_attack > 0:
            ok = True
            for attack in attacks:
                n_defs = len(
                    [
                        d
                        for d in by_attack.get(attack, {})
                        if d not in EXCLUDED_ATTACK_DEFENSES
                    ]
                )
                if n_defs < args.min_defenses_per_attack:
                    ok = False
                    break
            if not ok:
                skipped_insufficient_defenses += 1
                continue

        baseline_plot_data = None
        if baseline_candidates:
            by_seed = defaultdict(list)
            for rec in baseline_candidates:
                by_seed[rec["seed"]].append(rec)
            picked = []
            for seed, same_seed_records in by_seed.items():
                chosen = sorted(same_seed_records, key=lambda r: r["file"])[-1]
                picked.append((seed, chosen))
            picked.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

            curves = [load_curve(rec["file"]) for _, rec in picked]
            curves = [c for c in curves if c]
            if curves:
                epochs, means, stds = aggregate_mean_std(curves)
                if epochs is not None:
                    baseline_plot_data = {
                        "epochs": epochs,
                        "means": means,
                        "stds": stds,
                        "n": len(curves),
                        "seeds": [seed for seed, _ in picked],
                    }

        fig, axes = plt.subplots(
            1, 4, figsize=INITIAL_FIGSIZE, sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes).flatten()
        plotted_any = False
        used_seeds = set()
        legend_handles = {}

        per_curve_seed_counts = []

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
            ax.set_xlabel("Epoch", fontsize=BASE_FONT_SIZE * FONT_SCALE)
            ax.set_ylabel("Test Accuracy", fontsize=BASE_FONT_SIZE * FONT_SCALE)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=BASE_FONT_SIZE * FONT_SCALE)

            defense_records = by_attack.get(attack, {})
            plotted_on_ax = False
            for defense in sorted(defense_records.keys(), key=defense_sort_key):
                if defense in EXCLUDED_ATTACK_DEFENSES:
                    continue
                # Use at most one run per seed. If duplicated logs exist for the same seed,
                # keep the lexicographically latest path deterministically.
                by_seed = defaultdict(list)
                for rec in defense_records[defense]:
                    by_seed[rec["seed"]].append(rec)
                picked = []
                for seed, same_seed_records in by_seed.items():
                    chosen = sorted(same_seed_records, key=lambda r: r["file"])[-1]
                    picked.append((seed, chosen))
                picked.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

                curves = [load_curve(rec["file"]) for _, rec in picked]
                curves = [c for c in curves if c]
                if not curves:
                    continue

                epochs, means, stds = aggregate_mean_std(curves)
                if epochs is None:
                    continue

                n_seeds = len(curves)
                color = DEFENSE_COLORS.get(defense, None)
                line = ax.plot(
                    epochs,
                    means,
                    label=defense,
                    linewidth=1.8,
                    color=color,
                )[0]
                lower = np.clip(means - stds, 0.0, 1.0)
                upper = np.clip(means + stds, 0.0, 1.0)
                ax.fill_between(epochs, lower, upper, color=color, alpha=0.16, linewidth=0)
                if defense not in legend_handles:
                    legend_handles[defense] = line

                used_seeds.update(seed for seed, _ in picked)
                per_curve_seed_counts.append(n_seeds)
                plotted_on_ax = True
                plotted_any = True

            if baseline_plot_data is not None:
                b_epochs = baseline_plot_data["epochs"]
                b_means = baseline_plot_data["means"]
                b_stds = baseline_plot_data["stds"]
                b_n = baseline_plot_data["n"]
                b_lower = np.clip(b_means - b_stds, 0.0, 1.0)
                b_upper = np.clip(b_means + b_stds, 0.0, 1.0)
                b_line = ax.plot(
                    b_epochs,
                    b_means,
                    label=BASELINE_LABEL,
                    linewidth=2.0,
                    color=BASELINE_COLOR,
                    linestyle="--",
                )[0]
                ax.fill_between(
                    b_epochs,
                    b_lower,
                    b_upper,
                    color=BASELINE_COLOR,
                    alpha=0.12,
                    linewidth=0,
                )
                if BASELINE_LABEL not in legend_handles:
                    legend_handles[BASELINE_LABEL] = b_line
                used_seeds.update(baseline_plot_data["seeds"])
                per_curve_seed_counts.append(b_n)
                plotted_on_ax = True
                plotted_any = True

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
            skipped_empty_curves += 1
            continue

        info = key_to_dict(group_key)
        epoch_values = sorted(
            {r["epochs"] for r in records},
            key=lambda x: int(x) if str(x).isdigit() else str(x),
        )
        epochs_tag = ",".join(epoch_values)
        seeds_used = sorted(used_seeds, key=lambda x: int(x) if x.isdigit() else x)
        min_curve_n = min(per_curve_seed_counts) if per_curve_seed_counts else 0
        max_curve_n = max(per_curve_seed_counts) if per_curve_seed_counts else 0
        title = (
            f"{info['iid']} | {info['dataset']}_{info['model']} | {info['algo']} | "
            f"adv={info['adv']} | alpha={info['alpha']} | epochs_src={epochs_tag} | "
            f"cfg={info['cfg']} | mean±std over available seeds"
        )
        if seeds_used:
            title += f" ({','.join(seeds_used)}; curve n={min_curve_n}-{max_curve_n})"

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

        out_name = make_output_name(info, group_fields, epochs_tag)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        png_path = args.out_dir / f"{out_name}.png"
        pdf_path = args.out_dir / f"{out_name}.pdf"
        fig.savefig(png_path, dpi=args.dpi)
        fig.savefig(pdf_path, format="pdf")
        plt.close(fig)

        row = {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}
        row.update(info)
        row["epochs_tag"] = epochs_tag
        row["seeds_used"] = ",".join(seeds_used)
        row["n_seeds"] = len(seeds_used)
        row["min_curve_n"] = min_curve_n
        row["max_curve_n"] = max_curve_n
        row["baseline_n"] = baseline_plot_data["n"] if baseline_plot_data else 0
        csv_rows.append(row)
        generated += 1

    csv_path = args.out_dir / "plot_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "figure_png",
            "figure_pdf",
            *group_fields,
            "epochs_tag",
            "seeds_used",
            "n_seeds",
            "min_curve_n",
            "max_curve_n",
            "baseline_n",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary_lines = [
        f"log_root={args.logs_root}",
        f"output_dir={args.out_dir}",
        f"total_txt={len(all_txt)}",
        f"parsed_after_seed_filter={len(parsed)}",
        f"filtered_by_seed={filtered_by_seed}",
        f"filtered_by_epoch_cfg={filtered_by_epoch_cfg}",
        f"filtered_buggy_fltrust={filtered_buggy_fltrust}",
        f"unparsed_txt={len(unparsed)}",
        f"groups_total={len(grouped)}",
        f"generated_figures={generated}",
        f"skipped_missing_attacks={skipped_missing_attacks}",
        f"skipped_insufficient_defenses={skipped_insufficient_defenses}",
        f"skipped_empty_curves={skipped_empty_curves}",
        f"max_epoch={args.max_epoch}",
        f"attacks={','.join(attacks)}",
        f"seed_filter={','.join(sorted(seed_filter)) if seed_filter else 'ALL'}",
        f"min_defenses_per_attack={args.min_defenses_per_attack}",
        "flat_output=True",
        f"group_by_epochs={args.group_by_epochs}",
        f"exclude_config_epochs={','.join(sorted(excluded_config_epochs)) if excluded_config_epochs else 'NONE'}",
        f"filter_buggy_fltrust={args.filter_buggy_fltrust}",
        f"subplot_target_ratio={TARGET_SUBPLOT_RATIO}",
        f"subplot_target_height_in={TARGET_SUBPLOT_HEIGHT_INCH}",
    ]
    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"index_csv={csv_path}")
    if unparsed:
        sample = "\n".join(unparsed[:20])
        print("unparsed_samples:\n" + sample)


if __name__ == "__main__":
    main()
