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


FILENAME_RE = re.compile(
    r"^(?P<dataset>[^_]+)_(?P<model>[^_]+)_(?P<iid>iid|non-iid)_(?P<attack>[^_]+)_(?P<defense>[^_]+)"
    r"_(?P<epochs>\d+)_(?P<num_clients>\d+)_(?P<lr>[0-9.]+)_(?P<algo>[^_]+)_adv(?P<adv>[0-9.]+)"
    r"_seed(?P<seed>\d+)(?:_alpha(?P<alpha>[0-9.]+))?_cfg(?P<cfg>.+?)(?:_exp(?P<expid>\d+))?\.txt$"
)
EPOCH_RE = re.compile(r"^Epoch\s+(?P<epoch>\d+)\b")
NUM_ADV_RE = re.compile(r"\bnum_adv:\s*([0-9.]+)")
REP_THRESHOLD_RE = re.compile(r"'reputation_threshold':\s*([0-9.]+)")
FLOAT_RE = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")

ATTACKS = ["ALIE", "FangAttack", "MinMax", "MinSum"]
GROUP_FIELDS = [
    "iid",
    "dataset",
    "model",
    "num_clients",
    "lr",
    "algo",
    "cfg",
]
ADV_COLORS = {
    "0.1": "#1f77b4",
    "0.2": "#ff7f0e",
    "0.3": "#2ca02c",
    "0.4": "#d62728",
}
ALPHA_LINESTYLES = {
    "NA": "-",
    "0.1": "-",
    "0.5": "--",
    "1": ":",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot TriGuardFL reputation curves. Each figure is 2x4: top row benign "
            "mean reputation, bottom row malicious mean reputation; columns are attacks."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/logs_from_slurm_split_20260301"),
        help="Root folder containing split logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/home/fengye/scratch/FL_Poison/replot_from_slurm_no300_20260301/triguard_reputation_curves"
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=200,
        help="Only epochs < max-epoch are used.",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated seed filter. Empty means all seeds.",
    )
    parser.add_argument(
        "--exclude-config-epochs",
        type=str,
        default="300",
        help="Comma-separated configured epochs in filename to exclude.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG DPI.",
    )
    return parser.parse_args()


def parse_log_name(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    meta = m.groupdict()
    if meta["alpha"] is None:
        meta["alpha"] = "NA"
    if meta["expid"] is None:
        meta["expid"] = "NA"
    meta["file"] = str(path)
    return meta


def parse_header_params(first_line: str):
    num_adv_m = NUM_ADV_RE.search(first_line or "")
    rep_thr_m = REP_THRESHOLD_RE.search(first_line or "")
    num_adv = int(round(float(num_adv_m.group(1)))) if num_adv_m else None
    rep_thr = float(rep_thr_m.group(1)) if rep_thr_m else None
    return num_adv, rep_thr


def parse_reputation_curve(path: Path, max_epoch: int):
    # epoch -> np.ndarray(num_clients,)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    if not lines:
        return {}, None, None

    num_adv, rep_thr = parse_header_params(lines[0])
    curve = {}
    pending_rep = None
    collecting_rep = False
    rep_buf = ""

    for raw in lines:
        line = raw.strip()
        if collecting_rep:
            rep_buf += " " + line
            if "]" in line:
                vals = [float(x) for x in FLOAT_RE.findall(rep_buf)]
                pending_rep = np.asarray(vals, dtype=float) if vals else None
                collecting_rep = False
                rep_buf = ""
            continue

        if line.startswith("Reputation:"):
            rep_content = line.split("Reputation:", 1)[1].strip()
            if "]" in rep_content:
                vals = [float(x) for x in FLOAT_RE.findall(rep_content)]
                pending_rep = np.asarray(vals, dtype=float) if vals else None
            else:
                collecting_rep = True
                rep_buf = rep_content
            continue

        m_epoch = EPOCH_RE.match(line)
        if m_epoch and pending_rep is not None:
            e = int(m_epoch.group("epoch"))
            if e < max_epoch:
                curve[e] = pending_rep
            pending_rep = None

    return curve, num_adv, rep_thr


def sanitize(s: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def group_key(meta: dict):
    return tuple((k, meta[k]) for k in GROUP_FIELDS)


def key_to_dict(key):
    return dict(key)


def adv_alpha_label(adv: str, alpha: str):
    return f"adv={adv},a={alpha}"


def aggregate_epoch_mean_std(curves):
    # curves: list[dict epoch->float]
    by_epoch = defaultdict(list)
    for c in curves:
        for e, v in c.items():
            if np.isfinite(v):
                by_epoch[e].append(float(v))
    if not by_epoch:
        return None, None, None
    epochs = np.array(sorted(by_epoch.keys()), dtype=float)
    means = np.array([np.mean(by_epoch[int(e)]) for e in epochs], dtype=float)
    stds = np.array([np.std(by_epoch[int(e)]) for e in epochs], dtype=float)
    return epochs, means, stds


def main():
    args = parse_args()
    if args.seed_list.strip():
        seed_filter = {s.strip() for s in args.seed_list.split(",") if s.strip()}
    else:
        seed_filter = set()
    if args.exclude_config_epochs.strip():
        excluded_config_epochs = {
            x.strip() for x in args.exclude_config_epochs.split(",") if x.strip()
        }
    else:
        excluded_config_epochs = set()

    if not args.logs_root.exists():
        raise FileNotFoundError(f"logs root not found: {args.logs_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_txt = sorted(args.logs_root.rglob("*.txt"))
    parsed = []
    unparsed = []
    filtered_seed = 0
    filtered_epoch_cfg = 0
    filtered_not_triguard = 0
    filtered_not_attack4 = 0
    for path in all_txt:
        meta = parse_log_name(path)
        if meta is None:
            unparsed.append(str(path))
            continue
        if meta["defense"] != "TriGuardFL":
            filtered_not_triguard += 1
            continue
        if meta["attack"] not in ATTACKS:
            filtered_not_attack4 += 1
            continue
        if seed_filter and meta["seed"] not in seed_filter:
            filtered_seed += 1
            continue
        if excluded_config_epochs and meta["epochs"] in excluded_config_epochs:
            filtered_epoch_cfg += 1
            continue
        parsed.append(meta)

    grouped = defaultdict(list)
    for meta in parsed:
        grouped[group_key(meta)].append(meta)

    cache = {}

    def load(path_str: str):
        if path_str not in cache:
            cache[path_str] = parse_reputation_curve(Path(path_str), args.max_epoch)
        return cache[path_str]

    plot_rows = []
    generated = 0
    skipped = 0
    short_runs = 0

    for gk, records in grouped.items():
        info = key_to_dict(gk)
        # Deduplicate by (attack, adv, alpha, seed): keep lexicographically latest file
        dedup = {}
        for r in records:
            key = (r["attack"], r["adv"], r["alpha"], r["seed"])
            if key not in dedup or r["file"] > dedup[key]["file"]:
                dedup[key] = r

        # attack -> setting_label -> list[curve_dict] for benign/malicious
        benign_runs = defaultdict(lambda: defaultdict(list))
        malicious_runs = defaultdict(lambda: defaultdict(list))
        threshold_vals = []
        used_seeds = set()

        for (attack, adv, alpha, seed), meta in sorted(dedup.items()):
            curve_vec, num_adv, rep_thr = load(meta["file"])
            if rep_thr is not None:
                threshold_vals.append(rep_thr)
            if not curve_vec:
                short_runs += 1
                continue
            n_clients = int(meta["num_clients"])
            if num_adv is None:
                # fallback from adv ratio if header misses num_adv
                num_adv = int(round(float(meta["adv"]) * n_clients))
            num_adv = max(0, min(n_clients, int(num_adv)))
            mal_idx = np.arange(0, num_adv, dtype=int)
            ben_idx = np.arange(num_adv, n_clients, dtype=int)
            if mal_idx.size == 0 or ben_idx.size == 0:
                continue

            ben_curve = {}
            mal_curve = {}
            for epoch, rep in curve_vec.items():
                if rep.size < n_clients:
                    continue
                repv = rep[:n_clients]
                ben_curve[epoch] = float(np.mean(repv[ben_idx]))
                mal_curve[epoch] = float(np.mean(repv[mal_idx]))
            if not ben_curve or not mal_curve:
                continue

            label = adv_alpha_label(adv, alpha)
            benign_runs[attack][label].append(ben_curve)
            malicious_runs[attack][label].append(mal_curve)
            used_seeds.add(seed)

        if not benign_runs:
            skipped += 1
            continue

        fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharex=True, sharey=True)
        legend_handles = {}
        plotted_any = False

        # Stable setting order: adv asc, alpha NA/0.1/0.5/1
        def setting_sort_key(lbl: str):
            # lbl: adv=x,a=y
            m = re.match(r"adv=([0-9.]+),a=(.+)$", lbl)
            if not m:
                return (999, 999, lbl)
            adv_s, alpha_s = m.group(1), m.group(2)
            try:
                adv_v = float(adv_s)
            except Exception:
                adv_v = 999.0
            alpha_order = {"NA": -1, "0.1": 1, "0.5": 2, "1": 3}
            return (adv_v, alpha_order.get(alpha_s, 999), lbl)

        for col, attack in enumerate(ATTACKS):
            ax_top = axes[0, col]
            ax_bot = axes[1, col]
            ax_top.set_title(attack, fontweight="bold")
            ax_top.set_ylabel("Benign Reputation")
            ax_bot.set_ylabel("Malicious Reputation")
            ax_bot.set_xlabel("Epoch")
            ax_top.grid(alpha=0.25)
            ax_bot.grid(alpha=0.25)

            settings = sorted(
                set(benign_runs.get(attack, {}).keys())
                | set(malicious_runs.get(attack, {}).keys()),
                key=setting_sort_key,
            )
            for setting in settings:
                # parse style from label
                m = re.match(r"adv=([0-9.]+),a=(.+)$", setting)
                adv_s = m.group(1) if m else "0.1"
                alpha_s = m.group(2) if m else "NA"
                color = ADV_COLORS.get(adv_s, "#444444")
                ls = ALPHA_LINESTYLES.get(alpha_s, "-.")

                ben_epochs, ben_mean, ben_std = aggregate_epoch_mean_std(
                    benign_runs.get(attack, {}).get(setting, [])
                )
                mal_epochs, mal_mean, mal_std = aggregate_epoch_mean_std(
                    malicious_runs.get(attack, {}).get(setting, [])
                )
                if ben_epochs is None or mal_epochs is None:
                    continue

                h = ax_top.plot(
                    ben_epochs,
                    ben_mean,
                    color=color,
                    linestyle=ls,
                    linewidth=1.8,
                    label=setting,
                )[0]
                ax_top.fill_between(
                    ben_epochs,
                    np.clip(ben_mean - ben_std, 0.0, 1.0),
                    np.clip(ben_mean + ben_std, 0.0, 1.0),
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
                ax_bot.plot(
                    mal_epochs,
                    mal_mean,
                    color=color,
                    linestyle=ls,
                    linewidth=1.8,
                    label=setting,
                )
                ax_bot.fill_between(
                    mal_epochs,
                    np.clip(mal_mean - mal_std, 0.0, 1.0),
                    np.clip(mal_mean + mal_std, 0.0, 1.0),
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )

                if setting not in legend_handles:
                    legend_handles[setting] = h
                plotted_any = True

            thr = float(np.mean(threshold_vals)) if threshold_vals else 0.6
            ax_top.axhline(thr, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
            ax_bot.axhline(thr, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
            ax_top.set_ylim(0.0, 1.0)
            ax_bot.set_ylim(0.0, 1.0)

            if not settings:
                ax_top.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_top.transAxes)
                ax_bot.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_bot.transAxes)

        if not plotted_any:
            plt.close(fig)
            skipped += 1
            continue

        seeds_sorted = sorted(used_seeds, key=lambda x: int(x) if x.isdigit() else x)
        title = (
            f"TriGuardFL reputation curves | {info['iid']} | {info['dataset']}_{info['model']} | "
            f"{info['algo']} | cfg={info['cfg']} | epochs<{args.max_epoch} | "
            f"mean±std over seeds {','.join(seeds_sorted) if seeds_sorted else 'NA'}"
        )
        fig.suptitle(title, fontsize=12)

        if legend_handles:
            order = sorted(legend_handles.keys(), key=setting_sort_key)
            fig.legend(
                [legend_handles[k] for k in order],
                order,
                loc="lower center",
                ncol=min(6, max(1, len(order))),
                frameon=True,
                bbox_to_anchor=(0.5, 0.01),
                fontsize=9,
            )

        fig.tight_layout(rect=[0, 0.07, 1, 0.95])

        out_base = sanitize(
            f"triguard_repcurve_{info['iid']}_{info['dataset']}_{info['model']}_{info['algo']}"
            f"_clients{info['num_clients']}_lr{info['lr']}_cfg{info['cfg']}_epLT{args.max_epoch}"
        )
        png = args.out_dir / f"{out_base}.png"
        pdf = args.out_dir / f"{out_base}.pdf"
        fig.savefig(png, dpi=args.dpi)
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

        row = {"figure_png": str(png), "figure_pdf": str(pdf)}
        row.update(info)
        row["seeds_used"] = ",".join(seeds_sorted)
        plot_rows.append(row)
        generated += 1

    index_csv = args.out_dir / "plot_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["figure_png", "figure_pdf", *GROUP_FIELDS, "seeds_used"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(plot_rows)

    summary = [
        f"log_root={args.logs_root}",
        f"out_dir={args.out_dir}",
        f"total_txt={len(all_txt)}",
        f"parsed_triguard_attack4={len(parsed)}",
        f"filtered_not_triguard={filtered_not_triguard}",
        f"filtered_not_attack4={filtered_not_attack4}",
        f"filtered_seed={filtered_seed}",
        f"filtered_epoch_cfg={filtered_epoch_cfg}",
        f"unparsed={len(unparsed)}",
        f"groups={len(grouped)}",
        f"generated_figures={generated}",
        f"skipped_groups={skipped}",
        f"runs_short_or_empty={short_runs}",
        f"max_epoch={args.max_epoch}",
        f"seed_filter={','.join(sorted(seed_filter)) if seed_filter else 'ALL'}",
        f"exclude_config_epochs={','.join(sorted(excluded_config_epochs)) if excluded_config_epochs else 'NONE'}",
        f"index_csv={index_csv}",
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
