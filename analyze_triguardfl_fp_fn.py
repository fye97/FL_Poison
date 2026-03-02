#!/usr/bin/env python3
import argparse
import ast
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
HEADER_NUM_ADV_RE = re.compile(r"\bnum_adv:\s*([0-9.]+)")
DETECTED_RE = re.compile(r"^Detected Attackers:\s*(\[[^\]]*\])\s*$")
EPOCH_RE = re.compile(r"^Epoch\s+(?P<epoch>\d+)\b")

ATTACK_COLORS = {
    "NoAttack": "#111111",
    "ALIE": "#1f77b4",
    "FangAttack": "#ff7f0e",
    "MinMax": "#2ca02c",
    "MinSum": "#d62728",
}
ATTACK_ORDER = ["NoAttack", "ALIE", "FangAttack", "MinMax", "MinSum"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute TriGuardFL false-positive/false-negative rates from split logs "
            "and generate summary plots."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/logs_from_slurm_split"),
        help="Root folder containing split logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/triguardfl_fp_fn_stats"),
        help="Output directory for CSV and figures.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=300,
        help="Only include epochs < max-epoch.",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated seed filter. Empty means all seeds.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG DPI.",
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
    return parser.parse_args()


def parse_detected_list(raw: str):
    try:
        arr = ast.literal_eval(raw)
        if isinstance(arr, (list, tuple)):
            return [int(x) for x in arr]
    except Exception:
        pass
    return [int(x) for x in re.findall(r"-?\d+", raw)]


def safe_mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, 0
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def aggregate_by_attack_epoch(records, metric_key):
    bucket = defaultdict(list)
    for r in records:
        bucket[(r["attack"], r["epoch"])].append(r[metric_key])
    rows = []
    for (attack, epoch), vals in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        mean, std, n = safe_mean_std(vals)
        rows.append(
            {
                "attack": attack,
                "epoch": epoch,
                f"{metric_key}_mean": mean,
                f"{metric_key}_std": std,
                f"{metric_key}_n": n,
            }
        )
    return rows


def main():
    args = parse_args()
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

    if not args.logs_root.exists():
        raise FileNotFoundError(f"logs-root not found: {args.logs_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parsed_files = 0
    triguard_files = 0
    used_files = 0
    filtered_seed = 0
    filtered_epoch_cfg = 0
    rows = []
    run_final_rows = []
    dropped_no_header_num_adv = 0

    for path in sorted(args.logs_root.rglob("*.txt")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        parsed_files += 1
        meta = m.groupdict()
        if meta["alpha"] is None:
            meta["alpha"] = "NA"
        if meta["expid"] is None:
            meta["expid"] = "NA"

        if meta["defense"] != "TriGuardFL":
            continue
        triguard_files += 1
        if seed_filter and meta["seed"] not in seed_filter:
            filtered_seed += 1
            continue
        if excluded_config_epochs and meta["epochs"] in excluded_config_epochs:
            filtered_epoch_cfg += 1
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        if not lines:
            continue

        num_clients = int(meta["num_clients"])
        attack_name = meta["attack"]

        num_adv_match = HEADER_NUM_ADV_RE.search(lines[0])
        if num_adv_match is None and attack_name != "NoAttack":
            # Without num_adv we cannot determine ground truth attackers robustly.
            dropped_no_header_num_adv += 1
            continue
        num_adv = int(round(float(num_adv_match.group(1)))) if num_adv_match else 0

        if attack_name == "NoAttack":
            true_attackers = set()
        else:
            true_attackers = set(range(max(0, min(num_clients, num_adv))))

        current_detected = None
        run_rows = []
        for line in lines:
            line = line.strip()
            md = DETECTED_RE.match(line)
            if md:
                current_detected = set(parse_detected_list(md.group(1)))
                continue

            me = EPOCH_RE.match(line)
            if me and current_detected is not None:
                epoch = int(me.group("epoch"))
                if epoch >= args.max_epoch:
                    current_detected = None
                    continue

                predicted = {i for i in current_detected if 0 <= i < num_clients}

                tp = len(predicted & true_attackers)
                fp = len(predicted - true_attackers)
                fn = len(true_attackers - predicted)
                tn = num_clients - tp - fp - fn

                benign_count = num_clients - len(true_attackers)
                fpr = float(fp / benign_count) if benign_count > 0 else np.nan
                fnr = float(fn / len(true_attackers)) if len(true_attackers) > 0 else np.nan

                rec = {
                    "file_path": str(path),
                    "dataset": meta["dataset"],
                    "model": meta["model"],
                    "iid": meta["iid"],
                    "attack": attack_name,
                    "defense": meta["defense"],
                    "epochs_cfg": int(meta["epochs"]),
                    "num_clients": num_clients,
                    "adv_cfg": float(meta["adv"]),
                    "num_adv": int(num_adv),
                    "alpha": meta["alpha"],
                    "algo": meta["algo"],
                    "cfg": meta["cfg"],
                    "seed": int(meta["seed"]),
                    "expid": meta["expid"],
                    "epoch": epoch,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "fpr": fpr,
                    "fnr": fnr,
                    "detected_count": len(predicted),
                    "true_attacker_count": len(true_attackers),
                }
                rows.append(rec)
                run_rows.append(rec)
                current_detected = None

        if run_rows:
            used_files += 1
            final = max(run_rows, key=lambda r: r["epoch"])
            run_final_rows.append(final)

    if not rows:
        raise RuntimeError("No TriGuardFL detection records found.")

    # Save per-epoch detailed records.
    detail_csv = args.out_dir / "triguardfl_fp_fn_per_epoch_records.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Aggregation over attack+epoch.
    fpr_rows = aggregate_by_attack_epoch(rows, "fpr")
    fnr_rows = aggregate_by_attack_epoch(rows, "fnr")
    merged = {}
    for r in fpr_rows:
        merged[(r["attack"], r["epoch"])] = dict(r)
    for r in fnr_rows:
        merged.setdefault((r["attack"], r["epoch"]), {"attack": r["attack"], "epoch": r["epoch"]})
        merged[(r["attack"], r["epoch"])].update(r)
    agg_rows = [merged[k] for k in sorted(merged.keys(), key=lambda x: (x[0], x[1]))]

    agg_csv = args.out_dir / "triguardfl_fp_fn_attack_epoch_stats.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "attack",
            "epoch",
            "fpr_mean",
            "fpr_std",
            "fpr_n",
            "fnr_mean",
            "fnr_std",
            "fnr_n",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(agg_rows)

    # Final-epoch statistics per run.
    final_csv = args.out_dir / "triguardfl_fp_fn_run_final.csv"
    with final_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(run_final_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(run_final_rows)

    # Final summary by attack.
    bucket_final = defaultdict(lambda: {"fpr": [], "fnr": []})
    for r in run_final_rows:
        bucket_final[r["attack"]]["fpr"].append(r["fpr"])
        bucket_final[r["attack"]]["fnr"].append(r["fnr"])

    final_attack_rows = []
    for attack in sorted(bucket_final.keys()):
        fpr_mean, fpr_std, fpr_n = safe_mean_std(bucket_final[attack]["fpr"])
        fnr_mean, fnr_std, fnr_n = safe_mean_std(bucket_final[attack]["fnr"])
        final_attack_rows.append(
            {
                "attack": attack,
                "fpr_mean": fpr_mean,
                "fpr_std": fpr_std,
                "fpr_n": fpr_n,
                "fnr_mean": fnr_mean,
                "fnr_std": fnr_std,
                "fnr_n": fnr_n,
            }
        )

    final_attack_csv = args.out_dir / "triguardfl_fp_fn_final_by_attack.csv"
    with final_attack_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(final_attack_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final_attack_rows)

    # Figure 1: epoch curves by attack.
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    metric_specs = [
        ("fpr", "False Positive Rate (FPR)"),
        ("fnr", "False Negative Rate (FNR)"),
    ]
    for ax, (metric, title) in zip(axes, metric_specs):
        for attack in ATTACK_ORDER:
            points = [
                r
                for r in agg_rows
                if r["attack"] == attack and np.isfinite(r.get(f"{metric}_mean", np.nan))
            ]
            if not points:
                continue
            points = sorted(points, key=lambda r: r["epoch"])
            x = np.array([r["epoch"] for r in points], dtype=float)
            y = np.array([r[f"{metric}_mean"] for r in points], dtype=float)
            s = np.array([r.get(f"{metric}_std", np.nan) for r in points], dtype=float)
            color = ATTACK_COLORS.get(attack, None)
            linestyle = "--" if attack == "NoAttack" else "-"
            ax.plot(x, y, label=attack, color=color, linewidth=1.8, linestyle=linestyle)
            lower = np.clip(y - s, 0.0, 1.0)
            upper = np.clip(y + s, 0.0, 1.0)
            ax.fill_between(x, lower, upper, color=color, alpha=0.15, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(
        "TriGuardFL Detection Error Rates (mean±std over available runs)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path_png = args.out_dir / "triguardfl_fp_fn_vs_epoch_by_attack.png"
    fig_path_pdf = args.out_dir / "triguardfl_fp_fn_vs_epoch_by_attack.pdf"
    fig.savefig(fig_path_png, dpi=args.dpi)
    fig.savefig(fig_path_pdf, format="pdf")
    plt.close(fig)

    # Figure 2: final-epoch by attack.
    attacks_present = [a for a in ATTACK_ORDER if any(r["attack"] == a for r in final_attack_rows)]
    fpr_means = []
    fpr_stds = []
    fnr_means = []
    fnr_stds = []
    colors = []
    for a in attacks_present:
        row = next(r for r in final_attack_rows if r["attack"] == a)
        fpr_means.append(row["fpr_mean"])
        fpr_stds.append(row["fpr_std"])
        fnr_means.append(row["fnr_mean"])
        fnr_stds.append(row["fnr_std"])
        colors.append(ATTACK_COLORS.get(a, "#777777"))

    x = np.arange(len(attacks_present))
    width = 0.38
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(
        x - width / 2,
        fpr_means,
        width=width,
        yerr=fpr_stds,
        label="FPR",
        color="#1f77b4",
        alpha=0.85,
        capsize=3,
    )
    ax2.bar(
        x + width / 2,
        fnr_means,
        width=width,
        yerr=fnr_stds,
        label="FNR",
        color="#d62728",
        alpha=0.85,
        capsize=3,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(attacks_present, rotation=0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Rate")
    ax2.set_title("TriGuardFL Final-Epoch Detection Error Rates by Attack (mean±std)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2_path_png = args.out_dir / "triguardfl_fp_fn_final_by_attack.png"
    fig2_path_pdf = args.out_dir / "triguardfl_fp_fn_final_by_attack.pdf"
    fig2.savefig(fig2_path_png, dpi=args.dpi)
    fig2.savefig(fig2_path_pdf, format="pdf")
    plt.close(fig2)

    summary_lines = [
        f"logs_root={args.logs_root}",
        f"out_dir={args.out_dir}",
        f"parsed_files={parsed_files}",
        f"triguard_files={triguard_files}",
        f"filtered_seed={filtered_seed}",
        f"filtered_epoch_cfg={filtered_epoch_cfg}",
        f"used_files={used_files}",
        f"per_epoch_records={len(rows)}",
        f"run_final_records={len(run_final_rows)}",
        f"dropped_no_header_num_adv={dropped_no_header_num_adv}",
        f"max_epoch={args.max_epoch}",
        f"seed_filter={','.join(sorted(seed_filter)) if seed_filter else 'ALL'}",
        f"exclude_config_epochs={','.join(sorted(excluded_config_epochs)) if excluded_config_epochs else 'NONE'}",
        f"detail_csv={detail_csv}",
        f"epoch_stats_csv={agg_csv}",
        f"final_csv={final_csv}",
        f"final_by_attack_csv={final_attack_csv}",
        f"figure_epoch_png={fig_path_png}",
        f"figure_epoch_pdf={fig_path_pdf}",
        f"figure_final_png={fig2_path_png}",
        f"figure_final_pdf={fig2_path_pdf}",
    ]
    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
