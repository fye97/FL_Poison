#!/usr/bin/env python3
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


FILENAME_RE = re.compile(
    r"^(?P<dataset>[^_]+)_(?P<model>[^_]+)_(?P<iid>iid|non-iid)_(?P<attack>[^_]+)_(?P<defense>[^_]+)"
    r"_(?P<epochs>\d+)_(?P<num_clients>\d+)_(?P<lr>[0-9.]+)_(?P<algo>[^_]+)_adv(?P<adv>[0-9.]+)"
    r"_seed(?P<seed>\d+)(?:_alpha(?P<alpha>[0-9.]+))?_cfg(?P<cfg>.+?)(?:_exp(?P<expid>\d+))?\.txt$"
)
EPOCH_ACC_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+.*?Test Acc:\s*(?P<acc>[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose runs that did not reach the target epoch and summarize their causes."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/logs_from_slurm_split_20260312"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/logs_from_slurm_split_20260312/extraction_manifest.tsv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/fengye/FL_Poison/output/spreadsheet/short_run_diagnosis_epochs200_adv0.1-0.3"),
    )
    parser.add_argument(
        "--target-epoch",
        type=int,
        default=199,
        help="A run is considered complete only if this exact epoch exists.",
    )
    parser.add_argument(
        "--keep-epochs",
        type=str,
        default="200",
        help="Comma-separated configured epochs values to keep by filename metadata.",
    )
    parser.add_argument(
        "--max-adv",
        type=float,
        default=0.3,
        help="Keep only runs with adv <= this value.",
    )
    return parser.parse_args()


def parse_epoch_curve(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    curve = {}
    for match in EPOCH_ACC_RE.finditer(text):
        curve[int(match.group("epoch"))] = float(match.group("acc"))
    return text, curve


def load_manifest(manifest_path: Path):
    mapping = {}
    with manifest_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                mapping[parts[2]] = parts[0]
    return mapping


def classify_interrupt(err_text: str):
    err_lower = err_text.lower()
    if "due to time limit" in err_lower or (
        "cancelled at" in err_lower and "time limit" in err_lower
    ):
        return "time_limit", "slurm time limit"
    if "bus error" in err_lower:
        return "bus_error", "bus error"
    if "out of memory" in err_lower or "exceeded step memory limit" in err_lower:
        return "oom", "out of memory"

    if "traceback" in err_lower:
        for line in reversed(err_text.splitlines()):
            stripped = line.strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*Error:", stripped) or stripped.startswith(
                "OSError:"
            ):
                return "python_error", stripped
        return "python_error", "python traceback"

    return "unknown_interrupt", ""


def build_recommendations(detail_df: pd.DataFrame):
    if detail_df.empty:
        return [
            "No short runs were found under the requested filters.",
        ]

    lines = []
    cause_counts = detail_df["cause"].value_counts().to_dict()
    lines.append("Recommendations")
    lines.append("")
    lines.append(
        f"- Main cause is `{max(cause_counts, key=cause_counts.get)}` "
        f"({cause_counts[max(cause_counts, key=cause_counts.get)]}/{len(detail_df)} short runs)."
    )
    lines.append(
        "- Keep `target_epoch=199` as the acceptance rule; every dropped run here truly ended before epoch 199."
    )

    c10_time = detail_df[
        (detail_df["dataset"] == "CIFAR10") & (detail_df["cause"] == "time_limit")
    ]
    if not c10_time.empty:
        lines.append(
            "- For CIFAR10 jobs, the 12-hour walltime is too aggressive when one Slurm task runs 5 seeds sequentially."
        )
        lines.append(
            "- Recommended: either raise walltime to at least 24 hours, or set `num_experiments=1` and fan out seeds through the array dimension."
        )

    c100_time = detail_df[
        (detail_df["dataset"] == "CIFAR100") & (detail_df["cause"] == "time_limit")
    ]
    if not c100_time.empty:
        hot = (
            c100_time.groupby(["num_clients", "attack"])
            .size()
            .sort_values(ascending=False)
            .head(3)
        )
        hot_text = ", ".join(
            [f"{clients} clients + {attack} ({count})" for (clients, attack), count in hot.items()]
        )
        lines.append(
            f"- For CIFAR100, the slowest incomplete groups are: {hot_text}."
        )
        lines.append(
            "- Recommended: split 40/60-client FangAttack jobs into one seed per task, or increase walltime beyond 24 hours for those subsets."
        )

    bus_count = cause_counts.get("bus_error", 0)
    if bus_count:
        lines.append(
            f"- `{bus_count}` runs ended with `Bus error`; rerunning them without addressing the runtime fault is unlikely to help."
        )

    py_df = detail_df[detail_df["cause"] == "python_error"]
    if not py_df.empty:
        examples = ", ".join(py_df["cause_detail"].dropna().unique()[:3])
        lines.append(
            f"- Python exceptions need code-side fixes before rerun. Seen examples: {examples}."
        )

    lines.append("")
    lines.append("Practical Slurm changes")
    lines.append("")
    lines.append("- Change `num_experiments=5` to `num_experiments=1` in the submit scripts.")
    lines.append("- Add seeds to the array grid instead of looping over them inside a single job.")
    lines.append("- Keep 12h only for clearly fast CIFAR10 subsets; use a longer walltime for FangAttack-heavy jobs.")
    lines.append("- For CIFAR100 with 60 clients, treat FangAttack as a separate long-job class.")
    return lines


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keep_epochs = {item.strip() for item in args.keep_epochs.split(",") if item.strip()}
    manifest_map = load_manifest(args.manifest)

    records = []
    total_considered = 0
    total_short = 0

    for path in sorted(args.logs_root.rglob("*.txt")):
        if path.name == "extraction_summary.txt":
            continue
        match = FILENAME_RE.match(path.name)
        if not match:
            continue

        meta = match.groupdict()
        meta["alpha"] = meta["alpha"] if meta["alpha"] is not None else "NA"
        meta["expid"] = meta["expid"] if meta["expid"] is not None else "NA"

        if keep_epochs and meta["epochs"] not in keep_epochs:
            continue
        if float(meta["adv"]) > args.max_adv + 1e-12:
            continue

        total_considered += 1
        segment_text, curve = parse_epoch_curve(path)

        max_epoch = max(curve.keys()) if curve else -1
        if curve and max_epoch >= args.target_epoch and args.target_epoch in curve:
            continue

        total_short += 1
        source_out = manifest_map.get(str(path), "")
        source_err = str(Path(source_out).with_suffix(".err")) if source_out else ""
        err_text = (
            Path(source_err).read_text(encoding="utf-8", errors="ignore")
            if source_err and Path(source_err).exists()
            else ""
        )
        cause, cause_detail = classify_interrupt(err_text)

        if not curve:
            drop_reason = "empty_curve"
        elif max_epoch < args.target_epoch:
            drop_reason = "max_epoch_lt_target"
        else:
            drop_reason = "missing_exact_target_epoch"

        records.append(
            {
                "dataset": meta["dataset"],
                "model": meta["model"],
                "iid": meta["iid"],
                "attack": meta["attack"],
                "defense": meta["defense"],
                "algo": meta["algo"],
                "epochs_cfg": int(meta["epochs"]),
                "num_clients": int(meta["num_clients"]),
                "lr": float(meta["lr"]),
                "adv": float(meta["adv"]),
                "alpha": meta["alpha"],
                "seed": int(meta["seed"]),
                "expid": meta["expid"],
                "max_epoch": max_epoch,
                "drop_reason": drop_reason,
                "cause": cause,
                "cause_detail": cause_detail,
                "segment_file": str(path),
                "source_out": source_out,
                "source_err": source_err,
                "segment_has_training_finished": "training finished on" in segment_text.lower(),
            }
        )

    detail_df = pd.DataFrame(records).sort_values(
        by=[
            "dataset",
            "model",
            "iid",
            "num_clients",
            "attack",
            "defense",
            "adv",
            "alpha",
            "seed",
        ]
    )
    detail_csv = args.out_dir / "short_runs_detail.csv"
    detail_df.to_csv(detail_csv, index=False)

    overview_df = pd.DataFrame(
        [
            ("total_considered", total_considered),
            ("short_runs", total_short),
            ("target_epoch", args.target_epoch),
            ("max_adv", args.max_adv),
            ("keep_epochs", ",".join(sorted(keep_epochs))),
        ],
        columns=["metric", "value"],
    )
    by_cause_df = (
        detail_df.groupby(["cause", "cause_detail"]).size().reset_index(name="count")
        if not detail_df.empty
        else pd.DataFrame(columns=["cause", "cause_detail", "count"])
    )
    by_dataset_df = (
        detail_df.groupby(["dataset", "model", "cause"]).size().reset_index(name="count")
        if not detail_df.empty
        else pd.DataFrame(columns=["dataset", "model", "cause", "count"])
    )
    by_combo_df = (
        detail_df.groupby(["dataset", "num_clients", "attack", "defense", "cause"])
        .size()
        .reset_index(name="count")
        .sort_values(
            by=["dataset", "num_clients", "count", "attack", "defense"],
            ascending=[True, True, False, True, True],
        )
        if not detail_df.empty
        else pd.DataFrame(columns=["dataset", "num_clients", "attack", "defense", "cause", "count"])
    )

    summary_rows = []
    if not detail_df.empty:
        by_group = (
            detail_df.groupby(["dataset", "num_clients", "attack"])
            .size()
            .reset_index(name="short_count")
        )
        for _, row in by_group.iterrows():
            summary_rows.append(
                {
                    "dataset": row["dataset"],
                    "num_clients": row["num_clients"],
                    "attack": row["attack"],
                    "short_count": row["short_count"],
                }
            )
    group_summary_df = pd.DataFrame(summary_rows)

    xlsx_path = args.out_dir / "short_runs_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="overview", index=False)
        by_cause_df.to_excel(writer, sheet_name="by_cause", index=False)
        by_dataset_df.to_excel(writer, sheet_name="by_dataset", index=False)
        by_combo_df.to_excel(writer, sheet_name="by_combo", index=False)
        group_summary_df.to_excel(writer, sheet_name="by_group", index=False)
        detail_df.to_excel(writer, sheet_name="detail", index=False)

        for sheet_name, df in {
            "overview": overview_df,
            "by_cause": by_cause_df,
            "by_dataset": by_dataset_df,
            "by_combo": by_combo_df,
            "by_group": group_summary_df,
            "detail": detail_df,
        }.items():
            ws = writer.sheets[sheet_name]
            if not df.empty:
                for idx, column in enumerate(df.columns, start=1):
                    width = max(len(str(column)), int(df[column].astype(str).map(len).quantile(0.95))) + 2
                    ws.column_dimensions[chr(64 + idx) if idx <= 26 else ws.cell(row=1, column=idx).column_letter].width = min(
                        width, 60
                    )

    md_path = args.out_dir / "compute_canada_recommendations.md"
    md_path.write_text("\n".join(build_recommendations(detail_df)) + "\n", encoding="utf-8")

    txt_path = args.out_dir / "summary.txt"
    summary_lines = [
        f"logs_root={args.logs_root}",
        f"manifest={args.manifest}",
        f"out_dir={args.out_dir}",
        f"target_epoch={args.target_epoch}",
        f"keep_epochs={','.join(sorted(keep_epochs))}",
        f"max_adv={args.max_adv}",
        f"total_considered={total_considered}",
        f"short_runs={total_short}",
        f"detail_csv={detail_csv}",
        f"report_xlsx={xlsx_path}",
        f"recommendations_md={md_path}",
    ]
    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
