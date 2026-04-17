#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.utils.output_utils import perf_summary_path, run_log_path


BACKDOOR_ATTACKS = {"BadNets", "BadNets_image", "ModelReplacement", "DBA", "Neurotoxin", "AlterMin", "EdgeCase"}
ROUND_TIME_RE = re.compile(r"Round time: (?P<sec>[\d.]+)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CARAT experiment outputs into per-run and grouped CSV reports."
    )
    parser.add_argument(
        "--result-root",
        default="logs/local_runs",
        help="Root directory containing experiment outputs and jobmeta files.",
    )
    parser.add_argument(
        "--output-root",
        default="logs/reports/carat",
        help="Directory where summary CSVs will be written.",
    )
    parser.add_argument(
        "--spec-names",
        default="",
        help="Optional comma-separated CARAT spec names to include. Defaults to all discovered specs.",
    )
    return parser.parse_args()


def safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value


def safe_int(text: str | None) -> int | None:
    value = safe_float(text)
    if value is None:
        return None
    return int(value)


def read_jobmeta(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key.strip()] = value.strip()
    return payload


def read_metrics_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def last_numeric(rows: list[dict[str, str]], key: str) -> float | None:
    for row in reversed(rows):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def mean_numeric(rows: list[dict[str, str]], key: str) -> float | None:
    values = [safe_float(row.get(key)) for row in rows]
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def read_perf_summary(metrics_path: Path) -> dict | None:
    perf_path = perf_summary_path(metrics_path)
    if not perf_path.exists():
        return None
    try:
        with perf_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def parse_round_time_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values = [float(match.group("sec")) for match in ROUND_TIME_RE.finditer(text)]
    if not values:
        return None
    return sum(values) / len(values)


def parse_output_path(metrics_path: Path) -> dict[str, str | None]:
    dirname = metrics_path.parent.name
    attack_defense = metrics_path.parent.parent.name
    distribution = metrics_path.parent.parent.parent.name
    dataset_model = metrics_path.parent.parent.parent.parent.name
    algorithm = metrics_path.parent.parent.parent.parent.parent.name

    attack, defense = attack_defense.split("__", 1) if "__" in attack_defense else (attack_defense, "")
    dataset, model = dataset_model.rsplit("_", 1) if "_" in dataset_model else (dataset_model, "")

    meta: dict[str, str | None] = {
        "algorithm": algorithm,
        "dataset": dataset,
        "model": model,
        "distribution": distribution,
        "attack": attack,
        "defense": defense,
        "config_stem": None,
        "dirichlet_alpha": None,
        "epochs": None,
        "num_clients": None,
        "learning_rate": None,
        "num_adv": None,
        "effective_seed": None,
        "experiment_id": None,
    }
    patterns = {
        "epochs": r"ep([^_]+)",
        "num_clients": r"clients([^_]+)",
        "learning_rate": r"lr([^_]+)",
        "num_adv": r"adv([^_]+)",
        "effective_seed": r"seed([^_]+)",
        "experiment_id": r"exp([^_]+)",
        "dirichlet_alpha": r"alpha([^_]+)",
        "config_stem": r"cfg(.+)$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, dirname)
        if match:
            meta[key] = match.group(1)
    return meta


def category_for_record(record: dict[str, object]) -> str:
    attack = str(record.get("attack") or "")
    if attack == "NoAttack":
        return "clean"
    if attack in BACKDOOR_ATTACKS:
        return "backdoor"
    return "untargeted"


def build_run_record(jobmeta_path: Path) -> dict[str, object] | None:
    jobmeta = read_jobmeta(jobmeta_path)
    spec_name = jobmeta.get("spec_name", "")
    if not spec_name:
        return None

    metrics_path = Path(jobmeta.get("output_file") or "")
    if not metrics_path.exists():
        return None

    parsed = parse_output_path(metrics_path)
    rows = read_metrics_rows(metrics_path)
    if not rows:
        return None

    perf_summary = read_perf_summary(metrics_path)
    perf_overall = (perf_summary or {}).get("overall", {})
    perf_final_test = perf_overall.get("final_test_metrics", {})

    mean_round_time_sec = mean_numeric(rows, "round_time_sec")
    if mean_round_time_sec is None:
        mean_round_time_sec = safe_float(perf_overall.get("sec_per_round"))
    if mean_round_time_sec is None:
        mean_round_time_sec = parse_round_time_from_log(run_log_path(metrics_path))

    record: dict[str, object] = {
        "spec_name": spec_name,
        "category": category_for_record(parsed),
        "metrics_file": str(metrics_path),
        "jobmeta_file": str(jobmeta_path),
        "run_log_file": str(run_log_path(metrics_path)),
        "complete": (metrics_path.parent / "task.complete").exists(),
        "final_epoch": safe_int(rows[-1].get("epoch")),
        "row_count": len(rows),
        "algorithm": parsed["algorithm"],
        "dataset": parsed["dataset"],
        "model": parsed["model"],
        "distribution": parsed["distribution"],
        "dirichlet_alpha": parsed["dirichlet_alpha"] or "",
        "attack": parsed["attack"],
        "defense": parsed["defense"],
        "epochs": safe_int(parsed["epochs"]),
        "num_clients": safe_int(parsed["num_clients"]),
        "learning_rate": safe_float(parsed["learning_rate"]),
        "num_adv": safe_float(parsed["num_adv"]),
        "effective_seed": safe_int(jobmeta.get("effective_seed")) or safe_int(parsed["effective_seed"]),
        "experiment_id": safe_int(jobmeta.get("experiment_id")) or safe_int(parsed["experiment_id"]),
        "config_stem": parsed["config_stem"] or Path(jobmeta.get("config_file", "")).stem,
        "train_acc": last_numeric(rows, "train_acc"),
        "train_loss": last_numeric(rows, "train_loss"),
        "eval_acc": last_numeric(rows, "eval_acc"),
        "eval_loss": last_numeric(rows, "eval_loss"),
        "tail_acc": last_numeric(rows, "tail_acc"),
        "macro_acc": last_numeric(rows, "macro_acc"),
        "worst_class_acc": last_numeric(rows, "worst_class_acc"),
        "asr": last_numeric(rows, "asr"),
        "asr_loss": last_numeric(rows, "asr_loss"),
        "mean_round_time_sec": mean_round_time_sec,
        "perf_sec_per_round": safe_float(perf_overall.get("sec_per_round")),
        "perf_total_time_sec": safe_float(perf_overall.get("total_time_sec")),
        "perf_gpu_utilization_pct_avg": safe_float(perf_overall.get("gpu_utilization_pct_avg")),
        "perf_gpu_memory_peak_allocated_mb": safe_float(perf_overall.get("gpu_memory_peak_allocated_mb")),
    }

    if record["eval_acc"] is None:
        record["eval_acc"] = safe_float(perf_final_test.get("Test Acc"))
    if record["eval_loss"] is None:
        record["eval_loss"] = safe_float(perf_final_test.get("Test loss"))
    if record["tail_acc"] is None:
        record["tail_acc"] = safe_float(perf_final_test.get("Tail Acc"))
    if record["macro_acc"] is None:
        record["macro_acc"] = safe_float(perf_final_test.get("Macro Acc"))
    if record["worst_class_acc"] is None:
        record["worst_class_acc"] = safe_float(perf_final_test.get("Worst-Class Acc"))
    if record["asr"] is None:
        record["asr"] = safe_float(perf_final_test.get("ASR"))
    if record["asr_loss"] is None:
        record["asr_loss"] = safe_float(perf_final_test.get("ASR loss"))

    return record


def mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None, None, 0
    mean_value = sum(filtered) / len(filtered)
    std_value = statistics.stdev(filtered) if len(filtered) > 1 else 0.0
    return mean_value, std_value, len(filtered)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_grouped_summary(records: list[dict[str, object]], group_fields: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    metric_fields = [
        "train_acc",
        "train_loss",
        "eval_acc",
        "eval_loss",
        "tail_acc",
        "macro_acc",
        "worst_class_acc",
        "asr",
        "asr_loss",
        "mean_round_time_sec",
        "perf_sec_per_round",
        "perf_gpu_utilization_pct_avg",
        "perf_gpu_memory_peak_allocated_mb",
    ]

    for record in records:
        key = tuple(record.get(field) for field in group_fields)
        groups[key].append(record)

    summary_rows: list[dict[str, object]] = []
    for key, items in sorted(groups.items(), key=lambda item: item[0]):
        row = {field: value for field, value in zip(group_fields, key)}
        row["num_runs"] = len(items)
        for metric in metric_fields:
            mean_value, std_value, count = mean_std(record.get(metric) for record in items)
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_n"] = count
        summary_rows.append(row)
    return summary_rows


def collect_jobmeta_files(result_root: Path) -> list[Path]:
    return sorted(result_root.rglob("jobmeta.txt"))


def main() -> int:
    args = parse_args()
    result_root = Path(args.result_root).resolve()
    output_root = Path(args.output_root).resolve()
    allowed_specs = {item.strip() for item in args.spec_names.split(",") if item.strip()}

    records = []
    for jobmeta_path in collect_jobmeta_files(result_root):
        record = build_run_record(jobmeta_path)
        if record is None:
            continue
        if allowed_specs and record["spec_name"] not in allowed_specs:
            continue
        records.append(record)

    if not records:
        print(f"No CARAT runs found under {result_root}", file=sys.stderr)
        return 1

    all_fields = [
        "spec_name",
        "category",
        "complete",
        "algorithm",
        "dataset",
        "model",
        "distribution",
        "dirichlet_alpha",
        "attack",
        "defense",
        "epochs",
        "num_clients",
        "learning_rate",
        "num_adv",
        "effective_seed",
        "experiment_id",
        "config_stem",
        "final_epoch",
        "row_count",
        "train_acc",
        "train_loss",
        "eval_acc",
        "eval_loss",
        "tail_acc",
        "macro_acc",
        "worst_class_acc",
        "asr",
        "asr_loss",
        "mean_round_time_sec",
        "perf_sec_per_round",
        "perf_total_time_sec",
        "perf_gpu_utilization_pct_avg",
        "perf_gpu_memory_peak_allocated_mb",
        "metrics_file",
        "jobmeta_file",
        "run_log_file",
    ]
    write_csv(output_root / "all_runs.csv", records, all_fields)

    clean_records = [record for record in records if record["category"] == "clean"]
    untargeted_records = [record for record in records if record["category"] == "untargeted"]
    backdoor_records = [record for record in records if record["category"] == "backdoor"]

    grouped_fieldnames = [
        "spec_name",
        "dataset",
        "distribution",
        "dirichlet_alpha",
        "attack",
        "defense",
        "num_adv",
        "num_runs",
    ]
    metric_suffixes = [
        "train_acc",
        "train_loss",
        "eval_acc",
        "eval_loss",
        "tail_acc",
        "macro_acc",
        "worst_class_acc",
        "asr",
        "asr_loss",
        "mean_round_time_sec",
        "perf_sec_per_round",
        "perf_gpu_utilization_pct_avg",
        "perf_gpu_memory_peak_allocated_mb",
    ]
    summary_fields = list(grouped_fieldnames)
    for metric in metric_suffixes:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])

    clean_summary = build_grouped_summary(
        clean_records,
        ["spec_name", "dataset", "distribution", "dirichlet_alpha", "attack", "defense", "num_adv"],
    )
    untargeted_summary = build_grouped_summary(
        untargeted_records,
        ["spec_name", "dataset", "distribution", "dirichlet_alpha", "attack", "defense", "num_adv"],
    )
    backdoor_summary = build_grouped_summary(
        backdoor_records,
        ["spec_name", "dataset", "distribution", "dirichlet_alpha", "attack", "defense", "num_adv"],
    )
    runtime_summary = build_grouped_summary(
        records,
        ["spec_name", "dataset", "distribution", "dirichlet_alpha", "attack", "defense", "num_adv"],
    )

    write_csv(output_root / "clean_summary.csv", clean_summary, summary_fields)
    write_csv(output_root / "untargeted_summary.csv", untargeted_summary, summary_fields)
    write_csv(output_root / "backdoor_summary.csv", backdoor_summary, summary_fields)
    write_csv(output_root / "runtime_summary.csv", runtime_summary, summary_fields)

    print(f"Wrote {len(records)} run records to {output_root / 'all_runs.csv'}")
    print(f"Wrote clean summary to {output_root / 'clean_summary.csv'}")
    print(f"Wrote untargeted summary to {output_root / 'untargeted_summary.csv'}")
    print(f"Wrote backdoor summary to {output_root / 'backdoor_summary.csv'}")
    print(f"Wrote runtime summary to {output_root / 'runtime_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
