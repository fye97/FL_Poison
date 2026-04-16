from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


METRICS_FILENAME = "metrics.csv"


def derived_artifact_path(output_file: str | Path, subdir: str, suffix: str) -> Path:
    source = Path(output_file)
    parts = list(source.parts)
    if "logs" in parts:
        idx = parts.index("logs")
        derived = Path(*parts[: idx + 1], subdir, *parts[idx + 1 :])
    else:
        derived = source.parent / subdir / source.name
    path = derived.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_log_path(output_file: str | Path) -> Path:
    return derived_artifact_path(output_file, "run_logs", ".log")


def time_log_path(output_file: str | Path) -> Path:
    return derived_artifact_path(output_file, "time_logs", ".log")


def perf_summary_path(output_file: str | Path) -> Path:
    return derived_artifact_path(output_file, "perf_logs", ".json")


def torch_trace_dir(output_file: str | Path) -> Path:
    trace_path = derived_artifact_path(output_file, "torch_traces", "")
    trace_path.mkdir(parents=True, exist_ok=True)
    return trace_path


def _serialize_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def _parse_metrics_csv(filename: str | Path):
    with Path(filename).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "epoch" not in fieldnames or "train_acc" not in fieldnames:
            return None

        rows = list(reader)

    has_eval_metrics = any((row.get("eval_acc") or "").strip() for row in rows)
    acc_key = "eval_acc" if has_eval_metrics else "train_acc"
    loss_key = "eval_loss" if has_eval_metrics else "train_loss"
    epochs, accs, losses = [], [], []
    asrs, asr_losses = [], []

    for row in rows:
        epoch_text = (row.get("epoch") or "").strip()
        acc_text = (row.get(acc_key) or "").strip()
        loss_text = (row.get(loss_key) or "").strip()
        if not epoch_text or not acc_text or not loss_text:
            continue
        epochs.append(int(epoch_text))
        accs.append(float(acc_text))
        losses.append(float(loss_text))
        asrs.append(None)
        asr_losses.append(None)

    return epochs, accs, losses, asrs, asr_losses


def parse_logs(filename: str | Path):
    csv_metrics = _parse_metrics_csv(filename)
    if csv_metrics is not None:
        return csv_metrics

    content = Path(filename).read_text(encoding="utf-8")
    epochs, accs, losses, asrs, asr_losses = [], [], [], [], []
    regex = (
        r"Epoch (?P<epoch>\d+)\s.*?Test Acc: (?P<test_acc>[\d\.]+)\s.*?Test loss: (?P<test_loss>[\d\.]+)"
        r"(?:\s.*?ASR: (?P<asr>[\d\.]+))?(?:\s.*?ASR loss: (?P<asr_loss>[\d\.]+))?"
    )

    for match in re.finditer(regex, content):
        epochs.append(int(match.group("epoch")))
        accs.append(float(match.group("test_acc")))
        losses.append(float(match.group("test_loss")))

        asr = match.group("asr")
        asr_loss = match.group("asr_loss")
        asrs.append(float(asr) if asr else None)
        asr_losses.append(float(asr_loss) if asr_loss else None)

    return epochs, accs, losses, asrs, asr_losses


class EpochMetricsWriter:
    def __init__(self, output_file: str | Path, *, include_eval: bool):
        self.path = Path(output_file)
        self.include_eval = bool(include_eval)
        self.fieldnames = ["epoch", "train_acc", "train_loss"]
        if self.include_eval:
            self.fieldnames.extend(["eval_acc", "eval_loss"])
        self._handle = None
        self._writer = None

    def __enter__(self) -> "EpochMetricsWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None
        return False

    def write_row(
        self,
        *,
        epoch: int,
        train_acc: float,
        train_loss: float,
        eval_acc: float | None = None,
        eval_loss: float | None = None,
    ) -> None:
        if self._writer is None or self._handle is None:
            raise RuntimeError("EpochMetricsWriter must be used as a context manager")

        row = {
            "epoch": int(epoch),
            "train_acc": _serialize_metric(train_acc),
            "train_loss": _serialize_metric(train_loss),
        }
        if self.include_eval:
            row["eval_acc"] = _serialize_metric(eval_acc)
            row["eval_loss"] = _serialize_metric(eval_loss)

        self._writer.writerow(row)
        self._handle.flush()
