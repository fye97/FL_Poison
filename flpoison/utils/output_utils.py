from __future__ import annotations

import csv
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
