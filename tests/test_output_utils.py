import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.utils.output_utils import (
    EpochMetricsWriter,
    parse_logs,
    perf_summary_path,
    run_log_path,
    time_log_path,
    torch_trace_dir,
)


def test_epoch_metrics_writer_writes_train_only_rows(tmp_path):
    output_path = tmp_path / "logs" / "train" / "metrics.csv"

    with EpochMetricsWriter(output_path, include_eval=False) as writer:
        writer.write_row(epoch=0, train_acc=0.5, train_loss=1.25, round_time_sec=3.5)
        writer.write_row(epoch=1, train_acc=0.75, train_loss=0.5, round_time_sec=4.25)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"epoch": "0", "train_acc": "0.5", "train_loss": "1.25", "round_time_sec": "3.5"},
        {"epoch": "1", "train_acc": "0.75", "train_loss": "0.5", "round_time_sec": "4.25"},
    ]


def test_epoch_metrics_writer_writes_eval_columns_when_enabled(tmp_path):
    output_path = tmp_path / "logs" / "train" / "metrics.csv"

    with EpochMetricsWriter(output_path, include_eval=True) as writer:
        writer.write_row(
            epoch=2,
            train_acc=0.8,
            train_loss=0.4,
            round_time_sec=5.0,
            eval_acc=0.7,
            eval_loss=0.6,
            tail_acc=0.5,
            macro_acc=0.61,
            worst_class_acc=0.22,
            asr=0.9,
            asr_loss=0.3,
        )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "epoch": "2",
            "train_acc": "0.8",
            "train_loss": "0.4",
            "round_time_sec": "5",
            "eval_acc": "0.7",
            "eval_loss": "0.6",
            "tail_acc": "0.5",
            "macro_acc": "0.61",
            "worst_class_acc": "0.22",
            "asr": "0.9",
            "asr_loss": "0.3",
        }
    ]


def test_derived_artifact_paths_follow_logs_subtrees(tmp_path):
    output_path = tmp_path / "logs" / "local_runs" / "job42" / "metrics.csv"

    assert run_log_path(output_path) == tmp_path / "logs" / "run_logs" / "local_runs" / "job42" / "metrics.log"
    assert time_log_path(output_path) == tmp_path / "logs" / "time_logs" / "local_runs" / "job42" / "metrics.log"
    assert perf_summary_path(output_path) == tmp_path / "logs" / "perf_logs" / "local_runs" / "job42" / "metrics.json"

    trace_dir = torch_trace_dir(output_path)
    assert trace_dir == tmp_path / "logs" / "torch_traces" / "local_runs" / "job42" / "metrics"
    assert trace_dir.is_dir()


def test_parse_logs_reads_metrics_csv_prefer_eval_columns(tmp_path):
    output_path = tmp_path / "logs" / "train" / "metrics.csv"

    with EpochMetricsWriter(output_path, include_eval=True) as writer:
        writer.write_row(
            epoch=0,
            train_acc=0.5,
            train_loss=1.0,
            round_time_sec=3.0,
            eval_acc=0.25,
            eval_loss=1.5,
        )
        writer.write_row(
            epoch=1,
            train_acc=0.8,
            train_loss=0.4,
            round_time_sec=3.5,
            eval_acc=0.6,
            eval_loss=0.7,
            asr=0.2,
            asr_loss=1.1,
        )

    epochs, accs, losses, asrs, asr_losses = parse_logs(output_path)

    assert epochs == [0, 1]
    assert accs == [0.25, 0.6]
    assert losses == [1.5, 0.7]
    assert asrs == [None, 0.2]
    assert asr_losses == [None, 1.1]
