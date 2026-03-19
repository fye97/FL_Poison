import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.utils.global_utils import setup_logger
from flpoison.utils.performance_utils import RuntimeProfiler, perf_summary_path


def build_args():
    return SimpleNamespace(
        device=torch.device("cpu"),
        seed=7,
        dataset="MNIST",
        model="lenet",
        algorithm="FedSGD",
        attack="NoAttack",
        defense="Mean",
        batch_size=32,
        eval_batch_size=32,
        num_clients=2,
        num_adv=0,
        local_epochs=1,
        epochs=2,
        distribution="iid",
        dirichlet_alpha=0.5,
        cache_partition=True,
        num_workers=0,
        eval_interval=1,
        gpu_sample_interval_ms=50,
        cudnn_benchmark=True,
        allow_tf32=False,
        torch_profile=False,
    )


def test_runtime_profiler_writes_summary_json(tmp_path):
    output_path = tmp_path / "logs" / "train.txt"
    logger = setup_logger("test_runtime_profiler_writes_summary_json", str(output_path))
    profiler = RuntimeProfiler(build_args(), logger, output_path)

    profiler.start_round(0)
    profiler.add_client_stage(0, "sync", 0.10)
    profiler.add_client_stage(0, "data", 0.20)
    profiler.add_client_stage(0, "fwd_bwd", 0.30)
    profiler.add_client_stage(0, "gpu_compute", 0.25)
    profiler.add_client_stage(0, "opt_step", 0.05)
    profiler.add_client_stage(0, "pack_update", 0.02)
    profiler.begin_aggregation_breakdown()
    profiler.add_aggregation_substage("defense", 0.05)
    profiler.add_aggregation_substage("aggregate", 0.02)
    profiler.finish_aggregation(0.08)
    profiler.add_server_stage("logging", 0.01)
    profiler.finish_round(
        total_sec=0.50,
        train_acc=0.8,
        train_loss=1.2,
        train_samples=64,
        test_stats={"Test Acc": 0.75},
    )
    payload = profiler.finalize()

    assert perf_summary_path(output_path).exists()
    assert payload["overall"]["sec_per_round"] == pytest.approx(0.50)
    assert payload["overall"]["final_train_accuracy"] == pytest.approx(0.8)
    assert payload["overall"]["final_train_samples"] == 64
    assert payload["overall"]["final_test_metrics"]["Test Acc"] == pytest.approx(0.75)
    assert payload["metadata"]["cudnn_benchmark"] is True
    assert payload["metadata"]["allow_tf32"] is False
    assert payload["rounds"][0]["stage_times"]["defense"] == pytest.approx(0.05)
    assert payload["rounds"][0]["stage_times"]["aggregate"] == pytest.approx(0.03)
    assert payload["rounds"][0]["round_time_sec"] == pytest.approx(0.50)
    assert payload["rounds"][0]["train_metrics"]["train_samples"] == 64


def test_runtime_profiler_aggregation_falls_back_to_total_time(tmp_path):
    output_path = tmp_path / "logs" / "train.txt"
    logger = setup_logger("test_runtime_profiler_aggregation_falls_back_to_total_time", str(output_path))
    profiler = RuntimeProfiler(build_args(), logger, output_path)

    profiler.start_round(0)
    profiler.finish_aggregation(0.12)
    profiler.finish_round(total_sec=0.20, train_acc=0.5, train_loss=2.0, train_samples=32)
    payload = profiler.finalize()

    assert payload["rounds"][0]["aggregation_split_available"] is False
    assert payload["rounds"][0]["stage_times"]["aggregate"] == pytest.approx(0.12)
    assert payload["rounds"][0]["stage_times"]["defense"] == pytest.approx(0.0)
