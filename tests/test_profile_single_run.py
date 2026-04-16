import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PERF_DIR = ROOT / "tests" / "perf"
if str(PERF_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DIR))

from flpoison.utils.performance_utils import perf_summary_path
from profile_single_run import build_profile_config, resolve_perf_json


class _Args:
    model = None
    algorithm = None
    distribution = None
    epochs = None
    num_clients = None
    batch_size = None
    eval_batch_size = 1024
    evaluate = None
    local_epochs = None
    seed = None
    cudnn_benchmark = None
    allow_tf32 = None
    torch_profile = False
    gpu_sample_interval_ms = 100


def test_resolve_perf_json_finds_expected_file(tmp_path):
    output_path = tmp_path / "logs" / "perf_baseline" / "run" / "single_run_exp0.txt"
    perf_json = perf_summary_path(output_path)
    perf_json.parent.mkdir(parents=True, exist_ok=True)
    perf_json.write_text("{}", encoding="utf-8")

    assert resolve_perf_json(output_path) == perf_json


def test_resolve_perf_json_falls_back_to_single_candidate(tmp_path):
    output_path = tmp_path / "logs" / "perf_baseline" / "run" / "single_run.txt"
    fallback = perf_summary_path(output_path).parent / "single_run_exp0.json"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text("{}", encoding="utf-8")

    assert resolve_perf_json(output_path) == fallback


def test_build_profile_config_overrides_eval_batch_size(tmp_path):
    source_config = tmp_path / "config.yaml"
    source_config.write_text("batch_size: 64\neval_batch_size: 64\n", encoding="utf-8")

    profile_config, _ = build_profile_config(_Args(), source_config, tmp_path / "out")

    payload = profile_config.read_text(encoding="utf-8")
    assert "eval_batch_size: 1024" in payload


def test_build_profile_config_defaults_evaluate_to_false(tmp_path):
    source_config = tmp_path / "config.yaml"
    source_config.write_text("epochs: 20\n", encoding="utf-8")

    profile_config, _ = build_profile_config(_Args(), source_config, tmp_path / "out")

    payload = profile_config.read_text(encoding="utf-8")
    assert "evaluate: false" in payload


def test_build_profile_config_preserves_source_evaluate_setting(tmp_path):
    source_config = tmp_path / "config.yaml"
    source_config.write_text("epochs: 20\nevaluate: true\n", encoding="utf-8")

    profile_config, _ = build_profile_config(_Args(), source_config, tmp_path / "out")

    payload = profile_config.read_text(encoding="utf-8")
    assert "evaluate: true" in payload


def test_build_profile_config_applies_explicit_evaluate_override(tmp_path):
    source_config = tmp_path / "config.yaml"
    source_config.write_text("epochs: 20\nevaluate: false\n", encoding="utf-8")
    args = _Args()
    args.evaluate = True

    profile_config, _ = build_profile_config(args, source_config, tmp_path / "out")

    payload = profile_config.read_text(encoding="utf-8")
    assert "evaluate: true" in payload


def test_build_profile_config_writes_cuda_runtime_overrides(tmp_path):
    source_config = tmp_path / "config.yaml"
    source_config.write_text("cudnn_benchmark: true\nallow_tf32: false\n", encoding="utf-8")
    args = _Args()
    args.cudnn_benchmark = False
    args.allow_tf32 = True

    profile_config, _ = build_profile_config(args, source_config, tmp_path / "out")

    payload = profile_config.read_text(encoding="utf-8")
    assert "cudnn_benchmark: false" in payload
    assert "allow_tf32: true" in payload
