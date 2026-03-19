import sys
from pathlib import Path

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
    eval_interval = None
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


def test_build_profile_config_defaults_eval_interval_to_epochs(tmp_path):
    source_config = tmp_path / "config.yaml"
    source_config.write_text("epochs: 20\neval_interval: 1\n", encoding="utf-8")

    profile_config, _ = build_profile_config(_Args(), source_config, tmp_path / "out")

    payload = profile_config.read_text(encoding="utf-8")
    assert "eval_interval: 20" in payload


def test_build_profile_config_keeps_explicit_eval_interval_override(tmp_path):
    source_config = tmp_path / "config.yaml"
    source_config.write_text("epochs: 20\neval_interval: 1\n", encoding="utf-8")
    args = _Args()
    args.eval_interval = 5

    profile_config, _ = build_profile_config(args, source_config, tmp_path / "out")

    payload = profile_config.read_text(encoding="utf-8")
    assert "eval_interval: 5" in payload


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
