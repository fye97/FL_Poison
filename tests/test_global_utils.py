import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.utils.global_utils import (
    get_context_logger,
    print_filtered_args,
    setup_console_logger,
    setup_logger,
)


class _TTYStringIO(io.StringIO):
    def __init__(self, is_tty):
        super().__init__()
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


def test_setup_logger_colorizes_tty_stream_but_not_file(monkeypatch, tmp_path):
    monkeypatch.delenv("NO_COLOR", raising=False)
    output_path = tmp_path / "logs" / "train.txt"
    stream = _TTYStringIO(True)
    logger = setup_logger(
        "test_setup_logger_colorizes_tty_stream_but_not_file",
        str(output_path),
        level=logging.INFO,
        stream=True,
        color="auto",
        stream_target=stream,
    )

    logger.warning("warning message")

    assert "\x1b[" in stream.getvalue()
    assert "\x1b[" not in output_path.read_text(encoding="utf-8")


def test_setup_logger_respects_explicit_color_disable(tmp_path):
    output_path = tmp_path / "logs" / "train.txt"
    stream = _TTYStringIO(True)
    logger = setup_logger(
        "test_setup_logger_respects_explicit_color_disable",
        str(output_path),
        level=logging.INFO,
        stream=True,
        color=False,
        stream_target=stream,
    )

    logger.error("error message")

    assert "\x1b[" not in stream.getvalue()
    assert "\x1b[" not in output_path.read_text(encoding="utf-8")


def test_setup_console_logger_splits_stdout_and_stderr(monkeypatch):
    stdout = _TTYStringIO(False)
    stderr = _TTYStringIO(False)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    logger = setup_console_logger(
        "test_setup_console_logger_splits_stdout_and_stderr",
        level=logging.INFO,
        color=False,
    )

    logger.info("info message")
    logger.warning("warning message")

    assert "info message" in stdout.getvalue()
    assert "warning message" not in stdout.getvalue()
    assert "warning message" in stderr.getvalue()


def test_get_context_logger_prefers_args_logger(tmp_path):
    output_path = tmp_path / "logs" / "train.txt"
    base_logger = setup_logger(
        "test_get_context_logger_prefers_args_logger",
        str(output_path),
        level=logging.INFO,
    )
    args = SimpleNamespace(logger=base_logger)

    resolved = get_context_logger(args, logger_name="unused")

    assert resolved is base_logger


def test_print_filtered_args_formats_sectioned_summary(tmp_path):
    output_path = tmp_path / "logs" / "train.txt"
    logger = setup_logger(
        "test_print_filtered_args_formats_sectioned_summary",
        str(output_path),
        level=logging.INFO,
    )
    args = SimpleNamespace(
        seed=42,
        num_experiments=1,
        experiment_id=0,
        epochs=300,
        algorithm="FedAvg",
        optimizer="SGD",
        momentum=0.9,
        weight_decay=5.0e-4,
        learning_rate=0.05,
        dataset="CIFAR10",
        model="resnet18",
        distribution="iid",
        batch_size=64,
        eval_batch_size=64,
        num_clients=20,
        num_adv=4,
        attack="NoAttack",
        attack_params=None,
        defense="Mean",
        defense_params=None,
        device="cuda:0",
        gpu_idx=[0],
        log_stream=True,
        log_color="auto",
        output="logs/example.txt",
        logger=logger,
    )

    print_filtered_args(args, logger)
    payload = output_path.read_text(encoding="utf-8")

    assert "Configuration Summary" in payload
    assert "[Run]" in payload
    assert "[Data]" in payload
    assert "[Attack / Defense]" in payload
    assert "[Runtime]" in payload
    assert "[Output]" in payload
    assert "\n  seed" in payload
    assert "seed: 42, num_experiments" not in payload
