import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.cli.args import read_args


def test_read_args_benchmark_defaults_to_none(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["flpoison", "--config", "configs/FedSGD_MNIST_Lenet.yaml"],
    )

    _, cli_args = read_args()

    assert cli_args.benchmark is None


def test_read_args_benchmark_flag_enables_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["flpoison", "--config", "configs/FedSGD_MNIST_Lenet.yaml", "--benchmark"],
    )

    _, cli_args = read_args()

    assert cli_args.benchmark is True


def test_read_args_log_color_defaults_to_none(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["flpoison", "--config", "configs/FedSGD_MNIST_Lenet.yaml"],
    )

    _, cli_args = read_args()

    assert cli_args.log_color is None


def test_read_args_no_log_color_disables_color(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["flpoison", "--config", "configs/FedSGD_MNIST_Lenet.yaml", "--no-log_color"],
    )

    _, cli_args = read_args()

    assert cli_args.log_color is False
