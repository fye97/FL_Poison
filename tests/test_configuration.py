import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.fl.configuration import normalize_evaluation_config, override_args, single_preprocess


def test_normalize_evaluation_config_defaults_to_disabled():
    args = SimpleNamespace()

    normalize_evaluation_config(args)

    assert args.evaluate is False


def test_normalize_evaluation_config_enables_every_epoch_when_switch_is_true():
    args = SimpleNamespace(evaluate=True)

    normalize_evaluation_config(args)

    assert args.evaluate is True


def test_normalize_evaluation_config_disables_when_switch_is_false():
    args = SimpleNamespace(evaluate=False)

    normalize_evaluation_config(args)

    assert args.evaluate is False


def test_override_args_applies_evaluate_cli_switch():
    args = SimpleNamespace(
        attack="NoAttack",
        defense="Mean",
        evaluate=False,
    )
    cli_args = SimpleNamespace(
        config="configs/FedSGD_MNIST_Lenet.yaml",
        evaluate=True,
        attack=None,
        defense=None,
        defense_params=None,
        attack_params=None,
    )

    override_args(args, cli_args)

    assert args.evaluate is True


def test_single_preprocess_defaults_output_to_metrics_csv(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    args = SimpleNamespace(
        dataset="MNIST",
        gpu_idx=[0],
        log_stream=None,
        log_color=None,
        num_adv=0,
        num_clients=10,
        batch_size=64,
        evaluate=None,
        record_time=None,
        cudnn_benchmark=None,
        allow_tf32=None,
        torch_profile=None,
        gpu_sample_interval_ms=None,
        torch_profile_wait=None,
        torch_profile_warmup=None,
        torch_profile_active=None,
        torch_profile_repeat=None,
        torch_profile_record_shapes=None,
        torch_profile_memory=None,
        torch_profile_with_stack=None,
        attack="NoAttack",
        defense="Mean",
        algorithm="FedSGD",
        model="lenet",
        distribution="iid",
        epochs=5,
        learning_rate=0.01,
    )

    single_preprocess(args)

    assert args.output.endswith("_FedSGD_metrics.csv")
    assert args.log_stream is True
