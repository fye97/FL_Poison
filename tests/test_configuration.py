import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.fl.configuration import normalize_evaluation_config, override_args


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
