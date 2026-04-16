import importlib.util
import sys
from pathlib import Path


def _load_launch_module():
    module_path = Path(__file__).resolve().parent.parent / "exps" / "launch.py"
    spec = importlib.util.spec_from_file_location("exps_launch", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launch = _load_launch_module()
ExperimentTask = launch.ExperimentTask
RuntimeSpec = launch.RuntimeSpec
load_spec = launch.load_spec
task_command = launch.task_command


def test_load_spec_defaults_to_evaluate_enabled(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "name: smoke",
                "scenarios:",
                "  - algorithm: FedAvg",
                "    dataset: MNIST",
                "",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_spec(str(spec_path))

    assert spec.evaluate is True


def test_load_spec_honors_explicit_evaluate_false(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "name: smoke",
                "evaluate: false",
                "scenarios:",
                "  - algorithm: FedAvg",
                "    dataset: MNIST",
                "",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_spec(str(spec_path))

    assert spec.evaluate is False


def _task() -> ExperimentTask:
    return ExperimentTask(
        task_id=0,
        scenario_index=0,
        config_file=Path("configs/FedAvg_MNIST_Lenet.yaml"),
        config_name="FedAvg_MNIST_Lenet.yaml",
        algorithm="FedAvg",
        dataset="MNIST",
        distribution="iid",
        dirichlet_alpha="",
        im_iid_gamma="",
        model="lenet",
        epochs="1",
        num_clients="10",
        learning_rate="0.01",
        num_adv="0",
        seed_base=42,
        effective_seed=42,
        attack="NoAttack",
        defense="Mean",
        experiment_id=0,
    )


def test_task_command_emits_evaluate_override():
    enabled = task_command(
        python_bin="python",
        run_repo=Path("/tmp/repo"),
        config_path=Path("configs/FedAvg_MNIST_Lenet.yaml"),
        task=_task(),
        runtime=RuntimeSpec(),
        output_file=Path("logs/out.csv"),
        evaluate=True,
    )
    disabled = task_command(
        python_bin="python",
        run_repo=Path("/tmp/repo"),
        config_path=Path("configs/FedAvg_MNIST_Lenet.yaml"),
        task=_task(),
        runtime=RuntimeSpec(),
        output_file=Path("logs/out.csv"),
        evaluate=False,
    )

    assert "--evaluate" in enabled
    assert "--no-evaluate" not in enabled
    assert "--no-evaluate" in disabled
    assert "--evaluate" not in disabled
