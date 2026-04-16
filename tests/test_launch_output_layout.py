import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_launch_module():
    launch_path = ROOT / "exps" / "launch.py"
    spec = importlib.util.spec_from_file_location("flpoison_exps_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_task_output_dir_groups_attack_defense_and_run_metadata():
    launch = _load_launch_module()
    task = launch.ExperimentTask(
        task_id=0,
        scenario_index=0,
        config_file=ROOT / "configs" / "FedSGD_MNIST_Lenet.yaml",
        config_name="FedSGD_MNIST_Lenet.yaml",
        algorithm="FedSGD",
        dataset="MNIST",
        distribution="iid",
        dirichlet_alpha="",
        im_iid_gamma="",
        model="lenet",
        epochs="300",
        num_clients="50",
        learning_rate="0.01",
        num_adv="0",
        seed_base=7,
        effective_seed=7,
        attack="NoAttack",
        defense="Mean",
        experiment_id=0,
    )

    result_root = Path("/tmp/flpoison-results")
    out_dir = launch.task_output_dir(result_root, task)

    assert out_dir == (
        result_root
        / "FedSGD"
        / "MNIST_lenet"
        / "iid"
        / "NoAttack__Mean"
        / "ep300_clients50_lr0.01_adv0_seed7_exp0_cfgFedSGD_MNIST_Lenet"
    )
    assert launch.output_filename_for_task(task) == "metrics.csv"


def _build_task(launch, *, epochs="3"):
    return launch.ExperimentTask(
        task_id=0,
        scenario_index=0,
        config_file=ROOT / "configs" / "FedSGD_MNIST_Lenet.yaml",
        config_name="FedSGD_MNIST_Lenet.yaml",
        algorithm="FedSGD",
        dataset="MNIST",
        distribution="iid",
        dirichlet_alpha="",
        im_iid_gamma="",
        model="lenet",
        epochs=epochs,
        num_clients="50",
        learning_rate="0.01",
        num_adv="0",
        seed_base=7,
        effective_seed=7,
        attack="NoAttack",
        defense="Mean",
        experiment_id=0,
    )


def test_is_task_complete_accepts_full_metrics_without_marker(tmp_path):
    launch = _load_launch_module()
    task = _build_task(launch, epochs="3")
    result_root = tmp_path / "results"
    output_dir = launch.task_output_dir(result_root, task)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jobmeta.txt").write_text("task_id=0\n", encoding="utf-8")
    (output_dir / "metrics.csv").write_text(
        "epoch,train_acc,train_loss\n0,0.1,1.0\n1,0.2,0.9\n2,0.3,0.8\n",
        encoding="utf-8",
    )

    assert launch.is_task_complete(result_root, task) is True


def test_is_task_complete_rejects_partial_metrics(tmp_path):
    launch = _load_launch_module()
    task = _build_task(launch, epochs="3")
    result_root = tmp_path / "results"
    output_dir = launch.task_output_dir(result_root, task)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jobmeta.txt").write_text("task_id=0\n", encoding="utf-8")
    (output_dir / "metrics.csv").write_text(
        "epoch,train_acc,train_loss\n0,0.1,1.0\n1,0.2,0.9\n",
        encoding="utf-8",
    )

    assert launch.is_task_complete(result_root, task) is False


def test_filter_incomplete_task_ids_keeps_only_unfinished_tasks(tmp_path):
    launch = _load_launch_module()
    task0 = _build_task(launch, epochs="2")
    task1 = launch.ExperimentTask(**{**task0.__dict__, "task_id": 1, "effective_seed": 8, "experiment_id": 1})
    plan = launch.ExperimentPlan(spec=None, tasks=(task0, task1))
    result_root = tmp_path / "results"

    complete_dir = launch.task_output_dir(result_root, task0)
    complete_dir.mkdir(parents=True, exist_ok=True)
    (complete_dir / "jobmeta.txt").write_text("task_id=0\n", encoding="utf-8")
    (complete_dir / "metrics.csv").write_text(
        "epoch,train_acc,train_loss\n0,0.1,1.0\n1,0.2,0.9\n",
        encoding="utf-8",
    )

    incomplete_dir = launch.task_output_dir(result_root, task1)
    incomplete_dir.mkdir(parents=True, exist_ok=True)
    (incomplete_dir / "jobmeta.txt").write_text("task_id=1\n", encoding="utf-8")
    (incomplete_dir / "metrics.csv").write_text(
        "epoch,train_acc,train_loss\n0,0.1,1.0\n",
        encoding="utf-8",
    )

    assert launch.filter_incomplete_task_ids(plan, [0, 1], result_root) == [1]


def test_slurm_array_spec_compacts_sparse_ids():
    launch = _load_launch_module()

    assert launch.slurm_array_spec([1, 2, 3, 6, 8, 9]) == "1-3,6,8-9"
