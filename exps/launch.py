#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import queue
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flpoison.utils.config_utils import preset_relpath, resolve_config_path, resolve_preset_for_scenario
from flpoison.utils.global_utils import setup_console_logger


CONFIG_DEFAULT_FIELDS = (
    "dataset",
    "model",
    "epochs",
    "num_clients",
    "learning_rate",
    "algorithm",
    "seed",
    "num_adv",
    "distribution",
    "dirichlet_alpha",
    "im_iid_gamma",
    "attack",
    "defense",
)

DEFAULT_LOCAL_LOG_DIR = "logs/local_array"
DEFAULT_LOCAL_RESULT_ROOT = "logs/local_runs"
DEFAULT_SLURM_OUTPUT = "logs/slurm/%x_%A_%a.out"
DEFAULT_SLURM_ERROR = "logs/slurm/%x_%A_%a.err"
LOGGER = setup_console_logger("flpoison.launch", level=logging.INFO)


@dataclass(frozen=True)
class ScenarioSpec:
    algorithm: str
    dataset: str
    config: str = ""


@dataclass(frozen=True)
class DistributionChoice:
    kind: str
    dirichlet_alpha: str = ""
    im_iid_gamma: str = ""


@dataclass(frozen=True)
class RuntimeSpec:
    gpu_idx: int = 0
    num_workers: Optional[int] = None
    require_cuda: Optional[bool] = None
    cuda_retry_max: int = 3
    cuda_retry_sleep: int = 20
    cuda_requeue_on_fail: bool = True
    cuda_max_requeue: int = 2


@dataclass(frozen=True)
class SlurmSpec:
    account: str = ""
    time: str = ""
    gpus: str = ""
    cpus_per_task: int = 1
    mem: str = ""
    requeue: bool = True
    output: str = DEFAULT_SLURM_OUTPUT
    error: str = DEFAULT_SLURM_ERROR
    mail_user: str = ""
    mail_type: str = ""
    array_parallel: int = 1
    job_name: str = ""


@dataclass(frozen=True)
class ExperimentSpec:
    path: Path
    name: str
    description: str
    scenarios: Tuple[ScenarioSpec, ...]
    distributions: Tuple[DistributionChoice, ...]
    models: Tuple[str, ...]
    epochs: Tuple[str, ...]
    num_clients: Tuple[str, ...]
    learning_rates: Tuple[str, ...]
    num_advs: Tuple[str, ...]
    seeds: Tuple[str, ...]
    attacks: Tuple[str, ...]
    defenses: Tuple[str, ...]
    experiment_ids: Tuple[int, ...]
    runtime: RuntimeSpec
    slurm: SlurmSpec


@dataclass(frozen=True)
class ExperimentTask:
    task_id: int
    scenario_index: int
    config_file: Path
    config_name: str
    algorithm: str
    dataset: str
    distribution: str
    dirichlet_alpha: str
    im_iid_gamma: str
    model: str
    epochs: str
    num_clients: str
    learning_rate: str
    num_adv: str
    seed_base: int
    effective_seed: int
    attack: str
    defense: str
    experiment_id: int


@dataclass(frozen=True)
class ExperimentPlan:
    spec: ExperimentSpec
    tasks: Tuple[ExperimentTask, ...]

    @property
    def total(self) -> int:
        return len(self.tasks)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def exps_dir() -> Path:
    return Path(__file__).resolve().parent


def specs_dir() -> Path:
    return exps_dir() / "specs"


def have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_list(raw: Any, *, default: Optional[str] = None) -> Tuple[str, ...]:
    if raw is None:
      items = [default] if default is not None else []
    elif isinstance(raw, list):
      items = raw
    else:
      items = [raw]

    values: List[str] = []
    for item in items:
      text = normalize_text(item).strip()
      if text:
        values.append(text)
    if not values and default is not None:
      values.append(default)
    return tuple(values)


def parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = normalize_text(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def parse_optional_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    return parse_bool(value, default=False)


def parse_non_negative_int(value: Any, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"{field_name} must be an integer, got: {value}") from exc
    if out < 0:
        raise ValueError(f"{field_name} must be >= 0, got: {value}")
    return out


def parse_positive_int(value: Any, *, field_name: str) -> int:
    out = parse_non_negative_int(value, field_name=field_name)
    if out < 1:
        raise ValueError(f"{field_name} must be >= 1, got: {value}")
    return out


def parse_distribution_matrix(raw: Any) -> Tuple[DistributionChoice, ...]:
    if raw is None:
        return (DistributionChoice(kind="__cfg__"),)

    items = raw if isinstance(raw, list) else [raw]
    out: List[DistributionChoice] = []
    for item in items:
        if isinstance(item, str):
            kind = item.strip()
            if not kind:
                continue
            out.append(DistributionChoice(kind=kind))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"invalid distribution entry: {item}")

        kind = normalize_text(item.get("type") or item.get("distribution") or item.get("name")).strip()
        if not kind:
            raise ValueError(f"distribution entry is missing `type`: {item}")
        out.append(
            DistributionChoice(
                kind=kind,
                dirichlet_alpha=normalize_text(item.get("dirichlet_alpha")).strip(),
                im_iid_gamma=normalize_text(item.get("im_iid_gamma")).strip(),
            )
        )
    if not out:
        raise ValueError("matrix.distributions resolved to an empty list")
    return tuple(out)


def parse_scenarios(raw: Any) -> Tuple[ScenarioSpec, ...]:
    if raw is None:
        raise ValueError("spec is missing `scenarios`")
    if not isinstance(raw, list) or not raw:
        raise ValueError("`scenarios` must be a non-empty list")

    out: List[ScenarioSpec] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"scenario #{idx} must be a mapping, got: {item}")
        algorithm = normalize_text(item.get("algorithm")).strip()
        dataset = normalize_text(item.get("dataset")).strip()
        config = normalize_text(item.get("config")).strip()
        if not algorithm:
            raise ValueError(f"scenario #{idx} is missing `algorithm`")
        if not dataset:
            raise ValueError(f"scenario #{idx} is missing `dataset`")
        out.append(ScenarioSpec(algorithm=algorithm, dataset=dataset, config=config))
    return tuple(out)


def parse_experiment_ids(matrix_raw: dict[str, Any], repeats_raw: dict[str, Any]) -> Tuple[int, ...]:
    explicit = matrix_raw.get("experiment_ids")
    if explicit is not None:
        items = explicit if isinstance(explicit, list) else [explicit]
        values = tuple(parse_non_negative_int(item, field_name="matrix.experiment_ids") for item in items)
        if not values:
            raise ValueError("matrix.experiment_ids must not be empty")
        return values

    count = parse_positive_int(repeats_raw.get("count", 1), field_name="repeats.count")
    start = parse_non_negative_int(repeats_raw.get("start", 0), field_name="repeats.start")
    return tuple(range(start, start + count))


def load_spec(spec_identifier: str) -> ExperimentSpec:
    spec_path = resolve_spec_path(spec_identifier)
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid spec file: {spec_path}")

    matrix_raw = raw.get("matrix") or {}
    repeats_raw = raw.get("repeats") or {}
    runtime_raw = raw.get("runtime") or {}
    slurm_raw = raw.get("slurm") or {}
    if not isinstance(matrix_raw, dict):
        raise ValueError("`matrix` must be a mapping")
    if not isinstance(repeats_raw, dict):
        raise ValueError("`repeats` must be a mapping")
    if not isinstance(runtime_raw, dict):
        raise ValueError("`runtime` must be a mapping")
    if not isinstance(slurm_raw, dict):
        raise ValueError("`slurm` must be a mapping")

    return ExperimentSpec(
        path=spec_path,
        name=normalize_text(raw.get("name") or spec_path.stem).strip(),
        description=normalize_text(raw.get("description")).strip(),
        scenarios=parse_scenarios(raw.get("scenarios")),
        distributions=parse_distribution_matrix(matrix_raw.get("distributions")),
        models=normalize_list(matrix_raw.get("models"), default="__cfg__"),
        epochs=normalize_list(matrix_raw.get("epochs"), default="__cfg__"),
        num_clients=normalize_list(matrix_raw.get("num_clients"), default="__cfg__"),
        learning_rates=normalize_list(matrix_raw.get("learning_rates"), default="__cfg__"),
        num_advs=normalize_list(matrix_raw.get("num_advs"), default="__cfg__"),
        seeds=normalize_list(matrix_raw.get("seeds"), default="__cfg__"),
        attacks=normalize_list(matrix_raw.get("attacks"), default="__cfg__"),
        defenses=normalize_list(matrix_raw.get("defenses"), default="__cfg__"),
        experiment_ids=parse_experiment_ids(matrix_raw, repeats_raw),
        runtime=RuntimeSpec(
            gpu_idx=parse_non_negative_int(runtime_raw.get("gpu_idx", 0), field_name="runtime.gpu_idx"),
            num_workers=(
                None if runtime_raw.get("num_workers") in (None, "")
                else parse_positive_int(runtime_raw.get("num_workers"), field_name="runtime.num_workers")
            ),
            require_cuda=parse_optional_bool(runtime_raw.get("require_cuda")),
            cuda_retry_max=parse_positive_int(runtime_raw.get("cuda_retry_max", 3), field_name="runtime.cuda_retry_max"),
            cuda_retry_sleep=parse_positive_int(runtime_raw.get("cuda_retry_sleep", 20), field_name="runtime.cuda_retry_sleep"),
            cuda_requeue_on_fail=parse_bool(runtime_raw.get("cuda_requeue_on_fail", True), default=True),
            cuda_max_requeue=parse_non_negative_int(runtime_raw.get("cuda_max_requeue", 2), field_name="runtime.cuda_max_requeue"),
        ),
        slurm=SlurmSpec(
            account=normalize_text(slurm_raw.get("account")).strip(),
            time=normalize_text(slurm_raw.get("time")).strip(),
            gpus=normalize_text(slurm_raw.get("gpus")).strip(),
            cpus_per_task=parse_positive_int(slurm_raw.get("cpus_per_task", 1), field_name="slurm.cpus_per_task"),
            mem=normalize_text(slurm_raw.get("mem")).strip(),
            requeue=parse_bool(slurm_raw.get("requeue", True), default=True),
            output=normalize_text(slurm_raw.get("output") or DEFAULT_SLURM_OUTPUT).strip(),
            error=normalize_text(slurm_raw.get("error") or DEFAULT_SLURM_ERROR).strip(),
            mail_user=normalize_text(slurm_raw.get("mail_user")).strip(),
            mail_type=normalize_text(slurm_raw.get("mail_type")).strip(),
            array_parallel=parse_positive_int(slurm_raw.get("array_parallel", 1), field_name="slurm.array_parallel"),
            job_name=normalize_text(slurm_raw.get("job_name")).strip(),
        ),
    )


def resolve_spec_path(spec_identifier: str) -> Path:
    candidate = Path(spec_identifier)
    if candidate.exists():
        return candidate.resolve()

    if candidate.suffix:
        raise FileNotFoundError(f"spec not found: {spec_identifier}")

    for suffix in (".yaml", ".yml"):
        path = specs_dir() / f"{spec_identifier}{suffix}"
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"spec not found: {spec_identifier}")


def list_spec_files() -> List[Path]:
    paths: List[Path] = []
    if specs_dir().is_dir():
        for suffix in ("*.yaml", "*.yml"):
            paths.extend(sorted(specs_dir().glob(suffix)))
    return sorted(set(path.resolve() for path in paths))


def resolve_config_for_scenario(root: Path, scenario: ScenarioSpec) -> Path:
    if scenario.config:
        return resolve_config_path(scenario.config, root=root)
    return resolve_preset_for_scenario(scenario.algorithm, scenario.dataset, root=root)


def read_config_defaults(config_file: Path) -> dict[str, str]:
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid config file: {config_file}")
    out: dict[str, str] = {}
    for key in CONFIG_DEFAULT_FIELDS:
        out[key] = normalize_text(raw.get(key)).strip()
    return out


def resolve_matrix_value(raw_value: str, *, config_defaults: dict[str, str], config_key: str, label: str) -> str:
    value = raw_value
    if value in {"", "__cfg__"}:
        value = config_defaults.get(config_key, "")
    value = normalize_text(value).strip()
    if not value:
        raise ValueError(f"{label} resolved to empty using config defaults")
    return value


def resolve_distribution(choice: DistributionChoice, config_defaults: dict[str, str]) -> Tuple[str, str, str]:
    distribution = choice.kind
    if distribution in {"", "__cfg__"}:
        distribution = config_defaults.get("distribution", "")
    distribution = normalize_text(distribution).strip()
    if not distribution:
        raise ValueError("distribution resolved to empty using config defaults")

    dirichlet_alpha = ""
    im_iid_gamma = ""
    if distribution == "non-iid":
        dirichlet_alpha = normalize_text(choice.dirichlet_alpha or config_defaults.get("dirichlet_alpha", "")).strip()
        if not dirichlet_alpha:
            raise ValueError("non-iid distribution requires dirichlet_alpha")
    elif distribution == "class-imbalanced_iid":
        im_iid_gamma = normalize_text(choice.im_iid_gamma or config_defaults.get("im_iid_gamma", "")).strip()
        if not im_iid_gamma:
            raise ValueError("class-imbalanced_iid distribution requires im_iid_gamma")

    return distribution, dirichlet_alpha, im_iid_gamma


def build_plan(spec: ExperimentSpec) -> ExperimentPlan:
    root = repo_root()
    tasks: List[ExperimentTask] = []
    task_id = 0

    for scenario_index, scenario in enumerate(spec.scenarios):
        config_file = resolve_config_for_scenario(root, scenario)
        config_defaults = read_config_defaults(config_file)

        for distribution_choice, model, epochs, num_clients, learning_rate, num_adv, seed, attack, defense, experiment_id in product(
            spec.distributions,
            spec.models,
            spec.epochs,
            spec.num_clients,
            spec.learning_rates,
            spec.num_advs,
            spec.seeds,
            spec.attacks,
            spec.defenses,
            spec.experiment_ids,
        ):
            final_distribution, dirichlet_alpha, im_iid_gamma = resolve_distribution(distribution_choice, config_defaults)
            final_model = resolve_matrix_value(model, config_defaults=config_defaults, config_key="model", label="model")
            final_epochs = resolve_matrix_value(epochs, config_defaults=config_defaults, config_key="epochs", label="epochs")
            final_num_clients = resolve_matrix_value(num_clients, config_defaults=config_defaults, config_key="num_clients", label="num_clients")
            final_learning_rate = resolve_matrix_value(
                learning_rate, config_defaults=config_defaults, config_key="learning_rate", label="learning_rate"
            )
            final_num_adv = resolve_matrix_value(num_adv, config_defaults=config_defaults, config_key="num_adv", label="num_adv")
            final_seed_base = parse_non_negative_int(
                resolve_matrix_value(seed, config_defaults=config_defaults, config_key="seed", label="seed"),
                field_name="seed",
            )
            final_attack = resolve_matrix_value(attack, config_defaults=config_defaults, config_key="attack", label="attack")
            final_defense = resolve_matrix_value(defense, config_defaults=config_defaults, config_key="defense", label="defense")

            tasks.append(
                ExperimentTask(
                    task_id=task_id,
                    scenario_index=scenario_index,
                    config_file=config_file,
                    config_name=config_file.name,
                    algorithm=scenario.algorithm,
                    dataset=scenario.dataset,
                    distribution=final_distribution,
                    dirichlet_alpha=dirichlet_alpha,
                    im_iid_gamma=im_iid_gamma,
                    model=final_model,
                    epochs=final_epochs,
                    num_clients=final_num_clients,
                    learning_rate=final_learning_rate,
                    num_adv=final_num_adv,
                    seed_base=final_seed_base,
                    effective_seed=final_seed_base + experiment_id,
                    attack=final_attack,
                    defense=final_defense,
                    experiment_id=experiment_id,
                )
            )
            task_id += 1

    return ExperimentPlan(spec=spec, tasks=tuple(tasks))


def runtime_platform() -> str:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get("SLURM_TMPDIR"):
        return "compute_canada"
    return "local"


def should_require_cuda(runtime: RuntimeSpec, platform: str) -> bool:
    if runtime.require_cuda is not None:
        return runtime.require_cuda
    return platform == "compute_canada"


def resolve_python_bin(root: Path) -> str:
    override = os.environ.get("PYTHON_BIN")
    if override:
        return override

    venv_override = os.environ.get("VENV_PATH")
    if venv_override:
        candidate = Path(venv_override) / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    repo_venv = root / ".venv" / "bin" / "python"
    if repo_venv.exists():
        return str(repo_venv)

    if sys.executable:
        return sys.executable
    if have_cmd("python"):
        return "python"
    if have_cmd("python3"):
        return "python3"
    raise RuntimeError("no usable Python runtime found")


def resolve_data_root(code_root: Path) -> Path:
    override = os.environ.get("DATA_SRC_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    candidates: List[Path] = []
    scratch = os.environ.get("SCRATCH")
    project = os.environ.get("PROJECT")
    if scratch:
        candidates.append(Path(scratch) / "FL_Poison" / "data")
    if project:
        candidates.append(Path(project) / "FL_Poison" / "data")
    candidates.append(code_root / "data")

    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[-1].resolve()


def result_root_for_platform(root: Path, platform: str) -> Path:
    override = os.environ.get("RESULT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    if platform == "local":
        return (root / DEFAULT_LOCAL_RESULT_ROOT).resolve()

    scratch = os.environ.get("SCRATCH")
    if scratch:
        return (Path(scratch) / "FL_Poison" / "logs").resolve()
    return (Path.home() / "scratch" / "FL_Poison" / "logs").resolve()


def resolve_slurm_path(path_text: str, root: Path) -> str:
    text = normalize_text(path_text).strip()
    if not text:
        return ""
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((root / candidate).resolve())


def log_dir_for_task(result_root: Path, task: ExperimentTask) -> Path:
    return result_root / task.algorithm / f"{task.dataset}_{task.model}" / task.distribution


def output_filename_for_task(task: ExperimentTask) -> str:
    parts = [
        task.dataset,
        task.model,
        task.distribution,
        task.attack,
        task.defense,
        task.epochs,
        task.num_clients,
        task.learning_rate,
        task.algorithm,
        f"adv{task.num_adv}",
        f"seed{task.effective_seed}",
        f"exp{task.experiment_id}",
    ]
    if task.distribution == "non-iid" and task.dirichlet_alpha:
        parts.append(f"alpha{task.dirichlet_alpha}")
    if task.distribution == "class-imbalanced_iid" and task.im_iid_gamma:
        parts.append(f"gamma{task.im_iid_gamma}")
    parts.append(f"cfg{Path(task.config_name).stem}")
    return "_".join(parts) + ".txt"


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def log_block(title: str, rows: Sequence[str]) -> None:
    LOGGER.info(title)
    for row in rows:
        LOGGER.info("  %s", row)


def run_cmd_capture(cmd: Sequence[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> str:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.stdout.strip()


def sync_dir_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)

    if have_cmd("rsync"):
        subprocess.run(["rsync", "-a", f"{src}/", f"{dst}/"], check=False)
        return

    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def sync_code_to_local(src: Path, dst: Path) -> None:
    exclude = {".git", ".idea", ".venv", "logs", "running_caches", "data"}
    if have_cmd("rsync"):
        cmd = ["rsync", "-a", "--delete"]
        for pattern in sorted(exclude):
            cmd.extend(["--exclude", pattern])
        cmd.extend([f"{src}/", f"{dst}/"])
        subprocess.run(cmd, check=True)
        return

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*sorted(exclude)))


def copy_path_if_exists(src: Path, dst_root: Path) -> bool:
    if not src.exists():
        return False
    dst_root.mkdir(parents=True, exist_ok=True)
    target = dst_root / src.name
    if src.is_dir():
        if have_cmd("rsync"):
            subprocess.run(["rsync", "-a", f"{src}/", f"{target}/"], check=False)
        else:
            shutil.copytree(src, target, dirs_exist_ok=True)
    else:
        shutil.copy2(src, target)
    return True


def copy_dir_contents(src: Path, dst: Path) -> bool:
    if not src.is_dir():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    if have_cmd("rsync"):
        subprocess.run(["rsync", "-a", f"{src}/", f"{dst}/"], check=False)
        return True
    for child in src.iterdir():
        copy_path_if_exists(child, dst)
    return True


def stage_dataset_to_local(dataset_name: str, src_root: Path, dst_root: Path) -> None:
    copied_any = False
    mapping = {
        "MNIST": ["MNIST"],
        "FashionMNIST": ["FashionMNIST"],
        "EMNIST": ["EMNIST"],
        "CIFAR10": ["cifar-10-batches-py"],
        "CIFAR100": ["cifar-100-python"],
        "TinyImageNet": ["tiny-imagenet-200", "tiny-imagenet-200.zip"],
        "CINIC10": ["CINIC-10", "CINIC-10.tar.gz"],
        "CHMNIST": ["Kather_texture_2016_image_tiles_5000", "Kather_texture_2016_image_tiles_5000.zip"],
        "HAR": ["UCI HAR Dataset", "UCI_HAR_Dataset", "UCI HAR Dataset.zip", "uci_har_cache_v1.npz"],
    }
    for name in mapping.get(dataset_name, []):
        copied_any = copy_path_if_exists(src_root / name, dst_root) or copied_any
    if not copied_any and dataset_name not in mapping:
        copied_any = copy_dir_contents(src_root, dst_root)
    if not copied_any:
        LOGGER.warning(
            "No pre-staged dataset assets found for %s under %s; the job may trigger downloads.",
            dataset_name,
            src_root,
        )


def check_cuda(python_bin: str, *, retry_max: int, retry_sleep: int) -> bool:
    probe = (
        "import sys\n"
        "import torch\n"
        "ok=False\n"
        "try:\n"
        "    ok = torch.cuda.is_available()\n"
        "    if ok:\n"
        "        torch.zeros(1, device='cuda:0')\n"
        "except Exception as exc:\n"
        "    print(f'CUDA check failed: {exc}', file=sys.stderr)\n"
        "    ok = False\n"
        "sys.exit(0 if ok else 1)\n"
    )
    for attempt in range(1, retry_max + 1):
        proc = subprocess.run([python_bin, "-c", probe], check=False)
        if proc.returncode == 0:
            return True
        LOGGER.warning(
            "CUDA not available on %s (try %d/%d).",
            socket.gethostname(),
            attempt,
            retry_max,
        )
        if have_cmd("nvidia-smi"):
            subprocess.run(["nvidia-smi", "-L"], check=False)
        if attempt < retry_max:
            time.sleep(retry_sleep)
    return False


def maybe_requeue_for_cuda_failure(runtime: RuntimeSpec) -> bool:
    if not runtime.cuda_requeue_on_fail or not have_cmd("scontrol"):
        return False

    job_to_requeue = os.environ.get("SLURM_JOB_ID", "")
    if os.environ.get("SLURM_ARRAY_JOB_ID") and os.environ.get("SLURM_ARRAY_TASK_ID"):
        job_to_requeue = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"

    restart_count = parse_non_negative_int(os.environ.get("SLURM_RESTART_COUNT", "0"), field_name="SLURM_RESTART_COUNT")
    if not job_to_requeue or restart_count >= runtime.cuda_max_requeue:
        LOGGER.warning(
            "Max CUDA requeue attempts reached (%d) or job id unavailable.",
            runtime.cuda_max_requeue,
        )
        return False

    LOGGER.warning(
        "CUDA unavailable; requeueing %s (restart_count=%d).",
        job_to_requeue,
        restart_count,
    )
    if os.environ.get("SLURM_NODELIST"):
        subprocess.run(
            ["scontrol", "update", f"JobId={job_to_requeue}", f"ExcNodeList={os.environ['SLURM_NODELIST']}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    proc = subprocess.run(
        ["scontrol", "requeue", job_to_requeue],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode == 0:
        return True
    LOGGER.warning("scontrol requeue failed; falling back to a hard failure.")
    return False


def git_value(root: Path, *args: str) -> str:
    if not have_cmd("git"):
        return ""
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return (proc.stdout or "").strip()


def write_metadata(path: Path, items: List[Tuple[str, str]], cmd: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in items:
            handle.write(f"{key}={value}\n")
        handle.write(f"command={format_command(cmd)}\n")


def task_command(
    *,
    python_bin: str,
    run_repo: Path,
    config_path: Path,
    task: ExperimentTask,
    runtime: RuntimeSpec,
    output_file: Path,
) -> List[str]:
    cmd = [
        python_bin,
        "-u",
        "-m",
        "flpoison",
        f"--config={config_path}",
        "--algorithm",
        task.algorithm,
        "--dataset",
        task.dataset,
        "--distribution",
        task.distribution,
        "--model",
        task.model,
        "--epochs",
        task.epochs,
        "--num_clients",
        task.num_clients,
        "--learning_rate",
        task.learning_rate,
        "--num_adv",
        task.num_adv,
        "--seed",
        str(task.seed_base),
        "--num_experiments",
        "1",
        "--experiment_id",
        str(task.experiment_id),
        "--attack",
        task.attack,
        "--defense",
        task.defense,
        "--gpu_idx",
        str(runtime.gpu_idx),
        "--output",
        str(output_file),
    ]
    if task.distribution == "non-iid" and task.dirichlet_alpha:
        cmd.extend(["--dirichlet_alpha", task.dirichlet_alpha])
    if task.distribution == "class-imbalanced_iid" and task.im_iid_gamma:
        cmd.extend(["--im_iid_gamma", task.im_iid_gamma])
    if runtime.num_workers is not None:
        cmd.extend(["--num_workers", str(runtime.num_workers)])
    return cmd


def worker_main(args: argparse.Namespace) -> int:
    root = Path(os.environ.get("CODE_SRC_ROOT", str(repo_root()))).expanduser().resolve()
    spec = load_spec(args.spec)
    plan = build_plan(spec)

    task_id = args.task_id
    if task_id is None:
        env_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_task_id is None:
            LOGGER.error("worker requires --task-id or SLURM_ARRAY_TASK_ID")
            return 2
        task_id = parse_non_negative_int(env_task_id, field_name="SLURM_ARRAY_TASK_ID")

    if task_id < 0 or task_id >= plan.total:
        LOGGER.error("task id out of range 0..%d: %d", plan.total - 1, task_id)
        return 2

    task = plan.tasks[task_id]
    platform = runtime_platform()
    python_bin = resolve_python_bin(root)
    data_root = resolve_data_root(root)
    result_root = result_root_for_platform(root, platform)
    final_log_dir = log_dir_for_task(result_root, task)
    final_log_dir.mkdir(parents=True, exist_ok=True)

    output_basename = output_filename_for_task(task)
    final_output_file = final_log_dir / output_basename
    scratch_root = Path(os.environ.get("SCRATCH", str(Path.home() / "scratch"))).expanduser().resolve()

    run_repo = root
    runtime_output_file = final_output_file
    local_results_dir: Optional[Path] = None
    local_repo: Optional[Path] = None

    if os.environ.get("SLURM_TMPDIR"):
        slurm_tmpdir = Path(os.environ["SLURM_TMPDIR"]).expanduser().resolve()
        if slurm_tmpdir.exists():
            local_run_dir = slurm_tmpdir / f"flpoison_{os.environ.get('SLURM_JOB_ID', '0')}_{task.task_id}"
            local_repo = local_run_dir / "FL_Poison"
            local_results_dir = local_run_dir / "results"
            local_results_dir.mkdir(parents=True, exist_ok=True)
            sync_code_to_local(root, local_repo)
            if data_root.exists():
                stage_dataset_to_local(task.dataset, data_root, local_repo / "data")
            else:
                LOGGER.warning(
                    "DATA_SRC_ROOT not found: %s. Dataset may download and stress shared filesystems.",
                    data_root,
                )
            run_repo = local_repo
            runtime_output_file = local_results_dir / output_basename

    config_path = task.config_file
    if run_repo != root:
        try:
            config_path = run_repo / task.config_file.relative_to(root)
        except ValueError:
            config_path = task.config_file

    metadata_target_dir = local_results_dir if local_results_dir is not None else final_log_dir
    metadata_file = metadata_target_dir / f"{Path(output_basename).stem}_jobmeta.txt"

    task_cmd = task_command(
        python_bin=python_bin,
        run_repo=run_repo,
        config_path=config_path,
        task=task,
        runtime=spec.runtime,
        output_file=runtime_output_file,
    )

    log_block(
        "Job context:",
        [
            f"spec={spec.path}",
            f"task_id={task.task_id}",
            f"runtime_platform={platform}",
            f"host={socket.gethostname()}",
            f"job_id={os.environ.get('SLURM_JOB_ID', 'n/a')} array_task_id={os.environ.get('SLURM_ARRAY_TASK_ID', str(task.task_id))}",
            f"code_root={root}",
            f"data_root={data_root}",
            f"python_bin={python_bin}",
            f"scratch_root={scratch_root}",
            f"result_root={result_root}",
            f"slurm_tmpdir={os.environ.get('SLURM_TMPDIR', '')}",
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        ],
    )
    log_block(
        "Resolved experiment:",
        [
            f"config={task.config_name}",
            f"scenario_alg={task.algorithm} scenario_data={task.dataset}",
            f"distribution={task.distribution} dirichlet_alpha={task.dirichlet_alpha} im_iid_gamma={task.im_iid_gamma}",
            f"model={task.model} epochs={task.epochs} num_clients={task.num_clients} lr={task.learning_rate}",
            f"num_adv={task.num_adv} base_seed={task.seed_base} effective_seed={task.effective_seed}",
            f"experiment_id={task.experiment_id}",
            f"attack={task.attack} defense={task.defense}",
        ],
    )

    cuda_ok = check_cuda(
        python_bin,
        retry_max=spec.runtime.cuda_retry_max,
        retry_sleep=spec.runtime.cuda_retry_sleep,
    )
    require_cuda = should_require_cuda(spec.runtime, platform)
    if not cuda_ok:
        if require_cuda:
            if maybe_requeue_for_cuda_failure(spec.runtime):
                return 0
            LOGGER.error("CUDA still unavailable; aborting to avoid silent CPU fallback.")
            return 1
        LOGGER.warning("CUDA unavailable; continuing with local CPU/MPS fallback.")

    write_metadata(
        metadata_file,
        [
            ("spec", str(spec.path)),
            ("spec_name", spec.name),
            ("task_id", str(task.task_id)),
            ("runtime_platform", platform),
            ("job_id", os.environ.get("SLURM_JOB_ID", "")),
            ("array_job_id", os.environ.get("SLURM_ARRAY_JOB_ID", "")),
            ("array_task_id", os.environ.get("SLURM_ARRAY_TASK_ID", str(task.task_id))),
            ("host", socket.gethostname()),
            ("code_root", str(root)),
            ("data_root", str(data_root)),
            ("run_repo", str(run_repo)),
            ("result_root", str(result_root)),
            ("output_file", str(final_output_file)),
            ("runtime_output_file", str(runtime_output_file)),
            ("config_file", str(config_path)),
            ("cuda_required", str(require_cuda)),
            ("cuda_probe_ok", str(cuda_ok)),
            ("experiment_id", str(task.experiment_id)),
            ("base_seed", str(task.seed_base)),
            ("effective_seed", str(task.effective_seed)),
            ("python_bin", python_bin),
            ("python_version", run_cmd_capture([python_bin, "--version"])),
            ("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            ("git_commit", git_value(root, "rev-parse", "HEAD") or "unknown"),
            ("git_branch", git_value(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"),
        ],
        task_cmd,
    )

    log_block("Launch command:", [format_command(task_cmd)])

    proc: Optional[subprocess.Popen[str]] = None

    def sync_results() -> None:
        if local_results_dir is not None:
            sync_dir_contents(local_results_dir, final_log_dir)
            LOGGER.info("Saved results to: %s", final_log_dir)

    def on_signal(signum: int, _frame: Any) -> None:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        sync_results()
        raise SystemExit(128 + signum)

    old_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(os.environ.get("SLURM_CPUS_PER_TASK", spec.slurm.cpus_per_task)))
    env.setdefault("MKL_NUM_THREADS", env["OMP_NUM_THREADS"])

    try:
        proc = subprocess.Popen(task_cmd, cwd=str(run_repo), env=env, text=True)
        rc = proc.wait()
    finally:
        sync_results()
        signal.signal(signal.SIGINT, old_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, old_handlers[signal.SIGTERM])

    return int(rc)


def parse_ids(raw: str, total: int) -> List[int]:
    text = raw.strip()
    if text.lower() == "all":
        return list(range(total))
    out: List[int] = []
    for part in [item.strip() for item in text.split(",") if item.strip()]:
        if "-" in part or ":" in part:
            sep = "-" if "-" in part else ":"
            left, right = part.split(sep, 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"invalid range: {part}")
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    deduped: List[int] = []
    seen = set()
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def log_tail_has_success(log_path: Path) -> bool:
    try:
        data = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False
    lines = [line.strip() for line in data.splitlines() if line.strip()]
    return bool(lines) and lines[-1] == "LOCAL_ARRAY_EXIT_CODE=0"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


class TokenLock:
    def __init__(self, lock_dir: Path, tokens: int, poll_sec: float):
        if tokens < 1:
            raise ValueError("tokens must be >= 1")
        if poll_sec <= 0:
            raise ValueError("poll_sec must be > 0")
        self.lock_dir = lock_dir
        self.tokens = int(tokens)
        self.poll_sec = float(poll_sec)

        try:
            import fcntl  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Token locks require POSIX flock support") from exc

    def acquire(self):  # noqa: ANN001
        import fcntl

        ensure_dir(self.lock_dir)
        lock_files = [self.lock_dir / f"gpu_token_{idx}.lock" for idx in range(self.tokens)]
        for path in lock_files:
            if not path.exists():
                path.write_text("", encoding="utf-8")

        fh = None
        acquired_path = None
        while True:
            for path in lock_files:
                try:
                    handle = path.open("r+", encoding="utf-8")
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fh = handle
                        acquired_path = path
                        break
                    except BlockingIOError:
                        handle.close()
                except FileNotFoundError:
                    continue
            if fh is not None:
                break
            time.sleep(self.poll_sec)

        class _Ctx:
            def __enter__(self_inner):  # noqa: ANN001
                return str(acquired_path)

            def __exit__(self_inner, exc_type, exc, tb):  # noqa: ANN001
                try:
                    if fh is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                        fh.close()
                finally:
                    return False

        return _Ctx()


def local_worker_run_task(
    spec_path: Path,
    task_id: int,
    env: dict[str, str],
    log_path: Path,
    *,
    dry_run: bool,
) -> int:
    ensure_dir(log_path.parent)
    started = time.time()
    python_bin = resolve_python_bin(repo_root())
    cmd = [python_bin, str(Path(__file__).resolve()), "worker", str(spec_path), "--task-id", str(task_id)]

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"LOCAL_ARRAY_SPEC={spec_path}\n")
        handle.write(f"LOCAL_ARRAY_TASK_ID={task_id}\n")
        handle.write(f"LOCAL_ARRAY_STARTED_AT={format_ts(started)}\n")
        handle.write(f"LOCAL_ARRAY_CMD={format_command(cmd)}\n")
        handle.write(f"LOCAL_ARRAY_CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n\n")
        handle.flush()

        if dry_run:
            handle.write("LOCAL_ARRAY_DRY_RUN=1\n")
            handle.write("LOCAL_ARRAY_EXIT_CODE=0\n")
            return 0

        proc = subprocess.Popen(cmd, cwd=str(repo_root()), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
        ended = time.time()
        handle.write("\n")
        handle.write(f"LOCAL_ARRAY_ENDED_AT={format_ts(ended)}\n")
        handle.write(f"LOCAL_ARRAY_DURATION_SEC={ended - started:.1f}\n")
        handle.write(f"LOCAL_ARRAY_EXIT_CODE={rc}\n")
        return int(rc)


def local_main(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    plan = build_plan(spec)
    log_dir = (repo_root() / args.log_dir).resolve()

    if args.jobs < 1:
        LOGGER.error("--jobs must be >= 1")
        return 2
    if args.gpu_tokens < 1:
        LOGGER.error("--gpu-tokens must be >= 1")
        return 2
    if args.gpu_lock_poll <= 0:
        LOGGER.error("--gpu-lock-poll must be > 0")
        return 2

    ids = parse_ids(args.ids, plan.total)
    bad = [task_id for task_id in ids if task_id < 0 or task_id >= plan.total]
    if bad:
        LOGGER.error("task ids out of range 0..%d: %s", plan.total - 1, bad[:20])
        return 2

    if args.resume:
        filtered: List[int] = []
        for task_id in ids:
            log_path = log_dir / f"{spec.name}_task{task_id}.out"
            if not log_tail_has_success(log_path):
                filtered.append(task_id)
        ids = filtered

    LOGGER.info("SPEC=%s", spec.name)
    LOGGER.info("TOTAL=%d", plan.total)
    LOGGER.info("RUN_IDS=%d (first=%s, last=%s)", len(ids), ids[0] if ids else 'n/a', ids[-1] if ids else 'n/a')
    LOGGER.info("LOG_DIR=%s", log_dir)
    LOGGER.info("CUDA_VISIBLE_DEVICES=%s", args.cuda)
    LOGGER.info("JOBS=%d GPU_TOKENS=%d", args.jobs, args.gpu_tokens)
    if not ids:
        return 0

    failures: List[Tuple[int, int]] = []
    base_env = os.environ.copy()
    base_env.pop("SLURM_ARRAY_TASK_ID", None)
    base_env["CUDA_VISIBLE_DEVICES"] = args.cuda

    token_lock_dir = (log_dir / args.gpu_lock_dir).resolve()
    LOGGER.info("GPU_LOCK_DIR=%s GPU_LOCK_POLL=%s", token_lock_dir, args.gpu_lock_poll)

    def run_single(task_id: int) -> Tuple[int, int, Path]:
        log_path = log_dir / f"{spec.name}_task{task_id}.out"
        if args.gpu_tokens > 0:
            lock = TokenLock(token_lock_dir, args.gpu_tokens, args.gpu_lock_poll)
            with lock.acquire():
                rc = local_worker_run_task(spec.path, task_id, base_env, log_path, dry_run=args.dry_run)
        else:
            rc = local_worker_run_task(spec.path, task_id, base_env, log_path, dry_run=args.dry_run)
        return task_id, rc, log_path

    if args.jobs == 1:
        for idx, task_id in enumerate(ids, start=1):
            log_path = log_dir / f"{spec.name}_task{task_id}.out"
            LOGGER.info("[%d/%d] run task=%d -> %s", idx, len(ids), task_id, log_path)
            tid, rc, _ = run_single(task_id)
            if rc != 0:
                failures.append((tid, rc))
                LOGGER.error("FAIL task=%d rc=%d", tid, rc)
                if args.stop_on_fail:
                    break
    else:
        task_q: "queue.Queue[int]" = queue.Queue()
        result_q: "queue.Queue[Tuple[int, int, Path]]" = queue.Queue()
        stop_evt = threading.Event()
        for task_id in ids:
            task_q.put(task_id)

        def thread_worker() -> None:
            while not stop_evt.is_set():
                try:
                    task_id = task_q.get_nowait()
                except queue.Empty:
                    return
                try:
                    result_q.put(run_single(task_id))
                finally:
                    task_q.task_done()

        threads = [threading.Thread(target=thread_worker, daemon=True) for _ in range(args.jobs)]
        for thread in threads:
            thread.start()

        done = 0
        total_count = len(ids)
        while done < total_count:
            try:
                tid, rc, log_path = result_q.get(timeout=0.5)
            except queue.Empty:
                if stop_evt.is_set() and task_q.unfinished_tasks == 0:
                    break
                continue
            done += 1
            LOGGER.info("[%d/%d] done task=%d rc=%d -> %s", done, total_count, tid, rc, log_path)
            if rc != 0:
                failures.append((tid, rc))
                LOGGER.error("FAIL task=%d rc=%d", tid, rc)
                if args.stop_on_fail:
                    stop_evt.set()

        stop_evt.set()
        for thread in threads:
            thread.join(timeout=1.0)

    if failures:
        LOGGER.error("FAILED_TASKS: %s", ", ".join(f"{tid}:{rc}" for tid, rc in failures))
        return 1
    return 0


def chunk_ranges(start: int, end_inclusive: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    cur = start
    while cur <= end_inclusive:
        chunk_end = min(cur + chunk_size - 1, end_inclusive)
        yield cur, chunk_end
        cur = chunk_end + 1


def cc_main(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    plan = build_plan(spec)
    root = repo_root()

    if args.chunk_size < 1:
        LOGGER.error("--chunk-size must be >= 1")
        return 2
    if args.start_id < 0:
        LOGGER.error("--start-id must be >= 0")
        return 2

    end_id = plan.total - 1 if args.end_id is None else args.end_id
    if end_id < args.start_id:
        LOGGER.error("--end-id must be >= --start-id")
        return 2
    if end_id >= plan.total:
        LOGGER.error("--end-id must be <= %d", plan.total - 1)
        return 2

    array_parallel = args.array_parallel or spec.slurm.array_parallel
    entry_script = (exps_dir() / "slurm_array_entry.sh").resolve()
    job_name = spec.slurm.job_name or spec.name

    chunks = list(chunk_ranges(args.start_id, end_id, args.chunk_size))
    LOGGER.info("SPEC=%s", spec.name)
    LOGGER.info("SPEC_FILE=%s", spec.path)
    LOGGER.info("TOTAL=%d", plan.total)
    LOGGER.info("SUBMIT_RANGE=%d-%d", args.start_id, end_id)
    LOGGER.info("CHUNK_SIZE=%d", args.chunk_size)
    LOGGER.info("ARRAY_PARALLEL=%d", array_parallel)
    LOGGER.info("CHUNKS=%d", len(chunks))

    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        cmd: List[str] = ["sbatch", f"--array={chunk_start}-{chunk_end}%{array_parallel}"]

        if job_name:
            cmd.append(f"--job-name={job_name}")
        if spec.slurm.account:
            cmd.append(f"--account={spec.slurm.account}")
        if spec.slurm.time:
            cmd.append(f"--time={spec.slurm.time}")
        if spec.slurm.gpus:
            cmd.append(f"--gpus={spec.slurm.gpus}")
        if spec.slurm.cpus_per_task:
            cmd.append(f"--cpus-per-task={spec.slurm.cpus_per_task}")
        if spec.slurm.mem:
            cmd.append(f"--mem={spec.slurm.mem}")
        if spec.slurm.output:
            cmd.append(f"--output={resolve_slurm_path(spec.slurm.output, root)}")
        if spec.slurm.error:
            cmd.append(f"--error={resolve_slurm_path(spec.slurm.error, root)}")
        if spec.slurm.mail_user:
            cmd.append(f"--mail-user={spec.slurm.mail_user}")
        if spec.slurm.mail_type:
            cmd.append(f"--mail-type={spec.slurm.mail_type}")
        if spec.slurm.requeue:
            cmd.append("--requeue")

        cmd.extend(args.sbatch_arg)
        cmd.append(str(entry_script))
        cmd.append(str(spec.path))

        LOGGER.info("[%d/%d] %s", idx, len(chunks), format_command(cmd))
        if args.dry_run:
            continue

        proc = subprocess.run(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = (proc.stdout or "").rstrip()
        if output:
            LOGGER.info(output)
        if proc.returncode != 0:
            return int(proc.returncode)

    return 0


def plan_main(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    plan = build_plan(spec)

    dims = {
        "scenarios": len(spec.scenarios),
        "distributions": len(spec.distributions),
        "models": len(spec.models),
        "epochs": len(spec.epochs),
        "num_clients": len(spec.num_clients),
        "learning_rates": len(spec.learning_rates),
        "num_advs": len(spec.num_advs),
        "seeds": len(spec.seeds),
        "attacks": len(spec.attacks),
        "defenses": len(spec.defenses),
        "experiment_ids": len(spec.experiment_ids),
    }

    LOGGER.info("SPEC=%s", spec.name)
    LOGGER.info("SPEC_FILE=%s", spec.path)
    if spec.description:
        LOGGER.info("DESCRIPTION=%s", spec.description)
    LOGGER.info("TOTAL=%d", plan.total)
    LOGGER.info("GRID_DIMS")
    for key, value in dims.items():
        LOGGER.info("  %s=%s", key, value)
    LOGGER.info("SCENARIOS")
    for idx, scenario in enumerate(spec.scenarios):
        config = scenario.config or preset_relpath(scenario.algorithm, scenario.dataset).as_posix()
        LOGGER.info("  %d: algorithm=%s dataset=%s config=%s", idx, scenario.algorithm, scenario.dataset, config)
    LOGGER.info("RUN_EXAMPLES")
    LOGGER.info("  local: exps/run_local.sh %s", spec.path)
    LOGGER.info("  cc:    exps/run_cc.sh %s", spec.path)
    return 0


def list_main(_: argparse.Namespace) -> int:
    for path in list_spec_files():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        name = normalize_text(raw.get("name") or path.stem).strip()
        description = normalize_text(raw.get("description")).strip()
        if description:
            LOGGER.info("%s\t%s\t%s", name, path, description)
        else:
            LOGGER.info("%s\t%s", name, path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified experiment launcher: spec-driven planning, local execution, and Compute Canada submission."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    list_ap = sub.add_parser("list", help="List available experiment specs.")
    list_ap.set_defaults(func=list_main)

    plan_ap = sub.add_parser("plan", help="Show the resolved size and shape of an experiment matrix.")
    plan_ap.add_argument("spec", type=str, help="Spec path or spec name under exps/specs.")
    plan_ap.set_defaults(func=plan_main)

    local_ap = sub.add_parser("local", help="Run a spec locally by iterating task ids.")
    local_ap.add_argument("spec", type=str, help="Spec path or spec name under exps/specs.")
    local_ap.add_argument("--ids", type=str, default="all", help='Task ids to run, e.g. "0-31", "0,3,8-10", or "all".')
    local_ap.add_argument("--log-dir", type=str, default=DEFAULT_LOCAL_LOG_DIR, help="Per-task local runner logs.")
    local_ap.add_argument("--cuda", type=str, default="0", help='Value for CUDA_VISIBLE_DEVICES (default: "0").')
    local_ap.add_argument("--jobs", type=int, default=1, help="Max number of concurrent local worker processes.")
    local_ap.add_argument("--gpu-tokens", type=int, default=1, help="Concurrency limit enforced with file locks.")
    local_ap.add_argument("--gpu-lock-dir", type=str, default="gpu_locks", help="Directory for GPU token lock files.")
    local_ap.add_argument("--gpu-lock-poll", type=float, default=1.0, help="Polling interval when waiting for a token.")
    local_ap.add_argument("--resume", action="store_true", help="Skip tasks whose local log already ends with success.")
    local_ap.add_argument("--dry-run", action="store_true", help="Write local runner logs without launching workers.")
    local_ap.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failing task.")
    local_ap.set_defaults(func=local_main)

    cc_ap = sub.add_parser("cc", help="Submit a spec to Compute Canada in chunked Slurm arrays.")
    cc_ap.add_argument("spec", type=str, help="Spec path or spec name under exps/specs.")
    cc_ap.add_argument("--chunk-size", type=int, default=32, help="Number of task ids per sbatch submission.")
    cc_ap.add_argument("--array-parallel", type=int, default=None, help="Override spec.slurm.array_parallel.")
    cc_ap.add_argument("--start-id", type=int, default=0, help="First task id to submit.")
    cc_ap.add_argument("--end-id", type=int, default=None, help="Last task id to submit, inclusive.")
    cc_ap.add_argument("--sbatch-arg", action="append", default=[], help="Extra sbatch argument, e.g. --sbatch-arg=--qos=high.")
    cc_ap.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting them.")
    cc_ap.set_defaults(func=cc_main)

    worker_ap = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker_ap.add_argument("spec", type=str, help="Spec path or spec name under exps/specs.")
    worker_ap.add_argument("--task-id", type=int, default=None, help="Explicit task id. Defaults to SLURM_ARRAY_TASK_ID.")
    worker_ap.set_defaults(func=worker_main)

    return ap


def main(argv: Sequence[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv))
    try:
        return int(args.func(args))
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
