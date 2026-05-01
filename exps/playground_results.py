#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shlex
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import launch
from flpoison.utils.output_utils import run_log_path


def default_playground_root() -> Path:
    candidates = [
        Path.home() / "Projects" / "Poisoning_Resilient_Federated_Learning_Playground",
        Path.home() / "Project" / "Poisoning_Resilient_Federated_Learning_Playground",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_PLAYGROUND_ROOT = default_playground_root()
EPOCH_LINE_RE = re.compile(r"^Epoch\s+(?P<epoch>\d+)\b")
SUMMARY_KV_RE = re.compile(r"^\s{2,}(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<value>.*)$")


def parse_ids(raw: str, total: int) -> list[int]:
    return launch.parse_ids(raw, total)


def read_kv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_command_options(command: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not command:
        return out
    try:
        tokens = shlex.split(command)
    except ValueError:
        return out

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--"):
            idx += 1
            continue
        key_value = token[2:]
        if "=" in key_value:
            key, value = key_value.split("=", 1)
            out[key] = value
            idx += 1
            continue
        key = key_value
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            out[key] = tokens[idx + 1]
            idx += 2
        else:
            out[key] = "true"
            idx += 1
    return out


def parse_config_summary(run_log: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not run_log.exists():
        return out
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SUMMARY_KV_RE.match(line)
        if match:
            out[match.group("key")] = match.group("value").strip()
    return out


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_component(value: object) -> str:
    text = str(value).strip()
    return text.replace("/", "_").replace(" ", "")


def dataset_model_name(dataset: str, model: str) -> str:
    return f"{normalize_component(dataset)}_{normalize_component(model)}"


def config_stem_from_task(task: launch.ExperimentTask) -> str:
    return Path(task.config_name).stem


def legacy_filename(
    *,
    dataset: str,
    model: str,
    distribution: str,
    attack: str,
    defense: str,
    epochs: str,
    num_clients: str,
    learning_rate: str,
    algorithm: str,
    adv_ratio: str,
    seed: str,
    experiment_id: str,
    config_stem: str,
    dirichlet_alpha: str = "",
    im_iid_gamma: str = "",
) -> str:
    prefix = dataset_model_name(dataset, model)
    parts = [
        prefix,
        normalize_component(distribution),
        normalize_component(attack),
        normalize_component(defense),
        normalize_component(epochs),
        normalize_component(num_clients),
        normalize_component(learning_rate),
        normalize_component(algorithm),
        f"adv{normalize_component(adv_ratio)}",
        f"seed{normalize_component(seed)}",
    ]
    if distribution == "non-iid" and dirichlet_alpha:
        parts.append(f"alpha{normalize_component(dirichlet_alpha)}")
    if distribution == "class-imbalanced_iid" and im_iid_gamma:
        parts.append(f"gamma{normalize_component(im_iid_gamma)}")
    parts.extend([f"cfg{normalize_component(config_stem)}", f"exp{normalize_component(experiment_id)}"])
    return "_".join(parts) + ".txt"


def legacy_path_for_task(playground_root: Path, task: launch.ExperimentTask) -> Path:
    filename = legacy_filename(
        dataset=task.dataset,
        model=task.model,
        distribution=task.distribution,
        attack=task.attack,
        defense=task.defense,
        epochs=task.epochs,
        num_clients=task.num_clients,
        learning_rate=task.learning_rate,
        algorithm=task.algorithm,
        adv_ratio=task.num_adv,
        seed=str(task.effective_seed),
        experiment_id=str(task.experiment_id),
        config_stem=config_stem_from_task(task),
        dirichlet_alpha=task.dirichlet_alpha,
        im_iid_gamma=task.im_iid_gamma,
    )
    return (
        playground_root
        / task.algorithm
        / dataset_model_name(task.dataset, task.model)
        / task.distribution
        / filename
    )


def legacy_glob_for_task(playground_root: Path, task: launch.ExperimentTask) -> str:
    exact = legacy_path_for_task(playground_root, task)
    return exact.name.replace(f"_cfg{config_stem_from_task(task)}_", "_cfg*_")


def legacy_epoch_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if EPOCH_LINE_RE.match(line):
                count += 1
    return count


def is_legacy_complete(path: Path, expected_epochs: int) -> bool:
    if not path.exists():
        return False
    if expected_epochs <= 0:
        return True
    return legacy_epoch_count(path) >= expected_epochs


def find_legacy_match(
    playground_root: Path,
    task: launch.ExperimentTask,
    *,
    match_config: str,
) -> Path | None:
    exact = legacy_path_for_task(playground_root, task)
    expected_epochs = int(task.epochs)
    if is_legacy_complete(exact, expected_epochs):
        return exact
    if match_config != "any":
        return None
    parent = exact.parent
    for candidate in sorted(parent.glob(legacy_glob_for_task(playground_root, task))):
        if is_legacy_complete(candidate, expected_epochs):
            return candidate
    return None


def read_metrics_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_log_lines(run_log: Path) -> list[str]:
    if not run_log.exists():
        return []
    return run_log.read_text(encoding="utf-8", errors="replace").splitlines()


def extract_preamble(lines: Sequence[str]) -> list[str]:
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("Started on "):
            start = idx
            break
    if start is None:
        return []

    out: list[str] = []
    for line in lines[start:]:
        if EPOCH_LINE_RE.match(line):
            break
        if line.startswith("Global Epochs:"):
            break
        if line.startswith("CARAT "):
            continue
        if line.startswith("Metrics output |") or line.startswith("Run log |"):
            continue
        out.append(line)
    return out


def extract_finish_line(lines: Sequence[str]) -> str:
    for line in reversed(lines):
        if line.startswith("Training finished on "):
            return line
    return ""


def to_float(text: str | None, default: float = 0.0) -> float:
    if text is None or text == "":
        return default
    return float(text)


def scaled_loss(value: str | None, loss_scale: float) -> float:
    if loss_scale == 0:
        loss_scale = 1.0
    return to_float(value) / loss_scale


def parse_loss_scale(raw: str, summary: Mapping[str, str]) -> float:
    if raw == "auto":
        return to_float(summary.get("batch_size"), 64.0)
    return float(raw)


def legacy_header(
    *,
    summary: Mapping[str, str],
    metadata: Mapping[str, str],
    cli: Mapping[str, str],
    legacy_output: Path,
) -> str:
    config_file = metadata.get("config_file", "")
    config_stem = Path(config_file).stem if config_file else ""
    fields = [
        ("seed", metadata.get("effective_seed") or summary.get("seed") or cli.get("seed")),
        ("num_experiments", summary.get("num_experiments", "1")),
        ("experiment_id", metadata.get("experiment_id") or summary.get("experiment_id")),
        ("epochs", cli.get("epochs") or summary.get("epochs")),
        ("algorithm", cli.get("algorithm") or summary.get("algorithm")),
        ("optimizer", summary.get("optimizer")),
        ("momentum", summary.get("momentum")),
        ("weight_decay", summary.get("weight_decay")),
        ("lr_scheduler", summary.get("lr_scheduler")),
        ("milestones", summary.get("milestones")),
        ("num_clients", cli.get("num_clients") or summary.get("num_clients")),
        ("batch_size", summary.get("batch_size")),
        ("learning_rate", cli.get("learning_rate") or summary.get("learning_rate")),
        ("local_epochs", summary.get("local_epochs")),
        ("model", cli.get("model") or summary.get("model")),
        ("dataset", cli.get("dataset") or summary.get("dataset")),
        ("distribution", cli.get("distribution") or summary.get("distribution")),
        ("im_iid_gamma", cli.get("im_iid_gamma") or summary.get("im_iid_gamma")),
        ("tail_cls_from", summary.get("tail_cls_from")),
        ("dirichlet_alpha", cli.get("dirichlet_alpha") or summary.get("dirichlet_alpha")),
        ("cache_partition", summary.get("cache_partition")),
        ("gpu_idx", summary.get("gpu_idx")),
        ("num_workers", summary.get("num_workers")),
        ("record_time", summary.get("record_time")),
        ("log_stream", summary.get("log_stream")),
        ("num_adv", summary.get("num_adv")),
        ("attack", cli.get("attack") or summary.get("attack")),
        ("defense", cli.get("defense") or summary.get("defense")),
        ("attack_params", summary.get("attack_params")),
        ("defense_params", summary.get("defense_params")),
        ("benchmark", summary.get("benchmark", "False")),
        ("output", str(legacy_output)),
        ("num_training_sample", summary.get("num_training_sample")),
        ("num_channels", summary.get("num_channels")),
        ("num_classes", summary.get("num_classes")),
        ("mean", summary.get("mean")),
        ("std", summary.get("std")),
        ("device", summary.get("device")),
        ("seed_start", summary.get("seed_start") or cli.get("seed")),
        ("config", config_stem),
    ]
    return ", ".join(f"{key}: {value}" for key, value in fields if value not in (None, ""))


def legacy_path_from_artifacts(
    playground_root: Path,
    *,
    metadata: Mapping[str, str],
    summary: Mapping[str, str],
    cli: Mapping[str, str],
) -> Path:
    config_file = metadata.get("config_file", "")
    config_stem = Path(config_file).stem if config_file else "unknown_config"
    dataset = cli.get("dataset") or summary.get("dataset") or "unknown_dataset"
    model = cli.get("model") or summary.get("model") or "unknown_model"
    distribution = cli.get("distribution") or summary.get("distribution") or "unknown_distribution"
    algorithm = cli.get("algorithm") or summary.get("algorithm") or "unknown_algorithm"
    filename = legacy_filename(
        dataset=dataset,
        model=model,
        distribution=distribution,
        attack=cli.get("attack") or summary.get("attack") or "unknown_attack",
        defense=cli.get("defense") or summary.get("defense") or "unknown_defense",
        epochs=cli.get("epochs") or summary.get("epochs") or "unknown_epochs",
        num_clients=cli.get("num_clients") or summary.get("num_clients") or "unknown_clients",
        learning_rate=cli.get("learning_rate") or summary.get("learning_rate") or "unknown_lr",
        algorithm=algorithm,
        adv_ratio=cli.get("num_adv") or "unknown_adv",
        seed=metadata.get("effective_seed") or summary.get("seed") or cli.get("seed") or "unknown_seed",
        experiment_id=metadata.get("experiment_id") or summary.get("experiment_id") or "0",
        config_stem=config_stem,
        dirichlet_alpha=cli.get("dirichlet_alpha") or summary.get("dirichlet_alpha") or "",
        im_iid_gamma=cli.get("im_iid_gamma") or summary.get("im_iid_gamma") or "",
    )
    return playground_root / algorithm / dataset_model_name(dataset, model) / distribution / filename


def render_legacy_text(
    *,
    metrics_path: Path,
    run_log: Path,
    metadata: Mapping[str, str],
    summary: Mapping[str, str],
    cli: Mapping[str, str],
    legacy_output: Path,
    loss_scale: float,
    allow_partial: bool,
) -> str:
    rows = read_metrics_rows(metrics_path)
    lines = run_log_lines(run_log)
    finish_line = extract_finish_line(lines)
    if not finish_line and not allow_partial:
        raise ValueError(f"run is not finished: {metrics_path}")

    out = [legacy_header(summary=summary, metadata=metadata, cli=cli, legacy_output=legacy_output), ""]
    preamble = extract_preamble(lines)
    if preamble:
        out.extend(preamble)
    else:
        out.append("Starting Training...")

    for row in rows:
        epoch = row.get("epoch", "").strip()
        if not epoch:
            continue
        train_acc = to_float(row.get("train_acc"))
        train_loss = scaled_loss(row.get("train_loss"), loss_scale)
        test_acc = to_float(row.get("eval_acc"))
        test_loss = scaled_loss(row.get("eval_loss"), loss_scale)
        out.append(
            f"Epoch {int(epoch):<3}\t\t"
            f"Train Acc: {train_acc:.4f}\tTrain loss: {train_loss:.4f}\t"
            f"Test Acc: {test_acc:.4f}\tTest loss: {test_loss:.4f}"
        )

    if finish_line:
        out.append(finish_line)
    return "\n".join(out).rstrip() + "\n"


def artifact_paths_for_metrics(metrics_path: Path) -> tuple[Path, Path]:
    jobmeta = metrics_path.parent / "jobmeta.txt"
    return jobmeta, run_log_path(metrics_path)


def collect_metrics_from_result_root(result_root: Path) -> list[Path]:
    return sorted(result_root.rglob("metrics*.csv"))


def collect_metrics_from_spec(spec_path: str, ids_raw: str, result_root: Path) -> list[Path]:
    spec = launch.load_spec(spec_path)
    plan = launch.build_plan(spec)
    ids = parse_ids(ids_raw, plan.total)
    metrics_paths: list[Path] = []
    for task_id in ids:
        _, metrics_file, _ = launch.task_completion_artifacts(result_root, plan.tasks[task_id])
        if metrics_file.exists():
            metrics_paths.append(metrics_file)
    return metrics_paths


def cmd_missing(args: argparse.Namespace) -> int:
    playground_root = args.playground_root.expanduser().resolve()
    result_root = launch.result_root_for_platform(launch.repo_root(), "local")
    spec = launch.load_spec(args.spec)
    plan = launch.build_plan(spec)
    ids = parse_ids(args.ids, plan.total)

    missing: list[int] = []
    complete_playground = 0
    complete_local = 0
    for task_id in ids:
        task = plan.tasks[task_id]
        legacy_match = find_legacy_match(playground_root, task, match_config=args.match_config)
        if legacy_match is not None:
            complete_playground += 1
            if args.format == "table":
                print(f"{task_id}\tplayground\t{legacy_match}")
            continue
        if args.also_local and launch.is_task_complete(result_root, task):
            complete_local += 1
            if args.format == "table":
                print(f"{task_id}\tlocal\t{launch.task_output_dir(result_root, task)}")
            continue
        missing.append(task_id)
        if args.format == "table":
            print(f"{task_id}\tmissing\t{legacy_path_for_task(playground_root, task)}")

    if args.format == "ids":
        print(",".join(str(task_id) for task_id in missing))
    elif args.format == "summary":
        print(
            "total={total} playground_complete={pg} local_complete={local} missing={missing}".format(
                total=len(ids),
                pg=complete_playground,
                local=complete_local,
                missing=len(missing),
            )
        )
        if missing:
            print("missing_ids=" + ",".join(str(task_id) for task_id in missing))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    playground_root = args.playground_root.expanduser().resolve()
    metrics_paths = [path.expanduser().resolve() for path in args.metrics]
    if args.result_root:
        metrics_paths.extend(collect_metrics_from_result_root(args.result_root.expanduser().resolve()))
    if args.spec:
        result_root = args.result_root.expanduser().resolve() if args.result_root else launch.result_root_for_platform(launch.repo_root(), "local")
        metrics_paths.extend(collect_metrics_from_spec(args.spec, args.ids, result_root))
    metrics_paths = sorted(set(metrics_paths))

    exported = 0
    skipped = 0
    failed = 0
    for metrics_path in metrics_paths:
        try:
            jobmeta, run_log = artifact_paths_for_metrics(metrics_path)
            metadata = read_kv_file(jobmeta)
            cli = parse_command_options(metadata.get("command", ""))
            summary = parse_config_summary(run_log)
            legacy_output = legacy_path_from_artifacts(
                playground_root,
                metadata=metadata,
                summary=summary,
                cli=cli,
            )
            if legacy_output.exists() and not args.overwrite:
                skipped += 1
                print(f"skip exists\t{legacy_output}")
                continue
            loss_scale = parse_loss_scale(args.loss_scale, summary)
            text = render_legacy_text(
                metrics_path=metrics_path,
                run_log=run_log,
                metadata=metadata,
                summary=summary,
                cli=cli,
                legacy_output=legacy_output,
                loss_scale=loss_scale,
                allow_partial=args.allow_partial,
            )
            if args.dry_run:
                print(f"would export\t{metrics_path}\t->\t{legacy_output}")
                exported += 1
                continue
            legacy_output.parent.mkdir(parents=True, exist_ok=True)
            legacy_output.write_text(text, encoding="utf-8")
            exported += 1
            print(f"exported\t{legacy_output}")
        except Exception as exc:
            failed += 1
            print(f"failed\t{metrics_path}\t{exc}", file=sys.stderr)

    print(f"summary exported={exported} skipped={skipped} failed={failed}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reuse and export FL poisoning results in the legacy playground .txt format."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    missing = subparsers.add_parser("missing", help="List spec tasks missing from the playground result library.")
    missing.add_argument("spec", help="Experiment spec path or spec identifier.")
    missing.add_argument("--ids", default="all", help="Task ids/ranges to check; default: all.")
    missing.add_argument(
        "--playground-root",
        type=Path,
        default=DEFAULT_PLAYGROUND_ROOT,
        help="Root of Poisoning_Resilient_Federated_Learning_Playground.",
    )
    missing.add_argument(
        "--match-config",
        choices=("exact", "any"),
        default="exact",
        help="Use any cfg* legacy file as a semantic match when hyperparameters otherwise match.",
    )
    missing.add_argument(
        "--also-local",
        action="store_true",
        help="Treat completed local CSV runs as complete even if they have not been exported yet.",
    )
    missing.add_argument("--format", choices=("table", "summary", "ids"), default="summary")
    missing.set_defaults(func=cmd_missing)

    export = subparsers.add_parser("export", help="Export current CSV artifacts into playground .txt files.")
    export.add_argument("metrics", nargs="*", type=Path, help="metrics_exp*.csv files to export.")
    export.add_argument("--spec", help="Optional spec whose completed local task metrics should be exported.")
    export.add_argument("--ids", default="all", help="Task ids/ranges when --spec is used.")
    export.add_argument("--result-root", type=Path, help="Raw CSV result root; defaults to launcher local RESULT_ROOT.")
    export.add_argument(
        "--playground-root",
        type=Path,
        default=DEFAULT_PLAYGROUND_ROOT,
        help="Root of Poisoning_Resilient_Federated_Learning_Playground.",
    )
    export.add_argument("--overwrite", action="store_true", help="Overwrite existing playground .txt files.")
    export.add_argument("--dry-run", action="store_true", help="Show export targets without writing files.")
    export.add_argument(
        "--allow-partial",
        action="store_true",
        help="Export runs without a Training finished line. Use only for diagnostic partial curves.",
    )
    export.add_argument(
        "--loss-scale",
        default="auto",
        help="Divide CSV losses by this value. 'auto' uses batch_size and matches the legacy playground scale.",
    )
    export.set_defaults(func=cmd_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
