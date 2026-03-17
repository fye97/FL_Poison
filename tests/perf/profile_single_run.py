#!/usr/bin/env python3
"""
Run a single fixed-configuration profiling experiment and emit a concise baseline summary.

Example:
  python tests/perf/profile_single_run.py \
    --config configs/FedSGD_MNIST_Lenet.yaml \
    --defense Mean \
    --epochs 20 \
    --num-clients 10 \
    --batch-size 64 \
    --local-epochs 1 \
    --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import resolve_config_path
from performance_utils import perf_summary_path


def repo_root() -> Path:
    return ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single profiling experiment with a fixed baseline configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Repo-relative YAML config used as the baseline experiment template.",
    )
    parser.add_argument("--attack", default=None, help="Optional attack override.")
    parser.add_argument("--defense", default=None, help="Optional defense override.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument("--algorithm", default=None, help="Optional algorithm override.")
    parser.add_argument("--distribution", default=None, help="Optional distribution override.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional global-round override.")
    parser.add_argument("--num-clients", type=int, default=None, help="Optional client-count override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional train batch-size override.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Optional eval batch-size override.")
    parser.add_argument("--local-epochs", type=int, default=None, help="Optional local-epoch override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python executable to use. Default: .venv/bin/python if present, else current Python.",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Enable torch.profiler in addition to runtime timing.",
    )
    parser.add_argument(
        "--gpu-sample-interval-ms",
        type=int,
        default=100,
        help="GPU sampling interval used by runtime timing.",
    )
    parser.add_argument(
        "--extra-cli-args",
        default="",
        help="Extra CLI args appended to the main.py invocation.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for artifacts. Default: logs/perf_baseline/<timestamp>_<config-name>",
    )
    return parser.parse_args()


def resolve_python_bin(root: Path, requested: str | None) -> Path:
    if requested:
        return Path(os.path.abspath(str(Path(requested).expanduser())))
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return Path(os.path.abspath(str(venv_python)))
    return Path(os.path.abspath(sys.executable))


def default_output_root(root: Path, config_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return root / "logs" / "perf_baseline" / f"{stamp}_{config_path.stem}"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def build_profile_config(args: argparse.Namespace, source_config: Path, output_root: Path) -> tuple[Path, Path]:
    cfg = load_yaml(source_config)
    output_path = output_root / "single_run_exp0.txt"
    profile_config = output_root / "profile_config.yaml"

    overrides = {
        "model": args.model,
        "algorithm": args.algorithm,
        "distribution": args.distribution,
        "epochs": args.epochs,
        "num_clients": args.num_clients,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "local_epochs": args.local_epochs,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    cfg["output"] = str(output_path)
    cfg["record_time"] = True
    cfg["torch_profile"] = bool(args.torch_profile)
    cfg["gpu_sample_interval_ms"] = int(args.gpu_sample_interval_ms)
    cfg["cache_partition"] = True
    cfg["log_stream"] = False
    cfg["num_experiments"] = 1
    cfg["experiment_id"] = 0

    write_yaml(profile_config, cfg)
    return profile_config, output_path


def load_perf_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_perf_json(expected_output_path: Path) -> Path:
    expected_perf_json = perf_summary_path(expected_output_path)
    if expected_perf_json.exists():
        return expected_perf_json

    candidates = sorted(expected_perf_json.parent.glob(f"{expected_output_path.stem.split('_exp')[0]}*.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(expected_perf_json)


def print_summary(summary: dict[str, Any], perf_json: Path) -> None:
    metadata = summary.get("metadata", {})
    overall = summary.get("overall", {})
    final_test = overall.get("final_test_metrics", {})
    print("Baseline summary")
    print(
        f"config: model={metadata.get('model')} batch_size={metadata.get('batch_size')} "
        f"eval_batch_size={metadata.get('eval_batch_size')} "
        f"clients={metadata.get('num_clients')} local_epochs={metadata.get('local_epochs')} "
        f"distribution={metadata.get('distribution')} defense={metadata.get('defense')} "
        f"seed={metadata.get('seed')}"
    )
    print(
        f"sec/round={overall.get('sec_per_round')} "
        f"rounds/sec={overall.get('rounds_per_sec')} "
        f"sec/client={overall.get('sec_per_client')}"
    )
    print(
        f"gpu_util={overall.get('gpu_utilization_pct_avg')} "
        f"gpu_compute_ratio={overall.get('gpu_compute_ratio_avg')} "
        f"gpu_mem_peak_mb={overall.get('gpu_memory_peak_allocated_mb')}"
    )
    print(
        f"train_acc={overall.get('final_train_accuracy')} "
        f"val_acc={final_test.get('Test Acc')}"
    )
    print(f"perf_json={perf_json}")


def main() -> int:
    args = parse_args()
    root = repo_root()
    source_config = resolve_config_path(args.config, root=root)

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_output_root(root, source_config)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    python_bin = resolve_python_bin(root, args.python_bin)
    profile_config, output_path = build_profile_config(args, source_config, output_root)

    cmd = [str(python_bin), "main.py", f"--config={profile_config}"]
    if args.attack:
        cmd.append(f"--attack={args.attack}")
    if args.defense:
        cmd.append(f"--defense={args.defense}")
    cmd.extend(shlex.split(args.extra_cli_args))

    stdout_log = output_root / "runner.stdout.log"
    started = time.perf_counter()
    with stdout_log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND=" + shlex.join(cmd) + "\n")
        handle.write(f"CWD={root}\n\n")
        handle.flush()
        result = subprocess.run(
            cmd,
            cwd=str(root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wall_time = time.perf_counter() - started
    if result.returncode != 0:
        print(f"Profiling run failed. See {stdout_log}", file=sys.stderr)
        return result.returncode

    try:
        perf_json = resolve_perf_json(output_path)
    except FileNotFoundError as exc:
        print(f"Expected perf summary missing: {exc}", file=sys.stderr)
        return 1

    summary = load_perf_summary(perf_json)
    print_summary(summary, perf_json)
    print(f"wall_time_sec={wall_time}")
    print(f"runner_log={stdout_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
