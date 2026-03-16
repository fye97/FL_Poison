#!/usr/bin/env python3
"""
Benchmark the current workspace against the pre-optimization baseline tag.

Default experiment:
  python main.py --config=./configs/FedSGD_MNIST_config.yaml --attack=MinSum --defense=Mean

This script compares:
  - baseline: git tag/ref `v1.0.0`
  - current : the current workspace at repo root

It records:
  - end-to-end wall-clock time
  - optional `record_time` breakdowns emitted by the framework

This file intentionally does not start with `test_`, so pytest will not collect
it by default. Run it directly with Python when you want a performance check.

Example:
  python tests/perf/benchmark_fedsgd_mnist_minsum_mean.py
  python tests/perf/benchmark_fedsgd_mnist_minsum_mean.py --repeats 3 --warmup 1
  python tests/perf/benchmark_fedsgd_mnist_minsum_mean.py --extra-cli-args "--epochs 50 --num_clients 20"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


TIME_LOG_RE = re.compile(
    r"^(?P<subject>[^,]+), (?P<method>[A-Za-z_]+) averge time: (?P<avg>[0-9.]+) s, call time: (?P<calls>\d+)$"
)


@dataclass
class RunRecord:
    label: str
    run_index: int
    counted: bool
    repo_path: str
    git_commit: str
    config_path: str
    output_path: str
    runner_log_path: str
    wall_time_sec: float
    return_code: int
    time_breakdown: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current workspace performance against a baseline git ref."
    )
    parser.add_argument(
        "--baseline-ref",
        default="v1.0.0",
        help="Git ref used as the baseline worktree. Default: v1.0.0",
    )
    parser.add_argument(
        "--config",
        default="configs/FedSGD_MNIST_config.yaml",
        help="Repo-relative config path used in both baseline and current runs.",
    )
    parser.add_argument("--attack", default="MinSum", help="Attack override.")
    parser.add_argument("--defense", default="Mean", help="Defense override.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Measured runs per variant. Default: 1",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup runs per variant before measured runs. Default: 0",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python executable to use. Default: .venv/bin/python if present, else current Python.",
    )
    parser.add_argument(
        "--extra-cli-args",
        default="",
        help="Extra CLI overrides appended to each main.py invocation.",
    )
    parser.add_argument(
        "--no-record-time",
        action="store_true",
        help="Disable framework timing logs and only collect wall-clock time.",
    )
    parser.add_argument(
        "--no-cache-partition",
        action="store_true",
        help="Do not force cache_partition=True in the temporary benchmark config.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for benchmark artifacts. Default: logs/perf_eval/<timestamp>_fedsgd_mnist_minsum_mean",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_python_bin(root: Path, requested: str | None) -> Path:
    if requested:
        return Path(os.path.abspath(str(Path(requested).expanduser())))
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return Path(os.path.abspath(str(venv_python)))
    return Path(os.path.abspath(sys.executable))


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def ensure_worktree(root: Path, ref: str) -> tuple[Path, str]:
    commit = run_git(root, "rev-parse", f"{ref}^{{commit}}")
    safe_ref = re.sub(r"[^A-Za-z0-9_.-]+", "_", ref)
    worktree_root = Path(tempfile.gettempdir()) / "fl_poison_perf_worktrees"
    worktree_path = worktree_root / f"{safe_ref}_{commit[:7]}"

    if worktree_path.exists():
        head_commit = run_git(worktree_path, "rev-parse", "HEAD")
        if head_commit != commit:
            raise RuntimeError(
                f"Existing worktree {worktree_path} points to {head_commit}, expected {commit}. "
                "Remove it manually and rerun."
            )
        return worktree_path, commit

    worktree_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_path), commit],
        cwd=str(root),
        check=True,
    )
    return worktree_path, commit


def ensure_shared_data_dir(root: Path, worktree_path: Path) -> None:
    source_data = root / "data"
    target_data = worktree_path / "data"
    if not source_data.exists():
        return
    if target_data.is_symlink():
        return
    if target_data.exists():
        has_files = any(path.is_file() for path in target_data.rglob("*"))
        if has_files:
            return
        shutil.rmtree(target_data)
    target_data.symlink_to(source_data, target_is_directory=True)


def default_output_root(root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return root / "logs" / "perf_eval" / f"{stamp}_fedsgd_mnist_minsum_mean"


def temp_config_path(output_root: Path, label: str, run_index: int) -> Path:
    config_dir = output_root / "run_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / f"{label}_run{run_index:02d}.yaml"


def runner_log_path(output_root: Path, label: str, run_index: int) -> Path:
    log_dir = output_root / "runner_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{label}_run{run_index:02d}.stdout.log"


def experiment_output_path(output_root: Path, label: str, run_index: int) -> Path:
    return output_root / label / f"run{run_index:02d}.txt"


def time_log_path(output_path: Path) -> Path:
    return Path(str(output_path).replace("logs/", "logs/time_logs/", 1)).with_suffix(".log")


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


def build_temp_config(
    source_config: Path,
    temp_config: Path,
    output_path: Path,
    record_time: bool,
    cache_partition: bool,
) -> None:
    cfg = load_yaml(source_config)
    cfg["output"] = str(output_path)
    cfg["record_time"] = bool(record_time)
    cfg["log_stream"] = False
    cfg["num_experiments"] = 1
    cfg["experiment_id"] = 0
    if cache_partition:
        cfg["cache_partition"] = True
    write_yaml(temp_config, cfg)


def run_variant(
    *,
    label: str,
    repo_path: Path,
    git_commit: str,
    source_config: Path,
    python_bin: Path,
    attack: str,
    defense: str,
    extra_cli_args: list[str],
    output_root: Path,
    run_index: int,
    counted: bool,
    record_time: bool,
    cache_partition: bool,
) -> RunRecord:
    output_path = experiment_output_path(output_root, label, run_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = temp_config_path(output_root, label, run_index)
    build_temp_config(
        source_config=source_config,
        temp_config=config_path,
        output_path=output_path,
        record_time=record_time,
        cache_partition=cache_partition,
    )

    cmd = [
        str(python_bin),
        "main.py",
        f"--config={config_path}",
        f"--attack={attack}",
        f"--defense={defense}",
        *extra_cli_args,
    ]
    run_log = runner_log_path(output_root, label, run_index)
    started = time.perf_counter()
    with run_log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND=" + shlex.join(cmd) + "\n")
        handle.write(f"CWD={repo_path}\n\n")
        handle.flush()
        result = subprocess.run(
            cmd,
            cwd=str(repo_path),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    ended = time.perf_counter()

    breakdown = parse_time_log(time_log_path(output_path)) if record_time else {}
    return RunRecord(
        label=label,
        run_index=run_index,
        counted=counted,
        repo_path=str(repo_path),
        git_commit=git_commit,
        config_path=str(config_path),
        output_path=str(output_path),
        runner_log_path=str(run_log),
        wall_time_sec=ended - started,
        return_code=result.returncode,
        time_breakdown=breakdown,
    )


def parse_time_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    server_metrics: dict[str, float] = {}
    client_metrics: dict[str, list[float]] = {}

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = TIME_LOG_RE.match(line)
        if not match:
            continue
        subject = match.group("subject")
        method = match.group("method")
        avg = float(match.group("avg"))
        if subject.startswith("Client "):
            client_metrics.setdefault(method, []).append(avg)
        elif subject.startswith("Server"):
            server_metrics[method] = avg

    summarized_clients = {}
    for method, values in client_metrics.items():
        summarized_clients[method] = {
            "mean": statistics.mean(values),
            "max": max(values),
            "min": min(values),
            "num_clients": len(values),
        }

    return {
        "time_log_path": str(path),
        "server": server_metrics,
        "clients": summarized_clients,
    }


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def stdev_or_zero(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(records: list[RunRecord]) -> dict[str, Any]:
    grouped: dict[str, list[RunRecord]] = {}
    for record in records:
        if not record.counted:
            continue
        grouped.setdefault(record.label, []).append(record)

    summary: dict[str, Any] = {}
    for label, items in grouped.items():
        wall_times = [item.wall_time_sec for item in items]
        server_metrics: dict[str, list[float]] = {}
        client_mean_metrics: dict[str, list[float]] = {}

        for item in items:
            breakdown = item.time_breakdown or {}
            for method, value in breakdown.get("server", {}).items():
                server_metrics.setdefault(method, []).append(float(value))
            for method, stats in breakdown.get("clients", {}).items():
                client_mean_metrics.setdefault(method, []).append(float(stats["mean"]))

        summary[label] = {
            "num_runs": len(items),
            "wall_time_sec": {
                "mean": mean_or_none(wall_times),
                "stdev": stdev_or_zero(wall_times),
                "values": wall_times,
            },
            "server_time_sec": {
                method: {
                    "mean": mean_or_none(values),
                    "stdev": stdev_or_zero(values),
                }
                for method, values in sorted(server_metrics.items())
            },
            "client_mean_time_sec": {
                method: {
                    "mean": mean_or_none(values),
                    "stdev": stdev_or_zero(values),
                }
                for method, values in sorted(client_mean_metrics.items())
            },
        }

    baseline = summary.get("baseline", {})
    current = summary.get("current", {})
    base_wall = baseline.get("wall_time_sec", {}).get("mean")
    curr_wall = current.get("wall_time_sec", {}).get("mean")
    if base_wall and curr_wall:
        summary["speedup"] = {
            "baseline_over_current": base_wall / curr_wall if curr_wall > 0 else math.inf,
            "current_vs_baseline_delta_sec": curr_wall - base_wall,
            "current_vs_baseline_delta_pct": ((curr_wall - base_wall) / base_wall) * 100.0,
        }
    return summary


def build_summary_markdown(
    *,
    args: argparse.Namespace,
    baseline_commit: str,
    current_commit: str,
    output_root: Path,
    aggregate: dict[str, Any],
) -> str:
    lines = []
    lines.append("# Performance Summary")
    lines.append("")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Baseline ref: `{args.baseline_ref}` ({baseline_commit[:7]})")
    lines.append(f"- Current commit: `{current_commit[:7]}`")
    lines.append(f"- Command: `python main.py --config=./{args.config} --attack={args.attack} --defense={args.defense}`")
    if args.extra_cli_args:
        lines.append(f"- Extra CLI args: `{args.extra_cli_args}`")
    lines.append("")
    lines.append("| Variant | Runs | Mean wall time (s) | Std (s) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for label in ["baseline", "current"]:
        item = aggregate.get(label, {})
        wall = item.get("wall_time_sec", {})
        lines.append(
            f"| {label} | {item.get('num_runs', 0)} | {wall.get('mean', 0.0):.4f} | {wall.get('stdev', 0.0):.4f} |"
        )

    speedup = aggregate.get("speedup")
    if speedup:
        lines.append("")
        lines.append(
            f"- Speedup (baseline/current): `{speedup['baseline_over_current']:.4f}x`"
        )
        lines.append(
            f"- Current vs baseline delta: `{speedup['current_vs_baseline_delta_sec']:.4f}s` ({speedup['current_vs_baseline_delta_pct']:.2f}%)"
        )

    def append_breakdown(title: str, key: str) -> None:
        if key not in aggregate.get("baseline", {}) and key not in aggregate.get("current", {}):
            return
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Method | Baseline mean (s) | Current mean (s) |")
        lines.append("| --- | ---: | ---: |")
        methods = set(aggregate.get("baseline", {}).get(key, {}).keys()) | set(
            aggregate.get("current", {}).get(key, {}).keys()
        )
        for method in sorted(methods):
            baseline_mean = (
                aggregate.get("baseline", {}).get(key, {}).get(method, {}).get("mean", 0.0)
            )
            current_mean = (
                aggregate.get("current", {}).get(key, {}).get(method, {}).get("mean", 0.0)
            )
            lines.append(f"| {method} | {baseline_mean:.6f} | {current_mean:.6f} |")

    append_breakdown("Server Timing", "server_time_sec")
    append_breakdown("Client Mean Timing", "client_mean_time_sec")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    root = repo_root()
    python_bin = resolve_python_bin(root, args.python_bin)
    config_rel = Path(args.config)
    current_config = (root / config_rel).resolve()
    if not current_config.exists():
        raise FileNotFoundError(f"Config not found: {current_config}")

    baseline_root, baseline_commit = ensure_worktree(root, args.baseline_ref)
    ensure_shared_data_dir(root, baseline_root)
    baseline_config = (baseline_root / config_rel).resolve()
    if not baseline_config.exists():
        raise FileNotFoundError(f"Baseline config not found: {baseline_config}")

    current_commit = run_git(root, "rev-parse", "HEAD")
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root(root)
    output_root.mkdir(parents=True, exist_ok=True)

    extra_cli_args = shlex.split(args.extra_cli_args)
    record_time = not args.no_record_time
    cache_partition = not args.no_cache_partition
    total_runs = args.warmup + args.repeats

    records: list[RunRecord] = []
    variants = [
        ("baseline", baseline_root, baseline_commit, baseline_config),
        ("current", root, current_commit, current_config),
    ]

    for label, variant_root, variant_commit, variant_config in variants:
        for run_index in range(total_runs):
            counted = run_index >= args.warmup
            print(
                f"[{label}] run {run_index + 1}/{total_runs} "
                f"({'measured' if counted else 'warmup'})"
            )
            record = run_variant(
                label=label,
                repo_path=variant_root,
                git_commit=variant_commit,
                source_config=variant_config,
                python_bin=python_bin,
                attack=args.attack,
                defense=args.defense,
                extra_cli_args=extra_cli_args,
                output_root=output_root,
                run_index=run_index,
                counted=counted,
                record_time=record_time,
                cache_partition=cache_partition,
            )
            records.append(record)
            if record.return_code != 0:
                raise RuntimeError(
                    f"Run failed: {label} run_index={run_index}, see {record.runner_log_path}"
                )

    aggregate = aggregate_runs(records)
    summary = {
        "baseline_ref": args.baseline_ref,
        "baseline_commit": baseline_commit,
        "current_commit": current_commit,
        "config": args.config,
        "attack": args.attack,
        "defense": args.defense,
        "extra_cli_args": args.extra_cli_args,
        "record_time": record_time,
        "cache_partition": cache_partition,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "python_bin": str(python_bin),
        "output_root": str(output_root),
        "runs": [asdict(record) for record in records],
        "aggregate": aggregate,
    }

    summary_json = output_root / "summary.json"
    summary_md = output_root / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(
        build_summary_markdown(
            args=args,
            baseline_commit=baseline_commit,
            current_commit=current_commit,
            output_root=output_root,
            aggregate=aggregate,
        ),
        encoding="utf-8",
    )

    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    if "speedup" in aggregate:
        speedup = aggregate["speedup"]["baseline_over_current"]
        delta_pct = aggregate["speedup"]["current_vs_baseline_delta_pct"]
        print(f"Speedup (baseline/current): {speedup:.4f}x")
        print(f"Current vs baseline delta: {delta_pct:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
