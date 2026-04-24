#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import launch
import playground_results


def run(cmd: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None) -> int:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(list(cmd), cwd=str(cwd), env=env)


def capture(cmd: Sequence[str], *, cwd: Path) -> str:
    return subprocess.check_output(list(cmd), cwd=str(cwd), text=True).strip()


def parse_cuda_list(raw: str) -> list[str]:
    out = [item.strip() for item in raw.split(",") if item.strip()]
    if not out:
        raise ValueError("--cuda must contain at least one GPU id")
    return out


def query_gpus() -> dict[str, tuple[int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    gpus: dict[str, tuple[int, int]] = {}
    for row in csv.reader(output.splitlines()):
        if len(row) < 3:
            continue
        index = row[0].strip()
        free_mb = int(row[1].strip())
        util = int(row[2].strip())
        gpus[index] = (free_mb, util)
    return gpus


def choose_gpu(
    candidates: Sequence[str],
    *,
    min_free_mb: int,
    max_util: int,
) -> str | None:
    gpus = query_gpus()
    eligible: list[tuple[int, int, str]] = []
    for gpu in candidates:
        if gpu not in gpus:
            continue
        free_mb, util = gpus[gpu]
        if free_mb < min_free_mb:
            continue
        if max_util >= 0 and util > max_util:
            continue
        eligible.append((free_mb, -util, gpu))
    if not eligible:
        return None
    eligible.sort(reverse=True)
    return eligible[0][2]


def missing_ids(
    *,
    repo: Path,
    python_bin: str,
    spec: str,
    ids: str,
    match_config: str,
    also_local: bool,
) -> list[int]:
    cmd = [
        python_bin,
        "exps/playground_results.py",
        "missing",
        spec,
        "--ids",
        ids,
        "--match-config",
        match_config,
        "--format",
        "ids",
    ]
    if also_local:
        cmd.append("--also-local")
    output = capture(cmd, cwd=repo)
    if not output:
        return []
    return [int(item) for item in output.split(",") if item.strip()]


def wait_for_gpu(args: argparse.Namespace, candidates: Sequence[str]) -> str:
    while True:
        try:
            gpu = choose_gpu(
                candidates,
                min_free_mb=args.min_free_mb,
                max_util=args.max_util,
            )
        except Exception as exc:
            print(f"GPU query failed: {exc}; retrying in {args.poll_seconds}s", flush=True)
            gpu = None
        if gpu is not None:
            return gpu
        print(
            f"No eligible GPU yet; need free>={args.min_free_mb}MB util<={args.max_util}. "
            f"Retrying in {args.poll_seconds}s.",
            flush=True,
        )
        time.sleep(args.poll_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run only missing experiments and export each completed run to the playground library."
    )
    parser.add_argument("spec", help="Experiment spec path or identifier.")
    parser.add_argument("--ids", default="all", help="Task ids/ranges to consider.")
    parser.add_argument("--cuda", default="0,1", help="Candidate physical GPU ids.")
    parser.add_argument("--min-free-mb", type=int, default=12000)
    parser.add_argument(
        "--max-util",
        type=int,
        default=40,
        help="Maximum GPU utilization before launching a task. Use -1 to ignore utilization.",
    )
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--sleep-between", type=int, default=10)
    parser.add_argument("--match-config", choices=("exact", "any"), default="exact")
    parser.add_argument(
        "--also-local",
        action="store_true",
        help="Skip tasks already complete in local CSV output even before playground export.",
    )
    args = parser.parse_args(argv)

    repo = launch.repo_root()
    python_bin = sys.executable
    candidates = parse_cuda_list(args.cuda)

    pending = missing_ids(
        repo=repo,
        python_bin=python_bin,
        spec=args.spec,
        ids=args.ids,
        match_config=args.match_config,
        also_local=args.also_local,
    )
    print(f"initial_missing={len(pending)} ids={','.join(map(str, pending))}", flush=True)

    for task_id in pending:
        remaining = missing_ids(
            repo=repo,
            python_bin=python_bin,
            spec=args.spec,
            ids=str(task_id),
            match_config=args.match_config,
            also_local=False,
        )
        if not remaining:
            print(f"task {task_id}: already present in playground; skip", flush=True)
            continue

        gpu = wait_for_gpu(args, candidates)
        print(f"task {task_id}: launching on GPU {gpu}", flush=True)
        rc = run(
            [
                python_bin,
                "exps/launch.py",
                "local",
                args.spec,
                "--ids",
                str(task_id),
                "--resume",
                "--cuda",
                gpu,
                "--jobs",
                "1",
                "--gpu-lock-dir",
                f"gpu_locks_gpu{gpu}",
            ],
            cwd=repo,
        )
        if rc != 0:
            print(f"task {task_id}: launch failed rc={rc}; stopping", flush=True)
            return rc

        rc = run(
            [
                python_bin,
                "exps/playground_results.py",
                "export",
                "--spec",
                args.spec,
                "--ids",
                str(task_id),
            ],
            cwd=repo,
        )
        if rc != 0:
            print(f"task {task_id}: export failed rc={rc}; stopping", flush=True)
            return rc
        time.sleep(args.sleep_between)

    print("incremental run complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
