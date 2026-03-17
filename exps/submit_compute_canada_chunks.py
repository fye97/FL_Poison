#!/usr/bin/env python3
"""
Submit a shared experiment wrapper to Compute Canada in smaller Slurm array chunks.

Examples:
  python exps/submit_compute_canada_chunks.py exps/flpoison_CIFAR10.sh
  python exps/submit_compute_canada_chunks.py exps/flpoison_CIFAR10.sh --chunk-size 24 --dry-run
  ATTACKS_CSV=NoAttack DEFENSES_CSV=Mean python exps/submit_compute_canada_chunks.py exps/flpoison_CIFAR10.sh
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


TOTAL_RE = re.compile(r"^TOTAL combinations:\s*(\d+)\s*$")
ARRAY_HINT_RE = re.compile(r"--array=\d+-\d+%(\d+)")
SUBMIT_DEFAULTS_RE = re.compile(r"^SUBMIT_DEFAULTS\s+(.*)$")


def _repo_root_from_script(script_path: Path) -> Path:
    p = script_path.resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "main.py").exists() and (parent / "configs").is_dir():
            return parent
    return script_path.resolve().parent.parent


def _parse_submit_defaults(raw: str) -> dict:
    out = {}
    if not raw.strip():
        return out
    for token in shlex.split(raw):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value
    return out


def _parse_script_info(script_path: Path, cwd: Path, env: dict) -> Tuple[int, Optional[int], dict, str]:
    p = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = p.stdout or ""

    total: Optional[int] = None
    default_parallel: Optional[int] = None
    submit_defaults: dict = {}
    for line in out.splitlines():
        m_total = TOTAL_RE.match(line.strip())
        if m_total:
            total = int(m_total.group(1))
        if default_parallel is None:
            m_parallel = ARRAY_HINT_RE.search(line)
            if m_parallel:
                default_parallel = int(m_parallel.group(1))
        m_defaults = SUBMIT_DEFAULTS_RE.match(line.strip())
        if m_defaults:
            submit_defaults = _parse_submit_defaults(m_defaults.group(1))

    if total is None:
        raise RuntimeError(
            "Failed to parse TOTAL combinations from script output.\n"
            f"Script: {script_path}\n"
            "Expected a line like: 'TOTAL combinations: <N>'\n"
            "----- script output (first 2000 chars) -----\n"
            + out[:2000]
        )

    return total, default_parallel, submit_defaults, out


def _chunk_ranges(start: int, end_inclusive: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    cur = start
    while cur <= end_inclusive:
        chunk_end = min(cur + chunk_size - 1, end_inclusive)
        yield cur, chunk_end
        cur = chunk_end + 1


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", type=str, help="Path to a shared experiment wrapper such as exps/flpoison_CIFAR10.sh")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="How many task ids to include per sbatch submission (default: 32).",
    )
    ap.add_argument(
        "--array-parallel",
        type=int,
        default=None,
        help="Override the %%K part of --array=start-end%%K. Default: infer from script output.",
    )
    ap.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="First task id to submit (default: 0).",
    )
    ap.add_argument(
        "--end-id",
        type=int,
        default=None,
        help="Last task id to submit, inclusive (default: total-1).",
    )
    ap.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Extra argument to append before the script path, e.g. --sbatch-arg=--job-name=myjob",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting them.",
    )
    args = ap.parse_args(list(argv))

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2

    if args.chunk_size < 1:
        print("ERROR: --chunk-size must be >= 1", file=sys.stderr)
        return 2
    if args.start_id < 0:
        print("ERROR: --start-id must be >= 0", file=sys.stderr)
        return 2
    if args.array_parallel is not None and args.array_parallel < 1:
        print("ERROR: --array-parallel must be >= 1", file=sys.stderr)
        return 2

    repo_root = _repo_root_from_script(script_path)
    env = os.environ.copy()
    env.pop("SLURM_ARRAY_TASK_ID", None)

    total, inferred_parallel, submit_defaults, _script_output = _parse_script_info(script_path, repo_root, env)
    end_id = total - 1 if args.end_id is None else args.end_id
    if end_id < args.start_id:
        print("ERROR: --end-id must be >= --start-id", file=sys.stderr)
        return 2
    if end_id >= total:
        print(f"ERROR: --end-id must be <= {total - 1}", file=sys.stderr)
        return 2

    array_parallel = args.array_parallel if args.array_parallel is not None else inferred_parallel
    if array_parallel is None:
        array_parallel = 1

    chunks = list(_chunk_ranges(args.start_id, end_id, args.chunk_size))
    print(f"SCRIPT={script_path}")
    print(f"TOTAL={total}")
    print(f"SUBMIT_RANGE={args.start_id}-{end_id}")
    print(f"CHUNK_SIZE={args.chunk_size}")
    print(f"ARRAY_PARALLEL={array_parallel}")
    print(f"CHUNKS={len(chunks)}")

    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        cmd: List[str] = ["sbatch", f"--array={chunk_start}-{chunk_end}%{array_parallel}"]

        option_map = [
            ("account", "--account"),
            ("time", "--time"),
            ("gpus", "--gpus"),
            ("cpus_per_task", "--cpus-per-task"),
            ("mem", "--mem"),
            ("output", "--output"),
            ("error", "--error"),
            ("mail_user", "--mail-user"),
            ("mail_type", "--mail-type"),
        ]
        for key, flag in option_map:
            value = submit_defaults.get(key, "")
            if value:
                cmd.append(f"{flag}={value}")
        if submit_defaults.get("requeue", "1") == "1":
            cmd.append("--requeue")

        cmd.extend(args.sbatch_arg)
        cmd.append(str(script_path))

        print(f"[{idx}/{len(chunks)}] {_format_cmd(cmd)}")
        if args.dry_run:
            continue

        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print((proc.stdout or "").rstrip())
        if proc.returncode != 0:
            return int(proc.returncode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
