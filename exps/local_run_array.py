#!/usr/bin/env python3
"""
Run a Slurm-array style bash script locally by iterating SLURM_ARRAY_TASK_ID.

This is intended to mimic workflows like:
  sbatch --array=0-(TOTAL-1)%K exps/compute_canada_flpoison_*.sh

Example:
  python exps/local_run_array.py exps/compute_canada_flpoison_CIFAR10.sh
  python exps/local_run_array.py exps/compute_canada_flpoison_CIFAR10.sh --ids 0-99
  python exps/local_run_array.py exps/compute_canada_flpoison_CIFAR10.sh --resume
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


TOTAL_RE = re.compile(r"^TOTAL combinations:\s*(\d+)\s*$")


@dataclass(frozen=True)
class IdSpec:
    start: int
    end_inclusive: int

    def to_ids(self) -> List[int]:
        if self.end_inclusive < self.start:
            return []
        return list(range(self.start, self.end_inclusive + 1))


def _repo_root_from_script(script_path: Path) -> Path:
    # These scripts assume they are run from repo root (they cd there themselves),
    # but we run them with cwd at repo root for predictable relative paths.
    # exps/<script>.sh -> repo_root
    p = script_path.resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "main.py").exists() and (parent / "configs").is_dir():
            return parent
    # fallback: script's parent of exps
    return script_path.resolve().parent.parent


def _parse_total_from_script(script_path: Path, cwd: Path, env: dict) -> int:
    # Running without SLURM_ARRAY_TASK_ID makes the script print TOTAL and exit 0.
    p = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = p.stdout or ""
    for line in out.splitlines():
        m = TOTAL_RE.match(line.strip())
        if m:
            return int(m.group(1))
    # If the script didn't print TOTAL, show output for debugging.
    raise RuntimeError(
        "Failed to parse TOTAL combinations from script output.\n"
        f"Script: {script_path}\n"
        "Expected a line like: 'TOTAL combinations: <N>'\n"
        "----- script output (first 2000 chars) -----\n"
        + out[:2000]
    )


def _parse_ids(s: str, total: Optional[int]) -> List[int]:
    """
    Supported forms:
      - "0-299"
      - "0:299" (same as 0-299)
      - "0,3,9-12"
      - "all" (requires total)
    """
    s = s.strip()
    if s.lower() == "all":
        if total is None:
            raise ValueError("'all' requires total")
        return list(range(total))

    ids: List[int] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        if "-" in part or ":" in part:
            sep = "-" if "-" in part else ":"
            a, b = part.split(sep, 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                raise ValueError(f"Invalid range: {part}")
            ids.extend(range(a_i, b_i + 1))
        else:
            ids.append(int(part))
    # stable unique
    seen = set()
    out: List[int] = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _log_tail_has_success(log_path: Path) -> bool:
    try:
        # Read a small tail.
        data = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False
    lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
    if not lines:
        return False
    return lines[-1] == "LOCAL_ARRAY_EXIT_CODE=0"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _format_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _run_one(
    *,
    script_path: Path,
    cwd: Path,
    env: dict,
    task_id: int,
    log_path: Path,
    dry_run: bool,
) -> int:
    _ensure_dir(log_path.parent)
    started = time.time()

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"LOCAL_ARRAY_SCRIPT={script_path}\n")
        f.write(f"LOCAL_ARRAY_TASK_ID={task_id}\n")
        f.write(f"LOCAL_ARRAY_STARTED_AT={_format_ts(started)}\n")
        f.write(f"LOCAL_ARRAY_CWD={cwd}\n")
        f.write(f"LOCAL_ARRAY_CMD=bash {shlex.quote(str(script_path))}\n")
        f.write(f"LOCAL_ARRAY_CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
        f.write("\n")
        f.flush()

        if dry_run:
            f.write("LOCAL_ARRAY_DRY_RUN=1\n")
            f.write("LOCAL_ARRAY_EXIT_CODE=0\n")
            return 0

        p = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = p.wait()

        ended = time.time()
        f.write("\n")
        f.write(f"LOCAL_ARRAY_ENDED_AT={_format_ts(ended)}\n")
        f.write(f"LOCAL_ARRAY_DURATION_SEC={ended - started:.1f}\n")
        f.write(f"LOCAL_ARRAY_EXIT_CODE={rc}\n")
        return int(rc)


class _TokenLock:
    """
    N-token lock implemented via POSIX flock on N separate files.
    This works in restricted environments where multiprocessing SemLock is denied.
    """

    def __init__(self, lock_dir: Path, tokens: int, poll_sec: float):
        if tokens < 1:
            raise ValueError("tokens must be >= 1")
        if poll_sec <= 0:
            raise ValueError("poll_sec must be > 0")
        self.lock_dir = lock_dir
        self.tokens = int(tokens)
        self.poll_sec = float(poll_sec)

        try:
            import fcntl  # noqa: F401 (POSIX)
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Token locks require POSIX flock support") from e

    def acquire(self):  # noqa: ANN001
        import fcntl

        _ensure_dir(self.lock_dir)

        # Pre-create lock files to avoid races on first use.
        lock_files = [self.lock_dir / f"gpu_token_{i}.lock" for i in range(self.tokens)]
        for p in lock_files:
            if not p.exists():
                p.write_text("", encoding="utf-8")

        fh = None
        acquired_path = None
        while True:
            for p in lock_files:
                try:
                    f = p.open("r+", encoding="utf-8")
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fh = f
                        acquired_path = p
                        break
                    except BlockingIOError:
                        f.close()
                except FileNotFoundError:
                    # If the directory was recreated concurrently, retry.
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


def _worker_run_task(
    script_path_s: str,
    repo_root_s: str,
    base_env: dict,
    task_id: int,
    log_dir_s: str,
    dry_run: bool,
    token_lock_dir: Optional[str],
    token_lock_count: int,
    token_lock_poll: float,
) -> Tuple[int, int, str]:
    script_path = Path(script_path_s)
    repo_root = Path(repo_root_s)
    log_dir = Path(log_dir_s)
    log_path = log_dir / f"{script_path.name}_task{task_id}.out"

    env = dict(base_env)
    env["SLURM_ARRAY_TASK_ID"] = str(task_id)

    try:
        if token_lock_dir:
            tl = _TokenLock(Path(token_lock_dir), int(token_lock_count), float(token_lock_poll))
            ctx = tl.acquire()
        else:
            ctx = None

        if ctx is None:
            rc = _run_one(
                script_path=script_path,
                cwd=repo_root,
                env=env,
                task_id=task_id,
                log_path=log_path,
                dry_run=bool(dry_run),
            )
        else:
            with ctx:
                rc = _run_one(
                    script_path=script_path,
                    cwd=repo_root,
                    env=env,
                    task_id=task_id,
                    log_path=log_path,
                    dry_run=bool(dry_run),
                )
    finally:
        pass

    return (task_id, int(rc), str(log_path))


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", type=str, help="Path to exps/compute_canada_flpoison_*.sh")
    ap.add_argument(
        "--ids",
        type=str,
        default="all",
        help='Task ids to run, e.g. "0-299", "0,3,10-20", or "all" (default: all).',
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default="logs/local_array",
        help="Directory (relative to repo root) to write per-task logs.",
    )
    ap.add_argument(
        "--cuda",
        type=str,
        default="0",
        help='CUDA_VISIBLE_DEVICES to use (default: "0"). For 1x4090, keep this as 0.',
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Max number of worker processes to launch (default: 1).",
    )
    ap.add_argument(
        "--gpu-tokens",
        type=int,
        default=1,
        help="Max number of concurrent tasks allowed to run (default: 1). Use >1 only if you have >1 GPUs/MIG slices.",
    )
    ap.add_argument(
        "--gpu-lock-dir",
        type=str,
        default="gpu_locks",
        help="Directory to store GPU token lock files (relative to --log-dir by default).",
    )
    ap.add_argument(
        "--gpu-lock-poll",
        type=float,
        default=1.0,
        help="Polling interval (seconds) when waiting for a GPU token (default: 1.0).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose log tail indicates success (LOCAL_ARRAY_EXIT_CODE=0).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate log headers, do not execute.",
    )
    ap.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop at the first failing task (non-zero exit code).",
    )
    args = ap.parse_args(list(argv))

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2

    if args.jobs < 1:
        print("ERROR: --jobs must be >= 1", file=sys.stderr)
        return 2
    if args.gpu_tokens < 1:
        print("ERROR: --gpu-tokens must be >= 1", file=sys.stderr)
        return 2
    if args.gpu_lock_poll <= 0:
        print("ERROR: --gpu-lock-poll must be > 0", file=sys.stderr)
        return 2

    repo_root = _repo_root_from_script(script_path)
    log_dir = (repo_root / args.log_dir).resolve()

    base_env = os.environ.copy()
    # Ensure script prints TOTAL when SLURM_ARRAY_TASK_ID is missing.
    base_env.pop("SLURM_ARRAY_TASK_ID", None)
    base_env["CUDA_VISIBLE_DEVICES"] = args.cuda

    total = _parse_total_from_script(script_path, cwd=repo_root, env=base_env)
    ids = _parse_ids(args.ids, total=total)

    # bounds check
    bad = [i for i in ids if i < 0 or i >= total]
    if bad:
        print(f"ERROR: task ids out of range 0..{total-1}: {bad[:20]}", file=sys.stderr)
        return 2

    # Apply resume filter before scheduling.
    if args.resume:
        filtered: List[int] = []
        for tid in ids:
            log_path = log_dir / f"{script_path.name}_task{tid}.out"
            if _log_tail_has_success(log_path):
                continue
            filtered.append(tid)
        ids = filtered

    print(f"TOTAL={total}")
    print(f"RUN_IDS={len(ids)} (first={ids[0] if ids else 'n/a'}, last={ids[-1] if ids else 'n/a'})")
    print(f"LOG_DIR={log_dir}")
    print(f"CUDA_VISIBLE_DEVICES={args.cuda}")
    print(f"JOBS={args.jobs} GPU_TOKENS={args.gpu_tokens}")

    if not ids:
        return 0

    failures: List[Tuple[int, int]] = []

    # Token lock directory: absolute path under LOG_DIR by default.
    token_lock_dir = str((log_dir / args.gpu_lock_dir).resolve())
    print(f"GPU_LOCK_DIR={token_lock_dir} GPU_LOCK_POLL={args.gpu_lock_poll}")

    # Run either sequentially or via a process pool.
    if int(args.jobs) == 1:
        for k, task_id in enumerate(ids, start=1):
            log_path = log_dir / f"{script_path.name}_task{task_id}.out"
            print(f"[{k}/{len(ids)}] run task={task_id} -> {log_path}")
            tid, rc, _lp = _worker_run_task(
                str(script_path),
                str(repo_root),
                base_env,
                int(task_id),
                str(log_dir),
                bool(args.dry_run),
                token_lock_dir,
                int(args.gpu_tokens),
                float(args.gpu_lock_poll),
            )
            if rc != 0:
                failures.append((tid, rc))
                print(f"  FAIL task={tid} rc={rc}", file=sys.stderr)
                if args.stop_on_fail:
                    break
    else:
        # Multiprocessing primitives are often blocked in restricted environments.
        # Use threads to orchestrate multiple *subprocess* runs concurrently.
        task_q: "queue.Queue[int]" = queue.Queue()
        res_q: "queue.Queue[Tuple[int, int, str]]" = queue.Queue()
        stop_evt = threading.Event()

        for tid in ids:
            task_q.put(int(tid))

        def _thread_worker(worker_id: int) -> None:
            while not stop_evt.is_set():
                try:
                    tid = task_q.get_nowait()
                except queue.Empty:
                    return
                try:
                    task_id, rc, lp = _worker_run_task(
                        str(script_path),
                        str(repo_root),
                        base_env,
                        int(tid),
                        str(log_dir),
                        bool(args.dry_run),
                        token_lock_dir,
                        int(args.gpu_tokens),
                        float(args.gpu_lock_poll),
                    )
                    res_q.put((task_id, rc, lp))
                    if rc != 0 and args.stop_on_fail:
                        stop_evt.set()
                finally:
                    task_q.task_done()

        threads = [
            threading.Thread(target=_thread_worker, args=(i,), daemon=True)
            for i in range(int(args.jobs))
        ]
        for t in threads:
            t.start()

        done = 0
        total_n = len(ids)
        while done < total_n:
            try:
                tid, rc, lp = res_q.get(timeout=0.5)
            except queue.Empty:
                # If we are failing fast, stop when workers have no more output.
                if stop_evt.is_set() and task_q.unfinished_tasks == 0:
                    break
                continue
            done += 1
            print(f"[{done}/{total_n}] done task={tid} rc={rc} -> {lp}")
            if rc != 0:
                failures.append((tid, rc))
                print(f"  FAIL task={tid} rc={rc}", file=sys.stderr)

        stop_evt.set()
        for t in threads:
            t.join(timeout=1.0)

    if failures:
        print("FAILED_TASKS:", ", ".join(f"{tid}:{rc}" for tid, rc in failures), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
