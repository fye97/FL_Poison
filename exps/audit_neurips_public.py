#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPS = ROOT / "exps"
if str(EXPS) not in sys.path:
    sys.path.insert(0, str(EXPS))

import launch


DEFAULT_SPECS = (
    "exps/specs/CARAT/paper_neurips_core_public.yaml",
    "exps/specs/CARAT/paper_neurips_clean_public.yaml",
    "exps/specs/CARAT/paper_neurips_ablation_fang_alpha05.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit NeurIPS public CARAT experiment progress.")
    parser.add_argument("--result-root", default="logs/local_runs")
    parser.add_argument("--specs", nargs="*", default=list(DEFAULT_SPECS))
    parser.add_argument("--show-partial", type=int, default=12)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def mean_std(values: Iterable[float]) -> tuple[float, float, int] | None:
    items = list(values)
    if not items:
        return None
    return (
        sum(items) / len(items),
        statistics.stdev(items) if len(items) > 1 else 0.0,
        len(items),
    )


def task_key(task: launch.ExperimentTask) -> tuple[str, str, str, str, str]:
    alpha = task.dirichlet_alpha if task.distribution == "non-iid" else task.distribution
    return (alpha, task.attack, task.defense, task.config_name, task.epochs)


def fmt_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def main() -> int:
    args = parse_args()
    result_root = (ROOT / args.result_root).resolve()

    print(f"result_root={result_root}")
    all_partials: list[tuple[float, str]] = []

    for spec_path in args.specs:
        spec = launch.load_spec(spec_path)
        plan = launch.build_plan(spec)
        complete = 0
        partial = 0
        missing = 0
        failed_like = 0
        grouped_acc: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
        grouped_rows: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)

        for task in plan.tasks:
            output_dir = launch.task_output_dir(result_root, task)
            metrics_file = launch.resolve_existing_metrics_file(output_dir, task)
            rows = read_rows(metrics_file)
            is_complete = launch.is_task_complete(result_root, task)

            if is_complete:
                complete += 1
            elif rows:
                partial += 1
            else:
                missing += 1

            if rows:
                final_epoch = int(float(rows[-1].get("epoch") or 0))
                grouped_rows[task_key(task)].append(final_epoch)
                eval_acc = to_float(rows[-1].get("eval_acc"))
                if is_complete and eval_acc is not None:
                    grouped_acc[task_key(task)].append(eval_acc)
                try:
                    mtime = metrics_file.stat().st_mtime
                except OSError:
                    mtime = 0.0
                status = "complete" if is_complete else "partial"
                all_partials.append(
                    (
                        mtime,
                        f"{status} spec={spec.name} task={task.task_id} alpha={task.dirichlet_alpha or task.distribution} "
                        f"attack={task.attack} defense={task.defense} seed={task.effective_seed} "
                        f"epoch={final_epoch}/{task.epochs} eval_acc={fmt_metric(eval_acc)}",
                    )
                )
                if not is_complete and final_epoch >= int(task.epochs):
                    failed_like += 1

        print(
            f"\n[{spec.name}] total={plan.total} complete={complete} partial={partial} "
            f"missing={missing} suspicious={failed_like}"
        )
        for key in sorted(set(grouped_rows) | set(grouped_acc)):
            alpha, attack, defense, config_name, epochs = key
            row_max = max(grouped_rows.get(key, [0]))
            acc_stats = mean_std(grouped_acc.get(key, []))
            if acc_stats is None:
                acc_text = "complete_eval=NA"
            else:
                mean, std, count = acc_stats
                acc_text = f"complete_eval={mean:.4f}+-{std:.4f} n={count}"
            print(
                f"  alpha={alpha:>4} attack={attack:<10} defense={defense:<12} "
                f"rows_max={row_max:>3}/{epochs:<3} {acc_text} cfg={Path(config_name).stem}"
            )

    print("\n[recent metric files]")
    for _mtime, line in sorted(all_partials, reverse=True)[: args.show_partial]:
        print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
