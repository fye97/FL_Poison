#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


ATTACKS = ("ALIE", "FangAttack", "MinMax", "MinSum")
ALPHAS = ("1", "0.5")
DEFENSES = (
    "Mean",
    "NormClipping",
    "TrimmedMean",
    "MultiKrum",
    "FLTrust",
    "FLDetector",
    "CARAT",
)
MAIN_TEXT_DEFENSES = (
    "Mean",
    "NormClipping",
    "MultiKrum",
    "FLTrust",
    "FLDetector",
    "CARAT",
)

LEGACY_DEFENSES = {"Mean", "NormClipping", "MultiKrum", "FLDetector"}
FINAL_PROTOCOL_DEFENSES = {"TrimmedMean", "FLTrust", "CARAT"}
LEGACY_CFG = "FedAvg_CIFAR100_config"
FINAL_CFG = "FedAvg_CIFAR100_Resnet18_protocol"

EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+.*?Test Acc:\s*([0-9.]+)\s*Test loss:\s*([0-9.]+)"
)
NAME_RE = re.compile(
    r"_seed(?P<seed>\d+)_alpha(?P<alpha>[^_]+)_cfg(?P<cfg>.+)_exp(?P<exp>\d+)\.txt$"
)


@dataclass(frozen=True)
class RunResult:
    alpha: str
    attack: str
    defense: str
    source: str
    cfg: str
    seed: int
    exp: int
    epoch: int
    test_acc: float
    test_loss: float
    path: Path


def normalize_alpha(alpha: str) -> str:
    return "1" if alpha in {"1", "1.0"} else alpha


def alpha_tokens(alpha: str) -> tuple[str, ...]:
    return ("1", "1.0") if alpha == "1" else (alpha,)


def parse_last_epoch(path: Path) -> tuple[int, float, float] | None:
    last: tuple[int, float, float] | None = None
    with path.open(errors="ignore") as handle:
        for line in handle:
            match = EPOCH_RE.match(line)
            if not match:
                continue
            last = (int(match.group(1)), float(match.group(2)), float(match.group(3)))
    return last


def parse_seed_exp(path: Path) -> tuple[int, int]:
    match = NAME_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse seed/exp from {path.name}")
    return int(match.group("seed")), int(match.group("exp"))


def collect_runs(playground_root: Path, expected_rounds: int) -> list[RunResult]:
    result_dir = playground_root / "FedAvg" / "CIFAR100_resnet18" / "non-iid"
    rows: list[RunResult] = []

    for alpha in ALPHAS:
        for attack in ATTACKS:
            for defense in DEFENSES:
                if defense in LEGACY_DEFENSES:
                    cfg = LEGACY_CFG
                    source = "legacy_public_baseline"
                elif defense in FINAL_PROTOCOL_DEFENSES:
                    cfg = FINAL_CFG
                    source = "final_locked_protocol"
                else:
                    raise AssertionError(f"unhandled defense: {defense}")

                paths: set[Path] = set()
                for token in alpha_tokens(alpha):
                    pattern = (
                        f"CIFAR100_resnet18_non-iid_{attack}_{defense}_"
                        f"{expected_rounds}_20_0.05_FedAvg_adv0.2_seed*_"
                        f"alpha{token}_cfg{cfg}_exp*.txt"
                    )
                    paths.update(result_dir.glob(pattern))

                for path in sorted(paths):
                    parsed = parse_last_epoch(path)
                    if parsed is None:
                        continue
                    epoch, test_acc, test_loss = parsed
                    if epoch != expected_rounds - 1:
                        continue
                    seed, exp = parse_seed_exp(path)
                    rows.append(
                        RunResult(
                            alpha=alpha,
                            attack=attack,
                            defense=defense,
                            source=source,
                            cfg=cfg,
                            seed=seed,
                            exp=exp,
                            epoch=epoch,
                            test_acc=test_acc,
                            test_loss=test_loss,
                            path=path,
                        )
                    )
    return rows


def grouped_stats(rows: list[RunResult], metric: str) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        value = row.test_acc if metric == "accuracy" else row.test_loss
        grouped.setdefault((row.alpha, row.defense, row.attack), []).append(value)

    stats: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        stats[key] = {
            "n": float(len(values)),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return stats


def average_ranks(
    stats: dict[tuple[str, str, str], dict[str, float]],
    higher_is_better: bool,
    defenses: tuple[str, ...] = DEFENSES,
) -> dict[tuple[str, str], float]:
    rank_values: dict[tuple[str, str], list[int]] = {}
    for alpha in ALPHAS:
        for attack in ATTACKS:
            candidates: list[tuple[str, float]] = []
            for defense in defenses:
                item = stats.get((alpha, defense, attack))
                if item is not None:
                    candidates.append((defense, item["mean"]))
            candidates.sort(key=lambda item: item[1], reverse=higher_is_better)
            for rank, (defense, _) in enumerate(candidates, start=1):
                rank_values.setdefault((alpha, defense), []).append(rank)

    return {
        key: statistics.mean(values)
        for key, values in rank_values.items()
        if len(values) == len(ATTACKS)
    }


def write_detail_csv(rows: list[RunResult], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha",
                "attack",
                "defense",
                "source",
                "cfg",
                "seed",
                "exp",
                "epoch",
                "test_acc",
                "test_acc_pct",
                "test_loss",
                "path",
            ]
        )
        for row in sorted(rows, key=lambda r: (r.alpha != "1", r.attack, r.defense, r.seed)):
            writer.writerow(
                [
                    row.alpha,
                    row.attack,
                    row.defense,
                    row.source,
                    row.cfg,
                    row.seed,
                    row.exp,
                    row.epoch,
                    f"{row.test_acc:.6f}",
                    f"{100.0 * row.test_acc:.4f}",
                    f"{row.test_loss:.6f}",
                    str(row.path),
                ]
            )


def write_summary_csv(
    output: Path,
    stats: dict[tuple[str, str, str], dict[str, float]],
    ranks: dict[tuple[str, str], float],
    metric: str,
    defenses: tuple[str, ...] = DEFENSES,
) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha", "defense", "attack", "n", f"{metric}_mean", f"{metric}_std", "avg_rank"])
        for alpha in ALPHAS:
            for defense in defenses:
                for attack in ATTACKS:
                    item = stats.get((alpha, defense, attack))
                    if item is None:
                        continue
                    writer.writerow(
                        [
                            alpha,
                            defense,
                            attack,
                            int(item["n"]),
                            f"{item['mean']:.8f}",
                            f"{item['std']:.8f}",
                            f"{ranks[(alpha, defense)]:.4f}",
                        ]
                    )


def latex_defense_name(defense: str) -> str:
    if defense == "FLTrust":
        return "FLTrust (matched)"
    if defense == "CARAT":
        return r"\textbf{CARAT ($T=8$)}"
    return defense


def latex_metric_cell(value: float, std: float, metric: str, bold: bool) -> str:
    if metric == "accuracy":
        body = f"{100.0 * value:.2f} \\pm {100.0 * std:.2f}"
    else:
        body = f"{value:.4f} \\pm {std:.4f}"
    return rf"\textbf{{${body}$}}" if bold else rf"${body}$"


def best_by_cell(
    stats: dict[tuple[str, str, str], dict[str, float]],
    higher_is_better: bool,
    defenses: tuple[str, ...] = DEFENSES,
) -> dict[tuple[str, str], str]:
    best: dict[tuple[str, str], str] = {}
    for alpha in ALPHAS:
        for attack in ATTACKS:
            candidates = [
                (defense, stats[(alpha, defense, attack)]["mean"])
                for defense in defenses
                if (alpha, defense, attack) in stats
            ]
            candidates.sort(key=lambda item: item[1], reverse=higher_is_better)
            best[(alpha, attack)] = candidates[0][0]
    return best


def latex_table(
    stats: dict[tuple[str, str, str], dict[str, float]],
    ranks: dict[tuple[str, str], float],
    metric: str,
    defenses: tuple[str, ...] = DEFENSES,
    label: str | None = None,
    caption_extra: str = "",
) -> str:
    higher_is_better = metric == "accuracy"
    best_cells = best_by_cell(stats, higher_is_better, defenses)
    best_rank = {
        alpha: min(ranks[(alpha, defense)] for defense in defenses if (alpha, defense) in ranks)
        for alpha in ALPHAS
    }

    caption_metric = "test accuracy (\\%)" if metric == "accuracy" else "test loss"
    direction = "Higher is better" if metric == "accuracy" else "Lower is better"
    if label is None:
        label = "tab:cifar100-main-acc" if metric == "accuracy" else "tab:cifar100-main-loss"
    seed_note = (
        "Legacy public-baseline rows reuse five-seed logs where available; "
        "TrimmedMean, FLTrust, and CARAT are regenerated under the locked "
        "public-baseline protocol with three seeds and matched CARAT/FLTrust "
        "reference budgets."
    )
    if caption_extra:
        seed_note = f"{seed_note} {caption_extra}"

    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        rf"\caption{{Final {caption_metric} under four model-poisoning attacks on CIFAR-100/ResNet18 with $20\%$ malicious clients and non-i.i.d.\ client partitions. {direction}. {seed_note}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Dirichlet $\alpha$} & \textbf{Defense} & \textbf{ALIE} & \textbf{FangAttack} & \textbf{MinMax} & \textbf{MinSum} & \textbf{Avg. rank} \\",
        r"\midrule",
    ]

    for alpha_index, alpha in enumerate(ALPHAS):
        if alpha_index:
            lines.append(r"\midrule")
        display_alpha = "1.0" if alpha == "1" else alpha
        for defense in defenses:
            cells: list[str] = []
            for attack in ATTACKS:
                item = stats.get((alpha, defense, attack))
                if item is None:
                    cells.append(r"\multicolumn{1}{c}{--}")
                    continue
                cells.append(
                    latex_metric_cell(
                        item["mean"],
                        item["std"],
                        metric,
                        bold=(best_cells[(alpha, attack)] == defense),
                    )
                )
            rank = ranks.get((alpha, defense))
            if rank is None:
                rank_cell = r"\multicolumn{1}{c}{--}"
            elif abs(rank - best_rank[alpha]) < 1e-9:
                rank_cell = rf"\textbf{{${rank:.1f}$}}"
            else:
                rank_cell = rf"${rank:.1f}$"
            lines.append(
                rf"\textbf{{${display_alpha}$}} & {latex_defense_name(defense)} & "
                + " & ".join(cells)
                + rf" & {rank_cell} \\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_main_accuracy_table(
    stats: dict[tuple[str, str, str], dict[str, float]],
    ranks: dict[tuple[str, str], float],
) -> str:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Final test accuracy (\%) for the main-text public-baseline subset under four model-poisoning attacks on CIFAR-100/ResNet18 with $20\%$ malicious clients and non-i.i.d.\ client partitions. Higher is better. Legacy public-baseline rows reuse five-seed logs where available; FLTrust and CARAT are regenerated under the locked protocol with three seeds and matched CARAT/FLTrust reference budgets.}",
        r"\label{tab:cifar100-main-acc}",
        r"\definecolor{caratrow}{gray}{0.91}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c c l c c c c c}",
        r"\toprule",
        r"\multirow{2}{*}{\makecell[c]{Dataset\\Model}} &",
        r"\multirow{2}{*}{\makecell[c]{Non-IID\\setting}} &",
        r"\multirow{2}{*}{\makecell[c]{Defense\\rule}} &",
        r"\multicolumn{4}{c}{\textbf{Attacks}} &",
        r"\multirow{2}{*}{\makecell[c]{Avg.\\rank}} \\",
        r"\cmidrule(lr){4-7}",
        r"& & & \textbf{ALIE} & \textbf{FangAttack} & \textbf{MinMax} & \textbf{MinSum} & \\",
        r"\midrule",
    ]

    for alpha_index, alpha in enumerate(ALPHAS):
        if alpha_index:
            lines.append(r"\cmidrule(lr){1-8}")
        display_alpha = "1.0" if alpha == "1" else alpha
        row_count = len(MAIN_TEXT_DEFENSES)
        for defense_index, defense in enumerate(MAIN_TEXT_DEFENSES):
            prefix = []
            if defense_index == 0:
                prefix.append(rf"\multirow{{{row_count}}}{{*}}{{\makecell[c]{{CIFAR-100\\ResNet18}}}}")
                prefix.append(rf"& \multirow{{{row_count}}}{{*}}{{$\alpha={display_alpha}$}}")
            else:
                prefix.append("&")

            defense_name = latex_defense_name(defense)
            is_carat = defense == "CARAT"
            cells = []
            for attack in ATTACKS:
                item = stats[(alpha, defense, attack)]
                cells.append(latex_metric_cell(item["mean"], item["std"], "accuracy", bold=is_carat))
            rank = ranks[(alpha, defense)]
            rank_cell = rf"\textbf{{${rank:.1f}$}}" if is_carat else rf"${rank:.1f}$"
            if is_carat:
                lines.append(r"\rowcolor{caratrow}")
            lines.append(
                " ".join(prefix)
                + rf" & {defense_name} & "
                + " & ".join(cells)
                + rf" & {rank_cell} \\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_readme(
    output: Path,
    rows: list[RunResult],
    acc_stats: dict[tuple[str, str, str], dict[str, float]],
    acc_ranks: dict[tuple[str, str], float],
    acc_main_ranks: dict[tuple[str, str], float],
) -> None:
    counts: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        counts.setdefault((row.source, row.defense), set()).add(row.seed)

    full_leaders = []
    main_leaders = []
    for alpha in ALPHAS:
        full_ranked = sorted(
            ((defense, acc_ranks[(alpha, defense)]) for defense in DEFENSES),
            key=lambda item: item[1],
        )
        main_ranked = sorted(
            ((defense, acc_main_ranks[(alpha, defense)]) for defense in MAIN_TEXT_DEFENSES),
            key=lambda item: item[1],
        )
        full_leader = full_ranked[0][0]
        main_leader = main_ranked[0][0]
        carat_rank = acc_ranks[(alpha, "CARAT")]
        carat_main_rank = acc_main_ranks[(alpha, "CARAT")]
        carat_mean = statistics.mean(
            acc_stats[(alpha, "CARAT", attack)]["mean"] for attack in ATTACKS
        )
        full_leaders.append(
            f"alpha={alpha}: best average rank is {full_leader}; "
            f"CARAT average rank={carat_rank:.1f}, mean final accuracy={100.0 * carat_mean:.2f}%."
        )
        main_leaders.append(
            f"alpha={alpha}: best average rank in the main-text subset is {main_leader}; "
            f"CARAT main-text average rank={carat_main_rank:.1f}."
        )

    lines = [
        "# NeurIPS Core Public-Baseline Summary",
        "",
        "Scope: CIFAR-100/ResNet18, non-i.i.d., 20 clients, 20% malicious clients, 200 rounds, attacks ALIE/FangAttack/MinMax/MinSum.",
        "Excluded: TriGuardFL and all stress-test settings.",
        "",
        "Seed coverage:",
    ]
    for (source, defense), seeds in sorted(counts.items()):
        lines.append(f"- {defense}: {source}, n={len(seeds)}, seeds={','.join(map(str, sorted(seeds)))}")
    lines.extend(["", "Main-text accuracy observations:"])
    lines.extend(f"- {line}" for line in main_leaders)
    lines.extend(["", "Complete appendix accuracy observations:"])
    lines.extend(f"- {line}" for line in full_leaders)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_playground = repo_root.parent / "Poisoning_Resilient_Federated_Learning_Playground"
    default_output = repo_root / "output" / "spreadsheet" / "neurips_core_public"

    parser = argparse.ArgumentParser(description="Summarize final NeurIPS CARAT public-baseline results.")
    parser.add_argument("--playground-root", type=Path, default=default_playground)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--rounds", type=int, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_runs(args.playground_root, args.rounds)
    if not rows:
        raise SystemExit("no matching runs found")

    acc_stats = grouped_stats(rows, "accuracy")
    loss_stats = grouped_stats(rows, "loss")
    acc_ranks = average_ranks(acc_stats, higher_is_better=True)
    acc_main_ranks = average_ranks(acc_stats, higher_is_better=True, defenses=MAIN_TEXT_DEFENSES)
    loss_ranks = average_ranks(loss_stats, higher_is_better=False)

    write_detail_csv(rows, args.output_dir / "run_detail.csv")
    write_summary_csv(args.output_dir / "accuracy_summary.csv", acc_stats, acc_ranks, "accuracy")
    write_summary_csv(
        args.output_dir / "accuracy_main_summary.csv",
        acc_stats,
        acc_main_ranks,
        "accuracy",
        defenses=MAIN_TEXT_DEFENSES,
    )
    write_summary_csv(args.output_dir / "loss_summary.csv", loss_stats, loss_ranks, "loss")
    full_note = "This appendix table includes TrimmedMean for transparency."
    main_table = latex_main_accuracy_table(acc_stats, acc_main_ranks)
    full_table = latex_table(
        acc_stats,
        acc_ranks,
        "accuracy",
        defenses=DEFENSES,
        label="tab:cifar100-full-acc",
        caption_extra=full_note,
    )
    (args.output_dir / "accuracy_table.tex").write_text(main_table, encoding="utf-8")
    (args.output_dir / "accuracy_main_table.tex").write_text(main_table, encoding="utf-8")
    (args.output_dir / "accuracy_full_table.tex").write_text(full_table, encoding="utf-8")
    (args.output_dir / "loss_table.tex").write_text(latex_table(loss_stats, loss_ranks, "loss"), encoding="utf-8")
    write_readme(args.output_dir / "README.md", rows, acc_stats, acc_ranks, acc_main_ranks)

    expected_cells = len(ALPHAS) * len(ATTACKS) * len(DEFENSES)
    actual_cells = len(acc_stats)
    print(f"rows={len(rows)} cells={actual_cells}/{expected_cells} output={args.output_dir}")
    if actual_cells != expected_cells:
        raise SystemExit("missing summary cells")


if __name__ == "__main__":
    main()
