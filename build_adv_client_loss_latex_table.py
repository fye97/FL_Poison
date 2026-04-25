#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


FILENAME_RE = re.compile(
    r"^(?P<dataset>[^_]+)_(?P<model>[^_]+)_(?P<iid>iid|non-iid)_(?P<attack>[^_]+)_(?P<defense>[^_]+)"
    r"_(?P<epochs>\d+)_(?P<num_clients>\d+)_(?P<lr>[0-9.]+)_(?P<algo>[^_]+)_adv(?P<adv>[0-9.]+)"
    r"_seed(?P<seed>\d+)(?:_alpha(?P<alpha>[0-9.]+))?_cfg(?P<cfg>.+?)(?:_exp(?P<expid>\d+))?\.txt$"
)
EPOCH_METRIC_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+.*?Test Acc:\s*(?P<acc>[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+Test loss:\s*(?P<loss>[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
)

ATTACKS = ["ALIE", "FangAttack", "MinMax", "MinSum"]
DEFENSES = ["Mean", "NormClipping", "MultiKrum", "FLTrust", "FLDetector", "TriGuardFL"]
BASELINE_ATTACK = "NoAttack"
BASELINE_DEFENSE = "Mean"
BASELINE_LABEL = "NoAttack+Mean(Base)"
DISPLAY_DECIMALS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a LaTeX test-loss table for adv-fixed client-count comparison "
            "across iid/non-iid settings."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/logs_from_slurm_split_20260312"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/fengye/FL_Poison/output/spreadsheet/adv0.2_client_loss_tables"),
    )
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--adv", type=str, default="0.2")
    parser.add_argument("--lr", type=str, default="0.05")
    parser.add_argument("--cfg", type=str, default="FedAvg_CIFAR100_config")
    parser.add_argument(
        "--clients",
        type=str,
        default="20,40,60",
        help="Comma-separated client counts to compare.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="1,0.5,0.1",
        help="Comma-separated non-iid alpha values to compare.",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default="42,43,44,45,46",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=200,
        help="Use epochs < max-epoch; final loss means the last available logged epoch before this bound.",
    )
    parser.add_argument(
        "--exclude-config-epochs",
        type=str,
        default="300",
    )
    parser.add_argument(
        "--filter-buggy-fltrust",
        action="store_true",
    )
    parser.add_argument("--fltrust-tail-low", type=float, default=0.09)
    parser.add_argument("--fltrust-tail-high", type=float, default=0.11)
    parser.add_argument("--fltrust-tail-span-max", type=float, default=0.04)
    return parser.parse_args()


def parse_log_name(path: Path):
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    meta = match.groupdict()
    meta["alpha"] = meta["alpha"] if meta["alpha"] is not None else "NA"
    meta["expid"] = meta["expid"] if meta["expid"] is not None else "NA"
    meta["file"] = str(path)
    return meta


def parse_epoch_metrics(path: Path, max_epoch: int):
    curve = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EPOCH_METRIC_RE.search(line)
            if not match:
                continue
            epoch = int(match.group("epoch"))
            if epoch >= max_epoch:
                continue
            curve[epoch] = (
                float(match.group("acc")),
                float(match.group("loss")),
            )
    return curve


def reduce_final_loss(curve: dict):
    if not curve:
        return None
    last_epoch = max(curve.keys())
    return float(curve[last_epoch][1]), last_epoch


def is_buggy_fltrust(curve: dict, low: float, high: float, span_max: float):
    if not curve:
        return False
    epochs = sorted(curve.keys())
    accs = [curve[e][0] for e in epochs]
    tail = accs[-20:] if len(accs) >= 20 else accs
    tail_mean = float(np.mean(tail))
    tail_span = float(np.max(tail) - np.min(tail))
    return low <= tail_mean <= high and tail_span <= span_max


def fmt_cell(mean: float, std: float, bold: bool):
    text = f"{mean:.{DISPLAY_DECIMALS}f}$\\pm${std:.{DISPLAY_DECIMALS}f}"
    return f"\\textbf{{{text}}}" if bold else text


def table_label(dataset: str, model: str, adv: str):
    safe = re.sub(r"[^A-Za-z0-9]+", "", f"{dataset}{model}adv{adv}")
    return f"tablossclients{safe.lower()}"


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client_values = [int(x.strip()) for x in args.clients.split(",") if x.strip()]
    alpha_values = [x.strip() for x in args.alphas.split(",") if x.strip()]
    seed_filter = {x.strip() for x in args.seed_list.split(",") if x.strip()}
    excluded_config_epochs = {
        x.strip() for x in args.exclude_config_epochs.split(",") if x.strip()
    }

    curve_cache = {}

    def load_curve(file_path: str):
        if file_path not in curve_cache:
            curve_cache[file_path] = parse_epoch_metrics(Path(file_path), args.max_epoch)
        return curve_cache[file_path]

    # setting key -> attack -> defense -> list[meta]
    attacked = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    baseline = defaultdict(list)
    parsed = 0
    filtered_epoch_cfg = 0
    filtered_seed = 0
    filtered_buggy_fltrust = 0

    for path in sorted(args.logs_root.rglob("*.txt")):
        meta = parse_log_name(path)
        if meta is None:
            continue
        if meta["dataset"] != args.dataset or meta["model"] != args.model:
            continue
        if meta["algo"] != args.algorithm or meta["adv"] != args.adv:
            continue
        if meta["lr"] != args.lr or meta["cfg"] != args.cfg:
            continue
        if meta["epochs"] in excluded_config_epochs:
            filtered_epoch_cfg += 1
            continue
        if seed_filter and meta["seed"] not in seed_filter:
            filtered_seed += 1
            continue
        if int(meta["num_clients"]) not in client_values:
            continue
        if meta["iid"] == "non-iid" and meta["alpha"] not in alpha_values:
            continue
        if meta["iid"] == "iid":
            setting = ("iid", "NA", int(meta["num_clients"]))
        else:
            setting = ("non-iid", meta["alpha"], int(meta["num_clients"]))

        if args.filter_buggy_fltrust and meta["defense"] == "FLTrust":
            curve = load_curve(meta["file"])
            if is_buggy_fltrust(
                curve,
                args.fltrust_tail_low,
                args.fltrust_tail_high,
                args.fltrust_tail_span_max,
            ):
                filtered_buggy_fltrust += 1
                continue

        parsed += 1
        if meta["attack"] == BASELINE_ATTACK and meta["defense"] == BASELINE_DEFENSE:
            baseline[setting].append(meta)
        elif meta["attack"] in ATTACKS and meta["defense"] in DEFENSES:
            attacked[setting][meta["attack"]][meta["defense"]].append(meta)

    ordered_settings = [("iid", "NA", c) for c in client_values]
    for alpha in alpha_values:
        ordered_settings.extend([("non-iid", alpha, c) for c in client_values])

    def aggregate(metas):
        if not metas:
            return None
        by_seed = defaultdict(list)
        for meta in metas:
            by_seed[meta["seed"]].append(meta)

        vals = []
        last_epochs = []
        seeds_used = []
        for seed, seed_metas in by_seed.items():
            chosen = sorted(seed_metas, key=lambda x: x["file"])[-1]
            curve = load_curve(chosen["file"])
            reduced = reduce_final_loss(curve)
            if reduced is None:
                continue
            loss, last_epoch = reduced
            vals.append(loss)
            last_epochs.append(last_epoch)
            seeds_used.append(seed)
        if not vals:
            return None
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
            "seeds": ",".join(sorted(seeds_used, key=lambda x: int(x) if x.isdigit() else x)),
            "last_epoch_min": int(min(last_epochs)),
            "last_epoch_max": int(max(last_epochs)),
        }

    values = {}
    detail_rows = []
    for setting in ordered_settings:
        for attack in ATTACKS:
            for defense in DEFENSES:
                agg = aggregate(attacked[setting][attack][defense])
                values[(setting, attack, defense)] = agg
                if agg is not None:
                    detail_rows.append(
                        {
                            "iid": setting[0],
                            "alpha": setting[1],
                            "num_clients": setting[2],
                            "attack": attack,
                            "defense": defense,
                            **agg,
                        }
                    )
        agg = aggregate(baseline[setting])
        values[(setting, BASELINE_ATTACK, BASELINE_LABEL)] = agg
        if agg is not None:
            detail_rows.append(
                {
                    "iid": setting[0],
                    "alpha": setting[1],
                    "num_clients": setting[2],
                    "attack": BASELINE_ATTACK,
                    "defense": BASELINE_LABEL,
                    **agg,
                }
            )

    best = {}
    for attack in ATTACKS:
        for setting in ordered_settings:
            candidates = []
            for defense in DEFENSES:
                agg = values.get((setting, attack, defense))
                if agg is None:
                    continue
                candidates.append((round(agg["mean"], DISPLAY_DECIMALS), defense))
            if candidates:
                best_val = min(v for v, _ in candidates)
                best[(setting, attack)] = {
                    defense for v, defense in candidates if np.isclose(v, best_val)
                }

    def setting_label(setting):
        iid, alpha, clients = setting
        if iid == "iid":
            return f"IID-{clients}"
        return f"$\\\\alpha={alpha}$-{clients}"

    col_spec = "l l" + " c" * len(ordered_settings)
    lines = [
        r"\begin{table*}[t]",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\setlength{\tabcolsep}{3.5pt}",
        (
            r"\caption{Final available test loss before epoch 200 for the CIFAR-100 dataset "
            r"with ResNet18 under four model-poisoning attacks at adversary ratio $20\%$, "
            r"comparing IID and Non-IID settings across different client counts. "
            r"Each cell reports mean$\pm$std over available seeds. "
            r"If a run stopped early, its last recorded epoch before 200 is used.}"
        ),
        rf"\label{{{table_label(args.dataset, args.model, args.adv)}}}",
        r"\centering\scriptsize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\multirow{2}{*}{Attack} & \multirow{2}{*}{Defense} & \multicolumn{3}{c}{IID} & \multicolumn{9}{c}{Non-IID} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-14}",
        (
            "& & "
            + " & ".join([str(c) for c in client_values])
            + " & "
            + " & ".join(
                [f"{setting_label(('non-iid', alpha, c))}" for alpha in alpha_values for c in client_values]
            )
            + r" \\"
        ),
        r"\midrule",
    ]

    for attack in ATTACKS:
        for idx, defense in enumerate(DEFENSES):
            row = []
            if idx == 0:
                row.append(rf"\multirow{{{len(DEFENSES)}}}{{*}}{{{attack}}}")
            else:
                row.append("")
            row.append(defense)
            for setting in ordered_settings:
                agg = values.get((setting, attack, defense))
                if agg is None:
                    row.append("--")
                    continue
                row.append(
                    fmt_cell(
                        agg["mean"],
                        agg["std"],
                        defense in best.get((setting, attack), set()),
                    )
                )
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    baseline_row = ["Base", BASELINE_LABEL]
    for setting in ordered_settings:
        agg = values.get((setting, BASELINE_ATTACK, BASELINE_LABEL))
        if agg is None:
            baseline_row.append("--")
        else:
            baseline_row.append(fmt_cell(agg["mean"], agg["std"], False))
    lines.append(" & ".join(baseline_row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    out_tex = args.out_dir / f"adv{args.adv}_{args.dataset}_{args.model}_clients_loss_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")

    detail_csv = args.out_dir / f"adv{args.adv}_{args.dataset}_{args.model}_clients_loss_detail.csv"
    if detail_rows:
        with detail_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iid",
                    "alpha",
                    "num_clients",
                    "attack",
                    "defense",
                    "mean",
                    "std",
                    "n",
                    "seeds",
                    "last_epoch_min",
                    "last_epoch_max",
                ],
            )
            writer.writeheader()
            writer.writerows(detail_rows)

    summary_lines = [
        f"logs_root={args.logs_root}",
        f"out_dir={args.out_dir}",
        f"dataset={args.dataset}",
        f"model={args.model}",
        f"algorithm={args.algorithm}",
        f"adv={args.adv}",
        f"parsed_logs={parsed}",
        f"filtered_seed={filtered_seed}",
        f"filtered_epoch_cfg={filtered_epoch_cfg}",
        f"filtered_buggy_fltrust={filtered_buggy_fltrust}",
        f"table_tex={out_tex}",
        f"detail_csv={detail_csv}",
    ]
    summary_path = args.out_dir / f"adv{args.adv}_{args.dataset}_{args.model}_clients_loss_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
