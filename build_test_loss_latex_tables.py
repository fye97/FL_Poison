#!/usr/bin/env python3
import argparse
import csv
import hashlib
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

DEFAULT_ATTACKS = ["ALIE", "FangAttack", "MinMax", "MinSum"]
DEFAULT_DEFENSE_ORDER = [
    "Mean",
    "NormClipping",
    "MultiKrum",
    "FLTrust",
    "FLDetector",
    "TriGuardFL",
]
EXCLUDED_ATTACK_DEFENSES = {"Mean"}
BASELINE_ATTACK = "NoAttack"
BASELINE_DEFENSE = "Mean"
BASELINE_LABEL = "NoAttack+Mean(Base)"
DISPLAY_DECIMALS = 4

BASE_GROUP_FIELDS = [
    "iid",
    "dataset",
    "model",
    "num_clients",
    "lr",
    "algo",
    "adv",
    "alpha",
    "cfg",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build LaTeX tables for test loss from FL_Poison logs. "
            "Each table uses rows=defense, cols=attack, with mean±std over available seeds."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/logs_from_slurm"),
        help="Root folder containing *.txt logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/tables_test_loss_from_slurm"),
        help="Output root for .tex tables and CSV index.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=200,
        help="Only epochs < max-epoch are considered.",
    )
    parser.add_argument(
        "--loss-stat",
        type=str,
        default="final",
        choices=["final", "min", "mean"],
        help="How to reduce the test loss curve to one scalar per run.",
    )
    parser.add_argument(
        "--attacks",
        type=str,
        default=",".join(DEFAULT_ATTACKS),
        help="Comma-separated attacks used as table columns.",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default="42,43,44,45,46",
        help=(
            "Comma-separated seeds to include. Empty string means include all."
        ),
    )
    parser.add_argument(
        "--min-defenses-per-attack",
        type=int,
        default=2,
        help="Require at least this many defenses for each attack in one table.",
    )
    parser.add_argument(
        "--allow-partial-attacks",
        action="store_true",
        help="Allow table generation when some requested attacks are missing.",
    )
    parser.add_argument(
        "--merge-iid-noniid",
        action="store_true",
        help=(
            "If set, place iid and non-iid tables in same folder layout: "
            "dataset/adv_x.x/. Otherwise use iid/dataset/adv_x.x/."
        ),
    )
    parser.add_argument(
        "--group-by-epochs",
        action="store_true",
        help=(
            "If set, keep different configured epochs (e.g., 200 vs 300) in separate groups. "
            "Default is to merge them."
        ),
    )
    parser.add_argument(
        "--filter-buggy-fltrust",
        action="store_true",
        help=(
            "Filter FLTrust runs where tail test accuracy is suspiciously around 0.1."
        ),
    )
    parser.add_argument(
        "--fltrust-tail-low",
        type=float,
        default=0.09,
        help="Lower bound of suspicious FLTrust tail mean test accuracy.",
    )
    parser.add_argument(
        "--fltrust-tail-high",
        type=float,
        default=0.11,
        help="Upper bound of suspicious FLTrust tail mean test accuracy.",
    )
    parser.add_argument(
        "--fltrust-tail-span-max",
        type=float,
        default=0.04,
        help="Max allowed tail span (max-min) for suspicious FLTrust run.",
    )
    parser.add_argument(
        "--exclude-config-epochs",
        type=str,
        default="",
        help=(
            "Comma-separated configured epochs values to exclude by filename metadata, "
            "e.g. '300'."
        ),
    )
    return parser.parse_args()


def parse_log_name(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    meta = m.groupdict()
    if meta["alpha"] is None:
        meta["alpha"] = "NA"
    if meta["expid"] is None:
        meta["expid"] = "NA"
    meta["file"] = str(path)
    return meta


def sanitize(s: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def get_group_fields(args):
    fields = list(BASE_GROUP_FIELDS)
    if args.group_by_epochs:
        fields.append("epochs")
    return fields


def build_group_key(meta: dict, group_fields):
    return tuple((k, meta[k]) for k in group_fields)


def key_to_dict(key):
    return dict(key)


def parse_epoch_metrics(path: Path, max_epoch: int):
    # epoch -> (test_acc, test_loss)
    curve = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_METRIC_RE.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            if epoch >= max_epoch:
                continue
            acc = float(m.group("acc"))
            loss = float(m.group("loss"))
            curve[epoch] = (acc, loss)
    return curve


def reduce_loss(curve: dict, stat: str):
    if not curve:
        return None
    epochs = sorted(curve.keys())
    losses = np.array([curve[e][1] for e in epochs], dtype=float)
    if stat == "final":
        return float(losses[-1])
    if stat == "min":
        return float(np.min(losses))
    if stat == "mean":
        return float(np.mean(losses))
    raise ValueError(f"Unsupported loss stat: {stat}")


def is_buggy_fltrust(curve: dict, low: float, high: float, span_max: float):
    if not curve:
        return False
    epochs = sorted(curve.keys())
    accs = [curve[e][0] for e in epochs]
    tail = accs[-20:] if len(accs) >= 20 else accs
    tail_mean = float(np.mean(tail))
    tail_span = float(np.max(tail) - np.min(tail))
    return low <= tail_mean <= high and tail_span <= span_max


def format_cell(mean: float, std: float, bold=False):
    text = f"{mean:.{DISPLAY_DECIMALS}f}$\\pm${std:.{DISPLAY_DECIMALS}f}"
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def rounded_mean_for_display(v: float):
    return round(float(v), DISPLAY_DECIMALS)


def tied_rank_from_dict(value_dict):
    # Equivalent to MATLAB tiedrank on scalar scores; lower is better.
    items = [(k, v) for k, v in value_dict.items() if np.isfinite(v)]
    items.sort(key=lambda x: x[1])
    ranks = {k: np.nan for k in value_dict.keys()}

    i = 0
    n = len(items)
    while i < n:
        j = i
        while j + 1 < n and np.isclose(items[j + 1][1], items[i][1]):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg_rank
        i = j + 1
    return ranks


def compute_rankings(defenses, attacks, values, excluded_defenses=None):
    # testprint.m logic: per-attack column rank by ascending loss, then average rank.
    # To align with the table display, tie handling is based on rounded means
    # at DISPLAY_DECIMALS precision.
    excluded = set(excluded_defenses or [])
    rank_table = {}  # (defense, attack) -> rank
    for attack in attacks:
        col_value = {}
        for defense in defenses:
            if defense in excluded:
                continue
            v = values.get((defense, attack))
            if v is None:
                continue
            col_value[defense] = rounded_mean_for_display(v[0])
        if not col_value:
            continue
        tied = tied_rank_from_dict(col_value)
        for defense, r in tied.items():
            rank_table[(defense, attack)] = float(r)

    avg_rank = {}
    for defense in defenses:
        if defense in excluded:
            avg_rank[defense] = np.nan
            continue
        rs = [rank_table[(defense, a)] for a in attacks if (defense, a) in rank_table]
        avg_rank[defense] = float(np.mean(rs)) if rs else np.nan

    final_rank = tied_rank_from_dict(avg_rank)
    return rank_table, avg_rank, final_rank


def make_table_tex(
    info: dict,
    attacks,
    defenses,
    values,
    counts,
    loss_stat,
    epochs_tag,
    excluded_from_rank=None,
):
    # values[(defense, attack)] = (mean, std) or None
    # counts[(defense, attack)] = n
    excluded_from_rank = set(excluded_from_rank or [])
    rank_table, avg_rank, final_rank = compute_rankings(
        defenses, attacks, values, excluded_defenses=excluded_from_rank
    )
    header = " & ".join(["Defense"] + attacks + ["Avg. Rank", "Final Rank"]) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
    ]

    title = (
        f"Test loss ({loss_stat}, epoch < 200) for {info['dataset']}-{info['model']}, "
        f"{info['algo']}, adv={info['adv']}, {info['iid']}, alpha={info['alpha']}, "
        f"epochs_src={epochs_tag}. "
        f"Each cell reports mean$\\pm$std over available seeds."
    )
    lines.append(rf"\caption{{{title}}}")

    label_key = (
        f"{info['iid']}_{info['dataset']}_{info['model']}_{info['algo']}_"
        f"adv{info['adv']}_alpha{info['alpha']}"
    )
    label = sanitize(label_key)
    lines.append(rf"\label{{tab:testloss_{label}}}")
    lines.append(r"\begin{tabular}{l" + "c" * len(attacks) + r"}")
    lines[-1] = r"\begin{tabular}{l" + "c" * (len(attacks) + 2) + r"}"
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    # best (lowest mean) per attack column
    best_by_attack = {}
    for attack in attacks:
        candidates = []
        for defense in defenses:
            if defense in excluded_from_rank:
                continue
            v = values.get((defense, attack))
            if v is None:
                continue
            candidates.append((rounded_mean_for_display(v[0]), defense))
        if candidates:
            best_val = min(x[0] for x in candidates)
            best_by_attack[attack] = {
                defense for val, defense in candidates if np.isclose(val, best_val)
            }

    for defense in defenses:
        row = [defense]
        for attack in attacks:
            v = values.get((defense, attack))
            n = counts.get((defense, attack), 0)
            if v is None:
                row.append("--")
            else:
                is_best = defense in best_by_attack.get(attack, set())
                cell = format_cell(v[0], v[1], bold=is_best)
                cell += rf" \scriptsize{{(n={n})}}"
                row.append(cell)
        if defense in excluded_from_rank:
            row.append("--")
            row.append("--")
        else:
            ar = avg_rank.get(defense, np.nan)
            fr = final_rank.get(defense, np.nan)
            row.append(f"{ar:.2f}" if np.isfinite(ar) else "--")
            row.append(f"{fr:.1f}" if np.isfinite(fr) else "--")
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def make_output_name(info: dict, group_fields, epochs_tag: str):
    key_str = "__".join(f"{k}={info[k]}" for k in group_fields) + f"__epochs={epochs_tag}"
    short_hash = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:8]
    epochs_token = sanitize(str(epochs_tag).replace(",", "-"))
    base = (
        f"{info['iid']}_{info['dataset']}_{info['model']}_{info['algo']}"
        f"_adv{info['adv']}_alpha{info['alpha']}_epochs{epochs_token}"
    )
    return f"{sanitize(base)}_{short_hash}.tex"


def main():
    args = parse_args()
    group_fields = get_group_fields(args)
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    if len(attacks) != 4:
        raise ValueError("For 2x2-style tables, --attacks must contain exactly 4 attacks.")
    if not args.logs_root.exists():
        raise FileNotFoundError(f"Log directory not found: {args.logs_root}")

    if args.seed_list.strip():
        seed_filter = {s.strip() for s in args.seed_list.split(",") if s.strip()}
    else:
        seed_filter = set()
    if args.exclude_config_epochs.strip():
        excluded_config_epochs = {
            e.strip() for e in args.exclude_config_epochs.split(",") if e.strip()
        }
    else:
        excluded_config_epochs = set()

    all_txt = sorted(args.logs_root.rglob("*.txt"))
    parsed = []
    unparsed = []
    filtered_by_seed = 0
    filtered_by_epoch_cfg = 0

    for path in all_txt:
        meta = parse_log_name(path)
        if meta is None:
            unparsed.append(str(path))
            continue
        if seed_filter and meta["seed"] not in seed_filter:
            filtered_by_seed += 1
            continue
        if excluded_config_epochs and meta["epochs"] in excluded_config_epochs:
            filtered_by_epoch_cfg += 1
            continue
        parsed.append(meta)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for meta in parsed:
        grouped[build_group_key(meta, group_fields)].append(meta)

    curve_cache = {}

    def load_curve(file_path: str):
        if file_path not in curve_cache:
            curve_cache[file_path] = parse_epoch_metrics(Path(file_path), args.max_epoch)
        return curve_cache[file_path]

    csv_rows = []
    skipped_missing_attacks = 0
    skipped_insufficient_defenses = 0
    generated = 0
    filtered_buggy_fltrust = 0

    for group_key, records in grouped.items():
        # records_by_attack_defense[attack][defense] -> list[meta]
        records_by_attack_defense = defaultdict(lambda: defaultdict(list))
        baseline_metas = []
        for meta in records:
            if (
                meta["attack"] == BASELINE_ATTACK
                and meta["defense"] == BASELINE_DEFENSE
            ):
                baseline_metas.append(meta)
            if meta["attack"] not in attacks:
                continue
            if meta["defense"] in EXCLUDED_ATTACK_DEFENSES:
                continue
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
            records_by_attack_defense[meta["attack"]][meta["defense"]].append(meta)

        if not args.allow_partial_attacks:
            if any(attack not in records_by_attack_defense for attack in attacks):
                skipped_missing_attacks += 1
                continue

        if args.min_defenses_per_attack > 0:
            ok = True
            for attack in attacks:
                n_defs = len(records_by_attack_defense.get(attack, {}))
                if n_defs < args.min_defenses_per_attack:
                    ok = False
                    break
            if not ok:
                skipped_insufficient_defenses += 1
                continue

        defenses_present = set()
        for attack in attacks:
            defenses_present.update(records_by_attack_defense.get(attack, {}).keys())
        defenses = [
            d for d in DEFAULT_DEFENSE_ORDER if d in defenses_present
        ] + sorted(d for d in defenses_present if d not in DEFAULT_DEFENSE_ORDER)
        if baseline_metas:
            defenses.append(BASELINE_LABEL)

        def aggregate_runs(metas):
            if not metas:
                return None, 0
            by_seed = defaultdict(list)
            for m in metas:
                by_seed[m["seed"]].append(m)
            picked = []
            for seed, same_seed in by_seed.items():
                chosen = sorted(same_seed, key=lambda x: x["file"])[-1]
                picked.append((seed, chosen))
            picked.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

            run_vals = []
            for _, m in picked:
                curve = load_curve(m["file"])
                v = reduce_loss(curve, args.loss_stat)
                if v is not None:
                    run_vals.append(v)
            if not run_vals:
                return None, 0
            arr = np.array(run_vals, dtype=float)
            return (float(np.mean(arr)), float(np.std(arr))), int(arr.size)

        baseline_stats, baseline_n = aggregate_runs(baseline_metas)

        values = {}
        counts = {}
        for attack in attacks:
            for defense in defenses:
                if defense == BASELINE_LABEL:
                    values[(defense, attack)] = baseline_stats
                    counts[(defense, attack)] = baseline_n
                    continue
                metas = records_by_attack_defense.get(attack, {}).get(defense, [])
                if not metas:
                    values[(defense, attack)] = None
                    counts[(defense, attack)] = 0
                    continue
                stats, n = aggregate_runs(metas)
                values[(defense, attack)] = stats
                counts[(defense, attack)] = n

        info = key_to_dict(group_key)
        epoch_values = sorted(
            {r["epochs"] for r in records},
            key=lambda x: int(x) if str(x).isdigit() else str(x),
        )
        epochs_tag = ",".join(epoch_values)
        table_tex = make_table_tex(
            info=info,
            attacks=attacks,
            defenses=defenses,
            values=values,
            counts=counts,
            loss_stat=args.loss_stat,
            epochs_tag=epochs_tag,
            excluded_from_rank={BASELINE_LABEL},
        )

        if args.merge_iid_noniid:
            out_subdir = args.out_dir / info["dataset"] / f"adv_{info['adv']}"
        else:
            out_subdir = args.out_dir / info["iid"] / info["dataset"] / f"adv_{info['adv']}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / make_output_name(info, group_fields, epochs_tag)
        out_path.write_text(table_tex, encoding="utf-8")

        row = {"table_path": str(out_path)}
        row.update(info)
        row["epochs_tag"] = epochs_tag
        row["loss_stat"] = args.loss_stat
        csv_rows.append(row)
        generated += 1

    csv_path = args.out_dir / "table_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["table_path"] + group_fields + ["epochs_tag", "loss_stat"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary_lines = [
        f"log_root={args.logs_root}",
        f"output_dir={args.out_dir}",
        f"total_txt={len(all_txt)}",
        f"parsed_after_seed_filter={len(parsed)}",
        f"filtered_by_seed={filtered_by_seed}",
        f"filtered_by_epoch_cfg={filtered_by_epoch_cfg}",
        f"unparsed_txt={len(unparsed)}",
        f"groups_total={len(grouped)}",
        f"generated_tables={generated}",
        f"skipped_missing_attacks={skipped_missing_attacks}",
        f"skipped_insufficient_defenses={skipped_insufficient_defenses}",
        f"filtered_buggy_fltrust={filtered_buggy_fltrust}",
        f"max_epoch={args.max_epoch}",
        f"loss_stat={args.loss_stat}",
        f"attacks={','.join(attacks)}",
        f"seed_filter={','.join(sorted(seed_filter)) if seed_filter else 'ALL'}",
        f"min_defenses_per_attack={args.min_defenses_per_attack}",
        f"merge_iid_noniid={args.merge_iid_noniid}",
        f"group_by_epochs={args.group_by_epochs}",
        f"filter_buggy_fltrust={args.filter_buggy_fltrust}",
        f"exclude_config_epochs={','.join(sorted(excluded_config_epochs)) if excluded_config_epochs else 'NONE'}",
    ]
    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"table_index={csv_path}")
    if unparsed:
        sample = "\n".join(unparsed[:20])
        print("unparsed_samples:\n" + sample)


if __name__ == "__main__":
    main()
