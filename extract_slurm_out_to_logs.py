#!/usr/bin/env python3
import argparse
import re
from collections import Counter
from pathlib import Path


FILENAME_BASENAME_RE = re.compile(
    r"([A-Za-z0-9]+_[A-Za-z0-9]+_(?:iid|non-iid)_[A-Za-z0-9]+_[A-Za-z0-9]+"
    r"_[0-9]+_[0-9]+_[0-9.]+_[A-Za-z0-9]+_adv[0-9.]+_seed[0-9]+"
    r"(?:_alpha[0-9.]+)?_cfg[A-Za-z0-9_.-]+(?:_exp[0-9]+)?\.txt)"
)
FILENAME_META_RE = re.compile(
    r"^(?P<dataset>[^_]+)_(?P<model>[^_]+)_(?P<iid>iid|non-iid)_(?P<attack>[^_]+)_(?P<defense>[^_]+)"
    r"_(?P<epochs>\d+)_(?P<num_clients>\d+)_(?P<lr>[0-9.]+)_(?P<algo>[^_]+)_adv(?P<adv>[0-9.]+)"
    r"_seed(?P<seed>\d+)(?:_alpha(?P<alpha>[0-9.]+))?_cfg(?P<cfg>[A-Za-z0-9_.-]+)(?:_exp(?P<expid>\d+))?\.txt$"
)
SEED_LINE_RE = re.compile(r"^seed:\s*\d+,")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\b")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract FL_Poison logs from slurm *.out files. "
            "Each in-file seed run is split into an individual .txt log."
        )
    )
    parser.add_argument(
        "--slurm-dir",
        type=Path,
        default=Path("/home/fengye/slurm"),
        help="Directory containing slurm *.out files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Poison/logs_from_slurm_split"),
        help="Output root for extracted .txt logs.",
    )
    return parser.parse_args()


def score_text(text: str):
    epochs = [int(x) for x in EPOCH_RE.findall(text)]
    max_epoch = max(epochs) if epochs else -1
    n_epoch = len(epochs)
    return (max_epoch, n_epoch, len(text))


def segment_seed_runs(text: str):
    lines = text.splitlines()
    seed_idx = [i for i, line in enumerate(lines) if SEED_LINE_RE.match(line)]
    if not seed_idx:
        return []

    segments = []
    for k, start in enumerate(seed_idx):
        end = seed_idx[k + 1] if k + 1 < len(seed_idx) else len(lines)
        seg_lines = lines[start:end]
        seg_text = "\n".join(seg_lines).strip() + "\n"
        segments.append(seg_text)
    return segments


def main():
    args = parse_args()
    if not args.slurm_dir.exists():
        raise FileNotFoundError(f"slurm directory not found: {args.slurm_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_files = sorted(args.slurm_dir.glob("*.out"))

    written_new = 0
    replaced_existing = 0
    kept_existing = 0
    failed_out = 0
    failed_segments = 0
    total_segments = 0

    by_dataset = Counter()
    by_iid = Counter()
    by_adv = Counter()

    manifest_lines = []
    for src in out_files:
        text = src.read_text(encoding="utf-8", errors="ignore")
        segments = segment_seed_runs(text)
        if not segments:
            failed_out += 1
            continue

        for idx, seg_text in enumerate(segments, start=1):
            total_segments += 1
            names = FILENAME_BASENAME_RE.findall(seg_text)
            if not names:
                failed_segments += 1
                continue

            out_name = names[-1]
            meta_match = FILENAME_META_RE.match(out_name)
            if not meta_match:
                failed_segments += 1
                continue
            meta = meta_match.groupdict()
            dst_dir = args.out_dir / meta["algo"] / f"{meta['dataset']}_{meta['model']}" / meta["iid"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / out_name

            new_score = score_text(seg_text)
            take_new = True
            if dst.exists():
                old_text = dst.read_text(encoding="utf-8", errors="ignore")
                old_score = score_text(old_text)
                if new_score <= old_score:
                    take_new = False

            if dst.exists() and not take_new:
                kept_existing += 1
            else:
                if dst.exists():
                    replaced_existing += 1
                else:
                    written_new += 1
                dst.write_text(seg_text, encoding="utf-8")

            by_dataset[meta["dataset"]] += 1
            by_iid[meta["iid"]] += 1
            by_adv[meta["adv"]] += 1
            manifest_lines.append(
                f"{src}\tsegment={idx}\t{dst}\tmax_epoch={new_score[0]}\tepoch_lines={new_score[1]}"
            )

    manifest_path = args.out_dir / "extraction_manifest.tsv"
    manifest_path.write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""),
        encoding="utf-8",
    )

    result_txt = list(args.out_dir.rglob("*.txt"))
    summary_lines = [
        f"slurm_dir={args.slurm_dir}",
        f"out_dir={args.out_dir}",
        f"out_files={len(out_files)}",
        f"failed_out(no_seed_segment)={failed_out}",
        f"total_seed_segments={total_segments}",
        f"failed_segments(no_valid_filename)={failed_segments}",
        f"written_new={written_new}",
        f"replaced_existing={replaced_existing}",
        f"kept_existing={kept_existing}",
        f"result_txt_total={len(result_txt)}",
        f"dataset_seen={dict(by_dataset)}",
        f"iid_seen={dict(by_iid)}",
        f"adv_seen={dict(by_adv)}",
        f"manifest={manifest_path}",
    ]
    summary_path = args.out_dir / "extraction_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
