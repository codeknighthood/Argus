"""Phase 1 pipeline: raw VAST files -> cleaned records -> balanced dev and test samples.

Run from the project root (after scripts/download_data.py):
    python scripts/prepare_data.py
    python scripts/prepare_data.py --dev-per-label 150 --test-per-label 300 --max-per-topic 4 --seed 42

Writes to data/processed/:
    all.jsonl   every cleaned, de-duplicated record
    dev.jsonl   balanced sample from VAST's own dev file: iterate on prompts and compare strategies here
    test.jsonl  larger balanced sample from VAST's own test file: use it ONCE per method for the reported numbers

Dev and test come from different VAST files, which share almost no topics, so a strategy tuned on dev is not
being scored on the same topics. Each sample has the same number of Favor, Against and Neutral examples, and no topic
can contribute more than --max-per-topic examples of one label.

The older UKP corpus can still be used:  python scripts/prepare_data.py --source ukp --raw data/raw/ukp
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.data import (  # noqa: E402
    dedupe, load_ukp, load_vast, sample_balanced, save_jsonl, stats_table, stratified_dev_test,
)


def summary(records, name: str) -> str:
    labels = Counter(r.gold_stance.value for r in records)
    kinds = Counter(r.group or "-" for r in records)
    lines = [f"{name}: {len(records)} examples, {len({r.topic for r in records})} topics",
             "  labels: " + ", ".join(f"{k} {labels[k]}" for k in ("Favor", "Against", "Neutral")),
             "  how the topic was matched: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))]
    words = sorted(len(r.statement.split()) for r in records)
    if words:
        lines.append(f"  words per statement: median {words[len(words) // 2]}, longest {words[-1]}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["vast", "ukp"], default="vast")
    ap.add_argument("--raw", type=Path, default=None, help="folder with the raw files (default data/raw/<source>)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--dev-per-label", type=int, default=150)
    ap.add_argument("--test-per-label", type=int, default=300)
    ap.add_argument("--max-per-topic", type=int, default=4, help="cap per topic and label in each sample (VAST)")
    ap.add_argument("--dev-per-cell", type=int, default=10, help="UKP only: per topic and per label")
    ap.add_argument("--test-per-cell", type=int, default=30, help="UKP only: per topic and per label")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw = args.raw or ROOT / "data" / "raw" / args.source

    try:
        records = load_vast(raw, splits=["dev", "test"]) if args.source == "vast" else load_ukp(raw)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))                                  # a plain message, not a traceback
    print(f"Loaded {len(records)} records from {raw}")
    records, dups, conflicts = dedupe(records)
    print(f"Removed {dups} exact duplicates and {conflicts} records with conflicting labels")

    if args.source == "vast":
        dev_pool = [r for r in records if r.split == "dev"]
        test_pool = [r for r in records if r.split == "test"]
        if not dev_pool or not test_pool:
            sys.exit("Need both vast_dev.csv and vast_test.csv in " + str(raw) + ". Run: python scripts/download_data.py")
        dev = [r.model_copy(update={"split": "dev"}) for r in sample_balanced(dev_pool, args.dev_per_label, args.max_per_topic, args.seed)]
        test = [r.model_copy(update={"split": "test"}) for r in sample_balanced(test_pool, args.test_per_label, args.max_per_topic, args.seed)]
        shared = {r.topic for r in dev} & {r.topic for r in test}
    else:
        dev, test = stratified_dev_test(records, args.dev_per_cell, args.test_per_cell, args.seed)
        shared = set()
    save_jsonl(records, args.out / "all.jsonl")
    save_jsonl(dev, args.out / "dev.jsonl")
    save_jsonl(test, args.out / "test.jsonl")

    print()
    for name, rs in (("ALL (cleaned)", records), ("DEV sample", dev), ("TEST sample", test)):
        print(summary(rs, name) + "\n")
    if args.source == "ukp":
        print(stats_table(dev) + "\n")
    elif shared:
        print(f"{len(shared)} topics appear in both dev and test (of {len({r.topic for r in test})} in test).")
    print(f"Saved to {args.out}")
    print("Next:  python scripts/evaluate.py --limit 60")


if __name__ == "__main__":
    main()
