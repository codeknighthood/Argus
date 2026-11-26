"""Phase 2: run stance detection on the dev or test sample and score it.

Run from the project root:
    python scripts/run_stance.py                                  # zero-shot on the dev sample
    python scripts/run_stance.py --strategy few_shot --limit 60   # quick check on 60 random rows
    python scripts/run_stance.py --split test --strategy cot

Results go to results/stance_<strategy>_<split>.jsonl (one row per statement) and .json (the scores).
Everything the model answers is cached, so stopping and re-running only redoes what is missing.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.evaluation import Progress, load_split_or_exit, predict_records, take, write_json, write_jsonl  # noqa: E402
from argus.llm import make_ready_client  # noqa: E402
from argus.metrics import classification_report, format_report  # noqa: E402
from argus.schema import Stance  # noqa: E402
from argus.stance import STRATEGIES  # noqa: E402

LABELS = [s.value for s in Stance]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="zero_shot")
    ap.add_argument("--limit", type=int, default=None, help="score only a random subset of this size")
    ap.add_argument("--model", default=None, help="override the model in config.yaml")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    records = take(load_split_or_exit(args.data_dir, args.split), args.limit)
    llm = make_ready_client(args.model)
    print(f"Model: {llm.model} | strategy: {args.strategy} | split: {args.split} | n={len(records)}")

    progress = Progress(len(records), "stance")
    rows = predict_records(llm, records, args.strategy, progress)
    progress.close()

    report = classification_report([r["gold"] for r in rows], [r["pred"] for r in rows], LABELS)
    print("\n" + format_report(report))

    stem = f"stance_{args.strategy}_{args.split}"
    write_jsonl(rows, args.out_dir / f"{stem}.jsonl")
    write_json({"model": llm.model, "strategy": args.strategy, "split": args.split, "report": report},
               args.out_dir / f"{stem}.json")
    print(f"\nSaved {args.out_dir / (stem + '.jsonl')} and .json")


if __name__ == "__main__":
    main()
