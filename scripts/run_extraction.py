"""Phase 3: run argument extraction on the dev or test sample and score argument detection.

Run from the project root:
    python scripts/run_extraction.py --limit 60       # quick first look
    python scripts/run_extraction.py                  # full dev sample
    python scripts/run_extraction.py --split test

The UKP corpus says, for each sentence, whether it is an argument. So this scores the first half of
extraction, Argument vs NoArgument (Claim and Premise both count as Argument), with precision, recall
and F1. UKP has no claim/premise labels: to score those, annotate your own transcript
(scripts/make_annotation_sheet.py, then scripts/run_annotated_eval.py).

Results: results/extraction_<split>.jsonl (every prediction with the restated argument) and .json (scores).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.evaluation import (  # noqa: E402
    ARG_LABELS, Progress, load_split_or_exit, predict_arguments, take, write_json, write_jsonl,
)
from argus.llm import make_ready_client  # noqa: E402
from argus.metrics import classification_report, format_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--limit", type=int, default=None, help="score only a random subset of this size")
    ap.add_argument("--model", default=None, help="override the model in config.yaml")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    records = take(load_split_or_exit(args.data_dir, args.split), args.limit)
    llm = make_ready_client(args.model)
    print(f"Model: {llm.model} | split: {args.split} | n={len(records)}")

    progress = Progress(len(records), "extract")
    rows = predict_arguments(llm, records, progress)
    progress.close()

    scored = [r for r in rows if r["gold"] is not None]
    report = None
    if scored:
        report = classification_report([r["gold"] for r in scored], [r["pred"] for r in scored], ARG_LABELS)
        print("\nArgument detection against the dataset's labels:\n" + format_report(report))
    else:
        print("\nThis dataset (VAST) has no argument / non-argument labels, so there is nothing to score here.\n"
              "Score extraction on a transcript you annotate yourself:  python scripts/run_annotated_eval.py\n"
              "(see INSTALL.md, step 10). The table below still shows how roles line up with the stance labels.")

    # How the model's roles line up with the gold stance labels: a quick sanity check.
    table = Counter((r["gold_stance"], r["component"] or "ERROR") for r in rows)
    comps = ["Claim", "Premise", "None", "ERROR"]
    print("\nPredicted role by gold stance label")
    print(f"{'gold stance':<12}" + "".join(f"{c:>9}" for c in comps))
    for stance in ["Favor", "Against", "Neutral"]:
        print(f"{stance:<12}" + "".join(f"{table[(stance, c)]:>9}" for c in comps))

    stem = f"extraction_{args.split}"
    write_jsonl(rows, args.out_dir / f"{stem}.jsonl")
    write_json({"model": llm.model, "split": args.split, "report": report}, args.out_dir / f"{stem}.json")
    print(f"\nSaved {args.out_dir / (stem + '.jsonl')} and .json")


if __name__ == "__main__":
    main()
