"""Phase 6: compare the stance prompting strategies (and optionally several models) and analyse the mistakes.

Run from the project root:
    python scripts/evaluate.py --limit 60                              # quick look on 60 dev sentences
    python scripts/evaluate.py                                         # all three strategies on the dev sample
    python scripts/evaluate.py --split test                            # the numbers to REPORT (do this once)
    python scripts/evaluate.py --models qwen2.5:7b llama3.1:8b         # also compare two models
    python scripts/evaluate.py --consistency-runs 3                    # also check answer stability

How to use the splits: compare strategies on dev, pick the best one (set stance_strategy in config.yaml), then run
--split test once and report that. Choosing a strategy by looking at test scores makes the test numbers optimistic.

Writes to results/evaluation/:
    summary.md      the report: comparison with confidence intervals, significance tests, per-label and per-topic
                    scores, confusion matrices, kinds of mistakes, sentence features linked to errors
    summary.json    the same numbers for further processing
    stance_<run>_<split>.jsonl   every prediction with its explanation
    errors_<run>_<split>.csv     every mistake in random order, ready for you to tag with its cause
                                 (see docs/ERROR_ANALYSIS.md)

Answers are cached, so if you already ran scripts/run_stance.py with the same model this reuses those answers.
Cost on a first run: sentences x strategies x models model calls (about 720 per strategy for the default test sample).
"""
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.analysis import agreement_breakdown, pairwise_kappa, tag_error_rates  # noqa: E402
from argus.compare import (  # noqa: E402
    LABELS, best_run, pairwise_tests, run_stance_comparison, safe_name, write_error_csv,
)
from argus.evaluation import Progress, load_split_or_exit, self_consistency, take, write_json, write_jsonl  # noqa: E402
from argus.llm import load_config, make_client, make_ready_client  # noqa: E402
from argus.report import console_table, stance_report_md  # noqa: E402
from argus.stance import STRATEGIES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--strategies", nargs="+", choices=list(STRATEGIES), default=list(STRATEGIES))
    ap.add_argument("--models", nargs="+", default=None, help="models to compare (default: the one in config.yaml)")
    ap.add_argument("--limit", type=int, default=None, help="use only a random subset of this size")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for confidence intervals (0 = skip)")
    ap.add_argument("--consistency-runs", type=int, default=0, help="repeat the best run this many times at a "
                    "non-zero temperature to measure answer stability (0 = skip; try 3)")
    ap.add_argument("--consistency-temperature", type=float, default=0.7)
    ap.add_argument("--consistency-limit", type=int, default=100, help="sentences used for the consistency check")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "evaluation")
    args = ap.parse_args()

    records = take(load_split_or_exit(args.data_dir, args.split), args.limit)
    models = args.models or [load_config()["model"]]
    llms = {m: make_ready_client(m) for m in models}
    n_calls = len(records) * len(args.strategies) * len(models)
    print(f"Split: {args.split} | sentences: {len(records)} | strategies: {', '.join(args.strategies)} | "
          f"models: {', '.join(models)}")
    print(f"Up to {n_calls} model calls (answers already cached are free).")
    if args.split == "test":
        print("TEST split: report these numbers, but do not use them to choose a strategy.")

    runs = run_stance_comparison(llms, records, args.strategies, args.bootstrap,
                                 progress_factory=lambda total, name: Progress(total, name))
    best = best_run(runs)
    tests = pairwise_tests(runs)
    runs_rows = {r["name"]: r["rows"] for r in runs}

    consistency = None
    if args.consistency_runs > 1:
        subset = take(records, args.consistency_limit, seed=1)
        print(f"\nSelf-consistency of '{best['name']}': {args.consistency_runs} runs x {len(subset)} sentences "
              f"at temperature {args.consistency_temperature}")
        bar = Progress(len(subset) * args.consistency_runs, "consistency")
        consistency = self_consistency(
            subset, best["strategy"], args.consistency_runs,
            lambda k: make_client(best["model"], args.consistency_temperature, f"consistency-{k}"), LABELS, bar)
        bar.close()

    meta = {"split": args.split, "n": len(records), "models": models, "date": date.today().isoformat(),
            "bootstrap": args.bootstrap, "source": ", ".join(sorted({r.source for r in records}))}
    summary_md = stance_report_md(
        runs, tests, pairwise_kappa(runs_rows), agreement_breakdown(runs_rows),
        tag_error_rates(best["rows"]), best, meta, consistency)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(summary_md, encoding="utf-8")
    write_json({"meta": meta, "best": best["name"], "tests": tests, "consistency": consistency,
                "runs": [{k: v for k, v in r.items() if k != "rows"} for r in runs]}, out / "summary.json")
    for r in runs:
        stem = f"{safe_name(r['name'])}_{args.split}"
        write_jsonl(r["rows"], out / f"stance_{stem}.jsonl")
        n_err = write_error_csv(r, out / f"errors_{stem}.csv")
        print(f"  {r['name']}: {n_err} mistakes saved to errors_{stem}.csv")

    print("\n" + console_table(runs))
    print(f"\nBest macro-F1: {best['name']}")
    print(f"Full report: {out / 'summary.md'}")
    print("Next: open the errors_*.csv for the best run and tag the causes (docs/ERROR_ANALYSIS.md).")


if __name__ == "__main__":
    main()
