"""Phase 5: run the full ARGUS pipeline on a debate transcript.

Run from the project root:
    python scripts/run_pipeline.py                               # the sample transcript
    python scripts/run_pipeline.py --transcript my_debate.txt --target "the minimum wage"
    python scripts/run_pipeline.py --limit-segments 8            # just the first 8 sentences, to try it quickly

Transcript format, one turn per line:   SPEAKER NAME: what they said
(lines without a name continue the previous turn; use --no-speakers for plain text).

For every sentence the pipeline finds: its role (Claim / Premise / None), a standalone restatement of the argument,
the stance toward the target (Favor / Against / Neutral) with an explanation, and a Support / Attack link to the nearest
earlier argument it responds to. Output files, in --out-dir (default results/):
    <name>.json   full structured output (synopsis 5.7) plus the settings used
    <name>.csv    the same as a table you can open in Excel
    <name>.md     a compact readable table

Every model answer is cached, so stopping and re-running only redoes what is missing. Expect several model calls per
sentence (extraction, stance, and up to --relation-window relation checks), so a long transcript takes a while.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.evaluation import Progress  # noqa: E402
from argus.llm import load_config, make_ready_client  # noqa: E402
from argus.output import format_summary, summarize, to_markdown, write_csv, write_json  # noqa: E402
from argus.pipeline import STANCE_ON, ArgusPipeline, PipelineConfig  # noqa: E402
from argus.segment import segment_transcript  # noqa: E402
from argus.stance import STRATEGIES  # noqa: E402

SAMPLE = ROOT / "data" / "sample" / "sample_transcript.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript", type=Path, default=SAMPLE)
    ap.add_argument("--target", default=None, help='the policy under debate, e.g. "the minimum wage"')
    ap.add_argument("--strategy", choices=list(STRATEGIES), default=None, help="stance prompting strategy")
    ap.add_argument("--stance-on", choices=list(STANCE_ON), default=None)
    ap.add_argument("--context-window", type=int, default=None)
    ap.add_argument("--relation-window", type=int, default=None)
    ap.add_argument("--min-tokens", type=int, default=None)
    ap.add_argument("--no-speakers", action="store_true", help="the transcript has no 'NAME:' prefixes")
    ap.add_argument("--limit-segments", type=int, default=None, help="only process the first N sentences")
    ap.add_argument("--model", default=None, help="override the model in config.yaml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--name", default=None, help="output file name without extension (default: transcript name)")
    args = ap.parse_args()

    target = args.target
    if target is None:
        if args.transcript != SAMPLE:
            sys.exit('Please say what the debate is about, e.g.  --target "the minimum wage"')
        target = "the minimum wage"
    if not args.transcript.exists():
        sys.exit(f"Transcript not found: {args.transcript}")

    try:
        config = PipelineConfig.from_config(
            load_config(), stance_strategy=args.strategy, stance_on=args.stance_on,
            context_window=args.context_window, relation_window=args.relation_window,
            min_tokens=args.min_tokens, parse_speakers=False if args.no_speakers else None,
        )
    except ValueError as e:
        sys.exit(str(e))

    llm = make_ready_client(args.model)
    pipeline = ArgusPipeline(llm, config)

    text = args.transcript.read_text(encoding="utf-8")
    segments = segment_transcript(text, config.context_window, config.parse_speakers, config.min_tokens)
    if args.limit_segments:
        segments = segments[: args.limit_segments]
    if not segments:
        sys.exit("No sentences found in the transcript.")
    print(f"Model: {llm.model} | target: {target} | sentences: {len(segments)} | "
          f"stance: {config.stance_strategy} | relation window: {config.relation_window}")

    bar = Progress(len(segments), "pipeline")
    records = pipeline.run_segments(segments, target, lambda done, total: bar.update())
    bar.close()

    name = args.name or args.transcript.stem
    write_json(records, args.out_dir / f"{name}.json", meta={"target": target, **pipeline.describe()})
    write_csv(records, args.out_dir / f"{name}.csv")
    table = to_markdown(records)
    (args.out_dir / f"{name}.md").write_text(table + "\n", encoding="utf-8")

    print("\n" + table)
    print("\n" + format_summary(summarize(records)))
    print(f"\nSaved {args.out_dir / name}.json, .csv and .md")


if __name__ == "__main__":
    main()
