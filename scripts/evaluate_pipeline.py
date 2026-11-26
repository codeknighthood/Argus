"""Phase 6: score the WHOLE pipeline against a hand-annotated transcript.

Run from the project root, after filling data/annotation/components.csv and/or pairs.csv (Phases 3 and 4):
    python scripts/evaluate_pipeline.py                                   # the sample transcript
    python scripts/evaluate_pipeline.py --transcript my_debate.txt --target "the minimum wage"

Phases 3 and 4 scored extraction and relations one stage at a time, on the sentences exactly as written in the sheet.
Here the full pipeline runs on the transcript first (segmentation, extraction, then relations), and its output is
scored against your annotations. Mistakes can pile up across stages: a missed argument hides a link, and a link
to the wrong argument hides the right one. For every pair you marked Support or Attack that the pipeline missed,
the report says why (argument not detected, outside the relation window, linked to a nearer argument ...).

Stance is skipped by default (a transcript has no stance labels; use scripts/evaluate.py for stance); add
--with-stance to run it anyway. Output files, in --out-dir (default results/evaluation):
    pipeline_eval.md / .json      the report and its numbers
    pipeline_components.jsonl     every sentence: your label, the pipeline's label
    pipeline_relations.jsonl      every pair: your label, the pipeline's label, and the outcome

The sheets must come from the same transcript and settings (--context-window, --min-tokens) as this run; the script
checks and says so if they do not match.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.annotation import load_component_sheet, load_pair_sheet  # noqa: E402
from argus.evaluation import Progress, write_json, write_jsonl  # noqa: E402
from argus.llm import load_config, make_ready_client  # noqa: E402
from argus.metrics import format_report  # noqa: E402
from argus.pipeline import ArgusPipeline, PipelineConfig  # noqa: E402
from argus.pipeline_eval import (  # noqa: E402
    check_sentences, evaluate_components, evaluate_relations, index_records, pipeline_report_md,
)
from argus.segment import segment_transcript  # noqa: E402
from argus.stance import STRATEGIES  # noqa: E402

SAMPLE = ROOT / "data" / "sample" / "sample_transcript.txt"


def load_if_annotated(path: Path, loader, column: str):
    """Rows of a filled sheet, or None if the file is missing or has nothing annotated (with a note on stderr)."""
    if not path.exists():
        return None
    try:
        rows, blank = loader(path)
    except ValueError as e:
        sys.exit(str(e))                                  # a plain message, not a traceback
    if not rows:
        print(f"Note: {path} has no annotated rows ({blank} blank). Fill '{column}' to score it.", file=sys.stderr)
        return None
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript", type=Path, default=SAMPLE)
    ap.add_argument("--target", default=None, help='the policy under debate, e.g. "the minimum wage"')
    ap.add_argument("--components", type=Path, default=ROOT / "data" / "annotation" / "components.csv")
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "annotation" / "pairs.csv")
    ap.add_argument("--relation-window", type=int, default=None)
    ap.add_argument("--context-window", type=int, default=None)
    ap.add_argument("--min-tokens", type=int, default=None)
    ap.add_argument("--strategy", choices=list(STRATEGIES), default=None)
    ap.add_argument("--with-stance", action="store_true", help="also run the stance stage (slower)")
    ap.add_argument("--no-speakers", action="store_true", help="the transcript has no 'NAME:' prefixes")
    ap.add_argument("--model", default=None, help="override the model in config.yaml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "evaluation")
    args = ap.parse_args()

    target = args.target
    if target is None:
        if args.transcript != SAMPLE:
            sys.exit('Please say what the debate is about, e.g.  --target "the minimum wage"')
        target = "the minimum wage"
    if not args.transcript.exists():
        sys.exit(f"Transcript not found: {args.transcript}")

    comp_rows = load_if_annotated(args.components, load_component_sheet, "gold_component")
    pair_rows = load_if_annotated(args.pairs, load_pair_sheet, "gold_relation")
    if comp_rows is None and pair_rows is None:
        sys.exit(f"Nothing to score: no annotated rows in {args.components} or {args.pairs}. Make the sheets with "
                 "scripts/make_annotation_sheet.py, fill the gold columns, and run again.")

    try:
        config = PipelineConfig.from_config(
            load_config(), stance_strategy=args.strategy, stance_on=None if args.with_stance else "none",
            relation_window=args.relation_window, context_window=args.context_window, min_tokens=args.min_tokens,
            parse_speakers=False if args.no_speakers else None,
        )
        if not args.with_stance:
            config.stance_on = "none"                      # even if config.yaml says otherwise
    except ValueError as e:
        sys.exit(str(e))

    text = args.transcript.read_text(encoding="utf-8")
    segments = segment_transcript(text, config.context_window, config.parse_speakers, config.min_tokens)
    if not segments:
        sys.exit("No sentences found in the transcript.")

    # The sheets must describe the same sentences the pipeline will see, or every score would be about the wrong rows.
    expected = []
    for r in comp_rows or []:
        expected.append((int(r["index"]), r["sentence"]))
    for r in pair_rows or []:
        expected += [(int(r["a_index"]), r["sentence_a"]), (int(r["b_index"]), r["sentence_b"])]
    seg_by_index = {s.index: s for s in segments}
    problems = []
    for index, sentence in expected:
        seg = seg_by_index.get(index)
        if seg is None:
            problems.append(f"sentence #{index} is not in the transcript as split with these settings")
        elif seg.sentence.strip() != sentence.strip():
            problems.append(f"sentence #{index}: sheet has {sentence.strip()[:45]!r} but the transcript gives "
                            f"{seg.sentence.strip()[:45]!r}")
    if problems:
        shown = "\n  ".join(dict.fromkeys(problems[:6]))
        sys.exit("The annotation sheets do not match this transcript and these settings:\n  " + shown +
                 "\nMake the sheets again from the same transcript, or pass the same --context-window and "
                 "--min-tokens that you used for make_annotation_sheet.py.")

    llm = make_ready_client(args.model)
    pipeline = ArgusPipeline(llm, config)
    print(f"Model: {llm.model} | target: {target} | sentences: {len(segments)} | "
          f"relation window: {config.relation_window} | stance: {'on' if args.with_stance else 'off'}")
    bar = Progress(len(segments), "pipeline")
    records = pipeline.run_segments(segments, target, lambda done, total: bar.update())
    bar.close()

    recs = index_records(records)
    for message in check_sentences(recs, expected):      # belt and braces: the pipeline output itself
        sys.exit(f"Pipeline output does not match the sheet: {message}")

    components = evaluate_components(recs, comp_rows) if comp_rows else None
    relations = evaluate_relations(recs, pair_rows, config.relation_window) if pair_rows else None
    meta = {"transcript": args.transcript.name, "sentences": len(segments), "target": target, "model": llm.model,
            "relation_window": config.relation_window, "date": date.today().isoformat()}

    if components:
        print("\nClaim / Premise / None\n" + format_report(components["report"]))
        print(f"Cohen's kappa: {components['kappa']:.3f}")
    if relations:
        print("\nSupport / Attack / None\n" + format_report(relations["report"]))
        print(f"Cohen's kappa: {relations['kappa']:.3f}")
        print(f"Pairs you marked Support or Attack: {relations['gold_links']}; found and labelled correctly: "
              f"{relations['links_found_correctly']}")
        for reason, n in relations["misses"].items():
            print(f"  {n:>3}  {reason}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_eval.md").write_text(pipeline_report_md(meta, components, relations), encoding="utf-8")
    write_json({"meta": meta,
                "components": components and {k: v for k, v in components.items() if k != "rows"},
                "relations": relations and {k: v for k, v in relations.items() if k != "rows"}}, out / "pipeline_eval.json")
    if components:
        write_jsonl(components["rows"], out / "pipeline_components.jsonl")
    if relations:
        write_jsonl(relations["rows"], out / "pipeline_relations.jsonl")
    print(f"\nFull report: {out / 'pipeline_eval.md'}")


if __name__ == "__main__":
    main()
