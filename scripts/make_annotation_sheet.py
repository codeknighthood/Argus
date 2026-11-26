"""Turn a transcript into annotation sheets (Phase 3: claim/premise labels, Phase 4: relation labels).

Run from the project root:
    python scripts/make_annotation_sheet.py                       # both sheets, from the sample transcript
    python scripts/make_annotation_sheet.py --transcript my_debate.txt
    python scripts/make_annotation_sheet.py --kind pairs          # only the relation sheet

Two sheets are written to data/annotation/:
  components.csv  one row per sentence.      Fill `gold_component` with Claim, Premise or None.
  pairs.csv       one row per (A, B) pair.   Fill `gold_relation` with Support, Attack or None:
                  how does the LATER sentence B relate to the EARLIER sentence A?
Rules for both are in docs/ANNOTATION_GUIDE.md. Open in Excel, fill, save as CSV.

The transcript format is the usual one, one turn per line:   SPEAKER NAME: what they said

An existing sheet is never overwritten unless you pass --force, so your annotations are safe.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.annotation import COMPONENT_COLUMNS, PAIR_COLUMNS, components_sheet_rows, pairs_sheet_rows  # noqa: E402
from argus.evaluation import write_csv  # noqa: E402
from argus.segment import segment_transcript  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript", type=Path, default=ROOT / "data" / "sample" / "sample_transcript.txt")
    ap.add_argument("--kind", choices=["components", "pairs", "both"], default="both")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "annotation" / "components.csv")
    ap.add_argument("--pairs-out", type=Path, default=ROOT / "data" / "annotation" / "pairs.csv")
    ap.add_argument("--context-window", type=int, default=2, help="earlier sentences shown as context")
    ap.add_argument("--pair-window", type=int, default=3, help="each sentence is paired with this many before it")
    ap.add_argument("--min-tokens", type=int, default=3,
                    help="drop sentences shorter than this; keep it equal to the pipeline's min_tokens in config.yaml")
    ap.add_argument("--no-speakers", action="store_true", help="transcript has no 'NAME:' prefixes")
    ap.add_argument("--force", action="store_true", help="overwrite existing sheets (destroys annotations!)")
    args = ap.parse_args()

    wanted = []
    if args.kind in ("components", "both"):
        wanted.append(("components", args.out))
    if args.kind in ("pairs", "both"):
        wanted.append(("pairs", args.pairs_out))

    existing = [p for _, p in wanted if p.exists()]
    if existing and not args.force:        # check first, so nothing is half-written
        sys.exit("Already exists and may contain your annotations: " + ", ".join(str(p) for p in existing)
                 + "\nUse --force to overwrite, pick another file with --out / --pairs-out, "
                 "or choose just one sheet with --kind.")

    text = args.transcript.read_text(encoding="utf-8")
    segments = segment_transcript(text, context_window=args.context_window, parse_speakers=not args.no_speakers,
                                  min_tokens=args.min_tokens)

    for kind, path in wanted:
        if kind == "components":
            write_csv(components_sheet_rows(segments), path, COMPONENT_COLUMNS)
            print(f"components: {len(segments)} sentences -> {path}   (fill 'gold_component': Claim / Premise / None)")
        else:
            rows = pairs_sheet_rows(segments, args.pair_window)
            write_csv(rows, path, PAIR_COLUMNS)
            print(f"pairs:      {len(rows)} pairs     -> {path}   (fill 'gold_relation': Support / Attack / None)")


if __name__ == "__main__":
    main()
