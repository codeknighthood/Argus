"""Phase 3: how well do two annotators agree? (Cohen's kappa)

Run from the project root:
    python scripts/annotator_agreement.py annotatorA.csv annotatorB.csv

    python scripts/annotator_agreement.py pairsA.csv pairsB.csv --column gold_relation
    python scripts/annotator_agreement.py errorsA.csv errorsB.csv --column error_type     (Phase 6: causes of mistakes)

Both files are copies of the same sheet from make_annotation_sheet.py, filled in independently.
Rows are matched on the 'index' column (components sheet), 'pair_id' (pairs sheet) or 'id' (errors sheet); rows either person
left blank are ignored.
Report this kappa (before you discuss the disagreements) as evidence that the labels can be trusted.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.annotation import compare_annotators, interpret_kappa  # noqa: E402
from argus.evaluation import read_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_a", type=Path)
    ap.add_argument("file_b", type=Path)
    ap.add_argument("--column", default="gold_component", choices=["gold_component", "gold_relation", "error_type"])
    args = ap.parse_args()

    try:
        result = compare_annotators(read_csv(args.file_a), read_csv(args.file_b), args.column)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Could not compare the sheets: {e}")     # a plain message, not a traceback
    if result["n"] == 0:
        sys.exit("No rows were annotated by both people. Check the id column and the column you are comparing.")
    print(f"Rows annotated by both: {result['n']}")
    print(f"Raw agreement:          {result['agreement']:.1%}")
    print(f"Cohen's kappa:          {result['kappa']:.3f} ({interpret_kappa(result['kappa'])})")
    if result["disagreements"]:
        print(f"\nDisagreements ({len(result['disagreements'])}), to discuss:")
        for d in result["disagreements"][:30]:
            print(f"  [{d['index']}] A: {d['a']:<8} B: {d['b']:<8} | {d['sentence'][:80]}")


if __name__ == "__main__":
    main()
