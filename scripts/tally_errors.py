"""Phase 6: count the causes you tagged in an errors_*.csv (docs/ERROR_ANALYSIS.md).

Run from the project root, after filling the error_type column of a file written by scripts/evaluate.py:
    python scripts/tally_errors.py results/evaluation/errors_qwen2.5_7b___few_shot_dev.csv
    python scripts/tally_errors.py <file> --out results/evaluation/causes.md

error_type accepts a number or a name:
    1 implicit argument   2 ambiguous statement   3 missing context   4 label noise
    5 negation or sarcasm   6 wrong target   7 other
Rows you left blank are counted as "not tagged yet" and left out of the percentages.
Also prints which cause goes with which shape of mistake (polarity flip, over-reading, under-reading).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.analysis import ERROR_TYPES, cause_by_shape, tally_error_types  # noqa: E402
from argus.evaluation import read_csv  # noqa: E402


def build_text(rows: list[dict], tally: dict) -> str:
    tagged, total = tally["tagged"], tally["total"]
    lines = [f"Tagged {tagged} of {total} mistakes ({tally['untagged']} not tagged yet).", "",
             "| Cause | Mistakes | Share |", "|---|---|---|"]
    order = sorted(tally["counts"].items(), key=lambda kv: (-kv[1], ERROR_TYPES.index(kv[0])))
    lines += [f"| {cause} | {n} | {100 * n / tagged:.0f}% |" for cause, n in order]
    lines += ["", f"Shares are out of the {tagged} tagged rows, not out of all {total}; the percentages describe the "
                  f"{tagged} tagged rows only. Tag a random sample, not just the easy cases, or they will mislead.", ""]
    table = cause_by_shape(rows)
    kinds = sorted({k for shapes in table.values() for k in shapes})
    if kinds:
        lines += ["Cause by shape of the mistake:", "", "| Cause | " + " | ".join(kinds) + " |",
                  "|---|" + "---|" * len(kinds)]
        for cause, _ in order:
            lines.append(f"| {cause} | " + " | ".join(str(table.get(cause, {}).get(k, 0)) for k in kinds) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, help="an errors_*.csv with the error_type column filled in")
    ap.add_argument("--out", type=Path, default=None, help="also save the result as a markdown file")
    args = ap.parse_args()

    if not args.file.exists():
        sys.exit(f"File not found: {args.file}")
    rows = read_csv(args.file)
    if rows and "error_type" not in rows[0]:
        sys.exit(f"{args.file} has no 'error_type' column. Columns found: {list(rows[0])}")
    try:
        tally = tally_error_types(rows)
    except ValueError as e:
        sys.exit(str(e))                                  # a plain message, not a traceback
    if tally["tagged"] == 0:
        sys.exit(f"No tagged rows in {args.file}. Fill the error_type column first (see docs/ERROR_ANALYSIS.md).")

    text = build_text(rows, tally)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
