"""Look at the raw dataset before processing it.

Run from the project root:   python scripts/inspect_data.py [path]
Default path: data/raw/vast

Prints the files found, their columns, a few example rows, label counts and how long the texts are. If loading ever
fails, copy this output: it shows exactly what your copy of the data looks like.
(A folder of UKP .tsv files is also understood.)
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.data import VAST_KINDS, VAST_LABELS, find_ukp_files, find_vast_files  # noqa: E402


def inspect_vast(files: dict) -> None:
    csv.field_size_limit(10 ** 9)
    for split, f in files.items():
        with open(f, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            columns = reader.fieldnames
            rows = list(reader)
        print(f"\n--- {f.name} ({split}) ---\ncolumns: {columns}\nrows: {len(rows)}")
        label = Counter(VAST_LABELS[r["label"]].value if r.get("label") in VAST_LABELS else f"?{r.get('label')}" for r in rows)
        print("labels:", dict(label))
        print("how the topic was matched:", dict(Counter(VAST_KINDS.get(r.get("type_idx"), "?") for r in rows)))
        words = sorted(len((r.get("post") or "").split()) for r in rows)
        if words:
            print(f"words per comment: median {words[len(words) // 2]}, longest {words[-1]}; "
                  f"distinct topics: {len({r.get('new_topic') for r in rows})}")
        for r in rows[:2]:
            print({"topic": r.get("new_topic"), "label": r.get("label"), "post": (r.get("post") or "")[:90] + "..."})


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "vast"
    if not path.exists():
        print(f"{path} does not exist. Run:  python scripts/download_data.py")
        return
    vast = find_vast_files(path)
    if vast:
        print(f"Looking in: {path}\nFound VAST file(s): {', '.join(f.name for f in vast.values())}")
        inspect_vast(vast)
        return
    files = find_ukp_files(path)
    print(f"Looking in: {path}\nNo VAST files; found {len(files)} .tsv file(s) (UKP format):")
    for f in files:
        print("  ", f.relative_to(path) if path.is_dir() else f.name)
    if not files:
        print("Nothing recognisable here. For VAST run:  python scripts/download_data.py")
        return
    with open(files[0], "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = list(reader)
    print("columns:", reader.fieldnames, "| rows:", len(rows))


if __name__ == "__main__":
    main()
