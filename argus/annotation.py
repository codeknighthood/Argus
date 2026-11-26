"""Hand-annotation support for claim / premise labels (the UKP corpus has none).

Workflow: make a sheet from a transcript, fill in `gold_component` in Excel, score the LLM against it.
With two annotators, compare their sheets with Cohen's kappa (synopsis 5.8 asks for reliable evaluation
data; kappa is how you show the labels can be trusted). The labelling rules are in docs/ANNOTATION_GUIDE.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from .evaluation import read_csv
from .metrics import cohen_kappa
from .segment import Segment

COMPONENT_LABELS = ["Claim", "Premise", "None"]
COMPONENT_COLUMNS = ["index", "speaker", "sentence", "context", "gold_component"]

RELATION_LABELS = ["Support", "Attack", "None"]
PAIR_COLUMNS = ["pair_id", "a_index", "b_index", "speaker_a", "speaker_b", "sentence_a", "sentence_b", "gold_relation"]

_COMPONENT_ALIASES = {
    "claim": "Claim", "c": "Claim",
    "premise": "Premise", "p": "Premise",
    "none": "None", "n": "None", "no": "None", "non-argument": "None", "noargument": "None",
}
_RELATION_ALIASES = {
    "support": "Support", "supports": "Support", "s": "Support", "pro": "Support",
    "attack": "Attack", "attacks": "Attack", "a": "Attack", "rebut": "Attack", "con": "Attack",
    "none": "None", "n": "None", "no": "None", "unrelated": "None",
}


def _normalise(value: Optional[str], aliases: dict, allowed: str) -> Optional[str]:
    v = (value or "").strip().lower()
    if not v:
        return None
    if v not in aliases:
        raise ValueError(f"{value!r} is not one of {allowed}")
    return aliases[v]


def normalise_component(value: Optional[str]) -> Optional[str]:
    """'claim', ' Claim ', 'C' -> 'Claim'. Blank -> None (not yet annotated). Anything else is an error."""
    return _normalise(value, _COMPONENT_ALIASES, "Claim / Premise / None")


def normalise_relation(value: Optional[str]) -> Optional[str]:
    """'support', ' Attack ', 'S' -> 'Support' / 'Attack'. Blank -> None (not yet annotated)."""
    return _normalise(value, _RELATION_ALIASES, "Support / Attack / None")


def components_sheet_rows(segments: Sequence[Segment]) -> list[dict]:
    return [
        {"index": s.index, "speaker": s.speaker or "", "sentence": s.sentence,
         "context": s.context, "gold_component": ""}
        for s in segments
    ]


def pairs_sheet_rows(segments: Sequence[Segment], window: int = 3) -> list[dict]:
    """Every sentence B paired with each of the `window` sentences before it (A), nearest first.

    The annotator answers one question per row: how does B relate to A (Support / Attack / None)?
    """
    rows = []
    for b in segments:
        for back in range(1, window + 1):
            a_pos = b.index - back
            if a_pos < 0:
                break
            a = segments[a_pos]
            rows.append({
                "pair_id": f"{a.index}-{b.index}", "a_index": a.index, "b_index": b.index,
                "speaker_a": a.speaker or "", "speaker_b": b.speaker or "",
                "sentence_a": a.sentence, "sentence_b": b.sentence, "gold_relation": "",
            })
    return rows


def _load_sheet(
    path: Path, column: str, normaliser: Callable[[Optional[str]], Optional[str]]
) -> tuple[list[dict], int]:
    raw = read_csv(Path(path))
    if raw and column not in raw[0]:
        raise ValueError(f"{path} has no '{column}' column. Columns found: {list(raw[0])}")
    rows, blank, problems = [], 0, []
    for line_no, row in enumerate(raw, start=2):          # row 1 is the header
        try:
            gold = normaliser(row.get(column))
        except ValueError as e:
            problems.append(f"row {line_no}: {e}")
            continue
        if gold is None:
            blank += 1
            continue
        rows.append({**row, "gold": gold})
    if problems:
        raise ValueError("Fix these cells in the sheet and run again:\n  " + "\n  ".join(problems))
    return rows, blank


def load_component_sheet(path: Path) -> tuple[list[dict], int]:
    """Read a filled components sheet. Returns (annotated rows with a normalised `gold`, blank rows skipped).

    Every bad cell is reported at once, with its spreadsheet row number, so you can fix them in one pass.
    """
    return _load_sheet(path, "gold_component", normalise_component)


def load_pair_sheet(path: Path) -> tuple[list[dict], int]:
    """Read a filled pairs sheet. Same behaviour as load_component_sheet, for `gold_relation`."""
    return _load_sheet(path, "gold_relation", normalise_relation)


def compare_annotators(
    rows_a: Sequence[dict], rows_b: Sequence[dict], column: str = "gold_component", key: Optional[str] = None,
) -> dict:
    """Agreement between two annotators' sheets. Blank cells are ignored.

    Rows are matched on `key`: 'index' for the components sheet, 'pair_id' for the pairs sheet
    (chosen automatically from the column name unless given).
    """
    if column == "error_type":                       # Phase 6: causes of mistakes, tagged in errors_*.csv
        from .analysis import normalise_error_type as normaliser
        key = key or "id"
    else:
        normaliser = normalise_relation if column == "gold_relation" else normalise_component
        key = key or ("pair_id" if column == "gold_relation" else "index")

    def labels(rows):
        return {str(r[key]): normaliser(r.get(column)) for r in rows}

    a, b = labels(rows_a), labels(rows_b)
    shared = [i for i in a if i in b and a[i] is not None and b[i] is not None]
    text = {str(r[key]): r.get("sentence") or r.get("statement") or f"{r.get('sentence_a', '')} -> {r.get('sentence_b', '')}"
            for r in rows_a}
    la, lb = [a[i] for i in shared], [b[i] for i in shared]
    return {
        "n": len(shared),
        "agreement": sum(x == y for x, y in zip(la, lb)) / len(shared) if shared else 0.0,
        "kappa": cohen_kappa(la, lb),
        "disagreements": [{"index": i, "a": a[i], "b": b[i], "sentence": text.get(i, "")}
                          for i in shared if a[i] != b[i]],
    }


def interpret_kappa(k: float) -> str:
    """Common verbal scale (Landis and Koch, 1977)."""
    if k < 0:
        return "worse than chance"
    for limit, name in [(0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")]:
        if k <= limit:
            return name
    return "almost perfect"
