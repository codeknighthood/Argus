"""Phase 7: what the demo shows, kept separate from the screen code so it can be tested without a browser.

The pipeline returns one record per sentence. For reading a long debate quickly, three views are built from them:

* condensed view   only the sentences that carry an argument, grouped by stance, with repeated points merged.
                   This is the "less redundant conversation" the project is after: greetings, procedure and filler
                   are dropped, and several speakers making the same point appear once, with a count.
* argument map     a graph of the arguments with green (Support) and red (Attack) arrows between them.
* full table       every sentence with everything the pipeline decided (synopsis 5.7).
"""
from __future__ import annotations

import re
import textwrap
from typing import Sequence

from .output import records_to_rows
from .schema import ArgumentRecord

STANCE_ORDER = ["Favor", "Against", "Neutral", "Not assessed"]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def similarity(a: str, b: str) -> float:
    """Share of words two texts have in common (Jaccard), 0 to 1. Crude on purpose: it only has to spot near-repeats."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def argument_rows(records: Sequence[ArgumentRecord]) -> list[dict]:
    """One simple dict per sentence that the pipeline judged to be an argument."""
    rows = []
    for r in records:
        if not r.is_argument:
            continue
        rows.append({
            "index": r.segment_index, "speaker": r.speaker_context or "", "argument": r.identified_argument or r.original_statement,
            "statement": r.original_statement, "component": r.component.value if r.component else "",
            "stance": r.stance.value if r.stance else "Not assessed", "explanation": r.explanation or "",
            "relation": r.relation.value if r.relation else "", "related_to_index": r.related_to_index,
            "repeats": 1, "also_said_by": [],
        })
    return rows


def merge_repeats(rows: Sequence[dict], threshold: float = 0.7) -> list[dict]:
    """Merge arguments that say nearly the same thing with the same stance. The first one is kept.

    `repeats` counts how many sentences were merged (including the kept one) and `also_said_by` lists the other
    speakers. Two arguments are merged when they share at least `threshold` of their words.
    """
    kept: list[dict] = []
    for row in rows:
        for k in kept:
            if k["stance"] == row["stance"] and similarity(k["argument"], row["argument"]) >= threshold:
                k["repeats"] += 1
                if row["speaker"] and row["speaker"] != k["speaker"] and row["speaker"] not in k["also_said_by"]:
                    k["also_said_by"].append(row["speaker"])
                break
        else:
            kept.append({**row, "also_said_by": list(row["also_said_by"])})
    return kept


def condensed_view(records: Sequence[ArgumentRecord], merge: bool = True, threshold: float = 0.7) -> dict:
    """{'groups': {stance: [rows]}, 'sentences': n, 'arguments': n, 'shown': n, 'dropped': n}."""
    rows = argument_rows(records)
    shown = merge_repeats(rows, threshold) if merge else rows
    groups: dict[str, list[dict]] = {s: [] for s in STANCE_ORDER}
    for row in shown:
        groups.setdefault(row["stance"], []).append(row)
    return {
        "groups": {s: g for s, g in groups.items() if g},
        "sentences": len(records), "arguments": len(rows), "shown": len(shown),
        "dropped": len(records) - len(rows),
    }


_RELATION_COLOUR = {"Support": "#2e7d32", "Attack": "#c62828"}
_STANCE_FILL = {"Favor": "#e8f5e9", "Against": "#ffebee", "Neutral": "#eeeeee", "Not assessed": "#ffffff"}


def _dot_text(text: str, width: int = 34, limit: int = 110) -> str:
    text = text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
    return "\\n".join(textwrap.wrap(text.replace("\\", " ").replace('"', "'"), width)) or "-"


def argument_map_dot(records: Sequence[ArgumentRecord]) -> str:
    """Graphviz DOT source: a box per argument (coloured by stance) and an arrow from each argument to the earlier
    one it supports (green) or attacks (red). Arguments with no link are still drawn, so nothing disappears."""
    rows = argument_rows(records)
    lines = ["digraph arguments {", "  rankdir=LR;", '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
             '  edge [fontname="Helvetica", fontsize=9];']
    ids = {r["index"] for r in rows}
    for r in rows:
        who = f"{r['speaker']}: " if r["speaker"] else ""
        label = _dot_text(f"#{r['index']} {who}{r['argument']}")
        lines.append(f'  n{r["index"]} [label="{label}", fillcolor="{_STANCE_FILL.get(r["stance"], "#ffffff")}"];')
    for r in rows:
        if r["relation"] in _RELATION_COLOUR and r["related_to_index"] in ids:
            lines.append(f'  n{r["index"]} -> n{r["related_to_index"]} [label="{r["relation"]}", '
                         f'color="{_RELATION_COLOUR[r["relation"]]}", fontcolor="{_RELATION_COLOUR[r["relation"]]}"];')
    lines.append("}")
    return "\n".join(lines)


def table_rows(records: Sequence[ArgumentRecord], as_text: bool = False) -> list[dict]:
    """The full structured output (synopsis 5.7) as simple dicts, for a table or a download.

    `as_text=True` turns every cell into text, which is what an on-screen table wants: a column that mixes blanks and
    numbers (like related_to_index) otherwise cannot be shown without the table library repairing it."""
    rows = records_to_rows(records)
    return [{k: str(v) for k, v in row.items()} for row in rows] if as_text else rows
