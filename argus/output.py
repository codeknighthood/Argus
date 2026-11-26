"""Phase 5: writing the structured output of synopsis 5.7 as JSON, CSV and a readable table."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .evaluation import write_csv as _write_csv
from .evaluation import write_json as _write_json
from .schema import ArgumentRecord

# The nine columns listed in synopsis 5.7 come first, then bookkeeping columns.
SYNOPSIS_COLUMNS = [
    "policy_target", "original_statement", "identified_argument", "component", "relation",
    "related_to", "stance", "explanation", "speaker_context",
]
EXTRA_COLUMNS = ["segment_index", "is_argument", "related_to_index", "relation_explanation", "error"]
COLUMNS = SYNOPSIS_COLUMNS + EXTRA_COLUMNS


def record_to_row(rec: ArgumentRecord) -> dict:
    """Plain dict with enums as their text values and missing values as empty strings (for CSV and tables)."""
    data = rec.model_dump(mode="json")
    return {c: ("" if data.get(c) is None else data[c]) for c in COLUMNS}


def records_to_rows(records: Sequence[ArgumentRecord]) -> list[dict]:
    return [record_to_row(r) for r in records]


def write_json(records: Sequence[ArgumentRecord], path: Path, meta: Optional[dict] = None) -> None:
    """{"meta": settings and time, "records": [...]}. Missing values stay null in JSON."""
    payload = {
        "meta": {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(meta or {})},
        "records": [r.model_dump(mode="json") for r in records],
    }
    _write_json(payload, path)


def write_csv(records: Sequence[ArgumentRecord], path: Path) -> None:
    _write_csv(records_to_rows(records), path, COLUMNS)


def _cell(text: object, limit: int) -> str:
    s = str(text).replace("|", "/").replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 3].rstrip() + "..."


def to_markdown(records: Sequence[ArgumentRecord], width: int = 60) -> str:
    """A compact table: sentence number, speaker, statement, role, relation (with the sentence it points at), stance."""
    lines = ["| # | Speaker | Statement | Role | Relation | Stance |", "|---|---|---|---|---|---|"]
    for r in records:
        row = record_to_row(r)
        relation = row["relation"] or "-"
        if row["related_to_index"] != "":
            relation += f" -> #{row['related_to_index']}"
        lines.append("| " + " | ".join([
            str(row["segment_index"]), _cell(row["speaker_context"] or "-", 20), _cell(row["original_statement"], width),
            row["component"] or "-", relation, row["stance"] or "-",
        ]) + " |")
    return "\n".join(lines)


def summarize(records: Sequence[ArgumentRecord]) -> dict:
    """Headline counts for a finished run."""
    args = [r for r in records if r.is_argument]
    return {
        "sentences": len(records),
        "arguments": len(args),
        "claims": sum(1 for r in records if r.component and r.component.value == "Claim"),
        "premises": sum(1 for r in records if r.component and r.component.value == "Premise"),
        "stance": dict(Counter(r.stance.value for r in records if r.stance)),
        "relations": dict(Counter(r.relation.value for r in args if r.relation)),
        "speakers": dict(Counter(r.speaker_context for r in records if r.speaker_context)),
        "rows_with_errors": sum(1 for r in records if r.error),
    }


def format_summary(summary: dict) -> str:
    def counts(d: dict) -> str:
        return ", ".join(f"{k} {v}" for k, v in d.items()) or "none"

    lines = [
        f"Sentences analysed : {summary['sentences']}",
        f"Arguments found    : {summary['arguments']} ({summary['claims']} claims, {summary['premises']} premises)",
        f"Stance             : {counts(summary['stance'])}",
        f"Relations          : {counts(summary['relations'])}",
    ]
    if summary["rows_with_errors"]:
        lines.append(f"Rows with errors   : {summary['rows_with_errors']} (see the 'error' column)")
    return "\n".join(lines)
