"""Phase 6: error analysis (synopsis 5.8, "error analysis" and "reliability").

Two kinds of help, kept apart on purpose:

* AUTOMATIC CUES (`auto_tags`, `tag_error_rates`, `error_kind`). Cheap surface features of a sentence (has a
  negation, has a number, is a question ...) and the SHAPE of a mistake (polarity flip, over-reading ...).
  They show where mistakes cluster, but they do not say why a mistake happened.
* HUMAN CAUSES (`ERROR_TYPES`, `tally_error_types`). You read the mistakes and tag each with a cause. That is the
  real error analysis; the procedure is in docs/ERROR_ANALYSIS.md.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Optional, Sequence

from .metrics import cohen_kappa

# --------------------------------------------------------------------------- automatic cues
_WORD = re.compile(r"[A-Za-z0-9']+")
_PATTERNS = {
    "negation": re.compile(r"\b(not|no|never|nobody|nothing|none|neither|nor|cannot|without)\b|n't\b", re.I),
    "hedge": re.compile(r"\b(perhaps|maybe|might|may|could|possibly|probably|somewhat|seems?|appears?|arguably|likely)\b", re.I),
    "contrast": re.compile(r"\b(but|however|although|though|yet|whereas|while|nevertheless|despite)\b", re.I),
    "normative": re.compile(r"\b(should|ought|must|need to|needs to|have to|better|worse)\b", re.I),
    "first_person": re.compile(r"\b(i|we|my|our|me|us|i'm|i've|we've)\b", re.I),
    "number": re.compile(r"\d"),
    "question": re.compile(r"\?"),
}
_PRONOUN_STARTS = {"it", "this", "that", "they", "these", "those", "he", "she", "which", "such"}
SHORT_BELOW = 6          # fewer words than this = "short"
LONG_FROM = 35           # this many words or more = "long"

TAG_DESCRIPTIONS = {
    "negation": "contains not / no / never / n't ...",
    "hedge": "contains perhaps / might / could / probably ...",
    "contrast": "contains but / however / although ...",
    "normative": "contains should / must / ought ...",
    "first_person": "speaks as I / we",
    "number": "contains a digit",
    "pronoun_start": "starts with it / this / that / they ... (needs the sentence before it)",
    "question": "contains a question mark",
    "short": f"fewer than {SHORT_BELOW} words",
    "long": f"{LONG_FROM} or more words",
}


def auto_tags(statement: str) -> list[str]:
    """Surface features of a sentence, alphabetically. Cheap cues only: they say nothing about meaning."""
    text = statement or ""
    words = _WORD.findall(text)
    tags = {name for name, pat in _PATTERNS.items() if pat.search(text)}
    if words and words[0].lower() in _PRONOUN_STARTS:
        tags.add("pronoun_start")
    if len(words) < SHORT_BELOW:
        tags.add("short")
    if len(words) >= LONG_FROM:
        tags.add("long")
    return sorted(tags)


ERROR_KINDS = ["polarity flip", "over-reading", "under-reading", "failed call", "other"]


def error_kind(gold: Optional[str], pred: Optional[str]) -> Optional[str]:
    """The SHAPE of a stance mistake, or None if the prediction was right.

    polarity flip  Favor called Against or the reverse: the model read the direction wrongly.
    over-reading   gold Neutral but the model found a side: it read a stance into a neutral sentence.
    under-reading  gold has a side but the model said Neutral: it missed the stance.
    """
    if gold == pred:
        return None
    if pred in (None, "ERROR"):
        return "failed call"
    if {gold, pred} == {"Favor", "Against"}:
        return "polarity flip"
    if gold == "Neutral":
        return "over-reading"
    if pred == "Neutral":
        return "under-reading"
    return "other"


def tag_error_rates(rows: Sequence[dict], min_n: int = 5) -> list[dict]:
    """For each surface cue: how often are sentences WITH the cue wrong, compared with sentences without it?

    Cues with fewer than `min_n` sentences on either side are left out (too few to say anything). Sorted by the
    size of the gap, biggest first. A gap is a lead to read the mistakes, not a finding on its own.
    """
    tagged = [(set(auto_tags(r["statement"])), r["gold"] != r["pred"]) for r in rows]
    out = []
    for tag in TAG_DESCRIPTIONS:
        with_ = [wrong for tags, wrong in tagged if tag in tags]
        without = [wrong for tags, wrong in tagged if tag not in tags]
        if len(with_) < min_n or len(without) < min_n:
            continue
        rw, ro = sum(with_) / len(with_), sum(without) / len(without)
        out.append({"tag": tag, "description": TAG_DESCRIPTIONS[tag], "n_with": len(with_), "error_rate_with": rw,
                    "n_without": len(without), "error_rate_without": ro, "difference": rw - ro})
    return sorted(out, key=lambda t: (-t["difference"], t["tag"]))


def top_confusions(rows: Sequence[dict], k: int = 5) -> list[tuple[str, str, int]]:
    """The most common (gold, predicted) mistakes, ignoring failed calls. Biggest first."""
    counts = Counter((r["gold"], r["pred"]) for r in rows if r["gold"] != r["pred"] and r["pred"] not in (None, "ERROR"))
    return [(g, p, n) for (g, p), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def count_error_kinds(rows: Sequence[dict]) -> dict[str, int]:
    counts = Counter(error_kind(r["gold"], r["pred"]) for r in rows)
    counts.pop(None, None)
    return {kind: counts[kind] for kind in ERROR_KINDS if counts.get(kind)}


# --------------------------------------------------------------------------- agreement between runs
def _by_id(runs_rows: dict[str, Sequence[dict]]) -> tuple[list[str], dict[str, dict[str, dict]]]:
    """Rows of every run keyed by sentence id, restricted to ids that all runs have, in the first run's order."""
    names = list(runs_rows)
    keyed = {n: {r["id"]: r for r in runs_rows[n]} for n in names}
    if not names:
        return [], keyed
    ids = [r["id"] for r in runs_rows[names[0]] if all(r["id"] in keyed[n] for n in names)]
    return ids, keyed


def agreement_breakdown(runs_rows: dict[str, Sequence[dict]]) -> dict:
    """How the runs split on each sentence: all right, all wrong, or disagreeing (some right, some wrong)."""
    ids, keyed = _by_id(runs_rows)
    result = {"n": len(ids), "unanimous_correct": 0, "unanimous_wrong": 0, "split": 0}
    for i in ids:
        right = [keyed[n][i]["gold"] == keyed[n][i]["pred"] for n in keyed]
        if all(right):
            result["unanimous_correct"] += 1
        elif not any(right):
            result["unanimous_wrong"] += 1
        else:
            result["split"] += 1
    return result


def pairwise_kappa(runs_rows: dict[str, Sequence[dict]]) -> list[dict]:
    """Cohen's kappa between the predictions of every pair of runs (same sentences, matched by id)."""
    ids, keyed = _by_id(runs_rows)
    names = list(runs_rows)
    out = []
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a = [keyed[names[x]][i]["pred"] for i in ids]
            b = [keyed[names[y]][i]["pred"] for i in ids]
            out.append({"a": names[x], "b": names[y], "n": len(ids), "kappa": cohen_kappa(a, b),
                        "agree": sum(p == q for p, q in zip(a, b)) / len(ids) if ids else 0.0})
    return out


# --------------------------------------------------------------------------- human-tagged causes
ERROR_TYPES = [
    "implicit argument",       # 1 the stance is only implied; it needs inference the sentence does not state
    "ambiguous statement",     # 2 reasonable readers could disagree about the stance
    "missing context",         # 3 the sentence depends on the sentences around it, which the model did not see or use
    "label noise",             # 4 the gold label looks wrong; the model's answer is defensible
    "negation or sarcasm",     # 5 a negation, irony or sarcasm flipped the meaning
    "wrong target",            # 6 the sentence takes a side on something else (a related policy, a person)
    "other",                   # 7 none of the above
]
_ERROR_TYPE_ALIASES = {
    "implicit": "implicit argument", "ambiguous": "ambiguous statement", "context": "missing context",
    "missing": "missing context", "noise": "label noise", "label": "label noise", "negation": "negation or sarcasm",
    "sarcasm": "negation or sarcasm", "irony": "negation or sarcasm", "target": "wrong target",
    "wrong": "wrong target",
}


def normalise_error_type(value: Optional[str]) -> Optional[str]:
    """'3', 'missing context', 'Implicit' -> the cause name. Blank -> None (not tagged yet). Anything else is an error."""
    v = (value or "").strip().lower()
    if not v:
        return None
    if v.isdigit() and 1 <= int(v) <= len(ERROR_TYPES):
        return ERROR_TYPES[int(v) - 1]
    if v in ERROR_TYPES:
        return v
    if v in _ERROR_TYPE_ALIASES:
        return _ERROR_TYPE_ALIASES[v]
    legend = ", ".join(f"{i} = {name}" for i, name in enumerate(ERROR_TYPES, start=1))
    raise ValueError(f"{value!r} is not a cause. Use a number or a name: {legend}")


def tally_error_types(rows: Sequence[dict], column: str = "error_type") -> dict:
    """Count the tagged causes. Every bad cell is reported at once with its spreadsheet row number."""
    counts: Counter = Counter()
    problems, untagged = [], 0
    for line_no, row in enumerate(rows, start=2):               # row 1 of the sheet is the header
        try:
            cause = normalise_error_type(row.get(column))
        except ValueError as e:
            problems.append(f"row {line_no}: {e}")
            continue
        if cause is None:
            untagged += 1
        else:
            counts[cause] += 1
    if problems:
        raise ValueError("Fix these cells in the sheet and run again:\n  " + "\n  ".join(problems))
    return {"counts": {t: counts[t] for t in ERROR_TYPES if counts.get(t)}, "tagged": sum(counts.values()),
            "untagged": untagged, "total": len(rows)}


def cause_by_shape(rows: Sequence[dict], column: str = "error_type") -> dict[str, dict[str, int]]:
    """Cross-tab of tagged cause against the shape of the mistake (needs an `error_kind` column)."""
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        cause = normalise_error_type(row.get(column))
        if cause:
            table[cause][row.get("error_kind") or "other"] += 1
    return {c: dict(k) for c, k in table.items()}
