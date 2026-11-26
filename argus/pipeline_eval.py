"""Phase 6: score the full pipeline against a hand-annotated transcript (synopsis 5.8).

Phases 3 and 4 scored extraction and relations on their own, on clean inputs. Here the stages run together, so
mistakes can pile up: a missed argument can hide a link, and a link to the wrong argument can hide the right one.
For relations this module also records WHY each link was missed, which is the most useful part of the analysis.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from .annotation import COMPONENT_LABELS, RELATION_LABELS, interpret_kappa
from .metrics import classification_report, cohen_kappa, format_report
from .schema import ArgumentRecord


def index_records(records: Sequence[ArgumentRecord]) -> dict[int, ArgumentRecord]:
    return {r.segment_index: r for r in records}


def check_sentences(recs: dict[int, ArgumentRecord], expected: Sequence[tuple[int, str]]) -> list[str]:
    """Do the sheet's sentence numbers and texts match the pipeline's? If not, the sheet was made with other settings."""
    problems = []
    for index, text in expected:
        rec = recs.get(index)
        if rec is None:
            problems.append(f"sentence #{index} is not in the pipeline output")
        elif rec.original_statement.strip() != text.strip():
            problems.append(f"sentence #{index}: sheet has {text.strip()[:45]!r} but the pipeline has "
                            f"{rec.original_statement.strip()[:45]!r}")
    return problems


# --------------------------------------------------------------------------- components
def component_of(rec: Optional[ArgumentRecord]) -> str:
    if rec is None or rec.is_argument is None:
        return "ERROR"                          # no sentence, or the extraction call failed
    return rec.component.value if rec.component else "None"


def evaluate_components(recs: dict[int, ArgumentRecord], sheet_rows: Sequence[dict]) -> dict:
    rows = [{"index": int(r["index"]), "speaker": r.get("speaker", ""), "sentence": r["sentence"],
             "gold": r["gold"], "pred": component_of(recs.get(int(r["index"])))} for r in sheet_rows]
    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    return {"rows": rows, "report": classification_report(gold, pred, COMPONENT_LABELS),
            "kappa": cohen_kappa(gold, pred)}


# --------------------------------------------------------------------------- relations
def predict_pair(a: int, b: int, recs: dict[int, ArgumentRecord], relation_window: int) -> tuple[str, str]:
    """What the pipeline concluded about the pair (A earlier, B later), and the reason for that conclusion.

    The pipeline links each argument to ONE earlier argument, so for a pair it either says Support or Attack
    (B was linked to A) or None. The reason says why it said None when that is the case.
    """
    ra, rb = recs.get(a), recs.get(b)
    if ra is None or rb is None:
        return "None", "sentence missing from the pipeline output"
    if ra.is_argument is None or rb.is_argument is None:
        return "None", "extraction failed for A or B"
    if not rb.is_argument:
        return "None", "B was not detected as an argument"
    if not ra.is_argument:
        return "None", "A was not detected as an argument"
    if rb.related_to_index == a and rb.relation is not None:
        return rb.relation.value, "linked"
    if relation_window <= 0:
        return "None", "relation stage switched off"
    between = sum(1 for i, r in recs.items() if a < i < b and r.is_argument)
    if between >= relation_window:
        return "None", "A is outside the relation window"
    if rb.related_to_index is not None:
        return "None", "B was linked to a different argument (nearest-first rule)"
    if rb.error and "relation" in rb.error:
        return "None", "relation stage failed"
    return "None", "the model judged B unrelated to A"


def evaluate_relations(recs: dict[int, ArgumentRecord], pair_rows: Sequence[dict], relation_window: int) -> dict:
    rows = []
    for p in pair_rows:
        a, b = int(p["a_index"]), int(p["b_index"])
        pred, reason = predict_pair(a, b, recs, relation_window)
        gold = p["gold"]
        if pred == gold:
            outcome = "correct"
        elif gold == "None":                    # pred is Support or Attack here: the pipeline linked the pair
            outcome = "linked a pair the annotator marked None"
        elif reason == "linked":
            outcome = "linked, but labelled differently from the annotator"
        else:
            outcome = reason
        rows.append({"pair_id": p.get("pair_id"), "speaker_a": p.get("speaker_a", ""), "speaker_b": p.get("speaker_b", ""),
                     "sentence_a": p["sentence_a"], "sentence_b": p["sentence_b"], "gold": gold, "pred": pred,
                     "outcome": outcome})
    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    links = [r for r in rows if r["gold"] in ("Support", "Attack")]
    return {
        "rows": rows,
        "report": classification_report(gold, pred, RELATION_LABELS),
        "kappa": cohen_kappa(gold, pred),
        "misses": dict(Counter(r["outcome"] for r in rows if r["gold"] != r["pred"]).most_common()),
        "gold_links": len(links),
        "links_found_correctly": sum(r["gold"] == r["pred"] for r in links),
    }


# --------------------------------------------------------------------------- report
def pipeline_report_md(meta: dict, components: Optional[dict], relations: Optional[dict]) -> str:
    out = [
        "# ARGUS pipeline evaluation (annotated transcript)", "",
        f"- Transcript: {meta['transcript']} ({meta['sentences']} sentences)",
        f"- Target: {meta['target']}",
        f"- Model: {meta['model']} | relation window: {meta['relation_window']} | date: {meta['date']}",
        "- The whole pipeline ran first; its output is scored against your annotations.", "",
    ]
    if components:
        rep = components["report"]
        out += ["## Claim / Premise / None", "", "```", format_report(rep), "```", "",
                f"Cohen's kappa, pipeline vs your annotation: {components['kappa']:.3f} "
                f"({interpret_kappa(components['kappa'])}). Annotated sentences: {rep['n']}.", ""]
    if relations:
        rep = relations["report"]
        out += ["## Support / Attack / None between pairs of sentences", "", "```", format_report(rep), "```", "",
                f"Cohen's kappa, pipeline vs your annotation: {relations['kappa']:.3f} "
                f"({interpret_kappa(relations['kappa'])}). Annotated pairs: {rep['n']}.", ""]
        if relations["gold_links"]:
            out += [f"Of the {relations['gold_links']} pairs you marked Support or Attack, the pipeline found and "
                    f"labelled {relations['links_found_correctly']} correctly "
                    f"({100 * relations['links_found_correctly'] / relations['gold_links']:.0f}%).", ""]
        if relations["misses"]:
            out += ["### Why pairs were missed or mislabelled", "",
                    "| Reason | Pairs |", "|---|---|"] + [f"| {k} | {v} |" for k, v in relations["misses"].items()]
            out += ["", "This breakdown separates mistakes in earlier stages (an argument was not detected) from "
                        "limits of the linking rule (the pipeline links an argument to only the nearest earlier "
                        "argument it connects to, and only looks back a few arguments), and from plain "
                        "disagreements between the model and you.", ""]
    out += ["## How far to trust this", "",
            "- One transcript, annotated by the people on the project. A small sample gives only a rough picture; "
            "report the number of sentences and pairs with every figure.",
            "- Scores are only as good as the annotation. Report annotator agreement "
            "(scripts/annotator_agreement.py) alongside them.",
            "- Stance is scored separately on the UKP sentences (scripts/evaluate.py), because a transcript has no "
            "stance labels."]
    return "\n".join(out) + "\n"
