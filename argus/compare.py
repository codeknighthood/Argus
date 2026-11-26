"""Phase 6: compare prompting strategies (and models) on the stance task (synopsis 5.6 and 5.8).

Every run is scored on exactly the same sentences, so differences between runs come from the strategy or model,
not from the sample. A run is a dict:

    name, model, strategy   "qwen2.5:7b / few_shot" and its parts
    report                  accuracy, per-label precision / recall / F1, macro and weighted F1, confusion matrix
    ci                      95% bootstrap intervals for accuracy and macro-F1 ({} when bootstrap is 0)
    per_topic               {topic: (accuracy, n)} for the 10 largest topics, the rest pooled
    per_group               {kind of example: (accuracy, n)}, e.g. VAST's synthetic neutral examples
    rows                    every prediction (gold, pred, explanation ...)
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

from .analysis import auto_tags, error_kind
from .data import Record
from .evaluation import Progress, collapse_topics, per_group_accuracy, per_topic_accuracy, predict_records, write_csv
from .metrics import bootstrap_ci, classification_report, mcnemar_exact

LABELS = ["Favor", "Against", "Neutral"]
ERROR_CSV_COLUMNS = ["id", "topic", "statement", "gold", "pred", "error_kind", "auto_tags", "explanation",
                     "error_type", "notes"]


def run_stance_comparison(
    llms: dict, records: Sequence[Record], strategies: Sequence[str], bootstrap: int = 1000,
    progress_factory: Optional[Callable[[int, str], Progress]] = None,
) -> list[dict]:
    """Run every strategy with every model on `records` and score each run.

    `llms` maps a model name to a client. A failed model call never stops the comparison: that sentence is
    scored "ERROR", which counts as wrong.
    """
    runs = []
    for model, llm in llms.items():
        for strategy in strategies:
            name = f"{model} / {strategy}"
            bar = progress_factory(len(records), name) if progress_factory else None
            rows = predict_records(llm, records, strategy, bar)
            if bar:
                bar.close()
            gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
            run = {"name": name, "model": model, "strategy": strategy,
                   "report": classification_report(gold, pred, LABELS),
                   "ci": {}, "per_topic": collapse_topics(per_topic_accuracy(rows)),
                   "per_group": per_group_accuracy(rows), "rows": rows}
            if bootstrap > 0:
                run["ci"] = {m: bootstrap_ci(gold, pred, LABELS, m, bootstrap) for m in ("accuracy", "macro_f1")}
            runs.append(run)
    return runs


def best_run(runs: Sequence[dict]) -> dict:
    """Highest macro-F1, then accuracy. On a full tie the first run wins."""
    return max(runs, key=lambda r: (r["report"]["macro_f1"], r["report"]["accuracy"]))


def pairwise_tests(runs: Sequence[dict]) -> list[dict]:
    """Exact McNemar test for every pair of runs: is the difference more than luck?"""
    out = []
    for x in range(len(runs)):
        for y in range(x + 1, len(runs)):
            a, b = runs[x], runs[y]
            result = mcnemar_exact([r["gold"] == r["pred"] for r in a["rows"]],
                                   [r["gold"] == r["pred"] for r in b["rows"]])
            out.append({"a": a["name"], "b": b["name"], **result})
    return out


def error_rows(run: dict, seed: int = 0) -> list[dict]:
    """The mistakes of one run in a fixed random order, with blank `error_type` and `notes` to fill in.

    The order is shuffled (same shuffle every time) so that reading only the first 50 rows is a fair sample of
    all mistakes, not just the first topic in the file.
    """
    wrong = [r for r in run["rows"] if r["gold"] != r["pred"]]
    random.Random(seed).shuffle(wrong)
    return [{"id": r["id"], "topic": r["topic"], "statement": r["statement"], "gold": r["gold"], "pred": r["pred"],
             "error_kind": error_kind(r["gold"], r["pred"]), "auto_tags": "; ".join(auto_tags(r["statement"])),
             "explanation": r.get("explanation") or r.get("error") or "", "error_type": "", "notes": ""}
            for r in wrong]


def write_error_csv(run: dict, path: Path) -> int:
    rows = error_rows(run)
    write_csv(rows, path, ERROR_CSV_COLUMNS)
    return len(rows)


def safe_name(name: str) -> str:
    """A run name made safe for a file name on any system."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)
