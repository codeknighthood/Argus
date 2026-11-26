"""Evaluation metrics (synopsis 5.8): accuracy, precision, recall, F1, Cohen's kappa, and (Phase 6) a bootstrap
confidence interval and an exact McNemar test.

Written in plain Python so there is nothing extra to install and every number can be checked by hand.
Labels are plain strings. A prediction outside the label set (for example "ERROR" when the LLM call
failed) counts as wrong, so failures lower accuracy and recall instead of being hidden.
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def confusion_matrix(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    """Rows = gold label, columns = predicted label. Predictions outside `labels` are left out of the matrix."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        if g in index and p in index:
            matrix[index[g]][index[p]] += 1
    return matrix


def classification_report(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict:
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} items but pred has {len(pred)}")
    n = len(gold)
    per_class: dict[str, dict] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    present = [l for l in labels if per_class[l]["support"] > 0]
    total_support = sum(per_class[l]["support"] for l in labels)

    def mean(key: str) -> float:
        return sum(per_class[l][key] for l in present) / len(present) if present else 0.0

    return {
        "n": n,
        "labels": list(labels),
        "accuracy": sum(1 for g, p in zip(gold, pred) if g == p) / n if n else 0.0,
        "macro_precision": mean("precision"),
        "macro_recall": mean("recall"),
        "macro_f1": mean("f1"),
        "weighted_f1": (
            sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / total_support
            if total_support else 0.0
        ),
        "per_class": per_class,
        "confusion": confusion_matrix(gold, pred, labels),
        "unparsed": sum(1 for p in pred if p not in labels),
    }


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two label sequences, corrected for chance (1 = perfect, 0 = chance level).

    Used for inter-annotator agreement and for agreement between prompting strategies.
    """
    if len(a) != len(b):
        raise ValueError(f"a has {len(a)} items but b has {len(b)}")
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b))
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def format_confusion(report: dict) -> str:
    labels = report["labels"]
    width = max(len(l) for l in labels + ["gold \\ pred"])
    lines = [f"{'gold \\ pred':<{width}}  " + "  ".join(f"{l:>{max(len(l), 7)}}" for l in labels)]
    for label, row in zip(labels, report["confusion"]):
        lines.append(f"{label:<{width}}  " + "  ".join(f"{v:>{max(len(l), 7)}}" for l, v in zip(labels, row)))
    return "\n".join(lines)


def format_report(report: dict) -> str:
    lines = [f"{'label':<10} {'precision':>9} {'recall':>8} {'f1':>8} {'support':>8}"]
    for label in report["labels"]:
        m = report["per_class"][label]
        lines.append(f"{label:<10} {m['precision']:>9.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['support']:>8}")
    lines.append("")
    lines.append(f"accuracy {report['accuracy']:.3f} | macro-F1 {report['macro_f1']:.3f} "
                 f"| weighted-F1 {report['weighted_f1']:.3f} | n={report['n']}")
    if report["unparsed"]:
        lines.append(f"{report['unparsed']} prediction(s) failed (counted as wrong)")
    lines.append("")
    lines.append(format_confusion(report))
    return "\n".join(lines)


# --------------------------------------------------------------------------- Phase 6: how far to trust a number
def bootstrap_ci(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str], metric: str = "accuracy",
    n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for accuracy or macro-F1.

    Resamples the sentences with replacement `n_boot` times, scores each resample, and returns the middle
    (1 - alpha) of those scores. The seed is fixed, so the same data always gives the same interval.
    A score measured on 700 sentences is an estimate; the interval says how much it could move on other sentences.
    """
    if metric not in ("accuracy", "macro_f1"):
        raise ValueError("metric must be 'accuracy' or 'macro_f1'")
    n = len(gold)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    scores = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        g, p = [gold[i] for i in idx], [pred[i] for i in idx]
        if metric == "accuracy":
            scores.append(sum(1 for x, y in zip(g, p) if x == y) / n)
        else:
            scores.append(classification_report(g, p, labels)["macro_f1"])
    scores.sort()
    lo = scores[int((alpha / 2) * (len(scores) - 1))]
    hi = scores[int((1 - alpha / 2) * (len(scores) - 1))]
    return (lo, hi)


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Exact McNemar test: do two runs, scored on the SAME sentences, really differ?

    Only sentences where exactly one run is right matter (`a_only`, `b_only`). If the runs were equally good,
    those would split 50/50, so the p-value is the two-sided binomial probability of a split at least this uneven.
    A small p (below 0.05) means the difference is unlikely to be luck.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(f"a has {len(correct_a)} items but b has {len(correct_b)}")
    a_only = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    b_only = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(a_only, b_only)
    # log-space terms so large n does not overflow
    tail = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2))
               for i in range(k + 1))
    return {"a_only": a_only, "b_only": b_only, "p": min(1.0, 2 * tail)}
