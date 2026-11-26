"""Phase 6: turn the numbers of a stance comparison into a report you can read, check and quote (synopsis 5.8)."""
from __future__ import annotations

from typing import Optional, Sequence

from .analysis import count_error_kinds, top_confusions
from .evaluation import OTHER_TOPICS
from .metrics import format_confusion

LABELS = ["Favor", "Against", "Neutral"]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _display(runs: Sequence[dict]):
    """Short run names (just the strategy) when only one model is compared, full names otherwise."""
    single = len({r["model"] for r in runs}) <= 1
    return lambda run: run["strategy"] if single else run["name"]


def _with_ci(run: dict, metric: str) -> str:
    value = run["report"][metric]
    ci = run["ci"].get(metric)
    return f"{_pct(value)} ({_pct(ci[0])} to {_pct(ci[1])})" if ci else _pct(value)


def console_table(runs: Sequence[dict]) -> str:
    name = _display(runs)
    width = max([len("run")] + [len(name(r)) for r in runs])
    lines = [f"{'run':<{width}}  accuracy   macro-P   macro-R   macro-F1  failed"]
    for r in runs:
        rep = r["report"]
        lines.append(f"{name(r):<{width}}  {_pct(rep['accuracy']):>8}  {_pct(rep['macro_precision']):>8}  "
                     f"{_pct(rep['macro_recall']):>8}  {_pct(rep['macro_f1']):>8}  {rep['unparsed']:>6}")
    return "\n".join(lines)


def stance_report_md(
    runs: Sequence[dict], tests: Sequence[dict], kappas: Sequence[dict], breakdown: dict,
    tag_rows: Sequence[dict], best: dict, meta: dict, consistency: Optional[dict] = None,
) -> str:
    name = _display(runs)
    multi = len(runs) > 1
    out = ["# ARGUS stance evaluation", ""]
    out += [f"- Split: **{meta['split']}** ({meta['n']} examples" + (f" from {meta['source']}" if meta.get("source") else "") + ")",
            f"- Models: {', '.join(meta['models'])}",
            f"- Date: {meta['date']}"]
    if meta.get("bootstrap"):
        out.append(f"- Confidence intervals: 95% bootstrap over the sentences ({meta['bootstrap']} resamples)")
    else:
        out.append("- Confidence intervals: not computed (run with --bootstrap 1000 to get them)")
    if meta["split"] == "test":
        out += ["", "> TEST split: report these numbers, but do not use them to choose a strategy. "
                    "Choose on dev, then run test once."]
    out.append("")

    # 1. comparison table
    out += ["## 1. Strategy comparison", "",
            "| Run | Accuracy | Macro precision | Macro recall | Macro F1 | Failed calls |", "|---|---|---|---|---|---|"]
    for r in runs:
        rep = r["report"]
        out.append(f"| {name(r)} | {_with_ci(r, 'accuracy')} | {_pct(rep['macro_precision'])} | "
                   f"{_pct(rep['macro_recall'])} | {_with_ci(r, 'macro_f1')} | {rep['unparsed']} |")
    out += ["", "Macro scores average the three labels equally. A failed call counts as a wrong answer.", ""]
    if multi:
        ranked = sorted(runs, key=lambda r: (-r["report"]["macro_f1"], -r["report"]["accuracy"]))
        top, second = ranked[0], ranked[1]
        line = f"Highest macro-F1: **{name(top)}** ({_pct(top['report']['macro_f1'])}). " \
               f"Next: {name(second)} ({_pct(second['report']['macro_f1'])})."
        match = next((t for t in tests if {t["a"], t["b"]} == {top["name"], second["name"]}), None)
        if match:
            top_only, second_only = (match["a_only"], match["b_only"]) if match["a"] == top["name"] \
                else (match["b_only"], match["a_only"])
            verdict = "a statistically significant difference" if match["p"] < 0.05 \
                else "not a statistically significant difference"
            line += (f" Exact McNemar test between them: p = {match['p']:.3f}, {verdict} at the 0.05 level "
                     f"({top_only} sentences only {name(top)} gets right, {second_only} only {name(second)}).")
        out += [line, ""]

    # 2. per label
    out += ["## 2. F1 per label", "", "| Run | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for r in runs:
        out.append(f"| {name(r)} | " + " | ".join(f"{r['report']['per_class'][l]['f1']:.3f}" for l in LABELS) + " |")
    out.append("")

    # 3. per topic
    topics = sorted({t for r in runs for t in r["per_topic"]}, key=lambda t: (t == OTHER_TOPICS, t))
    out += ["## 3. Accuracy per topic", "", "| Topic | n | " + " | ".join(name(r) for r in runs) + " |",
            "|---|---|" + "---|" * len(runs)]
    for t in topics:
        n = next((r["per_topic"][t][1] for r in runs if t in r["per_topic"]), 0)
        cells = [_pct(r["per_topic"][t][0]) if t in r["per_topic"] else "-" for r in runs]
        out.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    out += ["", "Each topic has few examples, so differences of a few points between topics are noise. "
                "With many topics only the largest are listed and the rest are pooled.", ""]
    groups = sorted({g for r in runs for g in r.get("per_group", {})})
    if groups:
        out += ["Accuracy by how the example was built:", "",
                "| Kind of example | n | " + " | ".join(name(r) for r in runs) + " |", "|---|---|" + "---|" * len(runs)]
        for g in groups:
            n = next((r["per_group"][g][1] for r in runs if g in r["per_group"]), 0)
            out.append(f"| {g} | {n} | " + " | ".join(
                _pct(r["per_group"][g][0]) if g in r.get("per_group", {}) else "-" for r in runs) + " |")
        if "synthetic neutral" in groups:
            out += ["", "'Synthetic neutral' examples pair a comment with a topic taken from a different comment, so "
                        "Neutral is easy for them. They make up most of the Neutral examples in this data, which "
                        "raises the Neutral score and the macro-F1. Judge the model mainly on the other kinds, and on "
                        "Favor against Against."]
        out.append("")

    # 4. confusion matrices
    out += ["## 4. Confusion matrices", "", "Rows are the gold label, columns the model's answer.", ""]
    for r in runs:
        out += [f"**{name(r)}**", "", "```", format_confusion(r["report"]), "```", ""]

    section = 5
    if tests:
        out += [f"## {section}. Are the differences real?", "",
                "Exact McNemar test on the sentences where exactly one run is right.", "",
                "| Run A | Run B | Only A right | Only B right | p-value | Verdict |", "|---|---|---|---|---|---|"]
        for t in tests:
            verdict = "significant (p < 0.05)" if t["p"] < 0.05 else "not significant"
            out.append(f"| {t['a']} | {t['b']} | {t['a_only']} | {t['b_only']} | {t['p']:.3f} | {verdict} |")
        out += ["", "With many comparisons, a few small p-values appear by chance. Treat a p-value near 0.05 as weak.", ""]
    if multi and breakdown.get("n"):
        out += ["## 6. Agreement between runs", "",
                f"On {breakdown['n']} sentences: all runs right on {breakdown['unanimous_correct']}, "
                f"all runs wrong on {breakdown['unanimous_wrong']}, runs disagree on {breakdown['split']}.", ""]
        if kappas:
            out += ["| Run A | Run B | Same answer | Cohen's kappa |", "|---|---|---|---|"]
            out += [f"| {k['a']} | {k['b']} | {_pct(k['agree'])} | {k['kappa']:.3f} |" for k in kappas]
            out.append("")
        out += ["Sentences that every run gets wrong are the interesting ones: they point at label noise or at "
                "something no prompt fixes.", ""]

    # 7. shape of mistakes (best run)
    out += ["## 7. Shape of the mistakes", "", f"Best run: {name(best)}.", ""]
    kinds = count_error_kinds(best["rows"])
    if kinds:
        out += ["| Kind | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in kinds.items()]
        out += ["", "polarity flip = Favor called Against or the reverse; over-reading = a side found in a neutral "
                    "sentence; under-reading = a real stance missed (called Neutral).", ""]
        conf = top_confusions(best["rows"], 5)
        if conf:
            out += ["Most common confusions (gold to predicted): " +
                    ", ".join(f"{g} to {p} ({n})" for g, p, n in conf) + ".", ""]
    else:
        out += ["No mistakes to analyse.", ""]

    # 8. cues
    out += ["## 8. Sentence features linked to mistakes", ""]
    if tag_rows:
        out += ["| Feature | Sentences with it | Wrong with it | Sentences without | Wrong without |", "|---|---|---|---|---|"]
        for t in tag_rows:
            out.append(f"| {t['tag']} ({t['description']}) | {t['n_with']} | {_pct(t['error_rate_with'])} | "
                       f"{t['n_without']} | {_pct(t['error_rate_without'])} |")
        out += ["", "These are leads for reading the mistakes, not explanations. A feature can be linked to errors "
                    "only because it travels with something else.", ""]
    else:
        out += ["Too few sentences per feature to compare.", ""]

    if consistency:
        c = consistency
        out += ["## 9. Self-consistency", "",
                f"The best run was repeated {c['runs']} times at a non-zero temperature on {c['n']} sentences.", "",
                f"- Every run gave the same label on {_pct(c['all_same'])} of sentences.",
                f"- On average {_pct(c['mean_majority_share'])} of the runs agreed with the majority label.",
                "- Accuracy per run: " + ", ".join(_pct(a) for a in c["per_run_accuracy"]) + ".",
                f"- Accuracy of the majority vote: {_pct(c['majority_accuracy'])}.", "",
                "A model whose label changes between runs on the same sentence is not a stable annotator, "
                "whatever its accuracy.", ""]

    out += ["## How far to trust this", "",
            "- Each score comes from one sample of sentences. The intervals show how much it could move on "
            "another sample; quote them with the score.",
            "- The gold labels come from crowd annotators and contain mistakes, so a score of 100% is not "
            "reachable and a few 'model errors' are label errors.",
            "- One model on one corpus. Say so when you report results; do not claim they hold for other debates.",
            "- The mistake analysis (sections 7 and 8) uses automatic cues only. The cause of each mistake comes "
            "from your own tagging (docs/ERROR_ANALYSIS.md)."]
    return "\n".join(out) + "\n"
