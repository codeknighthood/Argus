"""Phase 6 tests: statistics, error analysis, strategy comparison, pipeline evaluation, report and scripts.

Run from the project root:  python -m unittest discover -s tests -t . -v
The rule-based fake model stands in for Ollama, so these check ARGUS's own logic, not model quality.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from argus.analysis import (
    ERROR_TYPES, agreement_breakdown, auto_tags, count_error_kinds, error_kind, normalise_error_type, pairwise_kappa,
    tag_error_rates, tally_error_types, top_confusions,
)
from argus.annotation import compare_annotators
from argus.compare import (
    ERROR_CSV_COLUMNS, best_run, error_rows, pairwise_tests, run_stance_comparison, safe_name, write_error_csv,
)
from argus.data import Record
from argus.evaluation import read_csv, self_consistency, write_csv
from argus.llm import LLMClient
from argus.metrics import bootstrap_ci, mcnemar_exact
from argus.pipeline import ArgusPipeline, PipelineConfig
from argus.pipeline_eval import (
    check_sentences, component_of, evaluate_components, evaluate_relations, index_records, pipeline_report_md,
    predict_pair,
)
from argus.report import console_table, stance_report_md
from argus.schema import ArgumentRecord, Component, Relation, Stance, StanceResult

from .fakes import FakeLLM
from .test_phase5 import DEMO, EXTRACTION, STANCE, TARGET, count

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Favor", "Against", "Neutral"]


def run_script(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=ROOT)


# --------------------------------------------------------------------------- statistics
class TestStatistics(unittest.TestCase):
    def test_mcnemar_by_hand(self):
        r = mcnemar_exact([False] * 5, [True] * 5)                       # 0 vs 5: p = 2 * 1 / 2**5
        self.assertEqual((r["a_only"], r["b_only"]), (0, 5))
        self.assertAlmostEqual(r["p"], 0.0625)
        self.assertAlmostEqual(mcnemar_exact([True] * 3 + [False] * 3, [False] * 3 + [True] * 3)["p"], 1.0)
        # 1 vs 9: n = 10, p = 2 * (C(10,0) + C(10,1)) / 2**10
        r = mcnemar_exact([True] * 1 + [False] * 9, [False] * 1 + [True] * 9)
        self.assertAlmostEqual(r["p"], 2 * 11 / 1024)

    def test_mcnemar_identical_runs_and_bad_input(self):
        self.assertEqual(mcnemar_exact([True, False], [True, False]), {"a_only": 0, "b_only": 0, "p": 1.0})
        with self.assertRaises(ValueError):
            mcnemar_exact([True], [True, False])

    def test_mcnemar_survives_large_samples(self):
        r = mcnemar_exact([True] * 600 + [False] * 300, [False] * 600 + [True] * 300)
        self.assertLess(r["p"], 1e-10)
        self.assertGreaterEqual(r["p"], 0.0)

    def test_bootstrap_perfect_and_empty(self):
        self.assertEqual(bootstrap_ci(["a", "b"] * 20, ["a", "b"] * 20, ["a", "b"], "accuracy", 100), (1.0, 1.0))
        self.assertEqual(bootstrap_ci([], [], ["a"], "accuracy", 10), (0.0, 0.0))

    def test_bootstrap_is_reproducible_and_brackets_the_estimate(self):
        gold = ["a", "b"] * 50
        pred = [g if i % 5 else ("b" if g == "a" else "a") for i, g in enumerate(gold)]        # 80% correct
        lo, hi = bootstrap_ci(gold, pred, ["a", "b"], "accuracy", 400, seed=3)
        self.assertEqual((lo, hi), bootstrap_ci(gold, pred, ["a", "b"], "accuracy", 400, seed=3))
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)
        lo_f, hi_f = bootstrap_ci(gold, pred, ["a", "b"], "macro_f1", 200)
        self.assertLess(lo_f, hi_f)

    def test_more_data_means_a_narrower_interval(self):
        def width(n):
            gold = ["a", "b"] * (n // 2)
            pred = [g if i % 5 else ("b" if g == "a" else "a") for i, g in enumerate(gold)]
            lo, hi = bootstrap_ci(gold, pred, ["a", "b"], "accuracy", 300)
            return hi - lo
        self.assertLess(width(400), width(40))

    def test_bootstrap_rejects_unknown_metric(self):
        with self.assertRaises(ValueError):
            bootstrap_ci(["a"], ["a"], ["a"], "auc")


# --------------------------------------------------------------------------- automatic cues and human labels
class TestAnalysis(unittest.TestCase):
    def test_auto_tags(self):
        self.assertEqual(auto_tags("I don't think so."), ["first_person", "negation", "short"])
        self.assertIn("question", auto_tags("Why would anyone do that?"))
        tags = auto_tags("Perhaps, but it could cost 20% more than the plan that we should adopt in the next year.")
        for expected in ("hedge", "contrast", "number", "normative", "first_person"):
            self.assertIn(expected, tags)
        self.assertIn("pronoun_start", auto_tags("That is exactly why it failed."))
        self.assertIn("long", auto_tags(" ".join(["word"] * 40)))
        self.assertEqual(auto_tags(""), ["short"])

    def test_error_kind(self):
        cases = [("Favor", "Favor", None), ("Favor", "Against", "polarity flip"), ("Against", "Favor", "polarity flip"),
                 ("Neutral", "Favor", "over-reading"), ("Neutral", "Against", "over-reading"),
                 ("Favor", "Neutral", "under-reading"), ("Against", "Neutral", "under-reading"),
                 ("Favor", "ERROR", "failed call"), ("Favor", None, "failed call")]
        for gold, pred, expected in cases:
            with self.subTest(gold=gold, pred=pred):
                self.assertEqual(error_kind(gold, pred), expected)

    def rows(self):
        out = []
        for i in range(10):    # statements with a negation: all wrong
            out.append({"id": f"n{i}", "statement": f"We will not accept this plan number {i}.", "gold": "Favor", "pred": "Against"})
        for i in range(10):    # statements without: all right
            out.append({"id": f"p{i}", "statement": f"A higher wage floor helps families number {i}.", "gold": "Favor", "pred": "Favor"})
        return out

    def test_tag_error_rates(self):
        rates = {t["tag"]: t for t in tag_error_rates(self.rows())}
        self.assertEqual((rates["negation"]["n_with"], rates["negation"]["error_rate_with"]), (10, 1.0))
        self.assertEqual(rates["negation"]["error_rate_without"], 0.0)
        self.assertAlmostEqual(rates["negation"]["difference"], 1.0)
        self.assertNotIn("number", rates)                        # every sentence has a digit: nothing to compare with
        self.assertIn(tag_error_rates(self.rows())[0]["tag"], ("negation", "first_person"))   # biggest gap first

    def test_rare_tags_are_left_out(self):
        self.assertEqual(tag_error_rates(self.rows()[:3], min_n=5), [])

    def test_confusions_and_kinds(self):
        rows = [{"gold": "Favor", "pred": "Against"}] * 3 + [{"gold": "Neutral", "pred": "Favor"}] * 2 \
               + [{"gold": "Favor", "pred": "Favor"}] * 4 + [{"gold": "Against", "pred": "ERROR"}]
        self.assertEqual(top_confusions(rows, 2), [("Favor", "Against", 3), ("Neutral", "Favor", 2)])
        self.assertEqual(count_error_kinds(rows), {"polarity flip": 3, "over-reading": 2, "failed call": 1})

    def test_agreement_is_matched_by_id_not_position(self):
        a = [{"id": "1", "gold": "Favor", "pred": "Favor"}, {"id": "2", "gold": "Against", "pred": "Against"},
             {"id": "3", "gold": "Neutral", "pred": "Favor"}, {"id": "4", "gold": "Favor", "pred": "Against"}]
        b = [{"id": "4", "gold": "Favor", "pred": "Favor"}, {"id": "3", "gold": "Neutral", "pred": "Favor"},
             {"id": "2", "gold": "Against", "pred": "Against"}, {"id": "1", "gold": "Favor", "pred": "Favor"}]
        result = agreement_breakdown({"A": a, "B": b})
        self.assertEqual(result, {"n": 4, "unanimous_correct": 2, "unanimous_wrong": 1, "split": 1})
        k = pairwise_kappa({"A": a, "B": b})
        self.assertEqual((k[0]["a"], k[0]["b"]), ("A", "B"))
        self.assertAlmostEqual(k[0]["agree"], 0.75)
        self.assertEqual(agreement_breakdown({})["n"], 0)

    def test_normalise_error_type(self):
        self.assertEqual(normalise_error_type("3"), "missing context")
        self.assertEqual(normalise_error_type(" Implicit "), "implicit argument")
        self.assertEqual(normalise_error_type("label noise"), "label noise")
        self.assertEqual(normalise_error_type("sarcasm"), "negation or sarcasm")
        self.assertIsNone(normalise_error_type("  "))
        self.assertIsNone(normalise_error_type(None))
        with self.assertRaises(ValueError) as cm:
            normalise_error_type("9")
        self.assertIn("1 = implicit argument", str(cm.exception))

    def test_tally_counts_and_reports_every_bad_cell(self):
        rows = [{"error_type": "1"}, {"error_type": "implicit argument"}, {"error_type": ""}, {"error_type": "4"}]
        t = tally_error_types(rows)
        self.assertEqual(t["counts"], {"implicit argument": 2, "label noise": 1})
        self.assertEqual((t["tagged"], t["untagged"], t["total"]), (3, 1, 4))
        with self.assertRaises(ValueError) as cm:
            tally_error_types([{"error_type": "1"}, {"error_type": "oops"}, {"error_type": "nope"}])
        self.assertIn("row 3", str(cm.exception))
        self.assertIn("row 4", str(cm.exception))

    def test_error_type_agreement_between_two_taggers(self):
        base = [{"id": str(i), "statement": f"s{i}", "error_type": t} for i, t in enumerate(["1", "2", "3", "1"])]
        other = [{**r} for r in base]
        other[2]["error_type"] = "label noise"
        r = compare_annotators(base, other, column="error_type")
        self.assertEqual((r["n"], r["agreement"]), (4, 0.75))
        self.assertEqual(r["disagreements"][0]["sentence"], "s2")
        self.assertEqual(len(ERROR_TYPES), 7)


# --------------------------------------------------------------------------- strategy comparison
class QuirkyFake(FakeLLM):
    """Like the rule-based fake, except chain-of-thought calls call every 'helps' sentence Neutral."""

    def structured(self, messages, schema):
        out = super().structured(messages, schema)
        prompt = messages[-1]["content"]
        if "reasoning" in schema.model_json_schema()["properties"] and "helps" in prompt:
            return schema.model_validate({"reasoning": "unsure", "stance": "Neutral"})
        return out


def make_records():
    recs = []
    for i in range(10):
        recs.append((f"A higher wage floor helps working families, case {i}.", Stance.FAVOR))
    for i in range(5):
        recs.append((f"A higher wage floor means layoffs, case {i}.", Stance.AGAINST))
    for i in range(5):
        recs.append((f"The bill was debated on Tuesday afternoon, case {i}.", Stance.NEUTRAL))
    return [Record(id=f"r{i}", source="t", topic="minimum_wage" if i % 2 else "gun_control", target=TARGET,
                   statement=s, gold_stance=g) for i, (s, g) in enumerate(recs)]


class TestComparison(unittest.TestCase):
    def setUp(self):
        self.records = make_records()
        self.runs = run_stance_comparison({"fake": QuirkyFake()}, self.records,
                                          ["zero_shot", "few_shot", "cot"], bootstrap=60)

    def test_every_strategy_is_scored(self):
        self.assertEqual([r["strategy"] for r in self.runs], ["zero_shot", "few_shot", "cot"])
        by = {r["strategy"]: r for r in self.runs}
        self.assertEqual(by["zero_shot"]["report"]["accuracy"], 1.0)
        self.assertEqual(by["few_shot"]["report"]["accuracy"], 1.0)
        self.assertAlmostEqual(by["cot"]["report"]["accuracy"], 0.5)        # the 10 'helps' sentences go wrong
        for r in self.runs:
            self.assertIn("accuracy", r["ci"])
            self.assertLessEqual(r["ci"]["accuracy"][0], r["report"]["accuracy"] + 1e-9)
            self.assertGreaterEqual(r["ci"]["accuracy"][1], r["report"]["accuracy"] - 1e-9)
            self.assertEqual(set(r["per_topic"]), {"minimum_wage", "gun_control"})

    def test_best_run_and_ties(self):
        self.assertEqual(best_run(self.runs)["strategy"], "zero_shot")        # tie with few_shot: first wins
        self.assertEqual(best_run(list(reversed(self.runs)))["strategy"], "few_shot")

    def test_pairwise_tests(self):
        t = {(x["a"], x["b"]): x for x in pairwise_tests(self.runs)}
        same = t[("fake / zero_shot", "fake / few_shot")]
        self.assertEqual((same["a_only"], same["b_only"], same["p"]), (0, 0, 1.0))
        differ = t[("fake / zero_shot", "fake / cot")]
        self.assertEqual((differ["a_only"], differ["b_only"]), (10, 0))
        self.assertAlmostEqual(differ["p"], 2 / 2 ** 10)
        self.assertEqual(len(t), 3)

    def test_without_bootstrap(self):
        runs = run_stance_comparison({"fake": FakeLLM()}, self.records, ["zero_shot"], bootstrap=0)
        self.assertEqual(runs[0]["ci"], {})

    def test_two_models_get_distinct_names(self):
        runs = run_stance_comparison({"m1": FakeLLM(), "m2": QuirkyFake()}, self.records, ["cot"], bootstrap=0)
        self.assertEqual([r["name"] for r in runs], ["m1 / cot", "m2 / cot"])
        self.assertGreater(runs[0]["report"]["accuracy"], runs[1]["report"]["accuracy"])

    def test_progress_factory_is_used(self):
        names = []

        class Quiet:
            def __init__(self, total, name):
                names.append((total, name))
            def update(self, n=1): pass
            def close(self): pass

        run_stance_comparison({"fake": FakeLLM()}, self.records, ["zero_shot"], 0, lambda t, n: Quiet(t, n))
        self.assertEqual(names, [(20, "fake / zero_shot")])

    def test_error_sheet_is_a_fixed_random_order_with_blank_causes(self):
        cot = next(r for r in self.runs if r["strategy"] == "cot")
        rows = error_rows(cot)
        self.assertEqual(len(rows), 10)
        self.assertEqual({r["error_kind"] for r in rows}, {"under-reading"})
        self.assertEqual({r["error_type"] for r in rows}, {""})
        self.assertEqual([r["id"] for r in rows], [r["id"] for r in error_rows(cot)])        # same order every time
        self.assertNotEqual([r["id"] for r in rows], sorted(r["id"] for r in rows))          # and it is shuffled
        self.assertEqual(list(rows[0]), ERROR_CSV_COLUMNS)

    def test_error_csv_roundtrip(self):
        cot = next(r for r in self.runs if r["strategy"] == "cot")
        with tempfile.TemporaryDirectory() as d:
            n = write_error_csv(cot, Path(d) / "e.csv")
            back = read_csv(Path(d) / "e.csv")
        self.assertEqual((n, len(back)), (10, 10))
        self.assertEqual(list(back[0]), ERROR_CSV_COLUMNS)

    def test_no_mistakes_gives_an_empty_sheet(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(write_error_csv(self.runs[0], Path(d) / "e.csv"), 0)

    def test_safe_name(self):
        self.assertEqual(safe_name("qwen2.5:7b / few_shot"), "qwen2.5_7b___few_shot")


# --------------------------------------------------------------------------- self-consistency and cache salt
class FlipFake(FakeLLM):
    """Run 1 of 3 calls everything Neutral; the other runs use the normal rules."""

    def __init__(self, run):
        super().__init__()
        self.run = run

    def structured(self, messages, schema):
        if self.run == 1:
            self.calls.append(messages[-1]["content"])
            return schema.model_validate({"stance": "Neutral", "explanation": "x"})
        return super().structured(messages, schema)


class TestConsistency(unittest.TestCase):
    def test_self_consistency(self):
        result = self_consistency(make_records(), "zero_shot", 3, lambda k: FlipFake(k), LABELS)
        self.assertEqual((result["runs"], result["n"]), (3, 20))
        self.assertEqual(result["per_run_accuracy"], [1.0, 0.25, 1.0])
        self.assertAlmostEqual(result["all_same"], 5 / 20)                    # only the 5 Neutral sentences never flip
        self.assertAlmostEqual(result["mean_majority_share"], (15 * 2 / 3 + 5) / 20)
        self.assertEqual(result["majority_accuracy"], 1.0)

    def test_perfectly_stable_model(self):
        result = self_consistency(make_records(), "zero_shot", 2, lambda k: FakeLLM(), LABELS)
        self.assertEqual((result["all_same"], result["mean_majority_share"]), (1.0, 1.0))

    def test_cache_salt_separates_runs_but_leaves_old_keys_alone(self):
        msgs = [{"role": "user", "content": "hello"}]
        with tempfile.TemporaryDirectory() as d:
            def client(**extra):
                return LLMClient({"model": "m", "temperature": 0.7, "cache_dir": d, **extra})
            plain = client()._cache_path(msgs, StanceResult)
            self.assertNotEqual(plain, client(cache_salt="run-1")._cache_path(msgs, StanceResult))
            self.assertNotEqual(client(cache_salt="run-1")._cache_path(msgs, StanceResult),
                                client(cache_salt="run-2")._cache_path(msgs, StanceResult))
            blob = json.dumps(["m", 0.7, "StanceResult", msgs], sort_keys=True)      # the key format from before Phase 6
            self.assertEqual(plain.name, hashlib.sha256(blob.encode()).hexdigest() + ".json")


# --------------------------------------------------------------------------- report
class TestReport(unittest.TestCase):
    def setUp(self):
        self.records = make_records()
        self.runs = run_stance_comparison({"fake": QuirkyFake()}, self.records,
                                          ["zero_shot", "few_shot", "cot"], bootstrap=40)
        self.tests = pairwise_tests(self.runs)
        rows = {r["name"]: r["rows"] for r in self.runs}
        self.meta = {"split": "dev", "n": 20, "models": ["fake"], "date": "2026-01-01", "bootstrap": 40}
        self.args = (self.runs, self.tests, pairwise_kappa(rows), agreement_breakdown(rows),
                     tag_error_rates(self.runs[2]["rows"], min_n=3), self.runs[0], self.meta)

    def test_sections_and_numbers(self):
        md = stance_report_md(*self.args)
        for heading in ("## 1. Strategy comparison", "## 2. F1 per label", "## 3. Accuracy per topic",
                        "## 4. Confusion matrices", "## 5. Are the differences real?", "## 6. Agreement between runs",
                        "## 7. Shape of the mistakes", "## How far to trust this"):
            self.assertIn(heading, md)
        self.assertIn("100.0%", md)
        self.assertIn("50.0%", md)
        self.assertIn("Highest macro-F1: **zero_shot**", md)
        self.assertIn("not a statistically significant difference", md)       # zero_shot vs few_shot are identical
        self.assertIn("95% bootstrap", md)
        self.assertNotIn("Self-consistency", md)
        self.assertNotIn("do not use them to choose", md)

    def test_significant_difference_is_called_significant(self):
        runs = [self.runs[0], self.runs[2]]
        md = stance_report_md(runs, pairwise_tests(runs), [], {"n": 0}, [], runs[0], self.meta)
        self.assertIn("a statistically significant difference", md)
        self.assertNotIn("not a statistically", md)

    def test_test_split_carries_a_warning(self):
        md = stance_report_md(*self.args[:6], {**self.meta, "split": "test"})
        self.assertIn("do not use them to choose a strategy", md)

    def test_consistency_block(self):
        c = {"runs": 3, "n": 20, "all_same": 0.25, "mean_majority_share": 0.75,
             "per_run_accuracy": [1.0, 0.25, 1.0], "majority_accuracy": 1.0}
        md = stance_report_md(*self.args, consistency=c)
        self.assertIn("## 9. Self-consistency", md)
        self.assertIn("25.0%", md)

    def test_single_run_and_no_bootstrap(self):
        run = run_stance_comparison({"fake": FakeLLM()}, self.records, ["zero_shot"], bootstrap=0)
        md = stance_report_md(run, [], [], {"n": 0}, [], run[0], {**self.meta, "bootstrap": 0})
        self.assertIn("Confidence intervals: not computed", md)
        self.assertNotIn("Highest macro-F1", md)
        self.assertNotIn("## 5.", md)

    def test_tables_are_well_formed(self):
        md = stance_report_md(*self.args)
        for line in md.splitlines():
            if line.startswith("|"):
                self.assertTrue(line.endswith("|"), line)

    def test_console_table(self):
        table = console_table(self.runs)
        self.assertEqual(len(table.splitlines()), 4)
        self.assertIn("macro-F1", table)
        self.assertIn("cot", table)


# --------------------------------------------------------------------------- pipeline evaluation
def demo_records(**config):
    return ArgusPipeline(FakeLLM(), PipelineConfig(**config)).run_transcript(DEMO, TARGET)


class TestPipelineEvaluation(unittest.TestCase):
    def setUp(self):
        self.recs = index_records(demo_records())

    def test_stance_none_skips_the_stage_entirely(self):
        llm = FakeLLM()
        recs = ArgusPipeline(llm, PipelineConfig(stance_on="none")).run_transcript(DEMO, TARGET)
        self.assertEqual(count(llm, STANCE), 0)
        self.assertEqual(count(llm, EXTRACTION), 6)
        self.assertTrue(all(r.stance is None and r.explanation is None for r in recs))
        self.assertEqual(recs[2].relation, Relation.SUPPORT)               # the other stages still ran

    def test_check_sentences(self):
        ok = [(i, r.original_statement) for i, r in self.recs.items()]
        self.assertEqual(check_sentences(self.recs, ok), [])
        problems = check_sentences(self.recs, [(1, "Something else entirely."), (99, "Nowhere.")])
        self.assertEqual(len(problems), 2)
        self.assertIn("#1", problems[0])
        self.assertIn("not in the pipeline output", problems[1])

    def test_component_of(self):
        self.assertEqual([component_of(self.recs[i]) for i in range(3)], ["None", "Claim", "Premise"])
        self.assertEqual(component_of(None), "ERROR")
        self.assertEqual(component_of(ArgumentRecord(policy_target="t", original_statement="s")), "ERROR")

    def test_evaluate_components(self):
        gold = ["None", "Claim", "Premise", "Premise", "Premise", "Claim"]      # the last one disagrees with the pipeline
        sheet = [{"index": str(i), "speaker": "", "sentence": self.recs[i].original_statement, "gold": g}
                 for i, g in enumerate(gold)]
        result = evaluate_components(self.recs, sheet)
        self.assertAlmostEqual(result["report"]["accuracy"], 5 / 6)
        self.assertEqual(result["rows"][5]["pred"], "Premise")
        self.assertLess(result["kappa"], 1.0)

    def test_extraction_failure_is_scored_as_an_error(self):
        recs = index_records(ArgusPipeline(FakeLLM(fail_when=lambda p: EXTRACTION in p and "layoffs" in p),
                                           PipelineConfig()).run_transcript(DEMO, TARGET))
        self.assertEqual(component_of(recs[5]), "ERROR")
        sheet = [{"index": "5", "sentence": recs[5].original_statement, "gold": "Premise"}]
        self.assertEqual(evaluate_components(recs, sheet)["report"]["unparsed"], 1)

    def test_predict_pair_reasons(self):
        window = 3
        cases = [((1, 2), ("Support", "linked")), ((3, 4), ("Attack", "linked")), ((4, 5), ("Support", "linked")),
                 ((1, 3), ("None", "B was linked to a different argument (nearest-first rule)")),
                 ((0, 1), ("None", "A was not detected as an argument")),
                 ((1, 0), ("None", "B was not detected as an argument")),
                 ((2, 99), ("None", "sentence missing from the pipeline output"))]
        for (a, b), expected in cases:
            with self.subTest(pair=(a, b)):
                self.assertEqual(predict_pair(a, b, self.recs, window), expected)

    def test_predict_pair_window_and_switched_off_stage(self):
        self.assertEqual(predict_pair(2, 5, self.recs, 1), ("None", "A is outside the relation window"))
        off = index_records(demo_records(relation_window=0))
        self.assertEqual(predict_pair(2, 3, off, 0), ("None", "relation stage switched off"))

    def test_predict_pair_other_reasons(self):
        a = ArgumentRecord(policy_target="t", original_statement="a", segment_index=0, is_argument=True,
                           component=Component.CLAIM)
        b = ArgumentRecord(policy_target="t", original_statement="b", segment_index=1, is_argument=True,
                           component=Component.PREMISE, relation=Relation.NONE)
        self.assertEqual(predict_pair(0, 1, {0: a, 1: b}, 3), ("None", "the model judged B unrelated to A"))
        b_failed = b.model_copy(update={"relation": None, "error": "relation: boom"})
        self.assertEqual(predict_pair(0, 1, {0: a, 1: b_failed}, 3), ("None", "relation stage failed"))
        unknown = a.model_copy(update={"is_argument": None})
        self.assertEqual(predict_pair(0, 1, {0: unknown, 1: b}, 3), ("None", "extraction failed for A or B"))

    def pairs(self, gold_by_pair):
        return [{"pair_id": f"{a}-{b}", "a_index": str(a), "b_index": str(b), "speaker_a": "", "speaker_b": "",
                 "sentence_a": self.recs[a].original_statement, "sentence_b": self.recs[b].original_statement, "gold": g}
                for (a, b), g in gold_by_pair.items()]

    def test_evaluate_relations_and_outcomes(self):
        result = evaluate_relations(self.recs, self.pairs({
            (1, 2): "Support",      # correct
            (1, 3): "Attack",       # missed: B was linked to a different argument
            (3, 4): "None",         # pipeline linked a pair the annotator marked None
            (4, 5): "Attack",       # linked, but the pipeline said Support
        }), relation_window=3)
        self.assertAlmostEqual(result["report"]["accuracy"], 1 / 4)
        self.assertEqual(result["gold_links"], 3)
        self.assertEqual(result["links_found_correctly"], 1)
        self.assertEqual(result["misses"], {
            "B was linked to a different argument (nearest-first rule)": 1,
            "linked a pair the annotator marked None": 1,
            "linked, but labelled differently from the annotator": 1,
        })
        self.assertEqual([r["outcome"] for r in result["rows"]][0], "correct")

    def test_report_text(self):
        sheet = [{"index": str(i), "speaker": "", "sentence": self.recs[i].original_statement, "gold": component_of(self.recs[i])}
                 for i in range(6)]
        comps = evaluate_components(self.recs, sheet)
        rels = evaluate_relations(self.recs, self.pairs({(1, 2): "Support", (1, 3): "Attack"}), 3)
        meta = {"transcript": "demo.txt", "sentences": 6, "target": TARGET, "model": "fake", "relation_window": 3,
                "date": "2026-01-01"}
        md = pipeline_report_md(meta, comps, rels)
        for text in ("# ARGUS pipeline evaluation", "## Claim / Premise / None", "## Support / Attack / None",
                     "### Why pairs were missed or mislabelled", "nearest-first rule", "found and labelled 1 correctly (50%)",
                     "How far to trust this"):
            self.assertIn(text, md)
        only_components = pipeline_report_md(meta, comps, None)
        self.assertNotIn("## Support / Attack / None", only_components)
        self.assertIn("Cohen's kappa", only_components)


# --------------------------------------------------------------------------- scripts (no Ollama needed)
class TestScripts(unittest.TestCase):
    def tagged_csv(self, directory, causes):
        rows = [{"id": str(i), "statement": f"s{i}", "gold": "Favor", "pred": "Against", "error_kind": "polarity flip",
                 "error_type": c} for i, c in enumerate(causes)]
        path = Path(directory) / "errors.csv"
        write_csv(rows, path, ["id", "statement", "gold", "pred", "error_kind", "error_type"])
        return path

    def test_tally_script(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "causes.md"
            p = run_script("tally_errors.py", str(self.tagged_csv(d, ["1", "implicit argument", "4", ""])), "--out", str(out))
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("Tagged 3 of 4 mistakes", p.stdout)
            self.assertIn("| implicit argument | 2 | 67% |", p.stdout)
            self.assertIn("Cause by shape of the mistake", p.stdout)
            self.assertIn("percentages describe the 3 tagged rows", p.stdout)
            self.assertIn("implicit argument", out.read_text(encoding="utf-8"))

    def test_tally_script_fails_politely(self):
        with tempfile.TemporaryDirectory() as d:
            for csv_causes, expected in [(["1", "oops"], "row 3"), (["", ""], "No tagged rows")]:
                with self.subTest(expected=expected):
                    p = run_script("tally_errors.py", str(self.tagged_csv(d, csv_causes)))
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn(expected, p.stderr)
                    self.assertNotIn("Traceback", p.stderr)
            p = run_script("tally_errors.py", str(Path(d) / "missing.csv"))
            self.assertIn("File not found", p.stderr)
            self.assertNotIn("Traceback", p.stderr)

    def test_evaluate_pipeline_needs_annotations_and_a_target(self):
        with tempfile.TemporaryDirectory() as d:
            nothing = run_script("evaluate_pipeline.py", "--components", str(Path(d) / "a.csv"), "--pairs", str(Path(d) / "b.csv"))
            self.assertNotEqual(nothing.returncode, 0)
            self.assertIn("Nothing to score", nothing.stderr)
            self.assertNotIn("Traceback", nothing.stderr)
            t = Path(d) / "debate.txt"
            t.write_text(DEMO, encoding="utf-8")
            no_target = run_script("evaluate_pipeline.py", "--transcript", str(t))
            self.assertIn("--target", no_target.stderr)
            self.assertNotIn("Traceback", no_target.stderr)

    def test_scripts_explain_a_missing_phase_one(self):
        with tempfile.TemporaryDirectory() as d:
            for script in ("evaluate.py", "run_stance.py", "run_extraction.py"):
                with self.subTest(script=script):
                    p = run_script(script, "--data-dir", d)
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn("prepare_data.py", p.stderr)
                    self.assertNotIn("Traceback", p.stderr)

    def test_error_type_agreement_script(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.tagged_csv(d, ["1", "2", "3"])
            p = run_script("annotator_agreement.py", str(a), str(a), "--column", "error_type")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("Cohen's kappa", p.stdout)


if __name__ == "__main__":
    unittest.main()
