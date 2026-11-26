"""Phase 4 tests: argument relations, pair annotation sheets, relation scoring.

Run from the project root:  python -m unittest discover -s tests -t . -v
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from argus.annotation import (
    PAIR_COLUMNS, compare_annotators, load_pair_sheet, normalise_relation, pairs_sheet_rows,
)
from argus.evaluation import predict_relations, read_csv, write_csv
from argus.metrics import classification_report
from argus.prompting import placeholders, render
from argus.relations import ArgItem, build_messages, classify_relation, find_relation
from argus.schema import Relation, RelationResult
from argus.segment import segment_transcript

from .fakes import FakeLLM

ROOT = Path(__file__).resolve().parent.parent
POS = "Full-time workers should not live in poverty."          # polarity +1 in the fake
NEG = "Small businesses run on thin margins."                  # polarity -1
NEUTRAL = "Tonight we debate raising the minimum wage."        # polarity 0


class TestRelationPrompt(unittest.TestCase):
    def test_placeholders_and_real_case_last(self):
        self.assertEqual(placeholders("relation_v1"),
                         {"target", "speaker_a", "argument_a", "speaker_b", "argument_b"})
        text = render("relation_v1", target="T", speaker_a="SA", argument_a="AA", speaker_b="SB", argument_b="BB")
        self.assertTrue(text.rstrip().endswith("Argument B: BB"))
        self.assertIn("Speaker of A: SA", text)
        for token in ("{target}", "{argument_a}", "{argument_b}", "{speaker_a}", "{speaker_b}"):
            self.assertNotIn(token, text)
        self.assertIn('"relation": "Support"', text)               # literal JSON survives

    def test_missing_speakers_are_marked_unknown(self):
        text = build_messages("t", "a", "b")[0]["content"]
        self.assertIn("Speaker of A: (unknown)", text)
        self.assertIn("Speaker of B: (unknown)", text)

    def test_schema_only_allows_three_relations(self):
        self.assertEqual(RelationResult.model_json_schema()["$defs"]["Relation"]["enum"], ["Support", "Attack", "None"])


class TestClassify(unittest.TestCase):
    def test_support_attack_none(self):
        llm = FakeLLM()
        self.assertEqual(classify_relation(llm, "t", POS, POS).relation, Relation.SUPPORT)
        self.assertEqual(classify_relation(llm, "t", POS, NEG).relation, Relation.ATTACK)
        self.assertEqual(classify_relation(llm, "t", POS, NEUTRAL).relation, Relation.NONE)

    def test_argument_order_in_the_prompt(self):
        llm = FakeLLM()
        classify_relation(llm, "t", "FIRST ARGUMENT", "SECOND ARGUMENT", "Ana", "Ben")
        prompt = llm.calls[0]
        self.assertLess(prompt.index("Argument A: FIRST ARGUMENT"), prompt.index("Argument B: SECOND ARGUMENT"))
        self.assertIn("Speaker of A: Ana", prompt)
        self.assertIn("Speaker of B: Ben", prompt)


class TestFindRelation(unittest.TestCase):
    def setUp(self):
        self.previous = [ArgItem(0, "A", NEG), ArgItem(1, "B", POS), ArgItem(2, "A", NEUTRAL)]
        self.current = ArgItem(3, "B", POS)

    def test_nearest_first_and_stops_at_the_first_link(self):
        llm = FakeLLM()
        result, linked = find_relation(llm, "t", self.current, self.previous, window=3)
        # nearest (index 2, neutral) is None, next (index 1, positive) supports: stop there, never ask about index 0
        self.assertEqual((result.relation, linked.index), (Relation.SUPPORT, 1))
        self.assertEqual(len(llm.calls), 2)
        self.assertIn(f"Argument A: {NEUTRAL}", llm.calls[0])
        self.assertIn(f"Argument A: {POS}", llm.calls[1])

    def test_window_limits_how_far_back_it_looks(self):
        llm = FakeLLM()
        result, linked = find_relation(llm, "t", self.current, self.previous, window=1)
        self.assertEqual((result.relation, linked), (Relation.NONE, None))
        self.assertEqual(len(llm.calls), 1)

    def test_attack_is_found_too(self):
        result, linked = find_relation(FakeLLM(), "t", ArgItem(4, "B", NEG), [ArgItem(3, "A", POS)])
        self.assertEqual((result.relation, linked.index), (Relation.ATTACK, 3))

    def test_nothing_to_relate_to(self):
        llm = FakeLLM()
        result, linked = find_relation(llm, "t", self.current, [], window=3)
        self.assertEqual((result.relation, linked, len(llm.calls)), (Relation.NONE, None, 0))
        self.assertIn("No earlier argument", result.explanation)

    def test_no_link_found_says_how_many_were_checked(self):
        result, linked = find_relation(FakeLLM(), "t", ArgItem(5, "B", NEUTRAL), self.previous, window=3)
        self.assertEqual((result.relation, linked), (Relation.NONE, None))
        self.assertIn("previous 3", result.explanation)


SAMPLE_TRANSCRIPT = "A: I support it. Workers deserve better pay.\nB: I disagree with that. Firms run on thin margins."


class TestPairSheet(unittest.TestCase):
    def setUp(self):
        self.segments = segment_transcript(SAMPLE_TRANSCRIPT, context_window=1)

    def test_pairs_are_nearest_first_and_never_look_before_the_start(self):
        rows = pairs_sheet_rows(self.segments, window=2)
        n = len(self.segments)
        self.assertEqual(len(rows), (n - 1) + (n - 2))              # 1 pair for index 1, 2 for each later index
        ids = [r["pair_id"] for r in rows]
        self.assertEqual(ids[:3], ["0-1", "1-2", "0-2"])            # for B=2: A=1 first, then A=0
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(r["a_index"] < r["b_index"] for r in rows))
        self.assertEqual(list(rows[0]), PAIR_COLUMNS)
        self.assertEqual({r["gold_relation"] for r in rows}, {""})

    def test_window_one_pairs_only_adjacent_sentences(self):
        rows = pairs_sheet_rows(self.segments, window=1)
        self.assertEqual([(r["a_index"], r["b_index"]) for r in rows],
                         [(i, i + 1) for i in range(len(self.segments) - 1)])

    def test_speakers_are_copied(self):
        rows = pairs_sheet_rows(self.segments, window=3)
        last = rows[-1]
        self.assertEqual(last["speaker_b"], "B")
        self.assertIn(last["speaker_a"], {"A", "B"})

    def test_normalise_relation(self):
        self.assertEqual(normalise_relation(" support "), "Support")
        self.assertEqual(normalise_relation("A"), "Attack")
        self.assertEqual(normalise_relation("none"), "None")
        self.assertIsNone(normalise_relation(""))
        with self.assertRaisesRegex(ValueError, "Support / Attack / None"):
            normalise_relation("suport")

    def _sheet(self, tmp, cells):
        rows = pairs_sheet_rows(self.segments, window=3)[:len(cells)]
        for row, cell in zip(rows, cells):
            row["gold_relation"] = cell
        path = Path(tmp) / "pairs.csv"
        write_csv(rows, path, PAIR_COLUMNS)
        return path

    def test_load_normalises_and_skips_blanks(self):
        with tempfile.TemporaryDirectory() as d:
            rows, blank = load_pair_sheet(self._sheet(d, ["support", "", "A", "None"]))
        self.assertEqual([r["gold"] for r in rows], ["Support", "Attack", "None"])
        self.assertEqual(blank, 1)

    def test_load_reports_every_bad_cell_with_its_row(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._sheet(d, ["Support", "suport", "Attack", "atack"])
            with self.assertRaises(ValueError) as cm:
                load_pair_sheet(path)
        self.assertIn("row 3", str(cm.exception))
        self.assertIn("row 5", str(cm.exception))

    def test_agreement_is_matched_on_pair_id(self):
        rows = pairs_sheet_rows(self.segments, window=3)[:5]
        a = [{**r, "gold_relation": c} for r, c in zip(rows, ["Support", "Attack", "None", "Support", "Attack"])]
        b = [{**r, "gold_relation": c} for r, c in zip(reversed(rows), reversed(["Support", "Attack", "None", "Attack", "Attack"]))]
        result = compare_annotators(a, b, column="gold_relation")       # b is in the opposite order: matching must not care
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["agreement"], 0.8)
        self.assertEqual(len(result["disagreements"]), 1)
        self.assertEqual(result["disagreements"][0]["index"], rows[3]["pair_id"])
        self.assertIn("->", result["disagreements"][0]["sentence"])


class TestPredictRelations(unittest.TestCase):
    PAIRS = [
        {"pair_id": "0-1", "speaker_a": "A", "speaker_b": "B", "sentence_a": POS, "sentence_b": NEG, "gold": "Attack"},
        {"pair_id": "1-2", "speaker_a": "B", "speaker_b": "B", "sentence_a": NEG, "sentence_b": NEG, "gold": "Support"},
        {"pair_id": "0-2", "speaker_a": "A", "speaker_b": "B", "sentence_a": POS, "sentence_b": NEUTRAL, "gold": "Attack"},
    ]

    def test_rows_carry_gold_and_prediction(self):
        rows = predict_relations(FakeLLM(), "the minimum wage", self.PAIRS)
        self.assertEqual([r["pred"] for r in rows], ["Attack", "Support", "None"])
        self.assertEqual([r["gold"] for r in rows], ["Attack", "Support", "Attack"])
        report = classification_report([r["gold"] for r in rows], [r["pred"] for r in rows], ["Support", "Attack", "None"])
        self.assertAlmostEqual(report["accuracy"], 2 / 3)

    def test_a_failed_call_is_recorded_and_the_run_continues(self):
        rows = predict_relations(FakeLLM(fail_when=lambda p: NEUTRAL in p), "t", self.PAIRS)
        self.assertEqual(rows[2]["pred"], "ERROR")
        self.assertIn("simulated", rows[2]["error"])
        self.assertEqual(rows[0]["pred"], "Attack")

    def test_speakers_reach_the_prompt(self):
        llm = FakeLLM()
        predict_relations(llm, "t", self.PAIRS[:1])
        self.assertIn("Speaker of A: A", llm.calls[0])
        self.assertIn("Speaker of B: B", llm.calls[0])


def run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=ROOT)


class TestScripts(unittest.TestCase):
    def test_sheet_generator_kinds_and_protection(self):
        with tempfile.TemporaryDirectory() as d:
            comp, pairs = Path(d) / "c.csv", Path(d) / "p.csv"
            only_pairs = run("make_annotation_sheet.py", "--kind", "pairs", "--out", str(comp), "--pairs-out", str(pairs))
            self.assertEqual(only_pairs.returncode, 0, only_pairs.stderr)
            self.assertTrue(pairs.exists())
            self.assertFalse(comp.exists())                                  # --kind pairs wrote nothing else
            self.assertGreaterEqual(len(read_csv(pairs)), 40)

            # "both" must refuse because pairs.csv exists, and must not half-write components.csv
            both = run("make_annotation_sheet.py", "--out", str(comp), "--pairs-out", str(pairs))
            self.assertNotEqual(both.returncode, 0)
            self.assertIn("--force", both.stderr)
            self.assertFalse(comp.exists())

            forced = run("make_annotation_sheet.py", "--out", str(comp), "--pairs-out", str(pairs), "--force")
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertTrue(comp.exists())

    def test_typo_in_a_pairs_sheet_gives_a_plain_message(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.csv"
            write_csv([{"pair_id": "0-1", "a_index": 0, "b_index": 1, "speaker_a": "A", "speaker_b": "B",
                        "sentence_a": "One sentence here.", "sentence_b": "Another sentence here.",
                        "gold_relation": "suport"}], bad, PAIR_COLUMNS)
            for script, args in [("run_relation_eval.py", ["--pairs", str(bad), "--target", "x"]),
                                 ("annotator_agreement.py", [str(bad), str(bad), "--column", "gold_relation"])]:
                with self.subTest(script=script):
                    p = run(script, *args)
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn("suport", p.stderr)
                    self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
