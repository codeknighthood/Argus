"""Phase 3 tests: argument extraction, argument-detection scoring, annotation tooling.

Run from the project root:  python -m unittest discover -s tests -t . -v
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from argus.annotation import (
    COMPONENT_COLUMNS, compare_annotators, components_sheet_rows, interpret_kappa, load_component_sheet,
    normalise_component,
)
from argus.data import Record
from argus.evaluation import predict_arguments, read_csv, write_csv
from argus.extract import build_messages, extract_argument
from argus.prompting import placeholders, render
from argus.schema import ComponentLabel, ExtractionResult, Stance
from argus.segment import segment_transcript

from .fakes import FakeLLM

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "sample_transcript.txt"


class StubLLM:
    """Returns a fixed ExtractionResult, to test how extract_argument tidies it."""

    def __init__(self, component, restated):
        self.result = ExtractionResult(component=component, identified_argument=restated, explanation="x")

    def structured(self, messages, schema):
        return self.result


class TestExtraction(unittest.TestCase):
    def test_prompt_renders_and_keeps_real_case_last(self):
        self.assertEqual(placeholders("extract_v1"), {"target", "context", "statement"})
        text = render("extract_v1", target="TT", context="CC", statement="SS")
        self.assertTrue(text.rstrip().endswith("Statement: SS"))
        for token in ("{target}", "{context}", "{statement}"):
            self.assertNotIn(token, text)
        self.assertIn('"component": "Claim"', text)        # literal JSON survives

    def test_empty_context_is_marked(self):
        self.assertIn("Context (may be empty): (none)", build_messages("t", "s")[0]["content"])

    def test_schema_only_allows_three_components(self):
        enum = ExtractionResult.model_json_schema()["$defs"]["ComponentLabel"]["enum"]
        self.assertEqual(enum, ["Claim", "Premise", "None"])

    def test_is_argument(self):
        self.assertTrue(ExtractionResult(component="Claim", identified_argument="x", explanation="e").is_argument)
        self.assertFalse(ExtractionResult(component="None", identified_argument="", explanation="e").is_argument)

    def test_none_always_has_an_empty_restatement(self):
        r = extract_argument(StubLLM(ComponentLabel.NONE, "leftover text"), "t", "Thanks all.")
        self.assertEqual((r.component, r.identified_argument), (ComponentLabel.NONE, ""))

    def test_argument_without_restatement_falls_back_to_the_sentence(self):
        r = extract_argument(StubLLM(ComponentLabel.PREMISE, "   "), "t", "  Small firms run on thin margins. ")
        self.assertEqual(r.identified_argument, "Small firms run on thin margins.")

    def test_restatement_is_stripped(self):
        r = extract_argument(StubLLM(ComponentLabel.CLAIM, "  Raise it. "), "t", "I agree.")
        self.assertEqual(r.identified_argument, "Raise it.")

    def test_with_the_rule_based_fake(self):
        llm = FakeLLM()
        claim = extract_argument(llm, "the minimum wage", "I support raising it.")
        premise = extract_argument(llm, "the minimum wage", "Small businesses run on thin margins.")
        none = extract_argument(llm, "the minimum wage", "Tonight we debate raising the minimum wage.")
        self.assertEqual([claim.component, premise.component, none.component],
                         [ComponentLabel.CLAIM, ComponentLabel.PREMISE, ComponentLabel.NONE])
        self.assertEqual(none.identified_argument, "")


def rec(i, statement, is_argument, stance=None):
    return Record(id=f"r{i}", source="test", topic="minimum_wage", target="the minimum wage",
                  statement=statement, is_argument=is_argument, gold_stance=stance)


class TestArgumentDetectionScoring(unittest.TestCase):
    def test_rows_carry_gold_pred_and_role(self):
        records = [
            rec(1, "Small businesses run on thin margins.", True, Stance.AGAINST),
            rec(2, "The bill was debated on Tuesday afternoon.", False, Stance.NEUTRAL),
            rec(3, "A higher wage floor helps working families.", False, Stance.NEUTRAL),   # gold says no; model says yes
            rec(4, "No gold label here at all.", None),
        ]
        rows = predict_arguments(FakeLLM(), records)
        self.assertEqual([r["gold"] for r in rows], ["Argument", "NoArgument", "NoArgument", None])
        self.assertEqual([r["pred"] for r in rows], ["Argument", "NoArgument", "Argument", "NoArgument"])
        self.assertEqual(rows[0]["component"], "Premise")
        self.assertEqual(rows[0]["gold_stance"], "Against")
        self.assertEqual(rows[1]["identified_argument"], "")

    def test_a_failed_call_is_recorded_and_the_run_continues(self):
        llm = FakeLLM(fail_when=lambda p: "Tuesday" in p)
        rows = predict_arguments(llm, [rec(1, "The bill was debated on Tuesday afternoon.", False),
                                       rec(2, "I support raising it.", True)])
        self.assertEqual(rows[0]["pred"], "ERROR")
        self.assertIn("simulated", rows[0]["error"])
        self.assertEqual(rows[1]["pred"], "Argument")


class TestAnnotation(unittest.TestCase):
    def test_normalise(self):
        self.assertEqual(normalise_component(" claim "), "Claim")
        self.assertEqual(normalise_component("P"), "Premise")
        self.assertEqual(normalise_component("none"), "None")
        self.assertIsNone(normalise_component("   "))
        self.assertIsNone(normalise_component(None))
        with self.assertRaises(ValueError):
            normalise_component("premice")

    def test_sample_transcript_is_usable(self):
        segments = segment_transcript(SAMPLE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(segments), 20)
        speakers = {s.speaker for s in segments}
        self.assertTrue({"MODERATOR", "DR. SMITH", "MR. JONES", "MS. PATEL", "AUDIENCE"} <= speakers)

    def test_sheet_rows(self):
        segments = segment_transcript("A: I support it. B: I disagree with that, sadly.", context_window=1)
        rows = components_sheet_rows(segments)
        self.assertEqual(list(rows[0]), COMPONENT_COLUMNS)
        self.assertEqual({r["gold_component"] for r in rows}, {""})
        self.assertEqual([r["index"] for r in rows], list(range(len(rows))))

    def _sheet(self, tmp, cells):
        path = Path(tmp) / "s.csv"
        rows = [{"index": i, "speaker": "A", "sentence": f"Sentence number {i} here.", "context": "", "gold_component": c}
                for i, c in enumerate(cells)]
        write_csv(rows, path, COMPONENT_COLUMNS)
        return path

    def test_load_sheet_normalises_and_skips_blanks(self):
        with tempfile.TemporaryDirectory() as d:
            rows, blank = load_component_sheet(self._sheet(d, ["claim", "", "P", "None"]))
        self.assertEqual([r["gold"] for r in rows], ["Claim", "Premise", "None"])
        self.assertEqual(blank, 1)

    def test_load_sheet_reports_every_bad_cell_with_its_row_number(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._sheet(d, ["Claim", "oops", "Premise", "nope"])
            with self.assertRaises(ValueError) as cm:
                load_component_sheet(path)
        message = str(cm.exception)
        self.assertIn("row 3", message)      # header is row 1, so index 1 is spreadsheet row 3
        self.assertIn("row 5", message)

    def test_load_sheet_without_the_column(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            write_csv([{"index": 0, "sentence": "s"}], path, ["index", "sentence"])
            with self.assertRaisesRegex(ValueError, "gold_component"):
                load_component_sheet(path)

    def test_compare_annotators(self):
        a = [{"index": i, "sentence": f"s{i}", "gold_component": c} for i, c in enumerate(["Claim", "Premise", "None", "Claim", ""])]
        b = [{"index": i, "sentence": f"s{i}", "gold_component": c} for i, c in enumerate(["Claim", "Premise", "None", "Premise", "Claim"])]
        r = compare_annotators(a, b)
        self.assertEqual(r["n"], 4)                       # the row one person left blank is ignored
        self.assertAlmostEqual(r["agreement"], 0.75)
        self.assertEqual(r["disagreements"], [{"index": "3", "a": "Claim", "b": "Premise", "sentence": "s3"}])
        self.assertGreater(r["kappa"], 0.5)
        self.assertEqual(compare_annotators(a, a)["kappa"], 1.0)

    def test_interpret_kappa(self):
        self.assertEqual(interpret_kappa(-0.1), "worse than chance")
        self.assertEqual(interpret_kappa(0.3), "fair")
        self.assertEqual(interpret_kappa(0.7), "substantial")
        self.assertEqual(interpret_kappa(0.9), "almost perfect")


class TestScriptsFailPolitely(unittest.TestCase):
    """A typo in the sheet must give a plain message, not a Python traceback."""

    def test_bad_cell_in_evaluation_and_agreement_scripts(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.csv"
            write_csv([{"index": 0, "speaker": "A", "sentence": "Some sentence here.", "context": "",
                        "gold_component": "premice"}], bad, COMPONENT_COLUMNS)
            for script, args in [("run_annotated_eval.py", ["--components", str(bad), "--target", "x"]),
                                 ("annotator_agreement.py", [str(bad), str(bad)])]:
                with self.subTest(script=script):
                    p = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                                       capture_output=True, text=True, cwd=ROOT)
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn("premice", p.stderr)
                    self.assertNotIn("Traceback", p.stderr)


class TestSheetScript(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "make_annotation_sheet.py"), *args],
                              capture_output=True, text=True, cwd=ROOT)

    def test_creates_the_sheet_and_refuses_to_overwrite_annotations(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "components.csv"
            first = self.run_script("--kind", "components", "--out", str(out))
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertGreaterEqual(len(read_csv(out)), 20)
            second = self.run_script("--kind", "components", "--out", str(out))
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("--force", second.stderr)
            self.assertEqual(self.run_script("--kind", "components", "--out", str(out), "--force").returncode, 0)


if __name__ == "__main__":
    unittest.main()
