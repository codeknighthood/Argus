"""Phase 5 tests: the integrated pipeline, structured output files, script helpers.

Run from the project root:  python -m unittest discover -s tests -t . -v
The rule-based fake model stands in for Ollama, so these check ARGUS's own logic.
"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from argus.evaluation import read_csv
from argus.llm import ensure_ready, load_config
from argus.output import (
    COLUMNS, SYNOPSIS_COLUMNS, format_summary, record_to_row, summarize, to_markdown, write_csv, write_json,
)
from argus.pipeline import ArgusPipeline, PipelineConfig
from argus.schema import ArgumentRecord, Component, Relation, Stance
from argus.segment import segment_transcript

from .fakes import FakeLLM, FakeOllama
from .test_phase2 import make_client

ROOT = Path(__file__).resolve().parent.parent

DEMO = """\
MODERATOR: Welcome everyone. Tonight we debate raising the minimum wage.
DR. SMITH: I support it. Full-time workers should not live in poverty.
Studies show modest increases barely affect employment.
MR. JONES: I disagree. Small businesses run on thin margins. A higher wage floor means layoffs.
"""
TARGET = "the minimum wage"

# Markers that identify which stage a prompt belongs to
EXTRACTION, STANCE, RELATION = "argument mining assistant", "Decide the stance", "how two arguments in a policy debate"


def count(llm: FakeLLM, marker: str) -> int:
    return sum(marker in p for p in llm.calls)


class TestConfig(unittest.TestCase):
    def test_defaults_and_validation(self):
        c = PipelineConfig()
        self.assertEqual((c.stance_strategy, c.stance_on, c.min_tokens), ("few_shot", "all", 3))
        for bad in ({"stance_strategy": "magic"}, {"stance_on": "some"}, {"relation_window": -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PipelineConfig(**bad)

    def test_from_config_uses_the_yaml_section_then_overrides(self):
        cfg = {"pipeline": {"stance_strategy": "cot", "relation_window": 5, "unknown_key": 1}}
        c = PipelineConfig.from_config(cfg, relation_window=None, context_window=4)
        self.assertEqual((c.stance_strategy, c.relation_window, c.context_window), ("cot", 5, 4))   # None is ignored

    def test_an_old_config_without_the_section_still_works(self):
        self.assertEqual(PipelineConfig.from_config({"model": "x"}), PipelineConfig())

    def test_the_shipped_config_yaml_is_valid(self):
        PipelineConfig.from_config(load_config())


class TestPipeline(unittest.TestCase):
    def run_demo(self, llm=None, **config):
        llm = llm or FakeLLM()
        pipeline = ArgusPipeline(llm, PipelineConfig(**config))
        return llm, pipeline.run_transcript(DEMO, TARGET)

    def test_full_run_produces_the_structured_output(self):
        _, recs = self.run_demo()
        self.assertEqual([r.segment_index for r in recs], list(range(6)))
        self.assertEqual([r.speaker_context for r in recs],
                         ["MODERATOR", "DR. SMITH", "DR. SMITH", "DR. SMITH", "MR. JONES", "MR. JONES"])
        self.assertEqual([r.is_argument for r in recs], [False, True, True, True, True, True])
        self.assertEqual([r.component for r in recs],
                         [None, Component.CLAIM] + [Component.PREMISE] * 4)
        self.assertEqual([r.stance for r in recs],
                         [Stance.NEUTRAL, Stance.FAVOR, Stance.FAVOR, Stance.FAVOR, Stance.AGAINST, Stance.AGAINST])
        # relations: each argument is linked to the nearest earlier argument it supports or attacks
        self.assertEqual([r.relation for r in recs],
                         [None, Relation.NONE, Relation.SUPPORT, Relation.SUPPORT, Relation.ATTACK, Relation.SUPPORT])
        self.assertEqual([r.related_to_index for r in recs], [None, None, 1, 2, 3, 4])
        self.assertEqual(recs[4].related_to, "Studies show modest increases barely affect employment.")
        self.assertTrue(all(r.error is None for r in recs))
        self.assertTrue(all(r.policy_target == TARGET for r in recs))

    def test_non_arguments_have_no_restatement_or_relation(self):
        _, recs = self.run_demo()
        self.assertIsNone(recs[0].identified_argument)
        self.assertIsNone(recs[0].relation)
        self.assertEqual(recs[1].identified_argument, "I support it.")

    def test_stance_on_arguments_skips_the_rest_and_says_so(self):
        llm, recs = self.run_demo(stance_on="arguments")
        self.assertEqual(count(llm, STANCE), 5)                      # the moderator's sentence got no stance call
        self.assertIsNone(recs[0].stance)
        self.assertIn("not evaluated", recs[0].explanation)
        self.assertEqual(recs[1].stance, Stance.FAVOR)

    def test_stance_on_all_asks_for_every_sentence(self):
        llm, _ = self.run_demo(stance_on="all")
        self.assertEqual((count(llm, EXTRACTION), count(llm, STANCE)), (6, 6))

    def test_relation_window_zero_turns_the_stage_off(self):
        llm, recs = self.run_demo(relation_window=0)
        self.assertEqual(count(llm, RELATION), 0)
        self.assertTrue(all(r.relation is None for r in recs))
        self.assertTrue(recs[1].is_argument)                         # everything else still works

    def test_relation_window_limits_how_far_back_it_looks(self):
        llm1, _ = self.run_demo(relation_window=1)
        llm3, _ = self.run_demo(relation_window=3)
        self.assertLessEqual(count(llm1, RELATION), count(llm3, RELATION))
        self.assertEqual(count(llm1, RELATION), 4)                   # one check per argument that has a predecessor

    def test_strategy_is_used(self):
        llm, _ = self.run_demo(stance_strategy="cot")
        self.assertTrue(any('"reasoning"' in p for p in llm.calls if STANCE in p))
        llm, _ = self.run_demo(stance_strategy="zero_shot")
        self.assertFalse(any("Examples (other topics" in p for p in llm.calls if STANCE in p))

    def test_context_is_passed_and_can_be_switched_off(self):
        llm, _ = self.run_demo(context_window=2)
        self.assertTrue(any("Context (may be empty): MODERATOR: Tonight we debate" in p for p in llm.calls))
        llm, _ = self.run_demo(context_window=0)
        self.assertFalse(any("Context (may be empty): MODERATOR" in p for p in llm.calls))

    def test_speakers_can_be_switched_off(self):
        _, recs = self.run_demo(parse_speakers=False)
        self.assertTrue(all(r.speaker_context is None for r in recs))

    def test_min_tokens_keeps_short_sentences(self):
        _, with_short = self.run_demo(min_tokens=2)
        _, default = self.run_demo()
        kept = {r.original_statement for r in with_short} - {r.original_statement for r in default}
        self.assertEqual(kept, {"Welcome everyone.", "I disagree."})  # the two 2-word sentences in the demo

    def test_progress_callback(self):
        seen = []
        ArgusPipeline(FakeLLM()).run_transcript(DEMO, TARGET, lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(i, 6) for i in range(1, 7)])

    def test_empty_transcript(self):
        llm = FakeLLM()
        self.assertEqual(ArgusPipeline(llm).run_transcript("   \n\n", TARGET), [])
        self.assertEqual(llm.calls, [])

    def test_describe_records_the_settings(self):
        d = ArgusPipeline(FakeLLM(), PipelineConfig(stance_strategy="cot")).describe()
        self.assertEqual((d["model"], d["stance_strategy"], d["relation_window"]), ("fake-model", "cot", 3))
        self.assertIn("argus_version", d)


class TestFailureIsolation(unittest.TestCase):
    """One failed model call must cost one field of one row, never the whole run."""

    def run_with(self, fail_when):
        return ArgusPipeline(FakeLLM(fail_when=fail_when), PipelineConfig()).run_transcript(DEMO, TARGET)

    def test_extraction_failure(self):
        recs = self.run_with(lambda p: EXTRACTION in p and "layoffs" in p)
        bad = recs[5]
        self.assertIsNone(bad.is_argument)
        self.assertIn("extraction", bad.error)
        self.assertEqual(bad.stance, Stance.AGAINST)                 # stance still ran
        self.assertIsNone(bad.relation)                              # unknown whether it is an argument
        self.assertEqual(len(recs), 6)
        self.assertTrue(all(r.error is None for r in recs[:5]))

    def test_stance_failure(self):
        recs = self.run_with(lambda p: STANCE in p and "Full-time workers" in p)
        self.assertIsNone(recs[2].stance)
        self.assertIn("stance", recs[2].error)
        self.assertEqual(recs[2].component, Component.PREMISE)       # extraction survived
        self.assertEqual(recs[2].relation, Relation.SUPPORT)         # and so did the relation

    def test_relation_failure_does_not_block_later_links(self):
        recs = self.run_with(lambda p: RELATION in p and "Argument B: Studies show" in p)
        self.assertIsNone(recs[3].relation)
        self.assertIn("relation", recs[3].error)
        self.assertEqual(recs[3].stance, Stance.FAVOR)
        self.assertEqual((recs[4].relation, recs[4].related_to_index), (Relation.ATTACK, 3))   # row 3 is still linkable

    def test_everything_failing_still_returns_every_row(self):
        recs = self.run_with(lambda p: True)
        self.assertEqual(len(recs), 6)
        self.assertTrue(all(r.error and r.stance is None for r in recs))


class TestOutput(unittest.TestCase):
    def setUp(self):
        _, self.recs = TestPipeline().run_demo()

    def test_columns_start_with_the_synopsis_five_seven_fields(self):
        self.assertEqual(COLUMNS[:9], SYNOPSIS_COLUMNS)
        self.assertEqual(SYNOPSIS_COLUMNS[:3], ["policy_target", "original_statement", "identified_argument"])

    def test_row_uses_plain_text_and_blank_for_missing(self):
        row = record_to_row(self.recs[0])
        self.assertEqual(row["stance"], "Neutral")
        self.assertEqual((row["component"], row["relation"], row["related_to_index"]), ("", "", ""))
        self.assertEqual(record_to_row(self.recs[2])["relation"], "Support")

    def test_json_has_meta_and_null_for_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.json"
            write_json(self.recs, path, meta={"target": TARGET})
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["meta"]["target"], TARGET)
        self.assertIn("created", data["meta"])
        self.assertEqual(len(data["records"]), 6)
        self.assertIsNone(data["records"][0]["component"])
        self.assertEqual(data["records"][1]["component"], "Claim")

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.csv"
            write_csv(self.recs, path)
            rows = read_csv(path)
            with open(path, encoding="utf-8-sig", newline="") as f:
                header = next(csv.reader(f))
        self.assertEqual(header, COLUMNS)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[4]["relation"], "Attack")
        self.assertEqual(rows[4]["related_to_index"], "3")

    def test_unicode_survives(self):
        rec = ArgumentRecord(policy_target="t", original_statement="Café हिन्दी")
        with tempfile.TemporaryDirectory() as d:
            write_csv([rec], Path(d) / "u.csv")
            write_json([rec], Path(d) / "u.json")
            self.assertEqual(read_csv(Path(d) / "u.csv")[0]["original_statement"], rec.original_statement)
            self.assertIn(rec.original_statement,
                          json.loads((Path(d) / "u.json").read_text(encoding="utf-8"))["records"][0]["original_statement"])

    def test_markdown_table_shows_links_and_is_safe(self):
        table = to_markdown(self.recs)
        self.assertIn("| Attack -> #3 |", table)
        self.assertEqual(table.splitlines()[0].count("|"), 7)
        nasty = ArgumentRecord(policy_target="t", original_statement="a | b\nand a very long statement " * 8, segment_index=0)
        line = to_markdown([nasty]).splitlines()[2]
        self.assertEqual(line.count("|"), 7)                         # the pipe inside the text did not add a column
        self.assertNotIn("\n", line)
        self.assertIn("...", line)

    def test_summary(self):
        s = summarize(self.recs)
        self.assertEqual((s["sentences"], s["arguments"], s["claims"], s["premises"]), (6, 5, 1, 4))
        self.assertEqual(s["stance"], {"Neutral": 1, "Favor": 3, "Against": 2})
        self.assertEqual(s["relations"], {"None": 1, "Support": 3, "Attack": 1})
        self.assertEqual(s["rows_with_errors"], 0)
        text = format_summary(s)
        self.assertIn("5 (1 claims, 4 premises)", text)
        self.assertNotIn("errors", text)

    def test_summary_mentions_errors_when_there_are_some(self):
        recs = ArgusPipeline(FakeLLM(fail_when=lambda p: True)).run_transcript(DEMO, TARGET)
        self.assertIn("Rows with errors   : 6", format_summary(summarize(recs)))


class TestScripts(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), *args],
                              capture_output=True, text=True, cwd=ROOT)

    def test_a_custom_transcript_needs_a_target(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "debate.txt"
            t.write_text(DEMO, encoding="utf-8")
            p = self.run_script("--transcript", str(t))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--target", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_missing_transcript(self):
        p = self.run_script("--transcript", "no_such_file.txt", "--target", "x")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("Transcript not found", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_ensure_ready_exits_with_a_plain_message(self):
        down = make_client(FakeOllama(list_error=ConnectionError("refused")), tempfile.mkdtemp())
        with self.assertRaises(SystemExit) as cm:
            ensure_ready(down)
        self.assertIn("Cannot reach Ollama", str(cm.exception))
        missing = make_client(FakeOllama(models=["other:1b"]), tempfile.mkdtemp())
        with self.assertRaises(SystemExit) as cm:
            ensure_ready(missing)
        self.assertIn("ollama pull", str(cm.exception))
        ok = make_client(FakeOllama(), tempfile.mkdtemp())
        self.assertIs(ensure_ready(ok), ok)

    def test_the_sheet_and_the_pipeline_segment_the_same_way_by_default(self):
        """Annotation sheets and the pipeline must agree on sentence numbers, or Phase 6 comparisons would be off."""
        text = (ROOT / "data" / "sample" / "sample_transcript.txt").read_text(encoding="utf-8")
        cfg = PipelineConfig.from_config(load_config())
        sheet = segment_transcript(text, context_window=2, parse_speakers=True)           # make_annotation_sheet defaults
        pipe = segment_transcript(text, cfg.context_window, cfg.parse_speakers, cfg.min_tokens)
        self.assertEqual([s.sentence for s in sheet], [s.sentence for s in pipe])


if __name__ == "__main__":
    unittest.main()
