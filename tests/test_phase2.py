"""Phase 2 tests (stance strategies, metrics, evaluation helpers, LLM client).

Run from the project root:  python -m unittest discover -s tests -t . -v
No Ollama and no dataset needed: a rule-based fake stands in for the model.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from argus.data import Record
from argus.evaluation import (
    Progress, errors_only, load_split, per_topic_accuracy, predict_records, read_csv, read_jsonl, take,
    write_csv, write_jsonl,
)
from argus.llm import LLMClient
from argus.metrics import classification_report, cohen_kappa, format_report
from argus.prompting import placeholders, render
from argus.schema import Stance, StanceCoT, StanceResult
from argus.stance import STRATEGIES, build_messages, predict_stance

from .fakes import FakeLLM, FakeOllama

LABELS = ["Favor", "Against", "Neutral"]


def rec(i: int, statement: str, gold: Stance, topic: str = "minimum_wage") -> Record:
    return Record(id=f"r{i}", source="test", topic=topic, target="the minimum wage",
                  statement=statement, gold_stance=gold)


class TestPrompting(unittest.TestCase):
    def test_every_stance_prompt_renders_fully(self):
        for strategy, (prompt_name, _) in STRATEGIES.items():
            with self.subTest(strategy=strategy):
                self.assertEqual(placeholders(prompt_name), {"target", "context", "statement"})
                text = render(prompt_name, target="T-VALUE", context="C-VALUE", statement="S-VALUE")
                for token in ("{target}", "{context}", "{statement}"):
                    self.assertNotIn(token, text)
                self.assertIn("T-VALUE", text)
                self.assertIn('"stance"', text)          # literal JSON braces survive rendering

    def test_inserted_text_is_never_substituted_again(self):
        text = render("stance_v1", target="x", context="y", statement="tricky {context} and {target}")
        self.assertIn("tricky {context} and {target}", text)

    def test_real_case_comes_last_in_few_shot(self):
        text = render("stance_fewshot_v1", target="TT", context="CC", statement="SS")
        self.assertTrue(text.rstrip().endswith("Statement: SS"))
        self.assertGreater(text.count("Statement:"), 3)   # the worked examples are there too


class TestStance(unittest.TestCase):
    def test_all_strategies_return_a_stance_result(self):
        llm = FakeLLM()
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                r = predict_stance(llm, strategy, "the minimum wage", "A higher wage floor means layoffs.")
                self.assertIsInstance(r, StanceResult)
                self.assertEqual(r.stance, Stance.AGAINST)

    def test_cot_reasoning_becomes_the_explanation(self):
        r = predict_stance(FakeLLM(), "cot", "the minimum wage", "I support it.")
        self.assertEqual(r.stance, Stance.FAVOR)
        self.assertTrue(r.explanation.startswith("fake reasoning"))

    def test_context_reaches_the_prompt_and_empty_context_is_marked(self):
        llm = FakeLLM()
        predict_stance(llm, "zero_shot", "t", "s", context="Earlier sentence.")
        predict_stance(llm, "zero_shot", "t", "s")
        self.assertIn("Context (may be empty): Earlier sentence.", llm.calls[0])
        self.assertIn("Context (may be empty): (none)", llm.calls[1])

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            build_messages("magic", "t", "s")

    def test_schemas_only_allow_the_three_labels(self):
        for schema in (StanceResult, StanceCoT):
            self.assertEqual(schema.model_json_schema()["$defs"]["Stance"]["enum"], LABELS)


class TestMetrics(unittest.TestCase):
    def test_known_example_by_hand(self):
        gold = ["Favor", "Favor", "Against", "Against", "Neutral", "Neutral"]
        pred = ["Favor", "Against", "Against", "Against", "Neutral", "Favor"]
        rep = classification_report(gold, pred, LABELS)
        self.assertAlmostEqual(rep["accuracy"], 4 / 6)
        self.assertAlmostEqual(rep["per_class"]["Favor"]["precision"], 0.5)
        self.assertAlmostEqual(rep["per_class"]["Favor"]["recall"], 0.5)
        self.assertAlmostEqual(rep["per_class"]["Against"]["precision"], 2 / 3)
        self.assertAlmostEqual(rep["per_class"]["Against"]["recall"], 1.0)
        self.assertAlmostEqual(rep["per_class"]["Neutral"]["f1"], 2 / 3)
        self.assertAlmostEqual(rep["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)
        self.assertEqual(rep["confusion"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])   # rows gold, cols predicted

    def test_failed_predictions_count_as_wrong(self):
        rep = classification_report(["Favor", "Against"], ["Favor", "ERROR"], LABELS)
        self.assertAlmostEqual(rep["accuracy"], 0.5)
        self.assertEqual(rep["unparsed"], 1)
        self.assertAlmostEqual(rep["per_class"]["Against"]["recall"], 0.0)

    def test_perfect_and_empty(self):
        rep = classification_report(LABELS, LABELS, LABELS)
        self.assertEqual((rep["accuracy"], rep["macro_f1"]), (1.0, 1.0))
        self.assertEqual(classification_report([], [], LABELS)["accuracy"], 0.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            classification_report(["Favor"], [], LABELS)

    def test_kappa(self):
        self.assertAlmostEqual(cohen_kappa([1, 1, 0, 0], [1, 0, 0, 1]), 0.0)     # agreement at chance level
        self.assertAlmostEqual(cohen_kappa(["a", "b", "a"], ["a", "b", "a"]), 1.0)
        self.assertAlmostEqual(cohen_kappa(["a", "a"], ["a", "a"]), 1.0)          # constant sequences
        self.assertLess(cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]), 0)

    def test_format_report_runs(self):
        rep = classification_report(["Favor", "Against"], ["Favor", "Favor"], LABELS)
        text = format_report(rep)
        self.assertIn("accuracy 0.500", text)
        self.assertIn("gold \\ pred", text)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.records = [
            rec(1, "A higher wage floor helps working families.", Stance.FAVOR),
            rec(2, "It will cause layoffs at small firms.", Stance.AGAINST),
            rec(3, "The bill was debated on Tuesday afternoon.", Stance.NEUTRAL, topic="other"),
            rec(4, "Employers will force cuts in hours.", Stance.FAVOR, topic="other"),   # model will get this wrong
        ]

    def test_predict_records(self):
        rows = predict_records(FakeLLM(), self.records, "zero_shot")
        self.assertEqual([r["pred"] for r in rows], ["Favor", "Against", "Neutral", "Against"])
        self.assertEqual([r["gold"] for r in rows], ["Favor", "Against", "Neutral", "Favor"])
        self.assertEqual(len(errors_only(rows)), 1)
        self.assertEqual(per_topic_accuracy(rows), {"minimum_wage": (1.0, 2), "other": (0.5, 2)})

    def test_one_failure_does_not_stop_the_run(self):
        llm = FakeLLM(fail_when=lambda p: "Tuesday" in p)
        rows = predict_records(llm, self.records, "few_shot")
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[2]["pred"], "ERROR")
        self.assertIn("simulated", rows[2]["error"])
        self.assertEqual(rows[0]["pred"], "Favor")

    def test_take_is_reproducible(self):
        many = [rec(i, f"Statement number {i} is here.", Stance.FAVOR) for i in range(50)]
        a, b = take(many, 10, seed=1), take(many, 10, seed=1)
        self.assertEqual([r.id for r in a], [r.id for r in b])
        self.assertEqual(len(a), 10)
        self.assertEqual([r.id for r in a], sorted((r.id for r in a), key=lambda s: int(s[1:])))  # order kept
        self.assertNotEqual([r.id for r in a], [r.id for r in take(many, 10, seed=2)])
        self.assertEqual(len(take(many, None)), 50)
        self.assertEqual(len(take(many, 500)), 50)

    def test_file_roundtrips_keep_unicode(self):
        rows = [{"id": "1", "statement": "Café prices — हिन्दी", "gold": "Favor"}]
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(rows, Path(d) / "x.jsonl")
            self.assertEqual(read_jsonl(Path(d) / "x.jsonl"), rows)
            write_csv(rows, Path(d) / "x.csv", ["id", "statement", "gold"])
            self.assertEqual(read_csv(Path(d) / "x.csv"), rows)

    def test_load_split_tells_you_what_to_do(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError) as cm:
            load_split(Path(d), "dev")
        self.assertIn("prepare_data.py", str(cm.exception))

    def test_progress_runs_quietly(self):
        import io
        buf = io.StringIO()
        p = Progress(2, "x", stream=buf)
        p.update()
        p.update()
        p.close()
        self.assertIn("2/2 (100%)", buf.getvalue())


def make_client(fake: FakeOllama, tmp: str, max_retries: int = 3) -> LLMClient:
    client = LLMClient({"model": "qwen2.5:7b", "temperature": 0.0, "num_ctx": 4096,
                        "max_retries": max_retries, "cache_dir": tmp})
    client.client = fake
    return client


GOOD = json.dumps({"stance": "Against", "explanation": "e"})


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.msgs = [{"role": "user", "content": "hello"}]

    def tearDown(self):
        self.tmp.cleanup()

    def test_ready_check_messages(self):
        down = make_client(FakeOllama(list_error=ConnectionError("refused")), self.tmp.name)
        with self.assertRaisesRegex(RuntimeError, "Cannot reach Ollama"):
            down.check_ready()
        missing = make_client(FakeOllama(models=["llama3.1:8b"]), self.tmp.name)
        with self.assertRaisesRegex(RuntimeError, "ollama pull qwen2.5:7b"):
            missing.check_ready()
        make_client(FakeOllama(models=["qwen2.5:7b"]), self.tmp.name).check_ready()
        make_client(FakeOllama(models=["qwen2.5:7b:latest"]), self.tmp.name).check_ready()

    def test_retry_on_invalid_output_then_cache(self):
        fake = FakeOllama(replies=[json.dumps({"stance": "Neither", "explanation": "x"}), GOOD])
        client = make_client(fake, self.tmp.name)
        self.assertEqual(client.structured(self.msgs, StanceResult).stance, Stance.AGAINST)
        self.assertEqual(len(fake.chat_calls), 2)
        self.assertIn("not valid", fake.chat_calls[1]["messages"][-1]["content"])   # the retry says what was wrong
        client.structured(self.msgs, StanceResult)
        self.assertEqual(len(fake.chat_calls), 2)                                  # second call came from the cache

    def test_schema_and_options_are_sent_to_ollama(self):
        fake = FakeOllama(replies=[GOOD])
        make_client(fake, self.tmp.name).structured(self.msgs, StanceResult)
        call = fake.chat_calls[0]
        self.assertEqual(call["format"]["$defs"]["Stance"]["enum"], LABELS)
        self.assertEqual(call["options"], {"temperature": 0.0, "num_ctx": 4096})

    def test_invalid_replies_are_not_cached(self):
        fake = FakeOllama(replies=["not json"] * 3 + [GOOD])
        client = make_client(fake, self.tmp.name)
        with patch("argus.llm.time.sleep"), self.assertRaises(RuntimeError):
            client.structured(self.msgs, StanceResult)
        self.assertEqual(client.structured(self.msgs, StanceResult).stance, Stance.AGAINST)
        self.assertEqual(len(fake.chat_calls), 4)

    def test_server_errors_are_retried_then_reported(self):
        fake = FakeOllama(chat_error=ConnectionError("boom"))
        client = make_client(fake, self.tmp.name, max_retries=3)
        with patch("argus.llm.time.sleep"), self.assertRaisesRegex(RuntimeError, "failed after 3 attempts"):
            client.structured(self.msgs, StanceResult)
        self.assertEqual(len(fake.chat_calls), 3)

    def test_corrupt_cache_entry_is_asked_again(self):
        fake = FakeOllama(replies=[GOOD])
        client = make_client(fake, self.tmp.name)
        client._cache_path(self.msgs, StanceResult).write_text("{broken", encoding="utf-8")
        self.assertEqual(client.structured(self.msgs, StanceResult).stance, Stance.AGAINST)
        self.assertEqual(len(fake.chat_calls), 1)

    def test_cache_is_separate_per_schema_and_per_model(self):
        client = make_client(FakeOllama(), self.tmp.name)
        a = client._cache_path(self.msgs, StanceResult)
        self.assertNotEqual(a, client._cache_path(self.msgs, StanceCoT))
        client.model = "other"
        self.assertNotEqual(a, client._cache_path(self.msgs, StanceResult))


if __name__ == "__main__":
    unittest.main()
