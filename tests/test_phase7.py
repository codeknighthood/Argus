"""Phase 7 tests: the VAST data source, the demo helpers, the setup checker and the Streamlit app.

Run from the project root:  python -m unittest discover -s tests -t . -v
No Ollama, no internet and no real data are needed: VAST files are small synthetic ones in the same format, the
download script talks to a local web server, and the app is run with the fake model.
"""
import csv
import functools
import http.server
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from unittest import mock

from argus.compare import run_stance_comparison
from argus.data import (
    Record, find_vast_files, load_jsonl, load_vast, sample_balanced, vast_topic_text,
)
from argus.demo import (
    argument_map_dot, argument_rows, condensed_view, merge_repeats, similarity, table_rows,
)
from argus.evaluation import OTHER_TOPICS, collapse_topics, per_group_accuracy, write_csv
from argus.pipeline import ArgusPipeline, PipelineConfig
from argus.report import stance_report_md
from argus.schema import ArgumentRecord, Component, Relation, Stance
from argus.setup_check import (
    check_config, check_data, check_ollama, check_packages, check_python, format_results, run_checks,
)

from .fakes import FakeLLM
from .test_phase5 import DEMO, TARGET
from .test_phase6 import make_records

ROOT = Path(__file__).resolve().parent.parent
VAST_COLUMNS = ["author", "post", "ori_topic", "new_topic", "label", "type_idx", "new_id", "topic_str"]


def run_script(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=ROOT)


def vast_rows(prefix="a", n_topics=6, per_topic=3):
    """A small synthetic VAST file: per topic some pro, con and extracted/corrected rows, plus synthetic neutrals."""
    rows = []
    for t in range(n_topics):
        topic = f"topic number {t}"
        for k in range(per_topic):
            for label, kind in (("1", "1"), ("0", "2")):
                rows.append({"author": "x", "post": f"{prefix} comment about {topic} point {k} label {label} " + "word " * 5,
                             "ori_topic": topic, "new_topic": topic, "label": label, "type_idx": kind,
                             "new_id": f"{prefix}{t}-{k}-{label}", "topic_str": f"topic number {t}"})
        # synthetic neutral: new_topic keeps the comment's OWN topic, topic_str holds the paired (unrelated) topic
        other = (t + 1) % n_topics
        rows.append({"author": "x", "post": f"{prefix} comment about {topic} point 0 label 1 " + "word " * 5,
                     "ori_topic": topic, "new_topic": topic, "label": "2", "type_idx": "4",
                     "new_id": f"{prefix}{t}-syn", "topic_str": f"topic number {other}"})
    return rows


def write_vast(directory, split, rows):
    path = Path(directory) / f"vast_{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=VAST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------- VAST loader
class TestVastLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_vast(self.dir, "dev", vast_rows("d"))
        write_vast(self.dir, "test", vast_rows("t"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_topic_text_gets_its_apostrophes_back(self):
        self.assertEqual(vast_topic_text("new york 's bike lanes"), "new york's bike lanes")
        self.assertEqual(vast_topic_text("parents ' religious beliefs"), "parents' religious beliefs")
        self.assertEqual(vast_topic_text("don 't panic"), "don't panic")
        self.assertEqual(vast_topic_text("school uniforms"), "school uniforms")

    def test_labels_splits_and_kinds(self):
        recs = load_vast(self.dir)
        self.assertEqual({r.split for r in recs}, {"dev", "test"})
        self.assertEqual({r.source for r in recs}, {"vast"})
        by = {(r.gold_stance, r.group) for r in recs}
        self.assertIn((Stance.FAVOR, "extracted topic"), by)
        self.assertIn((Stance.AGAINST, "corrected topic"), by)
        self.assertIn((Stance.NEUTRAL, "synthetic neutral"), by)
        self.assertEqual(len(recs), 2 * 6 * (3 * 2 + 1))

    def test_synthetic_neutral_uses_the_paired_topic_not_the_comments_own(self):
        recs = load_vast(self.dir, splits=["dev"])
        syn = next(r for r in recs if r.group == "synthetic neutral" and r.statement.startswith("d comment about topic number 0"))
        self.assertEqual(syn.target, "topic number 1")             # topic_str, mapped to readable text
        self.assertEqual(syn.topic, "topic_number_1")
        real = next(r for r in recs if r.group == "extracted topic" and "topic number 0 point 0" in r.statement)
        self.assertEqual(real.target, "topic number 0")

    def test_a_synthetic_neutral_never_contradicts_a_real_label_for_the_same_pair(self):
        from argus.data import dedupe
        kept, dups, conflicts = dedupe(load_vast(self.dir))
        self.assertEqual(conflicts, 0)

    def test_split_filter_and_file_lookup(self):
        self.assertEqual({r.split for r in load_vast(self.dir, splits=["test"])}, {"test"})
        self.assertEqual(set(find_vast_files(self.dir)), {"dev", "test"})
        self.assertEqual(set(find_vast_files(self.dir / "vast_dev.csv")), {"dev"})

    def test_bad_rows_are_skipped_with_one_warning(self):
        rows = vast_rows("d")
        rows.append({**rows[0], "label": "9", "new_id": "bad1"})
        rows.append({**rows[0], "post": "", "new_id": "bad2"})
        write_vast(self.dir, "dev", rows)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recs = load_vast(self.dir, splits=["dev"])
        self.assertEqual(len(recs), 6 * 7)
        self.assertEqual(sum("Skipped 2" in str(w.message) for w in caught), 1)

    def test_missing_files_and_wrong_file(self):
        with self.assertRaises(FileNotFoundError) as cm:
            load_vast(self.dir / "nope")
        self.assertIn("download_data.py", str(cm.exception))
        bad = self.dir / "other"
        bad.mkdir()
        (bad / "vast_dev.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            load_vast(bad)
        self.assertIn("missing columns", str(cm.exception))

    def test_utf8_bom_and_extra_columns_are_fine(self):
        path = self.dir / "vast_dev.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text("﻿" + text.replace("author", "author_name", 1).replace("topic_str", "topic_str,LexSim", 1)
                        .replace("\n", ",x\n", 1), encoding="utf-8")
        self.assertTrue(load_vast(self.dir, splits=["dev"]))


class TestSampling(unittest.TestCase):
    def pool(self):
        recs = []
        for t in range(20):
            for stance in (Stance.FAVOR, Stance.AGAINST, Stance.NEUTRAL):
                for k in range(6 if t == 0 else 2):                # topic 0 is much bigger than the rest
                    recs.append(Record(id=f"{t}-{stance.value}-{k}", source="vast", topic=f"t{t}", target=f"t{t}",
                                       statement=f"s {t} {stance.value} {k}", gold_stance=stance))
        return recs

    def test_balanced_capped_and_reproducible(self):
        a = sample_balanced(self.pool(), per_label=15, max_per_topic=2, seed=1)
        self.assertEqual(sorted(r.gold_stance.value for r in a).count("Favor"), 15)
        self.assertEqual({s: sum(r.gold_stance == s for r in a) for s in Stance}, {s: 15 for s in Stance})
        per_cell = {}
        for r in a:
            per_cell[(r.topic, r.gold_stance)] = per_cell.get((r.topic, r.gold_stance), 0) + 1
        self.assertLessEqual(max(per_cell.values()), 2)
        self.assertEqual([r.id for r in a], [r.id for r in sample_balanced(self.pool(), 15, 2, seed=1)])
        self.assertNotEqual([r.id for r in a], [r.id for r in sample_balanced(self.pool(), 15, 2, seed=2)])

    def test_gives_fewer_not_an_error_when_a_label_is_scarce(self):
        pool = [r for r in self.pool() if not (r.gold_stance == Stance.NEUTRAL and r.topic != "t0")]
        s = sample_balanced(pool, per_label=15, max_per_topic=2)
        self.assertEqual(sum(r.gold_stance == Stance.NEUTRAL for r in s), 2)

    def test_exclusions(self):
        pool = self.pool()
        first = sample_balanced(pool, 10, 2)
        second = sample_balanced(pool, 10, 2, exclude_ids={r.id for r in first})
        self.assertFalse({r.id for r in first} & {r.id for r in second})


class TestPrepareAndDownload(unittest.TestCase):
    def test_prepare_vast_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d) / "raw", Path(d) / "out"
            write_vast(raw, "dev", vast_rows("d", n_topics=8))
            write_vast(raw, "test", vast_rows("t", n_topics=8))
            p = run_script("prepare_data.py", "--raw", str(raw), "--out", str(out), "--dev-per-label", "5",
                           "--test-per-label", "8", "--max-per-topic", "2")
            self.assertEqual(p.returncode, 0, p.stderr)
            dev, test = load_jsonl(out / "dev.jsonl"), load_jsonl(out / "test.jsonl")
            self.assertEqual((len(dev), len(test)), (15, 24))
            self.assertEqual({r.split for r in dev}, {"dev"})
            self.assertFalse({r.id for r in dev} & {r.id for r in test})
            self.assertIn("labels: Favor 5, Against 5, Neutral 5", p.stdout)
            self.assertTrue((out / "all.jsonl").exists())

    def test_prepare_explains_missing_data_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            p = run_script("prepare_data.py", "--raw", str(Path(d) / "nothing"), "--out", str(Path(d) / "out"))
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("download_data.py", p.stderr)
            self.assertNotIn("Traceback", p.stderr)

    def serve(self, directory):
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
        handler.log_message = lambda *a, **k: None
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_download_script(self):
        with tempfile.TemporaryDirectory() as d:
            web, out = Path(d) / "web", Path(d) / "out"
            write_vast(web, "dev", vast_rows("d"))
            write_vast(web, "test", vast_rows("t"))
            url = self.serve(web)
            p = run_script("download_data.py", "--out", str(out), "--base-url", url)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue((out / "vast_dev.csv").exists() and (out / "vast_test.csv").exists())
            self.assertFalse(list(out.glob("*.part")))
            self.assertEqual(len(load_vast(out)), 2 * 6 * 7)

    def test_download_refuses_a_file_that_is_not_vast_and_prints_a_manual_route(self):
        with tempfile.TemporaryDirectory() as d:
            web, out = Path(d) / "web", Path(d) / "out"
            web.mkdir()
            (web / "vast_dev.csv").write_text("<html>not a csv</html>\n", encoding="utf-8")
            (web / "vast_test.csv").write_text("<html>not a csv</html>\n", encoding="utf-8")
            p = run_script("download_data.py", "--out", str(out), "--base-url", self.serve(web))
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("does not look like VAST", p.stderr)
            self.assertIn("save it as", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse((out / "vast_dev.csv").exists())

    def test_download_reports_a_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            web = Path(d) / "web"
            web.mkdir()
            p = run_script("download_data.py", "--out", str(Path(d) / "out"), "--base-url", self.serve(web))
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("Could not download", p.stderr)
            self.assertNotIn("Traceback", p.stderr)


# --------------------------------------------------------------------------- report helpers for many topics and kinds
class TestTopicsAndKinds(unittest.TestCase):
    def test_collapse_keeps_the_largest_and_pools_the_rest_by_weight(self):
        per_topic = {f"t{i}": (1.0, 10 - i) for i in range(5)}                 # sizes 10, 9, 8, 7, 6
        per_topic["small a"] = (0.0, 2)
        per_topic["small b"] = (1.0, 2)
        out = collapse_topics(per_topic, keep=5)
        self.assertEqual(set(out), {"t0", "t1", "t2", "t3", "t4", OTHER_TOPICS})
        self.assertEqual(out[OTHER_TOPICS], (0.5, 4))
        self.assertEqual(collapse_topics({"a": (1.0, 3)}, keep=5), {"a": (1.0, 3)})

    def test_per_group_accuracy(self):
        rows = [{"group": "x", "gold": "Favor", "pred": "Favor"}, {"group": "x", "gold": "Favor", "pred": "Against"},
                {"group": "y", "gold": "Neutral", "pred": "Neutral"}, {"group": None, "gold": "Favor", "pred": "Favor"}]
        self.assertEqual(per_group_accuracy(rows), {"x": (0.5, 2), "y": (1.0, 1)})

    def test_records_carry_their_group_into_the_comparison_and_report(self):
        recs = [r.model_copy(update={"group": "synthetic neutral" if r.gold_stance == Stance.NEUTRAL else "extracted topic"})
                for r in make_records()]
        runs = run_stance_comparison({"fake": FakeLLM()}, recs, ["zero_shot"], bootstrap=0)
        self.assertEqual(set(runs[0]["per_group"]), {"synthetic neutral", "extracted topic"})
        md = stance_report_md(runs, [], [], {"n": 0}, [], runs[0],
                              {"split": "dev", "n": 20, "models": ["fake"], "date": "d", "bootstrap": 0, "source": "vast"})
        self.assertIn("Accuracy by how the example was built", md)
        self.assertIn("| synthetic neutral | 5 |", md)
        self.assertIn("Neutral is easy", md)
        self.assertIn("20 examples from vast", md)

    def test_many_topics_give_a_short_table(self):
        recs = [Record(id=f"r{i}", source="t", topic=f"topic{i}", target=TARGET, statement=f"A higher wage floor helps case {i}.",
                       gold_stance=Stance.FAVOR) for i in range(40)]
        runs = run_stance_comparison({"fake": FakeLLM()}, recs, ["zero_shot"], bootstrap=0)
        self.assertEqual(len(runs[0]["per_topic"]), 11)
        md = stance_report_md(runs, [], [], {"n": 0}, [], runs[0], {"split": "dev", "n": 40, "models": ["f"], "date": "d", "bootstrap": 0})
        self.assertIn(OTHER_TOPICS, md)
        self.assertLess(md.count("| topic"), 12)


# --------------------------------------------------------------------------- demo helpers
class TestDemoHelpers(unittest.TestCase):
    def setUp(self):
        self.records = ArgusPipeline(FakeLLM(), PipelineConfig()).run_transcript(DEMO, TARGET)

    def test_similarity(self):
        self.assertEqual(similarity("A higher wage means layoffs", "a higher wage means layoffs"), 1.0)
        self.assertEqual(similarity("", ""), 0.0)
        self.assertLess(similarity("raising wages helps families", "bike lanes are dangerous"), 0.2)

    def test_argument_rows_only_contain_arguments(self):
        rows = argument_rows(self.records)
        self.assertEqual([r["index"] for r in rows], [1, 2, 3, 4, 5])             # sentence 0 is the moderator's welcome
        self.assertEqual(rows[1]["relation"], "Support")
        self.assertEqual(rows[1]["related_to_index"], 1)

    def row(self, text, speaker, stance="Favor", index=0):
        return {"index": index, "speaker": speaker, "argument": text, "stance": stance, "repeats": 1, "also_said_by": []}

    def test_merge_repeats(self):
        rows = [self.row("Raising the minimum wage reduces poverty among workers", "SMITH", index=1),
                self.row("raising the minimum wage reduces poverty among all workers", "LEE", index=2),
                self.row("Raising the minimum wage reduces poverty among workers", "SMITH", index=3),
                self.row("Raising the minimum wage reduces poverty among workers", "JONES", stance="Against", index=4),
                self.row("Small businesses will close", "JONES", index=5)]
        merged = merge_repeats(rows, 0.7)
        self.assertEqual([m["index"] for m in merged], [1, 4, 5])                  # same words but another stance stays
        self.assertEqual(merged[0]["repeats"], 3)
        self.assertEqual(merged[0]["also_said_by"], ["LEE"])
        self.assertEqual(len(merge_repeats(rows, 1.01)), 5)                        # nothing merges above 1
        self.assertEqual(rows[0]["repeats"], 1)                                    # inputs are not modified

    def test_condensed_view_counts(self):
        view = condensed_view(self.records)
        self.assertEqual((view["sentences"], view["arguments"], view["dropped"]), (6, 5, 1))
        self.assertEqual(sum(len(g) for g in view["groups"].values()), view["shown"])
        self.assertLessEqual(view["shown"], view["arguments"])
        self.assertEqual(set(view["groups"]), {"Favor", "Against"})
        unmerged = condensed_view(self.records, merge=False)
        self.assertEqual(unmerged["shown"], unmerged["arguments"])

    def test_condensed_view_with_stance_switched_off(self):
        records = ArgusPipeline(FakeLLM(), PipelineConfig(stance_on="none")).run_transcript(DEMO, TARGET)
        view = condensed_view(records)
        self.assertEqual(list(view["groups"]), ["Not assessed"])

    def test_argument_map(self):
        dot = argument_map_dot(self.records)
        self.assertTrue(dot.startswith("digraph") and dot.rstrip().endswith("}"))
        self.assertIn("n2 -> n1", dot)
        self.assertIn("Support", dot)
        self.assertIn("Attack", dot)
        self.assertNotIn("n0 ", dot)                                               # the welcome sentence is not drawn

    def test_argument_map_survives_awkward_text(self):
        rec = ArgumentRecord(policy_target="t", original_statement='He said "no" \\ then left', is_argument=True,
                             component=Component.CLAIM, segment_index=0, stance=Stance.AGAINST,
                             identified_argument='He said "no" \\ then left ' + "x" * 300)
        dot = argument_map_dot([rec])
        self.assertNotIn('"no"', dot)
        self.assertIn("...", dot)

    def test_table_rows_are_the_synopsis_columns(self):
        rows = table_rows(self.records)
        self.assertEqual(len(rows), 6)
        self.assertIn("policy_target", rows[0])
        text = table_rows(self.records, as_text=True)
        self.assertTrue(all(isinstance(v, str) for row in text for v in row.values()))


# --------------------------------------------------------------------------- setup checker
class FakeClient:
    def __init__(self, error=None):
        self.error = error

    def check_ready(self):
        if self.error:
            raise RuntimeError(self.error)

    def structured(self, messages, schema):
        return FakeLLM().structured(messages, schema)


class TestSetupCheck(unittest.TestCase):
    def test_python_version(self):
        self.assertEqual(check_python((3, 11, 4))["status"], "ok")
        old = check_python((3, 8, 10))
        self.assertEqual(old["status"], "fail")
        self.assertIn("python.org", old["fix"])

    def test_missing_package_gives_the_pip_command(self):
        def importer(name):
            if name in ("pysbd", "streamlit"):
                raise ImportError(name)
        results = {r["name"]: r for r in check_packages(importer)}
        self.assertEqual(results["package pysbd"]["status"], "fail")
        self.assertIn("pip install -r requirements.txt", results["package pysbd"]["fix"])
        self.assertEqual(results["package streamlit"]["status"], "warn")                 # optional
        self.assertEqual(results["package ollama"]["status"], "ok")

    def test_config(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(check_config(Path(d))[0]["status"], "fail")
            (Path(d) / "config.yaml").write_text("model: qwen2.5:7b\n", encoding="utf-8")
            result, cfg = check_config(Path(d))
            self.assertEqual((result["status"], cfg["model"]), ("ok", "qwen2.5:7b"))
            (Path(d) / "config.yaml").write_text("temperature: 0\n", encoding="utf-8")
            self.assertEqual(check_config(Path(d))[0]["status"], "fail")

    def test_ollama_messages(self):
        cfg = {"model": "qwen2.5:7b"}
        down = check_ollama(cfg, lambda c: FakeClient("Cannot reach Ollama. Install it ..."))
        self.assertEqual(down["status"], "fail")
        self.assertIn("ollama.com", down["fix"])
        missing = check_ollama(cfg, lambda c: FakeClient("Model 'qwen2.5:7b' is not downloaded. Run:  ollama pull qwen2.5:7b"))
        self.assertEqual(missing["fix"], "ollama pull qwen2.5:7b")
        ok, client = check_ollama(cfg, lambda c: FakeClient())
        self.assertEqual(ok["status"], "ok")

    def test_data_checks(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual([r["status"] for r in check_data(Path(d))], ["warn", "warn"])
            for rel in ("data/raw/vast/vast_dev.csv", "data/raw/vast/vast_test.csv", "data/processed/dev.jsonl", "data/processed/test.jsonl"):
                path = Path(d) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
            self.assertEqual([r["status"] for r in check_data(Path(d))], ["ok", "ok"])

    def test_run_checks_in_this_project(self):
        results = run_checks(ROOT, live=True, client_factory=lambda cfg: FakeClient())
        names = [r["name"] for r in results]
        self.assertIn("Live model call", names)
        self.assertEqual({r["status"] for r in results if r["name"] in ("Python", "Ollama", "Live model call", "config.yaml")}, {"ok"})
        self.assertNotIn("fail", {r["status"] for r in results})

    def test_a_failing_ollama_stops_before_the_live_call(self):
        results = run_checks(ROOT, live=True, client_factory=lambda cfg: FakeClient("Cannot reach Ollama."))
        self.assertNotIn("Live model call", [r["name"] for r in results])
        text = format_results(results)
        self.assertIn("[FAIL] Ollama", text)
        self.assertIn("fix:", text)
        self.assertIn("must be fixed", text)

    def test_format_all_ok(self):
        self.assertIn("Everything is ready.", format_results([{"name": "x", "status": "ok", "detail": "d", "fix": ""}]))

    def test_the_scripts_stay_polite_when_ollama_is_not_there(self):
        # Point the config at a port nothing listens on, in a copy of the project settings.
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "config.yaml"
            config.write_text("host: http://127.0.0.1:9\nmodel: qwen2.5:7b\n", encoding="utf-8")
            code = ("import sys; sys.path.insert(0, %r); from pathlib import Path; from argus.setup_check import run_checks, "
                    "format_results; r = run_checks(Path(%r)); print(format_results(r)); sys.exit(1 if any(x['status']=='fail' for x in r) else 0)"
                    % (str(ROOT), d))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=d)
            self.assertEqual(p.returncode, 1)
            self.assertIn("[FAIL] Ollama", p.stdout)
            self.assertNotIn("Traceback", p.stderr)


# --------------------------------------------------------------------------- the Streamlit app
try:
    from streamlit.testing.v1 import AppTest
except Exception:                                                                       # streamlit is optional
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit is not installed (pip install -r requirements-demo.txt)")
class TestApp(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()

    def make(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)

    def test_sample_transcript_end_to_end(self):
        with mock.patch("argus.llm.make_ready_client", lambda model=None: FakeLLM()):
            at = self.make().run()
            self.assertFalse(at.exception, at.exception)
            self.assertEqual(len(at.metric), 0)                                         # nothing before the button
            next(b for b in at.button if b.label == "Analyse debate").click()
            at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertFalse(at.error, [e.value for e in at.error])
        values = {m.label: m.value for m in at.metric}
        self.assertEqual(values["Sentences"], "23")
        self.assertGreater(int(values["Arguments"]), 0)
        self.assertEqual(len(at.tabs), 4)
        self.assertTrue(any("Favor" in s.value or "Against" in s.value for s in at.subheader))

    def test_ollama_down_gives_a_message_not_a_crash(self):
        def down(model=None):
            raise SystemExit("Cannot reach Ollama. Install it from https://ollama.com and make sure it is running.")
        with mock.patch("argus.llm.make_ready_client", down):
            at = self.make().run()
            next(b for b in at.button if b.label == "Analyse debate").click()
            at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any("Cannot reach Ollama" in e.value for e in at.error))
        self.assertEqual(len(at.metric), 0)

    def test_empty_pasted_transcript_is_refused(self):
        with mock.patch("argus.llm.make_ready_client", lambda model=None: FakeLLM()):
            at = self.make().run()
            at.radio[0].set_value("Paste or upload my own").run()
            next(b for b in at.button if b.label == "Analyse debate").click()
            at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any("Add a transcript" in w.value for w in at.warning))

    def test_pasted_transcript(self):
        with mock.patch("argus.llm.make_ready_client", lambda model=None: FakeLLM()):
            at = self.make().run()
            at.radio[0].set_value("Paste or upload my own").run()
            at.text_area[0].set_value(DEMO).run()
            next(b for b in at.button if b.label == "Analyse debate").click()
            at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual({m.label: m.value for m in at.metric}["Sentences"], "6")


if __name__ == "__main__":
    unittest.main()
