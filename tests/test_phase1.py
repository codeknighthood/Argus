"""Phase 1 tests. Run from the project root:  python -m unittest discover -s tests -v

The dataset rows here are SYNTHETIC (generated in a temp folder) so the tests need
no download. They check the logic, not the real corpus.
"""
import tempfile
import unittest
import warnings
from pathlib import Path

from argus.data import (
    clean_text, dedupe, load_jsonl, load_ukp, save_jsonl, stats_table, stratified_dev_test, ukp_label,
)
from argus.schema import Stance
from argus.segment import parse_turns, segment_transcript, segments_to_records

HEADER = "topic\tsentence_hash\tsentence\tannotation\tset\n"


def write_tsv(folder: Path, name: str, rows: list[tuple[str, str]], topic: str) -> None:
    (folder / "complete").mkdir(exist_ok=True)
    lines = [f"{topic}\th{i}\t{sent}\t{label}\ttrain" for i, (sent, label) in enumerate(rows)]
    (folder / "complete" / f"{name}.tsv").write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")


class TestCleaning(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("  Hello   world\t\n again\x00 "), "Hello world again")
        self.assertEqual(clean_text("ﬁne"), "fine")          # NFKC folds the "fi" ligature
        self.assertEqual(clean_text(None), "")

    def test_label_mapping(self):
        self.assertEqual(ukp_label("Argument_for"), (Stance.FAVOR, True))
        self.assertEqual(ukp_label("Argument_against"), (Stance.AGAINST, True))
        self.assertEqual(ukp_label("NoArgument"), (Stance.NEUTRAL, False))
        with self.assertRaises(ValueError):
            ukp_label("maybe")


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_map(self):
        write_tsv(self.dir, "minimum_wage", [
            ("A higher wage floor helps working families.", "Argument_for"),
            ("It will force small businesses to cut jobs.", "Argument_against"),
            ("The bill was debated on Tuesday afternoon.", "NoArgument"),
            ("Too short", "NoArgument"),                              # 2 tokens: dropped
        ], topic="minimum_wage")
        recs = load_ukp(self.dir)
        self.assertEqual(len(recs), 3)
        by = {r.statement: r for r in recs}
        self.assertEqual(by["A higher wage floor helps working families."].gold_stance, Stance.FAVOR)
        self.assertEqual(by["It will force small businesses to cut jobs."].gold_stance, Stance.AGAINST)
        neutral = by["The bill was debated on Tuesday afternoon."]
        self.assertEqual(neutral.gold_stance, Stance.NEUTRAL)
        self.assertFalse(neutral.is_argument)
        self.assertEqual(neutral.target, "the minimum wage")

    def test_loose_columns_and_topic_from_filename(self):
        (self.dir / "gun_control.tsv").write_text(
            "Sentence\tAnnotation\nBackground checks save lives every year.\tArgument_for\n", encoding="utf-8")
        recs = load_ukp(self.dir)
        self.assertEqual(recs[0].topic, "gun_control")

    def test_bad_file_is_skipped_not_fatal(self):
        write_tsv(self.dir, "cloning", [("Cloning could cure many diseases.", "Argument_for")], topic="cloning")
        (self.dir / "complete" / "junk.tsv").write_text("a\tb\n1\t2\n", encoding="utf-8")   # no sentence/label columns
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            recs = load_ukp(self.dir)
        self.assertEqual(len(recs), 1)
        self.assertTrue(any("junk.tsv" in str(x.message) for x in w))

    def test_all_empty_sentences_gives_clear_error(self):
        write_tsv(self.dir, "cloning", [("", "Argument_for"), ("", "NoArgument")], topic="cloning")
        with self.assertRaises(ValueError):
            load_ukp(self.dir)

    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            load_ukp(self.dir / "nope")


class TestQuality(unittest.TestCase):
    def _recs(self, rows):
        d = tempfile.TemporaryDirectory()
        write_tsv(Path(d.name), "minimum_wage", rows, topic="minimum_wage")
        recs = load_ukp(Path(d.name))
        d.cleanup()
        return recs

    def test_dedupe_and_conflicts(self):
        recs = self._recs([
            ("Same sentence appears twice here.", "Argument_for"),
            ("same sentence appears twice here.", "Argument_for"),          # duplicate (case-insensitive)
            ("This one has two different labels.", "Argument_for"),
            ("This one has two different labels.", "Argument_against"),     # conflict: both removed
            ("A perfectly ordinary unique sentence.", "NoArgument"),
        ])
        kept, dups, conflicts = dedupe(recs)
        self.assertEqual((len(kept), dups, conflicts), (2, 1, 2))

    def test_split_is_balanced_disjoint_deterministic(self):
        rows = []
        for label in ("Argument_for", "Argument_against", "NoArgument"):
            rows += [(f"Unique sentence number {i} about {label} topic.", label) for i in range(50)]
        recs = self._recs(rows)
        dev, test = stratified_dev_test(recs, dev_per_cell=5, test_per_cell=20, seed=7)
        self.assertEqual((len(dev), len(test)), (15, 60))
        self.assertFalse({r.id for r in dev} & {r.id for r in test})
        self.assertEqual({r.split for r in dev}, {"dev"})
        dev2, test2 = stratified_dev_test(recs, 5, 20, 7)
        self.assertEqual([r.id for r in dev], [r.id for r in dev2])        # same seed, same sample
        dev3, _ = stratified_dev_test(recs, 5, 20, 8)
        self.assertNotEqual([r.id for r in dev], [r.id for r in dev3])     # different seed differs
        self.assertIn("minimum_wage", stats_table(dev))

    def test_jsonl_roundtrip(self):
        recs = self._recs([("A sentence that roundtrips through disk.", "Argument_for")])
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(save_jsonl(recs, Path(d) / "x.jsonl"), 1)
            self.assertEqual(load_jsonl(Path(d) / "x.jsonl"), recs)


TRANSCRIPT = """\
MODERATOR: Welcome everyone. Tonight we debate the minimum wage.
DR. SMITH: I support it. Full-time workers should not live in poverty.
Studies show modest increases barely affect employment.
MR. JONES: I disagree. Dr. Smith ignores small businesses.
"""


class TestSegmentation(unittest.TestCase):
    def test_turns_and_continuation(self):
        turns = parse_turns(TRANSCRIPT)
        self.assertEqual([s for s, _ in turns], ["MODERATOR", "DR. SMITH", "MR. JONES"])
        self.assertIn("Studies show", turns[1][1])        # continuation line merged into Smith's turn

    def test_segments_have_speaker_and_context(self):
        segs = segment_transcript(TRANSCRIPT, context_window=2)
        speakers = [s.speaker for s in segs]
        self.assertEqual(speakers[0], "MODERATOR")
        self.assertIn("MR. JONES", speakers)
        self.assertEqual(segs[0].context, "")                       # nothing before the first sentence
        self.assertTrue(segs[2].context.startswith("MODERATOR:"))   # context carries speaker tags
        self.assertLessEqual(len(segs[-1].context.split(" MODERATOR:")), 2)
        # "Dr." must not be treated as a sentence boundary
        self.assertTrue(any("Dr. Smith ignores" in s.sentence for s in segs))

    def test_no_speaker_parsing(self):
        segs = segment_transcript("Plain text without labels. Another sentence follows it.", parse_speakers=False)
        self.assertTrue(all(s.speaker is None for s in segs))

    def test_segments_to_records_are_unlabelled(self):
        recs = segments_to_records(segment_transcript(TRANSCRIPT), target="the minimum wage")
        self.assertTrue(recs)
        self.assertTrue(all(r.gold_stance is None and r.target == "the minimum wage" for r in recs))
        self.assertEqual(len({r.id for r in recs}), len(recs))      # ids are unique


if __name__ == "__main__":
    unittest.main()
