"""Phase 1: dataset loading, cleaning and sampling (synopsis 5.1 and 5.2).

Everything downstream works on one common `Record`, whatever the source:
  (target, statement, context, speaker, gold label)
so the same stance / extraction code runs on benchmark data and on real transcripts.

Primary dataset: VAST, the "Varied Stance Topics" dataset (Allaway and McKeown, 2020). Comments from the New York
Times "Room for Debate" discussions, each paired with a topic and labelled pro / con / neutral toward it, which maps to
Favor / Against / Neutral. It is hosted openly on GitHub, so it needs no registration. It is a cross-target stance
dataset, which is the setting of synopsis section 2.3.

The UKP Sentential Argument Mining Corpus (Stab et al., 2018) is still supported as an optional second source
(`load_ukp`), but it sits behind an access form, so nothing in the default workflow needs it.
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel

from .schema import Stance


class Record(BaseModel):
    """One statement to analyse, in the same shape for every data source."""
    id: str
    source: str
    topic: str                      # machine name, e.g. "minimum_wage"
    target: str                     # human wording used in prompts, e.g. "the minimum wage"
    statement: str
    context: str = ""               # surrounding discussion, if any (5.2: keep context)
    speaker: Optional[str] = None   # 5.7: speaker/context where available
    gold_stance: Optional[Stance] = None   # None for unlabelled transcripts
    is_argument: Optional[bool] = None     # gold: does the sentence contain an argument?
    split: Optional[str] = None            # "dev" / "test" once sampled
    group: Optional[str] = None            # how the example was built (VAST: how the topic was found); for analysis


# Wording follows the corpus topic names, because annotators judged "for/against the topic".
UKP_TARGETS = {
    "abortion": "abortion",
    "cloning": "cloning",
    "death_penalty": "the death penalty",
    "gun_control": "gun control",
    "marijuana_legalization": "marijuana legalization",
    "minimum_wage": "the minimum wage",
    "nuclear_energy": "nuclear energy",
    "school_uniforms": "school uniforms",
}


# --------------------------------------------------------------------------- cleaning
_CONTROL = re.compile(r"[^\S \t\n]|[\x00-\x08\x0b-\x1f\x7f]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise unicode, drop control characters, collapse whitespace (synopsis 5.2)."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _CONTROL.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def n_tokens(text: str) -> int:
    return len(text.split())


def make_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- UKP loader
def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def ukp_label(raw: str) -> tuple[Stance, bool]:
    """Map a UKP annotation to (stance, is_argument). Raises on anything unknown."""
    n = _norm(raw)
    if n in ("noargument", "noarg", "none"):
        return Stance.NEUTRAL, False
    if n in ("argumentfor", "for", "support", "supporting", "pro"):
        return Stance.FAVOR, True
    if n in ("argumentagainst", "against", "oppose", "opposing", "con"):
        return Stance.AGAINST, True
    raise ValueError(f"Unknown UKP label: {raw!r}")


def _pick(columns: dict[str, str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return columns[c]
    return None


def find_ukp_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if (path / "complete").is_dir():           # the corpus ships a "complete" folder
        return sorted((path / "complete").glob("*.tsv"))
    return sorted(path.rglob("*.tsv"))


def load_ukp(path: Path, min_tokens: int = 3) -> list[Record]:
    """Load the UKP corpus TSV files into Records.

    Column names are matched loosely (topic / sentence / annotation / set), so small
    differences between releases do not break loading. Files that do not look like
    corpus files are skipped with a warning instead of crashing.
    """
    files = find_ukp_files(Path(path))
    if not files:
        raise FileNotFoundError(f"No .tsv files found under {path}. See README, 'Getting the data'.")

    records: list[Record] = []
    empty_sentences = 0
    for f in files:
        with open(f, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
            if not reader.fieldnames:
                continue
            cols = {_norm(c): c for c in reader.fieldnames}
            topic_col = _pick(cols, ["topic"])
            sent_col = _pick(cols, ["sentence", "sentenceoriginal", "text"])
            label_col = _pick(cols, ["annotation", "label"])
            if not (sent_col and label_col):
                warnings.warn(f"Skipping {f.name}: needs sentence + annotation columns, found {reader.fieldnames}")
                continue
            split_col = _pick(cols, ["set", "split"])

            for row in reader:
                topic = (row.get(topic_col) if topic_col else None) or f.stem
                topic = re.sub(r"[\s\-]+", "_", topic.strip().lower())
                statement = clean_text(row.get(sent_col) or "")
                if not statement:
                    empty_sentences += 1
                    continue
                if n_tokens(statement) < min_tokens:
                    continue
                try:
                    stance, is_arg = ukp_label(row.get(label_col) or "")
                except ValueError:
                    continue
                records.append(Record(
                    id=make_id("ukp", topic, statement),
                    source="ukp",
                    topic=topic,
                    target=UKP_TARGETS.get(topic, topic.replace("_", " ")),
                    statement=statement,
                    gold_stance=stance,
                    is_argument=is_arg,
                ))

    if not records and empty_sentences:
        raise ValueError(
            f"All {empty_sentences} sentences were empty. Some corpus releases ship only sentence "
            "hashes and URLs and need a retrieval step. Run scripts/inspect_data.py and send me the output."
        )
    return records


# --------------------------------------------------------------------------- VAST loader
VAST_FILES = {"dev": "vast_dev.csv", "test": "vast_test.csv", "train": "vast_train.csv"}
VAST_LABELS = {"0": Stance.AGAINST, "1": Stance.FAVOR, "2": Stance.NEUTRAL}      # the dataset's own coding
# `type_idx` in the dataset says how the (comment, topic) pair was made. Neutral pairs of type 4 are built by pairing a
# comment with an unrelated topic, so they are easier than the others; reports show accuracy for each kind.
VAST_KINDS = {"1": "extracted topic", "2": "corrected topic", "3": "listed topic", "4": "synthetic neutral"}


def vast_topic_text(raw: str) -> str:
    """The dataset stores topics tokenised and lowercased ("new york 's bike lanes"); put the apostrophes back."""
    text = re.sub(r"\s+'(s|t|re|ve|ll|d|m)\b", r"'\1", raw or "")
    text = re.sub(r"(\w)\s+'(\s|$)", r"\1'\2", text)
    return _SPACES.sub(" ", text).strip()


def find_vast_files(path: Path) -> dict[str, Path]:
    """{'dev': .../vast_dev.csv, 'test': ...} for the files that exist under `path` (a folder or a single file)."""
    path = Path(path)
    if path.is_file():
        for split, name in VAST_FILES.items():
            if path.name == name:
                return {split: path}
        return {"all": path}
    found = {}
    for split, name in VAST_FILES.items():
        hits = sorted(path.rglob(name)) if path.is_dir() else []
        if hits:
            found[split] = hits[0]
    return found


def load_vast(path: Path, min_tokens: int = 3, splits: Optional[Iterable[str]] = None) -> list[Record]:
    """Load VAST CSV files into Records. `split` on each record is the dataset's own split (dev / test / train).

    Reads only the columns it needs (post, new_topic, topic_str, label, type_idx), so extra or reordered columns
    do not matter. See the comment about synthetic neutral rows below: they need `topic_str`, not `new_topic`.
    Rows with an unknown label or an empty post are skipped and counted in a single warning.
    """
    files = find_vast_files(Path(path))
    if splits is not None:
        files = {s: f for s, f in files.items() if s in set(splits)}
    if not files:
        raise FileNotFoundError(
            f"No VAST files (vast_dev.csv, vast_test.csv) found under {path}. "
            "Run:  python scripts/download_data.py   (see INSTALL.md)")
    csv.field_size_limit(10 ** 9)
    raw_rows: list[tuple[str, dict]] = []
    for split, f in files.items():
        with open(f, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {"post", "new_topic", "label"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{f.name} is missing columns {sorted(missing)}; found {reader.fieldnames}. "
                                 "Is this the VAST file from github.com/emilyallaway/zero-shot-stance?")
            raw_rows += [(split, row) for row in reader]

    # Synthetic neutral rows (type 4) pair a comment with a topic taken from ANOTHER comment. The dataset leaves the
    # comment's own topic in `new_topic` and puts the paired topic in `topic_str` (tokenised and lemmatised). Reading
    # `new_topic` for these rows would give the comment its real topic with a Neutral label, which is a contradiction.
    # So for type 4 we use `topic_str`, mapped back to the readable wording that other rows show for the same topic.
    readable: dict[str, Counter] = defaultdict(Counter)
    for _, row in raw_rows:
        if (row.get("type_idx") or "").strip() != "4":
            key = _SPACES.sub(" ", row.get("topic_str") or "").strip()
            if key:
                readable[key][vast_topic_text(clean_text(row.get("new_topic") or ""))] += 1

    records: list[Record] = []
    skipped = 0
    for split, row in raw_rows:
        kind_code = (row.get("type_idx") or "").strip()
        if kind_code == "4":
            key = _SPACES.sub(" ", row.get("topic_str") or "").strip()
            topic_text = readable[key].most_common(1)[0][0] if readable.get(key) else key
        else:
            topic_text = vast_topic_text(clean_text(row.get("new_topic") or ""))
        statement = clean_text(row.get("post") or "")
        stance = VAST_LABELS.get((row.get("label") or "").strip())
        if not statement or not topic_text or stance is None or n_tokens(statement) < min_tokens:
            skipped += 1
            continue
        records.append(Record(
            id=make_id("vast", split, (row.get("new_id") or "").strip() or topic_text, statement),
            source="vast",
            topic=re.sub(r"[^a-z0-9]+", "_", topic_text.lower()).strip("_") or "topic",
            target=topic_text,
            statement=statement,
            gold_stance=stance,
            split=split,
            group=VAST_KINDS.get(kind_code),
        ))
    if skipped:
        warnings.warn(f"Skipped {skipped} VAST rows (empty post, empty topic, unknown label or too short).")
    return records


# --------------------------------------------------------------------------- quality
def dedupe(records: Iterable[Record]) -> tuple[list[Record], int, int]:
    """Drop exact duplicate (topic, statement) pairs.

    If the same sentence appears with DIFFERENT gold labels, the label is ambiguous,
    so every copy is removed. Returns (kept, n_duplicates_removed, n_conflicts_removed).
    """
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r.topic, r.statement.lower())].append(r)

    kept, dups, conflicts = [], 0, 0
    for group in groups.values():
        if len({r.gold_stance for r in group}) > 1:
            conflicts += len(group)
            continue
        kept.append(group[0])
        dups += len(group) - 1
    kept.sort(key=lambda r: r.id)
    return kept, dups, conflicts


def stratified_dev_test(
    records: list[Record], dev_per_cell: int = 10, test_per_cell: int = 30, seed: int = 42
) -> tuple[list[Record], list[Record]]:
    """Balanced, reproducible, non-overlapping dev and test samples.

    A "cell" is one (topic, gold stance) pair, so every topic and every label is
    represented equally. Iterate on prompts using DEV only; report final numbers on TEST.
    """
    cells: dict[tuple[str, Stance], list[Record]] = defaultdict(list)
    for r in records:
        if r.gold_stance is not None:
            cells[(r.topic, r.gold_stance)].append(r)

    rng = random.Random(seed)
    dev, test = [], []
    for key in sorted(cells, key=lambda k: (k[0], k[1].value)):
        items = sorted(cells[key], key=lambda r: r.id)   # sort first so the seed fully decides
        rng.shuffle(items)
        d = items[:dev_per_cell]
        t = items[dev_per_cell:dev_per_cell + test_per_cell]
        dev += [r.model_copy(update={"split": "dev"}) for r in d]
        test += [r.model_copy(update={"split": "test"}) for r in t]
    return dev, test


def sample_balanced(
    records: list[Record], per_label: int, max_per_topic: int = 4, seed: int = 42, exclude_ids: Iterable[str] = ()
) -> list[Record]:
    """A reproducible sample with the same number of Favor, Against and Neutral examples.

    VAST has hundreds of topics and a few of them have far more rows than the rest, so each (topic, label) cell is
    capped at `max_per_topic`. Within that, topics are visited in random order, so no topic dominates the sample.
    If a label has fewer than `per_label` examples after the cap, you get fewer of that label (never an error).
    """
    skip = set(exclude_ids)
    rng = random.Random(seed)
    out: list[Record] = []
    for stance in (Stance.FAVOR, Stance.AGAINST, Stance.NEUTRAL):
        cells: dict[str, list[Record]] = defaultdict(list)
        for r in sorted(records, key=lambda r: r.id):                 # sort first so the seed fully decides
            if r.gold_stance == stance and r.id not in skip:
                cells[r.topic].append(r)
        topics = sorted(cells)
        rng.shuffle(topics)
        for topic in topics:
            rng.shuffle(cells[topic])
            cells[topic] = cells[topic][:max_per_topic]
        picked: list[Record] = []
        # round-robin over topics: first one example from every topic, then a second, ...
        for depth in range(max_per_topic):
            for topic in topics:
                if len(picked) < per_label and depth < len(cells[topic]):
                    picked.append(cells[topic][depth])
        out += picked
    return sorted(out, key=lambda r: r.id)


def stats_table(records: list[Record]) -> str:
    """Plain-text table of counts per topic and gold stance."""
    counts: Counter = Counter((r.topic, r.gold_stance.value if r.gold_stance else "unlabelled") for r in records)
    topics = sorted({t for t, _ in counts})
    labels = ["Favor", "Against", "Neutral", "unlabelled"]
    labels = [l for l in labels if any(counts[(t, l)] for t in topics)]
    width = max([len(t) for t in topics] + [5])
    lines = [f"{'topic':<{width}}  " + "  ".join(f"{l:>8}" for l in labels) + f"  {'total':>6}"]
    for t in topics:
        row = [counts[(t, l)] for l in labels]
        lines.append(f"{t:<{width}}  " + "  ".join(f"{n:>8}" for n in row) + f"  {sum(row):>6}")
    totals = [sum(counts[(t, l)] for t in topics) for l in labels]
    lines.append(f"{'ALL':<{width}}  " + "  ".join(f"{n:>8}" for n in totals) + f"  {sum(totals):>6}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- io
def save_jsonl(records: Iterable[Record], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def load_jsonl(path: Path) -> list[Record]:
    with open(path, "r", encoding="utf-8") as f:
        return [Record.model_validate_json(line) for line in f if line.strip()]
