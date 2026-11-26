"""Shared helpers for running a stage over a dataset and scoring it (synopsis 5.8).

Used by the run_* scripts and by scripts/evaluate.py. A failed LLM call never stops a run: the
row gets the prediction "ERROR", which the metrics count as wrong, and the run carries on.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from .data import Record
from .extract import extract_argument
from .llm import LLMClient
from .relations import classify_relation
from .stance import predict_stance

ARG_LABELS = ["Argument", "NoArgument"]


class Progress:
    """Tiny console progress line with an ETA. ASCII only, so it works in any Windows console."""

    def __init__(self, total: int, label: str = "", stream=None):
        self.total, self.label, self.done = total, label, 0
        self.stream = stream or sys.stderr
        self.start = time.time()

    def update(self, n: int = 1) -> None:
        self.done += n
        elapsed = time.time() - self.start
        eta = elapsed / self.done * (self.total - self.done) if self.done else 0
        pct = 100 * self.done / self.total if self.total else 100
        self.stream.write(f"\r{self.label} {self.done}/{self.total} ({pct:3.0f}%)  ETA {int(eta // 60):02d}:{int(eta % 60):02d}   ")
        self.stream.flush()

    def close(self) -> None:
        self.stream.write("\n")
        self.stream.flush()


def take(records: Sequence[Record], limit: Optional[int], seed: int = 0) -> list[Record]:
    """A reproducible random subset of `limit` records (keeps the original order). None = all."""
    if not limit or limit >= len(records):
        return list(records)
    keep = set(random.Random(seed).sample(range(len(records)), limit))
    return [r for i, r in enumerate(records) if i in keep]


# --------------------------------------------------------------------------- stance
def predict_records(
    llm: LLMClient, records: Iterable[Record], strategy: str, progress: Optional[Progress] = None
) -> list[dict]:
    rows = []
    for r in records:
        row = {
            "id": r.id, "topic": r.topic, "target": r.target, "statement": r.statement,
            "gold": r.gold_stance.value if r.gold_stance else None, "group": r.group,
            "pred": None, "explanation": "", "error": None,
        }
        try:
            result = predict_stance(llm, strategy, r.target, r.statement, r.context)
            row["pred"], row["explanation"] = result.stance.value, result.explanation
        except Exception as e:  # keep going: one bad call must not lose a long run
            row["pred"], row["error"] = "ERROR", str(e)[:300]
        rows.append(row)
        if progress:
            progress.update()
    return rows


# --------------------------------------------------------------------------- argument detection
def predict_arguments(
    llm: LLMClient, records: Iterable[Record], progress: Optional[Progress] = None
) -> list[dict]:
    """Run extraction over records. `gold` is Argument / NoArgument when the dataset says so (UKP does)."""
    rows = []
    for r in records:
        row = {
            "id": r.id, "topic": r.topic, "target": r.target, "statement": r.statement,
            "gold": None if r.is_argument is None else ("Argument" if r.is_argument else "NoArgument"),
            "gold_stance": r.gold_stance.value if r.gold_stance else None,
            "pred": None, "component": None, "identified_argument": "", "explanation": "", "error": None,
        }
        try:
            result = extract_argument(llm, r.target, r.statement, r.context)
            row["pred"] = "Argument" if result.is_argument else "NoArgument"
            row["component"] = result.component.value
            row["identified_argument"] = result.identified_argument
            row["explanation"] = result.explanation
        except Exception as e:  # keep going: one bad call must not lose a long run
            row["pred"], row["error"] = "ERROR", str(e)[:300]
        rows.append(row)
        if progress:
            progress.update()
    return rows


# --------------------------------------------------------------------------- relations
def predict_relations(
    llm: LLMClient, target: str, pairs: Iterable[dict], progress: Optional[Progress] = None
) -> list[dict]:
    """Classify annotated (A, B) pairs. Each pair dict needs sentence_a, sentence_b and a normalised `gold`."""
    rows = []
    for p in pairs:
        row = {
            "pair_id": p.get("pair_id"), "speaker_a": p.get("speaker_a", ""), "speaker_b": p.get("speaker_b", ""),
            "sentence_a": p["sentence_a"], "sentence_b": p["sentence_b"],
            "gold": p.get("gold"), "pred": None, "explanation": "", "error": None,
        }
        try:
            result = classify_relation(llm, target, p["sentence_a"], p["sentence_b"],
                                       p.get("speaker_a") or None, p.get("speaker_b") or None)
            row["pred"], row["explanation"] = result.relation.value, result.explanation
        except Exception as e:  # keep going: one bad call must not lose a long run
            row["pred"], row["error"] = "ERROR", str(e)[:300]
        rows.append(row)
        if progress:
            progress.update()
    return rows


# --------------------------------------------------------------------------- analysis
def per_topic_accuracy(rows: Sequence[dict]) -> dict[str, tuple[float, int]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        groups[r["topic"]].append(r["gold"] == r["pred"])
    return {t: (sum(v) / len(v), len(v)) for t, v in sorted(groups.items())}


OTHER_TOPICS = "(all other topics)"


def per_group_accuracy(rows: Sequence[dict]) -> dict[str, tuple[float, int]]:
    """Accuracy for each kind of example (VAST: how the topic was matched). Empty if the rows carry no group."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        if r.get("group"):
            groups[r["group"]].append(r["gold"] == r["pred"])
    return {g: (sum(v) / len(v), len(v)) for g, v in sorted(groups.items())}


def collapse_topics(per_topic: dict[str, tuple[float, int]], keep: int = 10) -> dict[str, tuple[float, int]]:
    """Keep the `keep` topics with the most examples and pool the rest, so a corpus with hundreds of topics
    gives a readable table. Pooled accuracy is the example-weighted average, not an average of averages."""
    if len(per_topic) <= keep:
        return dict(per_topic)
    ranked = sorted(per_topic.items(), key=lambda kv: (-kv[1][1], kv[0]))
    kept = dict(ranked[:keep])
    rest = ranked[keep:]
    n = sum(v[1] for _, v in rest)
    kept[OTHER_TOPICS] = (sum(v[0] * v[1] for _, v in rest) / n, n)
    return kept


def errors_only(rows: Sequence[dict]) -> list[dict]:
    return [r for r in rows if r["gold"] != r["pred"]]


# --------------------------------------------------------------------------- io
def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: Sequence[dict], path: Path, columns: Sequence[str]) -> None:
    """UTF-8 with a BOM so Excel on Windows shows non-English characters correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_split(data_dir: Path, split: str) -> list[Record]:
    """Load dev / test / all from data/processed, with a clear message if Phase 1 has not been run."""
    from .data import load_jsonl

    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The data has not been prepared yet. Run these two commands:\n"
            "    python scripts/download_data.py\n    python scripts/prepare_data.py\n"
            "(see INSTALL.md, step 5)."
        )
    return load_jsonl(path)


def load_split_or_exit(data_dir: Path, split: str) -> list[Record]:
    """For scripts: like load_split, but stops with a plain message (no traceback) if Phase 1 has not been run."""
    try:
        return load_split(data_dir, split)
    except FileNotFoundError as e:
        raise SystemExit(str(e))


# --------------------------------------------------------------------------- Phase 6: self-consistency
def self_consistency(
    records: Sequence[Record], strategy: str, runs: int, client_factory: Callable[[int], LLMClient],
    labels: Sequence[str], progress: Optional[Progress] = None,
) -> dict:
    """How stable are the answers when the model is allowed to vary (non-zero temperature)?

    Asks the same questions `runs` times with a fresh client each time (`client_factory(k)` should give each run its
    own cache entries). Reports the share of sentences where every run gave the same label, the average share of
    runs that agree with the majority label, and the accuracy of the majority vote against the single runs.
    A model whose labels flip between runs on the same sentence is not a reliable annotator, whatever its accuracy.
    """
    per_run = [predict_records(client_factory(k), records, strategy, progress) for k in range(runs)]
    n = len(records)
    all_same, share_sum, majority = 0, 0.0, []
    for i in range(n):
        votes = Counter(per_run[k][i]["pred"] for k in range(runs))
        top = max(votes.values())
        winner = min((l for l, c in votes.items() if c == top),            # ties: first label in the label list
                     key=lambda l: (list(labels).index(l) if l in labels else len(labels), str(l)))
        majority.append(winner)
        all_same += len(votes) == 1
        share_sum += top / runs
    gold = [r["gold"] for r in per_run[0]] if per_run else []
    return {
        "runs": runs, "n": n,
        "all_same": all_same / n if n else 0.0,
        "mean_majority_share": share_sum / n if n else 0.0,
        "per_run_accuracy": [sum(r["gold"] == r["pred"] for r in rows) / n if n else 0.0 for rows in per_run],
        "majority_accuracy": sum(g == m for g, m in zip(gold, majority)) / n if n else 0.0,
    }
