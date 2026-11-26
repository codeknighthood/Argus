"""Phase 4: argument relations (synopsis 5.4).

Given two arguments, the LLM says whether the later one (B) supports, attacks or is unrelated to the earlier
one (A). In a transcript every argument is compared with the few arguments just before it, and linked to the
nearest one it connects to.

Design choice, stated plainly so it can be discussed in the report: candidates are checked nearest first and the
first Support or Attack wins. This matches how debates flow (people reply to what was just said) and keeps the
number of LLM calls low (at most `window` per argument), but it can miss a link to an argument further back
when a nearer one also happens to connect. `window` is configurable.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

from .llm import LLMClient
from .prompting import render
from .schema import Relation, RelationResult

PROMPT_NAME = "relation_v1"


@dataclass(frozen=True)
class ArgItem:
    """An argument ready to be linked: its position in the transcript, who said it, and its standalone text."""
    index: int
    speaker: Optional[str]
    text: str


def build_messages(
    target: str, argument_a: str, argument_b: str,
    speaker_a: Optional[str] = None, speaker_b: Optional[str] = None,
) -> list[dict]:
    text = render(
        PROMPT_NAME, target=target, argument_a=argument_a, argument_b=argument_b,
        speaker_a=speaker_a or "(unknown)", speaker_b=speaker_b or "(unknown)",
    )
    return [{"role": "user", "content": text}]


def classify_relation(
    llm: LLMClient, target: str, argument_a: str, argument_b: str,
    speaker_a: Optional[str] = None, speaker_b: Optional[str] = None,
) -> RelationResult:
    """How B (the later argument) relates to A (the earlier one)."""
    return llm.structured(build_messages(target, argument_a, argument_b, speaker_a, speaker_b), RelationResult)


def find_relation(
    llm: LLMClient, target: str, current: ArgItem, previous: Sequence[ArgItem], window: int = 3,
) -> tuple[RelationResult, Optional[ArgItem]]:
    """Link `current` to the nearest of the last `window` earlier arguments it supports or attacks.

    Returns (result, linked_argument). If nothing connects, the result is Relation.NONE and the link is None.
    """
    candidates = list(previous)[-window:] if window > 0 else []
    for candidate in reversed(candidates):                     # nearest first
        result = classify_relation(llm, target, candidate.text, current.text, candidate.speaker, current.speaker)
        if result.relation != Relation.NONE:
            return result, candidate
    return (
        RelationResult(
            relation=Relation.NONE,
            explanation=(f"No support or attack found among the previous {len(candidates)} argument(s)."
                         if candidates else "No earlier argument to relate to."),
        ),
        None,
    )
