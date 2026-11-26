"""Phase 1: turn a raw debate transcript into sentence-level segments with context.

Synopsis 5.2: "dividing lengthy debate transcripts into meaningful sentences or text
segments ... relevant contextual information will be retained". Each Segment keeps
the speaker and the previous few sentences, so the stance stage can judge a short
statement ("I disagree.") in the context of what came before it.

Expected transcript layout (the usual one for hearings and debates):

    SPEAKER NAME: what they said ...
    OTHER SPEAKER: their reply ...

Lines without a "NAME:" prefix are treated as a continuation of the previous turn.
"""
from __future__ import annotations

import re
from typing import Optional

import pysbd
from pydantic import BaseModel

from .data import Record, clean_text, make_id, n_tokens

# 1-4 capitalised words followed by a colon at the start of a line, e.g. "Dr. Smith:" / "MODERATOR:"
_SPEAKER = re.compile(r"^\s*((?:[A-Z][\w.'’\-]*)(?:\s+[A-Z][\w.'’\-]*){0,3})\s*:\s*(.*)$")

_segmenter: Optional[pysbd.Segmenter] = None


def _sentences(text: str) -> list[str]:
    global _segmenter
    if _segmenter is None:
        _segmenter = pysbd.Segmenter(language="en", clean=False)
    return [s.strip() for s in _segmenter.segment(text) if s.strip()]


class Segment(BaseModel):
    index: int
    turn: int
    speaker: Optional[str]
    sentence: str
    context: str       # the previous few sentences, speaker-tagged ("" for the first one)


def parse_turns(text: str, parse_speakers: bool = True) -> list[tuple[Optional[str], str]]:
    """Split a transcript into (speaker, text) turns."""
    turns: list[list] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SPEAKER.match(line) if parse_speakers else None
        if m:
            turns.append([m.group(1).strip(), m.group(2).strip()])
        elif turns:
            turns[-1][1] += " " + line          # continuation of the previous turn
        else:
            turns.append([None, line])
    return [(spk, clean_text(body)) for spk, body in turns if clean_text(body)]


def segment_transcript(
    text: str, context_window: int = 2, parse_speakers: bool = True, min_tokens: int = 3
) -> list[Segment]:
    """Sentence-split a transcript; attach speaker and the previous `context_window` sentences."""
    segments: list[Segment] = []
    history: list[str] = []                     # speaker-tagged earlier sentences
    for turn_idx, (speaker, body) in enumerate(parse_turns(text, parse_speakers)):
        for sent in _sentences(body):
            sent = clean_text(sent)
            if n_tokens(sent) < min_tokens:
                continue
            context = " ".join(history[-context_window:]) if context_window > 0 else ""
            segments.append(Segment(
                index=len(segments), turn=turn_idx, speaker=speaker, sentence=sent, context=context,
            ))
            history.append(f"{speaker}: {sent}" if speaker else sent)
    return segments


def segments_to_records(segments: list[Segment], target: str, source: str = "transcript") -> list[Record]:
    """Wrap segments as unlabelled Records so the same pipeline can process them."""
    return [
        Record(
            id=make_id(source, target, str(s.index), s.sentence),
            source=source,
            topic=re.sub(r"\W+", "_", target.lower()).strip("_"),
            target=target,
            statement=s.sentence,
            context=s.context,
            speaker=s.speaker,
        )
        for s in segments
    ]
