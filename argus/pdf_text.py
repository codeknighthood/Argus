"""Read a debate transcript from a PDF, so it can go through the same pipeline as a .txt file.

Real transcript PDFs are messy. Besides the words themselves they usually contain a title block, a header and a
footer repeated on every page, page numbers, time codes such as [19:04], notes such as [Applause] or [crosstalk], and
lines that wrap in the middle of a sentence. `extract_transcript` removes those and returns one line per speaker turn
("CHAIR OKAFOR: ..."), which is the format the rest of ARGUS reads.

It reads the text layer of the PDF. A scanned PDF (a photo of paper) has no text layer; ARGUS does not do OCR, and
says so instead of returning nothing.
"""
from __future__ import annotations

import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from .segment import _SPEAKER

_TIMECODE = re.compile(r"^\s*[\[(]?\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*[\])]?\s*[-–—]?\s*")
_NOTE = re.compile(r"\[[^\]\n]{1,60}\]")
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$|^\s*[-–—]\s*\d+\s*[-–—]\s*$", re.I)
_EDGE_LINES = 3          # header and footer are looked for in the first and last few lines of a page


@dataclass
class PdfTranscript:
    text: str                                   # one line per speaker turn
    pages: int
    turns: int
    speakers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)     # what was removed, for showing to the user


def _normalise(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def _label(line: str):
    m = _SPEAKER.match(line)
    return m.group(1).strip() if m else None


def extract_transcript(source: Union[bytes, str, Path, io.BufferedIOBase]) -> PdfTranscript:
    """Clean transcript text from a PDF (bytes, a path, or an open file). Raises ValueError with a plain message."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError("Reading PDFs needs the 'pypdf' package. Run:  pip install -r requirements-demo.txt") from None
    if isinstance(source, (str, Path)):
        source = Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        reader = PdfReader(source)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("This PDF is password protected. Remove the password and try again.")
        pages = [(page.extract_text() or "").splitlines() for page in reader.pages]
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read this file as a PDF ({type(e).__name__}).") from None
    if not any(line.strip() for page in pages for line in page):
        raise ValueError("No selectable text found in this PDF. It looks like a scan (a picture of paper); ARGUS reads "
                         "the text layer and does not do OCR. Use a PDF exported from a document or website instead.")

    notes: list[str] = []
    # 1. headers and footers: the same line (ignoring digits) near the top or bottom of most pages
    repeated: set[str] = set()
    if len(pages) >= 2:
        counts: Counter = Counter()
        for lines in pages:
            kept = [l for l in lines if l.strip()]
            edge = kept[:_EDGE_LINES] + kept[-_EDGE_LINES:]
            counts.update({_normalise(l) for l in edge})
        need = max(2, math.ceil(0.6 * len(pages)))
        repeated = {n for n, c in counts.items() if c >= need and n}
    lines: list[str] = []
    removed_edges = 0
    for page in pages:
        kept = [l for l in page if l.strip()]
        edge_ids = {id(l) for l in kept[:_EDGE_LINES] + kept[-_EDGE_LINES:]}
        for l in kept:
            if _PAGE_NUMBER.match(l) or (id(l) in edge_ids and _normalise(l) in repeated):
                removed_edges += 1
                continue
            lines.append(l.strip())
    if removed_edges:
        notes.append(f"removed {removed_edges} header, footer and page-number line(s)")

    # 2. time codes and bracketed notes; lines that held only those disappear
    cleaned, stripped = [], 0
    for l in lines:
        new = _NOTE.sub(" ", _TIMECODE.sub("", l, count=1))
        new = re.sub(r"\s+", " ", new).strip()
        stripped += new != l
        if new:
            cleaned.append(new)
    if stripped:
        notes.append("removed time codes and bracketed notes such as [Applause]")

    # 3. drop the title block: the body starts at the first speaker who speaks more than once
    labels = Counter(filter(None, (_label(l) for l in cleaned)))
    repeat_speakers = {s for s, n in labels.items() if n >= 2}
    start = next((i for i, l in enumerate(cleaned) if _label(l) in repeat_speakers), 0) if repeat_speakers else 0
    if start:
        notes.append(f"skipped {start} line(s) before the first speaker (title block)")
    body = cleaned[start:]

    # 4. one line per turn; join wrapped lines and words split by a hyphen at the end of a line
    turns: list[str] = []
    for l in body:
        if _label(l) is not None or not turns:
            turns.append(l)
        elif re.search(r"[A-Za-z]-$", turns[-1]) and l[:1].islower():
            turns[-1] = turns[-1][:-1] + l
        else:
            turns.append(turns.pop() + " " + l)
    speakers = list(dict.fromkeys(filter(None, (_label(t) for t in turns))))
    if not speakers:
        notes.append("no 'NAME:' speaker lines found; the text is used as plain prose")
    return PdfTranscript(text="\n".join(turns), pages=len(pages), turns=len(turns), speakers=speakers, notes=notes)
