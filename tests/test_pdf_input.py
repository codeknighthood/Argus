"""PDF transcripts: the two bundled demo PDFs, the cleaner, and the app picker."""
import io
from pathlib import Path

import pytest

pytest.importorskip("pypdf")
from argus.pdf_text import extract_transcript  # noqa: E402
from argus.segment import segment_transcript  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample"
COUNCIL = SAMPLES / "council_hearing_congestion_charge.pdf"
PANEL = SAMPLES / "tv_panel_compulsory_voting.pdf"


@pytest.mark.parametrize("path,turns,speakers", [(COUNCIL, 22, 8), (PANEL, 19, 6)])
def test_demo_pdfs_become_clean_turns(path, turns, speakers):
    got = extract_transcript(path)
    lines = got.text.splitlines()
    assert got.turns == turns == len(lines)
    assert len(got.speakers) == speakers
    assert all(": " in l for l in lines)
    for junk in ("Page ", "[Applause]", "[Crosstalk]", "Unofficial transcript", "CNV NETWORK", "19:0", "00:0"):
        assert junk not in got.text


def test_demo_pdfs_segment_with_speakers():
    segs = segment_transcript(extract_transcript(PANEL).text, 2, True, 3)
    assert len(segs) > 19
    assert {s.speaker for s in segs} >= {"HOST CALDER", "JORDAN BELL"}


def test_bytes_and_file_object_work():
    data = PANEL.read_bytes()
    assert extract_transcript(data).turns == extract_transcript(io.BytesIO(data)).turns == 19


def test_bad_pdf_gives_plain_message():
    with pytest.raises(ValueError):
        extract_transcript(b"not a pdf at all")


def test_app_lists_pdf_samples():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(SAMPLES.parent.parent / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.selectbox[0].options[0] == "Minimum wage (text)"
    assert any(t.value == "the minimum wage" for t in at.text_input)
    at.selectbox[0].select("TV panel: compulsory voting (PDF)").run()
    assert not at.exception
    assert any(t.value == "compulsory voting" for t in at.text_input)
    assert "HOST CALDER:" in at.text_area[0].value
