"""Show how a debate transcript becomes sentence segments with speaker and context.

Run from the project root:
    python scripts/segment_demo.py                  # built-in toy transcript
    python scripts/segment_demo.py path/to/file.txt # your own transcript (NAME: text lines)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.segment import segment_transcript  # noqa: E402

DEMO = """\
MODERATOR: Welcome everyone. Tonight we debate raising the minimum wage.
DR. SMITH: I support it. Full-time workers should not live in poverty.
Studies show modest increases barely affect employment.
MR. JONES: I disagree. Small businesses run on thin margins. A higher wage floor means layoffs.
"""


def main() -> None:
    text = Path(sys.argv[1]).read_text(encoding="utf-8") if len(sys.argv) > 1 else DEMO
    for s in segment_transcript(text, context_window=2):
        print(f"[{s.index}] ({s.speaker}) {s.sentence}")
        if s.context:
            print(f"      context: {s.context}")


if __name__ == "__main__":
    main()
