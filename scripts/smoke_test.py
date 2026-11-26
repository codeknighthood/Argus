"""Phase 0 check: one LLM call must return a valid, schema-checked stance.

Run from the project root:   python scripts/smoke_test.py
Run it twice: the second run should say "(from cache)" and return instantly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.llm import LLMClient, ensure_ready  # noqa: E402
from argus.schema import StanceResult    # noqa: E402

PROMPT = (ROOT / "argus" / "prompts" / "stance_v1.txt").read_text(encoding="utf-8")

SAMPLES = [
    {
        "target": "Raising the minimum wage",
        "context": "",
        "statement": "Workers who put in full-time hours should not have to live in poverty.",
    },
    {
        "target": "Raising the minimum wage",
        "context": "",
        "statement": "Small businesses already operate on thin margins and a higher wage floor will force layoffs.",
    },
]


def build_messages(sample: dict) -> list[dict]:
    # str.replace instead of str.format: the prompt contains literal { } for the JSON example.
    text = (PROMPT.replace("{target}", sample["target"])
                  .replace("{context}", sample["context"] or "(none)")
                  .replace("{statement}", sample["statement"]))
    return [{"role": "user", "content": text}]


def main() -> None:
    llm = ensure_ready(LLMClient())  # plain message if Ollama is not running or the model is not pulled
    print(f"Local model: {llm.model}\n")
    for s in SAMPLES:
        msgs = build_messages(s)
        cached = llm._cache_path(msgs, StanceResult).exists()
        result = llm.structured(msgs, StanceResult)
        print(f"Statement : {s['statement']}")
        print(f"Stance    : {result.stance.value}{'  (from cache)' if cached else ''}")
        print(f"Reason    : {result.explanation}\n")
    print("Phase 0 OK: valid structured output received.")


if __name__ == "__main__":
    main()
