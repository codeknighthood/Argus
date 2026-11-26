"""Prompt loading. Prompts live in argus/prompts/*.txt so each experiment is reproducible
(the file name carries the version, e.g. stance_v1) and prompts can be edited without touching code.

Placeholders look like {target}. Rendering is a single pass, so text that is inserted
(a statement that happens to contain "{context}", say) is never substituted again, and the
literal JSON braces in the prompts are left alone.
"""
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render(name: str, **values) -> str:
    text = load_prompt(name)
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), text)


def placeholders(name: str) -> set[str]:
    return set(_PLACEHOLDER.findall(load_prompt(name)))
