"""Phase 3: argument extraction (synopsis 5.3).

For one sentence of a debate, the LLM decides whether it is a Claim, a Premise or not an argument at all,
and rewrites the argument as a short self-contained sentence (pronouns resolved from the context). That
rewritten sentence is what the relation stage (Phase 4) compares, so "That is also why absences fell." becomes
something that makes sense on its own.
"""
from .llm import LLMClient
from .prompting import render
from .schema import ComponentLabel, ExtractionResult

PROMPT_NAME = "extract_v1"


def build_messages(target: str, statement: str, context: str = "") -> list[dict]:
    text = render(PROMPT_NAME, target=target, statement=statement, context=context or "(none)")
    return [{"role": "user", "content": text}]


def extract_argument(llm: LLMClient, target: str, statement: str, context: str = "") -> ExtractionResult:
    """Role of `statement` in the debate, with a self-contained restatement of the argument.

    Small models sometimes contradict themselves ("None" with a restatement, or "Claim" with an empty one),
    so the result is tidied: None always has an empty restatement, and an argument with no restatement
    falls back to the original sentence.
    """
    result = llm.structured(build_messages(target, statement, context), ExtractionResult)
    restated = result.identified_argument.strip()
    if result.component == ComponentLabel.NONE:
        restated = ""
    elif not restated:
        restated = statement.strip()
    return result.model_copy(update={"identified_argument": restated})
