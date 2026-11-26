"""Phase 2: stance detection (synopsis 5.5).

Three prompting strategies are compared in the evaluation (synopsis 5.6, "different prompting
strategies may be explored"):

  zero_shot  instructions only
  few_shot   instructions plus six worked examples on topics outside the benchmark
  cot        chain-of-thought: the model writes its reasoning first, then the label

All three return a StanceResult (stance + explanation), so the rest of ARGUS does not care
which strategy produced it.
"""
from typing import Type

from pydantic import BaseModel

from .llm import LLMClient
from .prompting import render
from .schema import StanceCoT, StanceResult

# strategy name -> (prompt file, schema the LLM must answer with)
STRATEGIES: dict[str, tuple[str, Type[BaseModel]]] = {
    "zero_shot": ("stance_v1", StanceResult),
    "few_shot": ("stance_fewshot_v1", StanceResult),
    "cot": ("stance_cot_v1", StanceCoT),
}


def build_messages(strategy: str, target: str, statement: str, context: str = "") -> list[dict]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose one of: {', '.join(STRATEGIES)}")
    prompt_name, _ = STRATEGIES[strategy]
    text = render(prompt_name, target=target, statement=statement, context=context or "(none)")
    return [{"role": "user", "content": text}]


def predict_stance(
    llm: LLMClient, strategy: str, target: str, statement: str, context: str = ""
) -> StanceResult:
    """Stance of `statement` toward `target`, using the chosen prompting strategy."""
    _, schema = STRATEGIES[strategy]
    result = llm.structured(build_messages(strategy, target, statement, context), schema)
    if isinstance(result, StanceCoT):
        return StanceResult(stance=result.stance, explanation=result.reasoning)
    return result
