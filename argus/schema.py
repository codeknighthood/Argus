"""Output schemas for ARGUS.

ArgumentRecord mirrors the structured output listed in synopsis section 5.7.
The other models are the small objects each stage asks the LLM to return.
They are passed to Ollama as JSON schemas (so the model can only produce valid
labels) and every reply is validated again with pydantic.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Stance(str, Enum):
    # Synopsis mixes "Neither" (intro, literature review) and "Neutral"
    # (objectives, methodology). We standardise on Neutral.
    FAVOR = "Favor"
    AGAINST = "Against"
    NEUTRAL = "Neutral"


class Component(str, Enum):
    CLAIM = "Claim"
    PREMISE = "Premise"


class ComponentLabel(str, Enum):
    """What the extraction stage returns: a component type, or None for non-arguments."""
    CLAIM = "Claim"
    PREMISE = "Premise"
    NONE = "None"


class Relation(str, Enum):
    SUPPORT = "Support"
    ATTACK = "Attack"
    NONE = "None"


# ---------------------------------------------------------------- per-stage LLM outputs
class StanceResult(BaseModel):
    """Stance stage, zero-shot and few-shot: one (target, statement) pair."""
    stance: Stance
    explanation: str = Field(description="One or two sentences justifying the stance.")


class StanceCoT(BaseModel):
    """Stance stage, chain-of-thought. `reasoning` comes first so the model thinks before it decides."""
    reasoning: str = Field(description="Short step-by-step reasoning before deciding.")
    stance: Stance


class ExtractionResult(BaseModel):
    """Argument extraction stage (synopsis 5.3): the role of one sentence in the debate."""
    component: ComponentLabel
    identified_argument: str = Field(
        description="The argument as one short self-contained sentence; empty string if component is None."
    )
    explanation: str = Field(description="One sentence justifying the component label.")

    @property
    def is_argument(self) -> bool:
        return self.component != ComponentLabel.NONE


class RelationResult(BaseModel):
    """Argument relation stage (synopsis 5.4): how argument B relates to the earlier argument A."""
    relation: Relation
    explanation: str = Field(description="One sentence justifying the relation.")


# ---------------------------------------------------------------- final output (synopsis 5.7)
class ArgumentRecord(BaseModel):
    """One row of the final structured output (synopsis 5.7)."""
    policy_target: str
    original_statement: str
    identified_argument: Optional[str] = None
    component: Optional[Component] = None
    relation: Optional[Relation] = None
    related_to: Optional[str] = None            # text of the argument this one supports / attacks
    stance: Optional[Stance] = None
    explanation: Optional[str] = None           # stance explanation
    speaker_context: Optional[str] = None       # speaker, where the transcript names one
    # bookkeeping, handy for linking rows and for debugging
    segment_index: Optional[int] = None
    is_argument: Optional[bool] = None
    related_to_index: Optional[int] = None
    relation_explanation: Optional[str] = None
    error: Optional[str] = None                 # set if an LLM call failed for this row
