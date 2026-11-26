"""Phase 5: the integrated ARGUS pipeline (synopsis 5.6 and 5.7).

    raw transcript
      -> segmentation: sentences with speaker and carried-over context     (Phase 1)
      -> extraction:   Claim / Premise / None + a standalone restatement   (Phase 3)
      -> stance:       Favor / Against / Neutral toward the policy target  (Phase 2)
      -> relations:    Support / Attack links to earlier arguments         (Phase 4)
      -> one ArgumentRecord per sentence, the structured output of synopsis 5.7

The LLM is used at every stage, which is why it is a shared client passed in, not a final step.
Each stage is wrapped separately: if one model call fails, that row records the error and the run carries on,
so a long transcript is never lost to one bad call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence

from . import __version__
from .extract import extract_argument
from .llm import LLMClient
from .relations import ArgItem, find_relation
from .schema import ArgumentRecord, Component
from .segment import Segment, segment_transcript
from .stance import STRATEGIES, predict_stance

STANCE_ON = ("all", "arguments", "none")
Progress = Callable[[int, int], None]          # called as progress(done, total) after each sentence


@dataclass
class PipelineConfig:
    stance_strategy: str = "few_shot"
    context_window: int = 2
    relation_window: int = 3
    stance_on: str = "all"
    min_tokens: int = 3
    parse_speakers: bool = True

    def __post_init__(self) -> None:
        if self.stance_strategy not in STRATEGIES:
            raise ValueError(f"stance_strategy must be one of {', '.join(STRATEGIES)}; got {self.stance_strategy!r}")
        if self.stance_on not in STANCE_ON:
            raise ValueError(f"stance_on must be one of {', '.join(STANCE_ON)}; got {self.stance_on!r}")
        for name in ("context_window", "relation_window", "min_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "PipelineConfig":
        """Defaults from the `pipeline:` section of config.yaml (missing keys fall back), then overrides.

        Overrides that are None are ignored, so command-line flags can be passed straight through.
        """
        values = dict(cfg.get("pipeline") or {})
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class ArgusPipeline:
    def __init__(self, llm: LLMClient, config: Optional[PipelineConfig] = None):
        self.llm = llm
        self.config = config or PipelineConfig()

    def describe(self) -> dict:
        """Settings that produced a result, saved next to it so a run can be reproduced."""
        return {"argus_version": __version__, "model": getattr(self.llm, "model", None), **asdict(self.config)}

    def run_transcript(self, text: str, target: str, progress: Optional[Progress] = None) -> list[ArgumentRecord]:
        cfg = self.config
        segments = segment_transcript(
            text, context_window=cfg.context_window, parse_speakers=cfg.parse_speakers, min_tokens=cfg.min_tokens,
        )
        return self.run_segments(segments, target, progress)

    def run_segments(
        self, segments: Sequence[Segment], target: str, progress: Optional[Progress] = None
    ) -> list[ArgumentRecord]:
        cfg, llm = self.config, self.llm
        records: list[ArgumentRecord] = []
        earlier: list[ArgItem] = []              # arguments seen so far, for the relation stage
        total = len(segments)

        for done, seg in enumerate(segments, start=1):
            rec = ArgumentRecord(
                policy_target=target, original_statement=seg.sentence,
                speaker_context=seg.speaker, segment_index=seg.index,
            )
            errors: list[str] = []

            # 1. extraction (5.3)
            try:
                ext = extract_argument(llm, target, seg.sentence, seg.context)
                rec.is_argument = ext.is_argument
                if ext.is_argument:
                    rec.component = Component(ext.component.value)
                    rec.identified_argument = ext.identified_argument
            except Exception as e:
                errors.append(f"extraction: {str(e)[:200]}")

            # 2. stance (5.5). "arguments" skips sentences we KNOW are not arguments; "none" skips the stage
            #    (used when scoring extraction and relations on a transcript, which has no stance labels).
            ask_stance = cfg.stance_on == "all" or (cfg.stance_on == "arguments" and rec.is_argument is not False)
            if ask_stance:
                try:
                    st = predict_stance(llm, cfg.stance_strategy, target, seg.sentence, seg.context)
                    rec.stance, rec.explanation = st.stance, st.explanation
                except Exception as e:
                    errors.append(f"stance: {str(e)[:200]}")
            elif cfg.stance_on == "arguments":
                rec.explanation = "Not an argument, so stance was not evaluated."

            # 3. relations (5.4), only for arguments, against the nearest earlier arguments
            if rec.is_argument:
                item = ArgItem(seg.index, seg.speaker, rec.identified_argument or seg.sentence)
                if cfg.relation_window > 0:
                    try:
                        result, linked = find_relation(llm, target, item, earlier, cfg.relation_window)
                        rec.relation, rec.relation_explanation = result.relation, result.explanation
                        if linked is not None:
                            rec.related_to, rec.related_to_index = linked.text, linked.index
                    except Exception as e:
                        errors.append(f"relation: {str(e)[:200]}")
                earlier.append(item)

            rec.error = "; ".join(errors) or None
            records.append(rec)
            if progress:
                progress(done, total)
        return records
