import pytest

from latintts.domain import (
    PronunciationPlan,
    PronunciationToken,
    ResolutionMethod,
)


def test_token_rejects_stress_outside_syllables() -> None:
    with pytest.raises(ValueError, match="stress_index"):
        PronunciationToken(
            surface="ave",
            normalized="ave",
            source_span=(0, 3),
            syllables=("a", "ve"),
            stress_index=2,
            ipa="ˈa.ve",
            model_phonemes=("ˈ", "a", "v", "e"),
            resolution_method=ResolutionMethod.RULE,
        )


def test_plan_rejects_token_span_outside_original_text() -> None:
    token = PronunciationToken(
        surface="ave",
        normalized="ave",
        source_span=(0, 3),
        syllables=("a", "ve"),
        stress_index=0,
        ipa="ˈa.ve",
        model_phonemes=("ˈ", "a", "v", "e"),
        resolution_method=ResolutionMethod.RULE,
    )
    with pytest.raises(ValueError, match="source_span"):
        PronunciationPlan(
            schema_version="1",
            rule_version="ecclesiastical-roman-v1",
            original_text="av",
            normalized_text="ave",
            tokens=(token,),
            phrase_phonemes=token.model_phonemes,
        )
