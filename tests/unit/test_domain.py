import pytest

from latintts.domain import (
    PronunciationPlan,
    PronunciationToken,
    ResolutionMethod,
)


def _token(*, source_span: tuple[int, int] = (0, 3)) -> PronunciationToken:
    return PronunciationToken(
        surface="ave",
        normalized="ave",
        source_span=source_span,
        syllables=("a", "ve"),
        stress_index=0,
        ipa="ˈa.ve",
        model_phonemes=("ˈ", "a", ".", "v", "e"),
        resolution_method=ResolutionMethod.RULE,
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


@pytest.mark.parametrize(
    "source_span",
    [(-1, 3), (0, 0), (2, 1)],
    ids=("negative-start", "empty", "reversed"),
)
def test_token_rejects_invalid_source_span(source_span: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="source_span must be a non-empty half-open range"):
        _token(source_span=source_span)


def test_token_rejects_empty_syllables() -> None:
    with pytest.raises(ValueError, match="syllables must not be empty"):
        PronunciationToken(
            surface="ave",
            normalized="ave",
            source_span=(0, 3),
            syllables=(),
            stress_index=0,
            ipa="ˈa.ve",
            model_phonemes=("ˈ", "a", ".", "v", "e"),
            resolution_method=ResolutionMethod.RULE,
        )


def test_plan_rejects_token_span_outside_original_text() -> None:
    token = _token()
    with pytest.raises(ValueError, match="source_span"):
        PronunciationPlan(
            schema_version="1",
            rule_version="ecclesiastical-roman-v1",
            original_text="av",
            normalized_text="ave",
            tokens=(token,),
            phrase_phonemes=token.model_phonemes,
        )
