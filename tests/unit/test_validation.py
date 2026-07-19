from typing import cast

import pytest

from latintts.domain import PronunciationOverride
from latintts.validation import validate_override


def test_override_rejects_unknown_model_phoneme() -> None:
    override = PronunciationOverride(
        ipa="ˈaunknown",
        model_phonemes=("ˈ", "a", "unknown"),
    )

    with pytest.raises(ValueError, match="unknown model phonemes"):
        validate_override(override, syllable_count=1)


def test_override_rejects_ipa_without_matching_model_tokens() -> None:
    override = PronunciationOverride(ipa="ˈa.ve")

    with pytest.raises(ValueError, match="ipa and model_phonemes"):
        validate_override(override, syllable_count=2)


def test_override_accepts_stress_only() -> None:
    validate_override(PronunciationOverride(stress_index=1), syllable_count=3)


def test_override_rejects_stress_marker_on_different_syllable() -> None:
    override = PronunciationOverride(
        ipa="ˈdo.mi.nus",
        model_phonemes=("ˈ", "d", "o", ".", "m", "i", ".", "n", "u", "s"),
    )

    with pytest.raises(ValueError, match="stress token"):
        validate_override(override, syllable_count=3, resolved_stress_index=1)


def test_override_accepts_non_initial_stress_with_structured_ipa_rendering() -> None:
    override = PronunciationOverride(
        ipa="deˈʃen.dit",
        model_phonemes=("d", "e", ".", "ˈ", "ʃ", "e", "n", ".", "d", "i", "t"),
    )

    validate_override(override, syllable_count=3, resolved_stress_index=1)


def test_override_rejects_structurally_mismatched_non_initial_stress_ipa() -> None:
    override = PronunciationOverride(
        ipa="de.ˈʃen.dit",
        model_phonemes=("d", "e", ".", "ˈ", "ʃ", "e", "n", ".", "d", "i", "t"),
    )

    with pytest.raises(ValueError, match="ipa and model_phonemes"):
        validate_override(override, syllable_count=3, resolved_stress_index=1)


@pytest.mark.parametrize(
    "override",
    [
        PronunciationOverride(),
        PronunciationOverride(ipa="ˈa", model_phonemes=("ˈ", "a", "ˈ")),
        PronunciationOverride(ipa="ˈa", model_phonemes=("ˈ", "a", ".")),
    ],
)
def test_override_rejects_incomplete_or_malformed_content(
    override: PronunciationOverride,
) -> None:
    with pytest.raises(ValueError):
        validate_override(override, syllable_count=1)


def test_override_requires_positive_syllable_count() -> None:
    with pytest.raises(ValueError, match="syllable_count must be positive"):
        validate_override(PronunciationOverride(stress_index=0), syllable_count=0)


def test_override_rejects_out_of_range_stress_index() -> None:
    with pytest.raises(ValueError, match="override stress_index"):
        validate_override(PronunciationOverride(stress_index=3), syllable_count=3)


@pytest.mark.parametrize("stress_index", [False, 1.0, "1"])
def test_override_rejects_non_integer_stress_index(stress_index: object) -> None:
    override = PronunciationOverride(stress_index=cast(int, stress_index))

    with pytest.raises(ValueError, match="override stress_index must be an integer"):
        validate_override(override, syllable_count=3)
