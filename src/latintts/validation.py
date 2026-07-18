from latintts.domain import PronunciationOverride
from latintts.g2p import PHONEME_INVENTORY

CONTROL_TOKENS = frozenset({"ˈ", "."})


def _render_structured_ipa(model_phonemes: tuple[str, ...]) -> tuple[str, int]:
    syllables: list[list[str]] = [[]]
    for token in model_phonemes:
        if token == ".":
            syllables.append([])
        else:
            syllables[-1].append(token)

    if any(not syllable or syllable == ["ˈ"] for syllable in syllables):
        raise ValueError("model_phonemes must contain a phoneme in every syllable")

    stress_index = next(index for index, syllable in enumerate(syllables) if "ˈ" in syllable)
    if syllables[stress_index][0] != "ˈ":
        raise ValueError("primary stress token must begin a model syllable")

    rendered: list[str] = []
    for index, syllable in enumerate(syllables):
        if index and index != stress_index:
            rendered.append(".")
        if index == stress_index:
            rendered.append("ˈ")
            syllable = syllable[1:]
        rendered.extend(syllable)
    return "".join(rendered), stress_index


def validate_override(
    override: PronunciationOverride,
    syllable_count: int,
    resolved_stress_index: int | None = None,
) -> None:
    if syllable_count < 1:
        raise ValueError("syllable_count must be positive")
    if override.stress_index is None and override.ipa is None and override.model_phonemes is None:
        raise ValueError("pronunciation override must change at least one field")
    if override.stress_index is not None and not 0 <= override.stress_index < syllable_count:
        raise ValueError("override stress_index must point to an existing syllable")
    if (override.ipa is None) != (override.model_phonemes is None):
        raise ValueError("ipa and model_phonemes must be supplied together")
    if override.model_phonemes is None:
        return

    unknown = set(override.model_phonemes) - PHONEME_INVENTORY - CONTROL_TOKENS
    if unknown:
        raise ValueError(f"unknown model phonemes: {sorted(unknown)}")
    if override.model_phonemes.count("ˈ") != 1:
        raise ValueError("model_phonemes must contain exactly one primary stress token")
    if override.model_phonemes.count(".") != syllable_count - 1:
        raise ValueError("model_phonemes syllable separators do not match syllables")

    rendered_ipa, stress_token_index = _render_structured_ipa(override.model_phonemes)
    if override.ipa != rendered_ipa:
        raise ValueError("ipa and model_phonemes must describe the same token sequence")
    if resolved_stress_index is not None and stress_token_index != resolved_stress_index:
        raise ValueError("model_phonemes stress token does not match resolved stress_index")
