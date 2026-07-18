from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from latintts.domain import (
    WORD_BOUNDARY_TOKEN,
    Diagnostic,
    PronunciationOverride,
    PronunciationPlan,
    PronunciationToken,
    ResolutionMethod,
)
from latintts.g2p import (
    G2PExceptionEntry,
    ecclesiastical_g2p,
    load_g2p_exceptions,
)
from latintts.normalization import (
    NormalizedWord,
    normalize_phrase,
    normalize_word,
    tokenize_words,
)
from latintts.stress import StressLexiconEntry, load_stress_lexicon, resolve_stress
from latintts.syllables import syllabify, syllable_ranges
from latintts.validation import validate_override


def _marked_syllable_index(
    word: NormalizedWord,
    ranges: tuple[tuple[int, int], ...],
) -> int | None:
    if word.marked_vowel_index is None:
        return None
    for syllable_index, (start, end) in enumerate(ranges):
        if start <= word.marked_vowel_index < end:
            return syllable_index
    raise ValueError("marked stress position does not belong to a syllable")


def _validate_token_override(
    override: PronunciationOverride,
    *,
    syllable_count: int,
    token_index: int,
    surface: str,
    resolved_stress_index: int | None = None,
) -> None:
    try:
        validate_override(override, syllable_count, resolved_stress_index)
    except ValueError as error:
        raise ValueError(
            f"invalid override for token {token_index} {surface!r}: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class Pronouncer:
    rule_version: str
    stress_lexicon: Mapping[str, StressLexiconEntry]
    g2p_exceptions: Mapping[str, G2PExceptionEntry]

    @classmethod
    def default(cls) -> Pronouncer:
        return cls(
            rule_version="ecclesiastical-roman-v1",
            stress_lexicon=load_stress_lexicon(),
            g2p_exceptions=load_g2p_exceptions(),
        )

    def analyze(
        self,
        text: str,
        overrides: Mapping[int, PronunciationOverride] | None = None,
    ) -> PronunciationPlan:
        override_map = dict(overrides or {})
        spans = tokenize_words(text)
        invalid_indexes = {
            index for index in override_map if type(index) is not int or not 0 <= index < len(spans)
        }
        if invalid_indexes:
            ordered_indexes = sorted(
                invalid_indexes,
                key=lambda value: (type(value).__name__, repr(value)),
            )
            raise ValueError(f"override token indexes do not exist: {ordered_indexes}")

        tokens: list[PronunciationToken] = []
        plan_warnings: list[Diagnostic] = []
        phrase_phonemes: list[str] = []
        for token_index, span in enumerate(spans):
            word = normalize_word(span.surface)
            ranges = syllable_ranges(word.normalized)
            syllables = syllabify(word.normalized)
            explicit_stress = _marked_syllable_index(word, ranges)
            override = override_map.get(token_index)
            if override is not None:
                _validate_token_override(
                    override,
                    syllable_count=len(syllables),
                    token_index=token_index,
                    surface=span.surface,
                )
            stress = resolve_stress(
                word.lookup_key,
                syllables,
                self.stress_lexicon,
                explicit_stress_index=explicit_stress,
                override_index=override.stress_index if override else None,
            )
            if override is not None:
                _validate_token_override(
                    override,
                    syllable_count=len(syllables),
                    token_index=token_index,
                    surface=span.surface,
                    resolved_stress_index=stress.stress_index,
                )
            g2p = ecclesiastical_g2p(
                word.normalized,
                syllables,
                stress.stress_index,
                lookup_key=word.lookup_key,
                exceptions=self.g2p_exceptions,
            )
            has_full_override = (
                override is not None
                and override.ipa is not None
                and override.model_phonemes is not None
            )
            if has_full_override:
                warning_items = stress.warnings
                applied_rule_ids = stress.applied_rule_ids
                source_ids = stress.source_ids
            else:
                warning_items = (*stress.warnings, *g2p.warnings)
                applied_rule_ids = tuple(
                    dict.fromkeys((*stress.applied_rule_ids, *g2p.applied_rule_ids))
                )
                source_ids = tuple(dict.fromkeys((*stress.source_ids, *g2p.source_ids)))
            warnings = tuple(
                replace(item, source_span=(span.start, span.end), token_index=token_index)
                for item in warning_items
            )
            if override is not None:
                method = ResolutionMethod.OVERRIDE
            elif word.lookup_key in self.g2p_exceptions:
                method = ResolutionMethod.EXCEPTION
            else:
                method = stress.method
            ipa = override.ipa if override and override.ipa is not None else g2p.ipa
            phonemes = (
                override.model_phonemes
                if override and override.model_phonemes is not None
                else g2p.phonemes
            )
            token = PronunciationToken(
                surface=span.surface,
                normalized=word.normalized,
                source_span=(span.start, span.end),
                syllables=syllables,
                stress_index=stress.stress_index,
                ipa=ipa,
                model_phonemes=phonemes,
                resolution_method=method,
                applied_rule_ids=applied_rule_ids,
                normalization_transforms=word.transformations,
                source_ids=source_ids,
                warnings=warnings,
                override=override,
            )
            if phrase_phonemes:
                phrase_phonemes.append(WORD_BOUNDARY_TOKEN)
            phrase_phonemes.extend(token.model_phonemes)
            tokens.append(token)
            plan_warnings.extend(warnings)

        return PronunciationPlan(
            schema_version="1",
            rule_version=self.rule_version,
            original_text=text,
            normalized_text=normalize_phrase(text),
            tokens=tuple(tokens),
            phrase_phonemes=tuple(phrase_phonemes),
            warnings=tuple(plan_warnings),
        )
