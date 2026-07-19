from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from latintts.domain import PronunciationOverride, PronunciationPlan
from latintts.normalization import tokenize_words
from latintts.pipeline import Pronouncer


class TranscriptInputError(ValueError):
    def __init__(self, message: str, code: str = "MANIFEST_SCHEMA_MISMATCH") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class TextCandidate:
    candidate_id: str
    source_id: str
    source_url: str
    source_version: str
    accessed_at: str
    source_sha256: str
    source_text: str


@dataclass(frozen=True, slots=True)
class SpokenUnit:
    unit_id: str
    ordinal: int
    text: str
    token_start_index: int
    token_end_index: int


@dataclass(frozen=True, slots=True)
class TextDifference:
    operation: str
    source_tokens: tuple[str, ...]
    spoken_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    candidate_id: str
    differences_from_selected: tuple[TextDifference, ...]


@dataclass(frozen=True, slots=True)
class AsrObservation:
    hypothesis: str
    differences: tuple[TextDifference, ...]
    confirmed_text: None = None


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    schema_version: str
    recording_id: str
    source_candidates: tuple[TextCandidate, ...]
    selected_candidate_id: str
    source_text: str
    spoken_text: str
    spoken_units: tuple[SpokenUnit, ...]
    normalized_text: str
    pronunciation_plan: dict[str, Any]
    differences: tuple[TextDifference, ...]
    candidate_comparisons: tuple[CandidateComparison, ...]
    state: str
    asr_observation: AsrObservation | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pronunciation_plan_to_dict(plan: PronunciationPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "rule_version": plan.rule_version,
        "original_text": plan.original_text,
        "normalized_text": plan.normalized_text,
        "tokens": [
            {
                "surface": token.surface,
                "normalized": token.normalized,
                "source_span": list(token.source_span),
                "syllables": list(token.syllables),
                "stress_index": token.stress_index,
                "ipa": token.ipa,
                "model_phonemes": list(token.model_phonemes),
                "resolution_method": token.resolution_method.value,
                "source_ids": list(token.source_ids),
                "warning_codes": [warning.code for warning in token.warnings],
            }
            for token in plan.tokens
        ],
        "phrase_phonemes": list(plan.phrase_phonemes),
        "warning_codes": [warning.code for warning in plan.warnings],
    }


def _differences(source: str, spoken: str) -> tuple[TextDifference, ...]:
    source_tokens = source.split()
    spoken_tokens = spoken.split()
    matcher = SequenceMatcher(a=source_tokens, b=spoken_tokens, autojunk=False)
    return tuple(
        TextDifference(tag, tuple(source_tokens[i1:i2]), tuple(spoken_tokens[j1:j2]))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def build_text_candidate(
    *,
    source_id: str,
    source_url: str,
    source_version: str,
    accessed_at: str,
    source_text: str,
) -> TextCandidate:
    if not all(
        value.strip()
        for value in (source_id, source_url, source_version, accessed_at, source_text)
    ):
        raise ValueError("text candidate fields must not be empty")
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    identity = hashlib.sha256(
        "\0".join((source_id, source_version, digest)).encode("utf-8")
    ).hexdigest()[:16]
    return TextCandidate(
        f"text-{identity}",
        source_id,
        source_url,
        source_version,
        accessed_at,
        digest,
        source_text,
    )


def build_transcript(
    *,
    recording_id: str,
    source_candidates: tuple[TextCandidate, ...],
    selected_candidate_id: str,
    spoken_unit_lines: tuple[str, ...],
    pronunciation_overrides: Mapping[int, PronunciationOverride] | None = None,
    pronouncer: Pronouncer | None = None,
) -> TranscriptRecord:
    candidates = {candidate.candidate_id: candidate for candidate in source_candidates}
    if len(candidates) != len(source_candidates) or selected_candidate_id not in candidates:
        raise ValueError("text candidates must be unique and contain the selected candidate")
    selected = candidates[selected_candidate_id]
    units_text = tuple(line.strip() for line in spoken_unit_lines if line.strip())
    if not units_text:
        raise ValueError("spoken transcript must contain at least one non-empty unit")
    spoken_text = "\n".join(units_text)
    plan = (pronouncer or Pronouncer.default()).analyze(
        spoken_text,
        overrides=pronunciation_overrides,
    )
    if plan.schema_version != "1":
        raise ValueError("PronunciationPlan schema_version must be '1'")
    if plan.rule_version != "ecclesiastical-roman-v1":
        raise ValueError("PronunciationPlan rule_version must be ecclesiastical-roman-v1")
    units_list: list[SpokenUnit] = []
    token_cursor = 0
    for index, text in enumerate(units_text, 1):
        token_count = len(tokenize_words(text))
        units_list.append(
            SpokenUnit(
                f"{recording_id}-unit-{index:04d}",
                index,
                text,
                token_cursor,
                token_cursor + token_count,
            )
        )
        token_cursor += token_count
    units = tuple(units_list)
    if token_cursor != len(plan.tokens):
        raise ValueError("spoken unit token ranges do not cover PronunciationPlan tokens")
    return TranscriptRecord(
        schema_version="1",
        recording_id=recording_id,
        source_candidates=source_candidates,
        selected_candidate_id=selected_candidate_id,
        source_text=selected.source_text,
        spoken_text=spoken_text,
        spoken_units=units,
        normalized_text=plan.normalized_text,
        pronunciation_plan=pronunciation_plan_to_dict(plan),
        differences=_differences(selected.source_text, spoken_text),
        candidate_comparisons=tuple(
            CandidateComparison(
                candidate.candidate_id,
                _differences(selected.source_text, candidate.source_text),
            )
            for candidate in source_candidates
            if candidate.candidate_id != selected_candidate_id
        ),
        state="TRANSCRIPT_CONFIRMED",
    )


def compare_asr_observation(spoken_text: str, hypothesis: str) -> AsrObservation:
    return AsrObservation(hypothesis, _differences(spoken_text, hypothesis))
