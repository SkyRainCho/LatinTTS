from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CorpusState(str, Enum):
    DISCOVERED = "DISCOVERED"
    INVENTORIED = "INVENTORIED"
    TEXT_CANDIDATES_READY = "TEXT_CANDIDATES_READY"
    TRANSCRIPT_CONFIRMED = "TRANSCRIPT_CONFIRMED"
    SEGMENTED = "SEGMENTED"
    PAIRED = "PAIRED"
    ALIGNED = "ALIGNED"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class IssueCode(str, Enum):
    INVENTORY_UNSUPPORTED_FORMAT = "INVENTORY_UNSUPPORTED_FORMAT"
    INVENTORY_HASH_MISMATCH = "INVENTORY_HASH_MISMATCH"
    RIGHTS_SCOPE_UNCONFIRMED = "RIGHTS_SCOPE_UNCONFIRMED"
    TRANSCRIPT_SOURCE_NOT_FOUND = "TRANSCRIPT_SOURCE_NOT_FOUND"
    TRANSCRIPT_AMBIGUOUS = "TRANSCRIPT_AMBIGUOUS"
    TRANSCRIPT_SPOKEN_MISMATCH = "TRANSCRIPT_SPOKEN_MISMATCH"
    PAUSE_CLASSES_AMBIGUOUS = "PAUSE_CLASSES_AMBIGUOUS"
    TAKE_COUNT_MISMATCH = "TAKE_COUNT_MISMATCH"
    TAKE_DURATION_MISMATCH = "TAKE_DURATION_MISMATCH"
    TAKE_TEXT_MISMATCH = "TAKE_TEXT_MISMATCH"
    ALIGNER_UNAVAILABLE = "ALIGNER_UNAVAILABLE"
    ALIGNMENT_TEXT_UNSUPPORTED = "ALIGNMENT_TEXT_UNSUPPORTED"
    ALIGNMENT_LOW_CONFIDENCE = "ALIGNMENT_LOW_CONFIDENCE"
    AUDIO_QUALITY_REJECTED = "AUDIO_QUALITY_REJECTED"
    MANIFEST_SCHEMA_MISMATCH = "MANIFEST_SCHEMA_MISMATCH"
    CACHE_ARTIFACT_INVALID = "CACHE_ARTIFACT_INVALID"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    PRONUNCIATION_NEEDS_REVIEW = "PRONUNCIATION_NEEDS_REVIEW"


_NEXT_STATES = {
    CorpusState.DISCOVERED: frozenset({CorpusState.INVENTORIED}),
    CorpusState.INVENTORIED: frozenset({CorpusState.TEXT_CANDIDATES_READY}),
    CorpusState.TEXT_CANDIDATES_READY: frozenset({CorpusState.TRANSCRIPT_CONFIRMED}),
    CorpusState.TRANSCRIPT_CONFIRMED: frozenset({CorpusState.SEGMENTED}),
    CorpusState.SEGMENTED: frozenset({CorpusState.PAIRED}),
    CorpusState.PAIRED: frozenset({CorpusState.ALIGNED}),
    CorpusState.ALIGNED: frozenset({CorpusState.REVIEWED}),
    CorpusState.REVIEWED: frozenset({CorpusState.APPROVED, CorpusState.REJECTED}),
    CorpusState.APPROVED: frozenset(),
    CorpusState.REJECTED: frozenset(),
}


def require_transition(current: CorpusState, target: CorpusState) -> None:
    if target not in _NEXT_STATES[current]:
        raise ValueError(f"illegal corpus state transition: {current.value} -> {target.value}")


@dataclass(frozen=True, slots=True)
class CorpusIssue:
    code: IssueCode
    message: str
    entity_id: str | None = None


class CorpusFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = IssueCode(code).value
