from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SourceSpan = tuple[int, int]


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ResolutionMethod(str, Enum):
    OVERRIDE = "override"
    EXCEPTION = "exception"
    LEXICON = "lexicon"
    RULE = "rule"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    message: str
    severity: Severity
    source_span: SourceSpan | None = None
    token_index: int | None = None


@dataclass(frozen=True, slots=True)
class PronunciationOverride:
    stress_index: int | None = None
    ipa: str | None = None
    model_phonemes: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class PronunciationToken:
    surface: str
    normalized: str
    source_span: SourceSpan
    syllables: tuple[str, ...]
    stress_index: int
    ipa: str
    model_phonemes: tuple[str, ...]
    resolution_method: ResolutionMethod
    applied_rule_ids: tuple[str, ...] = ()
    normalization_transforms: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    warnings: tuple[Diagnostic, ...] = ()
    override: PronunciationOverride | None = None

    def __post_init__(self) -> None:
        start, end = self.source_span
        if start < 0 or end <= start:
            raise ValueError("source_span must be a non-empty half-open range")
        if not self.syllables:
            raise ValueError("syllables must not be empty")
        if not 0 <= self.stress_index < len(self.syllables):
            raise ValueError("stress_index must point to an existing syllable")


@dataclass(frozen=True, slots=True)
class PronunciationPlan:
    schema_version: str
    rule_version: str
    original_text: str
    normalized_text: str
    tokens: tuple[PronunciationToken, ...]
    phrase_phonemes: tuple[str, ...]
    warnings: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for token in self.tokens:
            if token.source_span[1] > len(self.original_text):
                raise ValueError("token source_span exceeds original_text")
