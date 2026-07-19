from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from latintts.normalization import tokenize_words

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_digest(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_finite_number(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be finite")
    number = cast(int | float, value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return float(number)


def _require_ratio(value: object, field: str) -> float:
    number = _require_finite_number(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be between zero and one")
    return number


def _word_key(text: str) -> str:
    words = tokenize_words(text)
    if len(words) != 1:
        raise ValueError("word text must contain exactly one Unicode word")
    decomposed = unicodedata.normalize("NFD", words[0].surface).casefold()
    key = "".join(character for character in decomposed if character.isalpha())
    if not key:
        raise ValueError("word text must contain a Unicode word")
    return key.replace("æ", "ae").replace("œ", "oe")


def _text_word_keys(text: str, field: str) -> tuple[str, ...]:
    _require_nonempty_string(text, field)
    keys = tuple(_word_key(word.surface) for word in tokenize_words(text))
    if not keys:
        raise ValueError(f"{field} must contain at least one Unicode word")
    return keys


def _validate_json_value(value: object, field: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError(f"{field} must contain only finite JSON values")
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{field} object keys must be strings")
            _validate_json_value(item, f"{field}.{key}")
        return
    raise TypeError(f"{field} must contain only JSON values")


def alignment_cache_key(audio_sha256: str, text_sha256: str, config_sha256: str) -> str:
    """Return the content-addressed identity for one backend-neutral request."""
    payload = "\0".join(
        (
            _require_digest(audio_sha256, "audio_sha256"),
            _require_digest(text_sha256, "text_sha256"),
            _require_digest(config_sha256, "config_sha256"),
        )
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class AlignmentRequest:
    """Authoritative inputs to an alignment backend.

    ``spoken_text`` remains authoritative. Backends may derive ``alignment_text``
    for their own tokenization, but it is deliberately not accepted here.
    """

    audio_path: Path
    audio_sha256: str
    spoken_text: str
    config_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.audio_path, Path):
            raise TypeError("audio_path must be a Path")
        _require_digest(self.audio_sha256, "audio_sha256")
        _text_word_keys(self.spoken_text, "spoken_text")
        _require_digest(self.config_sha256, "config_sha256")

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.spoken_text.encode("utf-8")).hexdigest()

    @property
    def cache_key(self) -> str:
        return alignment_cache_key(self.audio_sha256, self.text_sha256, self.config_sha256)


@dataclass(frozen=True, slots=True)
class WordSpan:
    """One word-level half-open timing span, in seconds."""

    text: str
    start_seconds: float
    end_seconds: float
    score: float

    def __post_init__(self) -> None:
        _word_key(self.text)
        start = _require_finite_number(self.start_seconds, "word span start_seconds")
        end = _require_finite_number(self.end_seconds, "word span end_seconds")
        if start < 0 or end <= start:
            raise ValueError("word span must contain text and a positive half-open time range")
        _require_ratio(self.score, "word score")


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Backend-neutral word alignment evidence; never phoneme timing."""

    backend: str
    backend_version: str
    model_id: str
    model_revision: str
    model_license: str
    alignment_text: str
    words: tuple[WordSpan, ...]
    coverage: float
    mean_score: float
    alignment_level: Literal["word"]
    phoneme_timing_status: Literal["not_estimated"]
    warnings: tuple[str, ...]
    raw_output: dict[str, Any]

    def __post_init__(self) -> None:
        for value, field in (
            (self.backend, "backend"),
            (self.backend_version, "backend_version"),
            (self.model_id, "model_id"),
            (self.model_revision, "model_revision"),
            (self.model_license, "model_license"),
        ):
            _require_nonempty_string(value, field)
        _text_word_keys(self.alignment_text, "alignment_text")
        if type(self.words) is not tuple or not self.words:
            raise ValueError("words must contain at least one WordSpan in a tuple")
        if any(type(word) is not WordSpan for word in self.words):
            raise TypeError("words must contain only WordSpan values")
        if any(
            left.end_seconds > right.start_seconds
            for left, right in zip(self.words, self.words[1:], strict=False)
        ):
            raise ValueError("word spans must be ordered and non-overlapping")
        _require_ratio(self.coverage, "coverage")
        _require_ratio(self.mean_score, "mean_score")
        if self.alignment_level != "word" or self.phoneme_timing_status != "not_estimated":
            raise ValueError("pilot alignment must be word-level without phoneme timing")
        if type(self.warnings) is not tuple:
            raise TypeError("warnings must be a tuple")
        for warning in self.warnings:
            _require_nonempty_string(warning, "warnings")
        if type(self.raw_output) is not dict:
            raise TypeError("raw_output must be an object")
        _validate_json_value(self.raw_output, "raw_output")


class AlignmentBackend(Protocol):
    def align(self, request: AlignmentRequest) -> AlignmentResult: ...


def validate_alignment(
    result: AlignmentResult,
    request: AlignmentRequest,
    *,
    audio_duration_seconds: float,
) -> None:
    """Validate backend output against authoritative text and audio duration."""
    if type(result) is not AlignmentResult:
        raise TypeError("result must be an AlignmentResult")
    if type(request) is not AlignmentRequest:
        raise TypeError("request must be an AlignmentRequest")
    AlignmentRequest(
        audio_path=request.audio_path,
        audio_sha256=request.audio_sha256,
        spoken_text=request.spoken_text,
        config_sha256=request.config_sha256,
    )
    AlignmentResult(
        backend=result.backend,
        backend_version=result.backend_version,
        model_id=result.model_id,
        model_revision=result.model_revision,
        model_license=result.model_license,
        alignment_text=result.alignment_text,
        words=result.words,
        coverage=result.coverage,
        mean_score=result.mean_score,
        alignment_level=result.alignment_level,
        phoneme_timing_status=result.phoneme_timing_status,
        warnings=result.warnings,
        raw_output=result.raw_output,
    )
    duration = _require_finite_number(audio_duration_seconds, "audio_duration_seconds")
    if duration <= 0:
        raise ValueError("audio_duration_seconds must be positive")
    if result.words[-1].end_seconds > duration:
        raise ValueError("word span exceeds audio duration")

    aligned_keys = tuple(_word_key(word.text) for word in result.words)
    if _text_word_keys(result.alignment_text, "alignment_text") != aligned_keys:
        raise ValueError("alignment_text must map exactly to word spans")
    spoken_keys = _text_word_keys(request.spoken_text, "spoken_text")
    spoken_index = 0
    for aligned_key in aligned_keys:
        while spoken_index < len(spoken_keys) and spoken_keys[spoken_index] != aligned_key:
            spoken_index += 1
        if spoken_index == len(spoken_keys):
            raise ValueError("word spans must map to spoken_text in order")
        spoken_index += 1

    expected_coverage = len(aligned_keys) / len(spoken_keys)
    if not math.isclose(result.coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("coverage does not match mapped spoken_text words")
    expected_mean_score = sum(word.score for word in result.words) / len(result.words)
    if not math.isclose(result.mean_score, expected_mean_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("mean_score does not match word spans")


def result_to_dict(result: AlignmentResult) -> dict[str, Any]:
    """Serialize only the documented JSON fields of a validated result."""
    if type(result) is not AlignmentResult:
        raise TypeError("result must be an AlignmentResult")
    AlignmentResult(
        backend=result.backend,
        backend_version=result.backend_version,
        model_id=result.model_id,
        model_revision=result.model_revision,
        model_license=result.model_license,
        alignment_text=result.alignment_text,
        words=result.words,
        coverage=result.coverage,
        mean_score=result.mean_score,
        alignment_level=result.alignment_level,
        phoneme_timing_status=result.phoneme_timing_status,
        warnings=result.warnings,
        raw_output=result.raw_output,
    )
    raw = {
        "backend": result.backend,
        "backend_version": result.backend_version,
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "model_license": result.model_license,
        "alignment_text": result.alignment_text,
        "words": [
            {
                "text": word.text,
                "start_seconds": word.start_seconds,
                "end_seconds": word.end_seconds,
                "score": word.score,
            }
            for word in result.words
        ],
        "coverage": result.coverage,
        "mean_score": result.mean_score,
        "alignment_level": result.alignment_level,
        "phoneme_timing_status": result.phoneme_timing_status,
        "warnings": list(result.warnings),
        "raw_output": result.raw_output,
    }
    return raw


def result_from_dict(raw: dict[str, Any]) -> AlignmentResult:
    """Deserialize a canonical JSON result with no tolerated extra fields."""
    if type(raw) is not dict:
        raise TypeError("alignment result must be an object")
    expected = frozenset(field.name for field in fields(AlignmentResult))
    if set(raw) != expected:
        raise ValueError("alignment result must contain exact fields")
    words_raw = raw["words"]
    if type(words_raw) is not list:
        raise TypeError("alignment result words must be a JSON array")
    word_fields = frozenset(field.name for field in fields(WordSpan))
    words: list[WordSpan] = []
    for word_raw in words_raw:
        if type(word_raw) is not dict or set(word_raw) != word_fields:
            raise ValueError("alignment result words must contain exact fields")
        words.append(WordSpan(**word_raw))
    warnings_raw = raw["warnings"]
    if type(warnings_raw) is not list:
        raise TypeError("alignment result warnings must be a JSON array")
    if any(type(warning) is not str for warning in warnings_raw):
        raise TypeError("alignment result warnings must contain strings")
    if type(raw["raw_output"]) is not dict:
        raise TypeError("alignment result raw_output must be an object")
    return AlignmentResult(
        backend=raw["backend"],
        backend_version=raw["backend_version"],
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        model_license=raw["model_license"],
        alignment_text=raw["alignment_text"],
        words=tuple(words),
        coverage=raw["coverage"],
        mean_score=raw["mean_score"],
        alignment_level=raw["alignment_level"],
        phoneme_timing_status=raw["phoneme_timing_status"],
        warnings=tuple(warnings_raw),
        raw_output=raw["raw_output"],
    )
