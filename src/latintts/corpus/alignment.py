from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from latintts.normalization import tokenize_words

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODEL_REVISION = re.compile(r"[0-9a-f]{40,64}")
LATIN_ALIGNMENT_TRANSFORM_VERSION = "latin-uroman-v1"


def _require_digest(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_revision(value: object, field: str) -> str:
    if type(value) is not str or _MODEL_REVISION.fullmatch(value) is None:
        raise ValueError(f"{field} must be an immutable lowercase model revision")
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


def _freeze_json(value: object, field: str) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        raise ValueError(f"{field} must contain only finite JSON values")
    if type(value) is list:
        return tuple(_freeze_json(item, f"{field}[{index}]") for index, item in enumerate(value))
    if type(value) is dict:
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{field} object keys must be strings")
            frozen[key] = _freeze_json(item, f"{field}.{key}")
        return MappingProxyType(frozen)
    raise TypeError(f"{field} must contain only JSON values")


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: object, field: str) -> str:
    frozen = _freeze_json(value, field)
    return json.dumps(_thaw_json(frozen), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def transform_latin_for_alignment(
    text: str, *, version: str = LATIN_ALIGNMENT_TRANSFORM_VERSION
) -> str:
    """Return the versioned backend-only Latin/uroman-compatible token form."""
    if version != LATIN_ALIGNMENT_TRANSFORM_VERSION:
        raise ValueError("unsupported alignment transform version")
    _require_nonempty_string(text, "text")
    transformed: list[str] = []
    for token in tokenize_words(text):
        decomposed = unicodedata.normalize("NFD", token.surface).casefold()
        letters = "".join(character for character in decomposed if character.isalpha())
        normalized = (
            letters.replace("æ", "ae").replace("œ", "oe").replace("j", "i").replace("v", "u")
        )
        if not normalized:
            raise ValueError("alignment token normalizes to empty")
        transformed.append(normalized)
    if not transformed:
        raise ValueError("text must contain at least one Unicode word")
    return " ".join(transformed)


def _alignment_forms(text: str, field: str) -> tuple[str, ...]:
    transformed = transform_latin_for_alignment(text)
    forms = tuple(token.surface for token in tokenize_words(transformed))
    if not forms:
        raise ValueError(f"{field} must contain at least one alignment token")
    return forms


@dataclass(frozen=True, slots=True)
class AlignmentRequest:
    """Authoritative input and immutable provenance for a backend-neutral run."""

    audio_path: Path
    audio_sha256: str
    spoken_text: str
    segmentation_artifact_sha256: str
    backend: str
    backend_version: str
    model_id: str
    model_revision: str
    model_license: str
    config_sha256: str
    effective_parameters: dict[str, Any]
    alignment_transform_version: str = LATIN_ALIGNMENT_TRANSFORM_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.audio_path, Path):
            raise TypeError("audio_path must be a Path")
        _require_digest(self.audio_sha256, "audio_sha256")
        _require_nonempty_string(self.spoken_text, "spoken_text")
        transform_latin_for_alignment(self.spoken_text, version=self.alignment_transform_version)
        _require_digest(self.segmentation_artifact_sha256, "segmentation_artifact_sha256")
        for value, field in (
            (self.backend, "backend"),
            (self.backend_version, "backend_version"),
            (self.model_id, "model_id"),
            (self.model_license, "model_license"),
        ):
            _require_nonempty_string(value, field)
        _require_revision(self.model_revision, "model_revision")
        _require_digest(self.config_sha256, "config_sha256")
        if type(self.effective_parameters) is not dict or not self.effective_parameters:
            raise ValueError("effective_parameters must be a non-empty JSON object")
        frozen = _freeze_json(self.effective_parameters, "effective_parameters")
        object.__setattr__(self, "effective_parameters", frozen)

    @property
    def alignment_text(self) -> str:
        return transform_latin_for_alignment(
            self.spoken_text, version=self.alignment_transform_version
        )

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.spoken_text.encode("utf-8")).hexdigest()

    @property
    def alignment_text_sha256(self) -> str:
        return hashlib.sha256(self.alignment_text.encode("utf-8")).hexdigest()

    @property
    def cache_key(self) -> str:
        return alignment_cache_key(self)


def alignment_cache_key(request: AlignmentRequest) -> str:
    """Return a canonical content address for every effective alignment input."""
    if type(request) is not AlignmentRequest:
        raise TypeError("request must be an AlignmentRequest")
    payload = {
        "schema_version": "2",
        "audio_sha256": request.audio_sha256,
        "spoken_text_sha256": request.text_sha256,
        "alignment_text_sha256": request.alignment_text_sha256,
        "segmentation_artifact_sha256": request.segmentation_artifact_sha256,
        "backend": request.backend,
        "backend_version": request.backend_version,
        "model_id": request.model_id,
        "model_revision": request.model_revision,
        "model_license": request.model_license,
        "config_sha256": request.config_sha256,
        "effective_parameters": _thaw_json(request.effective_parameters),
        "alignment_transform_version": request.alignment_transform_version,
    }
    return hashlib.sha256(
        _canonical_json(payload, "alignment cache identity").encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AlignmentToken:
    """Reversible mapping from one backend alignment token to one spoken token."""

    alignment_token_index: int
    alignment_form: str
    spoken_token_index: int
    spoken_surface: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.alignment_token_index, "alignment_token_index"),
            (self.spoken_token_index, "spoken_token_index"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if len(_alignment_forms(self.alignment_form, "alignment_form")) != 1:
            raise ValueError("alignment_form must contain exactly one alignment token")
        _require_nonempty_string(self.spoken_surface, "spoken_surface")


@dataclass(frozen=True, slots=True)
class WordSpan:
    """One word-level half-open timing span mapped to its spoken token index."""

    text: str
    start_seconds: float
    end_seconds: float
    score: float
    spoken_token_index: int

    def __post_init__(self) -> None:
        if len(_alignment_forms(self.text, "word text")) != 1:
            raise ValueError("word text must contain exactly one alignment token")
        start = _require_finite_number(self.start_seconds, "word span start_seconds")
        end = _require_finite_number(self.end_seconds, "word span end_seconds")
        if start < 0 or end <= start:
            raise ValueError("word span must contain text and a positive half-open time range")
        _require_ratio(self.score, "word score")
        if type(self.spoken_token_index) is not int or self.spoken_token_index < 0:
            raise ValueError("spoken_token_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Immutable, word-only result with provenance and an integrity digest."""

    backend: str
    backend_version: str
    model_id: str
    model_revision: str
    model_license: str
    alignment_text: str
    tokens: tuple[AlignmentToken, ...]
    words: tuple[WordSpan, ...]
    coverage: float
    mean_score: float
    alignment_level: Literal["word"]
    phoneme_timing_status: Literal["not_estimated"]
    warnings: tuple[str, ...]
    raw_output: dict[str, Any]
    alignment_transform_version: str = LATIN_ALIGNMENT_TRANSFORM_VERSION
    integrity_sha256: str = ""

    def __post_init__(self) -> None:
        for value, field in (
            (self.backend, "backend"),
            (self.backend_version, "backend_version"),
            (self.model_id, "model_id"),
            (self.model_license, "model_license"),
        ):
            _require_nonempty_string(value, field)
        _require_revision(self.model_revision, "model_revision")
        forms = _alignment_forms(self.alignment_text, "alignment_text")
        if type(self.tokens) is not tuple or not self.tokens:
            raise ValueError("tokens must contain at least one AlignmentToken in a tuple")
        if any(type(token) is not AlignmentToken for token in self.tokens):
            raise TypeError("tokens must contain only AlignmentToken values")
        if len(forms) != len(self.tokens):
            raise ValueError("alignment_text must contain one form for every token")
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
        if self.alignment_transform_version != LATIN_ALIGNMENT_TRANSFORM_VERSION:
            raise ValueError("unsupported alignment transform version")
        if type(self.warnings) is not tuple:
            raise TypeError("warnings must be a tuple")
        for warning in self.warnings:
            _require_nonempty_string(warning, "warnings")
        if type(self.raw_output) is not dict:
            raise TypeError("raw_output must be an object")
        frozen_raw = _freeze_json(self.raw_output, "raw_output")
        object.__setattr__(self, "raw_output", frozen_raw)
        expected_integrity = _result_integrity(self)
        if (
            self.integrity_sha256
            and _require_digest(self.integrity_sha256, "integrity_sha256") != expected_integrity
        ):
            raise ValueError("alignment result integrity digest does not match content")
        object.__setattr__(self, "integrity_sha256", expected_integrity)


def _result_payload(result: AlignmentResult) -> dict[str, Any]:
    return {
        "backend": result.backend,
        "backend_version": result.backend_version,
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "model_license": result.model_license,
        "alignment_text": result.alignment_text,
        "tokens": [
            {
                "alignment_token_index": token.alignment_token_index,
                "alignment_form": token.alignment_form,
                "spoken_token_index": token.spoken_token_index,
                "spoken_surface": token.spoken_surface,
            }
            for token in result.tokens
        ],
        "words": [
            {
                "text": word.text,
                "start_seconds": word.start_seconds,
                "end_seconds": word.end_seconds,
                "score": word.score,
                "spoken_token_index": word.spoken_token_index,
            }
            for word in result.words
        ],
        "coverage": result.coverage,
        "mean_score": result.mean_score,
        "alignment_level": result.alignment_level,
        "phoneme_timing_status": result.phoneme_timing_status,
        "warnings": list(result.warnings),
        "raw_output": _thaw_json(result.raw_output),
        "alignment_transform_version": result.alignment_transform_version,
    }


def _result_integrity(result: AlignmentResult) -> str:
    return hashlib.sha256(
        _canonical_json(_result_payload(result), "alignment result").encode("utf-8")
    ).hexdigest()


class AlignmentBackend(Protocol):
    def align(self, request: AlignmentRequest) -> AlignmentResult: ...


def validate_alignment(
    result: AlignmentResult,
    request: AlignmentRequest,
    *,
    audio_duration_seconds: float,
) -> None:
    """Validate complete, reversible token and timing evidence against a request."""
    if type(result) is not AlignmentResult:
        raise TypeError("result must be an AlignmentResult")
    if type(request) is not AlignmentRequest:
        raise TypeError("request must be an AlignmentRequest")
    AlignmentRequest(
        audio_path=request.audio_path,
        audio_sha256=request.audio_sha256,
        spoken_text=request.spoken_text,
        segmentation_artifact_sha256=request.segmentation_artifact_sha256,
        backend=request.backend,
        backend_version=request.backend_version,
        model_id=request.model_id,
        model_revision=request.model_revision,
        model_license=request.model_license,
        config_sha256=request.config_sha256,
        effective_parameters=_thaw_json(request.effective_parameters),
        alignment_transform_version=request.alignment_transform_version,
    )
    _validated_result(result)
    duration = _require_finite_number(audio_duration_seconds, "audio_duration_seconds")
    if duration <= 0:
        raise ValueError("audio_duration_seconds must be positive")
    if result.words[-1].end_seconds > duration:
        raise ValueError("word span exceeds audio duration")
    for field in ("backend", "backend_version", "model_id", "model_revision", "model_license"):
        if getattr(result, field) != getattr(request, field):
            raise ValueError(f"alignment result {field} does not match request provenance")
    if result.alignment_transform_version != request.alignment_transform_version:
        raise ValueError("alignment transform provenance does not match request")
    spoken = tokenize_words(request.spoken_text)
    expected_forms = _alignment_forms(request.alignment_text, "alignment_text")
    expected_indexes = tuple(range(len(spoken)))
    if len(result.tokens) != len(spoken) or len(result.words) != len(spoken):
        raise ValueError("alignment tokens and word spans must cover every spoken token")
    if result.alignment_text != request.alignment_text:
        raise ValueError("alignment_text does not match the request transform")
    token_indexes = tuple(token.alignment_token_index for token in result.tokens)
    token_spoken_indexes = tuple(token.spoken_token_index for token in result.tokens)
    word_spoken_indexes = tuple(word.spoken_token_index for word in result.words)
    if token_indexes != expected_indexes:
        raise ValueError("alignment token indexes must be complete and in index order")
    if token_spoken_indexes != expected_indexes or word_spoken_indexes != expected_indexes:
        raise ValueError("spoken token indexes must be unique, complete, and in index order")
    for index, (token, surface, form, word) in enumerate(
        zip(result.tokens, spoken, expected_forms, result.words, strict=True)
    ):
        if token.spoken_surface != surface.surface or token.alignment_form != form:
            raise ValueError("alignment token mapping does not match spoken_text")
        if _alignment_forms(word.text, "word text") != (form,):
            raise ValueError(f"word span {index} does not match its alignment token")
    if result.coverage != 1.0:
        raise ValueError("complete token mapping requires coverage of one")
    expected_mean_score = sum(word.score for word in result.words) / len(result.words)
    if not math.isclose(result.mean_score, expected_mean_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("mean_score does not match word spans")


def _validated_result(result: AlignmentResult) -> None:
    AlignmentResult(
        backend=result.backend,
        backend_version=result.backend_version,
        model_id=result.model_id,
        model_revision=result.model_revision,
        model_license=result.model_license,
        alignment_text=result.alignment_text,
        tokens=result.tokens,
        words=result.words,
        coverage=result.coverage,
        mean_score=result.mean_score,
        alignment_level=result.alignment_level,
        phoneme_timing_status=result.phoneme_timing_status,
        warnings=result.warnings,
        raw_output=_thaw_json(result.raw_output),
        alignment_transform_version=result.alignment_transform_version,
        integrity_sha256=result.integrity_sha256,
    )


def result_to_dict(result: AlignmentResult) -> dict[str, Any]:
    """Serialize a result as a defensive, integrity-bearing canonical JSON object."""
    if type(result) is not AlignmentResult:
        raise TypeError("result must be an AlignmentResult")
    _validated_result(result)
    return _result_payload(result) | {"integrity_sha256": result.integrity_sha256}


def result_from_dict(raw: dict[str, Any]) -> AlignmentResult:
    """Deserialize an exact result schema and verify its canonical integrity digest."""
    if type(raw) is not dict:
        raise TypeError("alignment result must be an object")
    expected = frozenset(field.name for field in fields(AlignmentResult))
    if set(raw) != expected:
        raise ValueError("alignment result must contain exact fields")
    _require_digest(raw["integrity_sha256"], "integrity_sha256")
    tokens_raw = raw["tokens"]
    if type(tokens_raw) is not list:
        raise TypeError("alignment result tokens must be a JSON array")
    token_fields = frozenset(field.name for field in fields(AlignmentToken))
    tokens: list[AlignmentToken] = []
    for token_raw in tokens_raw:
        if type(token_raw) is not dict or set(token_raw) != token_fields:
            raise ValueError("alignment result tokens must contain exact fields")
        tokens.append(AlignmentToken(**token_raw))
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
        tokens=tuple(tokens),
        words=tuple(words),
        coverage=raw["coverage"],
        mean_score=raw["mean_score"],
        alignment_level=raw["alignment_level"],
        phoneme_timing_status=raw["phoneme_timing_status"],
        warnings=tuple(warnings_raw),
        raw_output=raw["raw_output"],
        alignment_transform_version=raw["alignment_transform_version"],
        integrity_sha256=raw["integrity_sha256"],
    )
