from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import wave
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol

from latintts.corpus.alignment import (
    AlignmentRequest,
    AlignmentResult,
    result_from_dict,
    result_to_dict,
    validate_alignment,
)
from latintts.corpus.audio import DerivedAudio, RunCommand, extract_analysis_segment
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.pauses import PauseAnalysis, PauseInterval
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from latintts.corpus.transcripts import SpokenUnit
from latintts.corpus.vad import SpeechInterval

_SAFE_COMPONENT_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_CANDIDATE_OUTPUT_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(
        {"codec": "pcm_s16le", "channels": 1, "sample_rate": 16000},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _require_safe_component(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or any(character not in _SAFE_COMPONENT_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field} must be a safe path component")
    return value


def require_safe_pairing_id(value: object, field: str) -> str:
    """Validate an identifier before it can influence pairing paths or artifact IDs."""
    return _require_safe_component(value, field)


@dataclass(frozen=True, slots=True)
class TextUnitWindow:
    unit_id: str
    text: str
    start_sample: int
    end_sample: int
    short_pause_ranges: tuple[tuple[int, int], ...]
    token_start_index: int = 0
    token_end_index: int = 0

    def __post_init__(self) -> None:
        _require_safe_component(self.unit_id, "unit_id")
        if type(self.text) is not str or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        for value, field in (
            (self.start_sample, "start_sample"),
            (self.end_sample, "end_sample"),
            (self.token_start_index, "token_start_index"),
            (self.token_end_index, "token_end_index"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.end_sample <= self.start_sample:
            raise ValueError("text unit window must be a positive half-open range")
        if self.token_end_index and self.token_end_index <= self.token_start_index:
            raise ValueError("token range must be positive when provided")
        if type(self.short_pause_ranges) is not tuple:
            raise TypeError("short_pause_ranges must be a tuple")
        previous_end = self.start_sample
        for pause in self.short_pause_ranges:
            if (
                type(pause) is not tuple
                or len(pause) != 2
                or any(type(value) is not int for value in pause)
            ):
                raise TypeError("short pause ranges must be integer pairs")
            start, end = pause
            if start < previous_end or end <= start or end > self.end_sample:
                raise ValueError("short pause ranges must be ordered inside the text unit")
            previous_end = end


@dataclass(frozen=True, slots=True)
class SplitEvidence:
    split_sample: int
    first_alignment_score: float
    second_alignment_score: float
    word_order_same: bool
    coverage: float
    first_audio: DerivedAudio | None = None
    second_audio: DerivedAudio | None = None
    first_alignment_cache_key: str | None = None
    second_alignment_cache_key: str | None = None
    first_result_integrity_sha256: str | None = None
    second_result_integrity_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.split_sample) is not int or self.split_sample < 0:
            raise ValueError("split_sample must be a non-negative integer")
        for value, field in (
            (self.first_alignment_score, "first_alignment_score"),
            (self.second_alignment_score, "second_alignment_score"),
            (self.coverage, "coverage"),
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between zero and one")
        if type(self.word_order_same) is not bool:
            raise TypeError("word_order_same must be a boolean")

    @property
    def rank(self) -> tuple[float, float]:
        return (min(self.first_alignment_score, self.second_alignment_score), self.coverage)


@dataclass(frozen=True, slots=True)
class TakeCandidate:
    take_index: int
    start_sample: int
    end_sample: int
    audio_relative_path: str = ""
    audio_sha256: str = ""
    alignment_cache_key: str = ""
    alignment_integrity_sha256: str = ""
    alignment_score: float = 0.0
    coverage: float = 0.0

    def __post_init__(self) -> None:
        if self.take_index not in (1, 2) or type(self.take_index) is not int:
            raise ValueError("take_index must be one or two")
        if (
            type(self.start_sample) is not int
            or type(self.end_sample) is not int
            or self.start_sample < 0
            or self.end_sample <= self.start_sample
        ):
            raise ValueError("take candidate must be a positive half-open sample range")
        evidence_values = (
            self.audio_relative_path,
            self.audio_sha256,
            self.alignment_cache_key,
            self.alignment_integrity_sha256,
        )
        if any(evidence_values) and not all(evidence_values):
            raise ValueError("take candidate quality provenance must be complete")
        for value, field in (
            (self.alignment_score, "alignment_score"),
            (self.coverage, "coverage"),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} must be a finite ratio")


@dataclass(frozen=True, slots=True)
class RepetitionGroup:
    repetition_group_id: str
    recording_id: str
    unit_id: str
    text: str
    takes: tuple[TakeCandidate, TakeCandidate]
    selected_evidence: SplitEvidence


@dataclass(frozen=True, slots=True)
class PairingParameters:
    minimum_duration_ratio: float
    maximum_duration_ratio: float
    minimum_score_margin: float

    def __post_init__(self) -> None:
        for value, field in (
            (self.minimum_duration_ratio, "minimum_duration_ratio"),
            (self.maximum_duration_ratio, "maximum_duration_ratio"),
            (self.minimum_score_margin, "minimum_score_margin"),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")
        if self.minimum_duration_ratio > self.maximum_duration_ratio:
            raise ValueError("duration ratio bounds are reversed")

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_duration_ratio": self.minimum_duration_ratio,
            "maximum_duration_ratio": self.maximum_duration_ratio,
            "minimum_score_margin": self.minimum_score_margin,
        }


@dataclass(frozen=True, slots=True)
class PairingGroupOutcome:
    unit_id: str
    status: Literal["selected", "review"]
    issue_code: str | None
    candidates: tuple[SplitEvidence, ...]
    group: RepetitionGroup | None


@dataclass(frozen=True, slots=True)
class PairingRecording:
    schema_version: str
    recording_id: str
    config_sha256: str
    segmentation_artifact_sha256: str
    analysis_audio_relative_path: str
    analysis_audio_sha256: str
    ffmpeg_version: str
    alignment_runtime_sha256s: tuple[str, ...]
    cache_key: str
    pairing_parameters: PairingParameters
    windows: tuple[TextUnitWindow, ...]
    groups: tuple[PairingGroupOutcome, ...]
    integrity_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.ffmpeg_version) is not str or not self.ffmpeg_version.strip():
            raise ValueError("ffmpeg_version must be a non-empty string")
        if type(self.alignment_runtime_sha256s) is not tuple or any(
            not _is_digest(value) for value in self.alignment_runtime_sha256s
        ):
            raise ValueError("alignment runtime identities must be SHA-256 digests")
        if len(set(self.alignment_runtime_sha256s)) != len(self.alignment_runtime_sha256s):
            raise ValueError("alignment runtime identities must be unique")
        expected = _pairing_integrity(self)
        if self.integrity_sha256 and self.integrity_sha256 != expected:
            raise ValueError("pairing artifact integrity digest does not match content")
        object.__setattr__(self, "integrity_sha256", expected)


@dataclass(frozen=True, slots=True)
class SelectedTakeAlignment:
    repetition_group_id: str
    unit_id: str
    take: TakeCandidate
    result: AlignmentResult


class PairingBackend(Protocol):
    def build_request(
        self,
        *,
        audio_path: Path,
        audio_sha256: str,
        spoken_text: str,
        segmentation_artifact_sha256: str,
        config_sha256: str,
    ) -> AlignmentRequest: ...

    def align(self, request: AlignmentRequest) -> AlignmentResult: ...


CandidateExtractor = Callable[..., DerivedAudio]

_DIGEST_LENGTH = 64


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain_json(item) for item in value]
    return value


def _canonical_digest(raw: object) -> str:
    canonical = json.dumps(
        _plain_json(raw), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _alignment_runtime_sha256(request: AlignmentRequest) -> str:
    return _canonical_digest(
        {
            "backend": request.backend,
            "backend_version": request.backend_version,
            "model_id": request.model_id,
            "model_revision": request.model_revision,
            "model_license": request.model_license,
            "effective_parameters": request.effective_parameters,
            "alignment_transform_version": request.alignment_transform_version,
        }
    )


def _require_exact(raw: object, expected: set[str], name: str) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != expected:
        raise ValueError(f"{name} must contain exact fields")
    return raw


def _window_to_dict(window: TextUnitWindow) -> dict[str, Any]:
    return {
        "unit_id": window.unit_id,
        "text": window.text,
        "start_sample": window.start_sample,
        "end_sample": window.end_sample,
        "short_pause_ranges": [list(item) for item in window.short_pause_ranges],
        "token_start_index": window.token_start_index,
        "token_end_index": window.token_end_index,
    }


def _window_from_dict(raw: object) -> TextUnitWindow:
    value = _require_exact(
        raw,
        {
            "unit_id",
            "text",
            "start_sample",
            "end_sample",
            "short_pause_ranges",
            "token_start_index",
            "token_end_index",
        },
        "pairing window",
    )
    pauses = value["short_pause_ranges"]
    if type(pauses) is not list or any(type(item) is not list or len(item) != 2 for item in pauses):
        raise TypeError("pairing short_pause_ranges must be arrays of pairs")
    return TextUnitWindow(
        value["unit_id"],
        value["text"],
        value["start_sample"],
        value["end_sample"],
        tuple((item[0], item[1]) for item in pauses),
        value["token_start_index"],
        value["token_end_index"],
    )


def _evidence_to_dict(evidence: SplitEvidence) -> dict[str, Any]:
    return {
        "split_sample": evidence.split_sample,
        "first_alignment_score": evidence.first_alignment_score,
        "second_alignment_score": evidence.second_alignment_score,
        "word_order_same": evidence.word_order_same,
        "coverage": evidence.coverage,
        "first_audio": evidence.first_audio.to_dict() if evidence.first_audio is not None else None,
        "second_audio": (
            evidence.second_audio.to_dict() if evidence.second_audio is not None else None
        ),
        "first_alignment_cache_key": evidence.first_alignment_cache_key,
        "second_alignment_cache_key": evidence.second_alignment_cache_key,
        "first_result_integrity_sha256": evidence.first_result_integrity_sha256,
        "second_result_integrity_sha256": evidence.second_result_integrity_sha256,
    }


def _evidence_from_dict(raw: object) -> SplitEvidence:
    value = _require_exact(
        raw,
        {
            "split_sample",
            "first_alignment_score",
            "second_alignment_score",
            "word_order_same",
            "coverage",
            "first_audio",
            "second_audio",
            "first_alignment_cache_key",
            "second_alignment_cache_key",
            "first_result_integrity_sha256",
            "second_result_integrity_sha256",
        },
        "split evidence",
    )
    first_audio_raw = value["first_audio"]
    second_audio_raw = value["second_audio"]
    if type(first_audio_raw) is not dict or type(second_audio_raw) is not dict:
        raise TypeError("split evidence audio values must be objects")
    return SplitEvidence(
        split_sample=value["split_sample"],
        first_alignment_score=value["first_alignment_score"],
        second_alignment_score=value["second_alignment_score"],
        word_order_same=value["word_order_same"],
        coverage=value["coverage"],
        first_audio=DerivedAudio.from_dict(first_audio_raw),
        second_audio=DerivedAudio.from_dict(second_audio_raw),
        first_alignment_cache_key=value["first_alignment_cache_key"],
        second_alignment_cache_key=value["second_alignment_cache_key"],
        first_result_integrity_sha256=value["first_result_integrity_sha256"],
        second_result_integrity_sha256=value["second_result_integrity_sha256"],
    )


def _take_to_dict(take: TakeCandidate) -> dict[str, Any]:
    return {
        "take_index": take.take_index,
        "start_sample": take.start_sample,
        "end_sample": take.end_sample,
        "audio_relative_path": take.audio_relative_path,
        "audio_sha256": take.audio_sha256,
        "alignment_cache_key": take.alignment_cache_key,
        "alignment_integrity_sha256": take.alignment_integrity_sha256,
        "alignment_score": take.alignment_score,
        "coverage": take.coverage,
    }


def _take_from_dict(raw: object) -> TakeCandidate:
    value = _require_exact(
        raw,
        {
            "take_index",
            "start_sample",
            "end_sample",
            "audio_relative_path",
            "audio_sha256",
            "alignment_cache_key",
            "alignment_integrity_sha256",
            "alignment_score",
            "coverage",
        },
        "take candidate",
    )
    return TakeCandidate(**value)


def _group_to_dict(group: RepetitionGroup) -> dict[str, Any]:
    return {
        "repetition_group_id": group.repetition_group_id,
        "recording_id": group.recording_id,
        "unit_id": group.unit_id,
        "text": group.text,
        "takes": [_take_to_dict(take) for take in group.takes],
        "selected_evidence": _evidence_to_dict(group.selected_evidence),
    }


def _group_from_dict(raw: object) -> RepetitionGroup:
    value = _require_exact(
        raw,
        {
            "repetition_group_id",
            "recording_id",
            "unit_id",
            "text",
            "takes",
            "selected_evidence",
        },
        "repetition group",
    )
    takes_raw = value["takes"]
    if type(takes_raw) is not list or len(takes_raw) != 2:
        raise ValueError("repetition group must contain exactly two takes")
    takes = tuple(_take_from_dict(take) for take in takes_raw)
    return RepetitionGroup(
        value["repetition_group_id"],
        value["recording_id"],
        value["unit_id"],
        value["text"],
        (takes[0], takes[1]),
        _evidence_from_dict(value["selected_evidence"]),
    )


def _pairing_payload(result: PairingRecording) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "recording_id": result.recording_id,
        "config_sha256": result.config_sha256,
        "segmentation_artifact_sha256": result.segmentation_artifact_sha256,
        "analysis_audio_relative_path": result.analysis_audio_relative_path,
        "analysis_audio_sha256": result.analysis_audio_sha256,
        "ffmpeg_version": result.ffmpeg_version,
        "alignment_runtime_sha256s": list(result.alignment_runtime_sha256s),
        "cache_key": result.cache_key,
        "pairing_parameters": result.pairing_parameters.to_dict(),
        "windows": [_window_to_dict(window) for window in result.windows],
        "groups": [
            {
                "unit_id": outcome.unit_id,
                "status": outcome.status,
                "issue_code": outcome.issue_code,
                "candidates": [_evidence_to_dict(item) for item in outcome.candidates],
                "group": _group_to_dict(outcome.group) if outcome.group is not None else None,
            }
            for outcome in result.groups
        ],
    }


def _pairing_integrity(result: PairingRecording) -> str:
    return _canonical_digest(_pairing_payload(result))


def pairing_to_dict(result: PairingRecording) -> dict[str, Any]:
    if type(result) is not PairingRecording:
        raise TypeError("result must be a PairingRecording")
    if result.integrity_sha256 != _pairing_integrity(result):
        raise ValueError("pairing artifact integrity digest does not match content")
    return _pairing_payload(result) | {"integrity_sha256": result.integrity_sha256}


def pairing_from_dict(raw: dict[str, Any]) -> PairingRecording:
    value = _require_exact(
        raw,
        {
            "schema_version",
            "recording_id",
            "config_sha256",
            "segmentation_artifact_sha256",
            "analysis_audio_relative_path",
            "analysis_audio_sha256",
            "ffmpeg_version",
            "alignment_runtime_sha256s",
            "cache_key",
            "pairing_parameters",
            "windows",
            "groups",
            "integrity_sha256",
        },
        "pairing recording",
    )
    if value["schema_version"] != "1":
        raise ValueError("pairing schema_version must be '1'")
    parameters_raw = _require_exact(
        value["pairing_parameters"],
        {"minimum_duration_ratio", "maximum_duration_ratio", "minimum_score_margin"},
        "pairing parameters",
    )
    parameters = PairingParameters(**parameters_raw)
    windows_raw = value["windows"]
    groups_raw = value["groups"]
    runtimes_raw = value["alignment_runtime_sha256s"]
    if (
        type(windows_raw) is not list
        or type(groups_raw) is not list
        or type(runtimes_raw) is not list
    ):
        raise TypeError("pairing windows and groups must be arrays")
    windows = tuple(_window_from_dict(window) for window in windows_raw)
    groups: list[PairingGroupOutcome] = []
    for group_raw in groups_raw:
        group_value = _require_exact(
            group_raw,
            {"unit_id", "status", "issue_code", "candidates", "group"},
            "pairing group outcome",
        )
        candidates_raw = group_value["candidates"]
        if type(candidates_raw) is not list:
            raise TypeError("pairing candidates must be an array")
        selected_raw = group_value["group"]
        selected = None if selected_raw is None else _group_from_dict(selected_raw)
        status = group_value["status"]
        issue_code = group_value["issue_code"]
        if status not in ("selected", "review") or type(status) is not str:
            raise ValueError("pairing outcome status is invalid")
        status_value: Literal["selected", "review"] = (
            "selected" if status == "selected" else "review"
        )
        if (status == "selected") != (selected is not None) or (status == "review") != (
            type(issue_code) is str
        ):
            raise ValueError("pairing outcome status, issue, and group disagree")
        groups.append(
            PairingGroupOutcome(
                group_value["unit_id"],
                status_value,
                issue_code,
                tuple(_evidence_from_dict(item) for item in candidates_raw),
                selected,
            )
        )
    supplied_integrity = value["integrity_sha256"]
    result = PairingRecording(
        value["schema_version"],
        value["recording_id"],
        value["config_sha256"],
        value["segmentation_artifact_sha256"],
        value["analysis_audio_relative_path"],
        value["analysis_audio_sha256"],
        value["ffmpeg_version"],
        tuple(runtimes_raw),
        value["cache_key"],
        parameters,
        windows,
        tuple(groups),
    )
    if len(result.windows) != len(result.groups) or any(
        window.unit_id != group.unit_id
        for window, group in zip(result.windows, result.groups, strict=True)
    ):
        raise ValueError("pairing groups must exactly cover windows in order")
    if not all(
        _is_digest(value)
        for value in (
            result.config_sha256,
            result.segmentation_artifact_sha256,
            result.analysis_audio_sha256,
            result.cache_key,
        )
    ):
        raise ValueError("pairing provenance must contain lowercase SHA-256 digests")
    expected_cache_key = _pairing_cache_key(
        result.recording_id,
        result.windows,
        result.analysis_audio_relative_path,
        result.analysis_audio_sha256,
        result.segmentation_artifact_sha256,
        result.config_sha256,
        result.pairing_parameters,
        result.ffmpeg_version,
    )
    if result.cache_key != expected_cache_key:
        raise ValueError("pairing cache identity does not match provenance")
    issue_codes = {
        "TAKE_COUNT_MISMATCH",
        "TAKE_DURATION_MISMATCH",
        "TAKE_TEXT_MISMATCH",
    }
    for window, outcome in zip(result.windows, result.groups, strict=True):
        expected_splits = {start + (end - start) // 2 for start, end in window.short_pause_ranges}
        candidate_splits = tuple(item.split_sample for item in outcome.candidates)
        missing_candidate_review = (
            outcome.status == "review"
            and outcome.issue_code == "TAKE_COUNT_MISMATCH"
            and not expected_splits
            and not candidate_splits
        )
        if not missing_candidate_review and (
            not outcome.candidates
            or len(set(candidate_splits)) != len(candidate_splits)
            or set(candidate_splits) != expected_splits
        ):
            raise ValueError("pairing candidates must exactly cover short-pause midpoints")
        if outcome.status == "review":
            if outcome.issue_code not in issue_codes:
                raise ValueError("pairing review issue code is invalid")
            continue
        group = outcome.group
        assert group is not None
        expected_group_id = (
            window.unit_id
            if window.unit_id.startswith(f"{result.recording_id}-")
            else f"{result.recording_id}-{window.unit_id}"
        )
        if (
            group.repetition_group_id != expected_group_id
            or group.recording_id != result.recording_id
            or group.unit_id != window.unit_id
            or group.text != window.text
        ):
            raise ValueError("selected repetition group identity does not match its window")
        if group.selected_evidence not in outcome.candidates:
            raise ValueError("selected evidence must be one of the preserved candidates")
        evidence = group.selected_evidence
        assert evidence.first_audio is not None
        assert evidence.second_audio is not None
        assert evidence.first_alignment_cache_key is not None
        assert evidence.second_alignment_cache_key is not None
        assert evidence.first_result_integrity_sha256 is not None
        assert evidence.second_result_integrity_sha256 is not None
        expected_takes = (
            TakeCandidate(
                1,
                window.start_sample,
                evidence.split_sample,
                evidence.first_audio.relative_path,
                evidence.first_audio.sha256,
                evidence.first_alignment_cache_key,
                evidence.first_result_integrity_sha256,
                evidence.first_alignment_score,
                evidence.coverage,
            ),
            TakeCandidate(
                2,
                evidence.split_sample,
                window.end_sample,
                evidence.second_audio.relative_path,
                evidence.second_audio.sha256,
                evidence.second_alignment_cache_key,
                evidence.second_result_integrity_sha256,
                evidence.second_alignment_score,
                evidence.coverage,
            ),
        )
        if group.takes != expected_takes:
            raise ValueError("selected take evidence does not match the chosen split")
    if supplied_integrity != result.integrity_sha256:
        raise ValueError("pairing artifact integrity digest does not match content")
    return result


def _validate_spoken_units(units: tuple[SpokenUnit, ...]) -> None:
    if type(units) is not tuple or not units:
        raise CorpusFailure("TRANSCRIPT_SPOKEN_MISMATCH", "spoken units must not be empty")
    token_cursor = 0
    unit_ids: set[str] = set()
    for ordinal, unit in enumerate(units, 1):
        if type(unit) is not SpokenUnit:
            raise TypeError("units must contain SpokenUnit values")
        if (
            unit.ordinal != ordinal
            or unit.token_start_index != token_cursor
            or unit.token_end_index <= unit.token_start_index
        ):
            raise CorpusFailure(
                "TRANSCRIPT_SPOKEN_MISMATCH",
                "spoken unit token ranges must be contiguous, complete, and ordered",
            )
        _require_safe_component(unit.unit_id, "unit_id")
        if unit.unit_id in unit_ids:
            raise CorpusFailure("TRANSCRIPT_SPOKEN_MISMATCH", "spoken unit IDs must be unique")
        unit_ids.add(unit.unit_id)
        token_cursor = unit.token_end_index


def _validate_pause_evidence(
    speech: tuple[SpeechInterval, ...], pause_analysis: PauseAnalysis
) -> None:
    if type(speech) is not tuple or not speech:
        raise CorpusFailure("TRANSCRIPT_SPOKEN_MISMATCH", "speech intervals must not be empty")
    if any(type(interval) is not SpeechInterval for interval in speech):
        raise TypeError("speech must contain SpeechInterval values")
    if any(left.end_sample > right.start_sample for left, right in pairwise(speech)):
        raise ValueError("speech intervals must be ordered and non-overlapping")
    if type(pause_analysis) is not PauseAnalysis:
        raise TypeError("pause_analysis must be a PauseAnalysis")
    expected_gaps = {(left.end_sample, right.start_sample) for left, right in pairwise(speech)}
    previous_end = speech[0].start_sample
    for pause in pause_analysis.pauses:
        if type(pause) is not PauseInterval:
            raise TypeError("pause analysis must contain PauseInterval values")
        if (
            pause.start_sample,
            pause.end_sample,
        ) not in expected_gaps or pause.start_sample < previous_end:
            raise ValueError("pause intervals must be ordered VAD speech gaps")
        previous_end = pause.end_sample


def map_text_units(
    units: tuple[SpokenUnit, ...],
    speech: tuple[SpeechInterval, ...],
    pause_analysis: PauseAnalysis,
    *,
    preserve_missing_take_candidates: bool = False,
) -> tuple[TextUnitWindow, ...]:
    """Map ordered spoken units to windows separated only by long-pause midpoints."""
    _validate_spoken_units(units)
    _validate_pause_evidence(speech, pause_analysis)
    long_pauses = tuple(pause for pause in pause_analysis.pauses if pause.kind == "long")
    if len(long_pauses) != len(units) - 1:
        raise CorpusFailure(
            "TRANSCRIPT_SPOKEN_MISMATCH",
            "long-pause windows do not match spoken unit count",
        )
    boundaries = (
        speech[0].start_sample,
        *(
            pause.start_sample + (pause.end_sample - pause.start_sample) // 2
            for pause in long_pauses
        ),
        speech[-1].end_sample,
    )
    windows: list[TextUnitWindow] = []
    for unit, start, end in zip(units, boundaries[:-1], boundaries[1:], strict=True):
        short_pauses = tuple(
            (pause.start_sample, pause.end_sample)
            for pause in pause_analysis.pauses
            if pause.kind == "short" and start <= pause.start_sample and pause.end_sample <= end
        )
        if not short_pauses and not preserve_missing_take_candidates:
            raise CorpusFailure(
                "TAKE_COUNT_MISMATCH", f"no short-pause split candidate for {unit.unit_id}"
            )
        windows.append(
            TextUnitWindow(
                unit.unit_id,
                unit.text,
                start,
                end,
                short_pauses,
                unit.token_start_index,
                unit.token_end_index,
            )
        )
    return tuple(windows)


def choose_split(
    recording_id: str,
    unit: TextUnitWindow,
    evidence: tuple[SplitEvidence, ...],
    *,
    sample_rate: int,
    minimum_duration_ratio: float,
    maximum_duration_ratio: float,
    minimum_score_margin: float,
) -> RepetitionGroup:
    if type(recording_id) is not str or not recording_id.strip():
        raise ValueError("recording_id must be a non-empty string")
    if type(unit) is not TextUnitWindow:
        raise TypeError("unit must be a TextUnitWindow")
    if type(evidence) is not tuple or any(type(item) is not SplitEvidence for item in evidence):
        raise TypeError("evidence must be a tuple of SplitEvidence values")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    for value, field in (
        (minimum_duration_ratio, "minimum_duration_ratio"),
        (maximum_duration_ratio, "maximum_duration_ratio"),
        (minimum_score_margin, "minimum_score_margin"),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and non-negative")
    if minimum_duration_ratio > maximum_duration_ratio:
        raise ValueError("duration ratio bounds are reversed")
    expected_splits = {start + (end - start) // 2 for start, end in unit.short_pause_ranges}
    if any(item.split_sample not in expected_splits for item in evidence):
        raise ValueError("split evidence must use a short-pause midpoint")
    if not evidence:
        raise CorpusFailure("TAKE_COUNT_MISMATCH", f"no split candidates for {unit.unit_id}")
    valid = tuple(item for item in evidence if item.word_order_same)
    if not valid:
        raise CorpusFailure("TAKE_TEXT_MISMATCH", f"no matching split for {unit.unit_id}")
    ordered = sorted(valid, key=lambda item: (item.rank, -item.split_sample), reverse=True)
    if len(ordered) > 1 and (
        ordered[0].rank == ordered[1].rank
        or ordered[0].rank[0] - ordered[1].rank[0] < minimum_score_margin
    ):
        raise CorpusFailure("TAKE_COUNT_MISMATCH", f"ambiguous split for {unit.unit_id}")
    selected = ordered[0]
    first_duration = selected.split_sample - unit.start_sample
    second_duration = unit.end_sample - selected.split_sample
    ratio = first_duration / second_duration
    if not minimum_duration_ratio <= ratio <= maximum_duration_ratio:
        raise CorpusFailure("TAKE_DURATION_MISMATCH", f"take duration ratio {ratio:.3f}")
    takes = (
        TakeCandidate(1, unit.start_sample, selected.split_sample),
        TakeCandidate(2, selected.split_sample, unit.end_sample),
    )
    repetition_group_id = (
        unit.unit_id
        if unit.unit_id.startswith(f"{recording_id}-")
        else f"{recording_id}-{unit.unit_id}"
    )
    return RepetitionGroup(
        repetition_group_id,
        recording_id,
        unit.unit_id,
        unit.text,
        takes,
        selected,
    )


def _pairing_cache_key(
    recording_id: str,
    windows: tuple[TextUnitWindow, ...],
    analysis_audio_relative_path: str,
    analysis_audio_sha256: str,
    segmentation_artifact_sha256: str,
    config_sha256: str,
    parameters: PairingParameters,
    ffmpeg_version: str,
) -> str:
    return _canonical_digest(
        {
            "schema_version": "1",
            "recording_id": recording_id,
            "windows": [_window_to_dict(window) for window in windows],
            "analysis_audio_relative_path": analysis_audio_relative_path,
            "analysis_audio_sha256": analysis_audio_sha256,
            "segmentation_artifact_sha256": segmentation_artifact_sha256,
            "config_sha256": config_sha256,
            "pairing_parameters": parameters.to_dict(),
            "ffmpeg_version": ffmpeg_version,
        }
    )


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _DIGEST_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def pairing_run_directory(paths: CorpusPaths, config_sha256: str, recording_id: str) -> Path:
    """Return a canonical, alias-free run directory anchored in the project root."""
    if type(paths) is not CorpusPaths:
        raise TypeError("paths must be CorpusPaths")
    if not _is_digest(config_sha256):
        raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
    _require_safe_component(recording_id, "recording_id")
    project = Path(os.path.abspath(paths.project_root))
    local = project / "local-data"
    derived_parent = local / "derived"
    derived = derived_parent / "corpus-v1"
    alignments = derived / "alignments"
    runs = alignments / "runs"
    config_root = runs / config_sha256
    recording_root = config_root / recording_id
    expected_fields = (
        (paths.project_root, project),
        (paths.local_data, local),
        (paths.raw_spoken, local / "raw" / "spoken"),
        (paths.raw_sung, local / "raw" / "sung"),
        (paths.normalized, derived / "normalized"),
        (paths.segments, derived / "segments"),
        (paths.alignments, alignments),
        (paths.manifests, local / "manifests"),
    )
    if any(actual != expected for actual, expected in expected_fields):
        raise ValueError("fixed corpus root fields do not match canonical project layout")
    fixed_chain = (
        project,
        local,
        derived_parent,
        derived,
        alignments,
        runs,
        config_root,
        recording_root,
    )
    if any(component.resolve() != component for component in fixed_chain):
        raise ValueError("pairing run root ancestor is an alias")
    return recording_root


def _validate_pairing_identity(
    result: PairingRecording,
    *,
    recording_id: str,
    windows: tuple[TextUnitWindow, ...],
    analysis_audio: DerivedAudio,
    segmentation_artifact_sha256: str,
    config_sha256: str,
    parameters: PairingParameters,
    ffmpeg_version: str,
) -> None:
    expected_key = _pairing_cache_key(
        recording_id,
        windows,
        analysis_audio.relative_path,
        analysis_audio.sha256,
        segmentation_artifact_sha256,
        config_sha256,
        parameters,
        ffmpeg_version,
    )
    if (
        result.recording_id != recording_id
        or result.windows != windows
        or result.analysis_audio_relative_path != analysis_audio.relative_path
        or result.analysis_audio_sha256 != analysis_audio.sha256
        or result.segmentation_artifact_sha256 != segmentation_artifact_sha256
        or result.config_sha256 != config_sha256
        or result.pairing_parameters != parameters
        or result.ffmpeg_version != ffmpeg_version
        or result.cache_key != expected_key
    ):
        raise ValueError("pairing cache identity or provenance does not match")


def _validate_pcm16_file(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getnchannels() != 1
                or handle.getsampwidth() != 2
                or handle.getframerate() != 16_000
                or handle.getcomptype() != "NONE"
            ):
                raise ValueError("candidate audio must be 16 kHz mono PCM16 WAV")
            return handle.getnframes()
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError("candidate audio must be a valid PCM WAV") from error


def _validate_analysis_reference(relative_path: str, sha256: str, paths: CorpusPaths) -> Path:
    path = paths.local_data / Path(*relative_path.split("/"))
    if (
        path.parent != paths.normalized
        or path.resolve() != path
        or not path.is_file()
        or path.is_symlink()
    ):
        raise ValueError("pairing analysis source path is missing, aliased, or non-canonical")
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError("pairing analysis source hash does not match current file")
    _validate_pcm16_file(path)
    return path


def _validate_candidate_file(
    audio: DerivedAudio,
    paths: CorpusPaths,
    *,
    analysis_relative_path: str,
    analysis_sha256: str,
    start_sample: int,
    end_sample: int,
) -> Path:
    if audio.mode != "candidate":
        raise ValueError("pairing evidence must reference candidate analysis WAVs")
    if (
        audio.source_relative_path != analysis_relative_path
        or audio.source_sha256 != analysis_sha256
    ):
        raise ValueError("candidate source does not match current analysis audio")
    if audio.output_config_sha256 != _CANDIDATE_OUTPUT_CONFIG_SHA256:
        raise ValueError("candidate output format is not 16 kHz mono PCM16")
    if audio.start_seconds != start_sample / 16_000 or audio.end_seconds != end_sample / 16_000:
        raise ValueError("candidate boundary does not match window split samples")
    if audio.metrics is None or abs(audio.metrics.sample_count - (end_sample - start_sample)) > 1:
        raise ValueError("candidate metrics sample_count does not match sample range")
    path = paths.local_data / Path(*audio.relative_path.split("/"))
    if path.resolve() != path or not path.is_file() or path.is_symlink():
        raise ValueError("pairing candidate audio is missing or aliased")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != audio.sha256:
        raise ValueError("pairing candidate audio hash does not match")
    actual_sample_count = _validate_pcm16_file(path)
    if (
        actual_sample_count != audio.metrics.sample_count
        or abs(actual_sample_count - (end_sample - start_sample)) > 1
    ):
        raise ValueError("candidate file sample_count does not match pairing evidence")
    return path


def _validate_request_audio(request: AlignmentRequest, path: Path, audio: DerivedAudio) -> None:
    if request.audio_path != path or request.audio_sha256 != audio.sha256:
        raise ValueError("alignment request audio path/hash does not match candidate")


def _result_matches_request_provenance(result: AlignmentResult, request: AlignmentRequest) -> None:
    for field in (
        "backend",
        "backend_version",
        "model_id",
        "model_revision",
        "model_license",
        "alignment_transform_version",
    ):
        if getattr(result, field) != getattr(request, field):
            raise ValueError(f"alignment result {field} does not match request provenance")


def _candidate_word_order_same(
    unit: TextUnitWindow,
    result: AlignmentResult,
    request: AlignmentRequest,
    audio_duration_seconds: float,
) -> bool:
    token_count = len(request.alignment_text.split())
    if unit.token_end_index and unit.token_end_index - unit.token_start_index != token_count:
        raise CorpusFailure(
            "TRANSCRIPT_SPOKEN_MISMATCH",
            f"unit token range does not cover text for {unit.unit_id}",
        )
    try:
        validate_alignment(
            result,
            request,
            audio_duration_seconds=audio_duration_seconds,
        )
    except ValueError:
        return False
    return True


def _alignment_cache_path(run_directory: Path, cache_key: str) -> Path:
    return run_directory / "alignment-cache" / f"{cache_key}.json"


def _read_cached_alignment(
    path: Path,
    request: AlignmentRequest,
    *,
    audio_duration_seconds: float,
    require_word_order: bool,
) -> AlignmentResult:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError("alignment cache must contain exactly one result")
    try:
        result = result_from_dict(rows[0])
    except (KeyError, TypeError) as error:
        raise ValueError(f"alignment cache result is invalid: {error}") from error
    _result_matches_request_provenance(result, request)
    if require_word_order:
        validate_alignment(result, request, audio_duration_seconds=audio_duration_seconds)
    return result


def _load_or_align(
    run_directory: Path,
    backend: PairingBackend,
    request: AlignmentRequest,
    *,
    audio_duration_seconds: float,
) -> AlignmentResult:
    path = _alignment_cache_path(run_directory, request.cache_key)
    if path.exists():
        try:
            return _read_cached_alignment(
                path,
                request,
                audio_duration_seconds=audio_duration_seconds,
                require_word_order=False,
            )
        except (OSError, ValueError) as error:
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "alignment cache is invalid") from error
    result = backend.align(request)
    if type(result) is not AlignmentResult:
        raise TypeError("pairing backend must return AlignmentResult")
    _result_matches_request_provenance(result, request)
    write_jsonl_atomic(path, (result_to_dict(result),))
    return result


@contextmanager
def _pairing_lock(run_directory: Path) -> Iterator[None]:
    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = run_directory / ".pairing.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", "pairing run is already locked") from error
    os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _enrich_group(group: RepetitionGroup) -> RepetitionGroup:
    evidence = group.selected_evidence
    if (
        evidence.first_audio is None
        or evidence.second_audio is None
        or evidence.first_alignment_cache_key is None
        or evidence.second_alignment_cache_key is None
        or evidence.first_result_integrity_sha256 is None
        or evidence.second_result_integrity_sha256 is None
    ):
        raise ValueError("selected split lacks take quality evidence")
    takes = (
        TakeCandidate(
            1,
            group.takes[0].start_sample,
            group.takes[0].end_sample,
            evidence.first_audio.relative_path,
            evidence.first_audio.sha256,
            evidence.first_alignment_cache_key,
            evidence.first_result_integrity_sha256,
            evidence.first_alignment_score,
            evidence.coverage,
        ),
        TakeCandidate(
            2,
            group.takes[1].start_sample,
            group.takes[1].end_sample,
            evidence.second_audio.relative_path,
            evidence.second_audio.sha256,
            evidence.second_alignment_cache_key,
            evidence.second_result_integrity_sha256,
            evidence.second_alignment_score,
            evidence.coverage,
        ),
    )
    return RepetitionGroup(
        group.repetition_group_id,
        group.recording_id,
        group.unit_id,
        group.text,
        takes,
        evidence,
    )


def materialize_reviewed_pairing(
    pairing: PairingRecording, selections: dict[str, int]
) -> PairingRecording:
    """Select only preserved split evidence after an explicit human correction."""
    if type(pairing) is not PairingRecording:
        raise TypeError("pairing must be a PairingRecording")
    if type(selections) is not dict or any(
        type(unit_id) is not str or not unit_id or type(split) is not int
        for unit_id, split in selections.items()
    ):
        raise TypeError("pairing selections must map unit IDs to integer split samples")
    validated = pairing_from_dict(pairing_to_dict(pairing))
    review_units = {outcome.unit_id for outcome in validated.groups if outcome.status == "review"}
    if set(selections) != review_units:
        raise ValueError("pairing selections must exactly cover review-required units")
    windows = {window.unit_id: window for window in validated.windows}
    corrected: list[PairingGroupOutcome] = []
    for outcome in validated.groups:
        if outcome.status == "selected":
            corrected.append(outcome)
            continue
        selected = tuple(
            candidate
            for candidate in outcome.candidates
            if candidate.split_sample == selections[outcome.unit_id]
        )
        if len(selected) != 1:
            raise ValueError("pairing selection must identify exactly one saved candidate")
        evidence = selected[0]
        window = windows[outcome.unit_id]
        group_id = (
            window.unit_id
            if window.unit_id.startswith(f"{validated.recording_id}-")
            else f"{validated.recording_id}-{window.unit_id}"
        )
        group = _enrich_group(
            RepetitionGroup(
                group_id,
                validated.recording_id,
                window.unit_id,
                window.text,
                (
                    TakeCandidate(1, window.start_sample, evidence.split_sample),
                    TakeCandidate(2, evidence.split_sample, window.end_sample),
                ),
                evidence,
            )
        )
        corrected.append(
            PairingGroupOutcome(outcome.unit_id, "selected", None, outcome.candidates, group)
        )
    result = PairingRecording(
        validated.schema_version,
        validated.recording_id,
        validated.config_sha256,
        validated.segmentation_artifact_sha256,
        validated.analysis_audio_relative_path,
        validated.analysis_audio_sha256,
        validated.ffmpeg_version,
        validated.alignment_runtime_sha256s,
        validated.cache_key,
        validated.pairing_parameters,
        validated.windows,
        tuple(corrected),
    )
    return pairing_from_dict(pairing_to_dict(result))


def _validate_cached_evidence(
    result: PairingRecording,
    paths: CorpusPaths,
    backend: PairingBackend,
    run_directory: Path,
    *,
    allow_reviewed_selection: bool = False,
) -> None:
    _validate_analysis_reference(
        result.analysis_audio_relative_path, result.analysis_audio_sha256, paths
    )
    runtime_sha256s: set[str] = set()
    for window, outcome in zip(result.windows, result.groups, strict=True):
        recomputed_candidates: list[SplitEvidence] = []
        for evidence in outcome.candidates:
            if (
                evidence.first_audio is None
                or evidence.second_audio is None
                or evidence.first_alignment_cache_key is None
                or evidence.second_alignment_cache_key is None
                or evidence.first_result_integrity_sha256 is None
                or evidence.second_result_integrity_sha256 is None
            ):
                raise ValueError("cached split evidence lacks provenance")
            alignments: list[AlignmentResult] = []
            requests: list[AlignmentRequest] = []
            for audio, cache_key, integrity, start_sample, end_sample in (
                (
                    evidence.first_audio,
                    evidence.first_alignment_cache_key,
                    evidence.first_result_integrity_sha256,
                    window.start_sample,
                    evidence.split_sample,
                ),
                (
                    evidence.second_audio,
                    evidence.second_alignment_cache_key,
                    evidence.second_result_integrity_sha256,
                    evidence.split_sample,
                    window.end_sample,
                ),
            ):
                if audio.ffmpeg_version != result.ffmpeg_version:
                    raise ValueError("cached candidate FFmpeg version does not match pairing")
                path = _validate_candidate_file(
                    audio,
                    paths,
                    analysis_relative_path=result.analysis_audio_relative_path,
                    analysis_sha256=result.analysis_audio_sha256,
                    start_sample=start_sample,
                    end_sample=end_sample,
                )
                request = backend.build_request(
                    audio_path=path,
                    audio_sha256=audio.sha256,
                    spoken_text=window.text,
                    segmentation_artifact_sha256=result.segmentation_artifact_sha256,
                    config_sha256=result.config_sha256,
                )
                _validate_request_audio(request, path, audio)
                runtime_sha256s.add(_alignment_runtime_sha256(request))
                if request.cache_key != cache_key or audio.metrics is None:
                    raise ValueError("cached alignment request identity does not match")
                alignment = _read_cached_alignment(
                    _alignment_cache_path(run_directory, cache_key),
                    request,
                    audio_duration_seconds=audio.metrics.sample_count / 16_000,
                    require_word_order=False,
                )
                if alignment.integrity_sha256 != integrity:
                    raise ValueError("cached alignment integrity does not match pairing evidence")
                requests.append(request)
                alignments.append(alignment)
            word_order_same = all(
                _candidate_word_order_same(
                    window,
                    alignment,
                    request,
                    audio.metrics.sample_count / 16_000,
                )
                for audio, alignment, request in zip(
                    (evidence.first_audio, evidence.second_audio),
                    alignments,
                    requests,
                    strict=True,
                )
                if audio is not None and audio.metrics is not None
            )
            recomputed = SplitEvidence(
                evidence.split_sample,
                alignments[0].mean_score,
                alignments[1].mean_score,
                word_order_same,
                min(alignments[0].coverage, alignments[1].coverage),
                evidence.first_audio,
                evidence.second_audio,
                requests[0].cache_key,
                requests[1].cache_key,
                alignments[0].integrity_sha256,
                alignments[1].integrity_sha256,
            )
            if recomputed != evidence:
                raise ValueError("cached split evidence does not match alignment results")
            recomputed_candidates.append(recomputed)

        candidates = tuple(recomputed_candidates)
        try:
            group = _enrich_group(
                choose_split(
                    result.recording_id,
                    window,
                    candidates,
                    sample_rate=16_000,
                    minimum_duration_ratio=(result.pairing_parameters.minimum_duration_ratio),
                    maximum_duration_ratio=(result.pairing_parameters.maximum_duration_ratio),
                    minimum_score_margin=result.pairing_parameters.minimum_score_margin,
                )
            )
        except CorpusFailure as error:
            if error.code not in {
                "TAKE_COUNT_MISMATCH",
                "TAKE_DURATION_MISMATCH",
                "TAKE_TEXT_MISMATCH",
            }:
                raise
            expected_outcome = PairingGroupOutcome(
                window.unit_id, "review", error.code, candidates, None
            )
        else:
            expected_outcome = PairingGroupOutcome(
                window.unit_id, "selected", None, candidates, group
            )
        human_selected_saved_candidate = (
            allow_reviewed_selection
            and outcome.status == "selected"
            and outcome.group is not None
            and outcome.candidates == candidates
            and outcome.group.selected_evidence in candidates
        )
        if outcome != expected_outcome and not human_selected_saved_candidate:
            raise ValueError("cached pairing decision does not match recomputed evidence")
    if tuple(sorted(runtime_sha256s)) != result.alignment_runtime_sha256s:
        raise ValueError("cached alignment runtime identities do not match pairing")


def pair_recording(
    recording_id: str,
    windows: tuple[TextUnitWindow, ...],
    analysis_audio: DerivedAudio,
    paths: CorpusPaths,
    backend: PairingBackend,
    *,
    segmentation_artifact_sha256: str,
    config_sha256: str,
    pairing_parameters: PairingParameters,
    ffmpeg_version: str,
    run_command: RunCommand = subprocess.run,
    extract_candidate: CandidateExtractor = extract_analysis_segment,
    allow_reviewed_selection: bool = False,
) -> PairingRecording:
    """Extract, align, score, and cache every two-take split candidate for one recording."""
    _require_safe_component(recording_id, "recording_id")
    if (
        type(windows) is not tuple
        or not windows
        or any(type(window) is not TextUnitWindow for window in windows)
    ):
        raise TypeError("windows must be a non-empty tuple of TextUnitWindow values")
    if type(analysis_audio) is not DerivedAudio or analysis_audio.mode != "analysis":
        raise TypeError("analysis_audio must be an analysis DerivedAudio")
    if type(paths) is not CorpusPaths:
        raise TypeError("paths must be CorpusPaths")
    if type(pairing_parameters) is not PairingParameters:
        raise TypeError("pairing_parameters must be PairingParameters")
    if type(ffmpeg_version) is not str or not ffmpeg_version.strip():
        raise ValueError("ffmpeg_version must be a non-empty string")
    unit_ids = tuple(window.unit_id for window in windows)
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unit_id values must be unique")
    if not _is_digest(segmentation_artifact_sha256) or not _is_digest(config_sha256):
        raise ValueError("pairing provenance hashes must be lowercase SHA-256 digests")
    if analysis_audio.config_sha256 != config_sha256:
        raise ValueError("analysis audio config provenance does not match pairing config")
    run_directory = pairing_run_directory(paths, config_sha256, recording_id)
    analysis_path = _validate_analysis_reference(
        analysis_audio.relative_path, analysis_audio.sha256, paths
    )
    if (
        analysis_audio.metrics is None
        or _validate_pcm16_file(analysis_path) != analysis_audio.metrics.sample_count
    ):
        raise ValueError("analysis audio metrics do not match current PCM file")
    pairing_path = run_directory / "pairing.json"
    if pairing_path.exists():
        try:
            rows = read_jsonl(pairing_path)
            if len(rows) != 1:
                raise ValueError("pairing cache must contain exactly one recording")
            try:
                cached = pairing_from_dict(rows[0])
            except (KeyError, TypeError) as error:
                raise ValueError("pairing cache payload is invalid") from error
            _validate_pairing_identity(
                cached,
                recording_id=recording_id,
                windows=windows,
                analysis_audio=analysis_audio,
                segmentation_artifact_sha256=segmentation_artifact_sha256,
                config_sha256=config_sha256,
                parameters=pairing_parameters,
                ffmpeg_version=ffmpeg_version,
            )
            _validate_cached_evidence(
                cached,
                paths,
                backend,
                run_directory,
                allow_reviewed_selection=allow_reviewed_selection,
            )
            return cached
        except (OSError, ValueError) as error:
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "pairing cache is invalid") from error

    with _pairing_lock(run_directory):
        outcomes: list[PairingGroupOutcome] = []
        runtime_sha256s: set[str] = set()
        for window in windows:
            evidence_items: list[SplitEvidence] = []
            for pause_start, pause_end in window.short_pause_ranges:
                split_sample = pause_start + (pause_end - pause_start) // 2
                audio_items: list[DerivedAudio] = []
                results: list[AlignmentResult] = []
                requests: list[AlignmentRequest] = []
                for start_sample, end_sample in (
                    (window.start_sample, split_sample),
                    (split_sample, window.end_sample),
                ):
                    audio = extract_candidate(
                        analysis_audio,
                        paths,
                        start_sample / 16_000,
                        end_sample / 16_000,
                        ffmpeg_version=ffmpeg_version,
                        run_command=run_command,
                    )
                    if audio.ffmpeg_version != ffmpeg_version:
                        raise ValueError("candidate audio FFmpeg version does not match pairing")
                    path = _validate_candidate_file(
                        audio,
                        paths,
                        analysis_relative_path=analysis_audio.relative_path,
                        analysis_sha256=analysis_audio.sha256,
                        start_sample=start_sample,
                        end_sample=end_sample,
                    )
                    request = backend.build_request(
                        audio_path=path,
                        audio_sha256=audio.sha256,
                        spoken_text=window.text,
                        segmentation_artifact_sha256=segmentation_artifact_sha256,
                        config_sha256=config_sha256,
                    )
                    _validate_request_audio(request, path, audio)
                    runtime_sha256s.add(_alignment_runtime_sha256(request))
                    if audio.metrics is None:
                        raise ValueError("candidate analysis audio must contain PCM metrics")
                    result = _load_or_align(
                        run_directory,
                        backend,
                        request,
                        audio_duration_seconds=audio.metrics.sample_count / 16_000,
                    )
                    audio_items.append(audio)
                    requests.append(request)
                    results.append(result)
                order_same = all(
                    _candidate_word_order_same(
                        window,
                        result,
                        request,
                        audio.metrics.sample_count / 16_000,
                    )
                    for audio, result, request in zip(audio_items, results, requests, strict=True)
                    if audio.metrics is not None
                )
                evidence_items.append(
                    SplitEvidence(
                        split_sample,
                        results[0].mean_score,
                        results[1].mean_score,
                        order_same,
                        min(results[0].coverage, results[1].coverage),
                        audio_items[0],
                        audio_items[1],
                        requests[0].cache_key,
                        requests[1].cache_key,
                        results[0].integrity_sha256,
                        results[1].integrity_sha256,
                    )
                )
            candidates = tuple(evidence_items)
            try:
                group = _enrich_group(
                    choose_split(
                        recording_id,
                        window,
                        candidates,
                        sample_rate=16_000,
                        minimum_duration_ratio=pairing_parameters.minimum_duration_ratio,
                        maximum_duration_ratio=pairing_parameters.maximum_duration_ratio,
                        minimum_score_margin=pairing_parameters.minimum_score_margin,
                    )
                )
            except CorpusFailure as error:
                if error.code not in {
                    "TAKE_COUNT_MISMATCH",
                    "TAKE_DURATION_MISMATCH",
                    "TAKE_TEXT_MISMATCH",
                }:
                    raise
                outcomes.append(
                    PairingGroupOutcome(window.unit_id, "review", error.code, candidates, None)
                )
            else:
                outcomes.append(
                    PairingGroupOutcome(window.unit_id, "selected", None, candidates, group)
                )
        pairing_result = PairingRecording(
            "1",
            recording_id,
            config_sha256,
            segmentation_artifact_sha256,
            analysis_audio.relative_path,
            analysis_audio.sha256,
            ffmpeg_version,
            tuple(sorted(runtime_sha256s)),
            _pairing_cache_key(
                recording_id,
                windows,
                analysis_audio.relative_path,
                analysis_audio.sha256,
                segmentation_artifact_sha256,
                config_sha256,
                pairing_parameters,
                ffmpeg_version,
            ),
            pairing_parameters,
            windows,
            tuple(outcomes),
        )
        write_jsonl_atomic(pairing_path, (pairing_to_dict(pairing_result),))
        return pairing_result


def load_selected_alignments(
    pairing: PairingRecording,
    paths: CorpusPaths,
    backend: PairingBackend,
    *,
    allow_reviewed_selection: bool = False,
) -> tuple[SelectedTakeAlignment, ...]:
    """Load and strictly validate selected alignment cache entries without recomputation."""
    run_directory = pairing_run_directory(paths, pairing.config_sha256, pairing.recording_id)
    _validate_cached_evidence(
        pairing,
        paths,
        backend,
        run_directory,
        allow_reviewed_selection=allow_reviewed_selection,
    )
    selected: list[SelectedTakeAlignment] = []
    for outcome in pairing.groups:
        if outcome.status == "review":
            continue
        group = outcome.group
        if group is None:
            raise ValueError("selected pairing outcome lacks a repetition group")
        evidence = group.selected_evidence
        audio_items = (evidence.first_audio, evidence.second_audio)
        for take, audio in zip(group.takes, audio_items, strict=True):
            if audio is None or audio.metrics is None:
                raise ValueError("selected take lacks candidate audio evidence")
            audio_path = _validate_candidate_file(
                audio,
                paths,
                analysis_relative_path=pairing.analysis_audio_relative_path,
                analysis_sha256=pairing.analysis_audio_sha256,
                start_sample=take.start_sample,
                end_sample=take.end_sample,
            )
            request = backend.build_request(
                audio_path=audio_path,
                audio_sha256=audio.sha256,
                spoken_text=group.text,
                segmentation_artifact_sha256=pairing.segmentation_artifact_sha256,
                config_sha256=pairing.config_sha256,
            )
            _validate_request_audio(request, audio_path, audio)
            if request.cache_key != take.alignment_cache_key:
                raise ValueError("selected take cache key does not match request")
            result = _read_cached_alignment(
                _alignment_cache_path(run_directory, request.cache_key),
                request,
                audio_duration_seconds=audio.metrics.sample_count / 16_000,
                require_word_order=True,
            )
            if result.integrity_sha256 != take.alignment_integrity_sha256:
                raise ValueError("selected take alignment integrity does not match")
            selected.append(
                SelectedTakeAlignment(group.repetition_group_id, group.unit_id, take, result)
            )
    return tuple(selected)
