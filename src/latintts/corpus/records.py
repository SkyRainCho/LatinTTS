from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime
from typing import Any, Literal

from latintts.corpus.domain import CorpusState, require_transition

UnknownBool = bool | Literal["unknown"]
ProcessingResult = Literal["success"]
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_schema_version(value: object) -> None:
    if value != "1" or type(value) is not str:
        raise ValueError("schema_version must be '1'")


def _require_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_positive_int(value: object, field: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")


def _require_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a timezone-aware ISO datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be a timezone-aware ISO datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware ISO datetime")
    return parsed


def _require_iso_date_or_datetime(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an ISO date or timezone-aware datetime")
    try:
        if len(value) == 10:
            date.fromisoformat(value)
            return
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO date or timezone-aware datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be an ISO date or timezone-aware datetime")


def require_exact_fields(raw: dict[str, Any], fields: frozenset[str], record: str) -> None:
    if set(raw) != fields:
        raise ValueError(f"{record} must contain exact fields; got {sorted(raw)}")


@dataclass(frozen=True, slots=True)
class RightsRecord:
    rights_id: str
    owner_id: str
    speaker_id: str
    allow_local_processing: bool
    allow_model_training: bool
    allow_internal_evaluation: bool
    allow_raw_release: UnknownBool
    allow_segment_release: UnknownBool
    allow_model_release: UnknownBool
    authorized_at: str
    basis: str
    notes: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.rights_id, "rights_id")
        _require_nonempty_string(self.owner_id, "owner_id")
        _require_nonempty_string(self.speaker_id, "speaker_id")
        boolean_fields = (
            self.allow_local_processing,
            self.allow_model_training,
            self.allow_internal_evaluation,
        )
        if any(type(value) is not bool for value in boolean_fields):
            raise TypeError("rights authorization fields must be booleans")
        release_fields = (
            self.allow_raw_release,
            self.allow_segment_release,
            self.allow_model_release,
        )
        if any(not (type(value) is bool or value == "unknown") for value in release_fields):
            raise TypeError("rights release fields must be booleans or unknown")
        _require_iso_date_or_datetime(self.authorized_at, "authorized_at")
        _require_nonempty_string(self.basis, "basis")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RightsRecord:
        require_exact_fields(raw, frozenset(field.name for field in fields(cls)), "rights row")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntakeRow:
    relative_path: str
    title_or_citation: str
    speaker_id: str
    rights_id: str
    notes: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.relative_path, "relative_path")
        _require_nonempty_string(self.title_or_citation, "title_or_citation")
        _require_nonempty_string(self.speaker_id, "speaker_id")
        _require_nonempty_string(self.rights_id, "rights_id")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")


@dataclass(frozen=True, slots=True)
class SourceCandidateIntake:
    source_id: str
    source_url: str
    source_version: str
    accessed_at: str
    source_file: str
    selected: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SourceCandidateIntake:
        require_exact_fields(
            raw,
            frozenset(field.name for field in fields(cls)),
            "source candidate intake",
        )
        for name in ("source_id", "source_url", "source_version", "accessed_at", "source_file"):
            if not isinstance(raw[name], str):
                raise TypeError(f"source candidate {name} must be a string")
        if type(raw["selected"]) is not bool:
            raise TypeError("source candidate selected must be a boolean")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscriptIntakeRow:
    recording_id: str
    source_candidates: tuple[SourceCandidateIntake, ...]
    spoken_units_file: str
    pronunciation_overrides_file: str
    confirmed: bool
    asr_hypothesis_file: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TranscriptIntakeRow:
        require_exact_fields(
            raw,
            frozenset(field.name for field in fields(cls)),
            "transcript intake row",
        )
        _require_nonempty_string(raw["recording_id"], "recording_id")
        if type(raw["source_candidates"]) is not list:
            raise TypeError("source_candidates must be an array")
        candidates_list: list[SourceCandidateIntake] = []
        for candidate in raw["source_candidates"]:
            if type(candidate) is not dict:
                raise TypeError("source candidate intake must be an object")
            candidates_list.append(SourceCandidateIntake.from_dict(candidate))
        candidates = tuple(candidates_list)
        for name in (
            "spoken_units_file",
            "pronunciation_overrides_file",
            "asr_hypothesis_file",
        ):
            if not isinstance(raw[name], str):
                raise TypeError(f"{name} must be a string")
        if type(raw["confirmed"]) is not bool:
            raise TypeError("confirmed must be a boolean")
        return cls(
            recording_id=raw["recording_id"],
            source_candidates=candidates,
            spoken_units_file=raw["spoken_units_file"],
            pronunciation_overrides_file=raw["pronunciation_overrides_file"],
            confirmed=raw["confirmed"],
            asr_hypothesis_file=raw["asr_hypothesis_file"],
        )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["source_candidates"] = [candidate.to_dict() for candidate in self.source_candidates]
        return raw


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str
    bit_rate: int | None

    def __post_init__(self) -> None:
        if type(self.duration_seconds) not in (int, float):
            raise TypeError("duration_seconds must be a number")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        _require_positive_int(self.sample_rate, "sample_rate")
        _require_positive_int(self.channels, "channels")
        _require_nonempty_string(self.codec, "codec")
        if self.bit_rate is not None:
            _require_positive_int(self.bit_rate, "bit_rate")


@dataclass(frozen=True, slots=True)
class RecordingRecord:
    schema_version: str
    recording_id: str
    relative_path: str
    sha256: str
    content_type: Literal["spoken", "sung"]
    title_or_citation: str
    speaker_id: str
    rights_id: str
    notes: str
    metadata: AudioMetadata
    state: CorpusState

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty_string(self.recording_id, "recording_id")
        _require_nonempty_string(self.relative_path, "relative_path")
        _require_sha256(self.sha256, "sha256")
        if self.content_type not in ("spoken", "sung") or not isinstance(self.content_type, str):
            raise ValueError("content_type must be spoken or sung")
        _require_nonempty_string(self.title_or_citation, "title_or_citation")
        _require_nonempty_string(self.speaker_id, "speaker_id")
        _require_nonempty_string(self.rights_id, "rights_id")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")
        if type(self.metadata) is not AudioMetadata:
            raise TypeError("metadata must be AudioMetadata")
        if type(self.state) is not CorpusState:
            raise TypeError("state must be CorpusState")

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["state"] = self.state.value
        return raw


@dataclass(frozen=True, slots=True)
class ProcessingEvent:
    schema_version: str
    event_id: str
    recording_id: str
    previous_state: CorpusState
    target_state: CorpusState
    input_sha256s: tuple[str, ...]
    config_sha256: str
    tool_versions: tuple[str, ...]
    started_at: str
    finished_at: str
    result: ProcessingResult

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty_string(self.event_id, "event_id")
        _require_nonempty_string(self.recording_id, "recording_id")
        if type(self.previous_state) is not CorpusState:
            raise TypeError("previous_state must be CorpusState")
        if type(self.target_state) is not CorpusState:
            raise TypeError("target_state must be CorpusState")
        require_transition(self.previous_state, self.target_state)
        if type(self.input_sha256s) is not tuple or not self.input_sha256s:
            raise ValueError("input_sha256s must be a non-empty tuple")
        for digest in self.input_sha256s:
            _require_sha256(digest, "input_sha256s")
        _require_sha256(self.config_sha256, "config_sha256")
        if type(self.tool_versions) is not tuple or not self.tool_versions:
            raise ValueError("tool_versions must be a non-empty tuple")
        for tool_version in self.tool_versions:
            _require_nonempty_string(tool_version, "tool_versions")
        started_at = _require_timestamp(self.started_at, "started_at")
        finished_at = _require_timestamp(self.finished_at, "finished_at")
        if finished_at < started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.result != "success" or type(self.result) is not str:
            raise ValueError("result must be success")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProcessingEvent:
        require_exact_fields(
            raw,
            frozenset(field.name for field in fields(cls)),
            "processing event",
        )
        if type(raw["input_sha256s"]) is not list:
            raise TypeError("processing event input_sha256s must be an array")
        if type(raw["tool_versions"]) is not list:
            raise TypeError("processing event tool_versions must be an array")
        return cls(
            schema_version=raw["schema_version"],
            event_id=raw["event_id"],
            recording_id=raw["recording_id"],
            previous_state=CorpusState(raw["previous_state"]),
            target_state=CorpusState(raw["target_state"]),
            input_sha256s=tuple(raw["input_sha256s"]),
            config_sha256=raw["config_sha256"],
            tool_versions=tuple(raw["tool_versions"]),
            started_at=raw["started_at"],
            finished_at=raw["finished_at"],
            result=raw["result"],
        )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["previous_state"] = self.previous_state.value
        raw["target_state"] = self.target_state.value
        raw["input_sha256s"] = list(self.input_sha256s)
        raw["tool_versions"] = list(self.tool_versions)
        return raw


def advance_recording(
    record: RecordingRecord,
    target: CorpusState,
    *,
    input_sha256s: tuple[str, ...],
    config_sha256: str,
    tool_versions: tuple[str, ...],
    started_at: str,
    finished_at: str,
    result: ProcessingResult,
) -> tuple[RecordingRecord, ProcessingEvent]:
    if type(target) is not CorpusState:
        raise TypeError("target must be CorpusState")
    require_transition(record.state, target)
    identity = {
        "recording_id": record.recording_id,
        "previous_state": record.state.value,
        "target_state": target.value,
        "inputs": input_sha256s,
        "config": config_sha256,
        "tools": tool_versions,
        "result": result,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    event = ProcessingEvent(
        "1",
        f"state-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}",
        record.recording_id,
        record.state,
        target,
        input_sha256s,
        config_sha256,
        tool_versions,
        started_at,
        finished_at,
        result,
    )
    return replace(record, state=target), event


@dataclass(frozen=True, slots=True)
class ReviewEvent:
    schema_version: str
    review_event_id: str
    entity_id: str
    field: str
    before: object
    after: object
    reason: str
    reviewer: str
    reviewed_at: str

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty_string(self.review_event_id, "review_event_id")
        _require_nonempty_string(self.entity_id, "entity_id")
        _require_nonempty_string(self.field, "field")
        _require_nonempty_string(self.reason, "reason")
        _require_nonempty_string(self.reviewer, "reviewer")
        _require_timestamp(self.reviewed_at, "reviewed_at")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReviewEvent:
        require_exact_fields(raw, frozenset(field.name for field in fields(cls)), "review event")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
