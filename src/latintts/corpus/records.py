from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

from latintts.corpus.domain import CorpusState, require_transition

UnknownBool = bool | Literal["unknown"]


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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RightsRecord:
        require_exact_fields(raw, frozenset(field.name for field in fields(cls)), "rights row")
        boolean_fields = (
            "allow_local_processing",
            "allow_model_training",
            "allow_internal_evaluation",
        )
        if any(type(raw[name]) is not bool for name in boolean_fields):
            raise TypeError("rights authorization fields must be booleans")
        release_fields = ("allow_raw_release", "allow_segment_release", "allow_model_release")
        if any(raw[name] not in (True, False, "unknown") for name in release_fields):
            raise TypeError("rights release fields must be booleans or unknown")
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


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str
    bit_rate: int | None


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
    result: str

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
    result: str,
) -> tuple[RecordingRecord, ProcessingEvent]:
    require_transition(record.state, target)
    identity = {
        "recording_id": record.recording_id,
        "previous_state": record.state.value,
        "target_state": target.value,
        "inputs": input_sha256s,
        "config": config_sha256,
        "tools": tool_versions,
        "started_at": started_at,
        "finished_at": finished_at,
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
