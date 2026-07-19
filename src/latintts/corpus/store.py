from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from latintts.corpus.domain import require_transition
from latintts.corpus.records import ProcessingEvent, RecordingRecord


class RecordingTransitionPersistenceError(RuntimeError):
    """The event is durable but the recording manifest still needs recovery."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank line {line_number} in {path}")
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonstandard_constant,
            )
        except ValueError as error:
            raise ValueError(f"invalid JSON at line {line_number} in {path}: {error}") from error
        if type(raw) is not dict:
            raise ValueError(f"line {line_number} must be a JSON object")
        rows.append(raw)
    return tuple(rows)


def _encode(row: dict[str, Any]) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_encode(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl_event(path: Path, row: dict[str, Any]) -> None:
    existing = read_jsonl(path) if path.exists() else ()
    write_jsonl_atomic(path, (*existing, row))


def persist_recording_transition(
    *,
    recordings_path: Path,
    events_path: Path,
    recordings: Iterable[RecordingRecord],
    event: ProcessingEvent,
) -> None:
    """Persist one transition in audit-first order without claiming cross-file atomicity."""
    target_records = tuple(recordings)
    require_transition(event.previous_state, event.target_state)
    target_matches = [
        record for record in target_records if record.recording_id == event.recording_id
    ]
    if len(target_matches) != 1:
        raise ValueError("target manifest must contain exactly one event recording_id")
    if target_matches[0].state is not event.target_state:
        raise ValueError("target manifest state must equal event target_state")
    existing_matches = [
        row for row in read_jsonl(recordings_path) if row.get("recording_id") == event.recording_id
    ]
    if len(existing_matches) != 1:
        raise ValueError("existing manifest must contain exactly one event recording_id")
    if existing_matches[0].get("state") != event.previous_state.value:
        raise ValueError("existing manifest state must equal event previous_state")
    append_jsonl_event(events_path, event.to_dict())
    try:
        write_jsonl_atomic(recordings_path, (record.to_dict() for record in target_records))
    except Exception as error:
        raise RecordingTransitionPersistenceError(
            f"event {event.event_id} persisted; recordings update failed; recovery required"
        ) from error
