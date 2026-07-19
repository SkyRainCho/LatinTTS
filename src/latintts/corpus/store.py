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
    if recordings_path.resolve() == events_path.resolve():
        raise ValueError("recordings_path and events_path must be distinct resolved paths")
    target_records = tuple(recordings)
    target_rows = tuple(record.to_dict() for record in target_records)
    require_transition(event.previous_state, event.target_state)
    target_matches = [
        record for record in target_records if record.recording_id == event.recording_id
    ]
    if len(target_matches) != 1:
        raise ValueError("target manifest must contain exactly one event recording_id")
    if target_matches[0].state is not event.target_state:
        raise ValueError("target manifest state must equal event target_state")
    existing_rows = read_jsonl(recordings_path)
    existing_matches = [
        row for row in existing_rows if row.get("recording_id") == event.recording_id
    ]
    if len(existing_matches) != 1:
        raise ValueError("existing manifest must contain exactly one event recording_id")
    if existing_matches[0].get("state") != event.previous_state.value:
        raise ValueError("existing manifest state must equal event previous_state")
    target_by_id = {row["recording_id"]: row for row in target_rows}
    existing_by_id = {row.get("recording_id"): row for row in existing_rows}
    if (
        len(target_by_id) != len(target_rows)
        or len(existing_by_id) != len(existing_rows)
        or set(target_by_id) != set(existing_by_id)
    ):
        raise ValueError("target manifest recording_id set and count must remain unchanged")
    for recording_id, existing_row in existing_by_id.items():
        target_row = target_by_id[recording_id]
        if recording_id == event.recording_id:
            existing_without_state = {
                key: value for key, value in existing_row.items() if key != "state"
            }
            target_without_state = {
                key: value for key, value in target_row.items() if key != "state"
            }
            if target_without_state != existing_without_state:
                raise ValueError("event recording may change only state")
        elif target_row != existing_row:
            raise ValueError("non-event recording must remain unchanged")
    append_jsonl_event(events_path, event.to_dict())
    try:
        write_jsonl_atomic(recordings_path, target_rows)
    except Exception as error:
        raise RecordingTransitionPersistenceError(
            f"event {event.event_id} persisted; recordings update failed; recovery required"
        ) from error
