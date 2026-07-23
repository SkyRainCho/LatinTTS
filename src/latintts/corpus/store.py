from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from collections.abc import Callable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

from latintts.corpus.domain import require_transition
from latintts.corpus.locking import corpus_mutation_lease
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


def read_jsonl(
    path: Path,
    *,
    directory_fd: int | None = None,
) -> tuple[dict[str, Any], ...]:
    if directory_fd is None:
        text = path.read_text(encoding="utf-8")
    else:  # pragma: no cover - exercised on POSIX hosts
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            text = handle.read()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
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


def jsonl_sha256(rows: Iterable[dict[str, Any]]) -> str:
    """Return the digest of the exact canonical bytes written by write_jsonl_atomic."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update((_encode(row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _file_sha256(path: Path, directory_fd: int | None) -> str:
    digest = hashlib.sha256()
    if directory_fd is None:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    descriptor = os.open(  # pragma: no cover - exercised on POSIX hosts
        path.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:  # pragma: no cover - exercised on POSIX hosts
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:  # pragma: no cover - exercised on POSIX hosts
        os.close(descriptor)


def write_jsonl_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    directory_fd: int | None = None,
) -> None:
    with corpus_mutation_lease(path):
        _write_jsonl_atomic_locked(path, rows, directory_fd=directory_fd)


def _write_jsonl_atomic_locked(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    directory_fd: int | None = None,
) -> None:
    if directory_fd is not None:  # pragma: no cover - exercised on POSIX hosts
        temporary_name = f".{path.name}.{secrets.token_hex(12)}"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(_encode(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)
        return
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
    with corpus_mutation_lease(path):
        existing = read_jsonl(path) if path.exists() else ()
        write_jsonl_atomic(path, (*existing, row))


def _read_jsonl_in_directory(
    path: Path,
    directory_fd: int | None,
) -> tuple[dict[str, Any], ...]:
    if directory_fd is None:
        return read_jsonl(path)
    return read_jsonl(path, directory_fd=directory_fd)  # pragma: no cover


def _write_jsonl_in_directory(
    path: Path,
    rows: Iterable[dict[str, Any]],
    directory_fd: int | None,
) -> None:
    if directory_fd is None:
        write_jsonl_atomic(path, rows)
    else:  # pragma: no cover - exercised on POSIX hosts
        write_jsonl_atomic(path, rows, directory_fd=directory_fd)


def validate_processing_events(
    rows: Iterable[dict[str, Any]],
) -> tuple[ProcessingEvent, ...]:
    """Decode and validate a complete processing journal without writing it."""
    try:
        decoded_events = tuple(ProcessingEvent.from_dict(row) for row in rows)
    except (KeyError, TypeError) as error:
        raise ValueError(f"processing event row is invalid: {error}") from error
    event_ids = [event.event_id for event in decoded_events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("processing events contain duplicate event_id")
    semantic_keys = [
        (
            event.recording_id,
            event.previous_state,
            event.target_state,
            event.input_sha256s,
            event.config_sha256,
            event.tool_versions,
            event.result,
        )
        for event in decoded_events
    ]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise ValueError("processing events contain duplicate semantic transition")
    last_target_by_recording: dict[str, object] = {}
    for event in decoded_events:
        last_target = last_target_by_recording.get(event.recording_id)
        if last_target is not None and event.previous_state is not last_target:
            raise ValueError("invalid processing event state chain")
        last_target_by_recording[event.recording_id] = event.target_state
    return decoded_events


def processing_event_exists(
    existing_events: tuple[dict[str, Any], ...],
    event: ProcessingEvent,
) -> bool:
    decoded_events = validate_processing_events(existing_events)
    semantic_keys = [
        (
            existing_event.recording_id,
            existing_event.previous_state,
            existing_event.target_state,
            existing_event.input_sha256s,
            existing_event.config_sha256,
            existing_event.tool_versions,
            existing_event.result,
        )
        for existing_event in decoded_events
    ]
    event_semantics = (
        event.recording_id,
        event.previous_state,
        event.target_state,
        event.input_sha256s,
        event.config_sha256,
        event.tool_versions,
        event.result,
    )
    matching_ids = [
        existing_event
        for existing_event in decoded_events
        if existing_event.event_id == event.event_id
    ]
    matching_transitions = [
        existing_event
        for existing_event, semantics in zip(decoded_events, semantic_keys, strict=True)
        if semantics == event_semantics
    ]
    if matching_ids and not matching_transitions:
        raise ValueError("existing processing event conflicts with transition event")
    if matching_transitions and matching_transitions[0].event_id != event.event_id:
        raise ValueError("semantic transition uses a different event_id")
    same_state_transitions = [
        existing_event
        for existing_event in decoded_events
        if existing_event.recording_id == event.recording_id
        and existing_event.previous_state is event.previous_state
        and existing_event.target_state is event.target_state
    ]
    if same_state_transitions and not matching_transitions:
        raise ValueError("existing processing event conflicts with transition inputs")
    return bool(matching_ids or matching_transitions)


def _validate_persisted_recording_ids(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        recording_id = row.get("recording_id")
        if type(recording_id) is not str or not recording_id:
            raise ValueError("existing manifest recording_id must be a non-empty string")


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
    with corpus_mutation_lease(recordings_path, events_path):
        _persist_recording_transition_locked(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=recordings,
            event=event,
        )


def _persist_recording_transition_locked(
    *,
    recordings_path: Path,
    events_path: Path,
    recordings: Iterable[RecordingRecord],
    event: ProcessingEvent,
) -> None:
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
    _validate_persisted_recording_ids(existing_rows)
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
    existing_events = read_jsonl(events_path) if events_path.exists() else ()
    if not processing_event_exists(existing_events, event):
        append_jsonl_event(events_path, event.to_dict())
    try:
        write_jsonl_atomic(recordings_path, target_rows)
    except Exception as error:
        raise RecordingTransitionPersistenceError(
            f"event {event.event_id} persisted; recordings update failed; recovery required"
        ) from error


def persist_recording_transitions(
    *,
    recordings_path: Path,
    events_path: Path,
    recordings: Iterable[RecordingRecord],
    events: Iterable[ProcessingEvent],
    _recordings_directory_fd: int | None = None,
    _events_directory_fd: int | None = None,
    _validate_durable_namespace: Callable[[], None] | None = None,
    _expected_recordings_sha256: str | None = None,
    _expected_events_sha256: str | None = None,
    _before_events_write: Callable[[], None] | None = None,
    _after_events_write: Callable[[str], None] | None = None,
    _before_recordings_write: Callable[[], None] | None = None,
    _after_recordings_write: Callable[[str], None] | None = None,
) -> None:
    """Publish a complete transition set audit-first, with one replace per durable file."""
    if recordings_path.resolve() == events_path.resolve():
        raise ValueError("recordings_path and events_path must be distinct resolved paths")
    with corpus_mutation_lease(recordings_path, events_path):
        _persist_recording_transitions_locked(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=recordings,
            events=events,
            _recordings_directory_fd=_recordings_directory_fd,
            _events_directory_fd=_events_directory_fd,
            _validate_durable_namespace=_validate_durable_namespace,
            _expected_recordings_sha256=_expected_recordings_sha256,
            _expected_events_sha256=_expected_events_sha256,
            _before_events_write=_before_events_write,
            _after_events_write=_after_events_write,
            _before_recordings_write=_before_recordings_write,
            _after_recordings_write=_after_recordings_write,
        )


def _persist_recording_transitions_locked(
    *,
    recordings_path: Path,
    events_path: Path,
    recordings: Iterable[RecordingRecord],
    events: Iterable[ProcessingEvent],
    _recordings_directory_fd: int | None = None,
    _events_directory_fd: int | None = None,
    _validate_durable_namespace: Callable[[], None] | None = None,
    _expected_recordings_sha256: str | None = None,
    _expected_events_sha256: str | None = None,
    _before_events_write: Callable[[], None] | None = None,
    _after_events_write: Callable[[str], None] | None = None,
    _before_recordings_write: Callable[[], None] | None = None,
    _after_recordings_write: Callable[[str], None] | None = None,
) -> None:
    target_records = tuple(recordings)
    target_rows = tuple(record.to_dict() for record in target_records)
    expected_events = tuple(events)
    if not target_records or not expected_events:
        raise ValueError("batch transition requires non-empty recordings and events")
    target_by_id = {record.recording_id: record for record in target_records}
    event_by_id = {event.recording_id: event for event in expected_events}
    if len(target_by_id) != len(target_records) or len(event_by_id) != len(expected_events):
        raise ValueError("batch transition identities must be unique")
    if not set(event_by_id) <= set(target_by_id):
        raise ValueError("event recording_ids must be a subset of target recordings")
    for recording_id, event in event_by_id.items():
        require_transition(event.previous_state, event.target_state)
        if target_by_id[recording_id].state is not event.target_state:
            raise ValueError("target manifest state must equal event target_state")

    if _validate_durable_namespace is not None:
        _validate_durable_namespace()
    if (
        _expected_recordings_sha256 is not None
        and _file_sha256(recordings_path, _recordings_directory_fd) != _expected_recordings_sha256
    ):
        raise ValueError("recordings snapshot changed before terminal persistence")
    existing_rows = _read_jsonl_in_directory(recordings_path, _recordings_directory_fd)
    _validate_persisted_recording_ids(existing_rows)
    existing_by_id = {row.get("recording_id"): row for row in existing_rows}
    if len(existing_by_id) != len(existing_rows) or set(existing_by_id) != set(target_by_id):
        raise ValueError("target manifest recording_id set and count must remain unchanged")
    if tuple(existing_by_id) != tuple(target_by_id):
        raise ValueError("recording order must remain unchanged")
    for recording_id, target_record in target_by_id.items():
        existing_row = existing_by_id[recording_id]
        target_row = target_record.to_dict()
        target_event = event_by_id.get(recording_id)
        if target_event is None:
            if target_row != existing_row:
                raise ValueError("non-event recording must remain unchanged")
            continue
        existing_state = existing_row.get("state")
        if existing_state not in {
            target_event.previous_state.value,
            target_event.target_state.value,
        }:
            raise ValueError("existing manifest state is outside the recoverable transition")
        if {key: value for key, value in existing_row.items() if key != "state"} != {
            key: value for key, value in target_row.items() if key != "state"
        }:
            raise ValueError("batch transition may change only recording state")

    if _events_directory_fd is None:
        events_exist = events_path.exists()
    else:  # pragma: no cover - exercised on POSIX hosts
        try:
            os.stat(events_path.name, dir_fd=_events_directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            events_exist = False
        else:
            events_exist = True
    existing_events = (
        _read_jsonl_in_directory(events_path, _events_directory_fd) if events_exist else ()
    )
    if _expected_events_sha256 is not None and (
        not events_exist
        or _file_sha256(events_path, _events_directory_fd) != _expected_events_sha256
    ):
        raise ValueError("processing snapshot changed before terminal persistence")
    missing: list[ProcessingEvent] = []
    for event in expected_events:
        exists = processing_event_exists(existing_events, event)
        existing_state = existing_by_id[event.recording_id]["state"]
        if existing_state == event.target_state.value and not exists:
            raise ValueError("terminal recording state lacks its durable processing event")
        if not exists:
            missing.append(event)
    prospective_rows = (*existing_events, *(event.to_dict() for event in missing))
    validate_processing_events(prospective_rows)
    if missing:
        if _validate_durable_namespace is not None:
            _validate_durable_namespace()
        if _before_events_write is not None:
            _before_events_write()
        _write_jsonl_in_directory(events_path, prospective_rows, _events_directory_fd)
        prospective_sha256 = jsonl_sha256(prospective_rows)
        if _after_events_write is not None:
            _after_events_write(prospective_sha256)
        if _validate_durable_namespace is not None:
            _validate_durable_namespace()
    try:
        if _validate_durable_namespace is not None:
            _validate_durable_namespace()
        if tuple(existing_rows) != target_rows:
            if _before_recordings_write is not None:
                _before_recordings_write()
            _write_jsonl_in_directory(recordings_path, target_rows, _recordings_directory_fd)
            if _after_recordings_write is not None:
                _after_recordings_write(jsonl_sha256(target_rows))
        if _validate_durable_namespace is not None:
            _validate_durable_namespace()
    except Exception as error:
        event_ids = ", ".join(event.event_id for event in expected_events)
        raise RecordingTransitionPersistenceError(
            f"events {event_ids} persisted; recordings update failed; recovery required"
        ) from error
