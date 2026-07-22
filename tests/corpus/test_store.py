from dataclasses import replace
from pathlib import Path

import pytest

from latintts.corpus import store
from latintts.corpus.domain import CorpusState
from latintts.corpus.records import (
    AudioMetadata,
    ProcessingEvent,
    RecordingRecord,
    advance_recording,
)
from latintts.corpus.store import append_jsonl_event, read_jsonl, write_jsonl_atomic


def _transition() -> tuple[RecordingRecord, RecordingRecord, ProcessingEvent]:
    original = RecordingRecord(
        schema_version="1",
        recording_id="rec-abc",
        relative_path="raw/spoken/a.wav",
        sha256="a" * 64,
        content_type="spoken",
        title_or_citation="Pater Noster",
        speaker_id="speaker-1",
        rights_id="rights-1",
        notes="",
        metadata=AudioMetadata(10.0, 48000, 1, "pcm_s16le", None),
        state=CorpusState.DISCOVERED,
    )
    updated, event = advance_recording(
        original,
        CorpusState.INVENTORIED,
        input_sha256s=("a" * 64,),
        config_sha256="b" * 64,
        tool_versions=("ffprobe=test",),
        started_at="2026-07-19T10:00:00+08:00",
        finished_at="2026-07-19T10:00:01+08:00",
        result="success",
    )
    return original, updated, event


def _other_recording() -> RecordingRecord:
    original, _, _ = _transition()
    return replace(
        original,
        recording_id="rec-other",
        relative_path="raw/spoken/other.wav",
        sha256="c" * 64,
    )


def test_atomic_jsonl_round_trip_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl_atomic(path, ({"text": "cælum", "ordinal": 1},))
    assert read_jsonl(path) == ({"ordinal": 1, "text": "cælum"},)


def test_read_jsonl_rejects_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"one"}\n\n', encoding="utf-8")
    with pytest.raises(ValueError, match="blank line 2"):
        read_jsonl(path)


def test_append_event_keeps_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    append_jsonl_event(path, {"id": "one"})
    append_jsonl_event(path, {"id": "two"})
    assert [row["id"] for row in read_jsonl(path)] == ["one", "two"]


def test_read_jsonl_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text('{"id":"one","id":"two"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate key.*id"):
        read_jsonl(path)


def test_persist_transition_writes_event_before_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    write_jsonl_atomic(recordings_path, (original.to_dict(),))
    destinations: list[Path] = []
    real_replace = store.os.replace

    def observe_replace(source: Path, destination: Path) -> None:
        destinations.append(Path(destination))
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", observe_replace)
    store.persist_recording_transition(
        recordings_path=recordings_path,
        events_path=events_path,
        recordings=(updated,),
        event=event,
    )

    assert destinations == [events_path, recordings_path]
    assert read_jsonl(events_path) == (event.to_dict(),)
    assert read_jsonl(recordings_path)[0]["state"] == "INVENTORIED"


def test_persist_transition_keeps_event_when_recordings_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    write_jsonl_atomic(recordings_path, (original.to_dict(),))
    real_replace = store.os.replace

    def fail_recordings_replace(source: Path, destination: Path) -> None:
        if Path(destination) == recordings_path:
            raise OSError("disk full")
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", fail_recordings_replace)
    with pytest.raises(RuntimeError, match=r"event.*persisted.*recovery required"):
        store.persist_recording_transition(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=(updated,),
            event=event,
        )

    assert read_jsonl(events_path) == (event.to_dict(),)
    assert read_jsonl(recordings_path)[0]["state"] == "DISCOVERED"


def test_persist_transition_recovers_manifest_without_duplicate_event(tmp_path: Path) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    write_jsonl_atomic(recordings_path, (original.to_dict(),))
    write_jsonl_atomic(events_path, (event.to_dict(),))

    store.persist_recording_transition(
        recordings_path=recordings_path,
        events_path=events_path,
        recordings=(updated,),
        event=event,
    )

    assert read_jsonl(events_path) == (event.to_dict(),)
    assert read_jsonl(recordings_path) == (updated.to_dict(),)


def test_persist_transitions_publishes_all_events_then_one_recordings_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    other = replace(
        original,
        recording_id="rec-other",
        relative_path="raw/spoken/other.wav",
        sha256="c" * 64,
    )
    other_updated, other_event = advance_recording(
        other,
        CorpusState.INVENTORIED,
        input_sha256s=(other.sha256,),
        config_sha256="b" * 64,
        tool_versions=("ffprobe=test",),
        started_at="2026-07-19T10:00:00+08:00",
        finished_at="2026-07-19T10:00:01+08:00",
        result="success",
    )
    write_jsonl_atomic(recordings_path, (original.to_dict(), other.to_dict()))
    destinations: list[Path] = []
    real_replace = store.os.replace

    def observe_replace(source: Path, destination: Path) -> None:
        destinations.append(Path(destination))
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", observe_replace)
    store.persist_recording_transitions(
        recordings_path=recordings_path,
        events_path=events_path,
        recordings=(updated, other_updated),
        events=(event, other_event),
    )

    assert destinations == [events_path, recordings_path]
    assert read_jsonl(events_path) == (event.to_dict(), other_event.to_dict())
    assert read_jsonl(recordings_path) == (updated.to_dict(), other_updated.to_dict())


def test_persist_transitions_preserves_unchanged_recordings_without_events(
    tmp_path: Path,
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    unselected = _other_recording()
    write_jsonl_atomic(recordings_path, (original.to_dict(), unselected.to_dict()))

    store.persist_recording_transitions(
        recordings_path=recordings_path,
        events_path=events_path,
        recordings=(updated, unselected),
        events=(event,),
    )

    assert read_jsonl(events_path) == (event.to_dict(),)
    assert read_jsonl(recordings_path) == (updated.to_dict(), unselected.to_dict())


@pytest.mark.parametrize("field", ("state", "notes"))
def test_persist_transitions_rejects_changes_to_recordings_without_events(
    tmp_path: Path, field: str
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    unselected = _other_recording()
    changed = (
        replace(unselected, state=CorpusState.INVENTORIED)
        if field == "state"
        else replace(unselected, notes="changed")
    )
    write_jsonl_atomic(recordings_path, (original.to_dict(), unselected.to_dict()))

    with pytest.raises(ValueError, match="non-event recording must remain unchanged"):
        store.persist_recording_transitions(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=(updated, changed),
            events=(event,),
        )

    assert read_jsonl(recordings_path) == (original.to_dict(), unselected.to_dict())
    assert not events_path.exists()


def test_persist_transitions_rejects_stale_expected_snapshot_before_any_write(
    tmp_path: Path,
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    write_jsonl_atomic(recordings_path, (original.to_dict(),))
    write_jsonl_atomic(events_path, ())
    before = {
        recordings_path: recordings_path.read_bytes(),
        events_path: events_path.read_bytes(),
    }

    with pytest.raises(ValueError, match="recordings snapshot changed"):
        store.persist_recording_transitions(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=(updated,),
            events=(event,),
            _expected_recordings_sha256="0" * 64,
            _expected_events_sha256=store.jsonl_sha256(()),
        )

    assert {path: path.read_bytes() for path in before} == before


def test_persist_transitions_hands_off_old_and_new_file_guards_in_order(
    tmp_path: Path,
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    original_rows = (original.to_dict(),)
    write_jsonl_atomic(recordings_path, original_rows)
    write_jsonl_atomic(events_path, ())
    calls: list[tuple[str, str | None]] = []

    store.persist_recording_transitions(
        recordings_path=recordings_path,
        events_path=events_path,
        recordings=(updated,),
        events=(event,),
        _expected_recordings_sha256=store.jsonl_sha256(original_rows),
        _expected_events_sha256=store.jsonl_sha256(()),
        _before_events_write=lambda: calls.append(("before-events", None)),
        _after_events_write=lambda digest: calls.append(("after-events", digest)),
        _before_recordings_write=lambda: calls.append(("before-recordings", None)),
        _after_recordings_write=lambda digest: calls.append(("after-recordings", digest)),
    )

    assert calls == [
        ("before-events", None),
        ("after-events", store.jsonl_sha256((event.to_dict(),))),
        ("before-recordings", None),
        ("after-recordings", store.jsonl_sha256((updated.to_dict(),))),
    ]


def test_persist_transitions_recovers_mixed_states_and_durable_events_without_duplicates(
    tmp_path: Path,
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    other = replace(
        original,
        recording_id="rec-other",
        relative_path="raw/spoken/other.wav",
        sha256="c" * 64,
    )
    other_updated, other_event = advance_recording(
        other,
        CorpusState.INVENTORIED,
        input_sha256s=(other.sha256,),
        config_sha256="b" * 64,
        tool_versions=("ffprobe=test",),
        started_at="2026-07-19T10:00:00+08:00",
        finished_at="2026-07-19T10:00:01+08:00",
        result="success",
    )
    write_jsonl_atomic(recordings_path, (updated.to_dict(), other.to_dict()))
    write_jsonl_atomic(events_path, (event.to_dict(), other_event.to_dict()))

    store.persist_recording_transitions(
        recordings_path=recordings_path,
        events_path=events_path,
        recordings=(updated, other_updated),
        events=(event, other_event),
    )

    assert read_jsonl(events_path) == (event.to_dict(), other_event.to_dict())
    assert read_jsonl(recordings_path) == (updated.to_dict(), other_updated.to_dict())


def test_persist_transitions_rejects_duplicate_event_id_in_prospective_batch_without_writes(
    tmp_path: Path,
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    other = replace(
        original,
        recording_id="rec-other",
        relative_path="raw/spoken/other.wav",
        sha256="c" * 64,
    )
    other_updated, other_event = advance_recording(
        other,
        CorpusState.INVENTORIED,
        input_sha256s=(other.sha256,),
        config_sha256="b" * 64,
        tool_versions=("ffprobe=test",),
        started_at="2026-07-19T10:00:00+08:00",
        finished_at="2026-07-19T10:00:01+08:00",
        result="success",
    )
    conflicting = replace(other_event, event_id=event.event_id)
    write_jsonl_atomic(recordings_path, (original.to_dict(), other.to_dict()))
    before_recordings = recordings_path.read_bytes()

    with pytest.raises(ValueError, match="duplicate event_id"):
        store.persist_recording_transitions(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=(updated, other_updated),
            events=(event, conflicting),
        )

    assert recordings_path.read_bytes() == before_recordings
    assert not events_path.exists()


def test_persist_transitions_rejects_prospective_state_chain_gap_without_writes(
    tmp_path: Path,
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    _, inventoried, inventoried_event = _transition()
    ready = replace(inventoried, state=CorpusState.TEXT_CANDIDATES_READY)
    confirmed, confirmed_event = advance_recording(
        ready,
        CorpusState.TRANSCRIPT_CONFIRMED,
        input_sha256s=(ready.sha256, "c" * 64),
        config_sha256="b" * 64,
        tool_versions=("prepare-text=test",),
        started_at="2026-07-19T10:00:02+08:00",
        finished_at="2026-07-19T10:00:03+08:00",
        result="success",
    )
    write_jsonl_atomic(recordings_path, (ready.to_dict(),))
    write_jsonl_atomic(events_path, (inventoried_event.to_dict(),))
    before_recordings = recordings_path.read_bytes()
    before_events = events_path.read_bytes()

    with pytest.raises(ValueError, match="state chain"):
        store.persist_recording_transitions(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=(confirmed,),
            events=(confirmed_event,),
        )

    assert recordings_path.read_bytes() == before_recordings
    assert events_path.read_bytes() == before_events


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("alias", "distinct resolved"),
        ("empty", "non-empty"),
        ("duplicate-events", "identities.*unique"),
        ("coverage", "event recording_ids must be a subset"),
        ("target-state", "target manifest state"),
        ("id-set", "recording_id set"),
        ("existing-state", "recoverable transition"),
        ("field-change", "change only recording state"),
        ("state-ahead", "lacks its durable"),
    ),
)
def test_persist_transitions_rejects_invalid_batch_contracts(
    tmp_path: Path, case: str, message: str
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    existing = (original,)
    targets = (updated,)
    events = (event,)
    if case == "alias":
        events_path = recordings_path
    elif case == "empty":
        events = ()
    elif case == "duplicate-events":
        events = (event, event)
    elif case == "coverage":
        events = (replace(event, recording_id="rec-other"),)
    elif case == "target-state":
        targets = (original,)
    elif case == "id-set":
        existing = (replace(original, recording_id="rec-other"),)
    elif case == "existing-state":
        existing = (replace(original, state=CorpusState.TEXT_CANDIDATES_READY),)
    elif case == "field-change":
        existing = (replace(original, notes="changed"),)
    else:
        existing = (updated,)
    write_jsonl_atomic(recordings_path, (record.to_dict() for record in existing))

    with pytest.raises(ValueError, match=message):
        store.persist_recording_transitions(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=targets,
            events=events,
        )


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("missing-field", "exact fields"),
        ("unknown-field", "exact fields"),
        ("illegal-enum", "not a valid CorpusState"),
        ("illegal-hash", "SHA-256"),
        ("illegal-time", "timezone-aware ISO datetime"),
    ),
)
def test_processing_event_recovery_rejects_malformed_history_rows(
    case: str,
    match: str,
) -> None:
    _, _, event = _transition()
    row = event.to_dict()
    if case == "missing-field":
        del row["schema_version"]
    elif case == "unknown-field":
        row["unexpected"] = True
    elif case == "illegal-enum":
        row["target_state"] = "UNKNOWN"
    elif case == "illegal-hash":
        row["config_sha256"] = "not-a-hash"
    elif case == "illegal-time":
        row["started_at"] = "2026-07-19T10:00:00"

    with pytest.raises((TypeError, ValueError), match=match):
        store.processing_event_exists((row,), event)


def test_processing_event_recovery_rejects_same_semantics_with_different_id() -> None:
    _, _, event = _transition()
    row = event.to_dict()
    row["event_id"] = "state-different-id"

    with pytest.raises(ValueError, match=r"semantic transition.*different event_id"):
        store.processing_event_exists((row,), event)


@pytest.mark.parametrize("duplicate", ("id", "semantics"))
def test_processing_event_recovery_rejects_duplicate_history_identity(duplicate: str) -> None:
    _, _, event = _transition()
    second = event.to_dict()
    if duplicate == "semantics":
        second["event_id"] = "state-other-deterministic-id"
    with pytest.raises(ValueError, match=r"duplicate (event_id|semantic transition)"):
        store.processing_event_exists((event.to_dict(), second), event)


def test_processing_event_recovery_rejects_same_id_with_conflicting_semantics() -> None:
    _, _, event = _transition()
    row = event.to_dict()
    row["config_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="conflicts with transition event"):
        store.processing_event_exists((row,), event)


def test_processing_event_recovery_rejects_invalid_history_chain_order() -> None:
    _, inventoried, _ = _transition()
    ready, ready_event = advance_recording(
        inventoried,
        CorpusState.TEXT_CANDIDATES_READY,
        input_sha256s=(inventoried.sha256,),
        config_sha256="b" * 64,
        tool_versions=("prepare-text=test",),
        started_at="2026-07-19T10:00:02+08:00",
        finished_at="2026-07-19T10:00:03+08:00",
        result="success",
    )
    _, confirmed_event = advance_recording(
        ready,
        CorpusState.TRANSCRIPT_CONFIRMED,
        input_sha256s=(inventoried.sha256, "c" * 64),
        config_sha256="b" * 64,
        tool_versions=("prepare-text=test",),
        started_at="2026-07-19T10:00:04+08:00",
        finished_at="2026-07-19T10:00:05+08:00",
        result="success",
    )

    with pytest.raises(ValueError, match="invalid processing event state chain"):
        store.processing_event_exists(
            (confirmed_event.to_dict(), ready_event.to_dict()),
            ready_event,
        )


@pytest.mark.parametrize(
    ("scenario", "match"),
    [
        ("target_missing", "target manifest.*exactly one"),
        ("target_duplicate", "target manifest.*exactly one"),
        ("target_state", "target_state"),
        ("existing_missing", "existing manifest.*exactly one"),
        ("existing_state", "previous_state"),
    ],
)
def test_persist_transition_preflights_manifest_consistency_before_event_append(
    tmp_path: Path, scenario: str, match: str
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    existing = original
    targets = (updated,)
    if scenario == "target_missing":
        targets = (replace(updated, recording_id="rec-other"),)
    elif scenario == "target_duplicate":
        targets = (updated, updated)
    elif scenario == "target_state":
        targets = (original,)
    elif scenario == "existing_missing":
        existing = replace(original, recording_id="rec-other")
    elif scenario == "existing_state":
        existing = updated
    write_jsonl_atomic(recordings_path, (existing.to_dict(),))

    with pytest.raises(ValueError, match=match):
        store.persist_recording_transition(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=targets,
            event=event,
        )

    assert not events_path.exists()
    assert read_jsonl(recordings_path) == (existing.to_dict(),)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_read_jsonl_rejects_nonstandard_numeric_constants(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "nonstandard.jsonl"
    path.write_text(f'{{"value":{constant}}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=rf"non-standard JSON constant.*{constant}"):
        read_jsonl(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_write_jsonl_rejects_nonfinite_numbers_without_replacing_existing(
    tmp_path: Path, value: float
) -> None:
    path = tmp_path / "finite-only.jsonl"
    write_jsonl_atomic(path, ({"value": 1.0},))
    with pytest.raises(ValueError):
        write_jsonl_atomic(path, ({"value": value},))
    assert read_jsonl(path) == ({"value": 1.0},)


@pytest.mark.parametrize(
    ("scenario", "match"),
    [
        ("other_state", "non-event recording.*unchanged"),
        ("delete_other", "recording_id.*count"),
        ("add_other", "recording_id.*count"),
        ("other_field", "non-event recording.*unchanged"),
        ("target_field", "event recording.*only state"),
    ],
)
def test_persist_transition_rejects_manifest_changes_outside_event_state(
    tmp_path: Path, scenario: str, match: str
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "processing-events.jsonl"
    original, updated, event = _transition()
    other = _other_recording()
    existing = (original, other)
    targets = (updated, other)
    if scenario == "other_state":
        targets = (updated, replace(other, state=CorpusState.INVENTORIED))
    elif scenario == "delete_other":
        targets = (updated,)
    elif scenario == "add_other":
        existing = (original,)
    elif scenario == "other_field":
        targets = (updated, replace(other, notes="changed"))
    elif scenario == "target_field":
        targets = (replace(updated, notes="changed"), other)
    write_jsonl_atomic(recordings_path, (record.to_dict() for record in existing))

    with pytest.raises(ValueError, match=match):
        store.persist_recording_transition(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=targets,
            event=event,
        )

    assert not events_path.exists()
    assert read_jsonl(recordings_path) == tuple(record.to_dict() for record in existing)


@pytest.mark.parametrize("alias_kind", ["same", "relative"])
def test_persist_transition_rejects_path_alias_before_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_kind: str
) -> None:
    recordings_path = tmp_path / "same.jsonl"
    events_path = recordings_path
    if alias_kind == "relative":
        events_path = tmp_path / "missing-directory" / ".." / "same.jsonl"
    _, updated, event = _transition()

    def reject_read(path: Path) -> tuple[dict[str, object], ...]:
        raise AssertionError(f"I/O attempted through {path}")

    monkeypatch.setattr(store, "read_jsonl", reject_read)
    with pytest.raises(ValueError, match="distinct resolved paths"):
        store.persist_recording_transition(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=(updated,),
            event=event,
        )

    assert not recordings_path.exists()


def test_persist_transition_rejects_symlink_alias_before_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_path = tmp_path / "recordings.jsonl"
    events_path = tmp_path / "events-link.jsonl"
    recordings_path.write_bytes(b"unchanged")
    try:
        events_path.symlink_to(recordings_path)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    _, updated, event = _transition()

    def reject_read(path: Path) -> tuple[dict[str, object], ...]:
        raise AssertionError(f"I/O attempted through {path}")

    monkeypatch.setattr(store, "read_jsonl", reject_read)
    with pytest.raises(ValueError, match="distinct resolved paths"):
        store.persist_recording_transition(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=(updated,),
            event=event,
        )

    assert recordings_path.read_bytes() == b"unchanged"
    assert events_path.is_symlink()
