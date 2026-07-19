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
