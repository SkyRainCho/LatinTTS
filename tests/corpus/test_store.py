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
