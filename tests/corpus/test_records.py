from dataclasses import replace

import pytest

from latintts.corpus.domain import CorpusState
from latintts.corpus.records import (
    AudioMetadata,
    RecordingRecord,
    ReviewEvent,
    advance_recording,
)


def _recording(state: CorpusState) -> RecordingRecord:
    return RecordingRecord(
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
        state=state,
    )


def test_recording_serialization_uses_state_value() -> None:
    record = _recording(CorpusState.INVENTORIED)
    assert record.to_dict()["state"] == "INVENTORIED"


def test_advance_recording_returns_event_and_new_immutable_record() -> None:
    record = _recording(CorpusState.DISCOVERED)
    updated, event = advance_recording(
        record,
        CorpusState.INVENTORIED,
        input_sha256s=("a" * 64,),
        config_sha256="b" * 64,
        tool_versions=("ffprobe=test",),
        started_at="2026-07-19T10:00:00+08:00",
        finished_at="2026-07-19T10:00:01+08:00",
        result="success",
    )
    assert record.state is CorpusState.DISCOVERED
    assert updated.state is CorpusState.INVENTORIED
    assert event.previous_state is CorpusState.DISCOVERED
    assert event.target_state is CorpusState.INVENTORIED


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"sha256": "not-a-sha256"}, "sha256"),
        ({"content_type": "music"}, "content_type"),
        ({"state": "INVENTORIED"}, "state"),
    ],
)
def test_recording_rejects_invalid_runtime_values(changes: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        replace(_recording(CorpusState.INVENTORIED), **changes)


def test_audio_metadata_rejects_boolean_sample_rate() -> None:
    with pytest.raises(TypeError, match="sample_rate"):
        AudioMetadata(10.0, True, 1, "pcm_s16le", None)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"result": "maybe"}, "result"),
        ({"config_sha256": "not-a-sha256"}, "config_sha256"),
        ({"started_at": "not-a-timestamp"}, "started_at"),
        (
            {
                "started_at": "2026-07-19T10:00:02+08:00",
                "finished_at": "2026-07-19T10:00:01+08:00",
            },
            "finished_at",
        ),
    ],
)
def test_processing_event_rejects_invalid_runtime_values(
    changes: dict[str, object], match: str
) -> None:
    arguments: dict[str, object] = {
        "input_sha256s": ("a" * 64,),
        "config_sha256": "b" * 64,
        "tool_versions": ("ffprobe=test",),
        "started_at": "2026-07-19T10:00:00+08:00",
        "finished_at": "2026-07-19T10:00:01+08:00",
        "result": "success",
    }
    arguments.update(changes)
    with pytest.raises((TypeError, ValueError), match=match):
        advance_recording(
            _recording(CorpusState.DISCOVERED),
            CorpusState.INVENTORIED,
            **arguments,
        )


def test_review_event_rejects_invalid_reviewed_at() -> None:
    with pytest.raises(ValueError, match="reviewed_at"):
        ReviewEvent(
            schema_version="1",
            review_event_id="review-1",
            entity_id="segment-1",
            field="segment_end",
            before=14.82,
            after=15.07,
            reason="final consonant truncated",
            reviewer="human",
            reviewed_at="yesterday",
        )
