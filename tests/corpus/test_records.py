from latintts.corpus.domain import CorpusState
from latintts.corpus.records import AudioMetadata, RecordingRecord, advance_recording


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
