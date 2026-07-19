from __future__ import annotations

import hashlib
from typing import Literal

from latintts.corpus.domain import CorpusState
from latintts.corpus.records import AudioMetadata, RecordingRecord


def recording(
    recording_id: str,
    duration_seconds: float,
    *,
    content_type: Literal["spoken", "sung"] = "spoken",
    state: CorpusState = CorpusState.INVENTORIED,
) -> RecordingRecord:
    digest = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()
    return RecordingRecord(
        schema_version="1",
        recording_id=recording_id,
        relative_path=f"raw/{content_type}/{recording_id}.wav",
        sha256=digest,
        content_type=content_type,
        title_or_citation=recording_id,
        speaker_id="speaker-1",
        rights_id="rights-1",
        notes="",
        metadata=AudioMetadata(duration_seconds, 48000, 1, "pcm_s16le", None),
        state=state,
    )
