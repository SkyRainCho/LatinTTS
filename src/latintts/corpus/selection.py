from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from latintts.corpus.domain import CorpusState
from latintts.corpus.records import RecordingRecord


@dataclass(frozen=True, slots=True)
class PilotSelection:
    schema_version: str
    strategy: str
    recording_ids: tuple[str, ...]
    inventory_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["recording_ids"] = list(self.recording_ids)
        raw["inventory_hashes"] = list(self.inventory_hashes)
        return raw


def select_pilot(
    records: tuple[RecordingRecord, ...],
    explicit_ids: tuple[str, ...] = (),
) -> PilotSelection:
    recording_ids = tuple(record.recording_id for record in records)
    if len(recording_ids) != len(set(recording_ids)):
        raise ValueError("records must not contain duplicate recording_id")
    eligible = {
        record.recording_id: record
        for record in records
        if record.content_type == "spoken" and record.state is CorpusState.INVENTORIED
    }
    if explicit_ids:
        if len(explicit_ids) not in (2, 3) or len(set(explicit_ids)) != len(explicit_ids):
            raise ValueError("explicit pilot selection must contain two or three unique IDs")
        missing = set(explicit_ids) - set(eligible)
        if missing:
            raise ValueError(f"pilot IDs are missing or not eligible: {sorted(missing)}")
        chosen = tuple(eligible[recording_id] for recording_id in explicit_ids)
        strategy = "explicit-v1"
    else:
        ordered = sorted(
            eligible.values(),
            key=lambda record: (record.metadata.duration_seconds, record.recording_id),
        )
        if len(ordered) < 2:
            raise ValueError("at least two eligible spoken recordings are required")
        if len(ordered) <= 3:
            chosen = tuple(ordered)
        else:
            median = statistics.median(record.metadata.duration_seconds for record in ordered)
            middle = min(
                ordered[1:-1],
                key=lambda record: (
                    abs(record.metadata.duration_seconds - median),
                    record.metadata.duration_seconds,
                    record.recording_id,
                ),
            )
            chosen = (ordered[0], middle, ordered[-1])
        strategy = "duration-short-median-long-v1"
    return PilotSelection(
        schema_version="1",
        strategy=strategy,
        recording_ids=tuple(record.recording_id for record in chosen),
        inventory_hashes=tuple(record.sha256 for record in chosen),
    )
