from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    title: str
    url: str
    source_kind: str
    authority_rank: int
    accessed_on: str
    locator: str
    license_or_terms: str
    usage_note: str


def load_source_registry() -> dict[str, SourceRecord]:
    raw = (
        files("latintts.resources")
        .joinpath("pronunciation_sources.json")
        .read_text(encoding="utf-8")
    )
    rows = json.loads(raw)
    records = [SourceRecord(**row) for row in rows]
    result = {record.source_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("pronunciation source IDs must be unique")
    if any(record.authority_rank < 1 for record in records):
        raise ValueError("authority_rank must be positive")
    return result
