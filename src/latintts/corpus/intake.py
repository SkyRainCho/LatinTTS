from __future__ import annotations

import csv
from pathlib import Path

from latintts.corpus.records import IntakeRow, RightsRecord
from latintts.corpus.store import read_jsonl

_INTAKE_FIELDS = (
    "relative_path",
    "title_or_citation",
    "speaker_id",
    "rights_id",
    "notes",
)


def read_intake(path: Path) -> tuple[IntakeRow, ...]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _INTAKE_FIELDS:
            raise ValueError("intake.csv must use the exact header")
        rows = tuple(IntakeRow(**row) for row in reader)
    if len({row.relative_path for row in rows}) != len(rows):
        raise ValueError("intake.csv contains duplicate relative_path values")
    return rows


def load_rights(path: Path) -> dict[str, RightsRecord]:
    records: dict[str, RightsRecord] = {}
    for raw in read_jsonl(path):
        record = RightsRecord.from_dict(raw)
        if not record.allow_local_processing or not record.allow_model_training:
            raise ValueError(f"{record.rights_id}: missing local or training authorization")
        if record.rights_id in records:
            raise ValueError(f"duplicate rights_id: {record.rights_id}")
        records[record.rights_id] = record
    return records
