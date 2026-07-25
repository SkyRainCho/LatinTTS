from pathlib import Path

import pytest

from latintts.corpus.intake import load_rights, read_intake
from latintts.corpus.records import RightsRecord


def _valid_rights() -> dict[str, object]:
    return {
        "rights_id": "r1",
        "owner_id": "o1",
        "speaker_id": "s1",
        "allow_local_processing": True,
        "allow_model_training": True,
        "allow_internal_evaluation": True,
        "allow_raw_release": "unknown",
        "allow_segment_release": "unknown",
        "allow_model_release": "unknown",
        "authorized_at": "2026-07-19",
        "basis": "owner",
        "notes": "",
    }


def test_intake_requires_exact_header(tmp_path: Path) -> None:
    path = tmp_path / "intake.csv"
    path.write_text("relative_path,title_or_citation\nraw/spoken/a.wav,Pater Noster\n")
    with pytest.raises(ValueError, match="exact header"):
        read_intake(path)


def test_rights_require_local_processing_and_training_authority(tmp_path: Path) -> None:
    path = tmp_path / "rights.jsonl"
    path.write_text(
        '{"rights_id":"r1","owner_id":"o1","speaker_id":"s1",'
        '"allow_local_processing":true,"allow_model_training":false,'
        '"allow_internal_evaluation":true,"allow_raw_release":"unknown",'
        '"allow_segment_release":"unknown","allow_model_release":"unknown",'
        '"authorized_at":"2026-07-19","basis":"owner","notes":""}\n'
    )
    with pytest.raises(ValueError, match="training authorization"):
        load_rights(path)


@pytest.mark.parametrize("value", [0, 1])
def test_rights_release_fields_reject_integer_booleans(value: int) -> None:
    raw = _valid_rights()
    raw["allow_raw_release"] = value
    with pytest.raises(TypeError, match="booleans or unknown"):
        RightsRecord.from_dict(raw)


def test_rights_reject_invalid_authorized_at() -> None:
    raw = _valid_rights()
    raw["authorized_at"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="authorized_at"):
        RightsRecord.from_dict(raw)
