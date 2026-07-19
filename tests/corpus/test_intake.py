from pathlib import Path

import pytest

from latintts.corpus.intake import load_rights, read_intake


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
