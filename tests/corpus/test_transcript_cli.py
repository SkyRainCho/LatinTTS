from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from latintts.corpus.cli import main
from latintts.corpus.domain import CorpusState
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from tests.corpus.factories import recording


def _set_up_pilot(tmp_path: Path) -> tuple[CorpusPaths, str]:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    record = recording("rec-1", 10.0)
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (record.to_dict(),))
    write_jsonl_atomic(
        paths.manifests / "pilot-selection.json",
        (
            {
                "schema_version": "1",
                "strategy": "explicit-v1",
                "recording_ids": [record.recording_id],
                "inventory_hashes": [record.sha256],
            },
        ),
    )
    return paths, record.recording_id


def _intake_row(recording_id: str, *, confirmed: bool = True) -> dict[str, object]:
    return {
        "recording_id": recording_id,
        "source_candidates": [
            {
                "source_id": "source-1",
                "source_url": "https://example.invalid/source-1",
                "source_version": "edition-1",
                "accessed_at": "2026-07-19",
                "source_file": "text/source-1.txt",
                "selected": True,
            }
        ],
        "spoken_units_file": "text/spoken-units.txt",
        "pronunciation_overrides_file": "",
        "confirmed": confirmed,
        "asr_hypothesis_file": "",
    }


def test_prepare_text_init_creates_exact_intake_skeleton(tmp_path: Path) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)

    assert main(["--project-root", str(tmp_path), "prepare-text", "--init"]) == 0

    assert read_jsonl(paths.manifests / "transcript-intake.jsonl") == (
        {
            "recording_id": recording_id,
            "source_candidates": [
                {
                    "source_id": "",
                    "source_url": "",
                    "source_version": "",
                    "accessed_at": "",
                    "source_file": "",
                    "selected": True,
                }
            ],
            "spoken_units_file": "",
            "pronunciation_overrides_file": "",
            "confirmed": False,
            "asr_hypothesis_file": "",
        },
    )


def test_prepare_text_rejects_unconfirmed_intake(tmp_path: Path, capsys) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    write_jsonl_atomic(
        paths.manifests / "transcript-intake.jsonl",
        (_intake_row(recording_id, confirmed=False),),
    )

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2

    assert "confirmed=true" in capsys.readouterr().err
    assert not (paths.manifests / "transcripts.jsonl").exists()
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "INVENTORIED"


def test_prepare_text_rejects_path_escape(tmp_path: Path, capsys) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    row = _intake_row(recording_id)
    source = dict(row["source_candidates"][0])  # type: ignore[index]
    source["source_file"] = "../outside.txt"
    row["source_candidates"] = [source]
    write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", (row,))

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2

    assert "escapes local-data" in capsys.readouterr().err
    assert not (paths.manifests / "transcripts.jsonl").exists()


def test_prepare_text_confirms_transcript_without_copying_asr_hypothesis(
    tmp_path: Path,
) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    text_dir = paths.local_data / "text"
    text_dir.mkdir()
    (text_dir / "source-1.txt").write_text(
        "Pater noster, qui es in caelis.", encoding="utf-8"
    )
    (text_dir / "source-2.txt").write_text(
        "Pater noster qui es in caelis.", encoding="utf-8"
    )
    (text_dir / "spoken-units.txt").write_text(
        "Pater noster\nqui es in caelis\n", encoding="utf-8"
    )
    (text_dir / "asr.txt").write_text(
        "Pater poster qui es in caelis", encoding="utf-8"
    )
    row = _intake_row(recording_id)
    row["source_candidates"] = [
        *row["source_candidates"],  # type: ignore[misc]
        {
            "source_id": "source-2",
            "source_url": "https://example.invalid/source-2",
            "source_version": "edition-2",
            "accessed_at": "2026-07-19",
            "source_file": "text/source-2.txt",
            "selected": False,
        },
    ]
    row["asr_hypothesis_file"] = "text/asr.txt"
    write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", (row,))

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0

    transcript = read_jsonl(paths.manifests / "transcripts.jsonl")[0]
    assert transcript["state"] == CorpusState.TRANSCRIPT_CONFIRMED.value
    assert transcript["source_text"] == "Pater noster, qui es in caelis."
    assert transcript["spoken_text"] == "Pater noster\nqui es in caelis"
    assert transcript["normalized_text"] == "Pater noster qui es in caelis"
    assert len(transcript["source_candidates"]) == 2
    assert transcript["asr_observation"]["hypothesis"] == "Pater poster qui es in caelis"
    assert transcript["asr_observation"]["confirmed_text"] is None
    assert transcript["asr_observation"]["differences"]
    assert transcript["pronunciation_plan"]["schema_version"] == "1"
    assert transcript["pronunciation_plan"]["rule_version"] == "ecclesiastical-roman-v1"
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == (
        CorpusState.TRANSCRIPT_CONFIRMED.value
    )
    events = read_jsonl(
        paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    )
    assert [(event["previous_state"], event["target_state"]) for event in events] == [
        ("INVENTORIED", "TEXT_CANDIDATES_READY"),
        ("TEXT_CANDIDATES_READY", "TRANSCRIPT_CONFIRMED"),
    ]
    assert "Pater poster" not in transcript["spoken_text"]


def test_prepare_text_rejects_non_object_intake_without_traceback(
    tmp_path: Path, capsys
) -> None:
    paths, _ = _set_up_pilot(tmp_path)
    (paths.manifests / "transcript-intake.jsonl").write_text(
        json.dumps(["not", "an", "object"]), encoding="utf-8"
    )

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2
    assert "MANIFEST_SCHEMA_MISMATCH" in capsys.readouterr().err


def _write_warning_transcript_inputs(paths: CorpusPaths, recording_id: str) -> dict[str, object]:
    text_dir = paths.local_data / "text"
    text_dir.mkdir()
    (text_dir / "source-1.txt").write_text("Fabula", encoding="utf-8")
    (text_dir / "spoken-units.txt").write_text("Fabula\n", encoding="utf-8")
    row = _intake_row(recording_id)
    write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", (row,))
    return row


def test_prepare_text_requires_review_for_unresolved_pronunciation_warning(
    tmp_path: Path, capsys
) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    _write_warning_transcript_inputs(paths, recording_id)

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 1

    assert capsys.readouterr().err.startswith("PRONUNCIATION_NEEDS_REVIEW:")
    transcript = read_jsonl(paths.manifests / "transcripts.jsonl")[0]
    assert transcript["state"] == "TRANSCRIPT_CONFIRMED"
    assert transcript["pronunciation_plan"]["warning_codes"] == [
        "PRONUNCIATION_NEEDS_REVIEW"
    ]
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == (
        "TRANSCRIPT_CONFIRMED"
    )
    review = read_jsonl(paths.manifests / f"pronunciation-review-{recording_id}.json")[0]
    assert review["spoken_text_sha256"] == hashlib.sha256(b"Fabula").hexdigest()
    assert review["review_tokens"] == [
        {
            "token_index": 0,
            "surface": "Fabula",
            "candidate_stress_index": 0,
            "warning_codes": ["PRONUNCIATION_NEEDS_REVIEW"],
        }
    ]


def test_prepare_text_accepts_matching_override_that_resolves_warning(tmp_path: Path) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    row = _write_warning_transcript_inputs(paths, recording_id)
    override = {
        "spoken_text_sha256": hashlib.sha256(b"Fabula").hexdigest(),
        "overrides": {
            "0": {
                "stress_index": 1,
                "ipa": None,
                "model_phonemes": None,
            }
        },
    }
    (paths.local_data / "text" / "overrides.json").write_text(
        json.dumps(override), encoding="utf-8"
    )
    row["pronunciation_overrides_file"] = "text/overrides.json"
    write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", (row,))

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0

    token = read_jsonl(paths.manifests / "transcripts.jsonl")[0]["pronunciation_plan"][
        "tokens"
    ][0]
    assert token["stress_index"] == 1
    assert token["resolution_method"] == "override"
    assert token["warning_codes"] == []
    assert not (paths.manifests / f"pronunciation-review-{recording_id}.json").exists()


def test_prepare_text_is_idempotent_for_unchanged_confirmed_inputs(tmp_path: Path) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    text_dir = paths.local_data / "text"
    text_dir.mkdir()
    (text_dir / "source-1.txt").write_text("Pater noster", encoding="utf-8")
    (text_dir / "spoken-units.txt").write_text("Pater noster\n", encoding="utf-8")
    write_jsonl_atomic(
        paths.manifests / "transcript-intake.jsonl",
        (_intake_row(recording_id),),
    )

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0
    transcript_before = (paths.manifests / "transcripts.jsonl").read_bytes()
    events_path = paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    events_before = events_path.read_bytes()

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0

    assert (paths.manifests / "transcripts.jsonl").read_bytes() == transcript_before
    assert events_path.read_bytes() == events_before


@pytest.mark.parametrize(
    ("stored_hash", "overrides", "message"),
    (
        ("0" * 64, {"0": {"stress_index": 1, "ipa": None, "model_phonemes": None}}, "hash"),
        (
            hashlib.sha256(b"Fabula").hexdigest(),
            {"word": {"stress_index": 1, "ipa": None, "model_phonemes": None}},
            "token indexes",
        ),
        (hashlib.sha256(b"Fabula").hexdigest(), {"0": {}}, "must not be empty"),
        (
            hashlib.sha256(b"Fabula").hexdigest(),
            {
                "0": {
                    "stress_index": 1,
                    "ipa": None,
                    "model_phonemes": None,
                    "unexpected": True,
                }
            },
            "exact fields",
        ),
    ),
)
def test_prepare_text_rejects_invalid_override_files(
    tmp_path: Path,
    capsys,
    stored_hash: str,
    overrides: dict[str, object],
    message: str,
) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    row = _write_warning_transcript_inputs(paths, recording_id)
    (paths.local_data / "text" / "overrides.json").write_text(
        json.dumps({"spoken_text_sha256": stored_hash, "overrides": overrides}),
        encoding="utf-8",
    )
    row["pronunciation_overrides_file"] = "text/overrides.json"
    write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", (row,))

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2

    assert message in capsys.readouterr().err
    assert not (paths.manifests / "transcripts.jsonl").exists()
