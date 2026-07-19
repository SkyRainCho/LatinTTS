from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from latintts.corpus.cli import main
from latintts.corpus.domain import CorpusState
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import TranscriptIntakeRow
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from tests.corpus.factories import recording


def _set_up_pilot(tmp_path: Path, recording_id: str = "rec-1") -> tuple[CorpusPaths, str]:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    record = recording(recording_id, 10.0)
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


def _review_path(paths: CorpusPaths, recording_id: str) -> Path:
    slug = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()
    return paths.manifests / f"pronunciation-review-{slug}.json"


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


def _set_up_clean_pilot_batch(
    tmp_path: Path,
    recording_ids: tuple[str, ...],
) -> tuple[CorpusPaths, tuple[object, ...]]:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    records = tuple(
        recording(recording_id, float(index + 1))
        for index, recording_id in enumerate(recording_ids)
    )
    write_jsonl_atomic(
        paths.manifests / "recordings.jsonl",
        (record.to_dict() for record in records),
    )
    write_jsonl_atomic(
        paths.manifests / "pilot-selection.json",
        (
            {
                "schema_version": "1",
                "strategy": "explicit-v1",
                "recording_ids": list(recording_ids),
                "inventory_hashes": [record.sha256 for record in records],
            },
        ),
    )
    text_dir = paths.local_data / "text"
    text_dir.mkdir()
    (text_dir / "source-1.txt").write_text("Pater noster", encoding="utf-8")
    (text_dir / "spoken-units.txt").write_text("Pater noster\n", encoding="utf-8")
    write_jsonl_atomic(
        paths.manifests / "transcript-intake.jsonl",
        (_intake_row(recording_id) for recording_id in recording_ids),
    )
    return paths, records


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
    original = (paths.manifests / "transcript-intake.jsonl").read_bytes()
    assert main(["--project-root", str(tmp_path), "prepare-text", "--init"]) == 0
    assert (paths.manifests / "transcript-intake.jsonl").read_bytes() == original


def test_prepare_text_init_rejects_invalid_existing_intake(tmp_path: Path) -> None:
    paths, _ = _set_up_pilot(tmp_path)
    intake_path = paths.manifests / "transcript-intake.jsonl"
    intake_path.write_text("\n", encoding="utf-8")
    original = intake_path.read_bytes()

    assert main(["--project-root", str(tmp_path), "prepare-text", "--init"]) == 2
    assert intake_path.read_bytes() == original


def test_prepare_text_init_requires_replace_for_different_existing_intake(
    tmp_path: Path, capsys
) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    intake_path = paths.manifests / "transcript-intake.jsonl"
    write_jsonl_atomic(intake_path, (_intake_row(recording_id),))
    original = intake_path.read_bytes()

    assert main(["--project-root", str(tmp_path), "prepare-text", "--init"]) == 2

    assert "--replace" in capsys.readouterr().err
    assert intake_path.read_bytes() == original

    assert main(["--project-root", str(tmp_path), "prepare-text", "--init", "--replace"]) == 0
    assert read_jsonl(intake_path)[0]["confirmed"] is False


def test_prepare_text_does_not_replace_existing_different_facts(tmp_path: Path) -> None:
    paths, _ = _set_up_clean_pilot_batch(tmp_path, ("rec-1",))
    transcripts_path = paths.manifests / "transcripts.jsonl"
    write_jsonl_atomic(transcripts_path, ({"recording_id": "different-facts"},))
    original_transcripts = transcripts_path.read_bytes()
    original_recordings = (paths.manifests / "recordings.jsonl").read_bytes()

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2

    assert transcripts_path.read_bytes() == original_transcripts
    assert (paths.manifests / "recordings.jsonl").read_bytes() == original_recordings
    assert not (paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl").exists()


@pytest.mark.parametrize(
    "case",
    (
        "missing-selection",
        "missing-intake",
        "duplicate-intake",
        "mismatched-intake-id",
        "mismatched-inventory-hash",
        "invalid-recording-state",
        "normal-replace",
    ),
)
def test_prepare_text_fails_closed_on_manifest_and_state_conflicts(
    tmp_path: Path,
    case: str,
) -> None:
    if case == "missing-selection":
        paths = CorpusPaths.from_project_root(tmp_path)
        paths.ensure_layout()
        assert main(["--project-root", str(tmp_path), "prepare-text", "--init"]) == 2
        return
    paths, records = _set_up_clean_pilot_batch(tmp_path, ("rec-1",))
    arguments = ["--project-root", str(tmp_path), "prepare-text"]
    if case == "missing-intake":
        (paths.manifests / "transcript-intake.jsonl").unlink()
    elif case == "duplicate-intake":
        row = _intake_row("rec-1")
        write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", (row, row))
    elif case == "mismatched-intake-id":
        write_jsonl_atomic(
            paths.manifests / "transcript-intake.jsonl",
            (_intake_row("other-recording"),),
        )
    elif case == "mismatched-inventory-hash":
        selection = dict(read_jsonl(paths.manifests / "pilot-selection.json")[0])
        selection["inventory_hashes"] = ["0" * 64]
        write_jsonl_atomic(paths.manifests / "pilot-selection.json", (selection,))
    elif case == "invalid-recording-state":
        write_jsonl_atomic(
            paths.manifests / "recordings.jsonl",
            (replace(records[0], state=CorpusState.SEGMENTED).to_dict(),),
        )
    elif case == "normal-replace":
        arguments.append("--replace")

    assert main(arguments) == 2
    assert not (paths.manifests / "transcripts.jsonl").exists()


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
    (text_dir / "source-1.txt").write_text("Pater noster, qui es in caelis.", encoding="utf-8")
    (text_dir / "source-2.txt").write_text("Pater noster qui es in caelis.", encoding="utf-8")
    (text_dir / "spoken-units.txt").write_text("Pater noster\nqui es in caelis\n", encoding="utf-8")
    (text_dir / "asr.txt").write_text("Pater poster qui es in caelis", encoding="utf-8")
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
    events = read_jsonl(paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl")
    assert [(event["previous_state"], event["target_state"]) for event in events] == [
        ("INVENTORIED", "TEXT_CANDIDATES_READY"),
        ("TEXT_CANDIDATES_READY", "TRANSCRIPT_CONFIRMED"),
    ]
    assert "Pater poster" not in transcript["spoken_text"]


def test_prepare_text_rejects_non_object_intake_without_traceback(tmp_path: Path, capsys) -> None:
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
    assert transcript["pronunciation_plan"]["warning_codes"] == ["PRONUNCIATION_NEEDS_REVIEW"]
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == ("TRANSCRIPT_CONFIRMED")
    review = read_jsonl(_review_path(paths, recording_id))[0]
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

    token = read_jsonl(paths.manifests / "transcripts.jsonl")[0]["pronunciation_plan"]["tokens"][0]
    assert token["stress_index"] == 1
    assert token["resolution_method"] == "override"
    assert token["warning_codes"] == []
    assert not _review_path(paths, recording_id).exists()


def test_prepare_text_allows_only_plan_update_to_resolve_confirmed_review(
    tmp_path: Path,
) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    row = _write_warning_transcript_inputs(paths, recording_id)

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 1
    original = read_jsonl(paths.manifests / "transcripts.jsonl")[0]
    original_facts = {key: value for key, value in original.items() if key != "pronunciation_plan"}
    events_path = paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    events_before = events_path.read_bytes()
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

    resolved = read_jsonl(paths.manifests / "transcripts.jsonl")[0]
    assert {
        key: value for key, value in resolved.items() if key != "pronunciation_plan"
    } == original_facts
    assert resolved["pronunciation_plan"]["warning_codes"] == []
    assert events_path.read_bytes() == events_before
    assert not _review_path(paths, recording_id).exists()


def test_pronunciation_review_filename_cannot_escape_manifests(tmp_path: Path) -> None:
    recording_id = "rec/../../escaped"
    paths, _ = _set_up_pilot(tmp_path, recording_id)
    _write_warning_transcript_inputs(paths, recording_id)

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 1

    review_path = _review_path(paths, recording_id)
    assert review_path.is_file()
    assert review_path.resolve().parent == paths.manifests.resolve()
    assert read_jsonl(review_path)[0]["recording_id"] == recording_id
    assert not (paths.local_data / "escaped.json").exists()


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


def test_prepare_text_resumes_mixed_legal_recording_states(tmp_path: Path) -> None:
    recording_ids = ("rec-inventoried", "rec-ready", "rec-confirmed")
    paths, records = _set_up_clean_pilot_batch(tmp_path, recording_ids)
    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0
    events_path = paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    full_events = read_jsonl(events_path)
    mixed_records = (
        records[0],
        replace(records[1], state=CorpusState.TEXT_CANDIDATES_READY),
        replace(records[2], state=CorpusState.TRANSCRIPT_CONFIRMED),
    )
    write_jsonl_atomic(
        paths.manifests / "recordings.jsonl",
        (record.to_dict() for record in mixed_records),
    )
    retained_events = tuple(
        event
        for event in full_events
        if event["recording_id"] == recording_ids[2]
        or (
            event["recording_id"] == recording_ids[1]
            and event["target_state"] == "TEXT_CANDIDATES_READY"
        )
    )
    write_jsonl_atomic(events_path, retained_events)

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0

    assert {
        row["recording_id"]: row["state"]
        for row in read_jsonl(paths.manifests / "recordings.jsonl")
    } == {recording_id: "TRANSCRIPT_CONFIRMED" for recording_id in recording_ids}
    events = read_jsonl(events_path)
    assert len(events) == 6
    assert len({event["event_id"] for event in events}) == 6
    assert {
        recording_id: sum(event["recording_id"] == recording_id for event in events)
        for recording_id in recording_ids
    } == {recording_id: 2 for recording_id in recording_ids}


def test_prepare_text_recovers_event_durable_manifest_without_duplicate(
    tmp_path: Path,
) -> None:
    paths, records = _set_up_clean_pilot_batch(tmp_path, ("rec-1",))
    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0
    events_path = paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    events_before = events_path.read_bytes()
    write_jsonl_atomic(
        paths.manifests / "recordings.jsonl",
        (replace(records[0], state=CorpusState.TEXT_CANDIDATES_READY).to_dict(),),
    )

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0

    assert events_path.read_bytes() == events_before
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == ("TRANSCRIPT_CONFIRMED")


def test_prepare_text_rejects_ready_state_without_durable_ready_event(
    tmp_path: Path,
) -> None:
    paths, records = _set_up_clean_pilot_batch(tmp_path, ("rec-1",))
    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0
    recordings_path = paths.manifests / "recordings.jsonl"
    transcripts_path = paths.manifests / "transcripts.jsonl"
    events_path = paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    write_jsonl_atomic(
        recordings_path,
        (replace(records[0], state=CorpusState.TEXT_CANDIDATES_READY).to_dict(),),
    )
    write_jsonl_atomic(events_path, ())
    recordings_before = recordings_path.read_bytes()
    transcripts_before = transcripts_path.read_bytes()
    events_before = events_path.read_bytes()

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2

    assert recordings_path.read_bytes() == recordings_before
    assert transcripts_path.read_bytes() == transcripts_before
    assert events_path.read_bytes() == events_before


def test_prepare_text_rejects_confirmed_state_without_durable_confirmed_event(
    tmp_path: Path,
) -> None:
    paths, _ = _set_up_clean_pilot_batch(tmp_path, ("rec-1",))
    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0
    events_path = paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    ready_event = next(
        event
        for event in read_jsonl(events_path)
        if event["target_state"] == "TEXT_CANDIDATES_READY"
    )
    write_jsonl_atomic(events_path, (ready_event,))
    recordings_before = (paths.manifests / "recordings.jsonl").read_bytes()

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2
    assert (paths.manifests / "recordings.jsonl").read_bytes() == recordings_before
    assert read_jsonl(events_path) == (ready_event,)


def test_prepare_text_rejects_confirmed_event_without_ready_predecessor(
    tmp_path: Path,
) -> None:
    paths, records = _set_up_clean_pilot_batch(tmp_path, ("rec-1",))
    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0
    recordings_path = paths.manifests / "recordings.jsonl"
    events_path = paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    confirmed_event = next(
        event
        for event in read_jsonl(events_path)
        if event["target_state"] == "TRANSCRIPT_CONFIRMED"
    )
    write_jsonl_atomic(recordings_path, (records[0].to_dict(),))
    write_jsonl_atomic(events_path, (confirmed_event,))
    recordings_before = recordings_path.read_bytes()
    events_before = events_path.read_bytes()

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2
    assert recordings_path.read_bytes() == recordings_before
    assert events_path.read_bytes() == events_before


def test_prepare_text_rejects_existing_transcript_row_count_conflict(
    tmp_path: Path,
) -> None:
    paths, _ = _set_up_clean_pilot_batch(tmp_path, ("rec-1",))
    transcripts_path = paths.manifests / "transcripts.jsonl"
    write_jsonl_atomic(
        transcripts_path,
        ({"recording_id": "one"}, {"recording_id": "unexpected"}),
    )
    original = transcripts_path.read_bytes()

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2
    assert transcripts_path.read_bytes() == original


def test_prepare_text_rejects_resumable_state_without_transcript_registry(
    tmp_path: Path,
) -> None:
    paths, records = _set_up_clean_pilot_batch(tmp_path, ("rec-1",))
    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 0
    (paths.manifests / "transcripts.jsonl").unlink()
    write_jsonl_atomic(
        paths.manifests / "recordings.jsonl",
        (replace(records[0], state=CorpusState.TEXT_CANDIDATES_READY).to_dict(),),
    )

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2
    assert not (paths.manifests / "transcripts.jsonl").exists()


def test_prepare_text_rejects_mismatched_pronunciation_review_template(
    tmp_path: Path,
) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    row = _write_warning_transcript_inputs(paths, recording_id)
    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 1
    review_path = _review_path(paths, recording_id)
    review = dict(read_jsonl(review_path)[0])
    review["recording_id"] = "other-recording"
    write_jsonl_atomic(review_path, (review,))
    (paths.local_data / "text" / "overrides.json").write_text(
        json.dumps(
            {
                "spoken_text_sha256": hashlib.sha256(b"Fabula").hexdigest(),
                "overrides": {
                    "0": {
                        "stress_index": 1,
                        "ipa": None,
                        "model_phonemes": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    row["pronunciation_overrides_file"] = "text/overrides.json"
    write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", (row,))

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2
    assert read_jsonl(review_path)[0]["recording_id"] == "other-recording"


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


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("invalid-json", "invalid JSON"),
        ("json-array", "JSON object"),
        ("duplicate-key", "duplicate key"),
        ("overrides-array", "overrides must be an object"),
        ("override-array", "values must be objects"),
        ("boolean-stress", "stress_index must be an integer"),
        ("numeric-ipa", "ipa must be a string"),
        ("numeric-phoneme", "array of strings"),
    ),
)
def test_prepare_text_rejects_strict_override_json_and_types(
    tmp_path: Path,
    capsys,
    case: str,
    message: str,
) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    row = _write_warning_transcript_inputs(paths, recording_id)
    digest = hashlib.sha256(b"Fabula").hexdigest()
    override_path = paths.local_data / "text" / "overrides.json"
    if case == "invalid-json":
        content = "{"
    elif case == "json-array":
        content = "[]"
    elif case == "duplicate-key":
        content = (
            '{"spoken_text_sha256":"'
            + digest
            + '","spoken_text_sha256":"'
            + digest
            + '","overrides":{}}'
        )
    else:
        overrides: object = {}
        if case == "overrides-array":
            overrides = []
        elif case == "override-array":
            overrides = {"0": []}
        elif case == "boolean-stress":
            overrides = {"0": {"stress_index": True, "ipa": None, "model_phonemes": None}}
        elif case == "numeric-ipa":
            overrides = {"0": {"stress_index": 1, "ipa": 7, "model_phonemes": None}}
        elif case == "numeric-phoneme":
            overrides = {"0": {"stress_index": 1, "ipa": None, "model_phonemes": [7]}}
        content = json.dumps({"spoken_text_sha256": digest, "overrides": overrides})
    override_path.write_text(content, encoding="utf-8")
    row["pronunciation_overrides_file"] = "text/overrides.json"
    write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", (row,))

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("no-candidates", "at least one candidate"),
        ("no-selection", "exactly one selected=true"),
        ("two-selections", "exactly one selected=true"),
        ("missing-source", "does not identify a file"),
        ("empty-source-id", "fields must not be empty"),
        ("bad-access-date", "ISO date"),
        ("empty-units", "at least one non-empty unit"),
        ("empty-asr", "must not be empty"),
    ),
)
def test_prepare_text_rejects_invalid_candidate_and_text_inputs(
    tmp_path: Path,
    capsys,
    case: str,
    message: str,
) -> None:
    paths, recording_id = _set_up_pilot(tmp_path)
    text_dir = paths.local_data / "text"
    text_dir.mkdir()
    (text_dir / "source-1.txt").write_text("Pater noster", encoding="utf-8")
    (text_dir / "spoken-units.txt").write_text("Pater noster\n", encoding="utf-8")
    row = _intake_row(recording_id)
    candidates = [dict(row["source_candidates"][0])]  # type: ignore[index]
    if case == "no-candidates":
        candidates = []
    elif case == "no-selection":
        candidates[0]["selected"] = False
    elif case == "two-selections":
        second = dict(candidates[0])
        second["source_id"] = "source-2"
        second["source_version"] = "edition-2"
        candidates.append(second)
    elif case == "missing-source":
        candidates[0]["source_file"] = "text/missing.txt"
    elif case == "empty-source-id":
        candidates[0]["source_id"] = ""
    elif case == "bad-access-date":
        candidates[0]["accessed_at"] = "19-07-2026"
    elif case == "empty-units":
        (text_dir / "spoken-units.txt").write_text("\n", encoding="utf-8")
    elif case == "empty-asr":
        (text_dir / "asr.txt").write_text("\n", encoding="utf-8")
        row["asr_hypothesis_file"] = "text/asr.txt"
    row["source_candidates"] = candidates
    write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", (row,))

    assert main(["--project-root", str(tmp_path), "prepare-text"]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "case",
    (
        "candidate-not-object",
        "candidate-field-not-string",
        "selected-not-bool",
        "candidates-not-array",
        "path-not-string",
        "confirmed-not-bool",
    ),
)
def test_transcript_intake_records_reject_wrong_json_types(case: str) -> None:
    raw = _intake_row("rec-1")
    candidates = [dict(raw["source_candidates"][0])]  # type: ignore[index]
    raw["source_candidates"] = candidates
    if case == "candidate-not-object":
        raw["source_candidates"] = ["candidate"]
    elif case == "candidate-field-not-string":
        candidates[0]["source_id"] = 7
    elif case == "selected-not-bool":
        candidates[0]["selected"] = 1
    elif case == "candidates-not-array":
        raw["source_candidates"] = {}
    elif case == "path-not-string":
        raw["spoken_units_file"] = 7
    elif case == "confirmed-not-bool":
        raw["confirmed"] = 1

    with pytest.raises((TypeError, ValueError)):
        TranscriptIntakeRow.from_dict(raw)
