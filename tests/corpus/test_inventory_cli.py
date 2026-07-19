import csv
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from latintts.corpus.cli import main
from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.inventory import inventory, inventory_from_manifests, write_intake_skeleton
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.store import read_jsonl, write_jsonl_atomic


def _fake_probe(command: list[str], **kwargs: object) -> CompletedProcess[str]:
    payload = {
        "format": {"duration": "3.0", "bit_rate": "256000"},
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "sample_rate": "48000",
                "channels": 1,
            }
        ],
    }
    return CompletedProcess(command, 0, json.dumps(payload), "")


def _write_completed_intake(paths: CorpusPaths, rows: list[list[str]]) -> None:
    with (paths.manifests / "intake.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["relative_path", "title_or_citation", "speaker_id", "rights_id", "notes"])
        writer.writerows(rows)


def _write_rights(paths: CorpusPaths) -> None:
    write_jsonl_atomic(
        paths.manifests / "rights.jsonl",
        (
            {
                "rights_id": "rights-1",
                "owner_id": "owner-1",
                "speaker_id": "speaker-1",
                "allow_local_processing": True,
                "allow_model_training": True,
                "allow_internal_evaluation": True,
                "allow_raw_release": "unknown",
                "allow_segment_release": "unknown",
                "allow_model_release": "unknown",
                "authorized_at": "2026-07-19",
                "basis": "written consent",
                "notes": "",
            },
        ),
    )


def test_intake_skeleton_lists_spoken_and_sung_without_overwriting(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    nested = paths.raw_spoken / "nested"
    nested.mkdir()
    (nested / "z.wav").write_bytes(b"a")
    (paths.raw_spoken / "spoken.wav").write_bytes(b"b")
    (paths.raw_sung / "chant.flac").write_bytes(b"c")
    (paths.raw_sung / "ignore.txt").write_text("not audio", encoding="utf-8")
    output = paths.manifests / "intake.csv"

    assert write_intake_skeleton(paths, output) is True

    original = output.read_bytes()
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "relative_path": "raw/spoken/nested/z.wav",
            "title_or_citation": "",
            "speaker_id": "",
            "rights_id": "",
            "notes": "",
        },
        {
            "relative_path": "raw/spoken/spoken.wav",
            "title_or_citation": "",
            "speaker_id": "",
            "rights_id": "",
            "notes": "",
        },
        {
            "relative_path": "raw/sung/chant.flac",
            "title_or_citation": "",
            "speaker_id": "",
            "rights_id": "",
            "notes": "",
        },
    ]
    assert write_intake_skeleton(paths, output) is False
    assert output.read_bytes() == original


def test_inventory_from_manifests_persists_every_transition_and_content_type(
    tmp_path: Path,
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "speech.wav").write_bytes(b"spoken bytes")
    (paths.raw_sung / "chant.flac").write_bytes(b"sung bytes")
    _write_completed_intake(
        paths,
        [
            ["raw/sung/chant.flac", "Chant", "speaker-1", "rights-1", "sung"],
            ["raw/spoken/speech.wav", "Speech", "speaker-1", "rights-1", "spoken"],
        ],
    )
    _write_rights(paths)

    records = inventory_from_manifests(
        paths,
        run_command=_fake_probe,
        timestamp="2026-07-19T12:00:00+00:00",
    )

    assert [record.relative_path for record in records] == [
        "raw/spoken/speech.wav",
        "raw/sung/chant.flac",
    ]
    assert [record.content_type for record in records] == ["spoken", "sung"]
    assert all(record.state is CorpusState.INVENTORIED for record in records)
    assert read_jsonl(paths.manifests / "recordings.jsonl") == tuple(
        record.to_dict() for record in records
    )
    events = read_jsonl(paths.alignments / "runs" / "inventory" / "processing-events.jsonl")
    assert len(events) == 2
    assert {event["recording_id"] for event in events} == {
        record.recording_id for record in records
    }
    assert all(event["previous_state"] == "DISCOVERED" for event in events)
    assert all(event["target_state"] == "INVENTORIED" for event in events)


def test_inventory_from_manifests_rejects_duplicate_recording_ids_before_write(
    tmp_path: Path,
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "one.wav").write_bytes(b"same bytes")
    (paths.raw_spoken / "two.wav").write_bytes(b"same bytes")
    _write_completed_intake(
        paths,
        [
            ["raw/spoken/one.wav", "One", "speaker-1", "rights-1", ""],
            ["raw/spoken/two.wav", "Two", "speaker-1", "rights-1", ""],
        ],
    )
    _write_rights(paths)

    with pytest.raises(ValueError, match="duplicate recording_id"):
        inventory_from_manifests(paths, run_command=_fake_probe)

    assert not (paths.manifests / "recordings.jsonl").exists()


def test_inventory_from_manifests_requires_every_rights_id_before_probe(
    tmp_path: Path,
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "speech.wav").write_bytes(b"speech")
    _write_completed_intake(
        paths,
        [["raw/spoken/speech.wav", "Speech", "speaker-1", "missing-rights", ""]],
    )
    _write_rights(paths)
    calls = 0

    def fail_if_called(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return _fake_probe(command, **kwargs)

    with pytest.raises(ValueError, match="unknown rights_id"):
        inventory_from_manifests(paths, run_command=fail_if_called)

    assert calls == 0
    assert not (paths.manifests / "recordings.jsonl").exists()


def test_inventory_cli_init_creates_only_intake_skeleton(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "speech.wav").write_bytes(b"speech")

    assert main(["--project-root", str(tmp_path), "inventory", "--init-intake"]) == 0

    assert (paths.manifests / "intake.csv").exists()
    assert not (paths.manifests / "recordings.jsonl").exists()


def test_inventory_cli_runs_inventory_from_manifests(tmp_path: Path, monkeypatch) -> None:
    received: list[CorpusPaths] = []

    def fake_inventory(paths: CorpusPaths) -> tuple[()]:
        received.append(paths)
        return ()

    monkeypatch.setattr("latintts.corpus.cli.inventory_from_manifests", fake_inventory)

    assert main(["--project-root", str(tmp_path), "inventory"]) == 0
    assert received == [CorpusPaths.from_project_root(tmp_path)]


def test_inventory_from_manifests_is_idempotent_for_unchanged_records(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "speech.wav").write_bytes(b"speech")
    _write_completed_intake(
        paths,
        [["raw/spoken/speech.wav", "Speech", "speaker-1", "rights-1", ""]],
    )
    _write_rights(paths)
    event_path = paths.alignments / "runs" / "inventory" / "processing-events.jsonl"

    first = inventory_from_manifests(
        paths,
        run_command=_fake_probe,
        timestamp="2026-07-19T12:00:00+00:00",
    )
    first_events = read_jsonl(event_path)
    second = inventory_from_manifests(
        paths,
        run_command=_fake_probe,
        timestamp="2026-07-19T13:00:00+00:00",
    )

    assert second == first
    assert read_jsonl(event_path) == first_events


def test_inventory_public_entry_point_is_manifest_inventory() -> None:
    assert inventory is inventory_from_manifests


def test_inventory_from_manifests_rejects_missing_transition_event_without_writes(
    tmp_path: Path,
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "speech.wav").write_bytes(b"speech")
    _write_completed_intake(
        paths,
        [["raw/spoken/speech.wav", "Speech", "speaker-1", "rights-1", ""]],
    )
    _write_rights(paths)
    event_path = paths.alignments / "runs" / "inventory" / "processing-events.jsonl"
    inventory_from_manifests(paths, run_command=_fake_probe)
    write_jsonl_atomic(event_path, ())
    recordings_path = paths.manifests / "recordings.jsonl"
    recordings_before = recordings_path.read_bytes()
    events_before = event_path.read_bytes()

    with pytest.raises(ValueError, match="existing inventory is inconsistent"):
        inventory_from_manifests(paths, run_command=_fake_probe)

    assert recordings_path.read_bytes() == recordings_before
    assert event_path.read_bytes() == events_before


def test_inventory_from_manifests_rejects_partial_state_without_writes(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "speech.wav").write_bytes(b"speech")
    _write_completed_intake(
        paths,
        [["raw/spoken/speech.wav", "Speech", "speaker-1", "rights-1", ""]],
    )
    _write_rights(paths)
    inventory_from_manifests(paths, run_command=_fake_probe)
    recordings_path = paths.manifests / "recordings.jsonl"
    event_path = paths.alignments / "runs" / "inventory" / "processing-events.jsonl"
    partial = dict(read_jsonl(recordings_path)[0])
    partial["state"] = "DISCOVERED"
    write_jsonl_atomic(recordings_path, (partial,))
    recordings_before = recordings_path.read_bytes()
    events_before = event_path.read_bytes()

    with pytest.raises(ValueError, match="existing inventory is inconsistent"):
        inventory_from_manifests(paths, run_command=_fake_probe)

    assert recordings_path.read_bytes() == recordings_before
    assert event_path.read_bytes() == events_before


def test_inventory_from_manifests_rejects_changed_audio_without_writes(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    audio = paths.raw_spoken / "speech.wav"
    audio.write_bytes(b"speech")
    _write_completed_intake(
        paths,
        [["raw/spoken/speech.wav", "Speech", "speaker-1", "rights-1", ""]],
    )
    _write_rights(paths)
    inventory_from_manifests(paths, run_command=_fake_probe)
    recordings_path = paths.manifests / "recordings.jsonl"
    event_path = paths.alignments / "runs" / "inventory" / "processing-events.jsonl"
    recordings_before = recordings_path.read_bytes()
    events_before = event_path.read_bytes()
    audio.write_bytes(b"changed speech")

    with pytest.raises(ValueError, match="existing inventory is inconsistent"):
        inventory_from_manifests(paths, run_command=_fake_probe)

    assert recordings_path.read_bytes() == recordings_before
    assert event_path.read_bytes() == events_before


def test_inventory_from_manifests_rejects_duplicate_transition_event_without_writes(
    tmp_path: Path,
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "speech.wav").write_bytes(b"speech")
    _write_completed_intake(
        paths,
        [["raw/spoken/speech.wav", "Speech", "speaker-1", "rights-1", ""]],
    )
    _write_rights(paths)
    inventory_from_manifests(paths, run_command=_fake_probe)
    recordings_path = paths.manifests / "recordings.jsonl"
    event_path = paths.alignments / "runs" / "inventory" / "processing-events.jsonl"
    event = read_jsonl(event_path)[0]
    write_jsonl_atomic(event_path, (event, event))
    recordings_before = recordings_path.read_bytes()
    events_before = event_path.read_bytes()

    with pytest.raises(ValueError, match="existing inventory is inconsistent"):
        inventory_from_manifests(paths, run_command=_fake_probe)

    assert recordings_path.read_bytes() == recordings_before
    assert event_path.read_bytes() == events_before


def test_inventory_cli_reports_corpus_failure_without_traceback(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    def unavailable(paths: CorpusPaths) -> tuple[()]:
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "ffprobe unavailable")

    monkeypatch.setattr("latintts.corpus.cli.inventory_from_manifests", unavailable)

    assert main(["--project-root", str(tmp_path), "inventory"]) == 1
    assert capsys.readouterr().err == "ALIGNER_UNAVAILABLE: ffprobe unavailable\n"


def test_inventory_cli_reports_missing_manifest_as_input_error(tmp_path: Path, capsys) -> None:
    assert main(["--project-root", str(tmp_path), "inventory"]) == 2
    assert capsys.readouterr().err == (
        "MANIFEST_SCHEMA_MISMATCH: missing required manifest: intake.csv\n"
    )


def test_inventory_cli_reports_strict_manifest_error(tmp_path: Path, capsys) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.manifests / "intake.csv").write_text("wrong,header\n", encoding="utf-8")

    assert main(["--project-root", str(tmp_path), "inventory"]) == 2
    assert capsys.readouterr().err == (
        "MANIFEST_SCHEMA_MISMATCH: intake.csv must use the exact header\n"
    )


def test_inventory_cli_does_not_swallow_programming_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def broken(paths: CorpusPaths) -> tuple[()]:
        raise RuntimeError("programming bug")

    monkeypatch.setattr("latintts.corpus.cli.inventory_from_manifests", broken)

    with pytest.raises(RuntimeError, match="programming bug"):
        main(["--project-root", str(tmp_path), "inventory"])


def test_empty_inventory_rejects_unexpected_event_without_writes(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    _write_completed_intake(paths, [])
    write_jsonl_atomic(paths.manifests / "rights.jsonl", ())
    inventory_from_manifests(paths, run_command=_fake_probe)
    recordings_path = paths.manifests / "recordings.jsonl"
    event_path = paths.alignments / "runs" / "inventory" / "processing-events.jsonl"
    write_jsonl_atomic(event_path, ({"unexpected": "event"},))
    recordings_before = recordings_path.read_bytes()
    events_before = event_path.read_bytes()

    with pytest.raises(ValueError, match="existing inventory is inconsistent"):
        inventory_from_manifests(paths, run_command=_fake_probe)

    assert recordings_path.read_bytes() == recordings_before
    assert event_path.read_bytes() == events_before
