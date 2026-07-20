from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from latintts.corpus import audio as audio_module
from latintts.corpus import cli
from latintts.corpus import manifest as manifest_module
from latintts.corpus import review as review_module
from latintts.corpus import store as store_module
from latintts.corpus.alignment import result_from_dict
from latintts.corpus.audio import DerivedAudio, PcmMetrics
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.manifest import build_approved_segments, build_manifest_corpus
from latintts.corpus.pairing import pairing_from_dict
from latintts.corpus.records import ReviewEvent, RightsRecord
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from tests.corpus.test_pairing import _audio_command
from tests.corpus.test_review_cli import (
    _aligned_project,
    _corrected_pairing_project,
    _set_decisions,
    _two_recording_aligned_project,
    export_review_bundle,
    import_review_bundle,
)


def test_build_manifest_cli_loads_config_and_dispatches(
    tmp_path: Path, monkeypatch: object
) -> None:
    config = tmp_path / "config" / "corpus" / "pilot-v1.json"
    config.parent.mkdir(parents=True)
    source = Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json"
    config.write_bytes(source.read_bytes())
    calls: list[tuple[Path, str]] = []

    def build(paths: object, corpus_config: object, *, ffmpeg_version: str) -> bool:
        calls.append((paths.project_root, corpus_config.raw["corpus_version"]))  # type: ignore[attr-defined]
        assert ffmpeg_version == "ffmpeg-test-1"
        return True

    monkeypatch.setattr(cli, "build_manifest_corpus", build, raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "_ffmpeg_version", lambda: "ffmpeg-test-1")  # type: ignore[attr-defined]

    assert (
        cli.main(
            [
                "--project-root",
                str(tmp_path),
                "build-manifest",
                "--config",
                "config/corpus/pilot-v1.json",
            ]
        )
        == 0
    )
    assert calls == [(tmp_path, "corpus-v1")]


@pytest.mark.parametrize(
    ("failure", "expected"), [(CorpusFailure("REVIEW_REQUIRED", "x"), 1), (ValueError("bad"), 2)]
)
def test_build_manifest_cli_maps_stable_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
    expected: int,
) -> None:
    config = tmp_path / "config" / "corpus" / "pilot-v1.json"
    config.parent.mkdir(parents=True)
    source = Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json"
    config.write_bytes(source.read_bytes())
    monkeypatch.setattr(
        cli,
        "build_manifest_corpus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(cli, "_ffmpeg_version", lambda: "ffmpeg-test-1")

    assert cli.main(["--project-root", str(tmp_path), "build-manifest"]) == expected
    assert ("REVIEW_REQUIRED" if expected == 1 else "MANIFEST_SCHEMA_MISMATCH") in (
        capsys.readouterr().err
    )


def _write_rights(paths: object) -> None:
    write_jsonl_atomic(
        paths.manifests / "rights.jsonl",  # type: ignore[attr-defined]
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
                "basis": "synthetic fixture",
                "notes": "",
            },
        ),
    )


def _reviewed_project(tmp_path: Path, *, decision: str = "approved") -> tuple[object, object]:
    paths, config = _aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group, decision=decision)
    assert import_review_bundle(paths, config)
    return paths, config


def test_build_manifest_extracts_approved_source_clips_and_advances_state(
    tmp_path: Path,
) -> None:
    paths, config = _aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    sample_counts: dict[Path, int] = {}

    def transcode(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        output = Path(command[-1])
        if output.suffix == ".flac":
            start = float(command[command.index("-ss") + 1])
            end = float(command[command.index("-to") + 1])
            sample_counts[output] = round((end - start) * 48_000)
            output.write_bytes(b"fLaCsynthetic" + str(sample_counts[output]).encode("ascii"))
            return CompletedProcess(command, 0, "", "")
        return _audio_command(command, **kwargs)

    def probe(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        samples = sample_counts[Path(command[-1])]
        payload = {
            "format": {"duration": str(samples / 48_000)},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "flac",
                    "sample_rate": "48000",
                    "channels": 1,
                    "duration_ts": str(samples),
                    "time_base": "1/48000",
                }
            ],
        }
        return CompletedProcess(command, 0, json.dumps(payload), "")

    assert build_manifest_corpus(
        paths,
        config,
        ffmpeg_version="ffmpeg-test-1",
        run_command=transcode,
        probe_command=probe,
        decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
    )

    rows = read_jsonl(paths.manifests / "segments.jsonl")
    assert len(rows) == 6
    assert [(row["text_unit_id"], row["take_index"]) for row in rows] == [
        (f"rec-1-unit-{unit:04d}", take) for unit in range(1, 4) for take in (1, 2)
    ]
    assert all(row["phoneme_timing_status"] == "not_estimated" for row in rows)
    assert all(row["review_event_ids"] for row in rows)
    assert rows[0]["spoken_text"].split()[0] == rows[0]["word_spans"][0]["text"]
    assert all(row["quality_metrics"]["sample_count"] > 0 for row in rows)
    assert all(row["quality_metric_audio_sha256"] != row["derived_audio_sha256"] for row in rows)
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "APPROVED"
    events = read_jsonl(paths.alignments / "runs" / config.digest / "processing-events.jsonl")
    assert events[-1]["previous_state"] == "REVIEWED"
    assert events[-1]["target_state"] == "APPROVED"
    assert build_manifest_corpus(
        paths,
        config,
        ffmpeg_version="ffmpeg-test-1",
        run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal rerun must validate existing clips without extraction")
        ),
    )

    rights_path = paths.manifests / "rights.jsonl"
    rights_rows = read_jsonl(rights_path)
    revoked = dict(rights_rows[0])
    revoked["allow_model_training"] = False
    write_jsonl_atomic(rights_path, (revoked,))
    with pytest.raises(CorpusFailure) as revoked_error:
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert revoked_error.value.code == "RIGHTS_SCOPE_UNCONFIRMED"
    write_jsonl_atomic(rights_path, rights_rows)

    manifest_path = paths.manifests / "segments.jsonl"
    original_manifest = read_jsonl(manifest_path)
    changed_manifest = [dict(row) for row in original_manifest]
    changed_manifest[0]["spoken_text"] = "stale terminal text"
    write_jsonl_atomic(manifest_path, changed_manifest)
    with pytest.raises(ValueError, match=r"manifest|segment|expected"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    write_jsonl_atomic(manifest_path, original_manifest)

    automatic_path = next(
        (paths.alignments / "runs" / config.digest / "rec-1" / "review").glob(  # type: ignore[attr-defined]
            "*/automatic.json"
        )
    )
    automatic_bytes = automatic_path.read_bytes()
    automatic = json.loads(automatic_bytes)
    automatic["takes"][0]["automatic_values"]["segment_end"] -= 0.01
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")
    with pytest.raises(ValueError, match=r"automatic|review|stale"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    automatic_path.write_bytes(automatic_bytes)

    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    terminal_events = read_jsonl(processing_path)
    write_jsonl_atomic(processing_path, (*terminal_events, terminal_events[-1]))
    with pytest.raises(ValueError, match=r"duplicate|history"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    write_jsonl_atomic(processing_path, terminal_events)

    clip = paths.resolve_local(rows[0]["derived_audio_relative_path"])
    clip.write_bytes(b"tampered final cache")
    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_build_manifest_rejects_tampered_raw_before_writing_any_final_clip(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    (paths.raw_spoken / "rec-1.wav").write_bytes(b"tampered after inventory")  # type: ignore[attr-defined]

    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("audio extraction must not start")
            ),
        )

    assert error.value.code == "INVENTORY_HASH_MISMATCH"
    assert not tuple((paths.segments / "lossless").glob("*.flac"))  # type: ignore[attr-defined]


@pytest.mark.parametrize("failure_point", ("later-take", "later-recording", "quality"))
def test_build_manifest_stages_all_audio_before_publishing_final_clips(
    tmp_path: Path, failure_point: str
) -> None:
    if failure_point == "later-recording":
        paths, config = _two_recording_aligned_project(tmp_path)
        _write_rights(paths)
        groups = export_review_bundle(paths, config)
        for group in groups:
            _set_decisions(group)
        assert import_review_bundle(paths, config)
    else:
        paths, config = _reviewed_project(tmp_path)
    before_lossless = set((paths.segments / "lossless").rglob("*.flac"))  # type: ignore[attr-defined]
    before_candidates = set((paths.segments / "candidates").rglob("*.wav"))  # type: ignore[attr-defined]
    sample_counts: dict[Path, int] = {}
    lossless_calls = 0

    def transcode(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        nonlocal lossless_calls
        output = Path(command[-1])
        if output.suffix == ".flac":
            lossless_calls += 1
            if (failure_point == "later-take" and lossless_calls == 2) or (
                failure_point == "later-recording" and lossless_calls == 7
            ):
                raise RuntimeError(f"injected {failure_point} extraction failure")
            start = float(command[command.index("-ss") + 1])
            end = float(command[command.index("-to") + 1])
            sample_counts[output] = round((end - start) * 48_000)
            output.write_bytes(b"fLaCsynthetic" + str(sample_counts[output]).encode("ascii"))
            return CompletedProcess(command, 0, "", "")
        if failure_point == "quality" and ".manifest-staging" in output.parts:
            raise RuntimeError("injected quality extraction failure")
        return _audio_command(command, **kwargs)

    def probe(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        target = Path(command[-1])
        samples = sample_counts.get(target)
        if samples is None:
            samples = int(target.read_bytes().removeprefix(b"fLaCsynthetic"))
        return CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "format": {"duration": str(samples / 48_000)},
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "flac",
                            "sample_rate": "48000",
                            "channels": 1,
                            "duration_ts": str(samples),
                            "time_base": "1/48000",
                        }
                    ],
                }
            ),
            "",
        )

    with pytest.raises(RuntimeError, match=failure_point):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=transcode,
            probe_command=probe,
            decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
        )

    assert set((paths.segments / "lossless").rglob("*.flac")) == before_lossless  # type: ignore[attr-defined]
    assert set((paths.segments / "candidates").rglob("*.wav")) == before_candidates  # type: ignore[attr-defined]


def test_build_manifest_recovers_after_manifest_replace_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _reviewed_project(tmp_path)
    sample_counts: dict[Path, int] = {}

    def transcode(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        output = Path(command[-1])
        if output.suffix == ".flac":
            start = float(command[command.index("-ss") + 1])
            end = float(command[command.index("-to") + 1])
            sample_counts[output] = round((end - start) * 48_000)
            output.write_bytes(b"fLaCsynthetic" + str(sample_counts[output]).encode("ascii"))
            return CompletedProcess(command, 0, "", "")
        return _audio_command(command, **kwargs)

    def probe(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        target = Path(command[-1])
        samples = sample_counts.get(target)
        if samples is None:
            samples = int(target.read_bytes().removeprefix(b"fLaCsynthetic"))
        return CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "format": {"duration": str(samples / 48_000)},
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "flac",
                            "sample_rate": "48000",
                            "channels": 1,
                            "duration_ts": str(samples),
                            "time_base": "1/48000",
                        }
                    ],
                }
            ),
            "",
        )

    real_write = manifest_module.write_jsonl_atomic

    def fail_manifest(path: Path, rows: object) -> None:
        if path == paths.manifests / "segments.jsonl":  # type: ignore[attr-defined]
            raise OSError("injected manifest replace failure")
        real_write(path, rows)  # type: ignore[arg-type]

    monkeypatch.setattr(manifest_module, "write_jsonl_atomic", fail_manifest)
    with pytest.raises(OSError, match="manifest replace"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=transcode,
            probe_command=probe,
            decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
        )
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "REVIEWED"  # type: ignore[attr-defined]
    monkeypatch.setattr(manifest_module, "write_jsonl_atomic", real_write)

    assert build_manifest_corpus(
        paths,  # type: ignore[arg-type]
        config,  # type: ignore[arg-type]
        ffmpeg_version="ffmpeg-test-1",
        run_command=transcode,
        probe_command=probe,
        decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
    )
    events = read_jsonl(
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    event_ids = [row["event_id"] for row in events]
    assert len(event_ids) == len(set(event_ids))


@pytest.mark.parametrize("checkpoint", ("events", "recordings"))
def test_build_manifest_recovers_batch_commit_checkpoints_without_duplicate_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group, decision="rejected", reason="fixture rejection")
    assert import_review_bundle(paths, config)
    recordings_path = paths.manifests / "recordings.jsonl"
    events_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    before_events = read_jsonl(events_path)
    real_replace = store_module.os.replace

    def fail_checkpoint(source: Path, destination: Path) -> None:
        target = events_path if checkpoint == "events" else recordings_path
        if Path(destination) == target:
            raise OSError(f"injected {checkpoint} replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(store_module.os, "replace", fail_checkpoint)
    expected_error = OSError if checkpoint == "events" else RuntimeError
    expected_message = checkpoint if checkpoint == "events" else "recovery required"
    with pytest.raises(expected_error, match=expected_message):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert all(row["state"] == "REVIEWED" for row in read_jsonl(recordings_path))
    if checkpoint == "events":
        assert read_jsonl(events_path) == before_events
    monkeypatch.setattr(store_module.os, "replace", real_replace)

    assert build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    terminal_events = read_jsonl(events_path)
    event_ids = [row["event_id"] for row in terminal_events]
    assert len(event_ids) == len(set(event_ids))
    assert all(row["state"] == "REJECTED" for row in read_jsonl(recordings_path))

    mixed = list(read_jsonl(recordings_path))
    mixed[0]["state"] = "REVIEWED"
    write_jsonl_atomic(recordings_path, mixed)
    assert build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert read_jsonl(events_path) == terminal_events
    assert all(row["state"] == "REJECTED" for row in read_jsonl(recordings_path))


def test_build_manifest_revalidates_review_snapshot_instead_of_trusting_events(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    automatic = next(
        (paths.alignments / "runs" / config.digest / "rec-1" / "review").glob(  # type: ignore[attr-defined]
            "*/automatic.json"
        )
    )
    raw = json.loads(automatic.read_text(encoding="utf-8"))
    raw["takes"][0]["automatic_values"]["segment_end"] -= 0.01
    automatic.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=r"automatic|review"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("clip extraction must not start")
            ),
        )


def test_build_manifest_accepts_strictly_bound_human_pairing_correction(
    tmp_path: Path,
) -> None:
    paths, config, backend, _ = _corrected_pairing_project(tmp_path)
    assert cli.pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("corrected pairing must use validated caches")
        ),
    )
    assert cli.align_corpus(paths, config, backend)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group, decision="rejected", reason="not selected for fixture")
    assert import_review_bundle(paths, config)
    _write_rights(paths)

    assert build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert read_jsonl(paths.manifests / "segments.jsonl") == ()
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "REJECTED"


def test_build_manifest_rejects_warning_in_approved_unit_before_extraction(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    transcript_path = paths.manifests / "transcripts.jsonl"  # type: ignore[attr-defined]
    transcript = read_jsonl(transcript_path)[0]
    transcript["pronunciation_plan"]["tokens"][0]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    transcript["pronunciation_plan"]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    write_jsonl_atomic(transcript_path, (transcript,))

    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("clip extraction must not start")
            ),
        )

    assert error.value.code == "PRONUNCIATION_NEEDS_REVIEW"


def test_build_manifest_preflights_all_units_before_any_final_clip(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    transcript_path = paths.manifests / "transcripts.jsonl"  # type: ignore[attr-defined]
    transcript = read_jsonl(transcript_path)[0]
    transcript["pronunciation_plan"]["tokens"][-1]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    transcript["pronunciation_plan"]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    write_jsonl_atomic(transcript_path, (transcript,))

    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("all unit inputs must be validated before extraction")
            ),
        )

    assert error.value.code == "PRONUNCIATION_NEEDS_REVIEW"
    assert not tuple((paths.segments / "lossless").glob("*.flac"))  # type: ignore[attr-defined]


def test_build_manifest_preflights_every_recording_before_any_final_clip(
    tmp_path: Path,
) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    transcript_path = paths.manifests / "transcripts.jsonl"
    transcripts = list(read_jsonl(transcript_path))
    transcripts[1]["pronunciation_plan"]["tokens"][-1]["warning_codes"] = [
        "PRONUNCIATION_NEEDS_REVIEW"
    ]
    transcripts[1]["pronunciation_plan"]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    write_jsonl_atomic(transcript_path, transcripts)

    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(
            paths,
            config,
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("all recordings must be preflighted before extraction")
            ),
        )

    assert error.value.code == "PRONUNCIATION_NEEDS_REVIEW"
    assert not tuple((paths.segments / "lossless").glob("*.flac"))


def test_build_manifest_rejects_duplicate_transcript_source_identity(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    transcript_path = paths.manifests / "transcripts.jsonl"  # type: ignore[attr-defined]
    transcript = read_jsonl(transcript_path)[0]
    transcript["source_candidates"].append(dict(transcript["source_candidates"][0]))
    write_jsonl_atomic(transcript_path, (transcript,))

    with pytest.raises(ValueError, match=r"source candidate|source_candidates"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("clip extraction must not start")
            ),
        )


@pytest.mark.parametrize("case", ("types", "version", "ffmpeg"))
def test_build_manifest_rejects_invalid_api_contract(tmp_path: Path, case: str) -> None:
    paths = cli.CorpusPaths.from_project_root(tmp_path)
    config = CorpusConfig({"schema_version": "1", "corpus_version": "corpus-v1"}, "a" * 64)
    with pytest.raises((CorpusFailure, TypeError, ValueError)):
        if case == "types":
            build_manifest_corpus(object(), config, ffmpeg_version="test")  # type: ignore[arg-type]
        elif case == "version":
            bad = CorpusConfig({"schema_version": "2", "corpus_version": "corpus-v2"}, "a" * 64)
            build_manifest_corpus(paths, bad, ffmpeg_version="test")
        else:
            build_manifest_corpus(paths, config, ffmpeg_version="")


def test_build_manifest_rejects_duplicate_rights_and_processing_identities(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    rights_path = paths.manifests / "rights.jsonl"  # type: ignore[attr-defined]
    rights = read_jsonl(rights_path)
    write_jsonl_atomic(rights_path, (*rights, rights[0]))
    with pytest.raises(ValueError, match="duplicate rights_id"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")  # type: ignore[arg-type]

    write_jsonl_atomic(rights_path, rights)
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    events = read_jsonl(processing_path)
    write_jsonl_atomic(processing_path, (*events, events[-1]))
    with pytest.raises(ValueError, match="duplicate identities"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("processing preflight must precede extraction")
            ),
        )


def _direct_builder_inputs(paths: object, config: CorpusConfig) -> dict[str, object]:
    recording = review_module._decode_recording(
        read_jsonl(paths.manifests / "recordings.jsonl")[0]  # type: ignore[attr-defined]
    )
    transcript = read_jsonl(paths.manifests / "transcripts.jsonl")[0]  # type: ignore[attr-defined]
    units = cli._decode_spoken_units(transcript, recording.recording_id)
    run_directory = (
        paths.alignments / "runs" / config.digest / recording.recording_id  # type: ignore[attr-defined]
    )
    pairing = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
    alignment = read_jsonl(run_directory / "alignment.json")[0]
    alignments = {
        (row["repetition_group_id"], row["take_index"]): result_from_dict(row["alignment_result"])
        for row in alignment["takes"]
    }
    segmentation = read_jsonl(run_directory / "segmentation.json")[0]
    rights = {
        row["rights_id"]: RightsRecord.from_dict(row)
        for row in read_jsonl(paths.manifests / "rights.jsonl")  # type: ignore[attr-defined]
    }
    events = tuple(
        ReviewEvent.from_dict(row)
        for row in read_jsonl(paths.manifests / "review.jsonl")  # type: ignore[attr-defined]
    )
    return {
        "recording": recording,
        "transcript": transcript,
        "units": units,
        "rights": rights,
        "pairing": pairing,
        "alignments": alignments,
        "review_events": events,
        "analysis_audio": DerivedAudio.from_dict(segmentation["analysis_audio"]),
        "paths": paths,
        "config": config,
        "ffmpeg_version": "ffmpeg-test-1",
        "_validate_only": True,
    }


@pytest.mark.parametrize(
    "case",
    (
        "missing-rights",
        "rights-scope",
        "pairing",
        "analysis",
        "duplicate-unit",
        "duplicate-event",
        "event-id",
        "outcome",
        "unit",
        "alignment",
    ),
)
def test_build_approved_segments_rejects_unbound_inputs(tmp_path: Path, case: str) -> None:
    paths, config = _reviewed_project(tmp_path)
    values = _direct_builder_inputs(paths, config)
    if case == "missing-rights":
        values["rights"] = {}
    elif case == "rights-scope":
        rights = values["rights"]
        assert isinstance(rights, dict)
        item = next(iter(rights.values()))
        rights[item.rights_id] = replace(item, allow_model_training=False)
    elif case == "pairing":
        values["pairing"] = replace(values["pairing"], recording_id="other", integrity_sha256="")  # type: ignore[arg-type]
    elif case == "analysis":
        pairing = values["pairing"]
        values["pairing"] = replace(
            pairing,
            analysis_audio_sha256="0" * 64,
            integrity_sha256="",  # type: ignore[arg-type]
        )
    elif case == "duplicate-unit":
        units = values["units"]
        values["units"] = (units[0], units[0], *units[1:])  # type: ignore[index]
    elif case == "duplicate-event":
        events = values["review_events"]
        values["review_events"] = (*events, events[0])  # type: ignore[misc,index]
    elif case == "event-id":
        events = values["review_events"]
        values["review_events"] = (replace(events[0], review_event_id="review-bad"), *events[1:])  # type: ignore[index]
    elif case == "outcome":
        pairing = values["pairing"]
        first = replace(
            pairing.groups[0], status="review", issue_code="TAKE_COUNT_MISMATCH", group=None
        )  # type: ignore[union-attr]
        values["pairing"] = replace(
            pairing,
            groups=(first, *pairing.groups[1:]),
            integrity_sha256="",  # type: ignore[union-attr]
        )
    elif case == "unit":
        values["units"] = values["units"][1:]  # type: ignore[index]
    else:
        alignments = dict(values["alignments"])  # type: ignore[arg-type]
        alignments.pop(next(iter(alignments)))
        values["alignments"] = alignments
    with pytest.raises((CorpusFailure, TypeError, ValueError)):
        build_approved_segments(**values)  # type: ignore[arg-type]


def test_build_approved_segments_rejects_extractor_provenance_drift(tmp_path: Path) -> None:
    paths, config = _reviewed_project(tmp_path)
    values = _direct_builder_inputs(paths, config)
    values["_validate_only"] = False
    analysis = values["analysis_audio"]
    values["extract_lossless"] = lambda *_args, **_kwargs: analysis
    values["extract_analysis"] = lambda *_args, **_kwargs: analysis

    with pytest.raises(ValueError, match="lossless derived clip provenance"):
        build_approved_segments(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("drift", ("lossless-interval", "quality-interval"))
def test_build_approved_segments_rejects_structurally_valid_wrong_extractor_interval(
    tmp_path: Path, drift: str
) -> None:
    paths, config = _reviewed_project(tmp_path)
    values = _direct_builder_inputs(paths, config)
    values["_validate_only"] = False

    def lossless(
        recording: object,
        _paths: object,
        start: float,
        end: float,
        *,
        ffmpeg_version: str,
        **_kwargs: object,
    ) -> DerivedAudio:
        output_config = audio_module._canonical_output_config(
            {
                "codec": "flac",
                "compression_level": 5,
                "channels": recording.metadata.channels,  # type: ignore[attr-defined]
                "sample_rate": recording.metadata.sample_rate,  # type: ignore[attr-defined]
            }
        )
        if drift == "lossless-interval":
            start += 0.01
        key = audio_module._cache_identity(
            mode="lossless",
            source_relative_path=recording.relative_path,  # type: ignore[attr-defined]
            source_sha256=recording.sha256,  # type: ignore[attr-defined]
            config_sha256=output_config,
            output_config_sha256=output_config,
            ffmpeg_version=ffmpeg_version,
            start_seconds=start,
            end_seconds=end,
        )
        return DerivedAudio(
            "1",
            "lossless",
            audio_module._artifact_relative_path("lossless", key),
            "d" * 64,
            recording.relative_path,  # type: ignore[attr-defined]
            recording.sha256,  # type: ignore[attr-defined]
            output_config,
            output_config,
            ffmpeg_version,
            key,
            start,
            end,
            None,
        )

    def quality(
        analysis: DerivedAudio,
        _paths: object,
        start: float,
        end: float,
        *,
        ffmpeg_version: str,
        **_kwargs: object,
    ) -> DerivedAudio:
        output_config = audio_module._canonical_output_config(
            {"codec": "pcm_s16le", "channels": 1, "sample_rate": 16000}
        )
        if drift == "quality-interval":
            end -= 0.01
        key = audio_module._cache_identity(
            mode="candidate",
            source_relative_path=analysis.relative_path,
            source_sha256=analysis.sha256,
            config_sha256=analysis.config_sha256,
            output_config_sha256=output_config,
            ffmpeg_version=ffmpeg_version,
            start_seconds=start,
            end_seconds=end,
        )
        return DerivedAudio(
            "1",
            "candidate",
            audio_module._artifact_relative_path("candidate", key),
            "e" * 64,
            analysis.relative_path,
            analysis.sha256,
            analysis.config_sha256,
            output_config,
            ffmpeg_version,
            key,
            start,
            end,
            PcmMetrics(100, 0.1, 0.1, 0.0, 0.0),
        )

    values["extract_lossless"] = lossless
    values["extract_analysis"] = quality
    with pytest.raises(ValueError, match=r"provenance|quality|lossless"):
        build_approved_segments(**values)  # type: ignore[arg-type]


def test_build_approved_segments_rejects_reviewed_word_text_stale_against_transcript(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    values = _direct_builder_inputs(paths, config)
    events = values["review_events"]
    assert isinstance(events, tuple)
    first = events[0]
    assert isinstance(first, ReviewEvent)
    alignment = next(iter(values["alignments"].values()))  # type: ignore[union-attr]
    before = alignment.words[0].text
    provisional = ReviewEvent(
        "1",
        "pending",
        first.entity_id,
        "word:0:text",
        before,
        "Mater",
        "manual transcription correction",
        "owner",
        "2026-07-19T11:59:00+08:00",
    )
    correction = replace(
        provisional,
        review_event_id=review_module._event_identity(provisional),
    )
    values["review_events"] = (correction, *events)

    with pytest.raises(ValueError, match=r"stale|text|transcript|pronunciation"):
        build_approved_segments(**values)  # type: ignore[arg-type]
