from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from latintts.corpus import cli
from latintts.corpus import review as review_module
from latintts.corpus.alignment import result_from_dict
from latintts.corpus.audio import DerivedAudio
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
