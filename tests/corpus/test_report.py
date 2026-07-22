from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from latintts.corpus import cli
from latintts.corpus import manifest as manifest_module
from latintts.corpus import report as report_module
from latintts.corpus import review as review_module
from latintts.corpus.cli import align_corpus, pair_corpus
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState, IssueCode
from latintts.corpus.pairing import (
    _pairing_cache_key,
    pairing_from_dict,
    pairing_to_dict,
)
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import (
    AudioMetadata,
    ProcessingEvent,
    RecordingRecord,
    advance_recording,
)
from latintts.corpus.report import (
    PilotMetrics,
    ReportTelemetry,
    build_report,
    classify_scale_readiness,
)
from latintts.corpus.review import read_textgrid, write_textgrid
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from tests.corpus.test_manifest_cli import (
    _build_with_fake_audio,
    _reviewed_project,
    _set_decisions,
    _two_recording_aligned_project,
    _write_rights,
    export_review_bundle,
    import_review_bundle,
)
from tests.corpus.test_pair_cli import _segment, _set_up
from tests.corpus.test_pairing import _audio_command, _FakeAligner
from tests.corpus.test_review_cli import _confirm_pairing_correction


def test_operator_guide_defines_complete_telemetry_and_error_recovery_contract() -> None:
    guide = (Path(__file__).parents[2] / "docs/corpus/alignment-pilot-operator-guide.md").read_text(
        encoding="utf-8"
    )

    assert "正式 `pair` 与 `align` 的全部 forced-alignment 工作负载" in guide
    assert "不能用仅命中 cache 的 `align`" in guide
    assert "停顿单元" in guide
    assert "pairing correction" in guide
    assert "模型权重与 Hugging Face cache" in guide
    assert "lock、临时文件、`report.json` 和 `report.md`" in guide
    assert "`.recovery.`" in guide
    assert "绝对路径" in guide
    for code in IssueCode:
        assert f"`{code.value}`" in guide


def _copy_config(project: Path) -> None:
    destination = project / "config" / "corpus" / "pilot-v1.json"
    destination.parent.mkdir(parents=True)
    source = Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json"
    destination.write_bytes(source.read_bytes())


def _recording_for_report_unit_tests() -> RecordingRecord:
    return RecordingRecord(
        schema_version="1",
        recording_id="rec-unit",
        relative_path="raw/spoken/unit.wav",
        sha256="a" * 64,
        content_type="spoken",
        title_or_citation="Unit fixture",
        speaker_id="speaker-unit",
        rights_id="rights-unit",
        notes="",
        metadata=AudioMetadata(
            duration_seconds=1.0,
            sample_rate=16_000,
            channels=1,
            codec="pcm_s16le",
            bit_rate=256_000,
        ),
        state=CorpusState.APPROVED,
    )


def test_report_cli_loads_config_dispatches_and_prints_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _copy_config(tmp_path)
    calls: list[tuple[Path, str]] = []

    def report(paths: object, config: object) -> object:
        calls.append((paths.project_root, config.raw["corpus_version"]))  # type: ignore[attr-defined]
        decision = type("Decision", (), {"value": "optimize"})()
        return type("Report", (), {"scale_decision": decision})()

    monkeypatch.setattr(cli, "build_report", report, raising=False)

    assert cli.main(["--project-root", str(tmp_path), "report"]) == 0
    assert calls == [(tmp_path, "corpus-v1")]
    assert capsys.readouterr().out == "corpus-report: optimize\n"


@pytest.mark.parametrize(
    ("failure", "exit_code", "message"),
    (
        (CorpusFailure("REVIEW_REQUIRED", "review first"), 1, "REVIEW_REQUIRED: review first\n"),
        (ValueError("bad telemetry"), 2, "MANIFEST_SCHEMA_MISMATCH: bad telemetry\n"),
    ),
)
def test_report_cli_maps_stable_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
    exit_code: int,
    message: str,
) -> None:
    _copy_config(tmp_path)
    monkeypatch.setattr(
        cli,
        "build_report",
        lambda *_args: (_ for _ in ()).throw(failure),
        raising=False,
    )

    assert cli.main(["--project-root", str(tmp_path), "report"]) == exit_code
    assert capsys.readouterr().err == message


def test_scale_readiness_is_scalable_at_all_green_boundaries() -> None:
    metrics = PilotMetrics(
        input_duration_seconds=600.0,
        speech_duration_seconds=500.0,
        auto_pairing_correct_ratio=0.95,
        boundary_unchanged_ratio=0.85,
        review_minutes_per_audio_minute=3.0,
        approved_speech_ratio=0.80,
        gpu_realtime_factor=0.2,
        peak_gpu_memory_bytes=4_000_000_000,
    )

    assert classify_scale_readiness(metrics).value == "scalable"


def test_any_red_metric_makes_batch_processing_not_ready() -> None:
    metrics = PilotMetrics(
        input_duration_seconds=600.0,
        speech_duration_seconds=500.0,
        auto_pairing_correct_ratio=0.84,
        boundary_unchanged_ratio=0.90,
        review_minutes_per_audio_minute=2.0,
        approved_speech_ratio=0.90,
        gpu_realtime_factor=0.2,
        peak_gpu_memory_bytes=4_000_000_000,
    )

    assert classify_scale_readiness(metrics).value == "not_ready"


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"auto_pairing_correct_ratio": 0.85}, "optimize"),
        ({"auto_pairing_correct_ratio": 0.949999}, "optimize"),
        ({"boundary_unchanged_ratio": 0.70}, "optimize"),
        ({"boundary_unchanged_ratio": 0.849999}, "optimize"),
        ({"review_minutes_per_audio_minute": 3.000001}, "optimize"),
        ({"review_minutes_per_audio_minute": 6.0}, "optimize"),
        ({"approved_speech_ratio": 0.60}, "optimize"),
        ({"approved_speech_ratio": 0.799999}, "optimize"),
    ),
)
def test_scale_readiness_uses_exact_yellow_boundaries(
    changes: dict[str, float], expected: str
) -> None:
    values: dict[str, float | int] = {
        "input_duration_seconds": 600.0,
        "speech_duration_seconds": 500.0,
        "auto_pairing_correct_ratio": 0.95,
        "boundary_unchanged_ratio": 0.85,
        "review_minutes_per_audio_minute": 3.0,
        "approved_speech_ratio": 0.80,
        "gpu_realtime_factor": 0.2,
        "peak_gpu_memory_bytes": 4_000_000_000,
    }
    values.update(changes)

    metrics = PilotMetrics(**values)  # type: ignore[arg-type]

    assert classify_scale_readiness(metrics).value == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("input_duration_seconds", 0.0),
        ("input_duration_seconds", True),
        ("speech_duration_seconds", 0.0),
        ("speech_duration_seconds", 601.0),
        ("auto_pairing_correct_ratio", -0.1),
        ("boundary_unchanged_ratio", 1.1),
        ("review_minutes_per_audio_minute", -0.1),
        ("approved_speech_ratio", math.nan),
        ("gpu_realtime_factor", math.inf),
        ("peak_gpu_memory_bytes", 0),
        ("peak_gpu_memory_bytes", True),
    ),
)
def test_pilot_metrics_reject_invalid_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "input_duration_seconds": 600.0,
        "speech_duration_seconds": 500.0,
        "auto_pairing_correct_ratio": 0.95,
        "boundary_unchanged_ratio": 0.85,
        "review_minutes_per_audio_minute": 3.0,
        "approved_speech_ratio": 0.80,
        "gpu_realtime_factor": 0.2,
        "peak_gpu_memory_bytes": 4_000_000_000,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError)):
        PilotMetrics(**values)  # type: ignore[arg-type]


def test_scale_readiness_rejects_non_metrics() -> None:
    with pytest.raises(TypeError, match="PilotMetrics"):
        classify_scale_readiness(object())  # type: ignore[arg-type]


def test_report_public_entrypoint_rejects_invalid_types_and_config(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)

    with pytest.raises(TypeError, match="CorpusPaths and CorpusConfig"):
        build_report(object(), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema 1 corpus-v1"):
        build_report(
            paths,
            CorpusConfig(
                raw={"schema_version": "2", "corpus_version": "corpus-v1"}, digest="a" * 64
            ),
        )


def _telemetry_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "1",
        "text_preparation_seconds": 10.0,
        "review_seconds": 20.0,
        "alignment_wall_seconds": 1.48,
        "gpu_retry_count": 0,
        "peak_gpu_memory_bytes": 4_000_000_000,
        "pilot_storage_bytes": 1_000,
    }
    row.update(changes)
    return row


def _completed_two_recording_project(
    tmp_path: Path, *, decision: str = "approved"
) -> tuple[object, object]:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    for group in export_review_bundle(paths, config):
        _set_decisions(group, decision=decision)
    assert import_review_bundle(paths, config)
    assert _build_with_fake_audio(paths, config)
    return paths, config


def _completed_corrected_pairing_project(tmp_path: Path) -> tuple[object, object]:
    paths, _ = _set_up(tmp_path)
    config_path = tmp_path / "config" / "corpus" / "pilot-v1.json"
    config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    config_raw["pairing"]["minimum_duration_ratio"] = 1.0
    config_raw["pairing"]["maximum_duration_ratio"] = 1.0
    config_path.write_text(json.dumps(config_raw), encoding="utf-8")
    config = CorpusConfig.load(config_path)
    source_one = paths.raw_spoken / "rec-1.wav"
    source_two = paths.raw_spoken / "rec-2.wav"
    shutil.copyfile(source_one, source_two)
    recordings = list(read_jsonl(paths.manifests / "recordings.jsonl"))
    second_recording = deepcopy(recordings[0])
    second_recording.update(
        recording_id="rec-2",
        relative_path="raw/spoken/rec-2.wav",
        title_or_citation="rec-2",
        sha256=hashlib.sha256(source_two.read_bytes()).hexdigest(),
    )
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (*recordings, second_recording))
    transcripts = list(read_jsonl(paths.manifests / "transcripts.jsonl"))
    second_transcript = deepcopy(transcripts[0])
    second_transcript["recording_id"] = "rec-2"
    for unit in second_transcript["spoken_units"]:
        unit["unit_id"] = unit["unit_id"].replace("rec-1-", "rec-2-")
    write_jsonl_atomic(paths.manifests / "transcripts.jsonl", (*transcripts, second_transcript))
    write_jsonl_atomic(
        paths.manifests / "pilot-selection.json",
        (
            {
                "schema_version": "1",
                "strategy": "explicit-v1",
                "recording_ids": ["rec-1", "rec-2"],
                "inventory_hashes": [recordings[0]["sha256"], second_recording["sha256"]],
            },
        ),
    )
    _segment(paths, config)
    backend = _FakeAligner()
    assert not pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    for correction in export_review_bundle(paths, config):
        _confirm_pairing_correction(correction)
    assert import_review_bundle(paths, config)
    assert align_corpus(paths, config, backend)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    _write_rights(paths)
    assert _build_with_fake_audio(paths, config)
    return paths, config


def _first_selected_run_directory(paths: object, config: object) -> Path:
    selection = read_jsonl(paths.manifests / "pilot-selection.json")[0]  # type: ignore[attr-defined]
    recording_id = selection["recording_ids"][0]
    return paths.alignments / "runs" / config.digest / recording_id  # type: ignore[attr-defined]


def _first_review_automatic_path(paths: object, config: object) -> Path:
    review_root = _first_selected_run_directory(paths, config) / "review"
    return next(review_root.glob("*/automatic.json"))


def _first_analysis_audio_path(paths: object, config: object) -> Path:
    segmentation = json.loads(
        (_first_selected_run_directory(paths, config) / "segmentation.json").read_text(
            encoding="utf-8"
        )
    )
    return paths.resolve_local(segmentation["analysis_audio"]["relative_path"])  # type: ignore[attr-defined]


def _first_lossless_audio_path(paths: object) -> Path:
    segment = read_jsonl(paths.manifests / "segments.jsonl")[0]  # type: ignore[attr-defined]
    return paths.resolve_local(segment["derived_audio_relative_path"])  # type: ignore[attr-defined]


def _completed_project_with_unselected_spoken(tmp_path: Path) -> tuple[object, object]:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    for group in export_review_bundle(paths, config):
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    recordings = list(read_jsonl(paths.manifests / "recordings.jsonl"))
    unselected_path = paths.raw_spoken / "rec-unselected.wav"
    unselected_path.write_bytes(b"unselected immutable fixture")
    unselected = deepcopy(recordings[0])
    unselected.update(
        recording_id="rec-unselected",
        relative_path="raw/spoken/rec-unselected.wav",
        sha256=hashlib.sha256(unselected_path.read_bytes()).hexdigest(),
        title_or_citation="unselected full-corpus item",
        state="INVENTORIED",
    )
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (*recordings, unselected))

    assert _build_with_fake_audio(paths, config)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    return paths, config


def test_manifest_build_preserves_unselected_spoken_inventory_for_projection(
    tmp_path: Path,
) -> None:
    paths, config = _completed_project_with_unselected_spoken(tmp_path)

    after = read_jsonl(paths.manifests / "recordings.jsonl")
    assert [row["recording_id"] for row in after] == [
        "rec-1",
        "rec-2",
        "rec-unselected",
    ]
    assert [row["state"] for row in after] == ["APPROVED", "APPROVED", "INVENTORIED"]
    assert len(read_jsonl(paths.manifests / "segments.jsonl")) == 12

    report = build_report(paths, config)
    assert report.metrics.input_duration_seconds == 20.0
    assert report.full_corpus_spoken_duration_seconds == 30.0
    assert report.text_unit_count == 6
    assert report.reading_count == 12
    assert report.projected_full_corpus_person_hours == pytest.approx(45.0 / 3600.0)
    assert report.projected_gpu_hours == pytest.approx(1.48 * 1.5 / 3600.0)
    assert report.projected_storage_bytes == 1_500


@pytest.mark.parametrize("tamper", ("changed", "missing"))
def test_build_report_rejects_selected_raw_drift(tmp_path: Path, tamper: str) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    selected = read_jsonl(paths.manifests / "recordings.jsonl")[0]
    source = paths.resolve_local(selected["relative_path"])
    if tamper == "changed":
        source.write_bytes(source.read_bytes() + b"tampered")
    else:
        source.unlink()

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)

    assert failure.value.code == "INVENTORY_HASH_MISMATCH"


@pytest.mark.parametrize("tamper", ("changed", "missing"))
def test_build_report_rejects_unselected_spoken_raw_drift(tmp_path: Path, tamper: str) -> None:
    paths, config = _completed_project_with_unselected_spoken(tmp_path)
    unselected = read_jsonl(paths.manifests / "recordings.jsonl")[-1]
    source = paths.resolve_local(unselected["relative_path"])
    if tamper == "changed":
        source.write_bytes(source.read_bytes() + b"tampered")
    else:
        source.unlink()

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)

    assert failure.value.code == "INVENTORY_HASH_MISMATCH"


def test_build_report_rejects_unselected_spoken_metadata_drift(tmp_path: Path) -> None:
    paths, config = _completed_project_with_unselected_spoken(tmp_path)
    recordings_path = paths.manifests / "recordings.jsonl"
    recordings = list(read_jsonl(recordings_path))
    recordings[-1]["metadata"]["duration_seconds"] = 1_000.0
    write_jsonl_atomic(recordings_path, recordings)

    with pytest.raises(ValueError, match=r"inventory.*evidence|terminal.*evidence"):
        build_report(paths, config)


@pytest.mark.parametrize("artifact", ("analysis", "lossless"))
@pytest.mark.parametrize("tamper", ("changed", "missing"))
def test_build_report_rejects_missing_or_tampered_derived_audio_without_publishing(
    tmp_path: Path, artifact: str, tamper: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    output_paths = (paths.manifests / "report.json", paths.manifests / "report.md")
    before = tuple(path.read_bytes() for path in output_paths)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    audio_path = (
        _first_analysis_audio_path(paths, config)
        if artifact == "analysis"
        else _first_lossless_audio_path(paths)
    )
    if tamper == "changed":
        audio_path.write_bytes(audio_path.read_bytes() + b"tampered")
    else:
        audio_path.unlink()

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)

    assert failure.value.code == "CACHE_ARTIFACT_INVALID"
    assert tuple(path.read_bytes() for path in output_paths) == before


@pytest.mark.parametrize("artifact", ("analysis", "lossless"))
def test_build_report_rejects_derived_audio_alias_without_publishing(
    tmp_path: Path, artifact: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    build_report(paths, config)
    output_paths = (paths.manifests / "report.json", paths.manifests / "report.md")
    before = tuple(path.read_bytes() for path in output_paths)
    audio_path = (
        _first_analysis_audio_path(paths, config)
        if artifact == "analysis"
        else _first_lossless_audio_path(paths)
    )
    real_path = audio_path.with_name(f"real-{audio_path.name}")
    audio_path.replace(real_path)
    try:
        audio_path.symlink_to(real_path)
    except OSError as error:
        real_path.replace(audio_path)
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)

    assert failure.value.code == "CACHE_ARTIFACT_INVALID"
    assert tuple(path.read_bytes() for path in output_paths) == before


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"unexpected": 1}, "exact fields"),
        ({"schema_version": 1}, "schema_version"),
        ({"text_preparation_seconds": "invalid"}, "text_preparation_seconds"),
        ({"text_preparation_seconds": -1.0}, "text_preparation_seconds"),
        ({"alignment_wall_seconds": math.nan}, "alignment_wall_seconds"),
        ({"gpu_retry_count": True}, "gpu_retry_count"),
        ({"peak_gpu_memory_bytes": 0}, "peak_gpu_memory_bytes"),
        ({"pilot_storage_bytes": 0}, "pilot_storage_bytes"),
    ),
)
def test_report_telemetry_strictly_validates_input(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ReportTelemetry.from_dict(_telemetry_row(**changes))


def test_report_telemetry_allows_zero_review_seconds() -> None:
    telemetry = ReportTelemetry.from_dict(_telemetry_row(review_seconds=0.0))

    assert telemetry.review_seconds == 0.0
    assert telemetry.to_dict() == _telemetry_row(review_seconds=0.0)


def test_report_single_row_and_recording_loaders_reject_empty_files(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    recordings_path = paths.manifests / "recordings.jsonl"
    write_jsonl_atomic(telemetry_path, ())
    write_jsonl_atomic(recordings_path, ())

    with pytest.raises(ValueError, match="exactly one row"):
        report_module._one_row(telemetry_path, "pilot telemetry")
    with pytest.raises(ValueError, match="unique recordings"):
        report_module._load_recordings(paths)
    recording = _recording_for_report_unit_tests().to_dict()
    write_jsonl_atomic(recordings_path, (recording, recording))
    with pytest.raises(ValueError, match="unique recordings"):
        report_module._load_recordings(paths)


@pytest.mark.parametrize(
    ("invalid_part", "message"),
    (
        ("identity", "identity"),
        ("vad", "16 kHz"),
        ("interval-array", "must be an array"),
        ("interval-fields", "exact fields"),
        ("interval-range", "half-open ranges"),
        ("empty", "zero VAD"),
    ),
)
def test_report_vad_interval_validation_rejects_each_invalid_layer(
    invalid_part: str, message: str
) -> None:
    recording = _recording_for_report_unit_tests()
    raw: dict[str, object] = {
        "schema_version": "1",
        "recording_id": recording.recording_id,
        "vad": {
            "sample_rate": 16_000,
            "speech_intervals": [{"start_sample": 0, "end_sample": 8_000}],
        },
    }
    vad = raw["vad"]
    assert type(vad) is dict
    if invalid_part == "identity":
        raw["recording_id"] = "rec-other"
    elif invalid_part == "vad":
        vad["sample_rate"] = 8_000
    elif invalid_part == "interval-array":
        vad["speech_intervals"] = "invalid"
    elif invalid_part == "interval-fields":
        vad["speech_intervals"] = [{"start_sample": 0}]
    elif invalid_part == "interval-range":
        vad["speech_intervals"] = [{"start_sample": 8_000, "end_sample": 8_000}]
    else:
        vad["speech_intervals"] = []

    with pytest.raises((TypeError, ValueError), match=message):
        report_module._vad_intervals(raw, recording, analysis_sample_count=16_000)


@pytest.mark.parametrize("issue", (1, "NOT_A_STABLE_ISSUE"))
def test_report_issue_counter_rejects_invalid_issue_values(issue: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        report_module._count_issue(Counter(), issue)


def test_terminal_evidence_requires_processing_history_for_every_recording() -> None:
    with pytest.raises(ValueError, match="processing history"):
        report_module._validate_terminal_processing_evidence(
            recordings=[_recording_for_report_unit_tests()],
            processing_rows=(),
            config_sha256="a" * 64,
            inventory_sha256="f" * 64,
            rights_sha256="b" * 64,
            transcripts_sha256="c" * 64,
            review_sha256="d" * 64,
            segments_sha256="e" * 64,
            pairing_sha256s={},
            alignment_sha256s={},
            attestations=(),
        )


def test_report_atomic_preparation_removes_temporary_file_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(report_module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        report_module._prepare_atomic(tmp_path / "report.json", b"report")

    assert not tuple(tmp_path.glob(".report.json.*"))


def test_report_atomic_preparation_preserves_primary_failure_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_unlink = type(tmp_path).unlink

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("primary fsync failure")

    def fail_temporary_cleanup(self: Path, missing_ok: bool = False) -> None:
        if self.parent == tmp_path and self.name.startswith(".report.json."):
            raise OSError("secondary cleanup failure")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(report_module.os, "fsync", fail_fsync)
    monkeypatch.setattr(type(tmp_path), "unlink", fail_temporary_cleanup)

    with pytest.raises(OSError, match="primary fsync failure"):
        report_module._prepare_atomic(tmp_path / "report.json", b"report")


def test_report_snapshot_rejects_metadata_drift_without_changing_output_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_fstat = report_module.os.fstat
    calls = 0

    def drift_during_second_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            json_path.write_bytes(before[0] + b"external drift")
            changed = real_fstat(descriptor)
            json_path.write_bytes(before[0])
            return changed
        return real_fstat(descriptor)

    monkeypatch.setattr(report_module.os, "fstat", drift_during_second_fstat)

    with pytest.raises(OSError, match="changed while snapshotting"):
        build_report(paths, config)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before
    assert not tuple(paths.manifests.glob(".recovery.*"))


def test_missing_prepared_markdown_fails_without_changing_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_prepare = report_module._prepare_atomic

    def remove_prepared_markdown(path: Path, content: bytes) -> Path:
        temporary = real_prepare(path, content)
        if path == markdown_path:
            temporary.unlink()
        return temporary

    monkeypatch.setattr(report_module, "_prepare_atomic", remove_prepared_markdown)

    with pytest.raises(OSError, match="prepared report output is missing"):
        build_report(paths, config)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before
    assert not tuple(paths.manifests.glob(".recovery.*"))
    assert not tuple(paths.manifests.glob(".report.*"))


def test_build_report_derives_every_metric_and_writes_deterministic_outputs(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.scale_decision.value == "scalable"
    assert report.text_unit_count == 6
    assert report.reading_count == 12
    assert report.repetition_group_count == 6
    assert report.approved_segment_count == 12
    assert report.metrics.input_duration_seconds == 20.0
    assert report.metrics.speech_duration_seconds == 10.4
    assert report.metrics.auto_pairing_correct_ratio == 1.0
    assert report.metrics.boundary_unchanged_ratio == 1.0
    assert report.metrics.review_minutes_per_audio_minute == 1.0
    assert report.metrics.approved_speech_ratio == 1.0
    assert report.metrics.gpu_realtime_factor == pytest.approx(0.074)
    assert report.approved_duration_seconds == pytest.approx(14.8)
    assert report.rejected_duration_seconds == 0.0
    assert report.unreviewed_duration_seconds == 0.0
    assert report.issue_code_counts == ()
    assert report.rejection_reason_counts == ()
    assert report.projected_full_corpus_person_hours == pytest.approx(30.0 / 3600.0)
    assert report.projected_gpu_hours == pytest.approx(1.48 / 3600.0)
    assert report.projected_storage_bytes == 1_000

    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    first = (json_path.read_bytes(), markdown_path.read_bytes())
    assert json.loads(json_path.read_text(encoding="utf-8")) == report.to_dict()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# LatinTTS Corpus Alignment Pilot Report" in markdown
    assert "Automatic pairing accepted without correction ratio" in markdown
    assert "Automatic pairing correct ratio" not in markdown
    assert "Approved effective take duration ratio" in markdown
    assert "Approved VAD speech ratio" not in markdown
    assert "Text preparation (seconds) | 10.000000" in markdown
    assert "Human review (seconds) | 20.000000" in markdown
    assert "GPU retries | 0" in markdown
    assert "Pilot storage (bytes) | 1000" in markdown

    assert build_report(paths, config) == report
    assert (json_path.read_bytes(), markdown_path.read_bytes()) == first


def test_build_report_rejects_zero_denominators_instead_of_emitting_nan(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(
        paths.manifests / "pilot-telemetry.json",
        (_telemetry_row(alignment_wall_seconds=0.0),),
    )

    with pytest.raises(ValueError, match="alignment_wall_seconds"):
        build_report(paths, config)

    assert not (paths.manifests / "report.json").exists()
    assert not (paths.manifests / "report.md").exists()


def test_build_report_counts_rejection_reasons_and_marks_not_ready(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path, decision="rejected")
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.scale_decision.value == "not_ready"
    assert report.approved_segment_count == 0
    assert report.approved_duration_seconds == 0.0
    assert report.rejected_duration_seconds == pytest.approx(14.8)
    assert report.rejection_reason_counts == (("listened in full", 12),)
    assert report.metrics.approved_speech_ratio == 0.0
    assert read_jsonl(paths.manifests / "segments.jsonl") == ()


def test_build_report_rejects_rejected_recording_with_approved_takes(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    recordings_path = paths.manifests / "recordings.jsonl"
    recordings = list(read_jsonl(recordings_path))
    recordings[0]["state"] = "REJECTED"
    write_jsonl_atomic(recordings_path, recordings)

    with pytest.raises(ValueError, match="terminal state"):
        build_report(paths, config)


def test_build_report_rejects_approved_recording_with_only_rejected_takes(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path, decision="rejected")
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    recordings_path = paths.manifests / "recordings.jsonl"
    recordings = list(read_jsonl(recordings_path))
    recordings[0]["state"] = "APPROVED"
    write_jsonl_atomic(recordings_path, recordings)

    with pytest.raises(ValueError, match="terminal state"):
        build_report(paths, config)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("terminal-event-missing", "processing"),
        ("rights-file-missing", "rights.jsonl"),
        ("recording-rights-id", "rights"),
        ("segment-rights-id", r"segment.*evidence"),
        ("transcript-selected-candidate", "selected source"),
        ("transcript-source-hash", "source candidate hash"),
        ("segment-source-text-id", r"segment.*evidence"),
    ),
)
def test_build_report_rejects_broken_terminal_evidence_chain(
    tmp_path: Path, tamper: str, message: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    recordings_path = paths.manifests / "recordings.jsonl"
    rights_path = paths.manifests / "rights.jsonl"
    transcripts_path = paths.manifests / "transcripts.jsonl"
    segments_path = paths.manifests / "segments.jsonl"
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"

    if tamper == "terminal-event-missing":
        events = tuple(
            row for row in read_jsonl(processing_path) if row["previous_state"] != "REVIEWED"
        )
        write_jsonl_atomic(processing_path, events)
    elif tamper == "rights-file-missing":
        rights_path.unlink()
    elif tamper == "recording-rights-id":
        recordings = list(read_jsonl(recordings_path))
        recordings[0]["rights_id"] = "rights-unknown"
        write_jsonl_atomic(recordings_path, recordings)
    elif tamper == "segment-rights-id":
        segments = list(read_jsonl(segments_path))
        segments[0]["rights_id"] = "rights-unknown"
        write_jsonl_atomic(segments_path, segments)
    elif tamper == "transcript-selected-candidate":
        transcripts = list(read_jsonl(transcripts_path))
        transcripts[0]["selected_candidate_id"] = "source-unknown"
        write_jsonl_atomic(transcripts_path, transcripts)
    elif tamper == "transcript-source-hash":
        transcripts = list(read_jsonl(transcripts_path))
        transcripts[0]["source_candidates"][0]["source_sha256"] = "f" * 64
        write_jsonl_atomic(transcripts_path, transcripts)
    else:
        segments = list(read_jsonl(segments_path))
        segments[0]["source_text_id"] = "source-unknown"
        write_jsonl_atomic(segments_path, segments)

    with pytest.raises((OSError, ValueError, CorpusFailure), match=message):
        build_report(paths, config)


def test_build_report_rejects_rights_without_internal_evaluation_permission(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    rights_path = paths.manifests / "rights.jsonl"
    rights = list(read_jsonl(rights_path))
    rights[0]["allow_internal_evaluation"] = False
    write_jsonl_atomic(rights_path, rights)

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)
    assert failure.value.code == "RIGHTS_SCOPE_UNCONFIRMED"


def _replace_terminal_events_with_old_rows(
    paths: object, config: CorpusConfig, version: str
) -> None:
    recordings = {
        raw["recording_id"]: review_module._decode_recording(raw)
        for raw in read_jsonl(paths.manifests / "recordings.jsonl")  # type: ignore[attr-defined]
    }
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    rewritten = []
    for raw in read_jsonl(processing_path):
        event = ProcessingEvent.from_dict(raw)
        if event.previous_state is not CorpusState.REVIEWED:
            rewritten.append(raw)
            continue
        recording = recordings[event.recording_id]
        if version == "v1":
            inputs = (
                event.input_sha256s[0],
                event.input_sha256s[2],
                event.input_sha256s[3],
                event.input_sha256s[4],
                event.input_sha256s[5],
                event.input_sha256s[6],
                event.input_sha256s[7],
            )
        else:
            inputs = (
                event.input_sha256s[0],
                event.input_sha256s[4],
                event.input_sha256s[5],
                event.input_sha256s[6],
                event.input_sha256s[7],
            )
        _, old_event = advance_recording(
            replace(recording, state=CorpusState.REVIEWED),
            recording.state,
            input_sha256s=inputs,
            config_sha256=event.config_sha256,
            tool_versions=("approved-manifest-v1", event.tool_versions[1]),
            started_at=event.started_at,
            finished_at=event.finished_at,
            result=event.result,
        )
        rewritten.append(old_event.to_dict())
    write_jsonl_atomic(processing_path, rewritten)


def _terminal_processing_by_recording(
    paths: object, config: CorpusConfig
) -> dict[str, ProcessingEvent]:
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    return {
        event.recording_id: event
        for event in (ProcessingEvent.from_dict(raw) for raw in read_jsonl(processing_path))
        if event.previous_state is CorpusState.REVIEWED
    }


def _attestation_path(paths: object, config: CorpusConfig) -> Path:
    return (
        paths.alignments  # type: ignore[attr-defined]
        / "runs"
        / config.digest
        / "manifest-attestations.jsonl"
    )


def _attestation_rows(
    old_events: dict[str, ProcessingEvent],
    v2_events: dict[str, ProcessingEvent],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "schema_version": "1",
            "terminal_event_id": old_events[recording_id].event_id,
            "evidence": v2_events[recording_id].to_dict(),
        }
        for recording_id in v2_events
    )


@pytest.mark.parametrize(
    ("version", "tamper_rights_metadata"),
    (("v1", False), ("v1", True), ("legacy", False), ("legacy", True)),
)
def test_build_report_requires_current_terminal_evidence_not_legacy(
    tmp_path: Path, version: str, tamper_rights_metadata: bool
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    _replace_terminal_events_with_old_rows(paths, config, version)
    if tamper_rights_metadata:
        rights_path = paths.manifests / "rights.jsonl"
        rights = list(read_jsonl(rights_path))
        rights[0]["basis"] = "changed after legacy manifest build"
        write_jsonl_atomic(rights_path, rights)

    expected = (
        r"processing manifest terminal event"
        if version == "v1" and tamper_rights_metadata
        else r"legacy terminal.*rebuild"
    )
    with pytest.raises(ValueError, match=expected):
        build_report(paths, config)


@pytest.mark.parametrize("version", ("v1", "legacy"))
def test_build_manifest_recovery_appends_v2_attestations_for_old_terminal_evidence(
    tmp_path: Path, version: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    _replace_terminal_events_with_old_rows(paths, config, version)
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    processing_before = processing_path.read_bytes()
    terminal_by_recording = {
        event.recording_id: event
        for event in (ProcessingEvent.from_dict(raw) for raw in read_jsonl(processing_path))
        if event.previous_state is CorpusState.REVIEWED
    }

    assert _build_with_fake_audio(paths, config)
    assert processing_path.read_bytes() == processing_before
    migrated = build_report(paths, config)

    attestation_path = paths.alignments / "runs" / config.digest / "manifest-attestations.jsonl"
    attestation_bytes = attestation_path.read_bytes()
    attestations = read_jsonl(attestation_path)
    assert len(attestations) == len(terminal_by_recording) == 2
    for raw in attestations:
        assert set(raw) == {"schema_version", "terminal_event_id", "evidence"}
        assert raw["schema_version"] == "1"
        evidence_raw = raw["evidence"]
        assert type(evidence_raw) is dict
        evidence = ProcessingEvent.from_dict(evidence_raw)
        terminal = terminal_by_recording[evidence.recording_id]
        assert raw["terminal_event_id"] == terminal.event_id
        assert evidence.previous_state is CorpusState.REVIEWED
        assert evidence.target_state is terminal.target_state
        assert len(evidence.input_sha256s) == 8
        assert evidence.tool_versions[0] == "approved-manifest-v2"
        identity = {
            "recording_id": evidence.recording_id,
            "previous_state": evidence.previous_state.value,
            "target_state": evidence.target_state.value,
            "inputs": evidence.input_sha256s,
            "config": evidence.config_sha256,
            "tools": evidence.tool_versions,
            "result": evidence.result,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        assert evidence.event_id == (
            "state-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        )

    assert _build_with_fake_audio(paths, config)
    assert attestation_path.read_bytes() == attestation_bytes
    assert build_report(paths, config).to_dict() == migrated.to_dict()


def test_fresh_v2_report_uses_manifest_builder_not_pairing_ffmpeg_version(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    recordings = {
        raw["recording_id"]: review_module._decode_recording(raw)
        for raw in read_jsonl(paths.manifests / "recordings.jsonl")
    }
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    rewritten = []
    for raw in read_jsonl(processing_path):
        event = ProcessingEvent.from_dict(raw)
        if event.previous_state is not CorpusState.REVIEWED:
            rewritten.append(raw)
            continue
        recording = recordings[event.recording_id]
        _, rebuilt = advance_recording(
            replace(recording, state=CorpusState.REVIEWED),
            recording.state,
            input_sha256s=event.input_sha256s,
            config_sha256=event.config_sha256,
            tool_versions=("approved-manifest-v2", "ffmpeg-manifest-2"),
            started_at=event.started_at,
            finished_at=event.finished_at,
            result=event.result,
        )
        rewritten.append(rebuilt.to_dict())
    write_jsonl_atomic(processing_path, rewritten)

    report = build_report(paths, config)

    assert report.approved_segment_count == 12


def test_manifest_attestation_append_preserves_prefix_and_is_byte_idempotent(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    v2_events = _terminal_processing_by_recording(paths, config)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    old_events = _terminal_processing_by_recording(paths, config)
    rows = _attestation_rows(old_events, v2_events)
    attestation_path = _attestation_path(paths, config)
    write_jsonl_atomic(attestation_path, rows[:1])
    existing_prefix = attestation_path.read_bytes()

    assert _build_with_fake_audio(paths, config)

    appended = attestation_path.read_bytes()
    assert appended.startswith(existing_prefix)
    assert len(read_jsonl(attestation_path)) == 2
    assert _build_with_fake_audio(paths, config)
    assert attestation_path.read_bytes() == appended


def test_manifest_attestation_first_publish_does_not_clobber_concurrent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    attestation_path = _attestation_path(paths, config)
    assert not attestation_path.exists()
    foreign_bytes = b"concurrent manifest attestation bytes\n"
    real_persist = manifest_module.persist_recording_transitions

    def persist_then_publish_foreign_file(**kwargs: object) -> None:
        real_persist(**kwargs)  # type: ignore[arg-type]
        attestation_path.write_bytes(foreign_bytes)

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        persist_then_publish_foreign_file,
    )

    with pytest.raises(FileExistsError):
        _build_with_fake_audio(paths, config)

    assert attestation_path.read_bytes() == foreign_bytes


def test_manifest_attestation_publish_preserves_primary_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    attestation_path = _attestation_path(paths, config)
    foreign_bytes = b"concurrent manifest attestation bytes\n"
    real_persist = manifest_module.persist_recording_transitions
    real_unlink = type(attestation_path).unlink

    def persist_then_publish_foreign_file(**kwargs: object) -> None:
        real_persist(**kwargs)  # type: ignore[arg-type]
        attestation_path.write_bytes(foreign_bytes)

    def fail_attestation_temporary_cleanup(self: Path, missing_ok: bool = False) -> None:
        if self.parent == attestation_path.parent and self.name.startswith(
            ".manifest-attestations.jsonl."
        ):
            raise OSError("secondary attestation cleanup failure")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        persist_then_publish_foreign_file,
    )
    monkeypatch.setattr(type(attestation_path), "unlink", fail_attestation_temporary_cleanup)

    with pytest.raises(FileExistsError):
        _build_with_fake_audio(paths, config)

    assert attestation_path.read_bytes() == foreign_bytes


def test_noncanonical_manifest_attestation_is_rejected_before_state_write(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    v2_events = _terminal_processing_by_recording(paths, config)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    old_events = _terminal_processing_by_recording(paths, config)
    recordings_path = paths.manifests / "recordings.jsonl"
    recordings = list(read_jsonl(recordings_path))
    for recording in recordings:
        recording["state"] = CorpusState.REVIEWED.value
    write_jsonl_atomic(recordings_path, recordings)
    attestation_path = _attestation_path(paths, config)
    noncanonical = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in _attestation_rows(old_events, v2_events)
    ).encode("utf-8")
    attestation_path.write_bytes(noncanonical)
    protected_paths = (
        recordings_path,
        paths.alignments / "runs" / config.digest / "processing-events.jsonl",
        paths.manifests / "review.jsonl",
        paths.manifests / "segments.jsonl",
        attestation_path,
    )
    before = {path: path.read_bytes() for path in protected_paths}

    with pytest.raises(ValueError, match="canonical"):
        _build_with_fake_audio(paths, config)

    assert {path: path.read_bytes() for path in protected_paths} == before


def test_unlinked_manifest_attestation_is_rejected_before_manifest_publication(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    recordings_path = paths.manifests / "recordings.jsonl"
    recording = review_module._decode_recording(read_jsonl(recordings_path)[0])
    timestamp = "1970-01-01T00:00:00+00:00"
    _, evidence = advance_recording(
        recording,
        CorpusState.APPROVED,
        input_sha256s=("a" * 64,) * 8,
        config_sha256=config.digest,
        tool_versions=("approved-manifest-v2", "ffmpeg-test-1"),
        started_at=timestamp,
        finished_at=timestamp,
        result="success",
    )
    attestation_path = _attestation_path(paths, config)
    write_jsonl_atomic(
        attestation_path,
        (
            {
                "schema_version": "1",
                "terminal_event_id": "state-ffffffffffffffff",
                "evidence": evidence.to_dict(),
            },
        ),
    )
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    protected_paths = (
        recordings_path,
        processing_path,
        paths.manifests / "review.jsonl",
        attestation_path,
    )
    protected_before = {path: path.read_bytes() for path in protected_paths}
    manifest_path = paths.manifests / "segments.jsonl"
    segment_files_before = {
        path.relative_to(paths.segments): path.read_bytes()
        for path in paths.segments.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match="extraneous manifest attestation"):
        _build_with_fake_audio(paths, config)

    assert {path: path.read_bytes() for path in protected_paths} == protected_before
    assert not manifest_path.exists()
    assert {
        path.relative_to(paths.segments): path.read_bytes()
        for path in paths.segments.rglob("*")
        if path.is_file()
    } == segment_files_before


def test_fresh_v2_report_rejects_extraneous_manifest_attestation(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    v2_events = _terminal_processing_by_recording(paths, config)
    first = next(iter(v2_events.values()))
    write_jsonl_atomic(
        _attestation_path(paths, config),
        (
            {
                "schema_version": "1",
                "terminal_event_id": first.event_id,
                "evidence": first.to_dict(),
            },
        ),
    )

    with pytest.raises(ValueError, match="extraneous manifest attestation"):
        build_report(paths, config)


@pytest.mark.parametrize(
    "tamper",
    (
        "outer-key",
        "nested-key",
        "event-id",
        "marker",
        "input-length",
        "terminal-event-id",
        "duplicate",
    ),
)
def test_old_terminal_report_rejects_invalid_manifest_attestation(
    tmp_path: Path, tamper: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    v2_events = _terminal_processing_by_recording(paths, config)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    old_events = _terminal_processing_by_recording(paths, config)
    rows = [deepcopy(row) for row in _attestation_rows(old_events, v2_events)]
    if tamper == "outer-key":
        rows[0]["unknown"] = True
    elif tamper == "nested-key":
        rows[0]["evidence"]["unknown"] = True  # type: ignore[index]
    elif tamper == "event-id":
        rows[0]["evidence"]["event_id"] = "state-0000000000000000"  # type: ignore[index]
    elif tamper == "marker":
        rows[0]["evidence"]["tool_versions"][0] = "unknown-manifest"  # type: ignore[index]
    elif tamper == "input-length":
        rows[0]["evidence"]["input_sha256s"].pop()  # type: ignore[index,union-attr]
    elif tamper == "terminal-event-id":
        rows[0]["terminal_event_id"] = "state-ffffffffffffffff"
    else:
        rows.append(deepcopy(rows[0]))
    write_jsonl_atomic(_attestation_path(paths, config), rows)

    with pytest.raises(ValueError, match="manifest attestation"):
        build_report(paths, config)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("selection-schema", "selection identity"),
        ("selection-hash", "spoken inventory"),
        ("recording-state", "terminal review states"),
        ("rights-speaker", "rights speaker"),
        ("analysis-audio", "analysis_audio"),
    ),
)
def test_build_report_rejects_invalid_pilot_identity_layers(
    tmp_path: Path, tamper: str, message: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    if tamper.startswith("selection-"):
        selection_path = paths.manifests / "pilot-selection.json"
        selection = dict(read_jsonl(selection_path)[0])
        if tamper == "selection-schema":
            selection["schema_version"] = "2"
        else:
            selection["inventory_hashes"][0] = "f" * 64
        write_jsonl_atomic(selection_path, (selection,))
    elif tamper == "recording-state":
        recordings_path = paths.manifests / "recordings.jsonl"
        recordings = list(read_jsonl(recordings_path))
        recordings[0]["state"] = "REVIEWED"
        write_jsonl_atomic(recordings_path, recordings)
    elif tamper == "rights-speaker":
        rights_path = paths.manifests / "rights.jsonl"
        rights = list(read_jsonl(rights_path))
        rights[0]["speaker_id"] = "speaker-other"
        write_jsonl_atomic(rights_path, rights)
    else:
        segmentation_path = _first_selected_run_directory(paths, config) / "segmentation.json"
        segmentation = dict(read_jsonl(segmentation_path)[0])
        segmentation["analysis_audio"] = []
        write_jsonl_atomic(segmentation_path, (segmentation,))

    with pytest.raises((CorpusFailure, TypeError, ValueError), match=message):
        build_report(paths, config)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("identity", "identity"),
        ("takes", "exactly two takes"),
        ("take-index", "take identities"),
        ("entity-id", "entity identity"),
        ("provenance-type", "take_provenance"),
        ("provenance-bounds", "take provenance"),
    ),
)
def test_build_report_rejects_invalid_review_automatic_layers(
    tmp_path: Path, tamper: str, message: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    automatic_path = _first_review_automatic_path(paths, config)
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    if tamper == "identity":
        automatic["unit_id"] = "unit-other"
    elif tamper == "takes":
        automatic["takes"] = []
    elif tamper == "take-index":
        automatic["takes"][1]["take_index"] = 1
    elif tamper == "entity-id":
        automatic["takes"][0]["entity_id"] = "review:other"
    elif tamper == "provenance-type":
        automatic["takes"][0]["take_provenance"] = []
    else:
        automatic["takes"][0]["take_provenance"]["source_start_sample"] += 1
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        build_report(paths, config)


def test_build_report_rejects_segmentation_provenance_tampering(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    segmentation_path = _first_selected_run_directory(paths, config) / "segmentation.json"
    segmentation = deepcopy(read_jsonl(segmentation_path)[0])
    segmentation["config_sha256"] = "f" * 64
    write_jsonl_atomic(segmentation_path, (segmentation,))

    with pytest.raises(ValueError, match="segmentation"):
        build_report(paths, config)

    assert not (paths.manifests / "report.json").exists()
    assert not (paths.manifests / "report.md").exists()


def test_build_report_rejects_review_automatic_binding_tampering(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    automatic_path = _first_review_automatic_path(paths, config)
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    automatic["artifact_binding"]["config_sha256"] = "f" * 64
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact binding"):
        build_report(paths, config)


def test_build_report_rejects_original_pairing_with_different_run_identity(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    run_directory = _first_selected_run_directory(paths, config)
    original = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
    changed_config = "f" * 64
    changed_cache = _pairing_cache_key(
        original.recording_id,
        original.windows,
        original.analysis_audio_relative_path,
        original.analysis_audio_sha256,
        original.segmentation_artifact_sha256,
        changed_config,
        original.pairing_parameters,
        original.ffmpeg_version,
    )
    stale_original = replace(
        original,
        config_sha256=changed_config,
        cache_key=changed_cache,
        integrity_sha256="",
    )
    write_jsonl_atomic(
        run_directory / "pairing-automatic.json",
        (pairing_to_dict(stale_original),),
    )

    with pytest.raises(ValueError, match=r"automatic pairing|pairing correction"):
        build_report(paths, config)


def test_corrected_pairing_metrics_use_original_automatic_evidence(tmp_path: Path) -> None:
    paths, config = _completed_corrected_pairing_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.metrics.auto_pairing_correct_ratio == pytest.approx(2 / 3)
    assert report.issue_code_counts == (("TAKE_DURATION_MISMATCH", 2),)
    assert report.scale_decision.value == "not_ready"
    assert report.approved_segment_count == 12


def test_build_report_does_not_fall_back_to_final_pairing_when_correction_event_exists(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    run_directory = _first_selected_run_directory(paths, config)
    pairing = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
    group = pairing.groups[0].group
    assert group is not None
    event = review_module._new_review_event(
        review_module._pairing_entity_id(pairing.recording_id, group.unit_id),
        "pairing_selected_split",
        None,
        group.selected_evidence.split_sample,
        {
            "reason": "human pairing correction",
            "reviewer": "owner",
            "reviewed_at": "2026-07-19T13:00:00+08:00",
        },
    )
    review_path = paths.manifests / "review.jsonl"
    write_jsonl_atomic(review_path, (*read_jsonl(review_path), event.to_dict()))

    with pytest.raises(ValueError, match=r"pairing correction|pairing history"):
        build_report(paths, config)


@pytest.mark.parametrize(
    "tamper",
    ("source_audio", "text_layers", "take_provenance"),
)
def test_build_report_rejects_review_automatic_evidence_tampering(
    tmp_path: Path, tamper: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    automatic_path = _first_review_automatic_path(paths, config)
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    if tamper == "source_audio":
        automatic["source_audio"]["sha256"] = "f" * 64
    elif tamper == "text_layers":
        automatic["text_layers"]["unit_spoken_text"] = "tampered text"
    else:
        automatic["takes"][0]["take_provenance"]["candidate_audio_sha256"] = "f" * 64
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")

    with pytest.raises(ValueError, match=r"review (source|text|take)|review automatic"):
        build_report(paths, config)


def test_build_report_rejects_unknown_review_entity(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    event = review_module._new_review_event(
        "review:unknown-entity",
        "review_decision",
        "unreviewed",
        "approved",
        {
            "reason": "orphan event",
            "reviewer": "owner",
            "reviewed_at": "2026-07-19T13:00:00+08:00",
        },
    )
    review_path = paths.manifests / "review.jsonl"
    write_jsonl_atomic(review_path, (*read_jsonl(review_path), event.to_dict()))

    with pytest.raises(ValueError, match=r"unknown.*entity"):
        build_report(paths, config)


@pytest.mark.parametrize(
    ("field", "tampered"),
    (
        ("config_sha256", "f" * 64),
        ("source_audio_sha256", "f" * 64),
        ("source_start_sample", 1),
        ("text_unit_id", "tampered-unit"),
    ),
)
def test_build_report_rejects_same_key_segment_provenance_tampering(
    tmp_path: Path, field: str, tampered: object
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    segments_path = paths.manifests / "segments.jsonl"
    segments = list(read_jsonl(segments_path))
    segments[0][field] = tampered
    write_jsonl_atomic(segments_path, segments)

    with pytest.raises(ValueError, match=r"segment.*evidence"):
        build_report(paths, config)


def test_boundary_moved_then_restored_is_still_counted_as_changed(tmp_path: Path) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    grid_path = groups[0] / "take-1.TextGrid"
    original = read_textgrid(grid_path)
    assert original.words[-1].end_seconds < original.take_end
    write_textgrid(
        grid_path,
        duration_seconds=original.duration_seconds,
        take_start=original.take_start,
        take_end=original.words[-1].end_seconds,
        words=original.words,
    )
    assert import_review_bundle(paths, config)
    write_textgrid(
        grid_path,
        duration_seconds=original.duration_seconds,
        take_start=original.take_start,
        take_end=original.take_end,
        words=original.words,
    )
    assert import_review_bundle(paths, config)
    assert _build_with_fake_audio(paths, config)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.metrics.boundary_unchanged_ratio == pytest.approx(11 / 12)


def test_rejection_reason_comes_from_final_effective_rejection(tmp_path: Path) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    _set_decisions(
        groups[0],
        decision="rejected",
        reason="initial rejection",
        reviewed_at="2026-07-19T12:01:00+08:00",
        take_indexes=(1,),
    )
    assert import_review_bundle(paths, config)
    _set_decisions(
        groups[0],
        decision="approved",
        reason="temporary approval",
        reviewed_at="2026-07-19T12:02:00+08:00",
        take_indexes=(1,),
    )
    assert import_review_bundle(paths, config)
    _set_decisions(
        groups[0],
        decision="rejected",
        reason="final rejection",
        reviewed_at="2026-07-19T12:03:00+08:00",
        take_indexes=(1,),
    )
    assert import_review_bundle(paths, config)
    assert _build_with_fake_audio(paths, config)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.rejection_reason_counts == (("final rejection", 1),)
    assert report.approved_segment_count == 11


def test_report_output_pair_rolls_back_when_second_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    failed = False

    def fail_markdown_once(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal failed
        if target == markdown_path and not failed:
            failed = True
            raise OSError("injected second publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_markdown_once)

    with pytest.raises(OSError, match="injected second publication failure"):
        build_report(paths, config)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before
    assert not tuple(paths.manifests.glob(".report.*"))
    assert not tuple(paths.manifests.glob(".recovery.*"))
    assert not tuple(paths.manifests.glob(".report-output-transaction.*"))


@pytest.mark.parametrize("target_name", ("report.json", "report.md"))
def test_first_report_publication_does_not_clobber_concurrent_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    target = paths.manifests / target_name
    other = (
        paths.manifests / ({"report.json": "report.md", "report.md": "report.json"}[target_name])
    )
    foreign_bytes = f"foreign {target_name}\n".encode()
    real_link = report_module.os.link
    injected = False

    def inject_before_link(
        source: Path, destination: Path, *args: object, **kwargs: object
    ) -> None:
        nonlocal injected
        if Path(destination) == target and not injected:
            injected = True
            target.write_bytes(foreign_bytes)
        real_link(source, destination, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(report_module.os, "link", inject_before_link)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert injected
    assert target.read_bytes() == foreign_bytes
    assert not other.exists()
    assert paths.manifests / ".report-output-transaction.json" in failure.value.recovery_paths


@pytest.mark.parametrize("target_name", ("report.json", "report.md"))
@pytest.mark.parametrize("drift", ("digest", "identity"))
def test_existing_report_drift_after_backup_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    drift: str,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    target = paths.manifests / target_name
    other = markdown_path if target == json_path else json_path
    other_before = other.read_bytes()
    foreign_bytes = f"foreign {drift} {target_name}\n".encode()
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_prepare = report_module._prepare_atomic
    injected = False

    def drift_after_backup(path: Path, content: bytes) -> Path:
        nonlocal injected
        temporary = real_prepare(path, content)
        if path.name == f"recovery.{target_name}" and not injected:
            injected = True
            if drift == "digest":
                target.write_bytes(foreign_bytes)
            else:
                replacement = target.with_name(f"foreign.{target.name}")
                replacement.write_bytes(foreign_bytes)
                replacement.replace(target)
        return temporary

    monkeypatch.setattr(report_module, "_prepare_atomic", drift_after_backup)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert injected
    assert target.read_bytes() == foreign_bytes
    assert other.read_bytes() == other_before
    assert any(
        path.is_file() and path.read_bytes() != foreign_bytes
        for path in failure.value.recovery_paths
    )


def test_report_rollback_refuses_to_overwrite_concurrent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign_bytes = b"foreign replacement after json publication\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_replace = report_module.os.replace
    injected = False

    def replace_then_take_over_before_second_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal injected
        if target == markdown_path and not injected:
            injected = True
            replacement = json_path.with_name("foreign.report.json")
            replacement.write_bytes(foreign_bytes)
            real_replace(replacement, json_path)
            raise OSError("injected second publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    monkeypatch.setattr(
        report_module,
        "_publish_report_output",
        replace_then_take_over_before_second_publication,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert injected
    assert json_path.read_bytes() == foreign_bytes
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_pair_commit_revalidates_json_after_markdown_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign_bytes = b"foreign replacement after json publish returned\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_replace = report_module.os.replace
    injected = False

    def publish_then_take_over_json(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal injected
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == json_path and not injected:
            injected = True
            replacement = json_path.with_name("foreign.report.json")
            replacement.write_bytes(foreign_bytes)
            real_replace(replacement, json_path)
        return published

    monkeypatch.setattr(report_module, "_publish_report_output", publish_then_take_over_json)

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert injected
    assert json_path.read_bytes() == foreign_bytes
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_publication_snapshot_failure_preserves_old_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_snapshot = report_module._prepared_report_snapshot

    def fail_published_json_snapshot(path: Path) -> report_module._ReportOutputSnapshot:
        if path == json_path:
            raise OSError("injected published JSON snapshot failure")
        return real_snapshot(path)

    monkeypatch.setattr(
        report_module,
        "_prepared_report_snapshot",
        fail_published_json_snapshot,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert "injected published JSON snapshot failure" in str(failure.value.__cause__)
    assert json_path.is_file()
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_publication_rejects_target_that_differs_from_prepared_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign_bytes = b"foreign JSON before publication verification\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_link = report_module.os.link
    real_replace = report_module.os.replace
    injected = False

    def link_then_take_over_json(source: Path, destination: Path) -> None:
        nonlocal injected
        real_link(source, destination)
        if Path(destination) == json_path and not injected:
            replacement = json_path.with_name("foreign.report.json")
            replacement.write_bytes(foreign_bytes)
            real_replace(replacement, json_path)
            injected = True

    monkeypatch.setattr(report_module.os, "link", link_then_take_over_json)

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert injected
    assert "foreign report output blocks recovery" in str(failure.value.__cause__)
    assert json_path.read_bytes() == foreign_bytes
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_pair_commit_preserves_concurrent_markdown_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign_bytes = b"foreign Markdown after publish returned\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_replace = report_module.os.replace
    injected = False

    def publish_then_take_over_markdown(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal injected
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == markdown_path and not injected:
            injected = True
            replacement = markdown_path.with_name("foreign.report.md")
            replacement.write_bytes(foreign_bytes)
            real_replace(replacement, markdown_path)
        return published

    monkeypatch.setattr(
        report_module,
        "_publish_report_output",
        publish_then_take_over_markdown,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert injected
    assert json_path.read_bytes() == old_json
    assert markdown_path.read_bytes() == foreign_bytes
    assert "foreign report output blocks recovery" in str(failure.value.__cause__)
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_markdown for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_rollback_requires_a_snapshot_for_existing_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_snapshot = report_module._prepared_report_snapshot
    real_publish = report_module._publish_report_output

    def omit_json_backup_snapshot(path: Path) -> report_module._ReportOutputSnapshot:
        if path.name.startswith(".recovery.report.json."):
            return None  # type: ignore[return-value]
        return real_snapshot(path)

    def fail_markdown_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected Markdown publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    monkeypatch.setattr(
        report_module,
        "_prepared_report_snapshot",
        omit_json_backup_snapshot,
    )
    monkeypatch.setattr(
        report_module,
        "_publish_report_output",
        fail_markdown_publication,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert "backup is invalid" in str(failure.value.__cause__)
    assert json_path.read_bytes() == old_json
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_rollback_detects_corrupted_restored_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_markdown = markdown_path.read_bytes()
    corrupt_bytes = b"corrupt restored JSON\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    old_json = json_path.read_bytes()
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link

    def fail_markdown_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected Markdown publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def corrupt_json_restore(source: Path, destination: Path) -> None:
        real_link(source, destination)
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            json_path.write_bytes(corrupt_bytes)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_markdown_publication)
    monkeypatch.setattr(report_module.os, "link", corrupt_json_restore)

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert "differs from old content" in str(failure.value.__cause__)
    assert json_path.read_bytes() == corrupt_bytes
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_output_pair_surfaces_rollback_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link

    def fail_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def fail_json_restore(source: Path, destination: Path) -> None:
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            raise OSError("injected rollback failure")
        real_link(source, destination)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_publication)
    monkeypatch.setattr(report_module.os, "link", fail_json_restore)

    with pytest.raises(report_module.ReportOutputRecoveryError, match="rollback failed") as failure:
        build_report(paths, config)

    assert isinstance(failure.value.__cause__, OSError)
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() in before for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_cli_maps_output_recovery_failure_and_lists_backups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link

    def fail_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def fail_json_restore(source: Path, destination: Path) -> None:
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            raise OSError("injected rollback failure")
        real_link(source, destination)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_publication)
    monkeypatch.setattr(report_module.os, "link", fail_json_restore)

    assert cli.main(["--project-root", str(tmp_path), "report"]) == 2
    stderr = capsys.readouterr().err
    assert stderr.startswith("REPORT_OUTPUT_RECOVERY_REQUIRED:")
    assert "recovery" in stderr
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    for path in recovery_backups:
        assert f"manifests/{path.name}" in stderr


def test_report_cli_uses_dedicated_recovery_code_without_absolute_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _copy_config(tmp_path)
    manifests = tmp_path / "local-data" / "manifests"
    backup = manifests / ".recovery.report.json.fixture"
    error = report_module.ReportOutputRecoveryError((backup,), manifests_root=manifests)
    monkeypatch.setattr(
        cli,
        "build_report",
        lambda *_args: (_ for _ in ()).throw(error),
    )

    assert cli.main(["--project-root", str(tmp_path), "report"]) == 2
    stderr = capsys.readouterr().err
    assert stderr.startswith("REPORT_OUTPUT_RECOVERY_REQUIRED:")
    assert "manifests/.recovery.report.json.fixture" in stderr
    assert str(tmp_path.resolve()) not in stderr


def test_existing_report_claim_race_preserves_foreign_and_old_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign = b"foreign writer won the claim race\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_rename = report_module.os.rename
    real_replace = report_module.os.replace
    injected = False

    def replace_before_claim(source: Path, destination: Path) -> None:
        nonlocal injected
        if Path(source) == json_path and Path(destination).name == "report.json.displaced":
            injected = True
            foreign_source = paths.manifests / "foreign.report.json"
            foreign_source.write_bytes(foreign)
            real_replace(foreign_source, json_path)
        real_rename(source, destination)

    monkeypatch.setattr(report_module.os, "rename", replace_before_claim)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert injected
    preserved = [
        path.read_bytes() for path in (json_path, *failure.value.recovery_paths) if path.is_file()
    ]
    assert foreign in preserved
    assert old_json in preserved
    assert markdown_path.read_bytes() == old_markdown


def test_failed_rollback_verification_keeps_immutable_old_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    old_json = json_path.read_bytes()
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link

    def fail_markdown_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target.name == "report.md":
            raise OSError("injected Markdown publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def corrupt_restored_json(source: Path, destination: Path) -> None:
        real_link(source, destination)
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            json_path.write_bytes(b"corrupt rollback target\n")

    monkeypatch.setattr(report_module, "_publish_report_output", fail_markdown_publication)
    monkeypatch.setattr(report_module.os, "link", corrupt_restored_json)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert any(
        path.is_file() and path.read_bytes() == old_json for path in failure.value.recovery_paths
    )


def test_report_recovers_durable_intent_left_after_json_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_pair = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    crashed = False

    def crash_after_json_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal crashed
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == json_path and not crashed:
            crashed = True
            raise SystemExit("simulated hard exit")
        return published

    monkeypatch.setattr(report_module, "_publish_report_output", crash_after_json_publication)
    with pytest.raises(SystemExit, match="simulated hard exit"):
        build_report(paths, config)

    intent_path = paths.manifests / ".report-output-transaction.json"
    assert crashed
    assert intent_path.is_file()
    assert json_path.read_bytes() != old_pair[0]
    assert markdown_path.read_bytes() == old_pair[1]

    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)
    report = build_report(paths, config)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report.to_dict()
    assert "Human review minutes / audio minute | 1.500000" in markdown_path.read_text(
        encoding="utf-8"
    )
    assert not intent_path.exists()


def test_report_rejects_malformed_recovery_intent_before_evidence_validation(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    outputs = (paths.manifests / "report.json", paths.manifests / "report.md")
    before = tuple(path.read_bytes() for path in outputs)
    telemetry_path.unlink()
    intent_path = paths.manifests / ".report-output-transaction.json"
    intent_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert tuple(path.read_bytes() for path in outputs) == before
    assert intent_path in failure.value.recovery_paths
    assert intent_path.is_file()


def test_intent_link_followed_by_directory_sync_failure_recovers_without_dangling_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    outputs = (paths.manifests / "report.json", paths.manifests / "report.md")
    before = tuple(path.read_bytes() for path in outputs)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    intent_path = paths.manifests / ".report-output-transaction.json"
    real_sync = report_module._sync_report_directory
    injected = False

    def fail_first_sync_after_intent_link(directory: Path) -> None:
        nonlocal injected
        if intent_path.is_file() and not injected:
            injected = True
            raise OSError("injected intent directory sync failure")
        real_sync(directory)

    monkeypatch.setattr(report_module, "_sync_report_directory", fail_first_sync_after_intent_link)

    with pytest.raises(OSError, match="intent directory sync failure"):
        build_report(paths, config)

    assert injected
    assert tuple(path.read_bytes() for path in outputs) == before
    assert not intent_path.exists()
    assert not tuple(paths.manifests.glob(".report-output-transaction.*"))


def test_first_report_recovers_durable_intent_left_after_json_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    real_publish = report_module._publish_report_output
    crashed = False

    def crash_after_json_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal crashed
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == json_path and not crashed:
            crashed = True
            raise SystemExit("simulated first-report hard exit")
        return published

    monkeypatch.setattr(report_module, "_publish_report_output", crash_after_json_publication)
    with pytest.raises(SystemExit, match="first-report hard exit"):
        build_report(paths, config)

    intent_path = paths.manifests / ".report-output-transaction.json"
    assert intent_path.is_file()
    assert json_path.is_file()
    assert not markdown_path.exists()

    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)
    report = build_report(paths, config)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report.to_dict()
    assert markdown_path.is_file()
    assert not intent_path.exists()


def test_report_recovery_rejects_unregistered_private_transaction_path_before_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    json_path = paths.manifests / "report.json"
    real_publish = report_module._publish_report_output

    def crash_after_json_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == json_path:
            raise SystemExit("simulated private-path crash")
        return published

    monkeypatch.setattr(report_module, "_publish_report_output", crash_after_json_publication)
    with pytest.raises(SystemExit, match="private-path crash"):
        build_report(paths, config)

    intent_path = paths.manifests / ".report-output-transaction.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    private_directory = paths.manifests / intent["transaction_directory"]
    unexpected = private_directory / "unregistered.evidence"
    unexpected.write_bytes(b"must not be silently removed")
    telemetry_path.unlink()
    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert "private" in str(failure.value.__cause__)
    assert unexpected.is_file()
    assert intent_path.is_file()


def test_report_output_cleanup_failure_does_not_mask_recovery_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link
    real_unlink = type(telemetry_path).unlink

    def fail_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def fail_json_restore(source: Path, destination: Path) -> None:
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            raise OSError("injected rollback failure")
        real_link(source, destination)

    def fail_report_temporary_cleanup(self: Path, missing_ok: bool = False) -> None:
        if self.parent == paths.manifests and self.name.startswith(".report."):
            raise OSError("injected cleanup failure")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_publication)
    monkeypatch.setattr(report_module.os, "link", fail_json_restore)
    monkeypatch.setattr(type(telemetry_path), "unlink", fail_report_temporary_cleanup)

    with pytest.raises(OSError, match="rollback failed") as failure:
        build_report(paths, config)

    assert isinstance(failure.value.__cause__, OSError)
    assert "injected rollback failure" in str(failure.value.__cause__)
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


@pytest.mark.parametrize("failed_prepare_call", (2, 3, 4))
def test_report_output_pair_preserves_existing_files_when_preparation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_prepare_call: int,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_prepare = report_module._prepare_atomic
    calls = 0

    def fail_prepare(path: Path, content: bytes) -> Path:
        nonlocal calls
        calls += 1
        if calls == failed_prepare_call:
            raise OSError("injected preparation failure")
        return real_prepare(path, content)

    monkeypatch.setattr(report_module, "_prepare_atomic", fail_prepare)

    with pytest.raises(OSError, match="injected preparation failure"):
        build_report(paths, config)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before
    assert not tuple(paths.manifests.glob(".report.*"))


def test_first_report_failure_leaves_neither_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    markdown_path = paths.manifests / "report.md"
    real_link = report_module.os.link

    def fail_markdown(source: Path, destination: Path) -> None:
        if Path(destination) == markdown_path:
            raise OSError("injected first publication failure")
        real_link(source, destination)

    monkeypatch.setattr(report_module.os, "link", fail_markdown)

    with pytest.raises(OSError, match="injected first publication failure"):
        build_report(paths, config)

    assert not (paths.manifests / "report.json").exists()
    assert not markdown_path.exists()
    assert not tuple(paths.manifests.glob(".report.*"))
