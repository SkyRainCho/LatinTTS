from __future__ import annotations

import hashlib
import json
import struct
import threading
import wave
from dataclasses import replace
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

import pytest

import latintts.corpus.audio as audio_module
from latintts.corpus.audio import (
    DerivedAudio,
    PcmMetrics,
    derive_analysis_audio,
    extract_analysis_segment,
    extract_lossless_segment,
    extract_review_wav,
)
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import RecordingRecord
from tests.corpus.factories import recording

FFMPEG_VERSION = "ffmpeg version 7.1.1-static"


def _write_pcm(
    path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
    frames: int = 16000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        if sample_width == 2:
            samples = (0,) * (frames * channels)
            handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        else:
            handle.writeframes(b"\x00" * frames * channels * sample_width)


def _fixture(tmp_path: Path) -> tuple[CorpusPaths, RecordingRecord, CorpusConfig, Path]:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    source = paths.raw_spoken / "source.wav"
    source.write_bytes(b"raw-source")
    record = replace(
        recording("rec-1", 1.0),
        relative_path=paths.relative_local(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    config = CorpusConfig({"analysis": {"sample_rate": 16000}}, "c" * 64)
    return paths, record, config, source


def _analysis_runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
    frames = 16000
    if "-ss" in command:
        start = float(command[command.index("-ss") + 1])
        end = float(command[command.index("-to") + 1])
        frames = round((end - start) * 16000)
    _write_pcm(Path(command[-1]), frames=frames)
    return CompletedProcess(command, 0, "", "")


def _derive(
    paths: CorpusPaths,
    record: RecordingRecord,
    config: CorpusConfig,
    *,
    version: str = FFMPEG_VERSION,
) -> DerivedAudio:
    return derive_analysis_audio(
        record,
        paths,
        config,
        ffmpeg_version=version,
        run_command=_analysis_runner,
    )


def test_analysis_cache_key_and_path_bind_source_config_and_ffmpeg_version(
    tmp_path: Path,
) -> None:
    paths, record, config, _ = _fixture(tmp_path)

    first = _derive(paths, record, config, version="ffmpeg version A")
    second = _derive(paths, record, config, version="ffmpeg version B")

    assert first.cache_key != second.cache_key
    assert first.relative_path != second.relative_path
    assert first.source_relative_path == record.relative_path
    assert first.source_sha256 == record.sha256
    assert first.config_sha256 == config.digest
    assert first.ffmpeg_version == "ffmpeg version A"
    assert first.cache_key in first.relative_path


def test_fixed_normalized_root_alias_is_rejected_before_temp_or_runner(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    aliased = replace(paths, normalized=paths.raw_spoken)
    calls = 0

    def forbidden(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        nonlocal calls
        calls += 1
        raise AssertionError("unsafe roots must fail before tool execution")

    with pytest.raises(ValueError, match="fixed corpus root"):
        derive_analysis_audio(
            record,
            aliased,
            config,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=forbidden,
        )

    assert calls == 0
    assert not tuple(paths.raw_spoken.glob(".*.lock"))


def test_symlinked_normalized_root_is_rejected_before_runner(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    paths.normalized.rmdir()
    try:
        paths.normalized.symlink_to(paths.raw_spoken, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="alias"):
        _derive(paths, record, config)


def test_raw_source_chain_must_match_recording_content_type(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    wrong = paths.raw_sung / "wrong.wav"
    wrong.write_bytes(b"raw-source")
    record = replace(record, relative_path=paths.relative_local(wrong))

    with pytest.raises(ValueError, match="source chain"):
        _derive(paths, record, config)


def test_source_hash_is_rechecked_after_ffmpeg_before_publish(tmp_path: Path) -> None:
    paths, record, config, source = _fixture(tmp_path)

    def mutating_runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        _write_pcm(Path(command[-1]))
        source.write_bytes(b"changed-during-decode")
        return CompletedProcess(command, 0, "", "")

    with pytest.raises(CorpusFailure) as error:
        derive_analysis_audio(
            record,
            paths,
            config,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=mutating_runner,
        )

    assert error.value.code == "INVENTORY_HASH_MISMATCH"
    assert not tuple(paths.normalized.glob("*.wav"))
    assert not tuple(paths.normalized.glob("*.lock"))


def test_candidate_source_must_be_canonical_normalized_artifact(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)
    copied = paths.segments / "copied.wav"
    copied.parent.mkdir(parents=True, exist_ok=True)
    copied.write_bytes(paths.resolve_local(analysis.relative_path).read_bytes())
    object.__setattr__(analysis, "relative_path", paths.relative_local(copied))

    with pytest.raises((TypeError, ValueError, CorpusFailure), match=r"source chain|metadata"):
        extract_analysis_segment(
            analysis,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=_analysis_runner,
        )


def test_clip_cache_key_binds_time_mode_source_output_config_and_version(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)

    first = extract_analysis_segment(
        analysis,
        paths,
        0.1,
        0.5,
        ffmpeg_version="ffmpeg version A",
        run_command=_analysis_runner,
    )
    second = extract_analysis_segment(
        analysis,
        paths,
        0.2,
        0.5,
        ffmpeg_version="ffmpeg version A",
        run_command=_analysis_runner,
    )
    third = extract_analysis_segment(
        analysis,
        paths,
        0.1,
        0.5,
        ffmpeg_version="ffmpeg version B",
        run_command=_analysis_runner,
    )

    assert len({first.cache_key, second.cache_key, third.cache_key}) == 3
    assert first.mode == "candidate"
    assert first.start_seconds == 0.1
    assert first.end_seconds == 0.5
    assert first.source_relative_path == analysis.relative_path
    assert first.output_config_sha256 == analysis.output_config_sha256


def test_cache_hit_revalidates_hash_format_and_metrics(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)
    candidate = extract_analysis_segment(
        analysis,
        paths,
        0.1,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=_analysis_runner,
    )
    output = paths.resolve_local(candidate.relative_path)
    _write_pcm(output, sample_width=3, frames=6400)
    forged = replace(candidate, sha256=hashlib.sha256(output.read_bytes()).hexdigest())

    with pytest.raises(CorpusFailure) as error:
        extract_analysis_segment(
            analysis,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            cached=forged,
            run_command=_analysis_runner,
        )

    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_candidate_cache_rejects_valid_pcm16_with_wrong_duration(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)
    candidate = extract_analysis_segment(
        analysis,
        paths,
        0.1,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=_analysis_runner,
    )
    output = paths.resolve_local(candidate.relative_path)
    _write_pcm(output, frames=16000)
    forged = replace(
        candidate,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        metrics=audio_module.measure_pcm16(output),
    )

    with pytest.raises(CorpusFailure) as error:
        extract_analysis_segment(
            analysis,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            cached=forged,
            run_command=_analysis_runner,
        )

    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_review_cache_cannot_reuse_pcm16_artifact(tmp_path: Path) -> None:
    paths, record, _, _ = _fixture(tmp_path)

    def review_runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        _write_pcm(
            Path(command[-1]),
            sample_rate=record.metadata.sample_rate,
            channels=record.metadata.channels,
            sample_width=3,
            frames=19200,
        )
        return CompletedProcess(command, 0, "", "")

    review = extract_review_wav(
        record,
        paths,
        0.1,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=review_runner,
    )
    output = paths.resolve_local(review.relative_path)
    _write_pcm(output, frames=6400)
    forged = replace(review, sha256=hashlib.sha256(output.read_bytes()).hexdigest())

    with pytest.raises(CorpusFailure) as error:
        extract_review_wav(
            record,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            cached=forged,
            run_command=review_runner,
        )

    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_cache_read_io_error_maps_to_cache_artifact_invalid(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)
    output = paths.resolve_local(analysis.relative_path)
    output.unlink()
    output.mkdir()

    with pytest.raises(CorpusFailure) as error:
        derive_analysis_audio(
            record,
            paths,
            config,
            ffmpeg_version=FFMPEG_VERSION,
            cached=analysis,
            run_command=_analysis_runner,
        )

    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def _flac_probe_payload(*, duration: float = 0.4) -> str:
    return json.dumps(
        {
            "format": {"duration": str(duration)},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "flac",
                    "sample_rate": "48000",
                    "channels": 1,
                }
            ],
        }
    )


def test_lossless_segment_is_validated_by_safe_ffprobe(tmp_path: Path) -> None:
    paths, record, _, _ = _fixture(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_ffmpeg(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"fLaCsynthetic")
        return CompletedProcess(command, 0, "", "")

    def fake_ffprobe(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, _flac_probe_payload(), "")

    derived = extract_lossless_segment(
        record,
        paths,
        0.1,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=fake_ffmpeg,
        probe_command=fake_ffprobe,
    )

    command, kwargs = calls[0]
    assert command[0] == "ffprobe"
    assert isinstance(command, list)
    assert command[-1].endswith(".flac")
    assert kwargs.get("shell") is None
    assert derived.mode == "lossless"


@pytest.mark.parametrize("failure", ("malformed", "failed", "wrong-layout"))
def test_lossless_probe_failure_never_publishes(tmp_path: Path, failure: str) -> None:
    paths, record, _, _ = _fixture(tmp_path)

    def fake_ffmpeg(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"fLaCsynthetic")
        return CompletedProcess(command, 0, "", "")

    def bad_probe(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        if failure == "failed":
            raise CalledProcessError(1, command, stderr="probe failed")
        if failure == "wrong-layout":
            return CompletedProcess(command, 0, _flac_probe_payload(duration=0.8), "")
        return CompletedProcess(command, 0, "not-json", "")

    with pytest.raises(CorpusFailure) as error:
        extract_lossless_segment(
            record,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=fake_ffmpeg,
            probe_command=bad_probe,
        )

    assert error.value.code == "AUDIO_QUALITY_REJECTED"
    assert not tuple((paths.segments / "lossless").glob("*.flac"))


def test_lossless_malformed_stream_maps_to_stable_audio_failure(tmp_path: Path) -> None:
    paths, record, _, _ = _fixture(tmp_path)

    def fake_ffmpeg(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"synthetic")
        return CompletedProcess(command, 0, "", "")

    def malformed_probe(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        payload = {"format": {"duration": "0.4"}, "streams": [None]}
        return CompletedProcess(command, 0, json.dumps(payload), "")

    with pytest.raises(CorpusFailure) as error:
        extract_lossless_segment(
            record,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=fake_ffmpeg,
            probe_command=malformed_probe,
        )
    assert error.value.code == "AUDIO_QUALITY_REJECTED"


def test_concurrent_same_key_has_one_producer_and_no_overwrite(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    producer_entered = threading.Event()
    release_producer = threading.Event()
    calls = 0
    results: list[DerivedAudio] = []
    errors: list[BaseException] = []

    def blocking_runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        nonlocal calls
        calls += 1
        producer_entered.set()
        assert release_producer.wait(timeout=2)
        _write_pcm(Path(command[-1]))
        return CompletedProcess(command, 0, "", "")

    def worker() -> None:
        try:
            results.append(
                derive_analysis_audio(
                    record,
                    paths,
                    config,
                    ffmpeg_version=FFMPEG_VERSION,
                    run_command=blocking_runner,
                )
            )
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    assert producer_entered.wait(timeout=2)
    second.start()
    release_producer.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not errors
    assert calls == 1
    assert len(results) == 2
    assert results[0] == results[1]
    assert not tuple(paths.normalized.glob("*.lock"))


def test_derived_audio_strict_round_trip_and_exact_fields(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    derived = _derive(paths, record, config)

    assert DerivedAudio.from_dict(derived.to_dict()) == derived
    unknown = derived.to_dict() | {"unknown": True}
    with pytest.raises(ValueError, match="exact fields"):
        DerivedAudio.from_dict(unknown)
    with pytest.raises(ValueError, match="SHA-256"):
        DerivedAudio.from_dict(derived.to_dict() | {"cache_key": "not-a-digest"})
    with pytest.raises(ValueError, match="canonical"):
        DerivedAudio.from_dict(derived.to_dict() | {"relative_path": "../escaped.wav"})
    with pytest.raises(ValueError, match="ffmpeg_version"):
        DerivedAudio.from_dict(derived.to_dict() | {"ffmpeg_version": ""})


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"schema_version": "2"}, "schema_version"),
        ({"mode": "other"}, "mode"),
        ({"relative_path": ""}, "canonical"),
        ({"start_seconds": 0.0}, "boundaries"),
        ({"metrics": None}, "metrics"),
        ({"source_relative_path": "derived/corpus-v1/normalized/x.wav"}, "raw source"),
        ({"config_sha256": "d" * 64}, "cache_key"),
    ),
)
def test_derived_audio_rejects_invalid_runtime_invariants(
    tmp_path: Path, changes: dict[str, object], error: str
) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    raw = _derive(paths, record, config).to_dict() | changes
    with pytest.raises((TypeError, ValueError), match=error):
        DerivedAudio.from_dict(raw)


def test_candidate_and_review_derived_audio_enforce_mode_specific_metrics(
    tmp_path: Path,
) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)
    candidate = extract_analysis_segment(
        analysis,
        paths,
        0.1,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=_analysis_runner,
    )
    with pytest.raises(ValueError, match="source path"):
        DerivedAudio.from_dict(
            candidate.to_dict() | {"source_relative_path": "raw/spoken/source.wav"}
        )

    review_raw = candidate.to_dict() | {
        "mode": "review",
        "metrics": candidate.metrics.to_dict() if candidate.metrics is not None else None,
    }
    with pytest.raises(TypeError, match="metrics"):
        DerivedAudio.from_dict(review_raw)


def test_strict_metric_decoders_reject_types_and_fields() -> None:
    with pytest.raises(TypeError, match="PCM metrics"):
        PcmMetrics(1, "bad", 0.0, 0.0, 0.0)  # type: ignore[arg-type]
    raw = PcmMetrics(1, 0.0, 0.0, 0.0, 1.0).to_dict()
    with pytest.raises(ValueError, match="exact fields"):
        PcmMetrics.from_dict(raw | {"unknown": 1})


def test_derived_audio_decoder_rejects_non_object_metrics(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    raw = _derive(paths, record, config).to_dict() | {"metrics": "invalid"}
    with pytest.raises(TypeError, match="object or null"):
        DerivedAudio.from_dict(raw)


def test_cache_rejects_invalid_metadata_provenance_metrics_and_missing_file(
    tmp_path: Path,
) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)
    candidate = extract_analysis_segment(
        analysis,
        paths,
        0.1,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=_analysis_runner,
    )

    object.__setattr__(candidate, "sha256", "invalid")
    with pytest.raises(CorpusFailure) as invalid_metadata:
        extract_analysis_segment(
            analysis,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            cached=candidate,
            run_command=_analysis_runner,
        )
    assert invalid_metadata.value.code == "CACHE_ARTIFACT_INVALID"

    candidate = extract_analysis_segment(
        analysis,
        paths,
        0.15,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=_analysis_runner,
    )
    object.__setattr__(candidate, "metrics", "invalid")
    with pytest.raises(CorpusFailure) as invalid_metrics:
        extract_analysis_segment(
            analysis,
            paths,
            0.15,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            cached=candidate,
            run_command=_analysis_runner,
        )
    assert invalid_metrics.value.code == "CACHE_ARTIFACT_INVALID"

    candidate = extract_analysis_segment(
        analysis,
        paths,
        0.2,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=_analysis_runner,
    )
    wrong_request = extract_analysis_segment(
        analysis,
        paths,
        0.3,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=_analysis_runner,
    )
    with pytest.raises(CorpusFailure) as provenance:
        extract_analysis_segment(
            analysis,
            paths,
            0.3,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            cached=candidate,
            run_command=_analysis_runner,
        )
    assert provenance.value.code == "CACHE_ARTIFACT_INVALID"

    wrong_metrics = replace(
        wrong_request,
        metrics=replace(wrong_request.metrics, sample_count=2)
        if wrong_request.metrics is not None
        else None,
    )
    with pytest.raises(CorpusFailure) as metrics:
        extract_analysis_segment(
            analysis,
            paths,
            0.3,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            cached=wrong_metrics,
            run_command=_analysis_runner,
        )
    assert metrics.value.code == "CACHE_ARTIFACT_INVALID"

    paths.resolve_local(wrong_request.relative_path).unlink()
    with pytest.raises(CorpusFailure) as missing:
        extract_analysis_segment(
            analysis,
            paths,
            0.3,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            cached=wrong_request,
            run_command=_analysis_runner,
        )
    assert missing.value.code == "CACHE_ARTIFACT_INVALID"


def test_invalid_analysis_source_media_maps_to_cache_artifact_invalid(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)
    output = paths.resolve_local(analysis.relative_path)
    _write_pcm(output, sample_rate=8000, frames=8000)
    forged = replace(analysis, sha256=hashlib.sha256(output.read_bytes()).hexdigest())

    with pytest.raises(CorpusFailure) as error:
        extract_analysis_segment(
            forged,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=_analysis_runner,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_candidate_source_hash_is_rechecked_after_ffmpeg(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)
    source = paths.resolve_local(analysis.relative_path)

    def mutating_runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        _write_pcm(Path(command[-1]), frames=6400)
        source.write_bytes(b"changed-analysis")
        return CompletedProcess(command, 0, "", "")

    with pytest.raises(CorpusFailure) as error:
        extract_analysis_segment(
            analysis,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=mutating_runner,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"
    assert not tuple((paths.segments / "candidates").glob("*.wav"))


def test_empty_ffmpeg_version_and_nonnumeric_boundary_are_rejected(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="ffmpeg_version"):
        derive_analysis_audio(
            record,
            paths,
            config,
            ffmpeg_version="",
            run_command=_analysis_runner,
        )
    with pytest.raises(TypeError, match="segment boundaries"):
        extract_review_wav(
            record,
            paths,
            "bad",  # type: ignore[arg-type]
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
        )


def test_missing_ffprobe_and_cache_probe_failure_have_stable_codes(tmp_path: Path) -> None:
    paths, record, _, _ = _fixture(tmp_path)

    def fake_ffmpeg(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"synthetic")
        return CompletedProcess(command, 0, "", "")

    def missing_probe(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise FileNotFoundError("ffprobe")

    with pytest.raises(CorpusFailure) as missing:
        extract_lossless_segment(
            record,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=fake_ffmpeg,
            probe_command=missing_probe,
        )
    assert missing.value.code == "ALIGNER_UNAVAILABLE"

    def good_probe(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(command, 0, _flac_probe_payload(), "")

    cached = extract_lossless_segment(
        record,
        paths,
        0.1,
        0.5,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=fake_ffmpeg,
        probe_command=good_probe,
    )
    with pytest.raises(CorpusFailure) as cache_failure:
        extract_lossless_segment(
            record,
            paths,
            0.1,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
            cached=cached,
            run_command=fake_ffmpeg,
            probe_command=missing_probe,
        )
    assert cache_failure.value.code == "CACHE_ARTIFACT_INVALID"


def test_existing_lock_times_out_without_temp_or_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    monkeypatch.setattr(audio_module, "_LOCK_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(audio_module, "_LOCK_POLL_SECONDS", 0.0)

    # The deterministic key can be learned in an isolated sibling root.
    other_paths, other_record, other_config, _ = _fixture(tmp_path / "other")
    derived = _derive(other_paths, other_record, other_config)
    lock_name = Path(derived.relative_path).name + ".lock"
    lock = paths.normalized / lock_name
    lock.write_text("busy", encoding="utf-8")

    with pytest.raises(CorpusFailure) as error:
        _derive(paths, record, config)
    assert error.value.code == "CACHE_ARTIFACT_INVALID"
    assert lock.exists()
    assert not tuple(paths.normalized.glob(".*.wav"))
