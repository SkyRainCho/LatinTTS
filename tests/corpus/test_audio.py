from __future__ import annotations

import hashlib
import struct
import wave
from dataclasses import replace
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

import pytest

from latintts.corpus.audio import (
    DerivedAudio,
    PcmMetrics,
    derive_analysis_audio,
    extract_analysis_segment,
    extract_lossless_segment,
    extract_review_wav,
    measure_pcm16,
)
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import RecordingRecord
from tests.corpus.factories import recording


def _write_pcm16(
    path: Path,
    samples: tuple[int, ...],
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _config() -> CorpusConfig:
    return CorpusConfig(raw={"analysis": {"sample_rate": 16000}}, digest="c" * 64)


def _record_and_source(tmp_path: Path) -> tuple[CorpusPaths, RecordingRecord, Path]:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    source = paths.raw_spoken / "recording with spaces.wav"
    source.write_bytes(b"synthetic raw audio")
    record = replace(
        recording("rec-1", 1.0),
        relative_path=paths.relative_local(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    return paths, record, source


def test_measure_pcm16_reports_peak_rms_clipping_and_silence(tmp_path: Path) -> None:
    path = tmp_path / "metrics.wav"
    _write_pcm16(path, (0, 0, 1000, -1000, 32767, -32768))

    metrics = measure_pcm16(path)

    assert metrics.sample_count == 6
    assert metrics.peak == 1.0
    assert metrics.clipped_sample_ratio == 2 / 6
    assert metrics.silent_sample_ratio == 2 / 6
    assert 0.57 < metrics.rms < 0.59


def test_derive_analysis_audio_uses_safe_command_and_atomic_publish(tmp_path: Path) -> None:
    paths, record, source = _record_and_source(tmp_path)
    final = paths.normalized / "rec-1.wav"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append((command, kwargs))
        temporary = Path(command[-1])
        assert temporary != final
        assert temporary.parent == final.parent
        assert not final.exists()
        _write_pcm16(temporary, (0, 1000, -1000))
        return CompletedProcess(command, 0, "", "")

    derived = derive_analysis_audio(record, paths, _config(), run_command=fake_run)

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        command[-1],
    ]
    assert kwargs == {
        "capture_output": True,
        "check": True,
        "encoding": "utf-8",
        "text": True,
    }
    assert isinstance(command, list)
    assert final.exists()
    assert not Path(command[-1]).exists()
    assert derived.relative_path == "derived/corpus-v1/normalized/rec-1.wav"
    assert derived.sha256 == hashlib.sha256(final.read_bytes()).hexdigest()
    assert derived.source_sha256 == record.sha256
    assert derived.config_sha256 == _config().digest
    assert derived.metrics.sample_count == 3


@pytest.mark.parametrize(
    ("channels", "sample_width", "message"),
    ((2, 2, "mono PCM16"), (1, 3, "mono PCM16")),
)
def test_measure_pcm16_rejects_multichannel_and_non_pcm16_wav(
    tmp_path: Path,
    channels: int,
    sample_width: int,
    message: str,
) -> None:
    path = tmp_path / "wrong-format.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(16000)
        handle.writeframes(b"\x00" * sample_width * channels)

    with pytest.raises(ValueError, match=message):
        measure_pcm16(path)


def test_measure_pcm16_rejects_empty_wav_and_invalid_silence_threshold(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    _write_pcm16(path, ())

    with pytest.raises(ValueError, match="no samples"):
        measure_pcm16(path)
    with pytest.raises(ValueError, match="silence_threshold"):
        measure_pcm16(path, silence_threshold=-1)


@pytest.mark.parametrize(
    "metrics",
    (
        (0, 0.0, 0.0, 0.0, 0.0),
        (1, float("nan"), 0.0, 0.0, 0.0),
        (1, 0.0, float("inf"), 0.0, 0.0),
        (1, 0.0, 0.0, -0.1, 0.0),
        (1, 0.0, 0.0, 0.0, 1.1),
    ),
)
def test_pcm_metrics_rejects_empty_nonfinite_and_out_of_range_values(
    metrics: tuple[int, float, float, float, float],
) -> None:
    with pytest.raises((TypeError, ValueError), match="PCM metrics"):
        PcmMetrics(*metrics)


def test_derive_analysis_audio_validates_source_hash_before_running(tmp_path: Path) -> None:
    paths, record, _ = _record_and_source(tmp_path)
    record = replace(record, sha256="0" * 64)
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return CompletedProcess(command, 0, "", "")

    with pytest.raises(CorpusFailure) as error:
        derive_analysis_audio(record, paths, _config(), run_command=fake_run)

    assert error.value.code == "INVENTORY_HASH_MISMATCH"
    assert calls == 0


def test_derive_analysis_audio_rejects_recording_id_path_escape(tmp_path: Path) -> None:
    paths, record, _ = _record_and_source(tmp_path)
    escaped = replace(record, recording_id="../../../../escaped")

    def forbidden_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise AssertionError("unsafe output path must be rejected before ffmpeg")

    with pytest.raises(ValueError, match="normalized"):
        derive_analysis_audio(escaped, paths, _config(), run_command=forbidden_run)

    assert not (tmp_path / "escaped.wav").exists()


def test_derive_analysis_audio_cache_hit_checks_hash_and_does_not_rerun(
    tmp_path: Path,
) -> None:
    paths, record, _ = _record_and_source(tmp_path)

    def producing_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        _write_pcm16(Path(command[-1]), (10, -10))
        return CompletedProcess(command, 0, "", "")

    cached = derive_analysis_audio(record, paths, _config(), run_command=producing_run)

    def forbidden_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise AssertionError("cache hit must not invoke ffmpeg")

    assert (
        derive_analysis_audio(
            record,
            paths,
            _config(),
            cached=cached,
            run_command=forbidden_run,
        )
        == cached
    )
    paths.resolve_local(cached.relative_path).write_bytes(b"tampered")
    with pytest.raises(CorpusFailure) as error:
        derive_analysis_audio(record, paths, _config(), cached=cached, run_command=forbidden_run)
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_derive_analysis_audio_refuses_untracked_overwrite_and_stale_cache(
    tmp_path: Path,
) -> None:
    paths, record, _ = _record_and_source(tmp_path)
    output = paths.normalized / "rec-1.wav"
    _write_pcm16(output, (1,))
    metrics = measure_pcm16(output)
    cached = DerivedAudio(
        paths.relative_local(output),
        hashlib.sha256(output.read_bytes()).hexdigest(),
        record.sha256,
        "d" * 64,
        metrics,
    )

    with pytest.raises(CorpusFailure) as untracked:
        derive_analysis_audio(record, paths, _config())
    assert untracked.value.code == "CACHE_ARTIFACT_INVALID"
    with pytest.raises(CorpusFailure) as stale:
        derive_analysis_audio(record, paths, _config(), cached=cached)
    assert stale.value.code == "CACHE_ARTIFACT_INVALID"


def test_derive_analysis_audio_rejects_cached_metrics_that_do_not_match_file(
    tmp_path: Path,
) -> None:
    paths, record, _ = _record_and_source(tmp_path)

    def producing_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        _write_pcm16(Path(command[-1]), (10, -10))
        return CompletedProcess(command, 0, "", "")

    cached = derive_analysis_audio(record, paths, _config(), run_command=producing_run)
    incorrect = replace(cached, metrics=replace(cached.metrics, sample_count=3))

    with pytest.raises(CorpusFailure) as error:
        derive_analysis_audio(record, paths, _config(), cached=incorrect)

    assert error.value.code == "CACHE_ARTIFACT_INVALID"


@pytest.mark.parametrize("failure", ("missing", "failed", "no-output", "invalid-wav"))
def test_derive_analysis_audio_fails_clearly_without_publishing_partial_output(
    tmp_path: Path,
    failure: str,
) -> None:
    paths, record, _ = _record_and_source(tmp_path)

    def failing_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        if failure == "missing":
            raise FileNotFoundError("ffmpeg")
        if failure == "failed":
            raise CalledProcessError(2, command, stderr="decode failed")
        if failure == "invalid-wav":
            Path(command[-1]).write_bytes(b"not a wave")
        return CompletedProcess(command, 0, "", "")

    expected_code = "ALIGNER_UNAVAILABLE" if failure == "missing" else "AUDIO_QUALITY_REJECTED"
    with pytest.raises(CorpusFailure) as error:
        derive_analysis_audio(record, paths, _config(), run_command=failing_run)

    assert error.value.code == expected_code
    assert not (paths.normalized / "rec-1.wav").exists()
    assert not tuple(paths.normalized.glob(".rec-1-*.wav"))


def _analysis_audio(paths: CorpusPaths) -> tuple[DerivedAudio, Path]:
    source = paths.normalized / "rec-1.wav"
    _write_pcm16(source, (0,) * 16000)
    return (
        DerivedAudio(
            paths.relative_local(source),
            hashlib.sha256(source.read_bytes()).hexdigest(),
            "a" * 64,
            "c" * 64,
            measure_pcm16(source),
        ),
        source,
    )


def test_extract_analysis_segment_uses_post_input_timestamps_and_pcm16(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    analysis, source = _analysis_audio(paths)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        commands.append(command)
        _write_pcm16(Path(command[-1]), (0, 20, -20))
        return CompletedProcess(command, 0, "", "")

    digest = extract_analysis_segment(
        analysis,
        paths,
        "derived/corpus-v1/segments/candidate.wav",
        0.25,
        0.75,
        run_command=fake_run,
    )

    command = commands[0]
    assert command.index("-i") < command.index("-ss") < command.index("-to")
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("-ss") + 1] == "0.25"
    assert command[command.index("-to") + 1] == "0.75"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    output = paths.resolve_local("derived/corpus-v1/segments/candidate.wav")
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("sample_rate", (8000, 16000))
def test_extract_analysis_segment_maps_invalid_source_to_cache_failure(
    tmp_path: Path,
    sample_rate: int,
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    source = paths.normalized / "broken.wav"
    _write_pcm16(source, () if sample_rate == 16000 else (0,), sample_rate=sample_rate)
    analysis = DerivedAudio(
        paths.relative_local(source),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "a" * 64,
        "c" * 64,
        PcmMetrics(1, 0.0, 0.0, 0.0, 1.0),
    )

    with pytest.raises(CorpusFailure) as error:
        extract_analysis_segment(
            analysis,
            paths,
            "derived/corpus-v1/segments/candidate.wav",
            0.0,
            0.1,
        )

    assert error.value.code == "CACHE_ARTIFACT_INVALID"


@pytest.mark.parametrize(
    ("mode", "suffix", "codec"),
    (("review", ".wav", "pcm_s24le"), ("lossless", ".flac", "flac")),
)
def test_raw_extraction_modes_preserve_layout_and_choose_explicit_codec(
    tmp_path: Path,
    mode: str,
    suffix: str,
    codec: str,
) -> None:
    paths, record, source = _record_and_source(tmp_path)
    commands: list[list[str]] = []
    relative = f"derived/corpus-v1/segments/clip{suffix}"

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        commands.append(command)
        output = Path(command[-1])
        if mode == "review":
            with wave.open(str(output), "wb") as handle:
                handle.setnchannels(record.metadata.channels)
                handle.setsampwidth(3)
                handle.setframerate(record.metadata.sample_rate)
                handle.writeframes(b"\x00" * 3 * record.metadata.channels)
        else:
            output.write_bytes(b"fLaCsynthetic")
        return CompletedProcess(command, 0, "", "")

    function = extract_review_wav if mode == "review" else extract_lossless_segment
    digest = function(record, paths, relative, 0.1, 0.9, run_command=fake_run)

    command = commands[0]
    assert command.index("-i") < command.index("-ss") < command.index("-to")
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("-c:a") + 1] == codec
    assert "-ac" not in command
    assert "-ar" not in command
    assert ("-compression_level" in command) is (mode == "lossless")
    if mode == "lossless":
        assert command[command.index("-compression_level") + 1] == "5"
    output = paths.resolve_local(relative)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("start", "end"),
    (
        (-0.1, 0.5),
        (0.5, 0.5),
        (0.7, 0.6),
        (0.0, 1.1),
        (float("nan"), 0.5),
        (0.0, float("inf")),
    ),
)
def test_extraction_rejects_invalid_or_out_of_bounds_times_before_running(
    tmp_path: Path,
    start: float,
    end: float,
) -> None:
    paths, record, _ = _record_and_source(tmp_path)

    def forbidden_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise AssertionError("invalid bounds must be rejected before ffmpeg")

    with pytest.raises((TypeError, ValueError), match="segment"):
        extract_review_wav(
            record,
            paths,
            "derived/corpus-v1/segments/review.wav",
            start,
            end,
            run_command=forbidden_run,
        )


def test_extraction_cache_hit_verifies_recorded_hash_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    analysis, _ = _analysis_audio(paths)
    relative = "derived/corpus-v1/segments/candidate.wav"

    def producing_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        _write_pcm16(Path(command[-1]), (1, 2))
        return CompletedProcess(command, 0, "", "")

    digest = extract_analysis_segment(
        analysis, paths, relative, 0.0, 0.5, run_command=producing_run
    )

    def forbidden_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise AssertionError("cache hit must not invoke ffmpeg")

    assert (
        extract_analysis_segment(
            analysis,
            paths,
            relative,
            0.0,
            0.5,
            expected_sha256=digest,
            run_command=forbidden_run,
        )
        == digest
    )
    output = paths.resolve_local(relative)
    output.write_bytes(b"tampered")
    with pytest.raises(CorpusFailure) as tampered:
        extract_analysis_segment(
            analysis,
            paths,
            relative,
            0.0,
            0.5,
            expected_sha256=digest,
            run_command=forbidden_run,
        )
    assert tampered.value.code == "CACHE_ARTIFACT_INVALID"
    with pytest.raises(CorpusFailure) as untracked:
        extract_analysis_segment(analysis, paths, relative, 0.0, 0.5, run_command=forbidden_run)
    assert untracked.value.code == "CACHE_ARTIFACT_INVALID"

    output.unlink()
    with pytest.raises(CorpusFailure) as missing:
        extract_analysis_segment(
            analysis,
            paths,
            relative,
            0.0,
            0.5,
            expected_sha256=digest,
            run_command=forbidden_run,
        )
    assert missing.value.code == "CACHE_ARTIFACT_INVALID"


@pytest.mark.parametrize("mode", ("analysis", "review", "lossless"))
def test_extraction_rejects_invalid_ffmpeg_output_and_cleans_temporary_file(
    tmp_path: Path,
    mode: str,
) -> None:
    paths, record, _ = _record_and_source(tmp_path)
    analysis, _ = _analysis_audio(paths)
    relative = f"derived/corpus-v1/segments/bad.{'flac' if mode == 'lossless' else 'wav'}"

    def invalid_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"invalid")
        return CompletedProcess(command, 0, "", "")

    with pytest.raises(CorpusFailure) as error:
        if mode == "analysis":
            extract_analysis_segment(analysis, paths, relative, 0.0, 0.5, run_command=invalid_run)
        elif mode == "review":
            extract_review_wav(record, paths, relative, 0.0, 0.5, run_command=invalid_run)
        else:
            extract_lossless_segment(record, paths, relative, 0.0, 0.5, run_command=invalid_run)
    assert error.value.code == "AUDIO_QUALITY_REJECTED"
    assert not paths.resolve_local(relative).exists()
    assert not tuple(paths.resolve_local(relative).parent.glob(".bad-*"))


def test_raw_extraction_rejects_missing_source_with_inventory_hash_failure(
    tmp_path: Path,
) -> None:
    paths, record, source = _record_and_source(tmp_path)
    source.unlink()

    with pytest.raises(CorpusFailure) as error:
        extract_review_wav(
            record,
            paths,
            "derived/corpus-v1/segments/review.wav",
            0.0,
            0.5,
        )

    assert error.value.code == "INVENTORY_HASH_MISMATCH"


def test_extraction_rejects_paths_outside_local_data_and_wrong_extensions(
    tmp_path: Path,
) -> None:
    paths, record, _ = _record_and_source(tmp_path)

    with pytest.raises(ValueError, match="local corpus path"):
        extract_review_wav(record, paths, "../escaped.wav", 0.0, 0.5)
    with pytest.raises(ValueError, match="derived"):
        extract_review_wav(record, paths, "raw/spoken/new-review.wav", 0.0, 0.5)
    with pytest.raises(ValueError, match=r"\.flac"):
        extract_lossless_segment(
            record,
            paths,
            "derived/corpus-v1/segments/not-lossless.wav",
            0.0,
            0.5,
        )
    analysis, _ = _analysis_audio(paths)
    with pytest.raises(ValueError, match=r"\.wav"):
        extract_analysis_segment(
            analysis,
            paths,
            "derived/corpus-v1/segments/not-analysis.flac",
            0.0,
            0.5,
        )
    with pytest.raises(ValueError, match=r"\.wav"):
        extract_review_wav(
            record,
            paths,
            "derived/corpus-v1/segments/not-review.flac",
            0.0,
            0.5,
        )
