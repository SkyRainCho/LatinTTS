from __future__ import annotations

import hashlib
import json
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

FFMPEG_VERSION = "ffmpeg version 7.1.1-static"


def _write_wav(
    path: Path,
    *,
    samples: tuple[int, ...] = (0, 1000, -1000),
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        if sample_width == 2:
            handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        else:
            handle.writeframes(b"\x00" * max(1, len(samples)) * channels * sample_width)


def _fixture(tmp_path: Path) -> tuple[CorpusPaths, RecordingRecord, CorpusConfig, Path]:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    source = paths.raw_spoken / "recording with spaces.wav"
    source.write_bytes(b"synthetic raw audio")
    record = replace(
        recording("rec-1", 1.0),
        relative_path=paths.relative_local(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    return paths, record, CorpusConfig({"analysis": {}}, "c" * 64), source


def _analysis_runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
    frames = 16000
    if "-ss" in command:
        start = float(command[command.index("-ss") + 1])
        end = float(command[command.index("-to") + 1])
        frames = round((end - start) * 16000)
    _write_wav(Path(command[-1]), samples=(0,) * frames)
    return CompletedProcess(command, 0, "", "")


def _derive(paths: CorpusPaths, record: RecordingRecord, config: CorpusConfig) -> DerivedAudio:
    return derive_analysis_audio(
        record,
        paths,
        config,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=_analysis_runner,
    )


def test_measure_pcm16_reports_peak_rms_clipping_and_silence(tmp_path: Path) -> None:
    path = tmp_path / "metrics.wav"
    _write_wav(path, samples=(0, 0, 1000, -1000, 32767, -32768))

    metrics = measure_pcm16(path)

    assert metrics.sample_count == 6
    assert metrics.peak == 1.0
    assert metrics.clipped_sample_ratio == 2 / 6
    assert metrics.silent_sample_ratio == 2 / 6
    assert 0.57 < metrics.rms < 0.59


@pytest.mark.parametrize(("channels", "width"), ((2, 2), (1, 3)))
def test_measure_pcm16_rejects_multichannel_and_sample_width(
    tmp_path: Path, channels: int, width: int
) -> None:
    path = tmp_path / "wrong.wav"
    _write_wav(path, channels=channels, sample_width=width)
    with pytest.raises(ValueError, match="mono PCM16"):
        measure_pcm16(path)


def test_measure_pcm16_rejects_empty_and_invalid_threshold(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    _write_wav(path, samples=())
    with pytest.raises(ValueError, match="no samples"):
        measure_pcm16(path)
    with pytest.raises(ValueError, match="silence_threshold"):
        measure_pcm16(path, -1)


@pytest.mark.parametrize(
    "values",
    (
        (0, 0.0, 0.0, 0.0, 0.0),
        (1, float("nan"), 0.0, 0.0, 0.0),
        (1, 0.0, float("inf"), 0.0, 0.0),
        (1, 0.0, 0.0, -0.1, 0.0),
        (1, 0.0, 0.0, 0.0, 1.1),
    ),
)
def test_pcm_metrics_reject_invalid_values(
    values: tuple[int, float, float, float, float],
) -> None:
    with pytest.raises((TypeError, ValueError), match="PCM metrics"):
        PcmMetrics(*values)


def test_derive_analysis_audio_uses_argument_array_and_atomic_temp(tmp_path: Path) -> None:
    paths, record, config, source = _fixture(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append((command, kwargs))
        temporary = Path(command[-1])
        assert temporary.parent == paths.normalized
        assert not tuple(paths.normalized.glob("analysis-*.wav"))
        _write_wav(temporary)
        return CompletedProcess(command, 0, "", "")

    derived = derive_analysis_audio(
        record,
        paths,
        config,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=fake_run,
    )

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
    output = paths.resolve_local(derived.relative_path)
    assert output.exists()
    assert not Path(command[-1]).exists()
    assert derived.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert derived.metrics is not None and derived.metrics.sample_count == 3


def test_analysis_cache_hit_revalidates_and_never_reruns(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    cached = _derive(paths, record, config)

    def forbidden(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise AssertionError("cache hit must not run ffmpeg")

    assert (
        derive_analysis_audio(
            record,
            paths,
            config,
            ffmpeg_version=FFMPEG_VERSION,
            cached=cached,
            run_command=forbidden,
        )
        == cached
    )
    with pytest.raises(CorpusFailure) as untracked:
        derive_analysis_audio(
            record,
            paths,
            config,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=forbidden,
        )
    assert untracked.value.code == "CACHE_ARTIFACT_INVALID"
    output = paths.resolve_local(cached.relative_path)
    output.write_bytes(b"tampered")
    with pytest.raises(CorpusFailure) as tampered:
        derive_analysis_audio(
            record,
            paths,
            config,
            ffmpeg_version=FFMPEG_VERSION,
            cached=cached,
            run_command=forbidden,
        )
    assert tampered.value.code == "CACHE_ARTIFACT_INVALID"


def test_analysis_rejects_source_hash_before_runner(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    record = replace(record, sha256="0" * 64)

    def forbidden(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise AssertionError("source mismatch must fail before ffmpeg")

    with pytest.raises(CorpusFailure) as error:
        derive_analysis_audio(
            record,
            paths,
            config,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=forbidden,
        )
    assert error.value.code == "INVENTORY_HASH_MISMATCH"


@pytest.mark.parametrize("failure", ("missing", "failed", "no-output", "invalid"))
def test_analysis_failure_cleans_temp_and_lock(tmp_path: Path, failure: str) -> None:
    paths, record, config, _ = _fixture(tmp_path)

    def failing(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        if failure == "missing":
            raise FileNotFoundError("ffmpeg")
        if failure == "failed":
            raise CalledProcessError(2, command, stderr="decode failed")
        if failure == "invalid":
            Path(command[-1]).write_bytes(b"invalid")
        return CompletedProcess(command, 0, "", "")

    expected = "ALIGNER_UNAVAILABLE" if failure == "missing" else "AUDIO_QUALITY_REJECTED"
    with pytest.raises(CorpusFailure) as error:
        derive_analysis_audio(
            record,
            paths,
            config,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=failing,
        )
    assert error.value.code == expected
    assert not tuple(paths.normalized.glob("*.wav"))
    assert not tuple(paths.normalized.glob("*.lock"))


def test_candidate_command_places_accurate_times_after_input(tmp_path: Path) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)
    source = paths.resolve_local(analysis.relative_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(command)
        _write_wav(Path(command[-1]), samples=(0,) * 8000)
        return CompletedProcess(command, 0, "", "")

    candidate = extract_analysis_segment(
        analysis,
        paths,
        0.25,
        0.75,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=fake_run,
    )

    command = calls[0]
    assert command.index("-i") < command.index("-ss") < command.index("-to")
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("-ss") + 1] == "0.25"
    assert command[command.index("-to") + 1] == "0.75"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert candidate.mode == "candidate"


def test_review_command_preserves_layout_with_pcm24(tmp_path: Path) -> None:
    paths, record, _, source = _fixture(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(command)
        _write_wav(
            Path(command[-1]),
            samples=(0,) * 38400,
            sample_rate=record.metadata.sample_rate,
            channels=record.metadata.channels,
            sample_width=3,
        )
        return CompletedProcess(command, 0, "", "")

    review = extract_review_wav(
        record,
        paths,
        0.1,
        0.9,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=fake_run,
    )

    command = calls[0]
    assert command.index("-i") < command.index("-ss") < command.index("-to")
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("-c:a") + 1] == "pcm_s24le"
    assert "-ac" not in command and "-ar" not in command
    assert review.mode == "review" and review.metrics is None


def test_lossless_command_preserves_layout_and_uses_compression_level_5(
    tmp_path: Path,
) -> None:
    paths, record, _, _ = _fixture(tmp_path)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"synthetic")
        return CompletedProcess(command, 0, "", "")

    def fake_probe(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        payload = {
            "format": {"duration": "0.8"},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "flac",
                    "sample_rate": "48000",
                    "channels": 1,
                    "duration_ts": "38400",
                    "time_base": "1/48000",
                }
            ],
        }
        return CompletedProcess(command, 0, json.dumps(payload), "")

    def fake_decode(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(command, 0, "", "")

    lossless = extract_lossless_segment(
        record,
        paths,
        0.1,
        0.9,
        ffmpeg_version=FFMPEG_VERSION,
        run_command=fake_run,
        probe_command=fake_probe,
        decode_command=fake_decode,
    )

    command = commands[0]
    assert command[command.index("-c:a") + 1] == "flac"
    assert command[command.index("-compression_level") + 1] == "5"
    assert "-ac" not in command and "-ar" not in command
    assert lossless.mode == "lossless"


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
def test_extraction_rejects_invalid_bounds_before_runner(
    tmp_path: Path, start: float, end: float
) -> None:
    paths, record, _, _ = _fixture(tmp_path)

    def forbidden(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise AssertionError("invalid bounds must fail before ffmpeg")

    with pytest.raises((TypeError, ValueError), match="segment"):
        extract_review_wav(
            record,
            paths,
            start,
            end,
            ffmpeg_version=FFMPEG_VERSION,
            run_command=forbidden,
        )


@pytest.mark.parametrize("mode", ("candidate", "review"))
def test_invalid_wav_output_is_not_published(tmp_path: Path, mode: str) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    analysis = _derive(paths, record, config)

    def invalid(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"invalid")
        return CompletedProcess(command, 0, "", "")

    with pytest.raises((ValueError, CorpusFailure)):
        if mode == "candidate":
            extract_analysis_segment(
                analysis,
                paths,
                0.1,
                0.5,
                ffmpeg_version=FFMPEG_VERSION,
                run_command=invalid,
            )
        else:
            extract_review_wav(
                record,
                paths,
                0.1,
                0.5,
                ffmpeg_version=FFMPEG_VERSION,
                run_command=invalid,
            )
    assert not tuple(paths.segments.rglob("*.lock"))


def test_missing_raw_source_maps_to_inventory_hash_mismatch(tmp_path: Path) -> None:
    paths, record, _, source = _fixture(tmp_path)
    source.unlink()
    with pytest.raises(CorpusFailure) as error:
        extract_review_wav(
            record,
            paths,
            0.0,
            0.5,
            ffmpeg_version=FFMPEG_VERSION,
        )
    assert error.value.code == "INVENTORY_HASH_MISMATCH"
