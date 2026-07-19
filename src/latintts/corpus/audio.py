from __future__ import annotations

import array
import hashlib
import math
import os
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import RecordingRecord

RunCommand = Callable[..., CompletedProcess[str]]
_SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class PcmMetrics:
    sample_count: int
    peak: float
    rms: float
    clipped_sample_ratio: float
    silent_sample_ratio: float

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("PCM metrics sample_count must be a positive integer")
        values = (self.peak, self.rms, self.clipped_sample_ratio, self.silent_sample_ratio)
        if any(type(value) not in (int, float) for value in values):
            raise TypeError("PCM metrics values must be numbers")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("PCM metrics values must be finite ratios from zero through one")


@dataclass(frozen=True, slots=True)
class DerivedAudio:
    relative_path: str
    sha256: str
    source_sha256: str
    config_sha256: str
    metrics: PcmMetrics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure_pcm16(path: Path, silence_threshold: int = 327) -> PcmMetrics:
    if type(silence_threshold) is not int or not 0 <= silence_threshold <= 32768:
        raise ValueError("silence_threshold must be an integer from 0 through 32768")
    total = squared = clipped = silent = peak = 0
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("analysis WAV must be mono PCM16")
        while frames := handle.readframes(65536):
            samples = array.array("h")
            samples.frombytes(frames)
            if sys.byteorder != "little":
                samples.byteswap()
            for sample in samples:
                magnitude = abs(sample)
                total += 1
                squared += sample * sample
                peak = max(peak, magnitude)
                clipped += magnitude >= 32760
                silent += magnitude <= silence_threshold
    if total == 0:
        raise ValueError("analysis WAV contains no samples")
    return PcmMetrics(
        sample_count=total,
        peak=min(peak / 32768.0, 1.0),
        rms=math.sqrt(squared / total) / 32768.0,
        clipped_sample_ratio=clipped / total,
        silent_sample_ratio=silent / total,
    )


def _cache_digest(destination: Path, expected_sha256: str | None) -> str | None:
    if destination.exists():
        if expected_sha256 is None:
            raise CorpusFailure(
                "CACHE_ARTIFACT_INVALID",
                f"refusing to overwrite untracked artifact: {destination.name}",
            )
        actual = _sha256_file(destination)
        if actual != expected_sha256:
            raise CorpusFailure(
                "CACHE_ARTIFACT_INVALID",
                f"cached artifact hash mismatch: {destination.name}",
            )
        return actual
    if expected_sha256 is not None:
        raise CorpusFailure(
            "CACHE_ARTIFACT_INVALID",
            f"recorded cache artifact is missing: {destination.name}",
        )
    return None


def _validate_source_hash(path: Path, expected_sha256: str, *, derived: bool) -> None:
    try:
        actual = _sha256_file(path)
    except OSError as error:
        code = "CACHE_ARTIFACT_INVALID" if derived else "INVENTORY_HASH_MISMATCH"
        raise CorpusFailure(code, f"audio source is unavailable: {path.name}") from error
    if actual != expected_sha256:
        code = "CACHE_ARTIFACT_INVALID" if derived else "INVENTORY_HASH_MISMATCH"
        raise CorpusFailure(code, f"audio source hash mismatch: {path.name}")


def _run_ffmpeg(command: list[str], run_command: RunCommand) -> None:
    try:
        run_command(
            command,
            capture_output=True,
            check=True,
            encoding="utf-8",
            text=True,
        )
    except FileNotFoundError as error:
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "ffmpeg not found in PATH") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
        message = stderr or f"ffmpeg exited with status {error.returncode}"
        raise CorpusFailure("AUDIO_QUALITY_REJECTED", message) from error


def _temporary_output(destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    return temporary


def _require_derived_output(paths: CorpusPaths, relative_path: str) -> Path:
    destination = paths.resolve_local(relative_path)
    derived_root = (paths.local_data / "derived").resolve()
    try:
        destination.relative_to(derived_root)
    except ValueError as error:
        raise ValueError("audio output must be under local-data/derived") from error
    return destination


def _analysis_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getnchannels() != 1
                or handle.getsampwidth() != 2
                or handle.getframerate() != 16000
            ):
                raise ValueError("analysis WAV must be 16 kHz mono PCM16")
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError("analysis WAV is invalid") from error
    if frame_count == 0:
        raise ValueError("analysis WAV contains no samples")
    return frame_count / frame_rate


def _validate_analysis_output(path: Path) -> PcmMetrics:
    _analysis_duration(path)
    return measure_pcm16(path)


def _validate_review_output(path: Path, sample_rate: int, channels: int) -> None:
    try:
        with wave.open(str(path), "rb") as handle:
            valid = (
                handle.getnchannels() == channels
                and handle.getsampwidth() == 3
                and handle.getframerate() == sample_rate
                and handle.getnframes() > 0
            )
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError("review WAV is invalid") from error
    if not valid:
        raise ValueError("review WAV must preserve layout and use PCM24")


def _validate_flac_output(path: Path) -> None:
    try:
        marker = path.read_bytes()[:4]
    except OSError as error:
        raise ValueError("lossless segment is invalid") from error
    if marker != b"fLaC":
        raise ValueError("lossless segment is invalid")


def _validate_segment_bounds(start_seconds: float, end_seconds: float, duration: float) -> None:
    for value in (start_seconds, end_seconds):
        if type(value) not in (int, float):
            raise TypeError("segment boundaries must be numbers")
        if not math.isfinite(value):
            raise ValueError("segment boundaries must be finite")
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("segment must be a positive range starting at or after zero")
    if end_seconds > duration:
        raise ValueError("segment end exceeds audio duration")


def _extract_atomic(
    *,
    source: Path,
    destination: Path,
    start_seconds: float,
    end_seconds: float,
    codec_arguments: list[str],
    expected_sha256: str | None,
    validate_output: Callable[[Path], object],
    run_command: RunCommand,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached_digest = _cache_digest(destination, expected_sha256)
    if cached_digest is not None:
        return cached_digest
    temporary = _temporary_output(destination)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ss",
        str(start_seconds),
        "-to",
        str(end_seconds),
        "-vn",
        *codec_arguments,
        str(temporary),
    ]
    try:
        _run_ffmpeg(command, run_command)
        try:
            validate_output(temporary)
        except (TypeError, ValueError) as error:
            raise CorpusFailure("AUDIO_QUALITY_REJECTED", str(error)) from error
        digest = _sha256_file(temporary)
        os.replace(temporary, destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def derive_analysis_audio(
    record: RecordingRecord,
    paths: CorpusPaths,
    config: CorpusConfig,
    *,
    cached: DerivedAudio | None = None,
    run_command: RunCommand = subprocess.run,
) -> DerivedAudio:
    source = paths.resolve_local(record.relative_path)
    destination = (paths.normalized / f"{record.recording_id}.wav").resolve()
    if destination.parent != paths.normalized.resolve():
        raise ValueError("analysis output must be directly under normalized")
    _validate_source_hash(source, record.sha256, derived=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if cached is not None:
        expected_path = paths.relative_local(destination)
        if (
            cached.relative_path != expected_path
            or cached.source_sha256 != record.sha256
            or cached.config_sha256 != config.digest
            or len(cached.sha256) != _SHA256_LENGTH
        ):
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "cached analysis metadata mismatch")
        _cache_digest(destination, cached.sha256)
        try:
            current_metrics = _validate_analysis_output(destination)
        except (TypeError, ValueError) as error:
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", str(error)) from error
        if current_metrics != cached.metrics:
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "cached analysis metrics mismatch")
        return cached
    _cache_digest(destination, None)

    temporary = _temporary_output(destination)
    command = [
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
        str(temporary),
    ]
    try:
        _run_ffmpeg(command, run_command)
        try:
            metrics = _validate_analysis_output(temporary)
        except (TypeError, ValueError) as error:
            raise CorpusFailure("AUDIO_QUALITY_REJECTED", str(error)) from error
        digest = _sha256_file(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return DerivedAudio(
        relative_path=paths.relative_local(destination),
        sha256=digest,
        source_sha256=record.sha256,
        config_sha256=config.digest,
        metrics=metrics,
    )


def extract_analysis_segment(
    analysis: DerivedAudio,
    paths: CorpusPaths,
    relative_path: str,
    start_seconds: float,
    end_seconds: float,
    *,
    expected_sha256: str | None = None,
    run_command: RunCommand = subprocess.run,
) -> str:
    source = paths.resolve_local(analysis.relative_path)
    destination = _require_derived_output(paths, relative_path)
    if destination.suffix.lower() != ".wav":
        raise ValueError("analysis segment output must use .wav")
    _validate_source_hash(source, analysis.sha256, derived=True)
    try:
        duration = _analysis_duration(source)
    except ValueError as error:
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", str(error)) from error
    _validate_segment_bounds(start_seconds, end_seconds, duration)
    return _extract_atomic(
        source=source,
        destination=destination,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        codec_arguments=["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"],
        expected_sha256=expected_sha256,
        validate_output=_validate_analysis_output,
        run_command=run_command,
    )


def extract_review_wav(
    record: RecordingRecord,
    paths: CorpusPaths,
    relative_path: str,
    start_seconds: float,
    end_seconds: float,
    *,
    expected_sha256: str | None = None,
    run_command: RunCommand = subprocess.run,
) -> str:
    source = paths.resolve_local(record.relative_path)
    destination = _require_derived_output(paths, relative_path)
    if destination.suffix.lower() != ".wav":
        raise ValueError("review output must use .wav")
    _validate_source_hash(source, record.sha256, derived=False)
    _validate_segment_bounds(start_seconds, end_seconds, record.metadata.duration_seconds)
    return _extract_atomic(
        source=source,
        destination=destination,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        codec_arguments=["-c:a", "pcm_s24le"],
        expected_sha256=expected_sha256,
        validate_output=lambda path: _validate_review_output(
            path, record.metadata.sample_rate, record.metadata.channels
        ),
        run_command=run_command,
    )


def extract_lossless_segment(
    record: RecordingRecord,
    paths: CorpusPaths,
    relative_path: str,
    start_seconds: float,
    end_seconds: float,
    *,
    expected_sha256: str | None = None,
    run_command: RunCommand = subprocess.run,
) -> str:
    source = paths.resolve_local(record.relative_path)
    destination = _require_derived_output(paths, relative_path)
    if destination.suffix.lower() != ".flac":
        raise ValueError("lossless segment output must use .flac")
    _validate_source_hash(source, record.sha256, derived=False)
    _validate_segment_bounds(start_seconds, end_seconds, record.metadata.duration_seconds)
    return _extract_atomic(
        source=source,
        destination=destination,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        codec_arguments=["-c:a", "flac", "-compression_level", "5"],
        expected_sha256=expected_sha256,
        validate_output=_validate_flac_output,
        run_command=run_command,
    )
