from __future__ import annotations

import array
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path, PurePosixPath
from subprocess import CompletedProcess
from typing import Any, Literal

from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.paths import CorpusPaths, require_canonical_descendant
from latintts.corpus.records import RecordingRecord

RunCommand = Callable[..., CompletedProcess[str]]
AudioMode = Literal["analysis", "candidate", "review", "lossless"]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DERIVED_PREFIX = "derived/corpus-v1/"
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.01


def _require_digest(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_canonical_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a canonical POSIX relative path")
    parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{field} must be a canonical POSIX relative path")
    return value


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

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PcmMetrics:
        expected = frozenset(field.name for field in fields(cls))
        if set(raw) != expected:
            raise ValueError("PCM metrics must contain exact fields")
        return cls(**raw)


def _cache_identity(
    *,
    mode: AudioMode,
    source_relative_path: str,
    source_sha256: str,
    config_sha256: str,
    output_config_sha256: str,
    ffmpeg_version: str,
    start_seconds: float | None,
    end_seconds: float | None,
) -> str:
    identity = {
        "schema_version": "1",
        "mode": mode,
        "source_relative_path": source_relative_path,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "output_config_sha256": output_config_sha256,
        "ffmpeg_version": ffmpeg_version,
        "start_seconds": float(start_seconds).hex() if start_seconds is not None else None,
        "end_seconds": float(end_seconds).hex() if end_seconds is not None else None,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_relative_path(mode: AudioMode, cache_key: str) -> str:
    if mode == "analysis":
        return f"{_DERIVED_PREFIX}normalized/analysis-{cache_key}.wav"
    if mode == "candidate":
        return f"{_DERIVED_PREFIX}segments/candidates/candidate-{cache_key}.wav"
    if mode == "review":
        return f"{_DERIVED_PREFIX}segments/review/review-{cache_key}.wav"
    return f"{_DERIVED_PREFIX}segments/lossless/lossless-{cache_key}.flac"


@dataclass(frozen=True, slots=True)
class DerivedAudio:
    schema_version: str
    mode: AudioMode
    relative_path: str
    sha256: str
    source_relative_path: str
    source_sha256: str
    config_sha256: str
    output_config_sha256: str
    ffmpeg_version: str
    cache_key: str
    start_seconds: float | None
    end_seconds: float | None
    metrics: PcmMetrics | None

    def __post_init__(self) -> None:
        if self.schema_version != "1" or type(self.schema_version) is not str:
            raise ValueError("DerivedAudio schema_version must be '1'")
        if self.mode not in ("analysis", "candidate", "review", "lossless") or not isinstance(
            self.mode, str
        ):
            raise ValueError("DerivedAudio mode is invalid")
        relative_path = _require_canonical_relative_path(
            self.relative_path, "DerivedAudio relative_path"
        )
        source_path = _require_canonical_relative_path(
            self.source_relative_path, "DerivedAudio source_relative_path"
        )
        for value, name in (
            (self.sha256, "DerivedAudio sha256"),
            (self.source_sha256, "DerivedAudio source_sha256"),
            (self.config_sha256, "DerivedAudio config_sha256"),
            (self.output_config_sha256, "DerivedAudio output_config_sha256"),
            (self.cache_key, "DerivedAudio cache_key"),
        ):
            _require_digest(value, name)
        if not isinstance(self.ffmpeg_version, str) or not self.ffmpeg_version.strip():
            raise ValueError("DerivedAudio ffmpeg_version must be a non-empty string")
        if self.mode == "analysis":
            if self.start_seconds is not None or self.end_seconds is not None:
                raise ValueError("analysis DerivedAudio must not have segment boundaries")
        else:
            _validate_segment_bounds(self.start_seconds, self.end_seconds, math.inf)
        if self.mode in ("analysis", "candidate"):
            if type(self.metrics) is not PcmMetrics:
                raise TypeError("PCM DerivedAudio metrics must be PcmMetrics")
        elif self.metrics is not None:
            raise TypeError("review and lossless DerivedAudio metrics must be null")
        if relative_path != _artifact_relative_path(self.mode, self.cache_key):
            raise ValueError("DerivedAudio relative_path does not match its cache identity")
        if self.mode == "candidate":
            if not source_path.startswith(f"{_DERIVED_PREFIX}normalized/analysis-"):
                raise ValueError("candidate DerivedAudio source path is not canonical")
        elif not source_path.startswith(("raw/spoken/", "raw/sung/")):
            raise ValueError("DerivedAudio raw source path is not canonical")
        expected_key = _cache_identity(
            mode=self.mode,
            source_relative_path=source_path,
            source_sha256=self.source_sha256,
            config_sha256=self.config_sha256,
            output_config_sha256=self.output_config_sha256,
            ffmpeg_version=self.ffmpeg_version,
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
        )
        if self.cache_key != expected_key:
            raise ValueError("DerivedAudio cache_key does not match provenance")

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["metrics"] = self.metrics.to_dict() if self.metrics is not None else None
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DerivedAudio:
        expected = frozenset(field.name for field in fields(cls))
        if set(raw) != expected:
            raise ValueError("DerivedAudio must contain exact fields")
        metrics_raw = raw["metrics"]
        if metrics_raw is not None and type(metrics_raw) is not dict:
            raise TypeError("DerivedAudio metrics must be an object or null")
        values = dict(raw)
        values["metrics"] = PcmMetrics.from_dict(metrics_raw) if metrics_raw is not None else None
        return cls(**values)


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


def _canonical_output_config(raw: dict[str, object]) -> str:
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _require_fixed_root(actual: Path, expected: Path) -> Path:
    lexical_actual = _lexical(actual)
    lexical_expected = _lexical(expected)
    if lexical_actual != lexical_expected:
        raise ValueError(f"fixed corpus root mismatch: {actual}")
    if actual.exists() and actual.resolve() != lexical_expected:
        raise ValueError(f"fixed corpus root aliases another location: {actual}")
    return lexical_expected


def _validate_fixed_roots(paths: CorpusPaths) -> None:
    project = _lexical(paths.project_root)
    local = _require_fixed_root(paths.local_data, project / "local-data")
    raw = local / "raw"
    derived = local / "derived" / "corpus-v1"
    _require_fixed_root(paths.raw_spoken, raw / "spoken")
    _require_fixed_root(paths.raw_sung, raw / "sung")
    _require_fixed_root(paths.normalized, derived / "normalized")
    _require_fixed_root(paths.segments, derived / "segments")
    _require_fixed_root(paths.alignments, derived / "alignments")
    _require_fixed_root(paths.manifests, local / "manifests")
    for fixed_parent in (local / "derived", derived):
        if fixed_parent.exists() and fixed_parent.resolve() != fixed_parent:
            raise ValueError(f"fixed corpus root aliases another location: {fixed_parent}")


def _prepare_output_root(paths: CorpusPaths, mode: AudioMode) -> Path:
    _validate_fixed_roots(paths)
    if mode == "analysis":
        root = _lexical(paths.normalized)
    else:
        folder = {"candidate": "candidates", "review": "review", "lossless": "lossless"}[mode]
        root = _lexical(paths.segments / folder)
    require_canonical_descendant(_lexical(paths.local_data), root, kind="derived output root")
    root.mkdir(parents=True, exist_ok=True)
    require_canonical_descendant(_lexical(paths.local_data), root, kind="derived output root")
    return root


def _raw_source(record: RecordingRecord, paths: CorpusPaths) -> Path:
    _validate_fixed_roots(paths)
    relative = _require_canonical_relative_path(record.relative_path, "recording source path")
    expected_prefix = f"raw/{record.content_type}/"
    if not relative.startswith(expected_prefix):
        raise ValueError("recording source chain does not match content_type")
    root = _lexical(paths.raw_spoken if record.content_type == "spoken" else paths.raw_sung)
    source = _lexical(paths.local_data / Path(*relative.split("/")))
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("recording source chain escapes its fixed raw root") from error
    if source.exists() and source.resolve() != source:
        raise ValueError("recording source chain contains an alias")
    return source


def _analysis_source(analysis: DerivedAudio, paths: CorpusPaths) -> Path:
    _validate_fixed_roots(paths)
    try:
        validated = DerivedAudio.from_dict(analysis.to_dict())
    except (TypeError, ValueError) as error:
        raise ValueError("candidate source chain metadata is invalid") from error
    if validated.mode != "analysis":
        raise ValueError("candidate source chain requires analysis DerivedAudio")
    source = _lexical(paths.local_data / Path(*validated.relative_path.split("/")))
    normalized = _lexical(paths.normalized)
    if source.parent != normalized:
        raise ValueError("candidate source chain must be directly under normalized")
    if source.exists() and source.resolve() != source:
        raise ValueError("candidate source chain contains an alias")
    return source


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


@contextmanager
def _exclusive_key_lock(destination: Path) -> Iterator[bool]:
    lock = destination.with_suffix(f"{destination.suffix}.lock")
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    contended = False
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            contended = True
            if time.monotonic() >= deadline:
                raise CorpusFailure(
                    "CACHE_ARTIFACT_INVALID", f"timed out waiting for cache lock: {lock.name}"
                ) from error
            time.sleep(_LOCK_POLL_SECONDS)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield contended
    finally:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            lock.unlink()


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


def _validate_analysis_output(path: Path, expected_sample_count: int | None = None) -> PcmMetrics:
    duration = _analysis_duration(path)
    actual_sample_count = round(duration * 16000)
    if expected_sample_count is not None and abs(actual_sample_count - expected_sample_count) > 1:
        raise ValueError("analysis WAV sample count does not match requested segment")
    return measure_pcm16(path)


def _validate_review_output(
    path: Path, sample_rate: int, channels: int, expected_sample_count: int
) -> None:
    try:
        with wave.open(str(path), "rb") as handle:
            valid = (
                handle.getnchannels() == channels
                and handle.getsampwidth() == 3
                and handle.getframerate() == sample_rate
                and handle.getnframes() > 0
                and abs(handle.getnframes() - expected_sample_count) <= 1
            )
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError("review WAV is invalid") from error
    if not valid:
        raise ValueError("review WAV must preserve layout and use PCM24")


def _validate_flac_output(
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    expected_sample_count: int,
    probe_command: RunCommand,
    decode_command: RunCommand,
) -> None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,sample_rate,channels,duration_ts,time_base",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = probe_command(
            command,
            capture_output=True,
            check=True,
            encoding="utf-8",
            text=True,
        )
    except FileNotFoundError as error:
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "ffprobe not found in PATH") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
        message = stderr or f"ffprobe exited with status {error.returncode}"
        raise CorpusFailure("AUDIO_QUALITY_REJECTED", message) from error
    try:
        raw = json.loads(completed.stdout)
        format_data = raw["format"]
        streams = raw["streams"]
        duration = float(format_data["duration"])
        audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
        stream = audio_streams[0]
        stream_sample_rate = int(stream["sample_rate"])
        stream_samples = (
            int(stream["duration_ts"]) * Fraction(stream["time_base"]) * stream_sample_rate
        )
        valid = (
            stream["codec_name"] == "flac"
            and stream_sample_rate == sample_rate
            and type(stream["channels"]) is int
            and stream["channels"] == channels
            and stream_samples.denominator == 1
            and abs(stream_samples.numerator - expected_sample_count) <= 1
            and math.isfinite(duration)
            and duration > 0
            and abs(duration - float(stream_samples / sample_rate)) <= 0.05
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise CorpusFailure("AUDIO_QUALITY_REJECTED", "invalid ffprobe FLAC output") from error
    if not valid:
        raise CorpusFailure("AUDIO_QUALITY_REJECTED", "FLAC media properties do not match")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    try:
        decode_command(
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
        message = stderr or f"ffmpeg decode exited with status {error.returncode}"
        raise CorpusFailure("AUDIO_QUALITY_REJECTED", message) from error


def _validate_segment_bounds(
    start_seconds: float | None, end_seconds: float | None, duration: float
) -> None:
    for value in (start_seconds, end_seconds):
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("segment boundaries must be numbers")
        if not math.isfinite(value):
            raise ValueError("segment boundaries must be finite")
    assert start_seconds is not None
    assert end_seconds is not None
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("segment must be a positive range starting at or after zero")
    if end_seconds > duration:
        raise ValueError("segment end exceeds audio duration")


@dataclass(frozen=True, slots=True)
class _ArtifactPlan:
    mode: AudioMode
    source: Path
    source_relative_path: str
    source_sha256: str
    config_sha256: str
    output_config_sha256: str
    ffmpeg_version: str
    cache_key: str
    destination: Path
    relative_path: str
    start_seconds: float | None
    end_seconds: float | None


def _make_plan(
    *,
    mode: AudioMode,
    paths: CorpusPaths,
    source: Path,
    source_relative_path: str,
    source_sha256: str,
    config_sha256: str,
    output_config_sha256: str,
    ffmpeg_version: str,
    start_seconds: float | None,
    end_seconds: float | None,
    staging_root: Path | None = None,
) -> _ArtifactPlan:
    if not isinstance(ffmpeg_version, str) or not ffmpeg_version.strip():
        raise ValueError("ffmpeg_version must be a non-empty string")
    _require_digest(source_sha256, "source_sha256")
    _require_digest(config_sha256, "config_sha256")
    _require_digest(output_config_sha256, "output_config_sha256")
    cache_key = _cache_identity(
        mode=mode,
        source_relative_path=source_relative_path,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        output_config_sha256=output_config_sha256,
        ffmpeg_version=ffmpeg_version,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    relative_path = _artifact_relative_path(mode, cache_key)
    if staging_root is None:
        root = _prepare_output_root(paths, mode)
        destination = _lexical(paths.local_data / Path(*relative_path.split("/")))
        if destination.parent != root:
            raise ValueError("derived target escapes its fixed mode root")
    else:
        _validate_fixed_roots(paths)
        local_root = _lexical(paths.local_data)
        segments_root = require_canonical_descendant(
            local_root,
            _lexical(paths.segments),
            kind="fixed segments root",
        )
        stage = _lexical(staging_root)
        try:
            stage.relative_to(segments_root)
        except ValueError as error:
            raise ValueError("audio staging root must be inside the fixed segments root") from error
        require_canonical_descendant(segments_root, stage, kind="audio staging root")
        final_root = _lexical(
            paths.normalized
            if mode == "analysis"
            else paths.segments
            / {"candidate": "candidates", "review": "review", "lossless": "lossless"}[mode]
        )
        require_canonical_descendant(local_root, final_root, kind="derived mode root")
        require_canonical_descendant(
            final_root,
            _lexical(paths.local_data / Path(*relative_path.split("/"))),
            kind="final derived target",
        )
        stage.mkdir(parents=True, exist_ok=True)
        require_canonical_descendant(segments_root, stage, kind="audio staging root")
        destination = _lexical(stage / Path(*relative_path.split("/")))
        try:
            destination.relative_to(stage)
        except ValueError as error:
            raise ValueError("staged derived target escapes its transaction root") from error
        require_canonical_descendant(stage, destination, kind="staged derived target")
        destination.parent.mkdir(parents=True, exist_ok=True)
        require_canonical_descendant(stage, destination, kind="staged derived target")
    if destination.is_symlink():
        raise ValueError("derived target is an alias")
    return _ArtifactPlan(
        mode,
        source,
        source_relative_path,
        source_sha256,
        config_sha256,
        output_config_sha256,
        ffmpeg_version,
        cache_key,
        destination,
        relative_path,
        start_seconds,
        end_seconds,
    )


def _metadata_matches(cached: DerivedAudio, plan: _ArtifactPlan) -> bool:
    return (
        cached.schema_version == "1"
        and cached.mode == plan.mode
        and cached.relative_path == plan.relative_path
        and cached.source_relative_path == plan.source_relative_path
        and cached.source_sha256 == plan.source_sha256
        and cached.config_sha256 == plan.config_sha256
        and cached.output_config_sha256 == plan.output_config_sha256
        and cached.ffmpeg_version == plan.ffmpeg_version
        and cached.cache_key == plan.cache_key
        and cached.start_seconds == plan.start_seconds
        and cached.end_seconds == plan.end_seconds
    )


def _build_derived(plan: _ArtifactPlan, sha256: str, metrics: PcmMetrics | None) -> DerivedAudio:
    return DerivedAudio(
        schema_version="1",
        mode=plan.mode,
        relative_path=plan.relative_path,
        sha256=sha256,
        source_relative_path=plan.source_relative_path,
        source_sha256=plan.source_sha256,
        config_sha256=plan.config_sha256,
        output_config_sha256=plan.output_config_sha256,
        ffmpeg_version=plan.ffmpeg_version,
        cache_key=plan.cache_key,
        start_seconds=plan.start_seconds,
        end_seconds=plan.end_seconds,
        metrics=metrics,
    )


def _validate_cached(
    plan: _ArtifactPlan,
    cached: DerivedAudio,
    validate_output: Callable[[Path], PcmMetrics | None],
) -> DerivedAudio:
    try:
        validated = DerivedAudio.from_dict(cached.to_dict())
    except (AttributeError, TypeError, ValueError) as error:
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", "cached metadata is invalid") from error
    if not _metadata_matches(validated, plan):
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", "cached provenance does not match request")
    if not plan.destination.exists() or plan.destination.is_symlink():
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", "recorded cache artifact is missing")
    try:
        actual_hash = _sha256_file(plan.destination)
        actual_metrics = validate_output(plan.destination)
    except (OSError, TypeError, ValueError, CorpusFailure) as error:
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", "cached media validation failed") from error
    if actual_hash != validated.sha256 or actual_metrics != validated.metrics:
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", "cached artifact content mismatch")
    return validated


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise CorpusFailure(
            "CACHE_ARTIFACT_INVALID", "cache artifact appeared during publication"
        ) from error
    temporary.unlink()


def _materialize(
    *,
    plan: _ArtifactPlan,
    cached: DerivedAudio | None,
    command_arguments: list[str],
    validate_output: Callable[[Path], PcmMetrics | None],
    run_command: RunCommand,
    source_is_derived: bool,
    allow_unrecorded_existing: bool = False,
) -> DerivedAudio:
    _validate_source_hash(plan.source, plan.source_sha256, derived=source_is_derived)
    if plan.destination.exists():
        if cached is None:
            if not allow_unrecorded_existing:
                raise CorpusFailure("CACHE_ARTIFACT_INVALID", "cache artifact lacks metadata")
            try:
                metrics = validate_output(plan.destination)
                digest = _sha256_file(plan.destination)
            except (OSError, TypeError, ValueError, CorpusFailure) as error:
                raise CorpusFailure(
                    "CACHE_ARTIFACT_INVALID", "staged cache artifact is invalid"
                ) from error
            return _build_derived(plan, digest, metrics)
        return _validate_cached(plan, cached, validate_output)
    if cached is not None:
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", "recorded cache artifact is missing")

    with _exclusive_key_lock(plan.destination) as contended:
        if plan.destination.exists():
            if not contended:
                raise CorpusFailure("CACHE_ARTIFACT_INVALID", "unexpected cache artifact")
            try:
                metrics = validate_output(plan.destination)
                digest = _sha256_file(plan.destination)
            except (OSError, TypeError, ValueError, CorpusFailure) as error:
                raise CorpusFailure(
                    "CACHE_ARTIFACT_INVALID", "concurrent cache artifact is invalid"
                ) from error
            return _build_derived(plan, digest, metrics)

        temporary = _temporary_output(plan.destination)
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(plan.source),
            *command_arguments,
            str(temporary),
        ]
        try:
            _run_ffmpeg(command, run_command)
            try:
                metrics = validate_output(temporary)
                digest = _sha256_file(temporary)
            except (OSError, TypeError, ValueError) as error:
                raise CorpusFailure("AUDIO_QUALITY_REJECTED", str(error)) from error
            _validate_source_hash(plan.source, plan.source_sha256, derived=source_is_derived)
            derived = _build_derived(plan, digest, metrics)
            _publish_no_replace(temporary, plan.destination)
            return derived
        finally:
            temporary.unlink(missing_ok=True)


def derive_analysis_audio(
    record: RecordingRecord,
    paths: CorpusPaths,
    config: CorpusConfig,
    *,
    ffmpeg_version: str,
    cached: DerivedAudio | None = None,
    run_command: RunCommand = subprocess.run,
) -> DerivedAudio:
    source = _raw_source(record, paths)
    output_config = _canonical_output_config(
        {"codec": "pcm_s16le", "channels": 1, "sample_rate": 16000}
    )
    plan = _make_plan(
        mode="analysis",
        paths=paths,
        source=source,
        source_relative_path=record.relative_path,
        source_sha256=record.sha256,
        config_sha256=config.digest,
        output_config_sha256=output_config,
        ffmpeg_version=ffmpeg_version,
        start_seconds=None,
        end_seconds=None,
    )
    return _materialize(
        plan=plan,
        cached=cached,
        command_arguments=["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"],
        validate_output=_validate_analysis_output,
        run_command=run_command,
        source_is_derived=False,
    )


def extract_analysis_segment(
    analysis: DerivedAudio,
    paths: CorpusPaths,
    start_seconds: float,
    end_seconds: float,
    *,
    ffmpeg_version: str,
    cached: DerivedAudio | None = None,
    run_command: RunCommand = subprocess.run,
    _staging_root: Path | None = None,
) -> DerivedAudio:
    source = _analysis_source(analysis, paths)
    try:
        duration = _analysis_duration(source)
    except ValueError as error:
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", str(error)) from error
    _validate_segment_bounds(start_seconds, end_seconds, duration)
    output_config = _canonical_output_config(
        {"codec": "pcm_s16le", "channels": 1, "sample_rate": 16000}
    )
    plan = _make_plan(
        mode="candidate",
        paths=paths,
        source=source,
        source_relative_path=analysis.relative_path,
        source_sha256=analysis.sha256,
        config_sha256=analysis.config_sha256,
        output_config_sha256=output_config,
        ffmpeg_version=ffmpeg_version,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        staging_root=_staging_root,
    )

    def validate_candidate(path: Path) -> PcmMetrics | None:
        expected_samples = round((end_seconds - start_seconds) * 16000)
        return _validate_analysis_output(path, expected_samples)

    return _materialize(
        plan=plan,
        cached=cached,
        command_arguments=[
            "-ss",
            str(start_seconds),
            "-to",
            str(end_seconds),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
        ],
        validate_output=validate_candidate,
        run_command=run_command,
        source_is_derived=True,
        allow_unrecorded_existing=_staging_root is not None,
    )


def extract_review_wav(
    record: RecordingRecord,
    paths: CorpusPaths,
    start_seconds: float,
    end_seconds: float,
    *,
    ffmpeg_version: str,
    cached: DerivedAudio | None = None,
    run_command: RunCommand = subprocess.run,
) -> DerivedAudio:
    source = _raw_source(record, paths)
    _validate_segment_bounds(start_seconds, end_seconds, record.metadata.duration_seconds)
    output_config = _canonical_output_config(
        {
            "codec": "pcm_s24le",
            "channels": record.metadata.channels,
            "sample_rate": record.metadata.sample_rate,
        }
    )
    plan = _make_plan(
        mode="review",
        paths=paths,
        source=source,
        source_relative_path=record.relative_path,
        source_sha256=record.sha256,
        config_sha256=output_config,
        output_config_sha256=output_config,
        ffmpeg_version=ffmpeg_version,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    def validate_review(path: Path) -> PcmMetrics | None:
        expected_samples = round((end_seconds - start_seconds) * record.metadata.sample_rate)
        _validate_review_output(
            path,
            record.metadata.sample_rate,
            record.metadata.channels,
            expected_samples,
        )
        return None

    return _materialize(
        plan=plan,
        cached=cached,
        command_arguments=[
            "-ss",
            str(start_seconds),
            "-to",
            str(end_seconds),
            "-vn",
            "-c:a",
            "pcm_s24le",
        ],
        validate_output=validate_review,
        run_command=run_command,
        source_is_derived=False,
    )


def extract_lossless_segment(
    record: RecordingRecord,
    paths: CorpusPaths,
    start_seconds: float,
    end_seconds: float,
    *,
    ffmpeg_version: str,
    cached: DerivedAudio | None = None,
    run_command: RunCommand = subprocess.run,
    probe_command: RunCommand = subprocess.run,
    decode_command: RunCommand = subprocess.run,
    _staging_root: Path | None = None,
) -> DerivedAudio:
    source = _raw_source(record, paths)
    _validate_segment_bounds(start_seconds, end_seconds, record.metadata.duration_seconds)
    output_config = _canonical_output_config(
        {
            "codec": "flac",
            "compression_level": 5,
            "channels": record.metadata.channels,
            "sample_rate": record.metadata.sample_rate,
        }
    )
    plan = _make_plan(
        mode="lossless",
        paths=paths,
        source=source,
        source_relative_path=record.relative_path,
        source_sha256=record.sha256,
        config_sha256=output_config,
        output_config_sha256=output_config,
        ffmpeg_version=ffmpeg_version,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        staging_root=_staging_root,
    )
    expected_samples = round((end_seconds - start_seconds) * record.metadata.sample_rate)

    def validate_flac(path: Path) -> PcmMetrics | None:
        _validate_flac_output(
            path,
            sample_rate=record.metadata.sample_rate,
            channels=record.metadata.channels,
            expected_sample_count=expected_samples,
            probe_command=probe_command,
            decode_command=decode_command,
        )
        return None

    return _materialize(
        plan=plan,
        cached=cached,
        command_arguments=[
            "-ss",
            str(start_seconds),
            "-to",
            str(end_seconds),
            "-vn",
            "-c:a",
            "flac",
            "-compression_level",
            "5",
        ],
        validate_output=validate_flac,
        run_command=run_command,
        source_is_derived=False,
        allow_unrecorded_existing=_staging_root is not None,
    )
