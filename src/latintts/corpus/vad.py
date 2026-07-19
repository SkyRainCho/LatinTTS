from __future__ import annotations

import importlib
import math
import wave
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from latintts.corpus.domain import CorpusFailure

_SAMPLE_RATE = 16000
_MODEL_VERSION = "6.2.1"


@dataclass(frozen=True, slots=True)
class VadFrame:
    start_sample: int
    probability: float

    def __post_init__(self) -> None:
        if type(self.start_sample) is not int or self.start_sample < 0:
            raise ValueError("VAD frame start_sample must be a non-negative integer")
        if (
            type(self.probability) not in (int, float)
            or not math.isfinite(self.probability)
            or not 0 <= self.probability <= 1
        ):
            raise ValueError("VAD frame probability must be a finite ratio from zero through one")


@dataclass(frozen=True, slots=True)
class SpeechInterval:
    start_sample: int
    end_sample: int

    def __post_init__(self) -> None:
        if (
            type(self.start_sample) is not int
            or type(self.end_sample) is not int
            or self.start_sample < 0
            or self.end_sample <= self.start_sample
        ):
            raise ValueError("speech interval must be a positive half-open range")


@dataclass(frozen=True, slots=True)
class VadResult:
    backend: str
    model_version: str
    sample_rate: int
    frames: tuple[VadFrame, ...]
    speech_intervals: tuple[SpeechInterval, ...]


class VadBackend(Protocol):
    def analyze(self, audio_path: Path) -> VadResult: ...


class SileroVadBackend:
    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        window_samples: int = 512,
    ) -> None:
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("threshold must be a finite ratio from zero through one")
        for value, name in (
            (min_speech_ms, "min_speech_ms"),
            (min_silence_ms, "min_silence_ms"),
            (window_samples, "window_samples"),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.threshold = float(threshold)
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.window_samples = window_samples

    def _load_dependencies(self) -> tuple[Any, Any, Any]:
        try:
            module = importlib.import_module("silero_vad")
            installed_version = metadata.version("silero-vad")
            if installed_version != _MODEL_VERSION:
                raise CorpusFailure(
                    "ALIGNER_UNAVAILABLE",
                    f"silero-vad {_MODEL_VERSION} is required; found {installed_version}",
                )
            return (
                module.load_silero_vad,
                module.read_audio,
                module.get_speech_timestamps,
            )
        except (AttributeError, ImportError, metadata.PackageNotFoundError, OSError) as error:
            raise CorpusFailure(
                "ALIGNER_UNAVAILABLE",
                "silero-vad 6.2.1 dependencies are unavailable",
            ) from error

    @staticmethod
    def _validate_audio(path: Path) -> int:
        try:
            with wave.open(str(path), "rb") as handle:
                valid = (
                    handle.getframerate() == _SAMPLE_RATE
                    and handle.getnchannels() == 1
                    and handle.getsampwidth() == 2
                    and handle.getcomptype() == "NONE"
                )
                sample_count = handle.getnframes()
        except (EOFError, OSError, wave.Error) as error:
            raise ValueError("VAD input must be a valid 16 kHz mono PCM16 WAV") from error
        if not valid:
            raise ValueError("VAD input must be a valid 16 kHz mono PCM16 WAV")
        if sample_count <= 0:
            raise ValueError("VAD input WAV must contain samples")
        return sample_count

    def analyze(self, audio_path: Path) -> VadResult:
        sample_count = self._validate_audio(audio_path)
        load_model, read_audio, get_speech_timestamps = self._load_dependencies()
        model = load_model()
        model.reset_states()
        try:
            audio = read_audio(str(audio_path), sampling_rate=_SAMPLE_RATE)
            if len(audio) != sample_count:
                raise ValueError("Silero audio sample count does not match the analysis WAV")
            frames = tuple(
                VadFrame(
                    start,
                    float(model(audio[start : start + self.window_samples], _SAMPLE_RATE).item()),
                )
                for start in range(0, sample_count, self.window_samples)
            )
            raw_intervals = get_speech_timestamps(
                audio,
                model,
                sampling_rate=_SAMPLE_RATE,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
                window_size_samples=self.window_samples,
                return_seconds=False,
            )
            intervals = self._decode_intervals(raw_intervals, sample_count)
            return VadResult(
                backend="silero-vad",
                model_version=_MODEL_VERSION,
                sample_rate=_SAMPLE_RATE,
                frames=frames,
                speech_intervals=intervals,
            )
        finally:
            model.reset_states()

    @staticmethod
    def _decode_intervals(raw_intervals: object, sample_count: int) -> tuple[SpeechInterval, ...]:
        if type(raw_intervals) is not list:
            raise ValueError("Silero speech timestamps must be a list")
        intervals: list[SpeechInterval] = []
        for raw in raw_intervals:
            if type(raw) is not dict or set(raw) != {"start", "end"}:
                raise ValueError("Silero speech timestamp must contain exact sample bounds")
            interval = SpeechInterval(raw["start"], raw["end"])
            if interval.end_sample > sample_count:
                raise ValueError("Silero speech timestamp exceeds audio sample bounds")
            if intervals and intervals[-1].end_sample > interval.start_sample:
                raise ValueError("Silero speech timestamps must be ordered and non-overlapping")
            intervals.append(interval)
        return tuple(intervals)
