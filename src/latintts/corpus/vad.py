from __future__ import annotations

import hashlib
import importlib
import math
import re
import wave
from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Protocol

from latintts.corpus.domain import CorpusFailure

_SAMPLE_RATE = 16000
_MODEL_VERSION = "6.2.1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class VadFrame:
    start_sample: int
    end_sample: int
    probability: float

    def __post_init__(self) -> None:
        if (
            type(self.start_sample) is not int
            or type(self.end_sample) is not int
            or self.start_sample < 0
            or self.end_sample <= self.start_sample
        ):
            raise ValueError("VAD frame must be a positive half-open sample range")
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
    model_sha256: str
    sample_rate: int
    frames: tuple[VadFrame, ...]
    speech_intervals: tuple[SpeechInterval, ...]


class VadBackend(Protocol):
    @property
    def model_sha256(self) -> str: ...

    def analyze(self, audio_path: Path) -> VadResult: ...


class SileroVadBackend:
    def __init__(
        self,
        *,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        max_speech_duration_s: float = 60.0,
        speech_pad_ms: int = 30,
        min_silence_at_max_speech: int = 98,
        use_max_poss_sil_at_max_speech: bool = True,
        sample_rate: int = 16000,
        window_samples: int = 512,
    ) -> None:
        for value, name in ((threshold, "threshold"), (neg_threshold, "neg_threshold")):
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite ratio from zero through one")
        if neg_threshold > threshold:
            raise ValueError("neg_threshold must not exceed threshold")
        for value, name, minimum in (
            (min_speech_duration_ms, "min_speech_duration_ms", 1),
            (min_silence_duration_ms, "min_silence_duration_ms", 1),
            (speech_pad_ms, "speech_pad_ms", 0),
            (min_silence_at_max_speech, "min_silence_at_max_speech", 1),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} is out of range")
        if type(max_speech_duration_s) not in (int, float):
            raise TypeError("max_speech_duration_s must be a number")
        if not math.isfinite(max_speech_duration_s) or max_speech_duration_s <= 0:
            raise ValueError("max_speech_duration_s must be finite and positive")
        if type(use_max_poss_sil_at_max_speech) is not bool:
            raise TypeError("use_max_poss_sil_at_max_speech must be a boolean")
        if type(sample_rate) is not int or sample_rate != _SAMPLE_RATE:
            raise ValueError("silero-vad 6.2.1 requires sample_rate=16000")
        self.threshold = float(threshold)
        self.neg_threshold = float(neg_threshold)
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.max_speech_duration_s = float(max_speech_duration_s)
        self.speech_pad_ms = speech_pad_ms
        self.min_silence_at_max_speech = min_silence_at_max_speech
        self.use_max_poss_sil_at_max_speech = use_max_poss_sil_at_max_speech
        self.sample_rate = sample_rate
        if type(window_samples) is not int:
            raise TypeError("window_samples must be an integer")
        if window_samples != 512:
            raise ValueError("silero-vad 6.2.1 requires window_samples=512")
        self.window_samples = window_samples

    def _load_dependencies(self) -> tuple[Any, Any, Any, Any]:
        try:
            module = importlib.import_module("silero_vad")
            installed_version = metadata.version("silero-vad")
            if installed_version != _MODEL_VERSION:
                raise CorpusFailure(
                    "ALIGNER_UNAVAILABLE",
                    f"silero-vad {_MODEL_VERSION} is required; found {installed_version}",
                )
            model_resource = (
                resources.files("silero_vad").joinpath("data").joinpath("silero_vad.jit")
            )
            if not model_resource.is_file():
                raise OSError("packaged silero_vad.jit is unavailable")
            return (
                module.load_silero_vad,
                module.read_audio,
                module.get_speech_timestamps,
                model_resource,
            )
        except (AttributeError, ImportError, metadata.PackageNotFoundError, OSError) as error:
            raise CorpusFailure(
                "ALIGNER_UNAVAILABLE",
                "silero-vad 6.2.1 dependencies are unavailable",
            ) from error

    @staticmethod
    def _hash_model_resource(model_resource: Any) -> str:
        digest = hashlib.sha256()
        try:
            with model_resource.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as error:
            raise CorpusFailure(
                "ALIGNER_UNAVAILABLE", "packaged silero_vad.jit cannot be hashed"
            ) from error
        return digest.hexdigest()

    @property
    def model_sha256(self) -> str:
        *_, model_resource = self._load_dependencies()
        return self._hash_model_resource(model_resource)

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
        load_model, read_audio, get_speech_timestamps, model_resource = self._load_dependencies()
        model_sha256 = self._hash_model_resource(model_resource)
        if _SHA256.fullmatch(model_sha256) is None:
            raise ValueError("Silero model hash must be a lowercase SHA-256 digest")
        model = load_model(onnx=False).to("cpu")
        model.reset_states()
        try:
            audio = read_audio(str(audio_path), sampling_rate=self.sample_rate)
            if len(audio) != sample_count:
                raise ValueError("Silero audio sample count does not match the analysis WAV")
            frames_list: list[VadFrame] = []
            for start in range(0, sample_count, self.window_samples):
                end = min(start + self.window_samples, sample_count)
                window = self._padded_window(audio, start, end)
                probability = float(model(window, self.sample_rate).item())
                frames_list.append(VadFrame(start, end, probability))
            frames = tuple(frames_list)
            raw_intervals = get_speech_timestamps(
                audio,
                model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                neg_threshold=self.neg_threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                max_speech_duration_s=self.max_speech_duration_s,
                speech_pad_ms=self.speech_pad_ms,
                min_silence_at_max_speech=self.min_silence_at_max_speech,
                use_max_poss_sil_at_max_speech=self.use_max_poss_sil_at_max_speech,
                return_seconds=False,
            )
            intervals = self._decode_intervals(raw_intervals, sample_count)
            return VadResult(
                backend="silero-vad",
                model_version=_MODEL_VERSION,
                model_sha256=model_sha256,
                sample_rate=self.sample_rate,
                frames=frames,
                speech_intervals=intervals,
            )
        finally:
            model.reset_states()

    def _padded_window(self, audio: Any, start: int, end: int) -> Any:
        window = audio[start:end]
        missing = self.window_samples - len(window)
        if missing == 0:
            return window
        if hasattr(audio, "new_zeros"):
            padded = audio.new_zeros(self.window_samples)
            padded[: len(window)] = window
            return padded
        return [*window, *([0.0] * missing)]

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
