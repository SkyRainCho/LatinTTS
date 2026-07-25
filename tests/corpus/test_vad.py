from __future__ import annotations

import hashlib
import importlib.metadata
import math
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from latintts.corpus.domain import CorpusFailure
from latintts.corpus.vad import SileroVadBackend, SpeechInterval, VadFrame, VadResult


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Model:
    def __init__(self, probabilities: tuple[float, ...]) -> None:
        self.probabilities = iter(probabilities)
        self.calls: list[tuple[list[float], int]] = []
        self.resets = 0
        self.devices: list[str] = []

    def to(self, device: str) -> _Model:
        self.devices.append(device)
        return self

    def reset_states(self) -> None:
        self.resets += 1

    def __call__(self, window: list[float], sample_rate: int) -> _Scalar:
        if len(window) != 512:
            raise ValueError("fake Silero model requires exactly 512 samples")
        self.calls.append((window, sample_rate))
        return _Scalar(next(self.probabilities))


def _write_wav(path: Path, *, rate: int = 16000, channels: int = 1, width: int = 2) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * width * channels * 1100)


def _write_model_resource(tmp_path: Path) -> Path:
    path = tmp_path / "silero_vad.jit"
    path.write_bytes(b"fake-silero-weight")
    return path


def test_silero_analyze_records_every_window_and_sample_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "analysis.wav"
    _write_wav(audio_path)
    audio = [0.0] * 1100
    model = _Model((0.1, 0.8, 0.4))
    model_path = tmp_path / "silero_vad.jit"
    model_path.write_bytes(b"fake-silero-weight")
    model_loads: list[dict[str, object]] = []
    reads: list[tuple[Path, int]] = []
    timestamp_calls: list[dict[str, object]] = []

    def read_audio(path: str, *, sampling_rate: int) -> list[float]:
        reads.append((Path(path), sampling_rate))
        return audio

    def get_speech_timestamps(
        samples: list[float], supplied_model: _Model, **kwargs: object
    ) -> list[dict[str, int]]:
        assert samples is audio
        assert supplied_model is model
        timestamp_calls.append(kwargs)
        return [{"start": 100, "end": 900}]

    backend = SileroVadBackend(
        threshold=0.5,
        neg_threshold=0.35,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        max_speech_duration_s=60.0,
        speech_pad_ms=30,
        min_silence_at_max_speech=98,
        use_max_poss_sil_at_max_speech=True,
        sample_rate=16000,
        window_samples=512,
    )

    def load_model(**kwargs: object) -> _Model:
        model_loads.append(kwargs)
        return model

    monkeypatch.setattr(
        backend,
        "_load_dependencies",
        lambda: (load_model, read_audio, get_speech_timestamps, model_path),
    )

    result = backend.analyze(audio_path)

    assert reads == [(audio_path, 16000)]
    assert [len(window) for window, _ in model.calls] == [512, 512, 512]
    assert all(rate == 16000 for _, rate in model.calls)
    assert model_loads == [{"onnx": False}]
    assert model.devices == ["cpu"]
    assert [(frame.start_sample, frame.end_sample) for frame in result.frames] == [
        (0, 512),
        (512, 1024),
        (1024, 1100),
    ]
    assert model.resets == 2
    assert result == VadResult(
        backend="silero-vad",
        model_version="6.2.1",
        model_sha256=hashlib.sha256(b"fake-silero-weight").hexdigest(),
        sample_rate=16000,
        frames=(
            VadFrame(0, 512, 0.1),
            VadFrame(512, 1024, 0.8),
            VadFrame(1024, 1100, 0.4),
        ),
        speech_intervals=(SpeechInterval(100, 900),),
    )
    assert timestamp_calls == [
        {
            "sampling_rate": 16000,
            "threshold": 0.5,
            "neg_threshold": 0.35,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 100,
            "max_speech_duration_s": 60.0,
            "speech_pad_ms": 30,
            "min_silence_at_max_speech": 98,
            "use_max_poss_sil_at_max_speech": True,
            "return_seconds": False,
        }
    ]
    assert backend.model_sha256 == hashlib.sha256(b"fake-silero-weight").hexdigest()


def test_silero_analyze_supports_no_speech(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_path = tmp_path / "silence.wav"
    _write_wav(audio_path)
    model = _Model((0.0, 0.0, 0.0))
    backend = SileroVadBackend()
    monkeypatch.setattr(
        backend,
        "_load_dependencies",
        lambda: (
            lambda **_kwargs: model,
            lambda *_args, **_kwargs: [0.0] * 1100,
            lambda *_a, **_k: [],
            _write_model_resource(tmp_path),
        ),
    )

    assert backend.analyze(audio_path).speech_intervals == ()
    assert model.resets == 2


def test_silero_analyze_resets_after_backend_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "analysis.wav"
    _write_wav(audio_path)
    model = _Model((math.nan,))
    backend = SileroVadBackend()
    monkeypatch.setattr(
        backend,
        "_load_dependencies",
        lambda: (
            lambda **_kwargs: model,
            lambda *_a, **_k: [0.0] * 1100,
            lambda *_a, **_k: [],
            _write_model_resource(tmp_path),
        ),
    )

    with pytest.raises(ValueError, match="probability"):
        backend.analyze(audio_path)

    assert model.resets == 2


@pytest.mark.parametrize(
    ("rate", "channels", "width"),
    ((8000, 1, 2), (16000, 2, 2), (16000, 1, 1)),
)
def test_silero_analyze_rejects_non_analysis_wav_before_loading_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rate: int,
    channels: int,
    width: int,
) -> None:
    audio_path = tmp_path / "wrong.wav"
    _write_wav(audio_path, rate=rate, channels=channels, width=width)
    backend = SileroVadBackend()
    loaded = False

    def load() -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError

    monkeypatch.setattr(backend, "_load_dependencies", load)

    with pytest.raises(ValueError, match="16 kHz mono PCM16"):
        backend.analyze(audio_path)

    assert not loaded


def test_silero_analyze_rejects_timestamps_outside_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "analysis.wav"
    _write_wav(audio_path)
    model = _Model((0.2, 0.2, 0.2))
    backend = SileroVadBackend()
    monkeypatch.setattr(
        backend,
        "_load_dependencies",
        lambda: (
            lambda **_kwargs: model,
            lambda *_a, **_k: [0.0] * 1100,
            lambda *_a, **_k: [{"start": 100, "end": 1101}],
            _write_model_resource(tmp_path),
        ),
    )

    with pytest.raises(ValueError, match="audio sample bounds"):
        backend.analyze(audio_path)


@pytest.mark.parametrize(
    ("raw_intervals", "message"),
    (
        ({"start": 0, "end": 1}, "must be a list"),
        ([{"start": 0, "end": 1, "seconds": 0.1}], "exact sample bounds"),
        (
            [{"start": 0, "end": 700}, {"start": 600, "end": 900}],
            "ordered and non-overlapping",
        ),
    ),
)
def test_silero_analyze_rejects_malformed_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_intervals: object,
    message: str,
) -> None:
    audio_path = tmp_path / "analysis.wav"
    _write_wav(audio_path)
    model = _Model((0.2, 0.2, 0.2))
    backend = SileroVadBackend()
    monkeypatch.setattr(
        backend,
        "_load_dependencies",
        lambda: (
            lambda **_kwargs: model,
            lambda *_a, **_k: [0.0] * 1100,
            lambda *_a, **_k: raw_intervals,
            _write_model_resource(tmp_path),
        ),
    )

    with pytest.raises(ValueError, match=message):
        backend.analyze(audio_path)


def test_silero_analyze_rejects_decoder_sample_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "analysis.wav"
    _write_wav(audio_path)
    model = _Model((0.2,))
    backend = SileroVadBackend()
    monkeypatch.setattr(
        backend,
        "_load_dependencies",
        lambda: (
            lambda **_kwargs: model,
            lambda *_a, **_k: [0.0],
            lambda *_a, **_k: [],
            _write_model_resource(tmp_path),
        ),
    )

    with pytest.raises(ValueError, match="sample count"):
        backend.analyze(audio_path)


@pytest.mark.parametrize("content", (b"not-a-wav", b""))
def test_silero_analyze_rejects_invalid_or_empty_wav(tmp_path: Path, content: bytes) -> None:
    audio_path = tmp_path / "bad.wav"
    if content:
        audio_path.write_bytes(content)
    else:
        _write_wav(audio_path)
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)

    with pytest.raises(ValueError, match=r"WAV|samples"):
        SileroVadBackend().analyze(audio_path)


def test_silero_missing_dependency_has_stable_failure_code(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = SileroVadBackend()

    def missing(_name: str) -> object:
        raise ModuleNotFoundError("silero_vad")

    monkeypatch.setattr("latintts.corpus.vad.importlib.import_module", missing)

    with pytest.raises(CorpusFailure) as error:
        backend._load_dependencies()

    assert error.value.code == "ALIGNER_UNAVAILABLE"


def test_silero_rejects_installed_version_different_from_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SileroVadBackend()
    fake_module = SimpleNamespace(
        load_silero_vad=lambda: object(),
        read_audio=lambda *_a, **_k: [],
        get_speech_timestamps=lambda *_a, **_k: [],
    )
    monkeypatch.setattr("latintts.corpus.vad.importlib.import_module", lambda _name: fake_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "6.3.0")

    with pytest.raises(CorpusFailure, match=r"6\.2\.1") as error:
        backend._load_dependencies()

    assert error.value.code == "ALIGNER_UNAVAILABLE"


def test_silero_loads_exactly_pinned_installed_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    functions = (lambda **_kwargs: object(), lambda *_a, **_k: [], lambda *_a, **_k: [])
    model_root = tmp_path / "package"
    model_path = model_root / "data" / "silero_vad.jit"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"fake-silero-weight")
    fake_module = SimpleNamespace(
        load_silero_vad=functions[0],
        read_audio=functions[1],
        get_speech_timestamps=functions[2],
    )
    monkeypatch.setattr("latintts.corpus.vad.importlib.import_module", lambda _name: fake_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "6.2.1")
    monkeypatch.setattr("latintts.corpus.vad.resources.files", lambda _name: model_root)

    assert SileroVadBackend()._load_dependencies() == (*functions, model_path)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"threshold": math.nan},
        {"neg_threshold": 0.6},
        {"min_speech_duration_ms": 0},
        {"min_silence_duration_ms": 1.5},
        {"max_speech_duration_s": "long"},
        {"max_speech_duration_s": math.inf},
        {"speech_pad_ms": -1},
        {"min_silence_at_max_speech": 0},
        {"use_max_poss_sil_at_max_speech": 1},
        {"sample_rate": 8000},
        {"window_samples": True},
        {"window_samples": 0},
    ),
)
def test_silero_backend_validates_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        SileroVadBackend(**kwargs)  # type: ignore[arg-type]


def test_silero_621_requires_512_sample_windows() -> None:
    with pytest.raises(ValueError, match="512"):
        SileroVadBackend(window_samples=256)


def test_silero_model_resource_read_failure_has_stable_code() -> None:
    class BrokenResource:
        def open(self, _mode: str) -> object:
            raise OSError("unreadable")

    with pytest.raises(CorpusFailure) as error:
        SileroVadBackend._hash_model_resource(BrokenResource())

    assert error.value.code == "ALIGNER_UNAVAILABLE"


def test_silero_rejects_invalid_computed_model_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "analysis.wav"
    _write_wav(audio_path)
    backend = SileroVadBackend()
    monkeypatch.setattr(
        backend,
        "_load_dependencies",
        lambda: (lambda **_kwargs: object(), object(), object(), object()),
    )
    monkeypatch.setattr(backend, "_hash_model_resource", lambda _resource: "invalid")

    with pytest.raises(ValueError, match="model hash"):
        backend.analyze(audio_path)


def test_silero_zero_padding_supports_tensor_like_audio() -> None:
    class TensorLike(list[float]):
        def new_zeros(self, size: int) -> TensorLike:
            return TensorLike([0.0] * size)

    audio = TensorLike([1.0] * 600)

    padded = SileroVadBackend()._padded_window(audio, 512, 600)

    assert len(padded) == 512
    assert padded[:88] == [1.0] * 88
    assert padded[88:] == [0.0] * 424


@pytest.mark.parametrize(
    "constructor",
    (
        lambda: VadFrame(-1, 1, 0.5),
        lambda: VadFrame(0, 1, 1.1),
        lambda: SpeechInterval(-1, 1),
        lambda: SpeechInterval(1, 1),
    ),
)
def test_vad_value_objects_validate_sample_boundaries(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]
