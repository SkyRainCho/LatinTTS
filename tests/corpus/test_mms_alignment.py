from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from latintts.corpus.alignment import AlignmentRequest, validate_alignment
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.mms_alignment import (
    ALIGNER_COMMIT,
    BackendRuntimeInfo,
    MmsCtcAligner,
    _installed_aligner_commit,
    create_alignment_backend,
    normalize_mms_output,
)


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _request(tmp_path: Path, **changes: object) -> AlignmentRequest:
    values: dict[str, object] = {
        "audio_path": tmp_path / "take.wav",
        "audio_sha256": _digest("audio"),
        "spoken_text": "Grátia plena.",
        "segmentation_artifact_sha256": _digest("segmentation"),
        "backend": "mms-ctc",
        "backend_version": "0.3.0",
        "model_id": "MahmoudAshraf/mms-300m-1130-forced-aligner",
        "model_revision": "f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        "model_license": "CC-BY-NC-4.0",
        "config_sha256": _digest("config"),
        "effective_parameters": {"language": "lat"},
    }
    values.update(changes)
    return AlignmentRequest(**values)  # type: ignore[arg-type]


def _alignment_config(**changes: object) -> CorpusConfig:
    alignment: dict[str, object] = {
        "backend": "mms-ctc",
        "model_id": "MahmoudAshraf/mms-300m-1130-forced-aligner",
        "model_revision": "f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        "language": "lat",
        "romanize": True,
        "split_size": "word",
        "star_frequency": "edges",
        "window_seconds": 30,
        "context_seconds": 2,
        "batch_size": 4,
        "dtype": "float16",
        "device": "cuda",
        "license": "CC-BY-NC-4.0",
    }
    alignment.update(changes)
    return CorpusConfig(raw={"alignment": alignment}, digest=_digest("config"))


def test_installed_aligner_commit_reads_pip_direct_url_provenance(monkeypatch) -> None:
    distribution = SimpleNamespace(
        read_text=lambda name: (
            '{"vcs_info":{"commit_id":"' + ALIGNER_COMMIT + '"}}'
            if name == "direct_url.json"
            else None
        )
    )
    monkeypatch.setattr(
        "latintts.corpus.mms_alignment.importlib.metadata.distribution",
        lambda name: distribution,
    )

    assert _installed_aligner_commit() == ALIGNER_COMMIT


def test_installed_aligner_commit_rejects_unverifiable_distribution(monkeypatch) -> None:
    distribution = SimpleNamespace(read_text=lambda name: None)
    monkeypatch.setattr(
        "latintts.corpus.mms_alignment.importlib.metadata.distribution",
        lambda name: distribution,
    )

    with pytest.raises(CorpusFailure, match="unverifiable"):
        _installed_aligner_commit()


def test_mms_output_keeps_alignment_text_non_authoritative() -> None:
    result = normalize_mms_output(
        spoken_text="Grátia plena.",
        text_starred=("", "Grátia", "plena.", ""),
        tokens_starred=("", "g r a t i a", "p l e n a", ""),
        raw_results=(
            {"start": 0.1, "end": 0.7, "text": "Grátia", "score": -0.1},
            {"start": 0.7, "end": 1.4, "text": "plena.", "score": -0.2},
        ),
        backend_version="0.3.0",
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
    )

    assert result.alignment_text == "gratia plena"
    assert [word.text for word in result.words] == ["Grátia", "plena."]
    assert [word.spoken_token_index for word in result.words] == [0, 1]
    assert result.phoneme_timing_status == "not_estimated"
    assert result.model_license == "CC-BY-NC-4.0"
    assert result.integrity_sha256


def test_mms_output_rejects_missing_or_duplicate_word_spans() -> None:
    with pytest.raises(CorpusFailure, match="one span per spoken token") as error:
        normalize_mms_output(
            spoken_text="Gratia plena",
            text_starred=("", "Gratia", "plena", ""),
            tokens_starred=("", "g r a t i a", "p l e n a", ""),
            raw_results=({"start": 0.1, "end": 0.7, "text": "Gratia", "score": 0.9},),
            backend_version="0.3.0",
            model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
            model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        )

    assert error.value.code == "ALIGNMENT_LOW_CONFIDENCE"


def test_mms_output_rejects_empty_backend_result() -> None:
    with pytest.raises(CorpusFailure, match="no word spans"):
        normalize_mms_output(
            spoken_text="Gratia",
            text_starred=("", "Gratia", ""),
            tokens_starred=("", "g r a t i a", ""),
            raw_results=(),
            backend_version="0.3.0",
            model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
            model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        )


def test_mms_output_rejects_backend_tokens_that_do_not_map_to_spoken_text() -> None:
    with pytest.raises(CorpusFailure, match="backend tokens do not match"):
        normalize_mms_output(
            spoken_text="Gratia",
            text_starred=("", "Gratia", ""),
            tokens_starred=("", "g r a c i a", ""),
            raw_results=({"start": 0.0, "end": 0.6, "text": "Gracia", "score": 0.9},),
            backend_version="0.3.0",
            model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
            model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        )


def test_mms_output_converts_tensor_scalars_and_normalizes_scores() -> None:
    result = normalize_mms_output(
        spoken_text="Gratia plena",
        text_starred=("", "Gratia", "plena", ""),
        tokens_starred=("", "g r a t i a", "p l e n a", ""),
        raw_results=(
            {"start": Scalar(0.1), "end": Scalar(0.7), "text": "Gratia", "score": Scalar(-1)},
            {"start": Scalar(0.7), "end": Scalar(1.4), "text": "plena", "score": Scalar(4)},
        ),
        backend_version="0.3.0",
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
    )

    assert result.words[0].score == pytest.approx(0.36787944117)
    assert result.words[1].score == 1.0
    assert result.raw_output["segments"][0]["raw_score"] == -1.0  # type: ignore[index]


def test_mms_aligner_calls_adapter_in_order_and_preserves_token_indexes(tmp_path: Path) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeAligner:
        @staticmethod
        def load_audio(*args: object) -> str:
            calls.append(("load_audio", *args))
            return "waveform"

        @staticmethod
        def generate_emissions(*args: object) -> tuple[str, float]:
            calls.append(("generate_emissions", *args))
            return "emissions", 0.02

        @staticmethod
        def preprocess_text(*args: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
            calls.append(("preprocess_text", *args))
            return ("", "g r a t i a", "p l e n a", ""), ("", "Grátia", "plena.", "")

        @staticmethod
        def get_alignments(*args: object) -> tuple[str, str, int]:
            calls.append(("get_alignments", *args))
            return "segments", "scores", 0

        @staticmethod
        def get_spans(*args: object) -> str:
            calls.append(("get_spans", *args))
            return "spans"

        @staticmethod
        def postprocess_results(*args: object) -> tuple[dict[str, object], ...]:
            calls.append(("postprocess_results", *args))
            return (
                {"start": 0.1, "end": 0.7, "text": "Grátia", "score": -0.1},
                {"start": 0.7, "end": 1.4, "text": "plena.", "score": -0.2},
            )

    model = SimpleNamespace(dtype="float16", device="cuda")
    dependencies = SimpleNamespace(aligner=FakeAligner(), model=model, tokenizer="tokenizer")
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=dependencies,
    )

    request = _request(tmp_path)
    result = backend.align(request)

    assert [call[0] for call in calls] == [
        "load_audio",
        "generate_emissions",
        "preprocess_text",
        "get_alignments",
        "get_spans",
        "postprocess_results",
    ]
    assert calls[1][-3:] == (30, 2, 4)
    assert calls[2][-5:] == (request.spoken_text, True, "lat", "word", "edges")
    assert [word.spoken_token_index for word in result.words] == [0, 1]
    validate_alignment(result, request, audio_duration_seconds=1.4)


def test_mms_lazy_load_uses_pinned_revision_trust_and_cache_policy(tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []
    revision = "f37ba6bf1673872e07519fb951866cb2a32a6d7f"
    snapshot = tmp_path / "cache" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"pinned weights")

    class FakeModel:
        dtype = "float16"
        device = "cuda"

        def to(self, device: str) -> FakeModel:
            calls.append(("model.to", device))
            self.device = device
            return self

        def eval(self) -> FakeModel:
            calls.append(("model.eval", True))
            return self

    class ModelFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            calls.append(("model.from_pretrained", (path, kwargs)))
            return FakeModel()

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> str:
            calls.append(("tokenizer.from_pretrained", (path, kwargs)))
            return "tokenizer"

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name() -> str:
            return "Fake CUDA"

        @staticmethod
        def max_memory_allocated() -> int:
            return 4096

    torch = SimpleNamespace(
        __version__="2.5.1", float16="float16", float32="float32", cuda=FakeCuda()
    )
    hub = SimpleNamespace(
        snapshot_download=lambda **kwargs: (
            calls.append(("snapshot_download", kwargs)) or str(snapshot)
        )
    )
    transformers = SimpleNamespace(
        __version__="4.48.0",
        AutoModelForCTC=ModelFactory,
        AutoTokenizer=TokenizerFactory,
    )
    modules = {
        "torch": torch,
        "huggingface_hub": hub,
        "ctc_forced_aligner": SimpleNamespace(__version__="0.3.0"),
        "transformers": transformers,
    }
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision=revision,
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        module_loader=modules.__getitem__,
        tool_commit_resolver=lambda: ALIGNER_COMMIT,
    )

    assert calls == []
    info = backend.runtime_info()

    assert isinstance(info, BackendRuntimeInfo)
    assert info.device_name == "Fake CUDA"
    assert len(info.model_weights_sha256) == 64
    snapshot_call = next(value for name, value in calls if name == "snapshot_download")
    assert snapshot_call == {
        "repo_id": "MahmoudAshraf/mms-300m-1130-forced-aligner",
        "revision": revision,
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
    }
    model_call = next(value for name, value in calls if name == "model.from_pretrained")
    assert model_call[1]["revision"] == revision  # type: ignore[index]
    assert model_call[1]["trust_remote_code"] is False  # type: ignore[index]
    assert model_call[1]["local_files_only"] is True  # type: ignore[index]


def test_create_alignment_backend_consumes_only_pinned_mms_config(tmp_path: Path) -> None:
    dependency = SimpleNamespace()

    backend = create_alignment_backend(
        _alignment_config(),
        dependencies=dependency,
        cache_dir=tmp_path / "cache",
        local_files_only=True,
    )

    assert isinstance(backend, MmsCtcAligner)
    assert backend.dependencies is dependency
    assert backend.window_seconds == 30
    assert backend.context_seconds == 2
    assert backend.batch_size == 4
    assert backend.local_files_only is True
    with pytest.raises(ValueError, match="model_revision"):
        create_alignment_backend(_alignment_config(model_revision="main"))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"extra": True}, "exact fields"),
        ({"window_seconds": 0}, "positive integer"),
        ({"dtype": "bfloat16"}, "dtype"),
        ({"device": "mps"}, "device"),
    ),
)
def test_create_alignment_backend_rejects_unsupported_runtime_config(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_alignment_backend(_alignment_config(**changes))


def test_mms_aligner_translates_cuda_oom_to_stable_corpus_failure(tmp_path: Path) -> None:
    class OutOfMemoryAligner:
        @staticmethod
        def load_audio(*args: object) -> None:
            raise RuntimeError("CUDA out of memory")

    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(
            aligner=OutOfMemoryAligner(),
            model=SimpleNamespace(dtype="float16", device="cuda"),
            tokenizer="tokenizer",
        ),
    )

    with pytest.raises(CorpusFailure, match="out of memory") as error:
        backend.align(_request(tmp_path))

    assert error.value.code == "ALIGNER_UNAVAILABLE"


def test_mms_aligner_translates_other_runtime_errors_to_stable_failure(tmp_path: Path) -> None:
    class BrokenAligner:
        @staticmethod
        def load_audio(*args: object) -> None:
            raise RuntimeError("kernel mismatch")

    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cpu",
        dtype="float32",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(
            aligner=BrokenAligner(),
            model=SimpleNamespace(dtype="float32", device="cpu"),
            tokenizer="tokenizer",
        ),
    )

    with pytest.raises(CorpusFailure, match="kernel mismatch"):
        backend.align(_request(tmp_path))


def test_runtime_info_requires_verified_loaded_dependencies() -> None:
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cpu",
        dtype="float32",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(),
    )

    with pytest.raises(CorpusFailure, match="runtime info is unavailable"):
        backend.runtime_info()


def test_mms_loader_uses_controlled_cpu_fallback_when_cuda_is_unavailable(tmp_path: Path) -> None:
    revision = "f37ba6bf1673872e07519fb951866cb2a32a6d7f"
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    loaded: dict[str, object] = {}

    class FakeModel:
        dtype = "float32"
        device = "cpu"

        def to(self, device: str) -> FakeModel:
            loaded["device"] = device
            return self

        def eval(self) -> FakeModel:
            return self

    class ModelFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            loaded["dtype"] = kwargs["torch_dtype"]
            return FakeModel()

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> str:
            return "tokenizer"

    class UnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    modules = {
        "torch": SimpleNamespace(
            __version__="2.5.1",
            float16="float16",
            float32="float32",
            cuda=UnavailableCuda(),
        ),
        "huggingface_hub": SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot)),
        "ctc_forced_aligner": SimpleNamespace(__version__="0.3.0"),
        "transformers": SimpleNamespace(
            __version__="4.48.0",
            AutoModelForCTC=ModelFactory,
            AutoTokenizer=TokenizerFactory,
        ),
    }
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision=revision,
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        module_loader=modules.__getitem__,
        tool_commit_resolver=lambda: ALIGNER_COMMIT,
        allow_cpu_fallback=True,
    )

    info = backend.runtime_info()

    assert loaded == {"dtype": "float32", "device": "cpu"}
    assert info.device_name == "CPU"
    assert info.peak_memory_bytes is None


def test_mms_aligner_rejects_request_provenance_before_backend_work(tmp_path: Path) -> None:
    class MustNotRun:
        @staticmethod
        def load_audio(*args: object) -> None:
            raise AssertionError("backend work must not start")

    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cpu",
        dtype="float32",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(
            aligner=MustNotRun(),
            model=SimpleNamespace(dtype="float32", device="cpu"),
            tokenizer="tokenizer",
        ),
    )

    with pytest.raises(ValueError, match="model_id"):
        backend.align(_request(tmp_path, model_id="other/model"))
