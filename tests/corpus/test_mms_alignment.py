from __future__ import annotations

import hashlib
import json
from dataclasses import replace
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
    ModelWeightFile,
    _installed_aligner_commit,
    create_alignment_backend,
    normalize_mms_output,
    resolve_model_weights,
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


def _resolved_runtime_info() -> BackendRuntimeInfo:
    return BackendRuntimeInfo(
        torch_version="2.5.1",
        transformers_version="4.48.0",
        device_name="Fake CUDA",
        peak_memory_bytes=4096,
        aligner_version="0.3.0",
        aligner_commit=ALIGNER_COMMIT,
        resolved_model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        model_weights_sha256="f" * 64,
        cuda_available=True,
        requested_device="cuda",
        requested_dtype="float16",
        resolved_device="cuda",
        resolved_dtype="float16",
        uroman_version="1.3.1",
        model_weight_manifest=(ModelWeightFile("model.safetensors", "e" * 64, 123),),
        measurement_scope="alignment-call",
    )


def _runtime_request(
    backend: MmsCtcAligner, tmp_path: Path, spoken_text: str = "Grátia plena."
) -> AlignmentRequest:
    return backend.build_request(
        audio_path=tmp_path / "take.wav",
        audio_sha256=_digest("audio"),
        spoken_text=spoken_text,
        segmentation_artifact_sha256=_digest("segment"),
        config_sha256=_digest("config"),
    )


def test_build_request_binds_resolved_runtime_identity_into_cache_key(tmp_path: Path) -> None:
    runtime = _resolved_runtime_info()
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(runtime_info=runtime),
    )

    request = backend.build_request(
        audio_path=tmp_path / "take.wav",
        audio_sha256=_digest("audio"),
        spoken_text="Ave Virgo",
        segmentation_artifact_sha256=_digest("segment"),
        config_sha256=_digest("config"),
    )

    effective = request.effective_parameters
    assert effective["requested_device"] == "cuda"
    assert effective["resolved_device"] == "cuda"
    assert effective["resolved_dtype"] == "float16"
    assert effective["resolved_model_revision"] == runtime.resolved_model_revision
    assert effective["model_weights_sha256"] == runtime.model_weights_sha256
    assert effective["model_weight_manifest"][0]["filename"] == "model.safetensors"  # type: ignore[index]
    assert effective["uroman_version"] == "1.3.1"
    assert request.cache_key != _request(tmp_path).cache_key


def test_align_rejects_request_built_for_different_resolved_runtime(tmp_path: Path) -> None:
    class MustNotRun:
        @staticmethod
        def load_audio(*args: object) -> None:
            raise AssertionError("runtime mismatch must fail before backend work")

    cuda_runtime = _resolved_runtime_info()
    cpu_runtime = replace(
        cuda_runtime,
        device_name="CPU",
        peak_memory_bytes=None,
        cuda_available=False,
        resolved_device="cpu",
        resolved_dtype="float32",
    )
    cuda_backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(
            runtime_info=cuda_runtime,
            aligner=MustNotRun(),
            model=SimpleNamespace(dtype="float16", device="cuda"),
            tokenizer="tokenizer",
        ),
    )
    cpu_backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(runtime_info=cpu_runtime),
    )
    request = cpu_backend.build_request(
        audio_path=tmp_path / "take.wav",
        audio_sha256=_digest("audio"),
        spoken_text="Gratia",
        segmentation_artifact_sha256=_digest("segment"),
        config_sha256=_digest("config"),
    )

    with pytest.raises(ValueError, match="runtime identity"):
        cuda_backend.align(request)


def test_align_stores_runtime_identity_inside_result_integrity(tmp_path: Path) -> None:
    class OneWordAligner:
        @staticmethod
        def load_audio(*args: object) -> str:
            return "waveform"

        @staticmethod
        def generate_emissions(*args: object) -> tuple[str, float]:
            return "emissions", 0.02

        @staticmethod
        def preprocess_text(*args: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
            return ("<star>", "g r a t i a", "<star>"), (
                "<star>",
                "gratia",
                "<star>",
            )

        @staticmethod
        def get_alignments(*args: object) -> tuple[str, str, int]:
            return "segments", "scores", 0

        @staticmethod
        def get_spans(*args: object) -> str:
            return "spans"

        @staticmethod
        def postprocess_results(*args: object) -> tuple[dict[str, object], ...]:
            return ({"start": 0.0, "end": 0.8, "text": "gratia", "score": 0.9},)

    runtime = _resolved_runtime_info()
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(
            runtime_info=runtime,
            aligner=OneWordAligner(),
            model=SimpleNamespace(dtype="float16", device="cuda"),
            tokenizer="tokenizer",
        ),
    )
    request = backend.build_request(
        audio_path=tmp_path / "take.wav",
        audio_sha256=_digest("audio"),
        spoken_text="Gratia",
        segmentation_artifact_sha256=_digest("segment"),
        config_sha256=_digest("config"),
    )

    result = backend.align(request)

    identity = result.raw_output["runtime_identity"]
    assert identity["model_weights_sha256"] == runtime.model_weights_sha256  # type: ignore[index]
    assert identity["resolved_device"] == "cuda"  # type: ignore[index]
    assert result.integrity_sha256
    validate_alignment(result, request, audio_duration_seconds=0.8)


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


def test_model_weight_manifest_prefers_actual_safetensors_loader_choice(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"safe")
    (tmp_path / "pytorch_model.bin").write_bytes(b"unused bin fallback")

    digest, manifest = resolve_model_weights(tmp_path)

    assert len(digest) == 64
    assert tuple(item.filename for item in manifest) == ("model.safetensors",)
    assert manifest[0].sha256 == hashlib.sha256(b"safe").hexdigest()


def test_model_weight_manifest_includes_index_and_all_unique_referenced_shards(
    tmp_path: Path,
) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 6},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                    "layer.2": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")

    _, manifest = resolve_model_weights(tmp_path)

    assert tuple(item.filename for item in manifest) == (
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )


def test_model_weight_manifest_accepts_explicit_cache_root_for_regular_snapshot(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    snapshot = cache / "models--latin" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")

    _, manifest = resolve_model_weights(snapshot, cache_root=cache)

    assert tuple(item.filename for item in manifest) == ("model.safetensors",)


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")


def test_model_weight_manifest_supports_hf_snapshot_symlinks_to_cache_blobs(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    repository = cache / "models--latin"
    snapshot = repository / "snapshots" / ("a" * 40)
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    index_blob = blobs / "index-hash"
    shard_blob = blobs / "shard-hash"
    index_blob.write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    shard_blob.write_bytes(b"shard")
    _symlink_or_skip(snapshot / "model.safetensors.index.json", Path("../../blobs/index-hash"))
    _symlink_or_skip(snapshot / "model-00001-of-00001.safetensors", Path("../../blobs/shard-hash"))

    _, manifest = resolve_model_weights(snapshot, cache_root=cache)

    assert tuple(item.filename for item in manifest) == (
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
    )
    assert manifest[1].sha256 == hashlib.sha256(b"shard").hexdigest()


def test_model_weight_manifest_rejects_snapshot_symlink_outside_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    snapshot = cache / "models--latin" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    _symlink_or_skip(snapshot / "model.safetensors", outside)

    with pytest.raises(CorpusFailure, match="outside the Hugging Face cache root"):
        resolve_model_weights(snapshot, cache_root=cache)


@pytest.mark.parametrize(
    "filename",
    ("../outside.safetensors", "C:/outside.safetensors", r"..\outside.safetensors"),
)
def test_model_weight_manifest_rejects_noncanonical_index_paths(
    tmp_path: Path, filename: str
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": filename}}), encoding="utf-8"
    )

    with pytest.raises(CorpusFailure, match="canonical relative path"):
        resolve_model_weights(tmp_path, cache_root=tmp_path)


@pytest.mark.parametrize("mode", ("missing", "extra", "conflicting-index"))
def test_model_weight_manifest_rejects_inconsistent_shard_sets(tmp_path: Path, mode: str) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    if mode != "missing":
        (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"one")
    if mode == "extra":
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"extra")
    if mode == "conflicting-index":
        (tmp_path / "other.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"other": "model-00001-of-00001.safetensors"}}),
            encoding="utf-8",
        )

    with pytest.raises(CorpusFailure, match=r"missing or extra|conflicting"):
        resolve_model_weights(tmp_path)


@pytest.mark.parametrize("mode", ("invalid-index", "unindexed-multiple", "no-weights"))
def test_model_weight_manifest_rejects_ambiguous_or_invalid_selection(
    tmp_path: Path, mode: str
) -> None:
    if mode == "invalid-index":
        (tmp_path / "model.safetensors.index.json").write_text("not-json", encoding="utf-8")
    elif mode == "unindexed-multiple":
        (tmp_path / "one.safetensors").write_bytes(b"one")
        (tmp_path / "two.safetensors").write_bytes(b"two")

    with pytest.raises(CorpusFailure, match=r"invalid|multiple|no unique"):
        resolve_model_weights(tmp_path)


def test_model_weight_manifest_rejects_mixed_format_shard_index(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "pytorch_model.bin"}}), encoding="utf-8"
    )

    with pytest.raises(CorpusFailure, match="mixes weight formats"):
        resolve_model_weights(tmp_path)


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


def test_mms_output_filters_literal_edge_star_sentinels() -> None:
    result = normalize_mms_output(
        spoken_text="Gratia plena",
        text_starred=("<star>", "Gratia", "plena", "<star>"),
        tokens_starred=("<star>", "g r a t i a", "p l e n a", "<star>"),
        raw_results=(
            {"start": 0.0, "end": 0.5, "text": "Gratia", "score": 0.9},
            {"start": 0.5, "end": 1.0, "text": "plena", "score": 0.8},
        ),
        backend_version="0.3.0",
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
    )

    assert result.alignment_text == "gratia plena"
    assert tuple(word.spoken_token_index for word in result.words) == (0, 1)


def test_mms_output_canonicalizes_uroman_tokens_with_task8_transform() -> None:
    result = normalize_mms_output(
        spoken_text="Ave Virgo Jesus Ave",
        text_starred=("<star>", "ave", "virgo", "jesus", "ave", "<star>"),
        tokens_starred=(
            "<star>",
            "a v e",
            "v i r g o",
            "j e s u s",
            "a v e",
            "<star>",
        ),
        raw_results=tuple(
            {
                "start": index * 0.5,
                "end": (index + 1) * 0.5,
                "text": text,
                "score": 0.9,
            }
            for index, text in enumerate(("ave", "virgo", "jesus", "ave"))
        ),
        backend_version="0.3.0",
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
    )

    assert result.alignment_text == "aue uirgo iesus aue"
    assert tuple(token.spoken_surface for token in result.tokens) == (
        "Ave",
        "Virgo",
        "Jesus",
        "Ave",
    )
    assert tuple(word.spoken_token_index for word in result.words) == (0, 1, 2, 3)


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
            return ("<star>", "g r a t i a", "p l e n a", "<star>"), (
                "<star>",
                "gratia",
                "plena",
                "<star>",
            )

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
    dependencies = SimpleNamespace(
        runtime_info=_resolved_runtime_info(),
        aligner=FakeAligner(),
        model=model,
        tokenizer="tokenizer",
    )
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

    request = _runtime_request(backend, tmp_path)
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
    assert calls[2][-5:] == (request.alignment_text, True, "lat", "word", "edges")
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
        def synchronize() -> None:
            calls.append(("cuda.synchronize", True))

        @staticmethod
        def reset_peak_memory_stats() -> None:
            calls.append(("cuda.reset_peak_memory_stats", True))

        @staticmethod
        def max_memory_allocated() -> int:
            calls.append(("cuda.max_memory_allocated", True))
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
        dependency_version_resolver=lambda name: {"uroman": "1.3.1"}[name],
    )

    assert calls == []
    info = backend.runtime_info()

    assert isinstance(info, BackendRuntimeInfo)
    assert info.device_name == "Fake CUDA"
    assert info.uroman_version == "1.3.1"
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
    measured = [
        name
        for name, _ in calls
        if name
        in {
            "cuda.synchronize",
            "cuda.reset_peak_memory_stats",
            "model.from_pretrained",
            "cuda.max_memory_allocated",
        }
    ]
    assert measured == [
        "cuda.synchronize",
        "cuda.reset_peak_memory_stats",
        "model.from_pretrained",
        "cuda.synchronize",
        "cuda.max_memory_allocated",
    ]


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
            runtime_info=_resolved_runtime_info(),
            aligner=OutOfMemoryAligner(),
            model=SimpleNamespace(dtype="float16", device="cuda"),
            tokenizer="tokenizer",
        ),
    )

    with pytest.raises(CorpusFailure, match="out of memory") as error:
        backend.align(_runtime_request(backend, tmp_path))

    assert error.value.code == "ALIGNER_UNAVAILABLE"


def test_mms_aligner_translates_other_runtime_errors_to_stable_failure(tmp_path: Path) -> None:
    class BrokenAligner:
        @staticmethod
        def load_audio(*args: object) -> None:
            raise RuntimeError("kernel mismatch")

    runtime = replace(
        _resolved_runtime_info(),
        requested_device="cpu",
        requested_dtype="float32",
        resolved_device="cpu",
        resolved_dtype="float32",
        device_name="CPU",
        peak_memory_bytes=None,
    )
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cpu",
        dtype="float32",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(
            runtime_info=runtime,
            aligner=BrokenAligner(),
            model=SimpleNamespace(dtype="float32", device="cpu"),
            tokenizer="tokenizer",
        ),
    )

    with pytest.raises(CorpusFailure, match="kernel mismatch"):
        backend.align(_runtime_request(backend, tmp_path))


def test_mms_aligner_translates_pinned_api_assertions_to_stable_failure(tmp_path: Path) -> None:
    class RejectingAligner:
        @staticmethod
        def load_audio(*args: object) -> None:
            raise AssertionError("unsupported waveform")

    runtime = _resolved_runtime_info()
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(
            runtime_info=runtime,
            aligner=RejectingAligner(),
            model=SimpleNamespace(dtype="float16", device="cuda"),
            tokenizer="tokenizer",
        ),
    )
    request = backend.build_request(
        audio_path=tmp_path / "take.wav",
        audio_sha256=_digest("audio"),
        spoken_text="Gratia",
        segmentation_artifact_sha256=_digest("segment"),
        config_sha256=_digest("config"),
    )

    with pytest.raises(CorpusFailure, match="unsupported waveform") as error:
        backend.align(request)

    assert error.value.code == "ALIGNER_UNAVAILABLE"


def test_mms_aligner_preserves_low_confidence_from_empty_postprocessed_spans(
    tmp_path: Path,
) -> None:
    class EmptySpanAligner:
        load_audio = staticmethod(lambda *args: "waveform")
        generate_emissions = staticmethod(lambda *args: ("emissions", 0.02))
        preprocess_text = staticmethod(
            lambda *args: (
                ("<star>", "g r a t i a", "<star>"),
                ("<star>", "gratia", "<star>"),
            )
        )
        get_alignments = staticmethod(lambda *args: ("segments", "scores", 0))
        get_spans = staticmethod(lambda *args: "spans")
        postprocess_results = staticmethod(lambda *args: ())

    runtime = _resolved_runtime_info()
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cuda",
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        dependencies=SimpleNamespace(
            runtime_info=runtime,
            aligner=EmptySpanAligner(),
            model=SimpleNamespace(dtype="float16", device="cuda"),
            tokenizer="tokenizer",
        ),
    )

    with pytest.raises(CorpusFailure, match="MMS returned no word spans") as error:
        backend.align(_runtime_request(backend, tmp_path, "Gratia"))

    assert error.value.code == "ALIGNMENT_LOW_CONFIDENCE"


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


def test_loader_does_not_swallow_unrelated_programmer_type_errors() -> None:
    modules = {
        "torch": SimpleNamespace(__version__="2.5.1"),
        "huggingface_hub": SimpleNamespace(
            snapshot_download=lambda **kwargs: (_ for _ in ()).throw(TypeError("programmer bug"))
        ),
        "ctc_forced_aligner": SimpleNamespace(__version__="0.3.0"),
        "transformers": SimpleNamespace(__version__="4.48.0"),
    }
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cpu",
        dtype="float32",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        module_loader=modules.__getitem__,
        tool_commit_resolver=lambda: ALIGNER_COMMIT,
    )

    with pytest.raises(TypeError, match="programmer bug"):
        backend.runtime_info()


def test_loader_rejects_snapshot_resolved_to_unpinned_commit(tmp_path: Path) -> None:
    wrong_snapshot = tmp_path / "snapshots" / ("a" * 40)
    wrong_snapshot.mkdir(parents=True)
    modules = {
        "torch": SimpleNamespace(__version__="2.5.1"),
        "huggingface_hub": SimpleNamespace(snapshot_download=lambda **kwargs: str(wrong_snapshot)),
        "ctc_forced_aligner": SimpleNamespace(__version__="0.3.0"),
        "transformers": SimpleNamespace(__version__="4.48.0"),
    }
    backend = MmsCtcAligner(
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
        device="cpu",
        dtype="float32",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        module_loader=modules.__getitem__,
        tool_commit_resolver=lambda: ALIGNER_COMMIT,
    )

    with pytest.raises(CorpusFailure, match="resolved model revision"):
        backend.runtime_info()


@pytest.mark.parametrize("requested_device", ("cuda", "cpu"))
def test_mms_loader_resolves_unsupported_cpu_half_precision_to_float32(
    tmp_path: Path, requested_device: str
) -> None:
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
        device=requested_device,
        dtype="float16",
        window_seconds=30,
        context_seconds=2,
        batch_size=4,
        module_loader=modules.__getitem__,
        tool_commit_resolver=lambda: ALIGNER_COMMIT,
        dependency_version_resolver=lambda name: "1.3.1",
        allow_cpu_fallback=True,
    )

    info = backend.runtime_info()

    assert loaded == {"dtype": "float32", "device": "cpu"}
    assert info.device_name == "CPU"
    assert info.peak_memory_bytes is None
    assert info.requested_dtype == "float16"
    assert info.resolved_dtype == "float32"
    request = _runtime_request(backend, tmp_path, "Gratia")
    assert request.effective_parameters["requested_device"] == requested_device
    assert request.effective_parameters["resolved_device"] == "cpu"
    assert request.effective_parameters["resolved_dtype"] == "float32"


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
