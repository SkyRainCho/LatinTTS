from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Any

from latintts.corpus.alignment import (
    AlignmentRequest,
    AlignmentResult,
    AlignmentToken,
    WordSpan,
    transform_latin_for_alignment,
)
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure
from latintts.normalization import tokenize_words

MODEL_LICENSE = "CC-BY-NC-4.0"
BACKEND_VERSION = "0.3.0"
ALIGNER_COMMIT = "11855d1de76af2b490dd2e8e2db2661805ae90a0"
MODEL_ID = "MahmoudAshraf/mms-300m-1130-forced-aligner"
MODEL_REVISION = "49402e9577b1158620820667c218cd494cc44486"
_ALIGNMENT_CONFIG_FIELDS = {
    "backend",
    "model_id",
    "model_revision",
    "language",
    "romanize",
    "split_size",
    "star_frequency",
    "window_seconds",
    "context_seconds",
    "batch_size",
    "dtype",
    "device",
    "license",
}


@dataclass(frozen=True, slots=True)
class ModelWeightFile:
    filename: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class BackendRuntimeInfo:
    torch_version: str
    transformers_version: str
    device_name: str
    peak_memory_bytes: int | None
    aligner_version: str = ""
    aligner_commit: str = ""
    resolved_model_revision: str = ""
    model_weights_sha256: str = ""
    cuda_available: bool = False
    requested_device: str = ""
    requested_dtype: str = ""
    resolved_device: str = ""
    resolved_dtype: str = ""
    uroman_version: str = ""
    model_weight_manifest: tuple[ModelWeightFile, ...] = ()
    measurement_scope: str = ""


def _installed_aligner_commit() -> str:
    try:
        direct_url = importlib.metadata.distribution("ctc-forced-aligner").read_text(
            "direct_url.json"
        )
        raw = json.loads(direct_url or "")
        commit = raw["vcs_info"]["commit_id"]
    except (
        importlib.metadata.PackageNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as error:
        raise CorpusFailure(
            "ALIGNER_UNAVAILABLE", "ctc-forced-aligner installed commit is unverifiable"
        ) from error
    if not isinstance(commit, str):
        raise CorpusFailure(
            "ALIGNER_UNAVAILABLE", "ctc-forced-aligner installed commit is unverifiable"
        )
    return commit


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_index_weight_path(snapshot: Path, index: Path, filename: str) -> Path:
    posix_path = PurePosixPath(filename)
    if (
        not filename
        or "\\" in filename
        or posix_path.is_absolute()
        or PureWindowsPath(filename).is_absolute()
        or posix_path.as_posix() != filename
        or ".." in posix_path.parts
    ):
        raise CorpusFailure(
            "ALIGNER_UNAVAILABLE", "MMS shard filename must be a canonical relative path"
        )
    candidate = index.parent.joinpath(*posix_path.parts)
    try:
        candidate.relative_to(snapshot)
    except ValueError as error:
        raise CorpusFailure(
            "ALIGNER_UNAVAILABLE", "MMS shard filename must be a canonical relative path"
        ) from error
    return candidate


def _validate_cached_weight_path(path: Path, cache_root: Path) -> None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(cache_root.resolve(strict=True))
    except FileNotFoundError as error:
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "MMS weight file is missing") from error
    except (OSError, RuntimeError, ValueError) as error:
        raise CorpusFailure(
            "ALIGNER_UNAVAILABLE", "MMS weight link resolves outside the Hugging Face cache root"
        ) from error
    if not resolved.is_file():
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "MMS weight path is not a file")


def _effective_hf_cache_root(snapshot: Path, configured_cache: Path | None) -> Path:
    if configured_cache is not None:
        return configured_cache
    repository_cache = snapshot.parent.parent
    if (
        snapshot.parent.name == "snapshots"
        and repository_cache.name.startswith("models--")
        and (repository_cache / "blobs").is_dir()
    ):
        return repository_cache
    return snapshot


def resolve_model_weights(
    snapshot: Path, *, cache_root: Path | None = None
) -> tuple[str, tuple[ModelWeightFile, ...]]:
    cache_root = snapshot if cache_root is None else cache_root
    safetensors = tuple(sorted(snapshot.rglob("*.safetensors")))
    indexes = tuple(sorted(snapshot.rglob("*.safetensors.index.json")))
    if indexes:
        if len(indexes) != 1:
            raise CorpusFailure("ALIGNER_UNAVAILABLE", "MMS snapshot has conflicting shard indexes")
        index = indexes[0]
        _validate_cached_weight_path(index, cache_root)
        try:
            raw_index = json.loads(index.read_text(encoding="utf-8"))
            weight_map = raw_index["weight_map"]
        except (json.JSONDecodeError, KeyError, OSError, TypeError) as error:
            raise CorpusFailure("ALIGNER_UNAVAILABLE", "MMS shard index is invalid") from error
        if (
            type(weight_map) is not dict
            or not weight_map
            or any(
                type(key) is not str or type(value) is not str for key, value in weight_map.items()
            )
        ):
            raise CorpusFailure("ALIGNER_UNAVAILABLE", "MMS shard index is invalid")
        referenced: set[Path] = set()
        for filename in weight_map.values():
            candidate = _canonical_index_weight_path(snapshot, index, filename)
            if candidate.suffix != ".safetensors":
                raise CorpusFailure("ALIGNER_UNAVAILABLE", "MMS shard index mixes weight formats")
            referenced.add(candidate)
        actual = set(safetensors)
        if actual != referenced:
            raise CorpusFailure(
                "ALIGNER_UNAVAILABLE", "MMS shard index has missing or extra weight files"
            )
        selected = (index, *sorted(referenced))
    elif len(safetensors) == 1:
        selected = safetensors
    elif len(safetensors) > 1:
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "multiple unindexed MMS safetensors files")
    else:
        bins = tuple(sorted(snapshot.rglob("*.bin")))
        if len(bins) != 1:
            raise CorpusFailure("ALIGNER_UNAVAILABLE", "MMS snapshot has no unique model weights")
        selected = bins
    for path in selected:
        _validate_cached_weight_path(path, cache_root)
    manifest = tuple(
        ModelWeightFile(
            filename=path.relative_to(snapshot).as_posix(),
            sha256=_file_digest(path),
            size_bytes=path.stat().st_size,
        )
        for path in selected
    )
    encoded = json.dumps(
        [item.to_dict() for item in manifest],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), manifest


class MmsCtcAligner:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        device: str,
        dtype: str,
        window_seconds: int,
        context_seconds: int,
        batch_size: int,
        dependencies: Any | None = None,
        cache_dir: Path | None = None,
        local_files_only: bool = True,
        module_loader: Callable[[str], Any] | None = None,
        tool_commit_resolver: Callable[[], str] | None = None,
        dependency_version_resolver: Callable[[str], str] | None = None,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self.dtype_name = dtype
        self.window_seconds = window_seconds
        self.context_seconds = context_seconds
        self.batch_size = batch_size
        self.dependencies = dependencies
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.module_loader = module_loader or importlib.import_module
        self.tool_commit_resolver = tool_commit_resolver or _installed_aligner_commit
        self.dependency_version_resolver = dependency_version_resolver or importlib.metadata.version
        self.allow_cpu_fallback = allow_cpu_fallback
        injected_runtime = (
            getattr(dependencies, "runtime_info", None) if dependencies is not None else None
        )
        self._runtime_info = (
            injected_runtime if isinstance(injected_runtime, BackendRuntimeInfo) else None
        )

    def _load(self) -> Any:
        if self.dependencies is not None:
            return self.dependencies
        try:
            torch = self.module_loader("torch")
            hub = self.module_loader("huggingface_hub")
            aligner = self.module_loader("ctc_forced_aligner")
            transformers = self.module_loader("transformers")
            aligner_version = str(aligner.__version__)
            aligner_commit = self.tool_commit_resolver()
            if aligner_version != BACKEND_VERSION or aligner_commit != ALIGNER_COMMIT:
                raise CorpusFailure(
                    "ALIGNER_UNAVAILABLE", "ctc-forced-aligner version or commit is not pinned"
                )
            model_path = Path(
                hub.snapshot_download(
                    repo_id=self.model_id,
                    revision=self.model_revision,
                    cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                    local_files_only=self.local_files_only,
                )
            ).resolve()
            if model_path.name != self.model_revision:
                raise CorpusFailure(
                    "ALIGNER_UNAVAILABLE", "MMS resolved model revision does not match the pin"
                )
            model_weights_sha256, model_weight_manifest = resolve_model_weights(
                model_path,
                cache_root=_effective_hf_cache_root(model_path, self.cache_dir),
            )
            runtime_device = self.device
            runtime_dtype = self.dtype_name
            cuda_available = bool(torch.cuda.is_available())
            if self.device.startswith("cuda") and not cuda_available:
                if not self.allow_cpu_fallback:
                    raise CorpusFailure("ALIGNER_UNAVAILABLE", "CUDA is unavailable")
                runtime_device = "cpu"
                runtime_dtype = "float32"
            if runtime_device == "cpu" and runtime_dtype == "float16":
                runtime_dtype = "float32"
            dtype = getattr(torch, runtime_dtype)
            if runtime_device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            model = (
                transformers.AutoModelForCTC.from_pretrained(
                    str(model_path),
                    revision=self.model_revision,
                    trust_remote_code=False,
                    local_files_only=self.local_files_only,
                    cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                    torch_dtype=dtype,
                )
                .to(runtime_device)
                .eval()
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(model_path),
                revision=self.model_revision,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
                cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                word_delimiter_token=None,
            )
            torch_version = str(torch.__version__)
            transformers_version = str(transformers.__version__)
            uroman_version = str(self.dependency_version_resolver("uroman"))
            if not torch_version or not transformers_version or not uroman_version:
                raise CorpusFailure("ALIGNER_UNAVAILABLE", "dependency versions are unavailable")
            device_name = (
                str(torch.cuda.get_device_name()) if runtime_device.startswith("cuda") else "CPU"
            )
            if runtime_device.startswith("cuda"):
                torch.cuda.synchronize()
            peak_memory = (
                int(torch.cuda.max_memory_allocated())
                if runtime_device.startswith("cuda")
                else None
            )
        except CorpusFailure:
            raise
        except (ImportError, AssertionError, OSError, RuntimeError, ValueError) as error:
            raise CorpusFailure("ALIGNER_UNAVAILABLE", str(error)) from error
        self.dependencies = SimpleNamespace(
            torch=torch,
            aligner=aligner,
            model=model,
            tokenizer=tokenizer,
        )
        self._runtime_info = BackendRuntimeInfo(
            torch_version=torch_version,
            transformers_version=transformers_version,
            device_name=device_name,
            peak_memory_bytes=peak_memory,
            aligner_version=aligner_version,
            aligner_commit=aligner_commit,
            resolved_model_revision=model_path.name,
            model_weights_sha256=model_weights_sha256,
            model_weight_manifest=model_weight_manifest,
            cuda_available=cuda_available,
            requested_device=self.device,
            requested_dtype=self.dtype_name,
            resolved_device=runtime_device,
            resolved_dtype=runtime_dtype,
            uroman_version=uroman_version,
            measurement_scope="model-load",
        )
        return self.dependencies

    def runtime_info(self) -> BackendRuntimeInfo:
        self._load()
        if self._runtime_info is None:
            raise CorpusFailure(
                "ALIGNER_UNAVAILABLE", "runtime info is unavailable for injected dependencies"
            )
        return self._runtime_info

    def _effective_parameters(self, runtime: BackendRuntimeInfo) -> dict[str, object]:
        return {
            "language": "lat",
            "romanize": True,
            "split_size": "word",
            "star_frequency": "edges",
            "window_seconds": self.window_seconds,
            "context_seconds": self.context_seconds,
            "batch_size": self.batch_size,
            "requested_device": runtime.requested_device,
            "requested_dtype": runtime.requested_dtype,
            "resolved_device": runtime.resolved_device,
            "resolved_dtype": runtime.resolved_dtype,
            "resolved_model_revision": runtime.resolved_model_revision,
            "model_weights_sha256": runtime.model_weights_sha256,
            "model_weight_manifest": [item.to_dict() for item in runtime.model_weight_manifest],
            "aligner_version": runtime.aligner_version,
            "aligner_commit": runtime.aligner_commit,
            "torch_version": runtime.torch_version,
            "transformers_version": runtime.transformers_version,
            "uroman_version": runtime.uroman_version,
            "model_license": MODEL_LICENSE,
        }

    def build_request(
        self,
        *,
        audio_path: Path,
        audio_sha256: str,
        spoken_text: str,
        segmentation_artifact_sha256: str,
        config_sha256: str,
    ) -> AlignmentRequest:
        runtime = self.runtime_info()
        return AlignmentRequest(
            audio_path=audio_path,
            audio_sha256=audio_sha256,
            spoken_text=spoken_text,
            segmentation_artifact_sha256=segmentation_artifact_sha256,
            backend="mms-ctc",
            backend_version=BACKEND_VERSION,
            model_id=self.model_id,
            model_revision=self.model_revision,
            model_license=MODEL_LICENSE,
            config_sha256=config_sha256,
            effective_parameters=self._effective_parameters(runtime),
        )

    def align(self, request: AlignmentRequest) -> AlignmentResult:
        expected_provenance = {
            "backend": "mms-ctc",
            "backend_version": BACKEND_VERSION,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_license": MODEL_LICENSE,
        }
        for field, expected in expected_provenance.items():
            if getattr(request, field) != expected:
                raise ValueError(f"alignment request {field} does not match the MMS backend")
        expected_request = self.build_request(
            audio_path=request.audio_path,
            audio_sha256=request.audio_sha256,
            spoken_text=request.spoken_text,
            segmentation_artifact_sha256=request.segmentation_artifact_sha256,
            config_sha256=request.config_sha256,
        )
        if request.effective_parameters != expected_request.effective_parameters:
            raise ValueError("alignment request runtime identity does not match loaded runtime")
        dependencies = self._load()
        try:
            waveform = dependencies.aligner.load_audio(
                str(request.audio_path), dependencies.model.dtype, dependencies.model.device
            )
            emissions, stride = dependencies.aligner.generate_emissions(
                dependencies.model,
                waveform,
                self.window_seconds,
                self.context_seconds,
                self.batch_size,
            )
            tokens_starred, text_starred = dependencies.aligner.preprocess_text(
                request.alignment_text, True, "lat", "word", "edges"
            )
            segments, scores, blank = dependencies.aligner.get_alignments(
                emissions, tokens_starred, dependencies.tokenizer
            )
            spans = dependencies.aligner.get_spans(tokens_starred, segments, blank)
            raw_results = dependencies.aligner.postprocess_results(
                text_starred, spans, stride, scores
            )
            return normalize_mms_output(
                spoken_text=request.spoken_text,
                text_starred=tuple(text_starred),
                tokens_starred=tuple(tokens_starred),
                raw_results=tuple(raw_results),
                backend_version=BACKEND_VERSION,
                model_id=self.model_id,
                model_revision=self.model_revision,
                runtime_identity=self._effective_parameters(self.runtime_info()),
            )
        except CorpusFailure:
            raise
        except (ImportError, AssertionError, OSError, RuntimeError, ValueError) as error:
            message = str(error)
            if "out of memory" in message.casefold():
                raise CorpusFailure(
                    "ALIGNER_UNAVAILABLE", "MMS alignment CUDA out of memory"
                ) from error
            raise CorpusFailure(
                "ALIGNER_UNAVAILABLE", f"MMS alignment failed: {message}"
            ) from error


def create_alignment_backend(
    config: CorpusConfig,
    *,
    dependencies: Any | None = None,
    cache_dir: Path | None = None,
    local_files_only: bool = True,
    module_loader: Callable[[str], Any] | None = None,
    tool_commit_resolver: Callable[[], str] | None = None,
    dependency_version_resolver: Callable[[str], str] | None = None,
    allow_cpu_fallback: bool = True,
) -> MmsCtcAligner:
    """Create the sole licensed, pinned alignment backend from corpus config."""
    alignment = config.raw.get("alignment")
    if type(alignment) is not dict or set(alignment) != _ALIGNMENT_CONFIG_FIELDS:
        raise ValueError("alignment config must contain exact fields")
    required = {
        "backend": "mms-ctc",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "language": "lat",
        "romanize": True,
        "split_size": "word",
        "star_frequency": "edges",
        "license": MODEL_LICENSE,
    }
    for field, expected in required.items():
        if alignment[field] != expected or type(alignment[field]) is not type(expected):
            raise ValueError(f"alignment.{field} must equal the pinned value")
    for field in ("window_seconds", "context_seconds", "batch_size"):
        if type(alignment[field]) is not int or alignment[field] <= 0:
            raise ValueError(f"alignment.{field} must be a positive integer")
    if alignment["dtype"] not in {"float16", "float32"}:
        raise ValueError("alignment.dtype is unsupported")
    if alignment["device"] not in {"cuda", "cpu"}:
        raise ValueError("alignment.device is unsupported")
    return MmsCtcAligner(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        device=alignment["device"],
        dtype=alignment["dtype"],
        window_seconds=alignment["window_seconds"],
        context_seconds=alignment["context_seconds"],
        batch_size=alignment["batch_size"],
        dependencies=dependencies,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        module_loader=module_loader,
        tool_commit_resolver=tool_commit_resolver,
        dependency_version_resolver=dependency_version_resolver,
        allow_cpu_fallback=allow_cpu_fallback,
    )


def _number(value: object) -> float:
    item = getattr(value, "item", None)
    scalar = item() if callable(item) else value
    return float(scalar)  # type: ignore[arg-type]


def _probability(value: object) -> float:
    number = _number(value)
    return max(0.0, min(1.0, math.exp(number) if number < 0 else number))


def normalize_mms_output(
    *,
    spoken_text: str,
    text_starred: tuple[str, ...],
    tokens_starred: tuple[str, ...],
    raw_results: tuple[dict[str, object], ...],
    backend_version: str,
    model_id: str,
    model_revision: str,
    runtime_identity: dict[str, object] | None = None,
) -> AlignmentResult:
    """Convert MMS output at the sole third-party/strict-contract boundary."""
    display_words = tuple(item for item in text_starred if item and item != "<star>")
    alignment_forms = tuple(
        transform_latin_for_alignment(item.replace(" ", ""))
        for item in tokens_starred
        if item and item != "<star>"
    )
    spoken_words = tokenize_words(spoken_text)
    if not raw_results:
        raise CorpusFailure("ALIGNMENT_LOW_CONFIDENCE", "MMS returned no word spans")
    if not (len(raw_results) == len(display_words) == len(alignment_forms) == len(spoken_words)):
        raise CorpusFailure(
            "ALIGNMENT_LOW_CONFIDENCE", "MMS did not return one span per spoken token"
        )
    if " ".join(alignment_forms) != transform_latin_for_alignment(spoken_text):
        raise CorpusFailure(
            "ALIGNMENT_LOW_CONFIDENCE", "MMS backend tokens do not match spoken text"
        )
    tokens = tuple(
        AlignmentToken(index, form, index, spoken.surface)
        for index, (form, spoken) in enumerate(zip(alignment_forms, spoken_words, strict=True))
    )
    words = tuple(
        WordSpan(
            text=display,
            start_seconds=_number(item["start"]),
            end_seconds=_number(item["end"]),
            score=_probability(item["score"]),
            spoken_token_index=index,
        )
        for index, (display, item) in enumerate(zip(display_words, raw_results, strict=True))
    )
    raw_output: dict[str, object] = {
        "spoken_text": spoken_text,
        "segments": [
            {
                "text": word.text,
                "start": word.start_seconds,
                "end": word.end_seconds,
                "raw_score": _number(item["score"]),
                "normalized_score": word.score,
                "spoken_token_index": word.spoken_token_index,
            }
            for item, word in zip(raw_results, words, strict=True)
        ],
    }
    if runtime_identity is not None:
        raw_output["runtime_identity"] = runtime_identity
    return AlignmentResult(
        backend="mms-ctc",
        backend_version=backend_version,
        model_id=model_id,
        model_revision=model_revision,
        model_license=MODEL_LICENSE,
        alignment_text=" ".join(alignment_forms),
        tokens=tokens,
        words=words,
        coverage=1.0,
        mean_score=statistics.fmean(word.score for word in words),
        alignment_level="word",
        phoneme_timing_status="not_estimated",
        warnings=(),
        raw_output=raw_output,
    )
