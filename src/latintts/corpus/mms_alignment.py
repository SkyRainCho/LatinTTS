from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
MODEL_REVISION = "f37ba6bf1673872e07519fb951866cb2a32a6d7f"
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


def _weights_digest(snapshot: Path) -> str:
    weights = tuple(
        path
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path.suffix in {".bin", ".pt", ".safetensors"}
    )
    if not weights:
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "MMS snapshot contains no model weights")
    digest = hashlib.sha256()
    for path in weights:
        digest.update(path.relative_to(snapshot).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


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
        self.allow_cpu_fallback = allow_cpu_fallback
        self._runtime_info: BackendRuntimeInfo | None = None

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
            model_weights_sha256 = _weights_digest(model_path)
            runtime_device = self.device
            runtime_dtype = self.dtype_name
            cuda_available = bool(torch.cuda.is_available())
            if self.device.startswith("cuda") and not cuda_available:
                if not self.allow_cpu_fallback:
                    raise CorpusFailure("ALIGNER_UNAVAILABLE", "CUDA is unavailable")
                runtime_device = "cpu"
                runtime_dtype = "float32"
            dtype = getattr(torch, runtime_dtype)
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
            if not torch_version or not transformers_version:
                raise CorpusFailure("ALIGNER_UNAVAILABLE", "dependency versions are unavailable")
            device_name = (
                str(torch.cuda.get_device_name()) if runtime_device.startswith("cuda") else "CPU"
            )
            peak_memory = (
                int(torch.cuda.max_memory_allocated())
                if runtime_device.startswith("cuda")
                else None
            )
        except CorpusFailure:
            raise
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
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
            cuda_available=cuda_available,
        )
        return self.dependencies

    def runtime_info(self) -> BackendRuntimeInfo:
        self._load()
        if self._runtime_info is None:
            raise CorpusFailure(
                "ALIGNER_UNAVAILABLE", "runtime info is unavailable for injected dependencies"
            )
        return self._runtime_info

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
                request.spoken_text, True, "lat", "word", "edges"
            )
            segments, scores, blank = dependencies.aligner.get_alignments(
                emissions, tokens_starred, dependencies.tokenizer
            )
            spans = dependencies.aligner.get_spans(tokens_starred, segments, blank)
            raw_results = dependencies.aligner.postprocess_results(
                text_starred, spans, stride, scores
            )
        except RuntimeError as error:
            message = str(error)
            if "out of memory" in message.casefold():
                raise CorpusFailure(
                    "ALIGNER_UNAVAILABLE", "MMS alignment CUDA out of memory"
                ) from error
            raise CorpusFailure(
                "ALIGNER_UNAVAILABLE", f"MMS alignment failed: {message}"
            ) from error
        return normalize_mms_output(
            spoken_text=request.spoken_text,
            text_starred=tuple(text_starred),
            tokens_starred=tuple(tokens_starred),
            raw_results=tuple(raw_results),
            backend_version=BACKEND_VERSION,
            model_id=self.model_id,
            model_revision=self.model_revision,
        )


def create_alignment_backend(
    config: CorpusConfig,
    *,
    dependencies: Any | None = None,
    cache_dir: Path | None = None,
    local_files_only: bool = True,
    module_loader: Callable[[str], Any] | None = None,
    tool_commit_resolver: Callable[[], str] | None = None,
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
) -> AlignmentResult:
    """Convert MMS output at the sole third-party/strict-contract boundary."""
    display_words = tuple(item for item in text_starred if item)
    alignment_forms = tuple(item.replace(" ", "") for item in tokens_starred if item)
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
    raw_output = {
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
