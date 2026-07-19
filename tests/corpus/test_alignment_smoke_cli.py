from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from latintts.corpus import mms_alignment as mms_alignment_module
from latintts.corpus.cli import alignment_smoke_test, main
from latintts.corpus.config import CorpusConfig
from latintts.corpus.mms_alignment import ALIGNER_COMMIT
from latintts.corpus.paths import CorpusPaths


def _config() -> CorpusConfig:
    return CorpusConfig(
        raw={
            "alignment": {
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
        },
        digest="a" * 64,
    )


def _cpu_loader_modules(snapshot: Path) -> dict[str, object]:
    class FakeModel:
        dtype = "float32"
        device = "cpu"

        def to(self, device: str) -> FakeModel:
            self.device = device
            return self

        def eval(self) -> FakeModel:
            return self

    class ModelFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            return FakeModel()

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> str:
            return "tokenizer"

    class UnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    return {
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


def test_alignment_smoke_writes_reproducible_runtime_audit_with_fake_modules(
    tmp_path: Path,
) -> None:
    revision = "f37ba6bf1673872e07519fb951866cb2a32a6d7f"
    snapshot = tmp_path / "cache" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"fake weights")

    class FakeModel:
        dtype = "float16"
        device = "cuda"

        def to(self, device: str) -> FakeModel:
            return self

        def eval(self) -> FakeModel:
            return self

    class Factory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            return FakeModel()

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> str:
            return "tokenizer"

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name() -> str:
            return "RTX 4080 Fake"

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def max_memory_allocated() -> int:
            return 123456

    modules = {
        "torch": SimpleNamespace(
            __version__="2.5.1", float16="float16", float32="float32", cuda=FakeCuda()
        ),
        "huggingface_hub": SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot)),
        "ctc_forced_aligner": SimpleNamespace(__version__="0.3.0"),
        "transformers": SimpleNamespace(
            __version__="4.48.0", AutoModelForCTC=Factory, AutoTokenizer=TokenizerFactory
        ),
    }
    commands: list[list[str]] = []
    command_options: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        command_options.append(kwargs)
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout="RTX 4080 Fake, 555.99, 16376 MiB\n")
        if command[-2:] == ["pip", "freeze"]:
            return SimpleNamespace(
                stdout=(
                    "torch==2.5.1\n"
                    "private @ https://alice:secret@example.invalid/repo.git\n"
                    "editable @ file:///C:/Users/Alice/private-package\n"
                )
            )
        return SimpleNamespace(stdout="deadbeef" * 5 + "\n")

    paths = CorpusPaths.from_project_root(tmp_path)
    exit_code = alignment_smoke_test(
        paths,
        _config(),
        cache_dir=tmp_path / "cache",
        module_loader=modules.__getitem__,
        tool_commit_resolver=lambda: ALIGNER_COMMIT,
        dependency_version_resolver=lambda name: "1.3.1",
        run_command=fake_run,
        now=lambda: datetime(2026, 7, 19, 8, 0, tzinfo=timezone.utc),
    )

    assert exit_code == 0
    runtime = paths.alignments / "runtime"
    smoke = json.loads((runtime / "smoke.json").read_text(encoding="utf-8"))
    environment = (runtime / "environment.txt").read_text(encoding="utf-8")
    assert smoke["success"] is True
    assert smoke["failure_code"] is None
    assert smoke["model_revision"] == revision
    assert smoke["model_license"] == "CC-BY-NC-4.0"
    assert smoke["aligner_commit"] == ALIGNER_COMMIT
    assert len(smoke["model_weights_sha256"]) == 64
    assert smoke["cuda_available"] is True
    assert smoke["requested_device"] == "cuda"
    assert smoke["requested_dtype"] == "float16"
    assert smoke["resolved_device"] == "cuda"
    assert smoke["resolved_dtype"] == "float16"
    assert smoke["uroman_version"] == "1.3.1"
    assert smoke["measurement_scope"] == "model-load"
    assert smoke["model_weight_manifest"][0]["filename"] == "model.safetensors"
    assert smoke["device_name"] == "RTX 4080 Fake"
    assert smoke["peak_memory_bytes"] == 123456
    assert smoke["timestamp"] == "2026-07-19T08:00:00+00:00"
    assert f"Run ID: {smoke['run_id']}" in environment
    assert smoke["environment_sha256"] == hashlib.sha256(environment.encode("utf-8")).hexdigest()
    assert "Python version:" in environment
    assert "torch==2.5.1" in environment
    assert "RTX 4080 Fake, 555.99" in environment
    assert "secret" not in environment
    assert "C:/Users/Alice" not in environment
    assert "private @ <redacted>" in environment
    assert "editable @ <redacted>" in environment
    assert [command[0] for command in commands] == [
        str(Path(__import__("sys").executable)),
        "nvidia-smi",
        "git",
    ]
    assert command_options[2]["cwd"] == paths.project_root
    assert all(options["timeout"] == 30 for options in command_options)


@pytest.mark.parametrize(
    ("nvidia_failure", "failure_name"),
    (
        (FileNotFoundError("nvidia-smi"), "FileNotFoundError"),
        (PermissionError(r"C:\Users\Alice\nvidia-smi"), "PermissionError"),
        (
            subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="unsupported"),
            "CalledProcessError",
        ),
    ),
)
def test_alignment_smoke_keeps_nvidia_smi_optional_for_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nvidia_failure: BaseException,
    failure_name: str,
) -> None:
    revision = "f37ba6bf1673872e07519fb951866cb2a32a6d7f"
    repository_cache = tmp_path / "hub" / "models--latin"
    snapshot = repository_cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repository_cache / "blobs").mkdir()
    (snapshot / "model.safetensors").write_bytes(b"fake weights")
    observed_roots: list[Path | None] = []
    original_resolver = mms_alignment_module.resolve_model_weights

    def recording_resolver(snapshot: Path, *, cache_root: Path | None = None):  # type: ignore[no-untyped-def]
        observed_roots.append(cache_root)
        return original_resolver(snapshot, cache_root=cache_root)

    monkeypatch.setattr(mms_alignment_module, "resolve_model_weights", recording_resolver)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[0] == "nvidia-smi":
            raise nvidia_failure
        if command[-2:] == ["pip", "freeze"]:
            return SimpleNamespace(stdout="torch==2.5.1\n")
        return SimpleNamespace(stdout="deadbeef" * 5 + "\n")

    paths = CorpusPaths.from_project_root(tmp_path)
    exit_code = alignment_smoke_test(
        paths,
        _config(),
        module_loader=_cpu_loader_modules(snapshot).__getitem__,
        tool_commit_resolver=lambda: ALIGNER_COMMIT,
        dependency_version_resolver=lambda name: "1.3.1",
        run_command=fake_run,
        now=lambda: datetime(2026, 7, 19, 10, 0, tzinfo=timezone.utc),
    )

    smoke = json.loads((paths.alignments / "runtime" / "smoke.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert smoke["success"] is True
    assert smoke["failure_code"] is None
    assert smoke["nvidia_smi"] == f"unavailable: {failure_name}"
    assert smoke["cuda_available"] is False
    assert smoke["requested_device"] == "cuda"
    assert smoke["requested_dtype"] == "float16"
    assert smoke["resolved_device"] == "cpu"
    assert smoke["resolved_dtype"] == "float32"
    assert len(smoke["model_weights_sha256"]) == 64
    assert smoke["model_weight_manifest"][0]["filename"] == "model.safetensors"
    assert observed_roots == [repository_cache]


def test_align_smoke_cli_uses_local_cache_policy_by_default(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_smoke(paths: CorpusPaths, config: CorpusConfig, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("latintts.corpus.cli.CorpusConfig.load", lambda path: _config())
    monkeypatch.setattr("latintts.corpus.cli.alignment_smoke_test", fake_smoke)

    exit_code = main(["--project-root", str(tmp_path), "align", "--smoke-test"])

    assert exit_code == 0
    assert captured["local_files_only"] is True
    assert captured["cache_dir"] == tmp_path / "local-data" / "cache" / "huggingface"


def test_alignment_smoke_propagates_programmer_type_error(tmp_path: Path) -> None:
    def programmer_bug(name: str) -> object:
        raise TypeError(f"unexpected loader contract for {name}")

    def successful_command(command: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="available\n")

    with pytest.raises(TypeError, match="unexpected loader contract"):
        alignment_smoke_test(
            CorpusPaths.from_project_root(tmp_path),
            _config(),
            module_loader=programmer_bug,
            run_command=successful_command,
        )


def test_align_smoke_cli_propagates_programmer_type_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("latintts.corpus.cli.CorpusConfig.load", lambda path: _config())
    monkeypatch.setattr(
        "latintts.corpus.cli.alignment_smoke_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("programmer bug")),
    )

    with pytest.raises(TypeError, match="programmer bug"):
        main(["--project-root", str(tmp_path), "align", "--smoke-test"])


def test_alignment_smoke_persists_stable_failure_when_dependencies_are_unavailable(
    tmp_path: Path,
) -> None:
    def missing_module(name: str) -> object:
        raise ImportError(name)

    def available_command(command: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="available\n")

    paths = CorpusPaths.from_project_root(tmp_path)
    exit_code = alignment_smoke_test(
        paths,
        _config(),
        module_loader=missing_module,
        run_command=available_command,
    )

    smoke = json.loads((paths.alignments / "runtime" / "smoke.json").read_text(encoding="utf-8"))
    environment = (paths.alignments / "runtime" / "environment.txt").read_text(encoding="utf-8")
    assert exit_code == 1
    assert smoke["success"] is False
    assert smoke["failure_code"] == "ALIGNER_UNAVAILABLE"
    assert smoke["model_weights_sha256"] is None
    assert smoke["pip_freeze"] == "available"
    assert smoke["message"] == "pinned alignment backend unavailable"
    assert str(Path(sys.executable).parent) not in environment


@pytest.mark.parametrize(
    ("failure", "failure_name"),
    (
        (PermissionError(r"C:\Users\Alice\private"), "PermissionError"),
        (OSError(r"C:\Users\Alice\private"), "OSError"),
        (
            subprocess.CalledProcessError(1, ["pip", "freeze"], stderr=r"C:\Users\Alice\private"),
            "CalledProcessError",
        ),
        (subprocess.TimeoutExpired(["pip", "freeze"], 30), "TimeoutExpired"),
        (
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, r"C:\Users\Alice\private"),
            "UnicodeDecodeError",
        ),
    ),
)
def test_alignment_smoke_persists_sanitized_failure_for_environment_command_errors(
    tmp_path: Path, failure: BaseException, failure_name: str
) -> None:
    module_calls: list[str] = []

    def backend_must_not_load(name: str) -> object:
        module_calls.append(name)
        raise AssertionError("backend should not load after environment capture failure")

    def failed_command(command: list[str], **kwargs: object) -> object:
        raise failure

    paths = CorpusPaths.from_project_root(tmp_path)
    exit_code = alignment_smoke_test(
        paths,
        _config(),
        module_loader=backend_must_not_load,
        run_command=failed_command,  # type: ignore[arg-type]
        now=lambda: datetime(2026, 7, 19, 9, 0, tzinfo=timezone.utc),
    )

    runtime = paths.alignments / "runtime"
    smoke_text = (runtime / "smoke.json").read_text(encoding="utf-8")
    smoke = json.loads(smoke_text)
    environment = (runtime / "environment.txt").read_text(encoding="utf-8")
    assert exit_code == 1
    assert module_calls == []
    assert smoke["success"] is False
    assert smoke["failure_code"] == "ALIGNER_UNAVAILABLE"
    assert smoke["message"] == "environment evidence unavailable"
    assert smoke["pip_freeze"] == f"unavailable: {failure_name}"
    assert f"Run ID: {smoke['run_id']}" in environment
    assert smoke["environment_sha256"] == hashlib.sha256(environment.encode()).hexdigest()
    assert "Alice" not in smoke_text + environment
    assert "private" not in smoke_text + environment
