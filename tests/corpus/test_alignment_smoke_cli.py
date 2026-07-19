from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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


def test_alignment_smoke_persists_stable_failure_when_dependencies_are_unavailable(
    tmp_path: Path,
) -> None:
    def missing_module(name: str) -> object:
        raise ImportError(name)

    def missing_command(command: list[str], **kwargs: object) -> object:
        raise FileNotFoundError(command[0])

    paths = CorpusPaths.from_project_root(tmp_path)
    exit_code = alignment_smoke_test(
        paths,
        _config(),
        module_loader=missing_module,
        run_command=missing_command,  # type: ignore[arg-type]
    )

    smoke = json.loads((paths.alignments / "runtime" / "smoke.json").read_text(encoding="utf-8"))
    environment = (paths.alignments / "runtime" / "environment.txt").read_text(encoding="utf-8")
    assert exit_code == 1
    assert smoke["success"] is False
    assert smoke["failure_code"] == "ALIGNER_UNAVAILABLE"
    assert smoke["model_weights_sha256"] is None
    assert smoke["pip_freeze"] == "unavailable: FileNotFoundError"
    assert smoke["message"] == "pinned alignment backend unavailable"
    assert str(Path(sys.executable).parent) not in environment
