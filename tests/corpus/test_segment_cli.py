from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from latintts.corpus.cli import _ffmpeg_version, _validate_vad_result, main, segment_corpus
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from latintts.corpus.vad import SpeechInterval, VadFrame, VadResult
from tests.corpus.factories import recording

_MODEL_SHA256 = hashlib.sha256(b"fake-silero-weight").hexdigest()


class _FakeVad:
    def __init__(
        self,
        speech: tuple[SpeechInterval, ...],
        model_sha256: str = _MODEL_SHA256,
    ) -> None:
        self.speech = speech
        self.model_sha256 = model_sha256
        self.calls = 0

    def analyze(self, audio_path: Path) -> VadResult:
        self.calls += 1
        with wave.open(str(audio_path), "rb") as handle:
            sample_count = handle.getnframes()
        frames = tuple(
            VadFrame(start, min(start + 512, sample_count), 0.8)
            for start in range(0, sample_count, 512)
        )
        return VadResult(
            "silero-vad",
            "6.2.1",
            self.model_sha256,
            16000,
            frames,
            self.speech,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redigest(config: CorpusConfig) -> CorpusConfig:
    digest = hashlib.sha256(
        json.dumps(
            config.raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return CorpusConfig(config.raw, digest)


def _set_up(tmp_path: Path, *, content_type: str = "spoken") -> tuple[CorpusPaths, CorpusConfig]:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    config_dir = tmp_path / "config" / "corpus"
    config_dir.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json",
        config_dir / "pilot-v1.json",
    )
    config = CorpusConfig.load(config_dir / "pilot-v1.json")
    raw_root = paths.raw_spoken if content_type == "spoken" else paths.raw_sung
    raw = raw_root / "rec-1.wav"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(raw), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160000)
    record = recording("rec-1", 10.0, content_type=content_type)  # type: ignore[arg-type]
    record = replace(
        record,
        sha256=_sha256(raw),
        metadata=replace(record.metadata, sample_rate=16000),
        state=CorpusState.TRANSCRIPT_CONFIRMED,
    )
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (record.to_dict(),))
    write_jsonl_atomic(
        paths.manifests / "pilot-selection.json",
        (
            {
                "schema_version": "1",
                "strategy": "explicit-v1",
                "recording_ids": ["rec-1"],
                "inventory_hashes": [record.sha256],
            },
        ),
    )
    write_jsonl_atomic(
        paths.manifests / "transcripts.jsonl",
        (
            {
                "schema_version": "1",
                "recording_id": "rec-1",
                "spoken_text": "Pater noster",
                "state": "TRANSCRIPT_CONFIRMED",
            },
        ),
    )
    return paths, config


def _transform(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
    source = Path(command[command.index("-i") + 1])
    shutil.copyfile(source, Path(command[-1]))
    return CompletedProcess(command, 0, "", "")


def _speech() -> tuple[SpeechInterval, ...]:
    return (
        SpeechInterval(0, 16000),
        SpeechInterval(19200, 32000),
        SpeechInterval(64000, 80000),
        SpeechInterval(84000, 96000),
        SpeechInterval(124800, 136000),
    )


def test_segment_writes_auditable_result_advances_state_and_reuses_it(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    backend = _FakeVad(_speech())

    assert segment_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_transform,
    )

    result_path = paths.alignments / "runs" / config.digest / "rec-1" / "segmentation.json"
    result = read_jsonl(result_path)[0]
    assert result["status"] == "success"
    assert result["config_sha256"] == config.digest
    assert result["vad"]["backend"] == "silero-vad"
    assert result["vad"]["model_version"] == "6.2.1"
    assert result["vad"]["model_sha256"] == _MODEL_SHA256
    assert result["vad"]["parameters"] == {
        "backend": "silero-vad",
        "model_version": "6.2.1",
        "threshold": 0.5,
        "neg_threshold": 0.35,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "max_speech_duration_s": 60.0,
        "speech_pad_ms": 30,
        "min_silence_at_max_speech": 98,
        "use_max_poss_sil_at_max_speech": True,
        "sample_rate": 16000,
        "window_samples": 512,
    }
    assert result["pause"]["parameters"] == {
        "profile_id": "pause-profile-v1",
        "minimum_gap_ms": 100,
        "minimum_gap_count": 4,
        "separation_ratio": 1.8,
        "maximum_iterations": 50,
        "minimum_cluster_size": 2,
        "maximum_cluster_imbalance_ratio": 3.0,
    }
    assert (
        result["vad"]["parameters_sha256"]
        == hashlib.sha256(
            json.dumps(config.raw["vad"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert (
        result["pause"]["parameters_sha256"]
        == hashlib.sha256(
            json.dumps(config.raw["pause"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert result["input_sha256s"] == [
        result["analysis_audio"]["source_sha256"],
        result["analysis_audio"]["sha256"],
        hashlib.sha256(b"Pater noster").hexdigest(),
        _MODEL_SHA256,
    ]
    cache_identity = {
        "schema_version": "1",
        "recording_id": "rec-1",
        "config_sha256": config.digest,
        "input_sha256s": result["input_sha256s"],
    }
    assert (
        result["cache_key"]
        == hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    assert [pause["kind"] for pause in result["pause"]["intervals"]] == [
        "short",
        "long",
        "short",
        "long",
    ]
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "SEGMENTED"
    events = read_jsonl(paths.alignments / "runs" / config.digest / "processing-events.jsonl")
    assert len(events) == 1
    artifact_before = result_path.read_bytes()
    events_before = events

    def unexpected(*_args: object, **_kwargs: object) -> CompletedProcess[str]:
        raise AssertionError("hash-valid segmentation should be reused")

    assert segment_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=unexpected,
    )
    assert backend.calls == 1
    assert result_path.read_bytes() == artifact_before
    assert read_jsonl(paths.alignments / "runs" / config.digest / "processing-events.jsonl") == (
        events_before
    )


def test_segment_records_ambiguous_pause_without_advancing(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    backend = _FakeVad((SpeechInterval(0, 16000),))

    assert not segment_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_transform,
    )

    result = read_jsonl(paths.alignments / "runs" / config.digest / "rec-1" / "segmentation.json")[
        0
    ]
    assert result["status"] == "failure"
    assert result["issue_code"] == "PAUSE_CLASSES_AMBIGUOUS"
    assert result["vad"]["speech_intervals"] == [{"end_sample": 16000, "start_sample": 0}]
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == ("TRANSCRIPT_CONFIRMED")


def test_segment_rejects_sung_before_audio_or_vad(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path, content_type="sung")
    backend = _FakeVad(_speech())

    with pytest.raises(ValueError, match="spoken"):
        segment_corpus(
            paths,
            config,
            backend,
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()),
        )

    assert backend.calls == 0


def test_segment_rejects_unpinned_model_config(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    config.raw["vad"]["model_version"] = "latest"
    config = _redigest(config)

    with pytest.raises(ValueError, match=r"6\.2\.1"):
        segment_corpus(
            paths,
            config,
            _FakeVad(_speech()),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_transform,
        )


def test_segment_rejects_config_mutation_without_matching_digest(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    config.raw["vad"]["threshold"] = 0.6

    with pytest.raises(ValueError, match="digest"):
        segment_corpus(
            paths,
            config,
            _FakeVad(_speech()),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_transform,
        )


def test_segment_rejects_invalid_vad_sample_boundaries(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    backend = _FakeVad((*_speech()[:-1], SpeechInterval(124800, 160001)))

    with pytest.raises(ValueError, match="sample bounds"):
        segment_corpus(
            paths,
            config,
            backend,
            ffmpeg_version="ffmpeg-test-1",
            run_command=_transform,
        )


@pytest.mark.parametrize(
    ("result", "message"),
    (
        (object(), "VadResult"),
        (VadResult("other", "6.2.1", _MODEL_SHA256, 16000, (), ()), "does not match"),
        (VadResult("silero-vad", "6.2.1", _MODEL_SHA256, 16000, (), ()), "consecutive"),
        (
            VadResult(
                "silero-vad",
                "6.2.1",
                _MODEL_SHA256,
                16000,
                (SimpleNamespace(start_sample=0),),  # type: ignore[arg-type]
                (),
            ),
            "VadFrame",
        ),
        (
            VadResult(
                "silero-vad",
                "6.2.1",
                _MODEL_SHA256,
                16000,
                (VadFrame(0, 10, 0.5),),
                (SimpleNamespace(start_sample=0, end_sample=1),),  # type: ignore[arg-type]
            ),
            "SpeechInterval",
        ),
        (
            VadResult(
                "silero-vad",
                "6.2.1",
                _MODEL_SHA256,
                16000,
                (VadFrame(0, 10, 0.5),),
                (SpeechInterval(5, 8), SpeechInterval(2, 4)),
            ),
            "ordered",
        ),
    ),
)
def test_segmentation_validates_backend_contract(result: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _validate_vad_result(  # type: ignore[arg-type]
            result,
            sample_count=10,
            window_samples=512,
            model_sha256=_MODEL_SHA256,
        )


def test_segmentation_rejects_frame_type_name_impostor() -> None:
    impostor = type("VadFrame", (), {"start_sample": 0})()
    result = VadResult(
        "silero-vad",
        "6.2.1",
        _MODEL_SHA256,
        16000,
        (impostor,),  # type: ignore[arg-type]
        (),
    )

    with pytest.raises(TypeError, match="VadFrame"):
        _validate_vad_result(
            result,
            sample_count=10,
            window_samples=512,
            model_sha256=_MODEL_SHA256,
        )


def test_segmentation_requires_immutable_vad_sequences() -> None:
    result = VadResult(
        "silero-vad",
        "6.2.1",
        _MODEL_SHA256,
        16000,
        [VadFrame(0, 10, 0.5)],  # type: ignore[arg-type]
        [],  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="tuples"):
        _validate_vad_result(
            result,
            sample_count=10,
            window_samples=512,
            model_sha256=_MODEL_SHA256,
        )


def test_segment_rejects_tampered_cached_result(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    backend = _FakeVad(_speech())
    assert segment_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_transform,
    )
    result_path = paths.alignments / "runs" / config.digest / "rec-1" / "segmentation.json"
    result = dict(read_jsonl(result_path)[0])
    result["config_sha256"] = "0" * 64
    write_jsonl_atomic(result_path, (result,))

    with pytest.raises(CorpusFailure) as error:
        segment_corpus(
            paths,
            config,
            backend,
            ffmpeg_version="ffmpeg-test-1",
            run_command=_transform,
        )

    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_segment_cache_rejects_changed_model_weight(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    backend = _FakeVad(_speech())
    assert segment_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_transform,
    )

    changed_backend = _FakeVad(_speech(), "1" * 64)
    with pytest.raises(CorpusFailure) as error:
        segment_corpus(
            paths,
            config,
            changed_backend,
            ffmpeg_version="ffmpeg-test-1",
            run_command=_transform,
        )

    assert error.value.code == "CACHE_ARTIFACT_INVALID"
    assert changed_backend.calls == 0


@pytest.mark.parametrize("target", ("derive_analysis_audio", "_validate_result_row"))
def test_segment_cached_core_programmer_type_error_propagates(
    target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _set_up(tmp_path)
    backend = _FakeVad(_speech())
    assert segment_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_transform,
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise TypeError("programmer bug")

    monkeypatch.setattr(f"latintts.corpus.cli.{target}", fail)

    with pytest.raises(TypeError, match="programmer bug"):
        segment_corpus(
            paths,
            config,
            backend,
            ffmpeg_version="ffmpeg-test-1",
            run_command=_transform,
        )


@pytest.mark.parametrize(("successful", "exit_code"), ((True, 0), (False, 1)))
def test_segment_cli_builds_pinned_backend_and_reports_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    successful: bool,
    exit_code: int,
) -> None:
    paths, config = _set_up(tmp_path)
    observed: list[tuple[CorpusPaths, CorpusConfig, str, object]] = []

    def fake_segment(
        supplied_paths: CorpusPaths,
        supplied_config: CorpusConfig,
        backend: object,
        *,
        ffmpeg_version: str,
    ) -> bool:
        observed.append((supplied_paths, supplied_config, ffmpeg_version, backend))
        return successful

    monkeypatch.setattr("latintts.corpus.cli.segment_corpus", fake_segment)
    monkeypatch.setattr("latintts.corpus.cli._ffmpeg_version", lambda: "ffmpeg-test-1")

    assert main(["--project-root", str(tmp_path), "segment"]) == exit_code

    assert len(observed) == 1
    supplied_paths, supplied_config, version, backend = observed[0]
    assert supplied_paths == paths
    assert supplied_config.digest == config.digest
    assert version == "ffmpeg-test-1"
    assert backend.__class__.__name__ == "SileroVadBackend"


def test_segment_recovers_event_durable_manifest_from_cached_result(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    backend = _FakeVad(_speech())
    assert segment_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_transform,
    )
    row = read_jsonl(paths.manifests / "recordings.jsonl")[0]
    row["state"] = "TRANSCRIPT_CONFIRMED"
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (row,))

    assert segment_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_transform,
    )

    assert backend.calls == 1
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "SEGMENTED"


def test_segment_rejects_segmented_state_without_transition_event(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    backend = _FakeVad(_speech())
    assert segment_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_transform,
    )
    (paths.alignments / "runs" / config.digest / "processing-events.jsonl").unlink()

    with pytest.raises(CorpusFailure, match="durable transition"):
        segment_corpus(
            paths,
            config,
            backend,
            ffmpeg_version="ffmpeg-test-1",
            run_command=_transform,
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("analysis", "sample_rate", 8000, "16 kHz"),
        ("vad", "extra", True, "exact fields"),
        ("pause", "extra", True, "exact fields"),
        ("pause", "profile_id", "latest", "pause-profile-v1"),
    ),
)
def test_segment_rejects_invalid_runtime_config(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    paths, config = _set_up(tmp_path)
    config.raw[section][field] = value
    config = _redigest(config)

    with pytest.raises(ValueError, match=message):
        segment_corpus(
            paths,
            config,
            _FakeVad(_speech()),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_transform,
        )


def test_ffmpeg_version_returns_exact_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "latintts.corpus.cli.subprocess.run",
        lambda *_a, **_k: CompletedProcess([], 0, "ffmpeg version 7.1\nmore\n", ""),
    )

    assert _ffmpeg_version() == "ffmpeg version 7.1"


@pytest.mark.parametrize("failure", (FileNotFoundError(), subprocess.CalledProcessError(1, [])))
def test_ffmpeg_version_maps_unavailable_tool(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("latintts.corpus.cli.subprocess.run", fail)

    with pytest.raises(CorpusFailure) as error:
        _ffmpeg_version()

    assert error.value.code == "ALIGNER_UNAVAILABLE"


def test_ffmpeg_version_rejects_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "latintts.corpus.cli.subprocess.run",
        lambda *_a, **_k: CompletedProcess([], 0, "", ""),
    )

    with pytest.raises(CorpusFailure, match="empty"):
        _ffmpeg_version()


@pytest.mark.parametrize(
    ("error", "exit_code", "prefix"),
    (
        (CorpusFailure("ALIGNER_UNAVAILABLE", "missing"), 1, "ALIGNER_UNAVAILABLE:"),
        (ValueError("bad manifest"), 2, "MANIFEST_SCHEMA_MISMATCH:"),
    ),
)
def test_segment_cli_maps_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    exit_code: int,
    prefix: str,
) -> None:
    _set_up(tmp_path)

    def fail() -> str:
        raise error

    monkeypatch.setattr("latintts.corpus.cli._ffmpeg_version", fail)

    assert main(["--project-root", str(tmp_path), "segment"]) == exit_code
    assert capsys.readouterr().err.startswith(prefix)


@pytest.mark.parametrize(
    ("section", "mutation"),
    (
        ("vad", ("delete", "neg_threshold", None)),
        ("vad", ("set", "unknown", True)),
        ("vad", ("set", "threshold", "high")),
        ("vad", ("set", "min_speech_duration_ms", "250")),
        ("vad", ("set", "use_max_poss_sil_at_max_speech", 1)),
        ("pause", ("delete", "minimum_cluster_size", None)),
        ("pause", ("set", "unknown", True)),
        ("pause", ("set", "minimum_gap_ms", "50")),
        ("pause", ("set", "maximum_cluster_imbalance_ratio", "many")),
    ),
)
def test_segment_cli_rejects_nested_config_without_output_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    section: str,
    mutation: tuple[str, str, object],
) -> None:
    raw = json.loads(
        (Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json").read_text(
            encoding="utf-8"
        )
    )
    invalid = deepcopy(raw)
    action, field, value = mutation
    if action == "delete":
        del invalid[section][field]
    else:
        invalid[section][field] = value
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(invalid), encoding="utf-8")

    assert (
        main(
            [
                "--project-root",
                str(tmp_path),
                "segment",
                "--config",
                str(config_path),
            ]
        )
        == 2
    )

    error = capsys.readouterr().err
    assert error.startswith("MANIFEST_SCHEMA_MISMATCH:")
    assert "Traceback" not in error
    assert not (tmp_path / "local-data").exists()
