from __future__ import annotations

import hashlib
import shutil
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from latintts.corpus.cli import align_corpus, main, pair_corpus, segment_corpus
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from latintts.corpus.transcripts import build_text_candidate, build_transcript
from latintts.corpus.vad import SpeechInterval
from tests.corpus.factories import recording
from tests.corpus.test_pairing import _audio_command, _FakeAligner
from tests.corpus.test_segment_cli import _FakeVad


def _set_up(tmp_path: Path, *, content_type: str = "spoken") -> tuple[CorpusPaths, CorpusConfig]:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    config_directory = tmp_path / "config" / "corpus"
    config_directory.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json",
        config_directory / "pilot-v1.json",
    )
    config = CorpusConfig.load(config_directory / "pilot-v1.json")
    raw_root = paths.raw_spoken if content_type == "spoken" else paths.raw_sung
    source = raw_root / "rec-1.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(
            b"\x01\x00" * 40_000
            + b"\x02\x00" * 40_000
            + b"\x03\x00" * 40_000
            + b"\x04\x00" * 40_000
        )
    record = replace(
        recording("rec-1", 10.0, content_type=content_type),  # type: ignore[arg-type]
        relative_path=f"raw/{content_type}/rec-1.wav",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
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
    candidate = build_text_candidate(
        source_id="fixture",
        source_url="https://example.invalid/fixture",
        source_version="1",
        accessed_at="2026-07-19",
        source_text="Pater noster\nqui es\nin caelis",
    )
    transcript = build_transcript(
        recording_id="rec-1",
        source_candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
        spoken_unit_lines=("Pater noster", "qui es", "in caelis"),
    )
    write_jsonl_atomic(paths.manifests / "transcripts.jsonl", (transcript.to_dict(),))
    return paths, config


def _speech() -> tuple[SpeechInterval, ...]:
    return (
        SpeechInterval(0, 12_800),
        SpeechInterval(16_000, 28_800),
        SpeechInterval(41_600, 54_400),
        SpeechInterval(57_600, 70_400),
        SpeechInterval(83_200, 96_000),
        SpeechInterval(99_200, 118_400),
    )


def _segment(paths: CorpusPaths, config: CorpusConfig) -> None:
    assert segment_corpus(
        paths,
        config,
        _FakeVad(_speech()),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )


def test_pair_corpus_advances_segmented_recording_and_align_reuses_selected_cache(
    tmp_path: Path,
) -> None:
    paths, config = _set_up(tmp_path)
    _segment(paths, config)
    backend = _FakeAligner()

    assert pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    pairing_path = paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json"
    pairing = read_jsonl(pairing_path)[0]
    assert [group["status"] for group in pairing["groups"]] == [
        "selected",
        "selected",
        "selected",
    ]
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "PAIRED"
    calls_after_pair = backend.calls
    assert pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    assert backend.calls == calls_after_pair

    assert align_corpus(paths, config, backend)
    assert backend.calls == calls_after_pair
    alignment_path = paths.alignments / "runs" / config.digest / "rec-1" / "alignment.json"
    alignment = read_jsonl(alignment_path)[0]
    assert len(alignment["takes"]) == 6
    assert all(take["alignment_result"]["coverage"] == 1.0 for take in alignment["takes"])
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "ALIGNED"
    assert align_corpus(paths, config, backend)
    assert backend.calls == calls_after_pair


def test_pair_and_align_preserve_review_issue_without_automatic_approval(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    _segment(paths, config)
    backend = _FakeAligner(mismatched=True)
    assert pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    assert align_corpus(paths, config, backend)
    alignment = read_jsonl(paths.alignments / "runs" / config.digest / "rec-1" / "alignment.json")[
        0
    ]
    assert alignment["status"] == "complete_with_issues"
    assert alignment["takes"] == []
    assert [issue["issue_code"] for issue in alignment["issues"]] == [
        "TAKE_TEXT_MISMATCH",
        "TAKE_TEXT_MISMATCH",
        "TAKE_TEXT_MISMATCH",
    ]
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "ALIGNED"


@pytest.mark.parametrize(
    ("content_type", "state"),
    (("sung", CorpusState.SEGMENTED), ("spoken", CorpusState.TRANSCRIPT_CONFIRMED)),
)
def test_pair_corpus_rejects_sung_or_wrong_state_before_writing(
    tmp_path: Path, content_type: str, state: CorpusState
) -> None:
    paths, config = _set_up(tmp_path, content_type=content_type)
    rows = read_jsonl(paths.manifests / "recordings.jsonl")
    rows[0]["state"] = state.value
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", rows)
    with pytest.raises(ValueError, match=r"spoken|SEGMENTED"):
        pair_corpus(
            paths,
            config,
            _FakeAligner(),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert not (paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json").exists()


def test_pair_corpus_rejects_tampered_strict_pairing_cache(tmp_path: Path) -> None:
    paths, config = _set_up(tmp_path)
    _segment(paths, config)
    backend = _FakeAligner()
    assert pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    path = paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json"
    raw = read_jsonl(path)[0]
    raw["unexpected"] = True
    write_jsonl_atomic(path, (raw,))
    with pytest.raises(CorpusFailure) as error:
        pair_corpus(
            paths,
            config,
            backend,
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_pair_cli_builds_backend_and_maps_pairing_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_directory = tmp_path / "config" / "corpus"
    config_directory.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json",
        config_directory / "pilot-v1.json",
    )
    sentinel = object()
    monkeypatch.setattr("latintts.corpus.cli.create_alignment_backend", lambda *_a, **_k: sentinel)
    monkeypatch.setattr("latintts.corpus.cli._ffmpeg_version", lambda: "ffmpeg-test-1")

    def fail(*args: object, **kwargs: object) -> bool:
        assert args[2] is sentinel
        raise CorpusFailure("TAKE_COUNT_MISMATCH", "review required")

    monkeypatch.setattr("latintts.corpus.cli.pair_corpus", fail)
    assert main(["--project-root", str(tmp_path), "pair"]) == 1
    assert "TAKE_COUNT_MISMATCH" in capsys.readouterr().err


def test_normal_align_cli_is_distinct_from_smoke_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_directory = tmp_path / "config" / "corpus"
    config_directory.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json",
        config_directory / "pilot-v1.json",
    )
    sentinel = object()
    monkeypatch.setattr("latintts.corpus.cli.create_alignment_backend", lambda *_a, **_k: sentinel)
    monkeypatch.setattr(
        "latintts.corpus.cli.align_corpus",
        lambda _paths, _config, backend: backend is sentinel,
    )
    assert main(["--project-root", str(tmp_path), "align"]) == 0
