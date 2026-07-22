from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import wave
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from latintts.corpus import cli
from latintts.corpus.config import CorpusConfig
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from tests.corpus.test_pairing import _audio_command, _FakeAligner
from tests.corpus.test_review_cli import _set_decisions
from tests.corpus.test_segment_cli import _FakeVad


def _write_pcm16(path: Path, sample: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(sample.to_bytes(2, "little", signed=True) * 160_000)


def _inventory_probe(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
    target = Path(command[-1])
    with wave.open(str(target), "rb") as reader:
        duration = reader.getnframes() / reader.getframerate()
        sample_rate = reader.getframerate()
        channels = reader.getnchannels()
    payload = {
        "format": {"duration": str(duration), "bit_rate": "256000"},
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "sample_rate": str(sample_rate),
                "channels": channels,
            }
        ],
    }
    return CompletedProcess(command, 0, json.dumps(payload), "")


class _FakeMedia:
    def __init__(self) -> None:
        self.flac_samples: dict[Path, int] = {}

    def run(self, command: list[str], **kwargs: object) -> CompletedProcess[str]:
        output = Path(command[-1])
        if output.suffix == ".flac":
            start = float(command[command.index("-ss") + 1])
            end = float(command[command.index("-to") + 1])
            samples = round((end - start) * 16_000)
            self.flac_samples[output] = samples
            output.write_bytes(b"fLaCsynthetic" + str(samples).encode("ascii"))
            return CompletedProcess(command, 0, "", "")
        if "pcm_s24le" in command:
            start = float(command[command.index("-ss") + 1])
            end = float(command[command.index("-to") + 1])
            with wave.open(str(output), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(3)
                writer.setframerate(16_000)
                writer.writeframes(b"\x01\x00\x00" * round((end - start) * 16_000))
            return CompletedProcess(command, 0, "", "")
        return _audio_command(command, **kwargs)

    def probe(self, command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        target = Path(command[-1])
        samples = self.flac_samples.get(target)
        if samples is None:
            samples = int(target.read_bytes().removeprefix(b"fLaCsynthetic"))
        payload = {
            "format": {"duration": str(samples / 16_000)},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "flac",
                    "sample_rate": "16000",
                    "channels": 1,
                    "duration_ts": str(samples),
                    "time_base": "1/16000",
                }
            ],
        }
        return CompletedProcess(command, 0, json.dumps(payload), "")

    @staticmethod
    def decode(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(command, 0, "", "")


def _write_intake(paths: CorpusPaths) -> None:
    with (paths.manifests / "intake.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["relative_path", "title_or_citation", "speaker_id", "rights_id", "notes"])
        for index in range(1, 4):
            writer.writerow(
                [
                    f"raw/spoken/recording-{index}.wav",
                    f"Synthetic prayer {index}",
                    "speaker-1",
                    "rights-1",
                    "synthetic integration fixture",
                ]
            )
    write_jsonl_atomic(
        paths.manifests / "rights.jsonl",
        (
            {
                "rights_id": "rights-1",
                "owner_id": "owner-1",
                "speaker_id": "speaker-1",
                "allow_local_processing": True,
                "allow_model_training": True,
                "allow_internal_evaluation": True,
                "allow_raw_release": "unknown",
                "allow_segment_release": "unknown",
                "allow_model_release": "unknown",
                "authorized_at": "2026-07-19",
                "basis": "synthetic fixture",
                "notes": "",
            },
        ),
    )


def _complete_transcript_intake(paths: CorpusPaths) -> None:
    selection = read_jsonl(paths.manifests / "pilot-selection.json")[0]
    text_root = paths.local_data / "text"
    text_root.mkdir()
    rows = []
    for recording_id in selection["recording_ids"]:
        source_path = text_root / f"{recording_id}-source.txt"
        units_path = text_root / f"{recording_id}-units.txt"
        source_path.write_text("Pater noster, qui es in caelis.", encoding="utf-8")
        units_path.write_text("Pater noster\nqui es\nin caelis\n", encoding="utf-8")
        rows.append(
            {
                "recording_id": recording_id,
                "source_candidates": [
                    {
                        "source_id": f"source-{recording_id}",
                        "source_url": "https://example.invalid/synthetic",
                        "source_version": "fixture-v1",
                        "accessed_at": "2026-07-19",
                        "source_file": paths.relative_local(source_path),
                        "selected": True,
                    }
                ],
                "spoken_units_file": paths.relative_local(units_path),
                "pronunciation_overrides_file": "",
                "confirmed": True,
                "asr_hypothesis_file": "",
            }
        )
    write_jsonl_atomic(paths.manifests / "transcript-intake.jsonl", rows)


def test_synthetic_three_recording_two_take_cli_pipeline_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path
    paths = CorpusPaths.from_project_root(project)
    paths.ensure_layout()
    config_root = project / "config" / "corpus"
    config_root.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json",
        config_root / "pilot-v1.json",
    )
    for index in range(1, 4):
        _write_pcm16(paths.raw_spoken / f"recording-{index}.wav", index)

    media = _FakeMedia()
    fake_aligner = _FakeAligner()
    real_inventory = cli.inventory_from_manifests
    real_segment = cli.segment_corpus
    real_pair = cli.pair_corpus
    real_export = cli.export_review_bundle
    real_import = cli.import_review_bundle
    real_build = cli.build_manifest_corpus
    monkeypatch.setattr(cli, "_ffmpeg_version", lambda: "ffmpeg-test-1")
    monkeypatch.setattr(
        cli,
        "inventory_from_manifests",
        lambda corpus_paths: real_inventory(
            corpus_paths,
            run_command=_inventory_probe,
            timestamp="2026-07-19T10:00:00+00:00",
        ),
    )
    monkeypatch.setattr(
        cli,
        "SileroVadBackend",
        lambda **_kwargs: _FakeVad(
            (
                cli.SpeechInterval(0, 12_800),
                cli.SpeechInterval(16_000, 28_800),
                cli.SpeechInterval(41_600, 54_400),
                cli.SpeechInterval(57_600, 70_400),
                cli.SpeechInterval(83_200, 96_000),
                cli.SpeechInterval(99_200, 118_400),
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "segment_corpus",
        lambda corpus_paths, config, backend, *, ffmpeg_version: real_segment(
            corpus_paths,
            config,
            backend,
            ffmpeg_version=ffmpeg_version,
            run_command=media.run,
        ),
    )
    monkeypatch.setattr(cli, "create_alignment_backend", lambda *_args, **_kwargs: fake_aligner)
    monkeypatch.setattr(
        cli,
        "pair_corpus",
        lambda corpus_paths, config, backend, *, ffmpeg_version: real_pair(
            corpus_paths,
            config,
            backend,
            ffmpeg_version=ffmpeg_version,
            run_command=media.run,
        ),
    )
    monkeypatch.setattr(
        cli,
        "export_review_bundle",
        lambda corpus_paths, config, *, ffmpeg_version: real_export(
            corpus_paths,
            config,
            ffmpeg_version=ffmpeg_version,
            run_command=media.run,
        ),
    )
    monkeypatch.setattr(
        cli,
        "import_review_bundle",
        lambda corpus_paths, config, *, ffmpeg_version: real_import(
            corpus_paths,
            config,
            ffmpeg_version=ffmpeg_version,
            run_command=media.run,
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_manifest_corpus",
        lambda corpus_paths, config, *, ffmpeg_version: real_build(
            corpus_paths,
            config,
            ffmpeg_version=ffmpeg_version,
            run_command=media.run,
            probe_command=media.probe,
            decode_command=media.decode,
        ),
    )

    heavy_modules = ("torch", "transformers", "silero_vad", "ctc_forced_aligner")
    imported_before = {
        name
        for name in sys.modules
        if any(name == root or name.startswith(f"{root}.") for root in heavy_modules)
    }
    arguments = ["--project-root", str(project)]
    assert cli.main([*arguments, "inventory", "--init-intake"]) == 0
    _write_intake(paths)
    assert cli.main([*arguments, "inventory"]) == 0
    recordings = read_jsonl(paths.manifests / "recordings.jsonl")
    selected_ids = [recordings[0]["recording_id"], recordings[1]["recording_id"]]
    assert (
        cli.main(
            [
                *arguments,
                "select-pilot",
                "--recording-id",
                selected_ids[0],
                "--recording-id",
                selected_ids[1],
            ]
        )
        == 0
    )
    assert cli.main([*arguments, "prepare-text", "--init"]) == 0
    _complete_transcript_intake(paths)
    assert cli.main([*arguments, "prepare-text"]) == 0
    assert cli.main([*arguments, "segment"]) == 0
    assert cli.main([*arguments, "pair"]) == 0
    pairing_calls = fake_aligner.calls
    assert cli.main([*arguments, "align"]) == 0
    assert fake_aligner.calls == pairing_calls
    assert cli.main([*arguments, "export-review"]) == 0
    config_digest = CorpusConfig.load(config_root / "pilot-v1.json").digest
    review_groups = sorted(
        group
        for recording_id in selected_ids
        for group in (paths.alignments / "runs" / config_digest / recording_id / "review").iterdir()
        if group.is_dir()
    )
    assert len(review_groups) == 6
    for group in review_groups:
        _set_decisions(group)
    assert cli.main([*arguments, "import-review"]) == 0
    assert cli.main([*arguments, "build-manifest"]) == 0
    write_jsonl_atomic(
        paths.manifests / "pilot-telemetry.json",
        (
            {
                "schema_version": "1",
                "text_preparation_seconds": 10.0,
                "review_seconds": 20.0,
                "alignment_wall_seconds": 1.48,
                "gpu_retry_count": 0,
                "peak_gpu_memory_bytes": 4_000_000_000,
                "pilot_storage_bytes": 1_000,
            },
        ),
    )
    state_paths = (
        paths.manifests / "recordings.jsonl",
        paths.manifests / "review.jsonl",
        paths.alignments / "runs" / config_digest / "processing-events.jsonl",
    )
    state_before = tuple(path.read_bytes() for path in state_paths)
    assert cli.main([*arguments, "report"]) == 0
    output_paths = (
        paths.manifests / "segments.jsonl",
        paths.manifests / "report.json",
        paths.manifests / "report.md",
    )
    first = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in output_paths)
    assert cli.main([*arguments, "report"]) == 0
    assert tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in output_paths) == first
    assert tuple(path.read_bytes() for path in state_paths) == state_before

    final_recordings = read_jsonl(paths.manifests / "recordings.jsonl")
    assert [row["state"] for row in final_recordings] == [
        "APPROVED",
        "APPROVED",
        "INVENTORIED",
    ]
    segments = read_jsonl(paths.manifests / "segments.jsonl")
    assert len(segments) == 12
    assert all(row["split"] == "unassigned" for row in segments)
    assert all(row["alignment_level"] == "word" for row in segments)
    assert all(row["phoneme_timing_status"] == "not_estimated" for row in segments)
    forbidden_phoneme_timing_fields = {
        "phoneme_spans",
        "phoneme_timestamps",
        "phoneme_start_seconds",
        "phoneme_end_seconds",
    }
    assert all(not forbidden_phoneme_timing_fields & set(row) for row in segments)
    report = json.loads((paths.manifests / "report.json").read_text(encoding="utf-8"))
    assert report["counts"] == {
        "approved_segments": 12,
        "readings": 12,
        "repetition_groups": 6,
        "text_units": 6,
    }
    assert report["metrics"]["input_duration_seconds"] == 20.0
    assert report["projection"]["full_corpus_spoken_duration_seconds"] == 30.0
    assert report["projection"]["storage_bytes"] == 1_500
    imported_after = {
        name
        for name in sys.modules
        if any(name == root or name.startswith(f"{root}.") for root in heavy_modules)
    }
    assert imported_after == imported_before
