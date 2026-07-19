import json
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

import pytest

from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.inventory import inventory_row, probe_audio, sha256_file
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import IntakeRow, RightsRecord


def _rights(*, speaker_id: str = "speaker-1") -> RightsRecord:
    return RightsRecord(
        rights_id="rights-1",
        owner_id="owner-1",
        speaker_id=speaker_id,
        allow_local_processing=True,
        allow_model_training=True,
        allow_internal_evaluation=True,
        allow_raw_release="unknown",
        allow_segment_release="unknown",
        allow_model_release="unknown",
        authorized_at="2026-07-19",
        basis="written consent",
        notes="",
    )


def _probe_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
    payload = {
        "format": {"duration": "2.5"},
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "flac",
                "sample_rate": "44100",
                "channels": 2,
            }
        ],
    }
    return CompletedProcess(command, 0, json.dumps(payload), "")


def test_probe_audio_uses_argument_array_and_parses_first_audio_stream(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "name with spaces.wav"
    audio.write_bytes(b"RIFF")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(command)
        payload = {
            "format": {"duration": "12.5", "bit_rate": "768000"},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "sample_rate": "48000",
                    "channels": 1,
                }
            ],
        }
        return CompletedProcess(command, 0, json.dumps(payload), "")

    metadata = probe_audio(audio, run_command=fake_run)
    assert metadata.duration_seconds == 12.5
    assert metadata.sample_rate == 48000
    assert calls[0][-1] == str(audio)
    assert "-of" in calls[0]


def test_sha256_file_hashes_original_bytes(tmp_path: Path) -> None:
    path = tmp_path / "audio.bin"
    path.write_bytes(b"LatinTTS")
    assert sha256_file(path) == ("405aa96624fbf5da83c15b35bdf3c2281312a87e47ef53a2fec298b751de601b")


def test_inventory_row_includes_sung_audio_as_inventoried(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    source = paths.raw_sung / "chant.flac"
    source.write_bytes(b"synthetic-chant")
    row = IntakeRow(
        relative_path="raw/sung/chant.flac",
        title_or_citation="Synthetic chant",
        speaker_id="speaker-1",
        rights_id="rights-1",
        notes="not eligible for pilot",
    )

    record = inventory_row(row, _rights(), paths, run_command=_probe_run)

    assert record.content_type == "sung"
    assert record.state is CorpusState.INVENTORIED
    assert record.relative_path == "raw/sung/chant.flac"
    assert record.recording_id == f"rec-{sha256_file(source)[:12]}"


def test_inventory_row_rejects_speaker_mismatch(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "speech.wav").write_bytes(b"synthetic-speech")
    row = IntakeRow("raw/spoken/speech.wav", "Speech", "speaker-1", "rights-1", "")

    with pytest.raises(ValueError, match="speaker mismatch"):
        inventory_row(row, _rights(speaker_id="speaker-2"), paths, run_command=_probe_run)


def test_inventory_row_rejects_unsupported_suffix_before_probe(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    source = paths.raw_spoken / "speech.txt"
    source.write_text("not audio", encoding="utf-8")
    row = IntakeRow("raw/spoken/speech.txt", "Speech", "speaker-1", "rights-1", "")

    with pytest.raises(CorpusFailure) as error:
        inventory_row(row, _rights(), paths, run_command=_probe_run)

    assert error.value.code == "INVENTORY_UNSUPPORTED_FORMAT"


def test_probe_audio_maps_malformed_ffprobe_json_to_unsupported_format(tmp_path: Path) -> None:
    audio = tmp_path / "bad.wav"
    audio.write_bytes(b"bad")

    def malformed_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(command, 0, "not-json", "")

    with pytest.raises(CorpusFailure) as error:
        probe_audio(audio, run_command=malformed_run)

    assert error.value.code == "INVENTORY_UNSUPPORTED_FORMAT"


def test_inventory_row_rejects_rights_record_for_another_rights_id(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "speech.wav").write_bytes(b"synthetic-speech")
    row = IntakeRow("raw/spoken/speech.wav", "Speech", "speaker-1", "rights-other", "")

    with pytest.raises(ValueError, match="rights mismatch"):
        inventory_row(row, _rights(), paths, run_command=_probe_run)


def test_probe_audio_maps_ffprobe_failure_without_stderr(tmp_path: Path) -> None:
    audio = tmp_path / "bad.wav"
    audio.write_bytes(b"bad")

    def failed_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise CalledProcessError(1, command)

    with pytest.raises(CorpusFailure) as error:
        probe_audio(audio, run_command=failed_run)

    assert error.value.code == "INVENTORY_UNSUPPORTED_FORMAT"
