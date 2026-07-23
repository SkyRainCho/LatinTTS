from __future__ import annotations

import hashlib
import json
import shutil
import wave
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from latintts.corpus import review as review_module
from latintts.corpus import store
from latintts.corpus.cli import align_corpus, main, pair_corpus
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.pairing import pairing_from_dict, pairing_to_dict
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.review import (
    ReviewedWordSpan,
    read_textgrid,
    write_textgrid,
)
from latintts.corpus.review import (
    export_review_bundle as _export_review_bundle,
)
from latintts.corpus.review import (
    import_review_bundle as _import_review_bundle,
)
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from tests.corpus.test_pair_cli import _segment, _set_up
from tests.corpus.test_pairing import _audio_command, _FakeAligner


def _review_audio_command(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
    destination = Path(command[-1])
    start = float(command[command.index("-ss") + 1])
    end = float(command[command.index("-to") + 1])
    sample_count = round((end - start) * 48_000)
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(3)
        writer.setframerate(48_000)
        writer.writeframes(b"\x01\x00\x00" * sample_count)
    return CompletedProcess(command, 0, "", "")


def export_review_bundle(paths: CorpusPaths, config: CorpusConfig) -> tuple[Path, ...]:
    return _export_review_bundle(
        paths,
        config,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_review_audio_command,
    )


def import_review_bundle(paths: CorpusPaths, config: CorpusConfig) -> bool:
    return _import_review_bundle(
        paths,
        config,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_review_audio_command,
    )


def _aligned_project(tmp_path: Path) -> tuple[CorpusPaths, CorpusConfig]:
    paths, config = _set_up(tmp_path)
    recording_path = paths.manifests / "recordings.jsonl"
    row = read_jsonl(recording_path)[0]
    row["title_or_citation"] = '<Pater & "Noster">'
    write_jsonl_atomic(recording_path, (row,))
    _segment(paths, config)
    backend = _FakeAligner()
    assert pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    assert align_corpus(paths, config, backend)
    return paths, config


def _review_required_project(
    tmp_path: Path,
) -> tuple[CorpusPaths, CorpusConfig, _FakeAligner]:
    paths, config = _set_up(tmp_path)
    config_path = tmp_path / "config" / "corpus" / "pilot-v1.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["pairing"]["minimum_duration_ratio"] = 1.0
    raw_config["pairing"]["maximum_duration_ratio"] = 1.0
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    config = CorpusConfig.load(config_path)
    _segment(paths, config)
    backend = _FakeAligner()
    assert not pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    return paths, config, backend


def _confirm_pairing_correction(correction: Path) -> None:
    automatic = json.loads((correction / "automatic.json").read_text(encoding="utf-8"))
    decision_path = correction / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    saved_splits = {
        unit["unit_id"]: unit["candidates"][0]["split_sample"] for unit in automatic["review_units"]
    }
    for row in decision["pairing_selections"]:
        row.update(
            split_sample=saved_splits[row["unit_id"]],
            reason="listened to both preserved candidates",
            reviewer="owner",
            reviewed_at="2026-07-19T12:00:00+08:00",
        )
    decision_path.write_text(json.dumps(decision), encoding="utf-8")


def _corrected_pairing_project(
    tmp_path: Path,
) -> tuple[CorpusPaths, CorpusConfig, _FakeAligner, Path]:
    paths, config, backend = _review_required_project(tmp_path)
    correction = export_review_bundle(paths, config)[0]
    _confirm_pairing_correction(correction)
    assert import_review_bundle(paths, config)
    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    return paths, config, backend, run_directory


def _bind_pairing_transition_to_prefix(processing_path: Path, prefix: bytes) -> None:
    rows = list(read_jsonl(processing_path))
    paired = next(row for row in rows if row["target_state"] == "PAIRED")
    paired["input_sha256s"][3] = hashlib.sha256(prefix).hexdigest()
    identity = {
        "recording_id": paired["recording_id"],
        "previous_state": paired["previous_state"],
        "target_state": paired["target_state"],
        "inputs": paired["input_sha256s"],
        "config": paired["config_sha256"],
        "tools": paired["tool_versions"],
        "result": paired["result"],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    paired["event_id"] = "state-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    write_jsonl_atomic(processing_path, rows)


def _place_correction_after_bound_prefix(review_path: Path, processing_path: Path) -> None:
    correction = read_jsonl(review_path)[0]
    unrelated = review_module._new_review_event(
        "review:unrelated-prefix",
        "review_decision",
        "unreviewed",
        "approved",
        {
            "reason": "earlier independent review",
            "reviewer": "owner",
            "reviewed_at": "2026-07-19T11:00:00+08:00",
        },
    )
    write_jsonl_atomic(review_path, (unrelated.to_dict(), correction))
    prefix = review_path.read_bytes().splitlines(keepends=True)[0]
    _bind_pairing_transition_to_prefix(processing_path, prefix)


def _file_snapshot(paths: tuple[Path, ...]) -> tuple[tuple[bool, bytes | None], ...]:
    return tuple((path.exists(), path.read_bytes() if path.exists() else None) for path in paths)


def _two_recording_aligned_project(tmp_path: Path) -> tuple[CorpusPaths, CorpusConfig]:
    paths, config = _set_up(tmp_path)
    source_one = paths.raw_spoken / "rec-1.wav"
    source_two = paths.raw_spoken / "rec-2.wav"
    shutil.copyfile(source_one, source_two)
    recordings = list(read_jsonl(paths.manifests / "recordings.jsonl"))
    second_recording = deepcopy(recordings[0])
    second_recording.update(
        recording_id="rec-2",
        relative_path="raw/spoken/rec-2.wav",
        title_or_citation="rec-2",
        sha256=hashlib.sha256(source_two.read_bytes()).hexdigest(),
    )
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (*recordings, second_recording))
    transcripts = list(read_jsonl(paths.manifests / "transcripts.jsonl"))
    second_transcript = deepcopy(transcripts[0])
    second_transcript["recording_id"] = "rec-2"
    for unit in second_transcript["spoken_units"]:
        unit["unit_id"] = unit["unit_id"].replace("rec-1-", "rec-2-")
    write_jsonl_atomic(paths.manifests / "transcripts.jsonl", (*transcripts, second_transcript))
    write_jsonl_atomic(
        paths.manifests / "pilot-selection.json",
        (
            {
                "schema_version": "1",
                "strategy": "explicit-v1",
                "recording_ids": ["rec-1", "rec-2"],
                "inventory_hashes": [recordings[0]["sha256"], second_recording["sha256"]],
            },
        ),
    )
    _segment(paths, config)
    backend = _FakeAligner()
    assert pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    assert align_corpus(paths, config, backend)
    return paths, config


def _two_recording_review_required_project(tmp_path: Path) -> tuple[CorpusPaths, CorpusConfig]:
    paths, config = _set_up(tmp_path)
    config_path = tmp_path / "config" / "corpus" / "pilot-v1.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["pairing"]["minimum_duration_ratio"] = 1.0
    raw_config["pairing"]["maximum_duration_ratio"] = 1.0
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    config = CorpusConfig.load(config_path)
    source_one = paths.raw_spoken / "rec-1.wav"
    source_two = paths.raw_spoken / "rec-2.wav"
    shutil.copyfile(source_one, source_two)
    recordings = list(read_jsonl(paths.manifests / "recordings.jsonl"))
    second_recording = deepcopy(recordings[0])
    second_recording.update(
        recording_id="rec-2",
        relative_path="raw/spoken/rec-2.wav",
        title_or_citation="rec-2",
        sha256=hashlib.sha256(source_two.read_bytes()).hexdigest(),
    )
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (*recordings, second_recording))
    transcripts = list(read_jsonl(paths.manifests / "transcripts.jsonl"))
    second_transcript = deepcopy(transcripts[0])
    second_transcript["recording_id"] = "rec-2"
    for unit in second_transcript["spoken_units"]:
        unit["unit_id"] = unit["unit_id"].replace("rec-1-", "rec-2-")
    write_jsonl_atomic(paths.manifests / "transcripts.jsonl", (*transcripts, second_transcript))
    write_jsonl_atomic(
        paths.manifests / "pilot-selection.json",
        (
            {
                "schema_version": "1",
                "strategy": "explicit-v1",
                "recording_ids": ["rec-1", "rec-2"],
                "inventory_hashes": [recordings[0]["sha256"], second_recording["sha256"]],
            },
        ),
    )
    _segment(paths, config)
    assert not pair_corpus(
        paths,
        config,
        _FakeAligner(),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    return paths, config


def test_export_review_bundle_writes_strict_human_artifacts_from_source_audio(
    tmp_path: Path,
) -> None:
    paths, config = _aligned_project(tmp_path)

    group_directories = export_review_bundle(paths, config)

    assert len(group_directories) == 3
    group = group_directories[0]
    assert group.parent == paths.alignments / "runs" / config.digest / "rec-1" / "review"
    assert {item.name for item in group.iterdir()} == {
        "index.html",
        "automatic.json",
        "decision.json",
        "take-1.wav",
        "take-1.TextGrid",
        "take-2.wav",
        "take-2.TextGrid",
    }
    with wave.open(str(group / "take-1.wav"), "rb") as reader:
        assert reader.getframerate() == 48_000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 3
    automatic = json.loads((group / "automatic.json").read_text(encoding="utf-8"))
    assert set(automatic) == {
        "schema_version",
        "recording_id",
        "repetition_group_id",
        "unit_id",
        "source_audio",
        "artifact_binding",
        "text_layers",
        "takes",
    }
    assert (
        automatic["source_audio"]["sha256"]
        == hashlib.sha256((paths.raw_spoken / "rec-1.wav").read_bytes()).hexdigest()
    )
    assert len(automatic["takes"]) == 2
    transcript = read_jsonl(paths.manifests / "transcripts.jsonl")[0]
    assert set(automatic["text_layers"]) == {
        "title_or_citation",
        "source_text",
        "spoken_text",
        "normalized_text",
        "unit_spoken_text",
        "alignment_texts",
    }
    assert automatic["text_layers"]["source_text"] == transcript["source_text"]
    assert automatic["text_layers"]["spoken_text"] == transcript["spoken_text"]
    assert automatic["text_layers"]["normalized_text"] == transcript["normalized_text"]
    assert all(take["audio_sha256"] for take in automatic["takes"])
    assert all(take["audio_provenance"]["mode"] == "review" for take in automatic["takes"])
    assert all(
        take["take_provenance"]["candidate_audio_relative_path"] for take in automatic["takes"]
    )
    assert automatic["artifact_binding"]["alignment_artifact_sha256"]
    assert all(take["textgrid_sha256"] for take in automatic["takes"])
    decision = json.loads((group / "decision.json").read_text(encoding="utf-8"))
    assert set(decision) == {
        "schema_version",
        "recording_id",
        "repetition_group_id",
        "decisions",
    }
    assert [item["decision"] for item in decision["decisions"]] == [
        "unreviewed",
        "unreviewed",
    ]
    assert all(
        set(item) == {"entity_id", "decision", "reason", "reviewer", "reviewed_at"}
        for item in decision["decisions"]
    )
    assert all(
        item["reason"] == item["reviewer"] == item["reviewed_at"] == ""
        for item in decision["decisions"]
    )
    html = (group / "index.html").read_text(encoding="utf-8")
    assert '<Pater & "Noster">' not in html
    assert "&lt;Pater &amp; &quot;Noster&quot;&gt;" in html


@pytest.mark.parametrize(
    "case",
    (
        "pairing-row-count",
        "alignment-row-count",
        "alignment-identity",
        "alignment-takes-type",
        "alignment-take-type",
        "alignment-duplicate-take",
        "pairing-review-outcome",
        "alignment-missing-take",
        "alignment-take-mismatch",
        "candidate-missing",
    ),
)
def test_review_export_rejects_strict_pairing_alignment_artifact_drift(
    tmp_path: Path, case: str
) -> None:
    paths, config = _aligned_project(tmp_path)
    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    pairing_path = run_directory / "pairing.json"
    alignment_path = run_directory / "alignment.json"
    if case == "pairing-row-count":
        write_jsonl_atomic(pairing_path, ())
    elif case == "alignment-row-count":
        write_jsonl_atomic(alignment_path, ())
    elif case == "candidate-missing":
        pairing = pairing_from_dict(read_jsonl(pairing_path)[0])
        take = next(outcome.group.takes[0] for outcome in pairing.groups if outcome.group)
        paths.resolve_local(take.audio_relative_path).unlink()
    elif case == "pairing-review-outcome":
        pairing = pairing_from_dict(read_jsonl(pairing_path)[0])
        changed = replace(pairing.groups[0], status="review", group=None)
        pairing = replace(
            pairing,
            groups=(changed, *pairing.groups[1:]),
            integrity_sha256="",
        )
        write_jsonl_atomic(pairing_path, (pairing_to_dict(pairing),))
        alignment = read_jsonl(alignment_path)[0]
        alignment["pairing_artifact_sha256"] = hashlib.sha256(pairing_path.read_bytes()).hexdigest()
        write_jsonl_atomic(alignment_path, (alignment,))
    else:
        alignment = read_jsonl(alignment_path)[0]
        if case == "alignment-identity":
            alignment["status"] = "failed"
        elif case == "alignment-takes-type":
            alignment["takes"] = {}
        elif case == "alignment-take-type":
            alignment["takes"][0] = None
        elif case == "alignment-duplicate-take":
            alignment["takes"].append(deepcopy(alignment["takes"][0]))
        elif case == "alignment-missing-take":
            alignment["takes"].pop()
        else:
            alignment["takes"][0]["unit_id"] = "mismatched-unit"
        write_jsonl_atomic(alignment_path, (alignment,))

    with pytest.raises(ValueError, match=r"pairing|alignment|candidate"):
        export_review_bundle(paths, config)


def test_review_wav_duration_uses_actual_44100hz_frame_count_at_rounding_edge(
    tmp_path: Path,
) -> None:
    paths, _ = _set_up(tmp_path)
    recording = review_module._decode_recording(read_jsonl(paths.manifests / "recordings.jsonl")[0])
    recording = replace(
        recording,
        metadata=replace(recording.metadata, sample_rate=44_100),
    )
    destination = tmp_path / "rounding-edge.wav"

    def write_rounding_edge(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        output = Path(command[-1])
        start = float(command[command.index("-ss") + 1])
        end = float(command[command.index("-to") + 1])
        sample_count = round((end - start) * 44_100)
        with wave.open(str(output), "wb") as writer:
            writer.setnchannels(recording.metadata.channels)
            writer.setsampwidth(3)
            writer.setframerate(44_100)
            writer.writeframes(b"\x01\x00\x00" * sample_count)
        return CompletedProcess(command, 0, "", "")

    start_seconds = 1 / 16_000
    end_seconds = 3 / 16_000
    review_module._export_review_audio(
        recording,
        paths,
        destination,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        ffmpeg_version="ffmpeg-test-1",
        run_command=write_rounding_edge,
    )

    actual_duration = review_module._review_wav_duration(destination, recording)
    endpoint_rounding_duration = (
        round(end_seconds * 44_100) - round(start_seconds * 44_100)
    ) / 44_100
    assert actual_duration == 6 / 44_100
    assert endpoint_rounding_duration == 5 / 44_100


@pytest.mark.parametrize("case", ("format", "invalid-container"))
def test_review_wav_duration_rejects_invalid_trusted_media(tmp_path: Path, case: str) -> None:
    paths, _ = _set_up(tmp_path)
    recording = review_module._decode_recording(read_jsonl(paths.manifests / "recordings.jsonl")[0])
    path = tmp_path / "invalid.wav"
    if case == "format":
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(recording.metadata.channels)
            writer.setsampwidth(2)
            writer.setframerate(recording.metadata.sample_rate)
            writer.writeframes(b"\0\0")
    else:
        path.write_bytes(b"not a wave file")

    with pytest.raises(ValueError, match="WAV"):
        review_module._review_wav_duration(path, recording)


def test_review_required_pairing_can_only_materialize_a_saved_human_selection(
    tmp_path: Path,
) -> None:
    paths, config, backend = _review_required_project(tmp_path)

    correction = export_review_bundle(paths, config)[0]
    automatic = json.loads((correction / "automatic.json").read_text(encoding="utf-8"))
    decision_path = correction / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["schema_version"] == "2"
    assert decision["pairing_selections"][0]["split_sample"] is None
    assert export_review_bundle(paths, config) == (correction,)
    assert not import_review_bundle(paths, config)
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "SEGMENTED"

    saved_splits = {
        unit["unit_id"]: unit["candidates"][0]["split_sample"] for unit in automatic["review_units"]
    }
    for row in decision["pairing_selections"]:
        row.update(
            split_sample=saved_splits[row["unit_id"]],
            reason="listened to both preserved candidates",
            reviewer="owner",
            reviewed_at="2026-07-19T12:00:00+08:00",
        )
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    assert import_review_bundle(paths, config)
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "PAIRED"
    event = read_jsonl(paths.manifests / "review.jsonl")[-1]
    assert event["field"] == "pairing_selected_split"
    assert event["after"] in saved_splits.values()
    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    assert (run_directory / "pairing-automatic.json").is_file()
    corrected = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
    assert all(outcome.status == "selected" for outcome in corrected.groups)

    assert pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("repeated reviewed pair must use validated caches")
        ),
    )

    assert align_corpus(paths, config, backend)
    groups = export_review_bundle(paths, config)
    assert len(groups) == 3
    assert all(group.name != "pairing-correction" for group in groups)


def test_pairing_correction_retry_recovers_after_corrected_pairing_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config, _ = _review_required_project(tmp_path)
    correction = export_review_bundle(paths, config)[0]
    _confirm_pairing_correction(correction)
    real_persist = review_module.persist_recording_transition
    monkeypatch.setattr(
        review_module,
        "persist_recording_transition",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("after pairing replace")),
    )

    with pytest.raises(RuntimeError, match="after pairing replace"):
        import_review_bundle(paths, config)

    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    assert all(
        outcome.status == "selected"
        for outcome in pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0]).groups
    )
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "SEGMENTED"
    monkeypatch.setattr(review_module, "persist_recording_transition", real_persist)

    assert import_review_bundle(paths, config)
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "PAIRED"
    assert (
        len(
            [
                row
                for row in read_jsonl(run_directory.parent / "processing-events.jsonl")
                if row["target_state"] == "PAIRED"
            ]
        )
        == 1
    )


def test_pairing_correction_retry_binds_new_transition_to_durable_review_whitespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config, _ = _review_required_project(tmp_path)
    correction = export_review_bundle(paths, config)[0]
    _confirm_pairing_correction(correction)
    real_persist = review_module.persist_recording_transition
    monkeypatch.setattr(
        review_module,
        "persist_recording_transition",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("before transition persist")),
    )

    with pytest.raises(RuntimeError, match="before transition persist"):
        import_review_bundle(paths, config)
    monkeypatch.setattr(review_module, "persist_recording_transition", real_persist)

    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    processing_path = run_directory.parent / "processing-events.jsonl"
    assert not any(row["target_state"] == "PAIRED" for row in read_jsonl(processing_path))
    review_path = paths.manifests / "review.jsonl"
    review_rows = read_jsonl(review_path)
    review_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(", ", ": ")) + "\n"
            for row in review_rows
        ),
        encoding="utf-8",
    )
    durable_review_bytes = review_path.read_bytes()
    durable_review_sha256 = hashlib.sha256(durable_review_bytes).hexdigest()

    assert import_review_bundle(paths, config)

    assert review_path.read_bytes() == durable_review_bytes
    paired_events = [row for row in read_jsonl(processing_path) if row["target_state"] == "PAIRED"]
    assert len(paired_events) == 1
    assert paired_events[0]["input_sha256s"][3] == durable_review_sha256
    automatic = pairing_from_dict(read_jsonl(run_directory / "pairing-automatic.json")[0])
    corrected = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
    assert review_module.validate_pairing_review_snapshot(
        review_path,
        durable_review_sha256,
        automatic,
        corrected,
    )


def test_later_pairing_correction_preserves_whitespace_bound_review_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _two_recording_review_required_project(tmp_path)
    corrections = {
        json.loads((directory / "decision.json").read_text(encoding="utf-8"))["recording_id"]: (
            directory
        )
        for directory in export_review_bundle(paths, config)
    }
    _confirm_pairing_correction(corrections["rec-1"])
    real_persist = review_module.persist_recording_transition
    monkeypatch.setattr(
        review_module,
        "persist_recording_transition",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("before first transition")),
    )
    with pytest.raises(RuntimeError, match="before first transition"):
        import_review_bundle(paths, config)
    monkeypatch.setattr(review_module, "persist_recording_transition", real_persist)

    review_path = paths.manifests / "review.jsonl"
    review_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(", ", ": ")) + "\n"
            for row in read_jsonl(review_path)
        ),
        encoding="utf-8",
    )
    first_prefix = review_path.read_bytes()
    assert not import_review_bundle(paths, config)

    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    first_event = next(
        row
        for row in read_jsonl(processing_path)
        if row["recording_id"] == "rec-1" and row["target_state"] == "PAIRED"
    )
    assert first_event["input_sha256s"][3] == hashlib.sha256(first_prefix).hexdigest()

    _confirm_pairing_correction(corrections["rec-2"])
    assert import_review_bundle(paths, config)

    final_journal = review_path.read_bytes()
    assert final_journal.startswith(first_prefix)
    paired_events = {
        row["recording_id"]: row
        for row in read_jsonl(processing_path)
        if row["target_state"] == "PAIRED"
    }
    for recording_id in ("rec-1", "rec-2"):
        run_directory = paths.alignments / "runs" / config.digest / recording_id
        automatic = pairing_from_dict(read_jsonl(run_directory / "pairing-automatic.json")[0])
        corrected = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
        assert review_module.validate_pairing_review_snapshot(
            review_path,
            paired_events[recording_id]["input_sha256s"][3],
            automatic,
            corrected,
        )


def test_pairing_correction_retry_recovers_after_event_before_recordings_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config, _ = _review_required_project(tmp_path)
    correction = export_review_bundle(paths, config)[0]
    _confirm_pairing_correction(correction)
    recordings_path = paths.manifests / "recordings.jsonl"
    real_replace = store.os.replace

    def fail_recordings_replace(source: Path, destination: Path) -> None:
        if Path(destination) == recordings_path:
            raise OSError("recordings replace failed")
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", fail_recordings_replace)
    with pytest.raises(store.RecordingTransitionPersistenceError, match="recovery required"):
        import_review_bundle(paths, config)
    monkeypatch.setattr(store.os, "replace", real_replace)

    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    assert read_jsonl(recordings_path)[0]["state"] == "SEGMENTED"
    processing_path = run_directory.parent / "processing-events.jsonl"
    paired_events = [row for row in read_jsonl(processing_path) if row["target_state"] == "PAIRED"]
    assert len(paired_events) == 1
    bound_review_sha256 = paired_events[0]["input_sha256s"][3]

    review_path = paths.manifests / "review.jsonl"
    history = list(read_jsonl(review_path))
    appended = review_module._new_review_event(
        "review:unrelated",
        "review_decision",
        "unreviewed",
        "approved",
        {
            "reason": "later independent review",
            "reviewer": "owner",
            "reviewed_at": "2026-07-19T13:00:00+08:00",
        },
    )
    write_jsonl_atomic(review_path, (*history, appended.to_dict()))

    assert import_review_bundle(paths, config)
    assert read_jsonl(recordings_path)[0]["state"] == "PAIRED"
    recovered = [row for row in read_jsonl(processing_path) if row["target_state"] == "PAIRED"]
    assert len(recovered) == 1
    assert recovered[0]["input_sha256s"][3] == bound_review_sha256


def test_pairing_correction_retry_recovers_after_journal_before_automatic_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config, _ = _review_required_project(tmp_path)
    correction = export_review_bundle(paths, config)[0]
    _confirm_pairing_correction(correction)
    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    automatic_path = run_directory / "pairing-automatic.json"
    real_write_bytes = review_module._write_bytes_atomic

    def fail_automatic_write(path: Path, content: bytes) -> None:
        if Path(path) == automatic_path:
            raise RuntimeError("after review journal")
        real_write_bytes(path, content)

    monkeypatch.setattr(
        review_module,
        "_write_bytes_atomic",
        fail_automatic_write,
    )

    with pytest.raises(RuntimeError, match="after review journal"):
        import_review_bundle(paths, config)

    assert (paths.manifests / "review.jsonl").is_file()
    assert not automatic_path.exists()
    assert any(
        outcome.status == "review"
        for outcome in pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0]).groups
    )
    monkeypatch.setattr(review_module, "_write_bytes_atomic", real_write_bytes)

    assert import_review_bundle(paths, config)
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "PAIRED"


def test_pairing_correction_retry_recovers_after_automatic_before_corrected_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config, _ = _review_required_project(tmp_path)
    correction = export_review_bundle(paths, config)[0]
    _confirm_pairing_correction(correction)
    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    pairing_path = run_directory / "pairing.json"
    real_write_jsonl = review_module.write_jsonl_atomic

    def fail_pairing_write(path: Path, rows: object) -> None:
        if Path(path) == pairing_path:
            raise RuntimeError("after automatic pairing")
        real_write_jsonl(path, rows)  # type: ignore[arg-type]

    monkeypatch.setattr(review_module, "write_jsonl_atomic", fail_pairing_write)
    with pytest.raises(RuntimeError, match="after automatic pairing"):
        import_review_bundle(paths, config)
    monkeypatch.setattr(review_module, "write_jsonl_atomic", real_write_jsonl)

    assert (run_directory / "pairing-automatic.json").is_file()
    assert any(
        outcome.status == "review"
        for outcome in pairing_from_dict(read_jsonl(pairing_path)[0]).groups
    )
    assert import_review_bundle(paths, config)
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "PAIRED"


def test_align_rejects_non_deterministic_pairing_correction_event_id(tmp_path: Path) -> None:
    paths, config, backend, _ = _corrected_pairing_project(tmp_path)
    review_path = paths.manifests / "review.jsonl"
    rows = list(read_jsonl(review_path))
    rows[0]["review_event_id"] = "review-" + "0" * 64
    write_jsonl_atomic(review_path, rows)

    with pytest.raises(CorpusFailure, match="selected alignment cache"):
        align_corpus(paths, config, backend)


def test_align_rejects_changed_review_snapshot_prefix_even_when_json_is_semantically_same(
    tmp_path: Path,
) -> None:
    paths, config, backend, _ = _corrected_pairing_project(tmp_path)
    review_path = paths.manifests / "review.jsonl"
    lines = review_path.read_text(encoding="utf-8").splitlines()
    review_path.write_text(" \n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CorpusFailure, match="selected alignment cache"):
        align_corpus(paths, config, backend)


def test_align_accepts_valid_review_events_appended_after_bound_pairing_snapshot(
    tmp_path: Path,
) -> None:
    paths, config, backend, _ = _corrected_pairing_project(tmp_path)
    review_path = paths.manifests / "review.jsonl"
    rows = list(read_jsonl(review_path))
    suffix = review_module._new_review_event(
        "review-suffix",
        "review_decision",
        "unreviewed",
        "approved",
        {
            "reason": "later take review",
            "reviewer": "owner",
            "reviewed_at": "2026-07-19T13:00:00+08:00",
        },
    )
    write_jsonl_atomic(review_path, (*rows, suffix.to_dict()))

    assert align_corpus(paths, config, backend)


def test_align_rejects_non_deterministic_review_event_after_bound_correction_prefix(
    tmp_path: Path,
) -> None:
    paths, config, backend, run_directory = _corrected_pairing_project(tmp_path)
    review_path = paths.manifests / "review.jsonl"
    rows = list(read_jsonl(review_path))
    suffix = review_module._new_review_event(
        "review:later-suffix",
        "review_decision",
        "unreviewed",
        "approved",
        {
            "reason": "later take review",
            "reviewer": "owner",
            "reviewed_at": "2026-07-19T13:00:00+08:00",
        },
    ).to_dict()
    suffix["review_event_id"] = "review-" + "0" * 64
    write_jsonl_atomic(review_path, (*rows, suffix))
    observed_paths = (
        paths.manifests / "recordings.jsonl",
        review_path,
        run_directory.parent / "processing-events.jsonl",
        run_directory / "pairing-automatic.json",
        run_directory / "pairing.json",
        run_directory / "alignment.json",
    )
    before = _file_snapshot(observed_paths)

    with pytest.raises(CorpusFailure, match="selected alignment cache"):
        align_corpus(paths, config, backend)

    assert _file_snapshot(observed_paths) == before


def test_align_rejects_processing_event_snapshot_before_pairing_correction(
    tmp_path: Path,
) -> None:
    paths, config, backend, run_directory = _corrected_pairing_project(tmp_path)
    review_path = paths.manifests / "review.jsonl"
    processing_path = run_directory.parent / "processing-events.jsonl"
    _place_correction_after_bound_prefix(review_path, processing_path)
    observed_paths = (
        paths.manifests / "recordings.jsonl",
        review_path,
        processing_path,
        run_directory / "pairing-automatic.json",
        run_directory / "pairing.json",
        run_directory / "alignment.json",
    )
    before = _file_snapshot(observed_paths)

    with pytest.raises(CorpusFailure, match="selected alignment cache"):
        align_corpus(paths, config, backend)

    assert _file_snapshot(observed_paths) == before


def test_pairing_correction_recovery_rejects_snapshot_before_correction_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config, _ = _review_required_project(tmp_path)
    correction = export_review_bundle(paths, config)[0]
    _confirm_pairing_correction(correction)
    recordings_path = paths.manifests / "recordings.jsonl"
    real_replace = store.os.replace

    def fail_recordings_replace(source: Path, destination: Path) -> None:
        if Path(destination) == recordings_path:
            raise OSError("recordings replace failed")
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", fail_recordings_replace)
    with pytest.raises(store.RecordingTransitionPersistenceError, match="recovery required"):
        import_review_bundle(paths, config)
    monkeypatch.setattr(store.os, "replace", real_replace)

    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    review_path = paths.manifests / "review.jsonl"
    processing_path = run_directory.parent / "processing-events.jsonl"
    _place_correction_after_bound_prefix(review_path, processing_path)
    observed_paths = (
        recordings_path,
        review_path,
        processing_path,
        run_directory / "pairing-automatic.json",
        run_directory / "pairing.json",
    )
    before = _file_snapshot(observed_paths)

    with pytest.raises(ValueError, match=r"snapshot|correction"):
        import_review_bundle(paths, config)

    assert _file_snapshot(observed_paths) == before


def test_pairing_correction_preflight_rejects_old_event_before_appending_missing_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config, _ = _review_required_project(tmp_path)
    correction = export_review_bundle(paths, config)[0]
    _confirm_pairing_correction(correction)
    recordings_path = paths.manifests / "recordings.jsonl"
    real_replace = store.os.replace

    def fail_recordings_replace(source: Path, destination: Path) -> None:
        if Path(destination) == recordings_path:
            raise OSError("recordings replace failed")
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", fail_recordings_replace)
    with pytest.raises(store.RecordingTransitionPersistenceError, match="recovery required"):
        import_review_bundle(paths, config)
    monkeypatch.setattr(store.os, "replace", real_replace)

    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    review_path = paths.manifests / "review.jsonl"
    processing_path = run_directory.parent / "processing-events.jsonl"
    unrelated = review_module._new_review_event(
        "review:old-prefix-only",
        "review_decision",
        "unreviewed",
        "approved",
        {
            "reason": "earlier independent review",
            "reviewer": "owner",
            "reviewed_at": "2026-07-19T11:00:00+08:00",
        },
    )
    write_jsonl_atomic(review_path, (unrelated.to_dict(),))
    _bind_pairing_transition_to_prefix(processing_path, review_path.read_bytes())
    assert all(row["field"] != "pairing_selected_split" for row in read_jsonl(review_path))
    observed_paths = (
        review_path,
        run_directory / "pairing-automatic.json",
        run_directory / "pairing.json",
        processing_path,
        recordings_path,
    )
    before = _file_snapshot(observed_paths)

    with pytest.raises(ValueError, match=r"snapshot|correction"):
        import_review_bundle(paths, config)

    assert _file_snapshot(observed_paths) == before


@pytest.mark.parametrize("operation", ("pair", "align"))
def test_repeated_pair_and_align_reject_reparse_marked_automatic_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    paths, config, backend, run_directory = _corrected_pairing_project(tmp_path)
    automatic_path = (run_directory / "pairing-automatic.json").absolute()
    real_lstat = review_module.os.lstat

    def reparse_lstat(path: object) -> object:
        metadata = real_lstat(path)  # type: ignore[arg-type]
        if Path(path).absolute() == automatic_path:  # type: ignore[arg-type]
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=(
                    getattr(metadata, "st_file_attributes", 0)
                    | review_module.stat.FILE_ATTRIBUTE_REPARSE_POINT
                ),
            )
        return metadata

    monkeypatch.setattr(review_module.os, "lstat", reparse_lstat)

    with pytest.raises(CorpusFailure, match=r"CACHE_ARTIFACT_INVALID|cache is invalid"):
        if operation == "pair":
            pair_corpus(
                paths,
                config,
                backend,
                ffmpeg_version="ffmpeg-test-1",
                run_command=_audio_command,
            )
        else:
            align_corpus(paths, config, backend)


@pytest.mark.parametrize(
    "case",
    ("orphan-entity", "duplicate-entity", "unsaved-split", "automatic-mismatch"),
)
def test_ordinary_review_import_rejects_invalid_pairing_correction_history(
    tmp_path: Path, case: str
) -> None:
    paths, config, backend, run_directory = _corrected_pairing_project(tmp_path)
    assert align_corpus(paths, config, backend)
    export_review_bundle(paths, config)
    review_path = paths.manifests / "review.jsonl"
    rows = list(read_jsonl(review_path))
    original = rows[0]
    decision = {
        "reason": original["reason"],
        "reviewer": original["reviewer"],
        "reviewed_at": "2026-07-19T13:00:00+08:00",
    }
    if case == "orphan-entity":
        event = review_module._new_review_event(
            "pairing:6:orphan:4:unit",
            "pairing_selected_split",
            None,
            original["after"],
            decision,
        )
        rows.append(event.to_dict())
    elif case == "duplicate-entity":
        event = review_module._new_review_event(
            original["entity_id"],
            "pairing_selected_split",
            None,
            original["after"],
            decision,
        )
        rows.append(event.to_dict())
    elif case == "unsaved-split":
        event = review_module._new_review_event(
            original["entity_id"],
            "pairing_selected_split",
            None,
            original["after"] + 1,
            decision,
        )
        rows[0] = event.to_dict()
    else:
        shutil.copyfile(run_directory / "pairing.json", run_directory / "pairing-automatic.json")
    write_jsonl_atomic(review_path, rows)

    with pytest.raises(ValueError, match="pairing correction"):
        import_review_bundle(paths, config)


@pytest.mark.parametrize(
    "case",
    (
        "original-type",
        "events-type",
        "identity",
        "duplicate-unit",
        "automatic-selected-changed",
        "corrected-evidence-invalid",
        "orphan-entity",
    ),
)
def test_pairing_correction_event_validator_rejects_structural_drift(
    tmp_path: Path, case: str
) -> None:
    paths, _, _, run_directory = _corrected_pairing_project(tmp_path)
    original = pairing_from_dict(read_jsonl(run_directory / "pairing-automatic.json")[0])
    corrected = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
    events = tuple(
        review_module.ReviewEvent.from_dict(row)
        for row in read_jsonl(paths.manifests / "review.jsonl")
    )
    if case == "original-type":
        original = None  # type: ignore[assignment]
    elif case == "events-type":
        events = list(events)  # type: ignore[assignment]
    elif case == "identity":
        corrected = replace(corrected, recording_id="different-recording", integrity_sha256="")
    elif case == "duplicate-unit":
        corrected = replace(
            corrected,
            groups=(*corrected.groups, corrected.groups[0]),
            integrity_sha256="",
        )
    elif case == "automatic-selected-changed":
        index = next(
            index for index, outcome in enumerate(original.groups) if outcome.status == "selected"
        )
        changed = replace(corrected.groups[index], status="review", group=None)
        corrected = replace(
            corrected,
            groups=(*corrected.groups[:index], changed, *corrected.groups[index + 1 :]),
            integrity_sha256="",
        )
    elif case == "corrected-evidence-invalid":
        index = next(
            index for index, outcome in enumerate(original.groups) if outcome.status == "review"
        )
        changed = replace(corrected.groups[index], status="review", group=None)
        corrected = replace(
            corrected,
            groups=(*corrected.groups[:index], changed, *corrected.groups[index + 1 :]),
            integrity_sha256="",
        )
    else:
        prior = events[0]
        orphan = review_module._new_review_event(
            "pairing:5:rec-1:6:orphan",
            "pairing_selected_split",
            None,
            prior.after,
            {
                "reason": prior.reason,
                "reviewer": prior.reviewer,
                "reviewed_at": prior.reviewed_at,
            },
        )
        events = (orphan,)

    with pytest.raises((TypeError, ValueError), match=r"pairing|correction|corrected"):
        review_module.validate_pairing_correction_events(original, corrected, events)


@pytest.mark.parametrize(
    "case",
    (
        "missing-file",
        "extra-file",
        "automatic-stale",
        "decision-identity",
        "selections-not-list",
        "row-not-object",
        "unit-type",
        "unreviewed-metadata",
        "split-type",
        "missing-reason",
        "reviewed-at-type",
        "reviewed-at-invalid",
        "reviewed-at-naive",
        "unknown-unit",
    ),
)
def test_pairing_correction_rejects_schema_metadata_and_bundle_drift(
    tmp_path: Path, case: str
) -> None:
    paths, config, _ = _review_required_project(tmp_path)
    correction = export_review_bundle(paths, config)[0]
    automatic_path = correction / "automatic.json"
    decision_path = correction / "decision.json"
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    first = decision["pairing_selections"][0]
    saved_split = automatic["review_units"][0]["candidates"][0]["split_sample"]
    if case == "missing-file":
        (correction / "index.html").unlink()
    elif case == "extra-file":
        (correction / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            export_review_bundle(paths, config)
        return
    elif case == "automatic-stale":
        automatic["recording_id"] = "wrong"
        automatic_path.write_text(json.dumps(automatic), encoding="utf-8")
    elif case == "decision-identity":
        decision["recording_id"] = "wrong"
    elif case == "selections-not-list":
        decision["pairing_selections"] = {}
    elif case == "row-not-object":
        decision["pairing_selections"][0] = None
    elif case == "unit-type":
        first["unit_id"] = 7
    elif case == "unreviewed-metadata":
        first["reason"] = "not empty"
    elif case == "split-type":
        first.update(
            split_sample=True,
            reason="reason",
            reviewer="owner",
            reviewed_at="2026-07-19T12:00:00+08:00",
        )
    elif case == "missing-reason":
        first.update(
            split_sample=saved_split,
            reviewer="owner",
            reviewed_at="2026-07-19T12:00:00+08:00",
        )
    elif case == "reviewed-at-type":
        first.update(split_sample=saved_split, reason="reason", reviewer="owner", reviewed_at=7)
    elif case == "reviewed-at-invalid":
        first.update(
            split_sample=saved_split,
            reason="reason",
            reviewer="owner",
            reviewed_at="invalid",
        )
    elif case == "reviewed-at-naive":
        first.update(
            split_sample=saved_split,
            reason="reason",
            reviewer="owner",
            reviewed_at="2026-07-19T12:00:00",
        )
    elif case == "unknown-unit":
        first["unit_id"] = "unknown-unit"
    if case not in {"missing-file", "automatic-stale"}:
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        import_review_bundle(paths, config)


def _set_decisions(
    group: Path,
    *,
    decision: str = "approved",
    reason: str = "listened in full",
    reviewer: str = "owner",
    reviewed_at: str = "2026-07-19T12:00:00+08:00",
    take_indexes: tuple[int, ...] = (1, 2),
) -> None:
    path = group / "decision.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    for index, item in enumerate(raw["decisions"], 1):
        if index not in take_indexes:
            continue
        item.update(
            decision=decision,
            reason=reason,
            reviewer=reviewer,
            reviewed_at=reviewed_at,
        )
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def test_import_review_appends_take_events_and_advances_only_when_complete(
    tmp_path: Path,
) -> None:
    paths, config = _aligned_project(tmp_path)
    groups = export_review_bundle(paths, config)
    first_grid = groups[0] / "take-1.TextGrid"
    boundaries = read_textgrid(first_grid)
    changed_words = (
        ReviewedWordSpan(
            boundaries.words[0].text,
            boundaries.words[0].start_seconds + 0.01,
            boundaries.words[0].end_seconds,
        ),
        *boundaries.words[1:],
    )
    write_textgrid(
        first_grid,
        duration_seconds=boundaries.duration_seconds,
        take_start=boundaries.take_start,
        take_end=boundaries.take_end,
        words=changed_words,
    )
    _set_decisions(groups[0])

    assert not import_review_bundle(paths, config)
    events_path = paths.manifests / "review.jsonl"
    first_events = read_jsonl(events_path)
    assert [event["field"] for event in first_events] == [
        "word:0:start_seconds",
        "review_decision",
        "review_decision",
    ]
    assert all(event["before"] != event["after"] for event in first_events)
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "ALIGNED"

    _set_decisions(groups[1], reviewed_at="2026-07-19T12:01:00+08:00")
    _set_decisions(
        groups[2],
        decision="rejected",
        reason="background noise",
        reviewed_at="2026-07-19T12:02:00+08:00",
    )
    assert import_review_bundle(paths, config)
    assert len(read_jsonl(events_path)) == 7
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "REVIEWED"

    assert import_review_bundle(paths, config)
    assert len(read_jsonl(events_path)) == 7


def test_import_review_advances_each_complete_recording_independently(tmp_path: Path) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    groups = export_review_bundle(paths, config)
    by_recording = {
        recording_id: tuple(group for group in groups if group.parents[1].name == recording_id)
        for recording_id in ("rec-1", "rec-2")
    }
    assert all(len(items) == 3 for items in by_recording.values())
    for group in by_recording["rec-1"]:
        _set_decisions(group)
    _set_decisions(by_recording["rec-2"][0])

    assert not import_review_bundle(paths, config)
    states = {
        row["recording_id"]: row["state"]
        for row in read_jsonl(paths.manifests / "recordings.jsonl")
    }
    assert states == {"rec-1": "REVIEWED", "rec-2": "ALIGNED"}

    for group in by_recording["rec-2"][1:]:
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    assert {
        row["recording_id"]: row["state"]
        for row in read_jsonl(paths.manifests / "recordings.jsonl")
    } == {"rec-1": "REVIEWED", "rec-2": "REVIEWED"}


def test_import_review_rejects_conflicting_duplicate_submission(tmp_path: Path) -> None:
    paths, config = _aligned_project(tmp_path)
    groups = export_review_bundle(paths, config)
    _set_decisions(groups[0])
    assert not import_review_bundle(paths, config)
    _set_decisions(groups[0], reason="different reason")

    try:
        import_review_bundle(paths, config)
    except ValueError as error:
        assert "conflict" in str(error)
    else:
        raise AssertionError("conflicting duplicate review was accepted")


def test_import_review_rejects_missing_reason_extra_fields_and_tampered_audio(
    tmp_path: Path,
) -> None:
    paths, config = _aligned_project(tmp_path)
    groups = export_review_bundle(paths, config)
    decision_path = groups[0] / "decision.json"
    raw = json.loads(decision_path.read_text(encoding="utf-8"))
    raw["decisions"][0].update(
        decision="approved",
        reviewer="owner",
        reviewed_at="2026-07-19T12:00:00+08:00",
    )
    decision_path.write_text(json.dumps(raw), encoding="utf-8")
    try:
        import_review_bundle(paths, config)
    except ValueError as error:
        assert "reason" in str(error)
    else:
        raise AssertionError("review without reason was accepted")

    _set_decisions(groups[0])
    raw = json.loads(decision_path.read_text(encoding="utf-8"))
    raw["decisions"][0]["unexpected"] = True
    decision_path.write_text(json.dumps(raw), encoding="utf-8")
    try:
        import_review_bundle(paths, config)
    except ValueError as error:
        assert "exact fields" in str(error)
    else:
        raise AssertionError("review with extra decision field was accepted")

    del raw["decisions"][0]["unexpected"]
    decision_path.write_text(json.dumps(raw), encoding="utf-8")
    (groups[0] / "take-1.wav").write_bytes(b"tampered")
    try:
        import_review_bundle(paths, config)
    except ValueError as error:
        assert "audio" in str(error)
    else:
        raise AssertionError("review with tampered audio was accepted")


def test_export_review_is_idempotent_without_overwriting_human_files(tmp_path: Path) -> None:
    paths, config = _aligned_project(tmp_path)
    groups = export_review_bundle(paths, config)
    decision_path = groups[0] / "decision.json"
    grid_path = groups[0] / "take-1.TextGrid"
    _set_decisions(groups[0])
    decision_bytes = decision_path.read_bytes()
    grid_path.write_bytes(grid_path.read_bytes() + b"\n")
    grid_bytes = grid_path.read_bytes()

    assert export_review_bundle(paths, config) == groups
    assert decision_path.read_bytes() == decision_bytes
    assert grid_path.read_bytes() == grid_bytes


def test_review_cli_exports_and_imports_complete_human_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _aligned_project(tmp_path)
    monkeypatch.setattr("latintts.corpus.review.subprocess.run", _review_audio_command)
    monkeypatch.setattr("latintts.corpus.cli._ffmpeg_version", lambda: "ffmpeg-test-1")
    assert main(["--project-root", str(tmp_path), "export-review"]) == 0
    groups = tuple(
        sorted((paths.alignments / "runs" / config.digest / "rec-1" / "review").iterdir())
    )
    for index, group in enumerate(groups):
        _set_decisions(group, reviewed_at=f"2026-07-19T12:0{index}:00+08:00")

    assert main(["--project-root", str(tmp_path), "import-review"]) == 0
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "REVIEWED"


def test_import_review_persists_audit_events_before_materialized_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _aligned_project(tmp_path)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    destinations: list[Path] = []
    real_replace = store.os.replace

    def observe_replace(source: Path, destination: Path) -> None:
        destinations.append(Path(destination))
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", observe_replace)
    assert import_review_bundle(paths, config)

    assert destinations[-3:] == [
        paths.manifests / "review.jsonl",
        paths.alignments / "runs" / config.digest / "processing-events.jsonl",
        paths.manifests / "recordings.jsonl",
    ]


def test_review_correction_appends_new_before_after_event_without_rewriting_history(
    tmp_path: Path,
) -> None:
    paths, config = _aligned_project(tmp_path)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    review_path = paths.manifests / "review.jsonl"
    original_rows = read_jsonl(review_path)
    grid_path = groups[0] / "take-1.TextGrid"
    boundaries = read_textgrid(grid_path)
    changed = (
        ReviewedWordSpan(
            boundaries.words[0].text,
            boundaries.words[0].start_seconds + 0.01,
            boundaries.words[0].end_seconds,
        ),
        *boundaries.words[1:],
    )
    write_textgrid(
        grid_path,
        duration_seconds=boundaries.duration_seconds,
        take_start=boundaries.take_start,
        take_end=boundaries.take_end,
        words=changed,
    )
    _set_decisions(
        groups[0],
        reviewed_at="2026-07-19T13:00:00+08:00",
        take_indexes=(1,),
    )

    assert import_review_bundle(paths, config)
    corrected_rows = read_jsonl(review_path)
    assert corrected_rows[: len(original_rows)] == original_rows
    assert corrected_rows[-1]["field"] == "word:0:start_seconds"
    assert corrected_rows[-1]["before"] == boundaries.words[0].start_seconds
    assert corrected_rows[-1]["after"] == boundaries.words[0].start_seconds + 0.01
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "REVIEWED"


def test_import_review_rejects_tampered_history_and_nested_automatic_schema(
    tmp_path: Path,
) -> None:
    paths, config = _aligned_project(tmp_path)
    groups = export_review_bundle(paths, config)
    _set_decisions(groups[0])
    assert not import_review_bundle(paths, config)
    review_path = paths.manifests / "review.jsonl"
    rows = list(read_jsonl(review_path))
    rows[0]["before"] = "tampered"
    write_jsonl_atomic(review_path, rows)
    with pytest.raises(ValueError, match=r"event ID|before"):
        import_review_bundle(paths, config)
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "ALIGNED"

    review_path.unlink()
    automatic_path = groups[0] / "automatic.json"
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    automatic["takes"][0]["quality_metrics"]["unexpected"] = True
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")
    with pytest.raises(ValueError, match="exact fields"):
        import_review_bundle(paths, config)


def test_import_review_rebuilds_trusted_clip_instead_of_trusting_edited_hash(
    tmp_path: Path,
) -> None:
    paths, config = _aligned_project(tmp_path)
    group = export_review_bundle(paths, config)[0]
    audio_path = group / "take-1.wav"
    with wave.open(str(audio_path), "rb") as reader:
        parameters = reader.getparams()
        frame_count = reader.getnframes()
    with wave.open(str(audio_path), "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(b"\x02\x00\x00" * frame_count)
    automatic_path = group / "automatic.json"
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    automatic["takes"][0]["audio_sha256"] = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")

    with pytest.raises(ValueError, match=r"audio|provenance|trusted"):
        import_review_bundle(paths, config)


def test_import_review_rejects_bundle_after_exact_alignment_artifact_bytes_change(
    tmp_path: Path,
) -> None:
    paths, config = _aligned_project(tmp_path)
    export_review_bundle(paths, config)
    alignment_path = paths.alignments / "runs" / config.digest / "rec-1" / "alignment.json"
    alignment_path.write_text(
        alignment_path.read_text(encoding="utf-8").rstrip("\n") + " \n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"artifact|stale|binding"):
        import_review_bundle(paths, config)


def test_import_review_rejects_nested_bundle_drift(tmp_path: Path) -> None:
    paths, config = _aligned_project(tmp_path)
    group = export_review_bundle(paths, config)[0]
    automatic_path = group / "automatic.json"
    decision_path = group / "decision.json"
    grid_path = group / "take-1.TextGrid"
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    grid = grid_path.read_text(encoding="utf-8")
    grid_duration = read_textgrid(grid_path).duration_seconds

    mutations = (
        ("metric-type", "metric and version"),
        ("candidate-score", "candidate scores"),
        ("quality", "quality metrics"),
        ("version", "alignment versions"),
        ("warnings", "warnings"),
        ("candidate-hash", "candidate audio"),
        ("source-sample", "source sample"),
        ("automatic-value", "automatic values"),
        ("identity", "automatic identity"),
        ("source-type", "source_audio"),
        ("source-identity", "source audio identity"),
        ("text-type", "text_layers"),
        ("text-layer", "text layers"),
        ("alignment-text", "alignment text layers"),
        ("take-count", "exactly two takes"),
        ("take-type", "take must be an object"),
        ("take-identity", "take identity"),
        ("decision-identity", "decision identity"),
        ("decision-count", "exactly two decisions"),
        ("decision-entity", "entity_id must be unique"),
        ("entity", "entity identity"),
        ("grid-duration", "duration differs"),
    )
    for mutation, match in mutations:
        changed_automatic = deepcopy(automatic)
        changed_decision = deepcopy(decision)
        changed_grid = grid
        take = changed_automatic["takes"][0]
        if mutation == "metric-type":
            take["candidate_scores"] = None
        elif mutation == "candidate-score":
            take["candidate_scores"]["coverage"] = 0.1
        elif mutation == "quality":
            take["quality_metrics"]["mean_score"] = 0.1
        elif mutation == "version":
            take["alignment_versions"]["backend"] = "wrong"
        elif mutation == "warnings":
            take["warnings"] = ["invented"]
        elif mutation == "candidate-hash":
            take["candidate_audio_sha256"] = "0" * 64
        elif mutation == "source-sample":
            take["source_start_sample"] += 1
        elif mutation == "automatic-value":
            take["automatic_values"]["segment_end"] -= 0.01
        elif mutation == "identity":
            changed_automatic["recording_id"] = "wrong"
        elif mutation == "source-type":
            changed_automatic["source_audio"] = None
        elif mutation == "source-identity":
            changed_automatic["source_audio"]["sha256"] = "0" * 64
        elif mutation == "text-type":
            changed_automatic["text_layers"] = None
        elif mutation == "text-layer":
            changed_automatic["text_layers"]["spoken_text"] = "wrong"
        elif mutation == "alignment-text":
            changed_automatic["text_layers"]["alignment_texts"][0] = "wrong"
        elif mutation == "take-count":
            changed_automatic["takes"] = changed_automatic["takes"][:1]
        elif mutation == "take-type":
            changed_automatic["takes"][0] = None
        elif mutation == "take-identity":
            changed_automatic["takes"][1]["take_index"] = 1
        elif mutation == "decision-identity":
            changed_decision["recording_id"] = "wrong"
        elif mutation == "decision-count":
            changed_decision["decisions"] = changed_decision["decisions"][:1]
        elif mutation == "decision-entity":
            changed_decision["decisions"][1]["entity_id"] = changed_decision["decisions"][0][
                "entity_id"
            ]
        elif mutation == "entity":
            changed_automatic["takes"][0]["entity_id"] = "wrong"
        else:
            changed_grid = changed_grid.replace(
                f"xmax = {grid_duration}\ntiers?",
                f"xmax = {grid_duration + 0.1}\ntiers?",
                1,
            )
        automatic_path.write_text(json.dumps(changed_automatic), encoding="utf-8")
        decision_path.write_text(json.dumps(changed_decision), encoding="utf-8")
        grid_path.write_text(changed_grid, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            import_review_bundle(paths, config)
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    grid_path.write_text(grid, encoding="utf-8")


def test_export_review_rejects_existing_bundle_file_drift(tmp_path: Path) -> None:
    paths, config = _aligned_project(tmp_path)
    group = export_review_bundle(paths, config)[0]
    (group / "unexpected.txt").write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="exact file schema"):
        export_review_bundle(paths, config)


def test_export_review_rejects_invalid_transcript_layer_manifest(tmp_path: Path) -> None:
    paths, config = _aligned_project(tmp_path)
    transcript_path = paths.manifests / "transcripts.jsonl"
    row = read_jsonl(transcript_path)[0]
    write_jsonl_atomic(transcript_path, (row, row))
    with pytest.raises(ValueError, match="unique recording_id"):
        export_review_bundle(paths, config)
    empty = deepcopy(row)
    empty["source_text"] = ""
    write_jsonl_atomic(transcript_path, (empty,))
    with pytest.raises(ValueError, match="source_text"):
        export_review_bundle(paths, config)
    write_jsonl_atomic(transcript_path, ())
    with pytest.raises(ValueError, match="pilot selection"):
        export_review_bundle(paths, config)
