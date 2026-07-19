from __future__ import annotations

import hashlib
import json
import wave
from copy import deepcopy
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from latintts.corpus import store
from latintts.corpus.cli import align_corpus, main, pair_corpus
from latintts.corpus.config import CorpusConfig
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
        with pytest.raises((TypeError, ValueError), match=match):
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
