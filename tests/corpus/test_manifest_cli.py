from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from latintts.corpus import audio as audio_module
from latintts.corpus import cli
from latintts.corpus import manifest as manifest_module
from latintts.corpus import review as review_module
from latintts.corpus import store as store_module
from latintts.corpus.alignment import result_from_dict
from latintts.corpus.audio import DerivedAudio, PcmMetrics
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.manifest import build_approved_segments, build_manifest_corpus
from latintts.corpus.pairing import pairing_from_dict
from latintts.corpus.records import ProcessingEvent, ReviewEvent, RightsRecord, advance_recording
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from tests.corpus.test_pairing import _audio_command
from tests.corpus.test_review_cli import (
    _aligned_project,
    _corrected_pairing_project,
    _set_decisions,
    _two_recording_aligned_project,
    export_review_bundle,
    import_review_bundle,
)


def test_build_manifest_cli_loads_config_and_dispatches(
    tmp_path: Path, monkeypatch: object
) -> None:
    config = tmp_path / "config" / "corpus" / "pilot-v1.json"
    config.parent.mkdir(parents=True)
    source = Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json"
    config.write_bytes(source.read_bytes())
    calls: list[tuple[Path, str]] = []

    def build(paths: object, corpus_config: object, *, ffmpeg_version: str) -> bool:
        calls.append((paths.project_root, corpus_config.raw["corpus_version"]))  # type: ignore[attr-defined]
        assert ffmpeg_version == "ffmpeg-test-1"
        return True

    monkeypatch.setattr(cli, "build_manifest_corpus", build, raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "_ffmpeg_version", lambda: "ffmpeg-test-1")  # type: ignore[attr-defined]

    assert (
        cli.main(
            [
                "--project-root",
                str(tmp_path),
                "build-manifest",
                "--config",
                "config/corpus/pilot-v1.json",
            ]
        )
        == 0
    )
    assert calls == [(tmp_path, "corpus-v1")]


@pytest.mark.parametrize(
    ("failure", "expected"), [(CorpusFailure("REVIEW_REQUIRED", "x"), 1), (ValueError("bad"), 2)]
)
def test_build_manifest_cli_maps_stable_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
    expected: int,
) -> None:
    config = tmp_path / "config" / "corpus" / "pilot-v1.json"
    config.parent.mkdir(parents=True)
    source = Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json"
    config.write_bytes(source.read_bytes())
    monkeypatch.setattr(
        cli,
        "build_manifest_corpus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(cli, "_ffmpeg_version", lambda: "ffmpeg-test-1")

    assert cli.main(["--project-root", str(tmp_path), "build-manifest"]) == expected
    assert ("REVIEW_REQUIRED" if expected == 1 else "MANIFEST_SCHEMA_MISMATCH") in (
        capsys.readouterr().err
    )


def _write_rights(paths: object) -> None:
    write_jsonl_atomic(
        paths.manifests / "rights.jsonl",  # type: ignore[attr-defined]
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


def _reviewed_project(tmp_path: Path, *, decision: str = "approved") -> tuple[object, object]:
    paths, config = _aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group, decision=decision)
    assert import_review_bundle(paths, config)
    return paths, config


def _build_with_fake_audio(
    paths: object,
    config: CorpusConfig,
    *,
    after_transcode: Callable[[], None] | None = None,
    before_publish_link: Callable[[Path], None] | None = None,
) -> bool:
    sample_counts: dict[Path, int] = {}

    def transcode(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        output = Path(command[-1])
        if output.suffix == ".flac":
            start = float(command[command.index("-ss") + 1])
            end = float(command[command.index("-to") + 1])
            sample_counts[output] = round((end - start) * 48_000)
            output.write_bytes(b"fLaCsynthetic" + str(sample_counts[output]).encode("ascii"))
            if after_transcode is not None:
                after_transcode()
            return CompletedProcess(command, 0, "", "")
        return _audio_command(command, **kwargs)

    def probe(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        target = Path(command[-1])
        samples = sample_counts.get(target)
        if samples is None:
            samples = int(target.read_bytes().removeprefix(b"fLaCsynthetic"))
        return CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "format": {"duration": str(samples / 48_000)},
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "flac",
                            "sample_rate": "48000",
                            "channels": 1,
                            "duration_ts": str(samples),
                            "time_base": "1/48000",
                        }
                    ],
                }
            ),
            "",
        )

    return build_manifest_corpus(
        paths,  # type: ignore[arg-type]
        config,
        ffmpeg_version="ffmpeg-test-1",
        run_command=transcode,
        probe_command=probe,
        decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
        _before_publish_link=before_publish_link,
    )


def _make_directory_alias(alias: Path, target: Path) -> None:
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks unavailable: {error}")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True,
            check=False,
            encoding="utf-8",
            text=True,
        )
        if completed.returncode:
            pytest.skip(f"directory aliases unavailable: {error}; {completed.stderr}")


def test_build_manifest_accepts_strict_review_prefix_with_post_review_corrections(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    review_root = paths.alignments / "runs" / config.digest / "rec-1" / "review"  # type: ignore[attr-defined]
    group = sorted(path for path in review_root.iterdir() if path.is_dir())[0]
    grid_path = group / "take-1.TextGrid"
    boundaries = review_module.read_textgrid(grid_path)
    changed_words = (
        review_module.ReviewedWordSpan(
            boundaries.words[0].text,
            boundaries.words[0].start_seconds + 0.01,
            boundaries.words[0].end_seconds,
        ),
        *boundaries.words[1:],
    )
    review_module.write_textgrid(
        grid_path,
        duration_seconds=boundaries.duration_seconds,
        take_start=boundaries.take_start,
        take_end=boundaries.take_end,
        words=changed_words,
    )
    _set_decisions(group, reviewed_at="2026-07-19T13:00:00+08:00", take_indexes=(1,))
    _set_decisions(
        group,
        decision="rejected",
        reason="corrected rejection",
        reviewed_at="2026-07-19T13:01:00+08:00",
        take_indexes=(2,),
    )
    assert import_review_bundle(paths, config)

    assert _build_with_fake_audio(paths, config)
    rows = read_jsonl(paths.manifests / "segments.jsonl")  # type: ignore[attr-defined]
    assert len(rows) == 5
    first = next(row for row in rows if row["text_unit_id"].endswith("0001"))
    assert first["take_index"] == 1
    assert first["word_spans"][0]["start_seconds"] == pytest.approx(0.01)
    journal = read_jsonl(paths.manifests / "review.jsonl")  # type: ignore[attr-defined]
    entity = next(
        event["entity_id"]
        for event in journal
        if event["review_event_id"] == first["review_event_ids"][0]
    )
    assert first["review_event_ids"] == [
        event["review_event_id"] for event in journal if event["entity_id"] == entity
    ]


@pytest.mark.parametrize("corruption", ("non-prefix-digest", "illegal-suffix"))
def test_build_manifest_rejects_invalid_review_prefix_or_suffix_without_writes(
    tmp_path: Path,
    corruption: str,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    review_path = paths.manifests / "review.jsonl"  # type: ignore[attr-defined]
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    if corruption == "non-prefix-digest":
        rows = list(read_jsonl(processing_path))
        index = next(
            index
            for index, row in enumerate(rows)
            if row["previous_state"] == "ALIGNED" and row["target_state"] == "REVIEWED"
        )
        old = ProcessingEvent.from_dict(rows[index])
        recording = review_module._decode_recording(
            read_jsonl(paths.manifests / "recordings.jsonl")[0]  # type: ignore[attr-defined]
        )
        _, replacement = advance_recording(
            replace(recording, state=CorpusState.ALIGNED),
            CorpusState.REVIEWED,
            input_sha256s=(*old.input_sha256s[:2], "f" * 64),
            config_sha256=old.config_sha256,
            tool_versions=old.tool_versions,
            started_at=old.started_at,
            finished_at=old.finished_at,
            result=old.result,
        )
        rows[index] = replacement.to_dict()
        write_jsonl_atomic(processing_path, rows)
    else:
        illegal = review_module._new_review_event(
            "review:unknown",
            "review_decision",
            "unreviewed",
            "approved",
            {
                "reason": "orphan suffix",
                "reviewer": "owner",
                "reviewed_at": "2026-07-20T00:00:00+08:00",
            },
        )
        write_jsonl_atomic(review_path, (*read_jsonl(review_path), illegal.to_dict()))
    protected = (
        review_path,
        processing_path,
        paths.manifests / "recordings.jsonl",  # type: ignore[attr-defined]
    )
    before = {path: path.read_bytes() for path in protected}

    with pytest.raises(ValueError, match=r"prefix|review|entity|processing"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("invalid review history must fail before extraction")
            ),
        )

    assert {path: path.read_bytes() for path in protected} == before
    assert not (paths.manifests / "segments.jsonl").exists()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "corruption",
    ("missing", "raw", "analysis", "transcript", "model", "config", "tools"),
)
def test_build_manifest_rejects_missing_or_tampered_segmented_event_without_durable_writes(
    tmp_path: Path,
    corruption: str,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    run_directory = paths.alignments / "runs" / config.digest  # type: ignore[attr-defined]
    processing_path = run_directory / "processing-events.jsonl"
    rows = list(read_jsonl(processing_path))
    segmented_index = next(
        index
        for index, row in enumerate(rows)
        if row["recording_id"] == "rec-1"
        and row["previous_state"] == "TRANSCRIPT_CONFIRMED"
        and row["target_state"] == "SEGMENTED"
    )
    segmented = ProcessingEvent.from_dict(rows[segmented_index])
    if corruption == "missing":
        del rows[segmented_index]
    else:
        inputs = list(segmented.input_sha256s)
        input_index = {"raw": 0, "analysis": 1, "transcript": 2, "model": 3}
        if corruption in input_index:
            index = input_index[corruption]
            inputs[index] = "f" * 64 if inputs[index] != "f" * 64 else "e" * 64
        event_config = segmented.config_sha256
        if corruption == "config":
            event_config = "f" * 64 if event_config != "f" * 64 else "e" * 64
        tools = segmented.tool_versions
        if corruption == "tools":
            tools = ("tampered-segmentation=1",)
        recording = review_module._decode_recording(
            read_jsonl(paths.manifests / "recordings.jsonl")[0]  # type: ignore[attr-defined]
        )
        _, replacement = advance_recording(
            replace(recording, state=CorpusState.TRANSCRIPT_CONFIRMED),
            CorpusState.SEGMENTED,
            input_sha256s=tuple(inputs),
            config_sha256=event_config,
            tool_versions=tools,
            started_at=segmented.started_at,
            finished_at=segmented.finished_at,
            result=segmented.result,
        )
        rows[segmented_index] = replacement.to_dict()
    write_jsonl_atomic(processing_path, rows)
    protected = (
        processing_path,
        paths.manifests / "recordings.jsonl",  # type: ignore[attr-defined]
        paths.manifests / "review.jsonl",  # type: ignore[attr-defined]
        run_directory / "rec-1" / "segmentation.json",
        run_directory / "rec-1" / "pairing.json",
        run_directory / "rec-1" / "alignment.json",
    )
    before = {path: path.read_bytes() for path in protected}
    lossless_before = tuple((paths.segments / "lossless").glob("*"))  # type: ignore[attr-defined]
    candidates_before = tuple((paths.segments / "candidates").glob("*"))  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match=r"SEGMENTED|segmentation|processing history"):
        _build_with_fake_audio(paths, config)  # type: ignore[arg-type]

    assert {path: path.read_bytes() for path in protected} == before
    assert not (paths.manifests / "segments.jsonl").exists()  # type: ignore[attr-defined]
    assert tuple((paths.segments / "lossless").glob("*")) == lossless_before  # type: ignore[attr-defined]
    assert tuple((paths.segments / "candidates").glob("*")) == candidates_before  # type: ignore[attr-defined]


@pytest.mark.parametrize("alias_component", ("staging-parent", "mode-root"))
def test_build_manifest_rejects_output_component_alias_before_runner_or_publish(
    tmp_path: Path,
    alias_component: str,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    outside = tmp_path / f"outside-{alias_component}"
    outside.mkdir()
    alias = (
        paths.segments / ".manifest-staging"  # type: ignore[attr-defined]
        if alias_component == "staging-parent"
        else paths.segments / "lossless"  # type: ignore[attr-defined]
    )
    alias.parent.mkdir(parents=True, exist_ok=True)
    _make_directory_alias(alias, outside)
    protected = (
        paths.manifests / "review.jsonl",  # type: ignore[attr-defined]
        paths.alignments / "runs" / config.digest / "processing-events.jsonl",  # type: ignore[attr-defined]
        paths.manifests / "recordings.jsonl",  # type: ignore[attr-defined]
    )
    before = {path: path.read_bytes() for path in protected}

    with pytest.raises(ValueError, match=r"alias|reparse|canonical"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("output alias must be rejected before the runner")
            ),
        )

    assert {path: path.read_bytes() for path in protected} == before
    assert not tuple(outside.rglob("*"))


def test_build_manifest_rechecks_final_mode_root_alias_before_link_or_state_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    outside = tmp_path / "outside-race"
    outside.mkdir()
    lossless_root = paths.segments / "lossless"  # type: ignore[attr-defined]
    injected = False

    def inject_alias() -> None:
        nonlocal injected
        if injected:
            return
        injected = True
        _make_directory_alias(lossless_root, outside)

    real_link = os.link

    def guarded_link(source: Path, destination: Path) -> None:
        if ".manifest-staging" not in Path(destination).parts:
            raise AssertionError("reparse mode root must be rejected before hard-link publication")
        real_link(source, destination)

    monkeypatch.setattr(manifest_module.os, "link", guarded_link)
    protected = (
        paths.manifests / "review.jsonl",  # type: ignore[attr-defined]
        paths.alignments / "runs" / config.digest / "processing-events.jsonl",  # type: ignore[attr-defined]
        paths.manifests / "recordings.jsonl",  # type: ignore[attr-defined]
    )
    before = {path: path.read_bytes() for path in protected}

    with pytest.raises(ValueError, match=r"alias|reparse|canonical"):
        _build_with_fake_audio(paths, config, after_transcode=inject_alias)

    assert injected
    assert {path: path.read_bytes() for path in protected} == before
    assert not (paths.manifests / "segments.jsonl").exists()  # type: ignore[attr-defined]
    assert not tuple(outside.rglob("*"))


def test_build_manifest_pins_final_directory_against_last_check_link_race(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    outside = tmp_path / "outside-last-check-race"
    outside.mkdir()
    attempted = False
    replaced = False

    def replace_mode_root(mode_root: Path) -> None:
        nonlocal attempted, replaced
        attempted = True
        mode_root.rmdir()
        _make_directory_alias(mode_root, outside)
        replaced = True

    protected = (
        paths.manifests / "review.jsonl",  # type: ignore[attr-defined]
        paths.alignments / "runs" / config.digest / "processing-events.jsonl",  # type: ignore[attr-defined]
        paths.manifests / "recordings.jsonl",  # type: ignore[attr-defined]
    )
    before = {path: path.read_bytes() for path in protected}

    with pytest.raises(ValueError, match=r"publication|directory|alias|reparse|canonical"):
        _build_with_fake_audio(paths, config, before_publish_link=replace_mode_root)

    assert attempted
    assert not replaced
    assert {path: path.read_bytes() for path in protected} == before
    assert not (paths.manifests / "segments.jsonl").exists()  # type: ignore[attr-defined]
    assert not tuple(outside.rglob("*"))


def test_build_manifest_holds_publication_guard_through_terminal_state_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    real_persist = manifest_module.persist_recording_transitions
    blocked = False

    def persist_with_rename_attempt(**kwargs: object) -> None:
        nonlocal blocked
        lossless_root = paths.segments / "lossless"  # type: ignore[attr-defined]
        try:
            lossless_root.rename(tmp_path / "moved-lossless")
        except OSError:
            blocked = True
        else:
            raise AssertionError("publication mode root was not pinned through state commit")
        real_persist(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        persist_with_rename_attempt,
    )

    assert _build_with_fake_audio(paths, config)
    assert blocked
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "APPROVED"  # type: ignore[attr-defined]


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing contract")
@pytest.mark.parametrize("reuse_existing", (False, True))
def test_build_manifest_pins_new_and_existing_final_artifacts_through_terminal_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reuse_existing: bool,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    recordings_path = paths.manifests / "recordings.jsonl"  # type: ignore[attr-defined]
    if reuse_existing:
        assert _build_with_fake_audio(paths, config)
        recordings = []
        for row in read_jsonl(recordings_path):
            reviewed = dict(row)
            reviewed["state"] = "REVIEWED"
            recordings.append(reviewed)
        write_jsonl_atomic(recordings_path, recordings)
        write_jsonl_atomic(
            processing_path,
            (
                row
                for row in read_jsonl(processing_path)
                if not (
                    row["previous_state"] == "REVIEWED"
                    and row["target_state"] in {"APPROVED", "REJECTED"}
                )
            ),
        )

    real_persist = manifest_module.persist_recording_transitions
    blocked_write = False
    blocked_rename = False

    def persist_with_artifact_attacks(**kwargs: object) -> None:
        nonlocal blocked_write, blocked_rename
        segment = read_jsonl(paths.manifests / "segments.jsonl")[0]  # type: ignore[attr-defined]
        final = paths.local_data / Path(*segment["derived_audio_relative_path"].split("/"))  # type: ignore[attr-defined]
        try:
            final.write_bytes(b"tampered during terminal commit")
        except OSError:
            blocked_write = True
        try:
            final.rename(final.with_suffix(".moved"))
        except OSError:
            blocked_rename = True
        if not blocked_write or not blocked_rename:
            raise AssertionError(
                "final artifact was not pinned through terminal commit: "
                f"write={blocked_write}, rename={blocked_rename}"
            )
        real_persist(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        persist_with_artifact_attacks,
    )

    assert _build_with_fake_audio(paths, config)
    assert blocked_write and blocked_rename
    segment = read_jsonl(paths.manifests / "segments.jsonl")[0]  # type: ignore[attr-defined]
    final = paths.local_data / Path(*segment["derived_audio_relative_path"].split("/"))  # type: ignore[attr-defined]
    assert hashlib.sha256(final.read_bytes()).hexdigest() == segment["derived_audio_sha256"]
    assert read_jsonl(recordings_path)[0]["state"] == "APPROVED"


@pytest.mark.skipif(os.name != "nt", reason="Windows directory-sharing contract")
def test_build_manifest_pins_manifest_namespace_before_terminal_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    real_persist = manifest_module.persist_recording_transitions
    moved = tmp_path / "moved-manifests"
    blocked = False

    def persist_with_namespace_attack(**kwargs: object) -> None:
        nonlocal blocked
        recordings = (paths.manifests / "recordings.jsonl").read_bytes()  # type: ignore[attr-defined]
        try:
            paths.manifests.rename(moved)  # type: ignore[attr-defined]
        except OSError:
            blocked = True
        else:
            paths.manifests.mkdir()  # type: ignore[attr-defined]
            (paths.manifests / "recordings.jsonl").write_bytes(recordings)  # type: ignore[attr-defined]
        real_persist(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        persist_with_namespace_attack,
    )

    assert _build_with_fake_audio(paths, config)
    assert blocked
    assert not moved.exists()
    assert (paths.manifests / "segments.jsonl").is_file()  # type: ignore[attr-defined]
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "APPROVED"  # type: ignore[attr-defined]


@pytest.mark.skipif(os.name != "nt", reason="Windows directory-sharing contract")
def test_build_manifest_pins_processing_namespace_during_event_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    processing_parent = processing_path.parent
    moved = tmp_path / "moved-processing-run"
    real_write = store_module.write_jsonl_atomic
    blocked = False
    attempted = False

    def write_with_namespace_attack(
        path: Path,
        rows: object,
        **kwargs: object,
    ) -> None:
        nonlocal blocked, attempted
        real_write(path, rows, **kwargs)  # type: ignore[arg-type]
        if path != processing_path or attempted:
            return
        attempted = True
        try:
            processing_parent.rename(moved)
        except OSError:
            blocked = True

    monkeypatch.setattr(store_module, "write_jsonl_atomic", write_with_namespace_attack)

    assert _build_with_fake_audio(paths, config)
    assert attempted and blocked
    assert not moved.exists()
    assert any(
        row["previous_state"] == "REVIEWED" and row["target_state"] == "APPROVED"
        for row in read_jsonl(processing_path)
    )
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "APPROVED"  # type: ignore[attr-defined]


@pytest.mark.skipif(os.name != "nt", reason="Windows directory-sharing contract")
def test_build_manifest_pins_output_ancestor_chain_through_terminal_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    real_persist = manifest_module.persist_recording_transitions
    ancestor = paths.local_data / "derived" / "corpus-v1"  # type: ignore[attr-defined]
    moved = tmp_path / "moved-corpus-version"
    blocked = False

    def persist_with_ancestor_attack(**kwargs: object) -> None:
        nonlocal blocked
        try:
            ancestor.rename(moved)
        except OSError:
            blocked = True
        real_persist(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        persist_with_ancestor_attack,
    )

    assert _build_with_fake_audio(paths, config)
    assert blocked
    assert not moved.exists()


def test_output_namespace_guard_enforces_single_writer_and_releases_lock(tmp_path: Path) -> None:
    paths, config = _reviewed_project(tmp_path)
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    first = manifest_module._open_output_namespace_guard(paths, processing_path, {})  # type: ignore[arg-type]
    failures: list[BaseException] = []

    def contend() -> None:
        try:
            contender = manifest_module._open_output_namespace_guard(paths, processing_path, {})  # type: ignore[arg-type]
            contender.close()
        except BaseException as error:
            failures.append(error)

    try:
        worker = threading.Thread(target=contend)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
    finally:
        first.close()

    assert len(failures) == 1
    assert isinstance(failures[0], (BlockingIOError, OSError))
    reopened = manifest_module._open_output_namespace_guard(paths, processing_path, {})  # type: ignore[arg-type]
    reopened.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace-handle cleanup contract")
def test_output_namespace_guard_releases_lock_when_post_acquisition_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    real_validate = manifest_module._OutputNamespaceGuard.validate

    def fail_validation(_guard: object) -> None:
        raise ValueError("injected namespace validation failure")

    monkeypatch.setattr(manifest_module._OutputNamespaceGuard, "validate", fail_validation)
    with pytest.raises(ValueError, match="namespace validation"):
        manifest_module._open_output_namespace_guard(paths, processing_path, {})  # type: ignore[arg-type]
    monkeypatch.setattr(manifest_module._OutputNamespaceGuard, "validate", real_validate)

    reopened = manifest_module._open_output_namespace_guard(paths, processing_path, {})  # type: ignore[arg-type]
    reopened.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing contract")
def test_build_manifest_fails_before_durable_writes_when_existing_final_has_writer(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    assert _build_with_fake_audio(paths, config)
    recordings_path = paths.manifests / "recordings.jsonl"  # type: ignore[attr-defined]
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    recordings = []
    for row in read_jsonl(recordings_path):
        reviewed = dict(row)
        reviewed["state"] = "REVIEWED"
        recordings.append(reviewed)
    write_jsonl_atomic(recordings_path, recordings)
    write_jsonl_atomic(
        processing_path,
        (
            row
            for row in read_jsonl(processing_path)
            if not (
                row["previous_state"] == "REVIEWED"
                and row["target_state"] in {"APPROVED", "REJECTED"}
            )
        ),
    )
    segment = read_jsonl(paths.manifests / "segments.jsonl")[0]  # type: ignore[attr-defined]
    final = paths.local_data / Path(*segment["derived_audio_relative_path"].split("/"))  # type: ignore[attr-defined]
    protected = (
        paths.manifests / "segments.jsonl",  # type: ignore[attr-defined]
        processing_path,
        recordings_path,
    )
    before = {path: path.read_bytes() for path in protected}
    shutil.rmtree(paths.segments / ".manifest-staging")  # type: ignore[attr-defined]
    writer = manifest_module._windows_open_handle(
        final,
        desired_access=0x40000000,
        share_mode=0x2,
        flags=0x00200000,
    )
    try:
        with pytest.raises(OSError):
            _build_with_fake_audio(paths, config)
    finally:
        manifest_module._windows_close_handle(writer)

    assert {path: path.read_bytes() for path in protected} == before


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-release contract")
def test_build_manifest_releases_file_and_namespace_guards_after_terminal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)

    def fail_terminal_commit(**_kwargs: object) -> None:
        raise RuntimeError("injected terminal failure")

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        fail_terminal_commit,
    )
    with pytest.raises(RuntimeError, match="terminal failure"):
        _build_with_fake_audio(paths, config)

    segment = read_jsonl(paths.manifests / "segments.jsonl")[0]  # type: ignore[attr-defined]
    final = paths.local_data / Path(*segment["derived_audio_relative_path"].split("/"))  # type: ignore[attr-defined]
    moved_final = final.with_suffix(".released")
    final.rename(moved_final)
    moved_final.rename(final)

    for directory, moved in (
        (paths.manifests, tmp_path / "released-manifests"),  # type: ignore[attr-defined]
        (
            paths.alignments / "runs" / config.digest,  # type: ignore[attr-defined]
            tmp_path / "released-processing",
        ),
    ):
        directory.rename(moved)
        moved.rename(directory)


@pytest.mark.skipif(os.name != "nt", reason="Windows artifact-handle contract")
def test_windows_artifact_guard_rejects_handle_drift_and_releases_invalid_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "mode-root"
    root.mkdir()
    artifact = root / "artifact.flac"
    artifact.write_bytes(b"guarded audio")
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    directory = manifest_module._open_publication_guard(root)
    try:
        with pytest.raises(CorpusFailure, match="published audio artifact"):
            manifest_module._open_artifact_guard(artifact, "0" * 64, directory)
        moved = root / "invalid-open-released.flac"
        artifact.rename(moved)
        moved.rename(artifact)

        guard = manifest_module._open_artifact_guard(artifact, sha256, directory)
        try:
            monkeypatch.setattr(
                manifest_module._PublicationGuard,
                "validate",
                lambda _guard: None,
            )
            monkeypatch.setattr(
                manifest_module,
                "_windows_handle_identity",
                lambda *_args, **_kwargs: (0, 0, 0),
            )
            with pytest.raises(ValueError, match="identity changed"):
                guard.validate()
            monkeypatch.setattr(
                manifest_module,
                "_windows_handle_identity",
                lambda *_args, **_kwargs: guard.identity,
            )
            monkeypatch.setattr(
                manifest_module,
                "_windows_digest_handle",
                lambda _handle: "f" * 64,
            )
            with pytest.raises(CorpusFailure, match="published audio artifact"):
                guard.validate()
        finally:
            guard.close()
    finally:
        directory.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows artifact-handle cleanup contract")
def test_windows_link_hash_failure_and_artifact_guard_close_failure_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mode-root"
    root.mkdir()
    staged = tmp_path / "staged.flac"
    staged.write_bytes(b"staged audio")
    final = root / "final.flac"
    directory = manifest_module._open_publication_guard(root)
    try:
        with pytest.raises(CorpusFailure, match="staged audio artifact"):
            manifest_module._windows_link_from_pinned_directory(
                staged,
                final,
                directory,
                "0" * 64,
            )
        moved = tmp_path / "released-staged.flac"
        staged.rename(moved)
        moved.rename(staged)
    finally:
        directory.close()

    class BrokenArtifactGuard:
        def close(self) -> None:
            raise OSError("injected artifact close failure")

    with pytest.raises(OSError, match="artifact close"):
        manifest_module._close_artifact_guards({"artifact": BrokenArtifactGuard()})  # type: ignore[dict-item]


@pytest.mark.skipif(os.name != "nt", reason="Windows artifact-handle cleanup contract")
def test_pin_existing_artifacts_deduplicates_and_closes_on_conflict(tmp_path: Path) -> None:
    paths = cli.CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    final = paths.segments / "lossless" / "fixture.flac"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"existing audio")
    sha256 = hashlib.sha256(final.read_bytes()).hexdigest()

    class Artifact:
        mode = "lossless"
        relative_path = "derived/corpus-v1/segments/lossless/fixture.flac"

        def __init__(self, digest: str) -> None:
            self.sha256 = digest

    artifact = Artifact(sha256)
    publication = manifest_module._publication_guards(paths, (artifact,))  # type: ignore[arg-type]
    try:
        pinned = manifest_module._pin_existing_artifacts(
            paths,
            (artifact, artifact),  # type: ignore[arg-type]
            publication,
        )
        manifest_module._close_artifact_guards(pinned)
        with pytest.raises(CorpusFailure, match="duplicate final audio artifact conflicts"):
            manifest_module._pin_existing_artifacts(
                paths,
                (artifact, Artifact("f" * 64)),  # type: ignore[arg-type]
                publication,
            )
        moved = final.with_suffix(".released")
        final.rename(moved)
        moved.rename(final)
    finally:
        manifest_module._close_publication_guards(publication)


@pytest.mark.skipif(os.name != "nt", reason="Windows publication-handle contract")
@pytest.mark.parametrize("case", ("missing", "junction"))
def test_windows_publication_guard_rejects_invalid_root(tmp_path: Path, case: str) -> None:
    root = tmp_path / "mode-root"
    if case == "junction":
        outside = tmp_path / "outside"
        outside.mkdir()
        _make_directory_alias(root, outside)

    with pytest.raises((OSError, ValueError), match=r"cannot find|找不到|reparse|access"):
        manifest_module._open_publication_guard(root)


@pytest.mark.skipif(os.name != "nt", reason="Windows publication-handle contract")
def test_windows_publication_guard_detects_lexical_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "mode-root"
    root.mkdir()
    guard = manifest_module._open_publication_guard(root)
    try:
        monkeypatch.setattr(
            manifest_module,
            "_windows_handle_identity",
            lambda *_args, **_kwargs: (0, 0, 0),
        )
        with pytest.raises(ValueError, match="identity changed"):
            guard.validate()
    finally:
        guard.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows publication-handle contract")
def test_windows_publication_rejects_source_target_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "mode-root"
    root.mkdir()
    staged = tmp_path / "staged.flac"
    staged.write_bytes(b"audio")
    final = root / "final.flac"
    guard = manifest_module._open_publication_guard(root)
    identities = iter(((1, 2, 3), (4, 5, 6)))
    try:
        monkeypatch.setattr(
            manifest_module,
            "_windows_handle_identity",
            lambda *_args, **_kwargs: next(identities),
        )
        with pytest.raises(ValueError, match="identity differs"):
            manifest_module._windows_link_from_pinned_directory(staged, final, guard)
    finally:
        guard.close()


def test_publication_guard_helpers_fail_closed_for_unsupported_mode_and_close_error(
    tmp_path: Path,
) -> None:
    paths = cli.CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    source_relative = "raw/spoken/source.wav"
    source_sha256 = "a" * 64
    config_sha256 = "b" * 64
    output_sha256 = "c" * 64
    cache_key = audio_module._cache_identity(
        mode="review",
        source_relative_path=source_relative,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        output_config_sha256=output_sha256,
        ffmpeg_version="ffmpeg-test-1",
        start_seconds=0.0,
        end_seconds=1.0,
    )
    artifact = DerivedAudio(
        "1",
        "review",
        audio_module._artifact_relative_path("review", cache_key),
        "d" * 64,
        source_relative,
        source_sha256,
        config_sha256,
        output_sha256,
        "ffmpeg-test-1",
        cache_key,
        0.0,
        1.0,
        None,
    )
    with pytest.raises(ValueError, match="unsupported audio mode"):
        manifest_module._publication_guards(paths, (artifact,))

    class BrokenGuard:
        def close(self) -> None:
            raise OSError("injected guard close failure")

    with pytest.raises(OSError, match="guard close"):
        manifest_module._close_publication_guards({"lossless": BrokenGuard()})  # type: ignore[dict-item]


def test_build_manifest_extracts_approved_source_clips_and_advances_state(
    tmp_path: Path,
) -> None:
    paths, config = _aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    sample_counts: dict[Path, int] = {}

    def transcode(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        output = Path(command[-1])
        if output.suffix == ".flac":
            start = float(command[command.index("-ss") + 1])
            end = float(command[command.index("-to") + 1])
            sample_counts[output] = round((end - start) * 48_000)
            output.write_bytes(b"fLaCsynthetic" + str(sample_counts[output]).encode("ascii"))
            return CompletedProcess(command, 0, "", "")
        return _audio_command(command, **kwargs)

    def probe(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        target = Path(command[-1])
        samples = sample_counts.get(target)
        if samples is None:
            samples = int(target.read_bytes().removeprefix(b"fLaCsynthetic"))
        payload = {
            "format": {"duration": str(samples / 48_000)},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "flac",
                    "sample_rate": "48000",
                    "channels": 1,
                    "duration_ts": str(samples),
                    "time_base": "1/48000",
                }
            ],
        }
        return CompletedProcess(command, 0, json.dumps(payload), "")

    assert build_manifest_corpus(
        paths,
        config,
        ffmpeg_version="ffmpeg-test-1",
        run_command=transcode,
        probe_command=probe,
        decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
    )

    rows = read_jsonl(paths.manifests / "segments.jsonl")
    assert len(rows) == 6
    assert [(row["text_unit_id"], row["take_index"]) for row in rows] == [
        (f"rec-1-unit-{unit:04d}", take) for unit in range(1, 4) for take in (1, 2)
    ]
    assert all(row["phoneme_timing_status"] == "not_estimated" for row in rows)
    assert all(row["review_event_ids"] for row in rows)
    assert rows[0]["spoken_text"].split()[0] == rows[0]["word_spans"][0]["text"]
    assert all(row["quality_metrics"]["sample_count"] > 0 for row in rows)
    assert all(row["quality_metric_audio_sha256"] != row["derived_audio_sha256"] for row in rows)
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "APPROVED"
    events = read_jsonl(paths.alignments / "runs" / config.digest / "processing-events.jsonl")
    assert events[-1]["previous_state"] == "REVIEWED"
    assert events[-1]["target_state"] == "APPROVED"
    assert build_manifest_corpus(
        paths,
        config,
        ffmpeg_version="ffmpeg-test-1",
        run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal rerun must validate existing clips without extraction")
        ),
    )

    rights_path = paths.manifests / "rights.jsonl"
    rights_rows = read_jsonl(rights_path)
    revoked = dict(rights_rows[0])
    revoked["allow_model_training"] = False
    write_jsonl_atomic(rights_path, (revoked,))
    with pytest.raises(CorpusFailure) as revoked_error:
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert revoked_error.value.code == "RIGHTS_SCOPE_UNCONFIRMED"
    write_jsonl_atomic(rights_path, rights_rows)

    manifest_path = paths.manifests / "segments.jsonl"
    original_manifest = read_jsonl(manifest_path)
    changed_manifest = [dict(row) for row in original_manifest]
    changed_manifest[0]["spoken_text"] = "stale terminal text"
    write_jsonl_atomic(manifest_path, changed_manifest)
    with pytest.raises(ValueError, match=r"manifest|segment|expected"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    write_jsonl_atomic(manifest_path, original_manifest)

    automatic_path = next(
        (paths.alignments / "runs" / config.digest / "rec-1" / "review").glob(  # type: ignore[attr-defined]
            "*/automatic.json"
        )
    )
    automatic_bytes = automatic_path.read_bytes()
    automatic = json.loads(automatic_bytes)
    automatic["takes"][0]["automatic_values"]["segment_end"] -= 0.01
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")
    with pytest.raises(ValueError, match=r"automatic|review|stale"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    automatic_path.write_bytes(automatic_bytes)

    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    processing_rows = read_jsonl(processing_path)
    current_recording = review_module._decode_recording(
        read_jsonl(paths.manifests / "recordings.jsonl")[0]
    )
    reviewed_recording = replace(current_recording, state=CorpusState.REVIEWED)
    _, legacy_event = advance_recording(
        reviewed_recording,
        CorpusState.APPROVED,
        input_sha256s=(
            reviewed_recording.sha256,
            review_module._digest(
                paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json"
            ),
            review_module._digest(
                paths.alignments / "runs" / config.digest / "rec-1" / "alignment.json"
            ),
            review_module._digest(paths.manifests / "review.jsonl"),
            review_module._digest(manifest_path),
        ),
        config_sha256=config.digest,
        tool_versions=("approved-manifest-v1", "ffmpeg-test-1"),
        started_at="1970-01-01T00:00:00+00:00",
        finished_at="1970-01-01T00:00:00+00:00",
        result="success",
    )
    write_jsonl_atomic(processing_path, (*processing_rows[:-1], legacy_event.to_dict()))
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (reviewed_recording.to_dict(),))
    assert build_manifest_corpus(
        paths,
        config,
        ffmpeg_version="ffmpeg-test-1",
        run_command=transcode,
        probe_command=probe,
        decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
    )
    assert read_jsonl(processing_path)[-1] == legacy_event.to_dict()

    terminal_events = read_jsonl(processing_path)
    write_jsonl_atomic(processing_path, (*terminal_events, terminal_events[-1]))
    with pytest.raises(ValueError, match=r"duplicate|history"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    write_jsonl_atomic(processing_path, terminal_events)

    clip = paths.resolve_local(rows[0]["derived_audio_relative_path"])
    clip.write_bytes(b"tampered final cache")
    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_build_manifest_rejects_tampered_raw_before_writing_any_final_clip(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    (paths.raw_spoken / "rec-1.wav").write_bytes(b"tampered after inventory")  # type: ignore[attr-defined]

    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("audio extraction must not start")
            ),
        )

    assert error.value.code == "INVENTORY_HASH_MISMATCH"
    assert not tuple((paths.segments / "lossless").glob("*.flac"))  # type: ignore[attr-defined]


@pytest.mark.parametrize("failure_point", ("later-take", "later-recording", "quality"))
def test_build_manifest_stages_all_audio_before_publishing_final_clips(
    tmp_path: Path, failure_point: str
) -> None:
    if failure_point == "later-recording":
        paths, config = _two_recording_aligned_project(tmp_path)
        _write_rights(paths)
        groups = export_review_bundle(paths, config)
        for group in groups:
            _set_decisions(group)
        assert import_review_bundle(paths, config)
    else:
        paths, config = _reviewed_project(tmp_path)
    before_lossless = set((paths.segments / "lossless").rglob("*.flac"))  # type: ignore[attr-defined]
    before_candidates = set((paths.segments / "candidates").rglob("*.wav"))  # type: ignore[attr-defined]
    sample_counts: dict[Path, int] = {}
    lossless_calls = 0

    def transcode(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        nonlocal lossless_calls
        output = Path(command[-1])
        if output.suffix == ".flac":
            lossless_calls += 1
            if (failure_point == "later-take" and lossless_calls == 2) or (
                failure_point == "later-recording" and lossless_calls == 7
            ):
                raise RuntimeError(f"injected {failure_point} extraction failure")
            start = float(command[command.index("-ss") + 1])
            end = float(command[command.index("-to") + 1])
            sample_counts[output] = round((end - start) * 48_000)
            output.write_bytes(b"fLaCsynthetic" + str(sample_counts[output]).encode("ascii"))
            return CompletedProcess(command, 0, "", "")
        if failure_point == "quality" and ".manifest-staging" in output.parts:
            raise RuntimeError("injected quality extraction failure")
        return _audio_command(command, **kwargs)

    def probe(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        target = Path(command[-1])
        samples = sample_counts.get(target)
        if samples is None:
            samples = int(target.read_bytes().removeprefix(b"fLaCsynthetic"))
        return CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "format": {"duration": str(samples / 48_000)},
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "flac",
                            "sample_rate": "48000",
                            "channels": 1,
                            "duration_ts": str(samples),
                            "time_base": "1/48000",
                        }
                    ],
                }
            ),
            "",
        )

    with pytest.raises(RuntimeError, match=failure_point):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=transcode,
            probe_command=probe,
            decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
        )

    assert set((paths.segments / "lossless").rglob("*.flac")) == before_lossless  # type: ignore[attr-defined]
    assert set((paths.segments / "candidates").rglob("*.wav")) == before_candidates  # type: ignore[attr-defined]


def test_build_manifest_recovers_after_manifest_replace_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _reviewed_project(tmp_path)
    sample_counts: dict[Path, int] = {}

    def transcode(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        output = Path(command[-1])
        if output.suffix == ".flac":
            start = float(command[command.index("-ss") + 1])
            end = float(command[command.index("-to") + 1])
            sample_counts[output] = round((end - start) * 48_000)
            output.write_bytes(b"fLaCsynthetic" + str(sample_counts[output]).encode("ascii"))
            return CompletedProcess(command, 0, "", "")
        return _audio_command(command, **kwargs)

    def probe(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        target = Path(command[-1])
        samples = sample_counts.get(target)
        if samples is None:
            samples = int(target.read_bytes().removeprefix(b"fLaCsynthetic"))
        return CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "format": {"duration": str(samples / 48_000)},
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "flac",
                            "sample_rate": "48000",
                            "channels": 1,
                            "duration_ts": str(samples),
                            "time_base": "1/48000",
                        }
                    ],
                }
            ),
            "",
        )

    real_write = manifest_module.write_jsonl_atomic

    def fail_manifest(path: Path, rows: object) -> None:
        if path == paths.manifests / "segments.jsonl":  # type: ignore[attr-defined]
            raise OSError("injected manifest replace failure")
        real_write(path, rows)  # type: ignore[arg-type]

    monkeypatch.setattr(manifest_module, "write_jsonl_atomic", fail_manifest)
    with pytest.raises(OSError, match="manifest replace"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=transcode,
            probe_command=probe,
            decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
        )
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "REVIEWED"  # type: ignore[attr-defined]
    monkeypatch.setattr(manifest_module, "write_jsonl_atomic", real_write)

    assert build_manifest_corpus(
        paths,  # type: ignore[arg-type]
        config,  # type: ignore[arg-type]
        ffmpeg_version="ffmpeg-test-1",
        run_command=transcode,
        probe_command=probe,
        decode_command=lambda command, **_kwargs: CompletedProcess(command, 0, "", ""),
    )
    events = read_jsonl(
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    event_ids = [row["event_id"] for row in events]
    assert len(event_ids) == len(set(event_ids))


@pytest.mark.parametrize("checkpoint", ("events", "recordings"))
def test_build_manifest_recovers_batch_commit_checkpoints_without_duplicate_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group, decision="rejected", reason="fixture rejection")
    assert import_review_bundle(paths, config)
    recordings_path = paths.manifests / "recordings.jsonl"
    events_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    before_events = read_jsonl(events_path)
    real_replace = store_module.os.replace

    def fail_checkpoint(source: Path, destination: Path) -> None:
        target = events_path if checkpoint == "events" else recordings_path
        if Path(destination) == target:
            raise OSError(f"injected {checkpoint} replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(store_module.os, "replace", fail_checkpoint)
    expected_error = OSError if checkpoint == "events" else RuntimeError
    expected_message = checkpoint if checkpoint == "events" else "recovery required"
    with pytest.raises(expected_error, match=expected_message):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert all(row["state"] == "REVIEWED" for row in read_jsonl(recordings_path))
    if checkpoint == "events":
        assert read_jsonl(events_path) == before_events
    monkeypatch.setattr(store_module.os, "replace", real_replace)

    assert build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    terminal_events = read_jsonl(events_path)
    event_ids = [row["event_id"] for row in terminal_events]
    assert len(event_ids) == len(set(event_ids))
    assert all(row["state"] == "REJECTED" for row in read_jsonl(recordings_path))


def _materialize_mixed_legacy_terminal_state(
    paths: object, config: CorpusConfig, version: str
) -> tuple[str, str]:
    recordings_path = paths.manifests / "recordings.jsonl"  # type: ignore[attr-defined]
    events_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    recordings = list(read_jsonl(recordings_path))
    rec1 = review_module._decode_recording(recordings[0])
    rec2 = review_module._decode_recording(recordings[1])
    terminal_rows = [
        ProcessingEvent.from_dict(row)
        for row in read_jsonl(events_path)
        if row["previous_state"] == "REVIEWED"
    ]
    rec1_current = next(event for event in terminal_rows if event.recording_id == rec1.recording_id)
    if version == "v1":
        inputs = (
            rec1_current.input_sha256s[0],
            rec1_current.input_sha256s[2],
            rec1_current.input_sha256s[3],
            rec1_current.input_sha256s[4],
            rec1_current.input_sha256s[5],
            rec1_current.input_sha256s[6],
            rec1_current.input_sha256s[7],
        )
    else:
        inputs = (
            rec1_current.input_sha256s[0],
            rec1_current.input_sha256s[4],
            rec1_current.input_sha256s[5],
            rec1_current.input_sha256s[6],
            rec1_current.input_sha256s[7],
        )
    _, legacy = advance_recording(
        replace(rec1, state=CorpusState.REVIEWED),
        rec1.state,
        input_sha256s=inputs,
        config_sha256=rec1_current.config_sha256,
        tool_versions=("approved-manifest-v1", rec1_current.tool_versions[1]),
        started_at=rec1_current.started_at,
        finished_at=rec1_current.finished_at,
        result=rec1_current.result,
    )
    nonterminal_rows = tuple(
        row for row in read_jsonl(events_path) if row["previous_state"] != "REVIEWED"
    )
    write_jsonl_atomic(events_path, (*nonterminal_rows, legacy.to_dict()))
    recordings[1] = replace(rec2, state=CorpusState.REVIEWED).to_dict()
    write_jsonl_atomic(recordings_path, recordings)
    return legacy.event_id, rec2.recording_id


@pytest.mark.parametrize("old_version", ("v1", "legacy"))
@pytest.mark.parametrize("rec1_decision", ("rejected", "approved"))
def test_build_manifest_recovers_mixed_legacy_subset_and_appends_only_missing_current_event(
    tmp_path: Path,
    rec1_decision: str,
    old_version: str,
) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        recording_id = json.loads((group / "automatic.json").read_text(encoding="utf-8"))[
            "recording_id"
        ]
        decision = rec1_decision if recording_id == "rec-1" else "rejected"
        _set_decisions(group, decision=decision, reason=f"fixture {decision}")
    assert import_review_bundle(paths, config)
    builder = _build_with_fake_audio if rec1_decision == "approved" else build_manifest_corpus
    if builder is build_manifest_corpus:
        assert builder(paths, config, ffmpeg_version="ffmpeg-test-1")  # type: ignore[arg-type]
    else:
        assert builder(paths, config)
    events_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    recordings_path = paths.manifests / "recordings.jsonl"
    legacy_id, missing_recording_id = _materialize_mixed_legacy_terminal_state(
        paths, config, old_version
    )
    mixed_events = events_path.read_bytes()

    if builder is build_manifest_corpus:
        assert builder(paths, config, ffmpeg_version="ffmpeg-test-1")  # type: ignore[arg-type]
    else:
        assert builder(paths, config)

    rows = read_jsonl(events_path)
    terminals = [
        ProcessingEvent.from_dict(row) for row in rows if row["previous_state"] == "REVIEWED"
    ]
    assert [event.event_id for event in terminals[:-1]] == [legacy_id]
    assert terminals[-1].recording_id == missing_recording_id
    assert len(terminals[-1].input_sha256s) == 8
    assert events_path.read_bytes().startswith(mixed_events)
    assert all(row["state"] in {"APPROVED", "REJECTED"} for row in read_jsonl(recordings_path))
    durable = {path: path.read_bytes() for path in (events_path, recordings_path)}

    if builder is build_manifest_corpus:
        assert builder(paths, config, ffmpeg_version="ffmpeg-test-1")  # type: ignore[arg-type]
    else:
        assert builder(paths, config)
    assert {path: path.read_bytes() for path in durable} == durable


@pytest.mark.parametrize("conflict", ("legacy-input", "manifest-row", "terminal-state"))
def test_build_manifest_mixed_legacy_recovery_conflicts_fail_without_writes(
    tmp_path: Path,
    conflict: str,
) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        recording_id = json.loads((group / "automatic.json").read_text(encoding="utf-8"))[
            "recording_id"
        ]
        decision = "approved" if recording_id == "rec-1" else "rejected"
        _set_decisions(group, decision=decision, reason=f"fixture {decision}")
    assert import_review_bundle(paths, config)
    assert _build_with_fake_audio(paths, config)
    _materialize_mixed_legacy_terminal_state(paths, config, "legacy")
    events_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    recordings_path = paths.manifests / "recordings.jsonl"
    manifest_path = paths.manifests / "segments.jsonl"
    if conflict == "legacy-input":
        events = list(read_jsonl(events_path))
        terminal = next(row for row in events if row["previous_state"] == "REVIEWED")
        terminal["input_sha256s"][1] = "f" * 64
        write_jsonl_atomic(events_path, events)
    elif conflict == "manifest-row":
        segments = list(read_jsonl(manifest_path))
        segments[0]["spoken_text"] += " corrupt"
        write_jsonl_atomic(manifest_path, segments)
    else:
        recordings = list(read_jsonl(recordings_path))
        recordings[0]["state"] = "REJECTED"
        write_jsonl_atomic(recordings_path, recordings)
    protected = (
        paths.manifests / "review.jsonl",
        manifest_path,
        events_path,
        recordings_path,
    )
    before = {path: path.read_bytes() for path in protected}
    clips_before = {path: path.read_bytes() for path in paths.segments.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match=r"event|manifest|recording|state|evidence|history"):
        _build_with_fake_audio(paths, config)

    assert {path: path.read_bytes() for path in protected} == before
    assert {path: path.read_bytes() for path in paths.segments.rglob("*") if path.is_file()} == (
        clips_before
    )


def test_build_manifest_classifies_audit_ahead_before_any_writable_review_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group, decision="rejected", reason="fixture rejection")
    assert import_review_bundle(paths, config)
    recordings_path = paths.manifests / "recordings.jsonl"
    events_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    real_replace = store_module.os.replace

    def fail_recordings(source: Path, destination: Path) -> None:
        if Path(destination) == recordings_path:
            raise OSError("injected recordings replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(store_module.os, "replace", fail_recordings)
    with pytest.raises(RuntimeError, match="recovery required"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    monkeypatch.setattr(store_module.os, "replace", real_replace)
    assert all(row["state"] == "REVIEWED" for row in read_jsonl(recordings_path))

    decision_path = groups[0] / "decision.json"
    original_decision = decision_path.read_bytes()
    changed = json.loads(original_decision)
    changed["decisions"][0].update(
        decision="approved",
        reason="stale changed decision",
        reviewer="owner",
        reviewed_at="2026-07-20T00:00:00+08:00",
    )
    decision_path.write_text(json.dumps(changed), encoding="utf-8")
    protected = (
        paths.manifests / "review.jsonl",
        paths.manifests / "segments.jsonl",
        events_path,
        recordings_path,
    )
    before = {path: path.read_bytes() for path in protected}
    clips_before = {path: path.read_bytes() for path in paths.segments.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match=r"review|decision|durable|replay|processing"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")

    assert {path: path.read_bytes() for path in protected} == before
    assert {path: path.read_bytes() for path in paths.segments.rglob("*") if path.is_file()} == (
        clips_before
    )
    decision_path.write_bytes(original_decision)

    transcripts_path = paths.manifests / "transcripts.jsonl"
    original_transcripts = transcripts_path.read_bytes()
    transcripts = list(read_jsonl(transcripts_path))
    transcripts[0]["pronunciation_plan"]["tokens"][0]["ipa"] += "x"
    write_jsonl_atomic(transcripts_path, transcripts)
    before = {path: path.read_bytes() for path in protected}
    clips_before = {path: path.read_bytes() for path in paths.segments.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match=r"transcript|pronunciation|review|processing"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")

    assert {path: path.read_bytes() for path in protected} == before
    assert {path: path.read_bytes() for path in paths.segments.rglob("*") if path.is_file()} == (
        clips_before
    )
    transcripts_path.write_bytes(original_transcripts)
    assert build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    terminal_events = read_jsonl(events_path)
    assert len({row["event_id"] for row in terminal_events}) == len(terminal_events)

    mixed = list(read_jsonl(recordings_path))
    mixed[0]["state"] = "REVIEWED"
    write_jsonl_atomic(recordings_path, mixed)
    assert build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert read_jsonl(events_path) == terminal_events
    assert all(row["state"] == "REJECTED" for row in read_jsonl(recordings_path))


def test_build_manifest_revalidates_review_snapshot_instead_of_trusting_events(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    automatic = next(
        (paths.alignments / "runs" / config.digest / "rec-1" / "review").glob(  # type: ignore[attr-defined]
            "*/automatic.json"
        )
    )
    raw = json.loads(automatic.read_text(encoding="utf-8"))
    raw["takes"][0]["automatic_values"]["segment_end"] -= 0.01
    automatic.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=r"automatic|review"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("clip extraction must not start")
            ),
        )


def test_build_manifest_accepts_strictly_bound_human_pairing_correction(
    tmp_path: Path,
) -> None:
    paths, config, backend, _ = _corrected_pairing_project(tmp_path)
    assert cli.pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("corrected pairing must use validated caches")
        ),
    )
    assert cli.align_corpus(paths, config, backend)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group, decision="rejected", reason="not selected for fixture")
    assert import_review_bundle(paths, config)
    _write_rights(paths)

    assert build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")
    assert read_jsonl(paths.manifests / "segments.jsonl") == ()
    assert read_jsonl(paths.manifests / "recordings.jsonl")[0]["state"] == "REJECTED"


def test_build_manifest_rejects_warning_in_approved_unit_before_extraction(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    transcript_path = paths.manifests / "transcripts.jsonl"  # type: ignore[attr-defined]
    transcript = read_jsonl(transcript_path)[0]
    transcript["pronunciation_plan"]["tokens"][0]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    transcript["pronunciation_plan"]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    write_jsonl_atomic(transcript_path, (transcript,))

    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("clip extraction must not start")
            ),
        )

    assert error.value.code == "PRONUNCIATION_NEEDS_REVIEW"


def test_build_manifest_preflights_all_units_before_any_final_clip(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    transcript_path = paths.manifests / "transcripts.jsonl"  # type: ignore[attr-defined]
    transcript = read_jsonl(transcript_path)[0]
    transcript["pronunciation_plan"]["tokens"][-1]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    transcript["pronunciation_plan"]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    write_jsonl_atomic(transcript_path, (transcript,))

    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("all unit inputs must be validated before extraction")
            ),
        )

    assert error.value.code == "PRONUNCIATION_NEEDS_REVIEW"
    assert not tuple((paths.segments / "lossless").glob("*.flac"))  # type: ignore[attr-defined]


def test_build_manifest_preflights_every_recording_before_any_final_clip(
    tmp_path: Path,
) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    transcript_path = paths.manifests / "transcripts.jsonl"
    transcripts = list(read_jsonl(transcript_path))
    transcripts[1]["pronunciation_plan"]["tokens"][-1]["warning_codes"] = [
        "PRONUNCIATION_NEEDS_REVIEW"
    ]
    transcripts[1]["pronunciation_plan"]["warning_codes"] = ["PRONUNCIATION_NEEDS_REVIEW"]
    write_jsonl_atomic(transcript_path, transcripts)

    with pytest.raises(CorpusFailure) as error:
        build_manifest_corpus(
            paths,
            config,
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("all recordings must be preflighted before extraction")
            ),
        )

    assert error.value.code == "PRONUNCIATION_NEEDS_REVIEW"
    assert not tuple((paths.segments / "lossless").glob("*.flac"))


def test_build_manifest_rejects_duplicate_transcript_source_identity(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    transcript_path = paths.manifests / "transcripts.jsonl"  # type: ignore[attr-defined]
    transcript = read_jsonl(transcript_path)[0]
    transcript["source_candidates"].append(dict(transcript["source_candidates"][0]))
    write_jsonl_atomic(transcript_path, (transcript,))

    with pytest.raises(ValueError, match=r"source candidate|source_candidates"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("clip extraction must not start")
            ),
        )


@pytest.mark.parametrize("case", ("types", "version", "ffmpeg"))
def test_build_manifest_rejects_invalid_api_contract(tmp_path: Path, case: str) -> None:
    paths = cli.CorpusPaths.from_project_root(tmp_path)
    config = CorpusConfig({"schema_version": "1", "corpus_version": "corpus-v1"}, "a" * 64)
    with pytest.raises((CorpusFailure, TypeError, ValueError)):
        if case == "types":
            build_manifest_corpus(object(), config, ffmpeg_version="test")  # type: ignore[arg-type]
        elif case == "version":
            bad = CorpusConfig({"schema_version": "2", "corpus_version": "corpus-v2"}, "a" * 64)
            build_manifest_corpus(paths, bad, ffmpeg_version="test")
        else:
            build_manifest_corpus(paths, config, ffmpeg_version="")


def test_build_manifest_rejects_duplicate_rights_and_processing_identities(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    rights_path = paths.manifests / "rights.jsonl"  # type: ignore[attr-defined]
    rights = read_jsonl(rights_path)
    write_jsonl_atomic(rights_path, (*rights, rights[0]))
    with pytest.raises(ValueError, match="duplicate rights_id"):
        build_manifest_corpus(paths, config, ffmpeg_version="ffmpeg-test-1")  # type: ignore[arg-type]

    write_jsonl_atomic(rights_path, rights)
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    events = read_jsonl(processing_path)
    write_jsonl_atomic(processing_path, (*events, events[-1]))
    with pytest.raises(ValueError, match=r"duplicate (identities|event_id)"):
        build_manifest_corpus(
            paths,  # type: ignore[arg-type]
            config,
            ffmpeg_version="ffmpeg-test-1",
            run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("processing preflight must precede extraction")
            ),
        )


def _direct_builder_inputs(paths: object, config: CorpusConfig) -> dict[str, object]:
    recording = review_module._decode_recording(
        read_jsonl(paths.manifests / "recordings.jsonl")[0]  # type: ignore[attr-defined]
    )
    transcript = read_jsonl(paths.manifests / "transcripts.jsonl")[0]  # type: ignore[attr-defined]
    units = cli._decode_spoken_units(transcript, recording.recording_id)
    run_directory = (
        paths.alignments / "runs" / config.digest / recording.recording_id  # type: ignore[attr-defined]
    )
    pairing = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
    alignment = read_jsonl(run_directory / "alignment.json")[0]
    alignments = {
        (row["repetition_group_id"], row["take_index"]): result_from_dict(row["alignment_result"])
        for row in alignment["takes"]
    }
    segmentation = read_jsonl(run_directory / "segmentation.json")[0]
    rights = {
        row["rights_id"]: RightsRecord.from_dict(row)
        for row in read_jsonl(paths.manifests / "rights.jsonl")  # type: ignore[attr-defined]
    }
    events = tuple(
        ReviewEvent.from_dict(row)
        for row in read_jsonl(paths.manifests / "review.jsonl")  # type: ignore[attr-defined]
    )
    return {
        "recording": recording,
        "transcript": transcript,
        "units": units,
        "rights": rights,
        "pairing": pairing,
        "alignments": alignments,
        "review_events": events,
        "analysis_audio": DerivedAudio.from_dict(segmentation["analysis_audio"]),
        "paths": paths,
        "config": config,
        "ffmpeg_version": "ffmpeg-test-1",
        "_validate_only": True,
    }


def test_processing_history_rejects_malformed_persisted_segmentation_vad(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    inputs = _direct_builder_inputs(paths, config)  # type: ignore[arg-type]
    recording = inputs["recording"]
    run_directory = (
        paths.alignments / "runs" / config.digest / recording.recording_id  # type: ignore[attr-defined,union-attr]
    )
    pairing_path = run_directory / "pairing.json"
    alignment_path = run_directory / "alignment.json"
    context = manifest_module._ManifestContext(
        0,
        recording,
        inputs["transcript"],
        inputs["units"],
        inputs["pairing"],
        inputs["alignments"],
        inputs["analysis_audio"],
        hashlib.sha256(pairing_path.read_bytes()).hexdigest(),
        hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
        run_directory,
    )
    segmentation_path = run_directory / "segmentation.json"
    segmentation = dict(read_jsonl(segmentation_path)[0])
    segmentation["vad"] = []
    write_jsonl_atomic(segmentation_path, (segmentation,))
    processing_rows = read_jsonl(run_directory.parent / "processing-events.jsonl")
    review_path = paths.manifests / "review.jsonl"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="segmentation VAD"):
        manifest_module._validate_processing_history(
            processing_rows,
            context,
            review_path,
            inputs["review_events"],
            config,
            paths,
        )


@pytest.mark.parametrize(
    "case",
    (
        "missing-rights",
        "rights-scope",
        "pairing",
        "analysis",
        "duplicate-unit",
        "duplicate-event",
        "event-id",
        "outcome",
        "unit",
        "alignment",
    ),
)
def test_build_approved_segments_rejects_unbound_inputs(tmp_path: Path, case: str) -> None:
    paths, config = _reviewed_project(tmp_path)
    values = _direct_builder_inputs(paths, config)
    if case == "missing-rights":
        values["rights"] = {}
    elif case == "rights-scope":
        rights = values["rights"]
        assert isinstance(rights, dict)
        item = next(iter(rights.values()))
        rights[item.rights_id] = replace(item, allow_model_training=False)
    elif case == "pairing":
        values["pairing"] = replace(values["pairing"], recording_id="other", integrity_sha256="")  # type: ignore[arg-type]
    elif case == "analysis":
        pairing = values["pairing"]
        values["pairing"] = replace(
            pairing,
            analysis_audio_sha256="0" * 64,
            integrity_sha256="",  # type: ignore[arg-type]
        )
    elif case == "duplicate-unit":
        units = values["units"]
        values["units"] = (units[0], units[0], *units[1:])  # type: ignore[index]
    elif case == "duplicate-event":
        events = values["review_events"]
        values["review_events"] = (*events, events[0])  # type: ignore[misc,index]
    elif case == "event-id":
        events = values["review_events"]
        values["review_events"] = (replace(events[0], review_event_id="review-bad"), *events[1:])  # type: ignore[index]
    elif case == "outcome":
        pairing = values["pairing"]
        first = replace(
            pairing.groups[0], status="review", issue_code="TAKE_COUNT_MISMATCH", group=None
        )  # type: ignore[union-attr]
        values["pairing"] = replace(
            pairing,
            groups=(first, *pairing.groups[1:]),
            integrity_sha256="",  # type: ignore[union-attr]
        )
    elif case == "unit":
        values["units"] = values["units"][1:]  # type: ignore[index]
    else:
        alignments = dict(values["alignments"])  # type: ignore[arg-type]
        alignments.pop(next(iter(alignments)))
        values["alignments"] = alignments
    with pytest.raises((CorpusFailure, TypeError, ValueError)):
        build_approved_segments(**values)  # type: ignore[arg-type]


def test_build_approved_segments_rejects_extractor_provenance_drift(tmp_path: Path) -> None:
    paths, config = _reviewed_project(tmp_path)
    values = _direct_builder_inputs(paths, config)
    values["_validate_only"] = False
    analysis = values["analysis_audio"]
    values["extract_lossless"] = lambda *_args, **_kwargs: analysis
    values["extract_analysis"] = lambda *_args, **_kwargs: analysis

    with pytest.raises(ValueError, match="lossless derived clip provenance"):
        build_approved_segments(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("drift", ("lossless-interval", "quality-interval"))
def test_build_approved_segments_rejects_structurally_valid_wrong_extractor_interval(
    tmp_path: Path, drift: str
) -> None:
    paths, config = _reviewed_project(tmp_path)
    values = _direct_builder_inputs(paths, config)
    values["_validate_only"] = False

    def lossless(
        recording: object,
        _paths: object,
        start: float,
        end: float,
        *,
        ffmpeg_version: str,
        **_kwargs: object,
    ) -> DerivedAudio:
        output_config = audio_module._canonical_output_config(
            {
                "codec": "flac",
                "compression_level": 5,
                "channels": recording.metadata.channels,  # type: ignore[attr-defined]
                "sample_rate": recording.metadata.sample_rate,  # type: ignore[attr-defined]
            }
        )
        if drift == "lossless-interval":
            start += 0.01
        key = audio_module._cache_identity(
            mode="lossless",
            source_relative_path=recording.relative_path,  # type: ignore[attr-defined]
            source_sha256=recording.sha256,  # type: ignore[attr-defined]
            config_sha256=output_config,
            output_config_sha256=output_config,
            ffmpeg_version=ffmpeg_version,
            start_seconds=start,
            end_seconds=end,
        )
        return DerivedAudio(
            "1",
            "lossless",
            audio_module._artifact_relative_path("lossless", key),
            "d" * 64,
            recording.relative_path,  # type: ignore[attr-defined]
            recording.sha256,  # type: ignore[attr-defined]
            output_config,
            output_config,
            ffmpeg_version,
            key,
            start,
            end,
            None,
        )

    def quality(
        analysis: DerivedAudio,
        _paths: object,
        start: float,
        end: float,
        *,
        ffmpeg_version: str,
        **_kwargs: object,
    ) -> DerivedAudio:
        output_config = audio_module._canonical_output_config(
            {"codec": "pcm_s16le", "channels": 1, "sample_rate": 16000}
        )
        if drift == "quality-interval":
            end -= 0.01
        key = audio_module._cache_identity(
            mode="candidate",
            source_relative_path=analysis.relative_path,
            source_sha256=analysis.sha256,
            config_sha256=analysis.config_sha256,
            output_config_sha256=output_config,
            ffmpeg_version=ffmpeg_version,
            start_seconds=start,
            end_seconds=end,
        )
        return DerivedAudio(
            "1",
            "candidate",
            audio_module._artifact_relative_path("candidate", key),
            "e" * 64,
            analysis.relative_path,
            analysis.sha256,
            analysis.config_sha256,
            output_config,
            ffmpeg_version,
            key,
            start,
            end,
            PcmMetrics(100, 0.1, 0.1, 0.0, 0.0),
        )

    values["extract_lossless"] = lossless
    values["extract_analysis"] = quality
    with pytest.raises(ValueError, match=r"provenance|quality|lossless"):
        build_approved_segments(**values)  # type: ignore[arg-type]


def test_build_approved_segments_rejects_reviewed_word_text_stale_against_transcript(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    values = _direct_builder_inputs(paths, config)
    events = values["review_events"]
    assert isinstance(events, tuple)
    first = events[0]
    assert isinstance(first, ReviewEvent)
    alignment = next(iter(values["alignments"].values()))  # type: ignore[union-attr]
    before = alignment.words[0].text
    provisional = ReviewEvent(
        "1",
        "pending",
        first.entity_id,
        "word:0:text",
        before,
        "Mater",
        "manual transcription correction",
        "owner",
        "2026-07-19T11:59:00+08:00",
    )
    correction = replace(
        provisional,
        review_event_id=review_module._event_identity(provisional),
    )
    values["review_events"] = (correction, *events)

    with pytest.raises(ValueError, match=r"stale|text|transcript|pronunciation"):
        build_approved_segments(**values)  # type: ignore[arg-type]


def test_build_manifest_takes_transaction_lock_before_loading_rights_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    rights_path = paths.manifests / "rights.jsonl"  # type: ignore[attr-defined]
    recordings_path = paths.manifests / "recordings.jsonl"  # type: ignore[attr-defined]
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    before = {
        recordings_path: recordings_path.read_bytes(),
        processing_path: processing_path.read_bytes(),
    }
    real_open = manifest_module._open_output_namespace_guard

    def revoke_then_open(*args: object, **kwargs: object) -> object:
        rows = list(read_jsonl(rights_path))
        rows[0]["allow_model_training"] = False
        write_jsonl_atomic(rights_path, rows)
        return real_open(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(manifest_module, "_open_output_namespace_guard", revoke_then_open)

    with pytest.raises(CorpusFailure) as error:
        _build_with_fake_audio(paths, config)

    assert error.value.code == "RIGHTS_SCOPE_UNCONFIRMED"
    assert not (paths.manifests / "segments.jsonl").exists()  # type: ignore[attr-defined]
    assert {path: path.read_bytes() for path in before} == before


def test_build_manifest_fails_on_busy_writer_lock_before_any_input_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    first = manifest_module._open_output_namespace_guard(paths, processing_path, {})  # type: ignore[arg-type]
    monkeypatch.setattr(
        manifest_module,
        "_load_selection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("input was loaded before acquiring the writer lock")
        ),
    )
    failures: list[BaseException] = []

    def contend() -> None:
        try:
            _build_with_fake_audio(paths, config)
        except BaseException as error:
            failures.append(error)

    try:
        worker = threading.Thread(target=contend)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
    finally:
        first.close()

    assert len(failures) == 1
    assert isinstance(failures[0], OSError)
    assert not (paths.manifests / "segments.jsonl").exists()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "drift",
    (
        "selection",
        "recordings",
        "rights",
        "transcripts",
        "review",
        "processing",
        "segmentation",
        "pairing",
        "alignment",
    ),
)
def test_build_manifest_rejects_authoritative_snapshot_drift_before_segments_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    recordings_path = paths.manifests / "recordings.jsonl"  # type: ignore[attr-defined]
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    terminal_before = tuple(
        row for row in read_jsonl(processing_path) if row["previous_state"] == "REVIEWED"
    )
    real_publish = manifest_module._publish_staged_audio

    def publish_then_drift(*args: object, **kwargs: object) -> object:
        published = real_publish(*args, **kwargs)  # type: ignore[arg-type]
        try:
            if drift == "selection":
                path = paths.manifests / "pilot-selection.json"  # type: ignore[attr-defined]
                rows = list(read_jsonl(path))
                rows[0]["strategy"] += "-mutatum"
                write_jsonl_atomic(path, rows)
            elif drift == "recordings":
                rows = list(read_jsonl(recordings_path))
                rows[0]["notes"] += " mutatum"
                write_jsonl_atomic(recordings_path, rows)
            elif drift == "rights":
                path = paths.manifests / "rights.jsonl"  # type: ignore[attr-defined]
                rows = list(read_jsonl(path))
                rows[0]["allow_model_training"] = False
                write_jsonl_atomic(path, rows)
            elif drift == "transcripts":
                path = paths.manifests / "transcripts.jsonl"  # type: ignore[attr-defined]
                rows = list(read_jsonl(path))
                rows[0]["normalized_text"] += " mutatum"
                write_jsonl_atomic(path, rows)
            elif drift == "review":
                path = paths.manifests / "review.jsonl"  # type: ignore[attr-defined]
                rows = read_jsonl(path)
                write_jsonl_atomic(path, (*rows, rows[-1]))
            elif drift == "processing":
                recording = review_module._decode_recording(read_jsonl(recordings_path)[0])
                discovered = replace(
                    recording,
                    recording_id="rec-unrelated",
                    state=CorpusState.DISCOVERED,
                )
                _, unrelated = advance_recording(
                    discovered,
                    CorpusState.INVENTORIED,
                    input_sha256s=(discovered.sha256,),
                    config_sha256=config.digest,
                    tool_versions=("fixture=1",),
                    started_at="2026-07-21T00:00:00+08:00",
                    finished_at="2026-07-21T00:00:01+08:00",
                    result="success",
                )
                write_jsonl_atomic(
                    processing_path,
                    (*read_jsonl(processing_path), unrelated.to_dict()),
                )
            else:
                path = (
                    paths.alignments  # type: ignore[attr-defined]
                    / "runs"
                    / config.digest
                    / "rec-1"
                    / f"{drift}.json"
                )
                rows = list(read_jsonl(path))
                rows[0]["schema_version"] = "mutatum"
                write_jsonl_atomic(path, rows)
        except Exception:
            manifest_module._close_artifact_guards(published)  # type: ignore[arg-type]
            raise
        return published

    monkeypatch.setattr(manifest_module, "_publish_staged_audio", publish_then_drift)

    with pytest.raises((CorpusFailure, OSError, ValueError)):
        _build_with_fake_audio(paths, config)

    assert not (paths.manifests / "segments.jsonl").exists()  # type: ignore[attr-defined]
    assert all(row["state"] == "REVIEWED" for row in read_jsonl(recordings_path))
    terminal_after = tuple(
        row for row in read_jsonl(processing_path) if row["previous_state"] == "REVIEWED"
    )
    assert terminal_after == terminal_before


def test_build_manifest_pins_segments_file_before_terminal_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    manifest_path = paths.manifests / "segments.jsonl"  # type: ignore[attr-defined]
    recordings_path = paths.manifests / "recordings.jsonl"  # type: ignore[attr-defined]
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    real_persist = manifest_module.persist_recording_transitions

    def replace_segments_then_persist(**kwargs: object) -> None:
        rows = list(read_jsonl(manifest_path))
        rows[0]["spoken_text"] += " mutatum"
        write_jsonl_atomic(manifest_path, rows)
        real_persist(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        replace_segments_then_persist,
    )

    with pytest.raises((CorpusFailure, OSError, ValueError)):
        _build_with_fake_audio(paths, config)

    assert all(row["state"] == "REVIEWED" for row in read_jsonl(recordings_path))
    assert not any(row["previous_state"] == "REVIEWED" for row in read_jsonl(processing_path))


@pytest.mark.parametrize("drift", ("events", "recordings"))
def test_build_manifest_detects_terminal_file_drift_immediately_after_atomic_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    recordings_path = paths.manifests / "recordings.jsonl"  # type: ignore[attr-defined]
    events_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    real_write = store_module.write_jsonl_atomic
    attacked = False

    def write_then_drift(path: Path, rows: object, **kwargs: object) -> None:
        nonlocal attacked
        materialized = tuple(rows)  # type: ignore[arg-type]
        real_write(path, materialized, **kwargs)  # type: ignore[arg-type]
        target = events_path if drift == "events" else recordings_path
        if path != target or attacked:
            return
        if drift == "events" and not any(
            row.get("previous_state") == "REVIEWED" for row in materialized
        ):
            return
        if drift == "recordings" and not any(
            row.get("state") == "APPROVED" for row in materialized
        ):
            return
        attacked = True
        changed = [dict(row) for row in materialized]
        changed[-1]["result" if drift == "events" else "notes"] = "mutatum"
        real_write(path, changed, **kwargs)

    monkeypatch.setattr(store_module, "write_jsonl_atomic", write_then_drift)

    expected = ValueError if drift == "events" else RuntimeError
    message = r"durable file bytes|snapshot|transaction" if drift == "events" else "recovery"
    with pytest.raises(expected, match=message):
        _build_with_fake_audio(paths, config)

    assert attacked


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing contract")
def test_build_manifest_holds_new_events_guard_through_recordings_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    recordings_path = paths.manifests / "recordings.jsonl"  # type: ignore[attr-defined]
    events_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    real_write = store_module.write_jsonl_atomic
    blocked = False

    def attack_events_during_recordings_write(path: Path, rows: object, **kwargs: object) -> None:
        nonlocal blocked
        materialized = tuple(rows)  # type: ignore[arg-type]
        if path == recordings_path and any(row.get("state") == "APPROVED" for row in materialized):
            try:
                real_write(events_path, read_jsonl(events_path))
            except OSError:
                blocked = True
            else:
                raise AssertionError("processing event file was not pinned through state replace")
        real_write(path, materialized, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        store_module,
        "write_jsonl_atomic",
        attack_events_during_recordings_write,
    )

    assert _build_with_fake_audio(paths, config)
    assert blocked


def test_terminal_event_digests_equal_the_consumed_authoritative_snapshot(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)

    assert _build_with_fake_audio(paths, config)

    events_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    terminal = next(row for row in read_jsonl(events_path) if row["previous_state"] == "REVIEWED")
    expected = tuple(
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            paths.manifests / "rights.jsonl",  # type: ignore[attr-defined]
            paths.manifests / "transcripts.jsonl",  # type: ignore[attr-defined]
        )
    )
    recordings = tuple(
        review_module._decode_recording(row)
        for row in read_jsonl(paths.manifests / "recordings.jsonl")  # type: ignore[attr-defined]
    )
    selection_ids = set(read_jsonl(paths.manifests / "pilot-selection.json")[0]["recording_ids"])  # type: ignore[attr-defined]
    inventory_sha256 = store_module.jsonl_sha256(
        (
            replace(recording, state=CorpusState.REVIEWED)
            if recording.recording_id in selection_ids
            else recording
        ).to_dict()
        for recording in recordings
    )
    assert terminal["tool_versions"][0] == "approved-manifest-v2"
    assert terminal["input_sha256s"][1] == inventory_sha256
    assert tuple(terminal["input_sha256s"][2:4]) == expected
    assert (
        terminal["input_sha256s"][6]
        == hashlib.sha256(
            (paths.manifests / "review.jsonl").read_bytes()  # type: ignore[attr-defined]
        ).hexdigest()
    )
    assert (
        terminal["input_sha256s"][7]
        == hashlib.sha256(
            (paths.manifests / "segments.jsonl").read_bytes()  # type: ignore[attr-defined]
        ).hexdigest()
    )


def test_build_manifest_preserves_business_failure_when_namespace_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    real_close = manifest_module._OutputNamespaceGuard.close

    def close_then_fail(guard: object) -> None:
        real_close(guard)  # type: ignore[arg-type]
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(manifest_module._OutputNamespaceGuard, "close", close_then_fail)
    monkeypatch.setattr(
        manifest_module,
        "_load_selection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("business failure")),
    )

    with pytest.raises(ValueError, match="business failure"):
        _build_with_fake_audio(paths, config)


def test_file_snapshot_cleanup_closes_every_guard_and_raises_first_failure() -> None:
    calls: list[str] = []

    class BrokenGuard:
        def __init__(self, name: str, failure: str | None) -> None:
            self.name = name
            self.failure = failure

        def close(self) -> None:
            calls.append(self.name)
            if self.failure is not None:
                raise OSError(self.failure)

    guards = {
        "first": BrokenGuard("first", "first failure"),
        "second": BrokenGuard("second", "second failure"),
        "third": BrokenGuard("third", None),
    }

    with pytest.raises(OSError, match="second failure"):
        manifest_module._close_file_snapshots(guards)  # type: ignore[arg-type]

    assert calls == ["third", "second", "first"]


def test_file_snapshot_rejects_expected_digest_conflict_and_use_after_close(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot.jsonl"
    write_jsonl_atomic(path, ({"value": "stable"},))
    guards: dict[str, object] = {}
    snapshot = manifest_module._add_file_snapshot(guards, path)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expected bytes"):
        manifest_module._add_file_snapshot(  # type: ignore[arg-type]
            guards,
            path,
            expected_sha256="0" * 64,
        )

    snapshot.close()
    with pytest.raises(ValueError, match="already closed"):
        snapshot.validate()
