from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from latintts.corpus.paths import CorpusPaths

DryRunOperation = Literal["READ", "WRITE", "RECOVER"]
_OPERATION_ORDER: dict[DryRunOperation, int] = {"READ": 0, "WRITE": 1, "RECOVER": 2}


@dataclass(frozen=True, slots=True)
class DryRunPath:
    operation: DryRunOperation
    relative_path: str

    def render(self) -> str:
        return f"{self.operation} {self.relative_path}"


def _project_relative(paths: CorpusPaths, path: Path, *, directory: bool = False) -> str:
    relative = path.relative_to(paths.project_root).as_posix()
    return f"{relative}/" if directory else relative


def _config_relative(paths: CorpusPaths, config_path: Path) -> str:
    candidate = config_path if config_path.is_absolute() else paths.project_root / config_path
    lexical = Path(os.path.abspath(candidate))
    root = Path(os.path.abspath(paths.project_root))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError("--dry-run config must be within the project root") from error
    if not relative.parts:
        raise ValueError("--dry-run config must identify a project file")
    return relative.as_posix()


def _join(base: str, suffix: str) -> str:
    return f"{base.rstrip('/')}/{suffix.lstrip('/')}"


def _sorted_plan(entries: set[DryRunPath]) -> tuple[DryRunPath, ...]:
    return tuple(
        sorted(
            entries,
            key=lambda entry: (_OPERATION_ORDER[entry.operation], entry.relative_path),
        )
    )


def plan_corpus_dry_run(
    command: str,
    paths: CorpusPaths,
    *,
    config_path: Path | None = None,
    init_intake: bool = False,
    init_transcript: bool = False,
    smoke_test: bool = False,
    allow_download: bool = False,
) -> tuple[DryRunPath, ...]:
    """Declare project paths without inspecting corpus state or invoking any backend."""
    entries: set[DryRunPath] = set()

    def add(operation: DryRunOperation, *relative_paths: str) -> None:
        entries.update(DryRunPath(operation, path) for path in relative_paths)

    local = _project_relative(paths, paths.local_data)
    raw_spoken = _project_relative(paths, paths.raw_spoken)
    raw_sung = _project_relative(paths, paths.raw_sung)
    normalized = _project_relative(paths, paths.normalized)
    segments = _project_relative(paths, paths.segments)
    alignments = _project_relative(paths, paths.alignments)
    manifests = _project_relative(paths, paths.manifests)
    runs = _join(alignments, "runs")
    lock = _join(manifests, ".corpus-mutation.lock")

    for directory in (
        paths.raw_spoken,
        paths.raw_sung,
        paths.normalized,
        paths.segments,
        paths.alignments,
        paths.manifests,
    ):
        add("WRITE", _project_relative(paths, directory, directory=True))

    if command == "doctor":
        return _sorted_plan(entries)

    if command == "inventory":
        add("WRITE", lock)
        if init_intake:
            add(
                "READ",
                _join(raw_spoken, "**/*"),
                _join(raw_sung, "**/*"),
                _join(manifests, "intake.csv"),
            )
            add("WRITE", _join(manifests, "intake.csv"))
            return _sorted_plan(entries)
        event_path = _join(runs, "inventory/processing-events.jsonl")
        recordings = _join(manifests, "recordings.jsonl")
        add(
            "READ",
            _join(manifests, "intake.csv"),
            _join(manifests, "rights.jsonl"),
            recordings,
            _join(raw_spoken, "**/*"),
            _join(raw_sung, "**/*"),
            event_path,
        )
        add("WRITE", recordings, event_path)
        add("RECOVER", recordings, event_path)
        return _sorted_plan(entries)

    if command == "select-pilot":
        add(
            "READ",
            _join(manifests, "recordings.jsonl"),
            _join(manifests, "pilot-selection.json"),
        )
        add("WRITE", lock, _join(manifests, "pilot-selection.json"))
        return _sorted_plan(entries)

    if command == "prepare-text":
        selection = _join(manifests, "pilot-selection.json")
        intake = _join(manifests, "transcript-intake.jsonl")
        add("READ", selection, intake)
        add("WRITE", lock)
        if init_transcript:
            add("WRITE", intake)
            return _sorted_plan(entries)
        recordings = _join(manifests, "recordings.jsonl")
        events = _join(runs, "prepare-text/processing-events.jsonl")
        add(
            "READ",
            recordings,
            _join(manifests, "transcripts.jsonl"),
            _join(manifests, "pronunciation-review-*.json"),
            events,
            _join(local, "**/*"),
        )
        add(
            "WRITE",
            recordings,
            _join(manifests, "transcripts.jsonl"),
            _join(manifests, "pronunciation-review-*.json"),
            events,
        )
        add("RECOVER", recordings, events)
        return _sorted_plan(entries)

    if config_path is None:
        raise ValueError(f"{command} dry-run requires a config path")
    config = _config_relative(paths, config_path)
    add("READ", config)
    add("WRITE", lock)

    selection = _join(manifests, "pilot-selection.json")
    recordings = _join(manifests, "recordings.jsonl")
    transcripts = _join(manifests, "transcripts.jsonl")
    review = _join(manifests, "review.jsonl")
    events = _join(runs, "*/processing-events.jsonl")
    recording_run = _join(runs, "*/*")
    raw_audio = _join(raw_spoken, "**/*")
    analysis_audio = _join(normalized, "analysis-*.wav")
    analysis_lock = _join(normalized, "analysis-*.wav.lock")
    candidates = _join(segments, "candidates/candidate-*.wav")
    candidate_locks = _join(segments, "candidates/candidate-*.wav.lock")
    alignment_cache = _join(recording_run, "alignment-cache/*.json")
    huggingface_cache = _join(local, "cache/huggingface/**/*")

    if command == "segment":
        segmentation = _join(recording_run, "segmentation.json")
        add(
            "READ",
            selection,
            recordings,
            transcripts,
            raw_audio,
            analysis_audio,
            segmentation,
            events,
        )
        add("WRITE", recordings, analysis_audio, analysis_lock, segmentation, events)
        add("RECOVER", recordings, events)
        return _sorted_plan(entries)

    if command == "pair":
        pairing = _join(recording_run, "pairing.json")
        add(
            "READ",
            selection,
            recordings,
            transcripts,
            review,
            raw_audio,
            analysis_audio,
            candidates,
            alignment_cache,
            huggingface_cache,
            _join(recording_run, "segmentation.json"),
            pairing,
            _join(recording_run, "pairing-automatic.json"),
            events,
        )
        add(
            "WRITE",
            recordings,
            candidates,
            candidate_locks,
            alignment_cache,
            _join(recording_run, ".pairing.lock"),
            pairing,
            events,
        )
        if allow_download:
            add("WRITE", huggingface_cache)
        add("RECOVER", recordings, events)
        return _sorted_plan(entries)

    if command == "align" and smoke_test:
        add("READ", ".git", ".git/**/*", huggingface_cache)
        add(
            "WRITE",
            _join(alignments, "runtime/environment.txt"),
            _join(alignments, "runtime/smoke.json"),
        )
        if allow_download:
            add("WRITE", huggingface_cache)
        return _sorted_plan(entries)

    if command == "align":
        alignment = _join(recording_run, "alignment.json")
        add(
            "READ",
            selection,
            recordings,
            review,
            analysis_audio,
            candidates,
            alignment_cache,
            huggingface_cache,
            _join(recording_run, "pairing.json"),
            _join(recording_run, "pairing-automatic.json"),
            alignment,
            events,
        )
        add("WRITE", recordings, alignment, events)
        if allow_download:
            add("WRITE", huggingface_cache)
        add("RECOVER", recordings, events)
        return _sorted_plan(entries)

    review_bundle = _join(recording_run, "review/**/*")
    pairing = _join(recording_run, "pairing.json")
    alignment = _join(recording_run, "alignment.json")
    review_audio = _join(segments, "review/review-*.wav")

    if command == "export-review":
        add(
            "READ",
            selection,
            transcripts,
            recordings,
            raw_audio,
            pairing,
            alignment,
            candidates,
            review_bundle,
        )
        add(
            "WRITE",
            review_audio,
            _join(segments, "review/review-*.wav.lock"),
            _join(recording_run, "review/*/index.html"),
            _join(recording_run, "review/*/automatic.json"),
            _join(recording_run, "review/*/decision.json"),
            _join(recording_run, "review/*/take-1.wav"),
            _join(recording_run, "review/*/take-1.TextGrid"),
            _join(recording_run, "review/*/take-2.wav"),
            _join(recording_run, "review/*/take-2.TextGrid"),
        )
        return _sorted_plan(entries)

    if command == "import-review":
        pairing_automatic = _join(recording_run, "pairing-automatic.json")
        add(
            "READ",
            selection,
            transcripts,
            recordings,
            review,
            raw_audio,
            pairing,
            pairing_automatic,
            alignment,
            candidates,
            review_audio,
            review_bundle,
            events,
        )
        add("WRITE", review, recordings, pairing, pairing_automatic, events)
        add("RECOVER", review, recordings, pairing, pairing_automatic, events)
        return _sorted_plan(entries)

    manifest_attestations = _join(runs, "*/manifest-attestations.jsonl")
    segment_manifest = _join(manifests, "segments.jsonl")
    lossless = _join(segments, "lossless/lossless-*.flac")

    if command == "build-manifest":
        pairing_automatic = _join(recording_run, "pairing-automatic.json")
        staging = _join(segments, ".manifest-staging/**/*")
        add(
            "READ",
            selection,
            recordings,
            _join(manifests, "rights.jsonl"),
            transcripts,
            review,
            segment_manifest,
            raw_audio,
            analysis_audio,
            candidates,
            review_audio,
            lossless,
            staging,
            _join(recording_run, "segmentation.json"),
            pairing,
            pairing_automatic,
            alignment,
            review_bundle,
            events,
            manifest_attestations,
        )
        add(
            "WRITE",
            review,
            recordings,
            segment_manifest,
            events,
            manifest_attestations,
            staging,
            candidates,
            lossless,
            pairing,
            pairing_automatic,
        )
        add(
            "RECOVER",
            review,
            recordings,
            segment_manifest,
            events,
            manifest_attestations,
            staging,
            pairing,
            pairing_automatic,
        )
        return _sorted_plan(entries)

    if command == "report":
        report_json = _join(manifests, "report.json")
        report_markdown = _join(manifests, "report.md")
        transaction_paths = (
            _join(manifests, ".report-output-transaction.json"),
            _join(manifests, ".report-output-transaction.rolled-back.json"),
            _join(manifests, ".report-output-transaction.completed.json"),
            _join(manifests, ".report-output-transaction.*"),
            _join(manifests, ".report-output-transaction.*/**/*"),
            _join(manifests, ".report.json.*"),
            _join(manifests, ".report.md.*"),
            _join(manifests, ".recovery.report.*"),
            _join(manifests, ".restore.report.*"),
        )
        intent_temporary = _join(manifests, "..report-output-transaction.json.*")
        add(
            "READ",
            _join(manifests, "pilot-telemetry.json"),
            recordings,
            selection,
            _join(manifests, "rights.jsonl"),
            transcripts,
            review,
            segment_manifest,
            report_json,
            report_markdown,
            raw_audio,
            analysis_audio,
            candidates,
            lossless,
            _join(recording_run, "segmentation.json"),
            pairing,
            _join(recording_run, "pairing-automatic.json"),
            alignment,
            _join(recording_run, "review/*/automatic.json"),
            events,
            manifest_attestations,
            *transaction_paths,
            intent_temporary,
        )
        add("WRITE", report_json, report_markdown, *transaction_paths, intent_temporary)
        add(
            "RECOVER",
            report_json,
            report_markdown,
            *transaction_paths,
        )
        return _sorted_plan(entries)

    raise ValueError(f"unsupported corpus dry-run command: {command}")
