from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, cast

from latintts.corpus.audio import DerivedAudio, _raw_source, _validate_source_hash
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState, IssueCode
from latintts.corpus.locking import corpus_mutation_lease
from latintts.corpus.manifest import _load_rights, _validate_transcript_sources
from latintts.corpus.pairing import pairing_from_dict
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import (
    RecordingRecord,
    ReviewEvent,
    SegmentRecord,
    advance_recording,
    require_exact_fields,
)
from latintts.corpus.review import (
    _AUTOMATIC_FIELDS,
    _AUTOMATIC_TAKE_FIELDS,
    _alignment_by_take,
    _artifact_binding,
    _decode_recording,
    _load_existing_review_events,
    _load_pairing_and_alignment,
    _load_selection,
    _load_transcript_layers,
    _read_json,
    _review_entity_id,
    _validate_existing_pairing_corrections,
    _validate_take_summary,
    replay_review_events,
)
from latintts.corpus.store import jsonl_sha256, read_jsonl, validate_processing_events

_TELEMETRY_FIELDS = frozenset(
    {
        "schema_version",
        "text_preparation_seconds",
        "review_seconds",
        "alignment_wall_seconds",
        "gpu_retry_count",
        "peak_gpu_memory_bytes",
        "pilot_storage_bytes",
    }
)
_BOUNDARY_FIELDS = frozenset({"segment_start", "segment_end"})
_SAMPLE_RATE = 16_000


class ScaleDecision(str, Enum):
    SCALABLE = "scalable"
    OPTIMIZE = "optimize"
    NOT_READY = "not_ready"


def _finite_number(value: object, field: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{field} must be a number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{field} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class PilotMetrics:
    input_duration_seconds: float
    speech_duration_seconds: float
    auto_pairing_correct_ratio: float
    boundary_unchanged_ratio: float
    review_minutes_per_audio_minute: float
    approved_speech_ratio: float
    gpu_realtime_factor: float
    peak_gpu_memory_bytes: int

    def __post_init__(self) -> None:
        input_duration = _finite_number(
            self.input_duration_seconds, "input_duration_seconds", positive=True
        )
        speech_duration = _finite_number(
            self.speech_duration_seconds, "speech_duration_seconds", positive=True
        )
        if speech_duration > input_duration:
            raise ValueError("speech_duration_seconds must not exceed input duration")
        for value, field in (
            (self.auto_pairing_correct_ratio, "auto_pairing_correct_ratio"),
            (self.boundary_unchanged_ratio, "boundary_unchanged_ratio"),
            (self.approved_speech_ratio, "approved_speech_ratio"),
        ):
            ratio = _finite_number(value, field)
            if ratio > 1:
                raise ValueError(f"{field} must be a finite ratio")
        _finite_number(
            self.review_minutes_per_audio_minute,
            "review_minutes_per_audio_minute",
        )
        _finite_number(self.gpu_realtime_factor, "gpu_realtime_factor", positive=True)
        if type(self.peak_gpu_memory_bytes) is not int:
            raise TypeError("peak_gpu_memory_bytes must be an integer")
        if self.peak_gpu_memory_bytes <= 0:
            raise ValueError("peak_gpu_memory_bytes must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReportTelemetry:
    schema_version: str
    text_preparation_seconds: float
    review_seconds: float
    alignment_wall_seconds: float
    gpu_retry_count: int
    peak_gpu_memory_bytes: int
    pilot_storage_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != "1" or type(self.schema_version) is not str:
            raise ValueError("telemetry schema_version must be '1'")
        _finite_number(self.text_preparation_seconds, "text_preparation_seconds")
        _finite_number(self.review_seconds, "review_seconds")
        _finite_number(self.alignment_wall_seconds, "alignment_wall_seconds", positive=True)
        for value, field, minimum in (
            (self.gpu_retry_count, "gpu_retry_count", 0),
            (self.peak_gpu_memory_bytes, "peak_gpu_memory_bytes", 1),
            (self.pilot_storage_bytes, "pilot_storage_bytes", 1),
        ):
            if type(value) is not int:
                raise TypeError(f"{field} must be an integer")
            if value < minimum:
                qualifier = "positive" if minimum else "non-negative"
                raise ValueError(f"{field} must be {qualifier}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReportTelemetry:
        if type(raw) is not dict or set(raw) != _TELEMETRY_FIELDS:
            got = sorted(raw) if type(raw) is dict else type(raw).__name__
            raise ValueError(f"pilot telemetry must contain exact fields; got {got}")
        return cls(**raw)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PilotReport:
    metrics: PilotMetrics
    scale_decision: ScaleDecision
    text_unit_count: int
    reading_count: int
    repetition_group_count: int
    approved_segment_count: int
    approved_duration_seconds: float
    rejected_duration_seconds: float
    unreviewed_duration_seconds: float
    issue_code_counts: tuple[tuple[str, int], ...]
    rejection_reason_counts: tuple[tuple[str, int], ...]
    telemetry: ReportTelemetry
    full_corpus_spoken_duration_seconds: float
    projected_full_corpus_person_hours: float
    projected_gpu_hours: float
    projected_storage_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1",
            "corpus_version": "corpus-v1",
            "scale_decision": self.scale_decision.value,
            "metrics": self.metrics.to_dict(),
            "counts": {
                "text_units": self.text_unit_count,
                "readings": self.reading_count,
                "repetition_groups": self.repetition_group_count,
                "approved_segments": self.approved_segment_count,
            },
            "durations_seconds": {
                "approved": self.approved_duration_seconds,
                "rejected": self.rejected_duration_seconds,
                "unreviewed": self.unreviewed_duration_seconds,
            },
            "issue_code_counts": dict(self.issue_code_counts),
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "timing_seconds": {
                "text_preparation": self.telemetry.text_preparation_seconds,
                "human_review": self.telemetry.review_seconds,
            },
            "gpu": {
                "alignment_wall_seconds": self.telemetry.alignment_wall_seconds,
                "realtime_factor": self.metrics.gpu_realtime_factor,
                "retry_count": self.telemetry.gpu_retry_count,
                "peak_memory_bytes": self.telemetry.peak_gpu_memory_bytes,
            },
            "projection": {
                "full_corpus_spoken_duration_seconds": (self.full_corpus_spoken_duration_seconds),
                "person_hours": self.projected_full_corpus_person_hours,
                "gpu_hours": self.projected_gpu_hours,
                "storage_bytes": self.projected_storage_bytes,
                "pilot_storage_bytes": self.telemetry.pilot_storage_bytes,
            },
        }


def classify_scale_readiness(metrics: PilotMetrics) -> ScaleDecision:
    if type(metrics) is not PilotMetrics:
        raise TypeError("metrics must be PilotMetrics")
    red = (
        metrics.auto_pairing_correct_ratio < 0.85
        or metrics.boundary_unchanged_ratio < 0.70
        or metrics.review_minutes_per_audio_minute > 6.0
        or metrics.approved_speech_ratio < 0.60
    )
    if red:
        return ScaleDecision.NOT_READY
    green = (
        metrics.auto_pairing_correct_ratio >= 0.95
        and metrics.boundary_unchanged_ratio >= 0.85
        and metrics.review_minutes_per_audio_minute <= 3.0
        and metrics.approved_speech_ratio >= 0.80
    )
    return ScaleDecision.SCALABLE if green else ScaleDecision.OPTIMIZE


def _one_row(path: Path, kind: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"{kind} must contain exactly one row")
    return rows[0]


def _load_telemetry(paths: CorpusPaths) -> ReportTelemetry:
    return ReportTelemetry.from_dict(
        _one_row(paths.manifests / "pilot-telemetry.json", "pilot telemetry")
    )


def _load_recordings(paths: CorpusPaths) -> tuple[RecordingRecord, ...]:
    recordings = tuple(
        _decode_recording(raw) for raw in read_jsonl(paths.manifests / "recordings.jsonl")
    )
    identities = tuple(recording.recording_id for recording in recordings)
    if not recordings or len(identities) != len(set(identities)):
        raise ValueError("recordings.jsonl must contain unique recordings")
    return recordings


def _vad_intervals(
    raw: dict[str, Any],
    recording: RecordingRecord,
    *,
    analysis_sample_count: int,
) -> tuple[tuple[int, int], ...]:
    recording_id = recording.recording_id
    if raw.get("schema_version") != "1" or raw.get("recording_id") != recording_id:
        raise ValueError("segmentation identity does not match pilot recording")
    vad = raw.get("vad")
    if type(vad) is not dict or vad.get("sample_rate") != _SAMPLE_RATE:
        raise ValueError("segmentation VAD must use 16 kHz")
    intervals_raw = vad.get("speech_intervals")
    if type(intervals_raw) is not list:
        raise TypeError("segmentation speech_intervals must be an array")
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for interval in intervals_raw:
        if type(interval) is not dict or set(interval) != {"start_sample", "end_sample"}:
            raise ValueError("segmentation speech interval must contain exact fields")
        start = interval["start_sample"]
        end = interval["end_sample"]
        if (
            type(start) is not int
            or type(end) is not int
            or start < cursor
            or end <= start
            or end > analysis_sample_count
            or end > math.ceil(recording.metadata.duration_seconds * _SAMPLE_RATE)
        ):
            raise ValueError("segmentation speech intervals must be ordered half-open ranges")
        intervals.append((start, end))
        cursor = end
    if not intervals:
        raise ValueError("segmentation contains zero VAD speech duration")
    return tuple(intervals)


def _count_issue(counter: Counter[str], value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError("issue_code must be a string or null")
    try:
        code = IssueCode(value).value
    except ValueError as error:
        raise ValueError(f"unknown issue_code: {value}") from error
    counter[code] += 1


def _validate_terminal_processing_evidence(
    *,
    recordings: list[RecordingRecord],
    processing_rows: tuple[dict[str, Any], ...],
    config_sha256: str,
    inventory_sha256: str,
    rights_sha256: str,
    transcripts_sha256: str,
    review_sha256: str,
    segments_sha256: str,
    pairing_sha256s: dict[str, str],
    alignment_sha256s: dict[str, str],
    ffmpeg_versions: dict[str, str],
) -> None:
    events = validate_processing_events(processing_rows)
    timestamp = "1970-01-01T00:00:00+00:00"
    for recording in recordings:
        recording_events = tuple(
            event for event in events if event.recording_id == recording.recording_id
        )
        if not recording_events:
            raise ValueError("processing history lacks a selected recording")
        base = replace(recording, state=CorpusState.REVIEWED)
        tool_versions = (
            "approved-manifest-v2",
            ffmpeg_versions[recording.recording_id],
        )
        _, current = advance_recording(
            base,
            recording.state,
            input_sha256s=(
                recording.sha256,
                inventory_sha256,
                rights_sha256,
                transcripts_sha256,
                pairing_sha256s[recording.recording_id],
                alignment_sha256s[recording.recording_id],
                review_sha256,
                segments_sha256,
            ),
            config_sha256=config_sha256,
            tool_versions=tool_versions,
            started_at=timestamp,
            finished_at=timestamp,
            result="success",
        )
        terminal_event = recording_events[-1]
        if terminal_event.tool_versions[0] != "approved-manifest-v2":
            raise ValueError(
                "legacy terminal processing evidence cannot support a trusted report; "
                "rebuild the manifest"
            )
        if terminal_event != current:
            raise ValueError("processing terminal event does not bind current manifest evidence")


def _canonical_report_bytes(report: PilotReport) -> bytes:
    return (
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _markdown_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _render_counts(title: str, counts: tuple[tuple[str, int], ...]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "| --- | ---: |"]
    if counts:
        lines.extend(f"| {_markdown_cell(key)} | {value} |" for key, value in counts)
    else:
        lines.append("| None | 0 |")
    return lines


def _render_markdown(report: PilotReport) -> bytes:
    metrics = report.metrics
    lines = [
        "# LatinTTS Corpus Alignment Pilot Report",
        "",
        f"Scale decision: **{report.scale_decision.value}**",
        "",
        "## Quality metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Input duration (seconds) | {metrics.input_duration_seconds:.6f} |",
        f"| VAD speech duration (seconds) | {metrics.speech_duration_seconds:.6f} |",
        "| Automatic pairing accepted without correction ratio | "
        f"{metrics.auto_pairing_correct_ratio:.6f} |",
        f"| Boundary unchanged ratio | {metrics.boundary_unchanged_ratio:.6f} |",
        f"| Human review minutes / audio minute | {metrics.review_minutes_per_audio_minute:.6f} |",
        f"| Approved effective take duration ratio | {metrics.approved_speech_ratio:.6f} |",
        f"| GPU real-time factor | {metrics.gpu_realtime_factor:.6f} |",
        f"| Peak GPU memory (bytes) | {metrics.peak_gpu_memory_bytes} |",
        "",
        "## Corpus counts and durations",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Text units | {report.text_unit_count} |",
        f"| Readings | {report.reading_count} |",
        f"| Repetition groups | {report.repetition_group_count} |",
        f"| Approved segments | {report.approved_segment_count} |",
        f"| Approved duration (seconds) | {report.approved_duration_seconds:.6f} |",
        f"| Rejected duration (seconds) | {report.rejected_duration_seconds:.6f} |",
        f"| Unreviewed duration (seconds) | {report.unreviewed_duration_seconds:.6f} |",
        "",
        "## Operational telemetry",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Text preparation (seconds) | {report.telemetry.text_preparation_seconds:.6f} |",
        f"| Human review (seconds) | {report.telemetry.review_seconds:.6f} |",
        f"| Alignment wall time (seconds) | {report.telemetry.alignment_wall_seconds:.6f} |",
        f"| GPU retries | {report.telemetry.gpu_retry_count} |",
        f"| Peak GPU memory (bytes) | {report.telemetry.peak_gpu_memory_bytes} |",
        f"| Pilot storage (bytes) | {report.telemetry.pilot_storage_bytes} |",
        "",
        "## Full-corpus projection",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        "| Full spoken inventory duration (seconds) | "
        f"{report.full_corpus_spoken_duration_seconds:.6f} |",
        f"| Person-hours | {report.projected_full_corpus_person_hours:.6f} |",
        f"| GPU hours | {report.projected_gpu_hours:.6f} |",
        f"| Storage bytes | {report.projected_storage_bytes} |",
        "",
        *_render_counts("Issue codes", report.issue_code_counts),
        "",
        *_render_counts("Rejection reasons", report.rejection_reason_counts),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _prepare_atomic(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_outputs(paths: CorpusPaths, report: PilotReport) -> None:
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    json_temporary = _prepare_atomic(json_path, _canonical_report_bytes(report))
    markdown_temporary: Path | None = None
    json_backup: Path | None = None
    markdown_backup: Path | None = None
    json_replaced = False
    markdown_replaced = False
    try:
        markdown_temporary = _prepare_atomic(markdown_path, _render_markdown(report))
        if json_path.exists():
            json_backup = _prepare_atomic(json_path, json_path.read_bytes())
        if markdown_path.exists():
            markdown_backup = _prepare_atomic(markdown_path, markdown_path.read_bytes())
        json_replaced = True
        os.replace(json_temporary, json_path)
        markdown_replaced = True
        os.replace(markdown_temporary, markdown_path)
    except BaseException:
        try:
            if markdown_replaced:
                if markdown_backup is None:
                    markdown_path.unlink(missing_ok=True)
                else:
                    os.replace(markdown_backup, markdown_path)
            if json_replaced:
                if json_backup is None:
                    json_path.unlink(missing_ok=True)
                else:
                    os.replace(json_backup, json_path)
        except BaseException as rollback_error:
            raise RuntimeError(
                "report output rollback failed; recovery required"
            ) from rollback_error
        raise
    finally:
        json_temporary.unlink(missing_ok=True)
        if markdown_temporary is not None:
            markdown_temporary.unlink(missing_ok=True)
        if json_backup is not None:
            json_backup.unlink(missing_ok=True)
        if markdown_backup is not None:
            markdown_backup.unlink(missing_ok=True)


def _build_report_locked(paths: CorpusPaths, config: CorpusConfig) -> PilotReport:
    telemetry = _load_telemetry(paths)
    recordings = _load_recordings(paths)
    recording_by_id = {recording.recording_id: recording for recording in recordings}
    selection = _load_selection(paths)
    if (
        selection.schema_version != "1"
        or len(selection.recording_ids) not in (2, 3)
        or len(selection.recording_ids) != len(set(selection.recording_ids))
        or len(selection.recording_ids) != len(selection.inventory_hashes)
    ):
        raise ValueError("pilot selection identity is invalid")
    selection_ids = set(selection.recording_ids)
    inventory_sha256 = jsonl_sha256(
        (
            replace(recording, state=CorpusState.REVIEWED)
            if recording.recording_id in selection_ids
            else recording
        ).to_dict()
        for recording in recordings
    )
    for inventory_recording in recordings:
        if inventory_recording.content_type == "spoken":
            _validate_source_hash(
                _raw_source(inventory_recording, paths),
                inventory_recording.sha256,
                derived=False,
            )
    pilot = []
    for recording_id, expected_hash in zip(
        selection.recording_ids, selection.inventory_hashes, strict=True
    ):
        recording = recording_by_id.get(recording_id)
        if (
            recording is None
            or recording.sha256 != expected_hash
            or recording.content_type != "spoken"
        ):
            raise ValueError("pilot selection does not match spoken inventory")
        if recording.state not in {CorpusState.APPROVED, CorpusState.REJECTED}:
            raise CorpusFailure("REVIEW_REQUIRED", "pilot report requires terminal review states")
        pilot.append(recording)
    input_duration = sum(recording.metadata.duration_seconds for recording in pilot)
    full_duration = sum(
        recording.metadata.duration_seconds
        for recording in recordings
        if recording.content_type == "spoken"
    )
    if input_duration <= 0 or full_duration < input_duration:
        raise ValueError("spoken inventory duration cannot project from the pilot")

    rights_path = paths.manifests / "rights.jsonl"
    rights = _load_rights(paths)
    rights_sha256 = hashlib.sha256(rights_path.read_bytes()).hexdigest()
    for recording in pilot:
        recording_rights = rights.get(recording.rights_id)
        if recording_rights is None:
            raise ValueError("recording rights_id does not exist")
        if recording_rights.speaker_id != recording.speaker_id:
            raise ValueError("recording rights speaker does not match recording")
        if not (
            recording_rights.allow_local_processing
            and recording_rights.allow_model_training
            and recording_rights.allow_internal_evaluation
        ):
            raise CorpusFailure(
                "RIGHTS_SCOPE_UNCONFIRMED",
                "pilot report requires local, training, and internal-evaluation rights",
            )

    transcripts_path = paths.manifests / "transcripts.jsonl"
    transcripts = _load_transcript_layers(paths, selection)
    transcripts_sha256 = hashlib.sha256(transcripts_path.read_bytes()).hexdigest()
    for recording in pilot:
        _validate_transcript_sources(transcripts[recording.recording_id], recording.recording_id)
    review_path = paths.manifests / "review.jsonl"
    review_events = _load_existing_review_events(review_path)
    review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
    processing_rows = read_jsonl(
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    )
    _validate_existing_pairing_corrections(paths, config, selection, recordings, review_events)
    events_by_entity: dict[str, tuple[ReviewEvent, ...]] = defaultdict(tuple)
    for event in review_events:
        events_by_entity[event.entity_id] = (*events_by_entity[event.entity_id], event)

    issue_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    vad_by_recording: dict[str, tuple[tuple[int, int], ...]] = {}
    text_unit_count = 0
    reading_count = 0
    repetition_group_count = 0
    unchanged_boundaries = 0
    approved_duration = 0.0
    rejected_duration = 0.0
    unreviewed_duration = 0.0
    total_automatic_groups = 0
    automatic_selected = 0
    decision_by_take: dict[tuple[str, str, int], str] = {}
    expected_segment_evidence: dict[tuple[str, str, int], dict[str, object]] = {}
    expected_review_entities: set[str] = set()
    pairing_sha256s: dict[str, str] = {}
    alignment_sha256s: dict[str, str] = {}
    ffmpeg_versions: dict[str, str] = {}

    for recording in pilot:
        run_directory, final_pairing, alignment = _load_pairing_and_alignment(
            paths, config, recording
        )
        pairing_sha256s[recording.recording_id] = hashlib.sha256(
            (run_directory / "pairing.json").read_bytes()
        ).hexdigest()
        alignment_sha256s[recording.recording_id] = hashlib.sha256(
            (run_directory / "alignment.json").read_bytes()
        ).hexdigest()
        ffmpeg_versions[recording.recording_id] = final_pairing.ffmpeg_version
        segmentation_path = run_directory / "segmentation.json"
        segmentation = _one_row(segmentation_path, "segmentation artifact")
        segmentation_sha256 = hashlib.sha256(segmentation_path.read_bytes()).hexdigest()
        analysis_raw = segmentation.get("analysis_audio")
        if type(analysis_raw) is not dict:
            raise TypeError("segmentation analysis_audio must be an object")
        analysis_audio = DerivedAudio.from_dict(analysis_raw)
        if (
            segmentation.get("schema_version") != "1"
            or segmentation.get("status") != "success"
            or segmentation.get("recording_id") != recording.recording_id
            or segmentation.get("config_sha256") != config.digest
            or final_pairing.recording_id != recording.recording_id
            or final_pairing.config_sha256 != config.digest
            or final_pairing.segmentation_artifact_sha256 != segmentation_sha256
            or analysis_audio.mode != "analysis"
            or analysis_audio.source_relative_path != recording.relative_path
            or analysis_audio.source_sha256 != recording.sha256
            or analysis_audio.config_sha256 != config.digest
            or analysis_audio.relative_path != final_pairing.analysis_audio_relative_path
            or analysis_audio.sha256 != final_pairing.analysis_audio_sha256
            or analysis_audio.metrics is None
        ):
            raise ValueError("segmentation provenance does not match pairing and recording")
        _count_issue(issue_counts, segmentation.get("issue_code"))
        speech_intervals = _vad_intervals(
            segmentation,
            recording,
            analysis_sample_count=analysis_audio.metrics.sample_count,
        )
        vad_by_recording[recording.recording_id] = speech_intervals

        automatic_path = run_directory / "pairing-automatic.json"
        automatic_pairing = final_pairing
        if automatic_path.exists():
            automatic_pairing = pairing_from_dict(
                _one_row(automatic_path, "automatic pairing artifact")
            )
        automatic_fixed_identity = (
            automatic_pairing.schema_version,
            automatic_pairing.recording_id,
            automatic_pairing.config_sha256,
            automatic_pairing.segmentation_artifact_sha256,
            automatic_pairing.analysis_audio_relative_path,
            automatic_pairing.analysis_audio_sha256,
            automatic_pairing.ffmpeg_version,
            automatic_pairing.alignment_runtime_sha256s,
            automatic_pairing.cache_key,
            automatic_pairing.pairing_parameters,
            automatic_pairing.windows,
            tuple(outcome.unit_id for outcome in automatic_pairing.groups),
        )
        final_fixed_identity = (
            final_pairing.schema_version,
            final_pairing.recording_id,
            final_pairing.config_sha256,
            final_pairing.segmentation_artifact_sha256,
            final_pairing.analysis_audio_relative_path,
            final_pairing.analysis_audio_sha256,
            final_pairing.ffmpeg_version,
            final_pairing.alignment_runtime_sha256s,
            final_pairing.cache_key,
            final_pairing.pairing_parameters,
            final_pairing.windows,
            tuple(outcome.unit_id for outcome in final_pairing.groups),
        )
        if automatic_fixed_identity != final_fixed_identity:
            raise ValueError("automatic pairing identity does not match final pairing")
        total_automatic_groups += len(automatic_pairing.groups)
        automatic_selected += sum(
            outcome.status == "selected" for outcome in automatic_pairing.groups
        )
        for outcome in automatic_pairing.groups:
            _count_issue(issue_counts, outcome.issue_code)

        transcript = transcripts[recording.recording_id]
        units = transcript.get("spoken_units")
        if type(units) is not list or not units:
            raise ValueError("transcript spoken_units must be a non-empty array")
        unit_ids = tuple(unit.get("unit_id") if type(unit) is dict else None for unit in units)
        if any(not isinstance(unit_id, str) or not unit_id for unit_id in unit_ids):
            raise ValueError("transcript spoken unit identity is invalid")
        if len(unit_ids) != len(set(unit_ids)) or len(unit_ids) != len(final_pairing.groups):
            raise ValueError("transcript and pairing text-unit counts differ")
        if tuple(outcome.unit_id for outcome in final_pairing.groups) != unit_ids:
            raise ValueError("transcript and pairing text-unit order differs")
        text_unit_count += len(unit_ids)
        decoded_alignment = _alignment_by_take(final_pairing, alignment, paths)

        for outcome in final_pairing.groups:
            if outcome.status != "selected" or outcome.group is None:
                raise CorpusFailure("REVIEW_REQUIRED", "final pairing still requires review")
            group = outcome.group
            repetition_group_count += 1
            automatic = _read_json(
                run_directory / "review" / group.repetition_group_id / "automatic.json"
            )
            require_exact_fields(automatic, _AUTOMATIC_FIELDS, "review automatic")
            if (
                automatic.get("schema_version") != "1"
                or automatic.get("recording_id") != recording.recording_id
                or automatic.get("repetition_group_id") != group.repetition_group_id
                or automatic.get("unit_id") != group.unit_id
            ):
                raise ValueError("review automatic identity does not match pairing")
            if automatic.get("source_audio") != {
                "relative_path": recording.relative_path,
                "sha256": recording.sha256,
            }:
                raise ValueError("review source audio identity does not match recording")
            if automatic.get("artifact_binding") != _artifact_binding(
                run_directory, final_pairing, alignment, config
            ):
                raise ValueError("review artifact binding is stale or does not match")
            expected_text_layers = {
                "title_or_citation": recording.title_or_citation,
                "source_text": transcript["source_text"],
                "spoken_text": transcript["spoken_text"],
                "normalized_text": transcript["normalized_text"],
                "unit_spoken_text": group.text,
                "alignment_texts": [
                    decoded_alignment[(group.repetition_group_id, index)][
                        "alignment_result"
                    ].alignment_text
                    for index in (1, 2)
                ],
            }
            if automatic.get("text_layers") != expected_text_layers:
                raise ValueError("review text layers do not match pairing and alignment")
            takes = automatic.get("takes")
            if type(takes) is not list or len(takes) != 2:
                raise ValueError("review automatic must contain exactly two takes")
            take_by_index = {take.get("take_index"): take for take in takes if type(take) is dict}
            if set(take_by_index) != {1, 2}:
                raise ValueError("review automatic take identities are invalid")
            for pairing_take in group.takes:
                take = take_by_index[pairing_take.take_index]
                require_exact_fields(take, _AUTOMATIC_TAKE_FIELDS, "review automatic take")
                entity_id = _review_entity_id(
                    recording.recording_id,
                    group.repetition_group_id,
                    pairing_take.take_index,
                )
                if take.get("entity_id") != entity_id:
                    raise ValueError("review entity identity does not match pairing")
                expected_review_entities.add(entity_id)
                provenance = take.get("take_provenance")
                if type(provenance) is not dict:
                    raise TypeError("review take_provenance must be an object")
                source_start = provenance.get("source_start_sample")
                source_end = provenance.get("source_end_sample")
                if (
                    type(source_start) is not int
                    or type(source_end) is not int
                    or source_start != pairing_take.start_sample
                    or source_end != pairing_take.end_sample
                ):
                    raise ValueError("review take provenance does not match pairing")
                expected_start = round(
                    pairing_take.start_sample / _SAMPLE_RATE * recording.metadata.sample_rate
                )
                expected_end = round(
                    pairing_take.end_sample / _SAMPLE_RATE * recording.metadata.sample_rate
                )
                automatic_values = _validate_take_summary(
                    take,
                    pairing_take,
                    decoded_alignment[(group.repetition_group_id, pairing_take.take_index)][
                        "alignment_result"
                    ],
                    recording,
                    (expected_end - expected_start) / recording.metadata.sample_rate,
                )
                entity_events = events_by_entity.get(entity_id, ())
                effective = replay_review_events(
                    automatic_values, entity_events, expected_entity_id=entity_id
                )
                decision = effective["review_decision"]
                if decision not in {"approved", "rejected", "unreviewed"}:
                    raise ValueError("effective review decision is invalid")
                duration = float(effective["segment_end"] - effective["segment_start"])
                decision_by_take[
                    (recording.recording_id, group.repetition_group_id, pairing_take.take_index)
                ] = decision
                identity = (
                    recording.recording_id,
                    group.repetition_group_id,
                    pairing_take.take_index,
                )
                if decision == "approved":
                    expected_segment_evidence[identity] = {
                        "config_sha256": config.digest,
                        "text_unit_id": group.unit_id,
                        "source_text_id": transcript["selected_candidate_id"],
                        "source_audio_sha256": recording.sha256,
                        "source_start_sample": round(
                            (pairing_take.start_sample / _SAMPLE_RATE + effective["segment_start"])
                            * recording.metadata.sample_rate
                        ),
                        "source_end_sample": round(
                            (pairing_take.start_sample / _SAMPLE_RATE + effective["segment_end"])
                            * recording.metadata.sample_rate
                        ),
                        "rights_id": recording.rights_id,
                    }
                reading_count += 1
                if decision == "approved":
                    approved_duration += duration
                elif decision == "rejected":
                    rejected_duration += duration
                    decision_events = tuple(
                        event
                        for event in entity_events
                        if event.field == "review_decision" and event.after == "rejected"
                    )
                    if not decision_events:
                        raise ValueError("rejected take lacks a rejection decision event")
                    rejection_counts[decision_events[-1].reason] += 1
                else:
                    unreviewed_duration += duration
                if decision != "unreviewed" and not any(
                    event.field in _BOUNDARY_FIELDS for event in entity_events
                ):
                    unchanged_boundaries += 1

    unknown_review_entities = {
        event.entity_id
        for event in review_events
        if event.field != "pairing_selected_split"
        and event.entity_id not in expected_review_entities
    }
    if unknown_review_entities:
        raise ValueError("review history contains an unknown review entity")

    for recording in pilot:
        recording_decisions = tuple(
            decision
            for (recording_id, _group_id, _take_index), decision in decision_by_take.items()
            if recording_id == recording.recording_id
        )
        if not recording_decisions:
            raise ValueError("pilot recording has no effective review decisions")
        expected_terminal: CorpusState | None = None
        if "approved" in recording_decisions:
            expected_terminal = CorpusState.APPROVED
        elif all(decision == "rejected" for decision in recording_decisions):
            expected_terminal = CorpusState.REJECTED
        if expected_terminal is not None and recording.state is not expected_terminal:
            raise ValueError("recording terminal state does not match effective review decisions")

    if reading_count == 0 or repetition_group_count == 0 or text_unit_count == 0:
        raise ValueError("pilot report ratio denominators must be non-zero")

    if total_automatic_groups != text_unit_count:
        raise ValueError("automatic pairing must cover every text unit")

    speech_samples = sum(
        end - start for intervals in vad_by_recording.values() for start, end in intervals
    )
    if speech_samples <= 0:
        raise ValueError("VAD speech duration denominator must be non-zero")
    reviewed_count = sum(decision != "unreviewed" for decision in decision_by_take.values())
    if reviewed_count <= 0:
        raise ValueError("reviewed take denominator must be non-zero")
    decision_duration = approved_duration + rejected_duration + unreviewed_duration
    if decision_duration <= 0:
        raise ValueError("review decision duration denominator must be non-zero")

    segments_path = paths.manifests / "segments.jsonl"
    segment_rows = tuple(SegmentRecord.from_dict(raw) for raw in read_jsonl(segments_path))
    segments_sha256 = hashlib.sha256(segments_path.read_bytes()).hexdigest()
    segment_keys = {
        (row.recording_id, row.repetition_group_id, row.take_index) for row in segment_rows
    }
    approved_keys = {
        identity for identity, decision in decision_by_take.items() if decision == "approved"
    }
    if len(segment_keys) != len(segment_rows) or segment_keys != approved_keys:
        raise ValueError("segments.jsonl does not exactly match approved review decisions")
    for row in segment_rows:
        expected = expected_segment_evidence[
            (row.recording_id, row.repetition_group_id, row.take_index)
        ]
        if any(getattr(row, field) != value for field, value in expected.items()):
            raise ValueError("segment row evidence does not match approved review provenance")

    _validate_terminal_processing_evidence(
        recordings=pilot,
        processing_rows=processing_rows,
        config_sha256=config.digest,
        inventory_sha256=inventory_sha256,
        rights_sha256=rights_sha256,
        transcripts_sha256=transcripts_sha256,
        review_sha256=review_sha256,
        segments_sha256=segments_sha256,
        pairing_sha256s=pairing_sha256s,
        alignment_sha256s=alignment_sha256s,
        ffmpeg_versions=ffmpeg_versions,
    )

    metrics = PilotMetrics(
        input_duration_seconds=input_duration,
        speech_duration_seconds=speech_samples / _SAMPLE_RATE,
        auto_pairing_correct_ratio=automatic_selected / total_automatic_groups,
        boundary_unchanged_ratio=unchanged_boundaries / reviewed_count,
        review_minutes_per_audio_minute=telemetry.review_seconds / input_duration,
        approved_speech_ratio=approved_duration / decision_duration,
        gpu_realtime_factor=telemetry.alignment_wall_seconds / input_duration,
        peak_gpu_memory_bytes=telemetry.peak_gpu_memory_bytes,
    )
    scale = full_duration / input_duration
    base_decision = classify_scale_readiness(metrics)
    report = PilotReport(
        metrics=metrics,
        scale_decision=(ScaleDecision.NOT_READY if unreviewed_duration > 0 else base_decision),
        text_unit_count=text_unit_count,
        reading_count=reading_count,
        repetition_group_count=repetition_group_count,
        approved_segment_count=len(segment_rows),
        approved_duration_seconds=approved_duration,
        rejected_duration_seconds=rejected_duration,
        unreviewed_duration_seconds=unreviewed_duration,
        issue_code_counts=tuple(sorted(issue_counts.items())),
        rejection_reason_counts=tuple(sorted(rejection_counts.items())),
        telemetry=telemetry,
        full_corpus_spoken_duration_seconds=full_duration,
        projected_full_corpus_person_hours=(
            (telemetry.text_preparation_seconds + telemetry.review_seconds) * scale / 3600
        ),
        projected_gpu_hours=telemetry.alignment_wall_seconds * scale / 3600,
        projected_storage_bytes=math.ceil(telemetry.pilot_storage_bytes * scale),
    )
    _write_outputs(paths, report)
    return report


def build_report(paths: CorpusPaths, config: CorpusConfig) -> PilotReport:
    """Validate pilot evidence, derive metrics, and atomically publish both reports."""
    if type(paths) is not CorpusPaths or type(config) is not CorpusConfig:
        raise TypeError("build_report requires CorpusPaths and CorpusConfig")
    if config.raw.get("schema_version") != "1" or config.raw.get("corpus_version") != "corpus-v1":
        raise ValueError("report requires schema 1 corpus-v1 configuration")
    with corpus_mutation_lease(
        paths.manifests / "report.json",
        paths.manifests / "report.md",
    ):
        return _build_report_locked(paths, config)
