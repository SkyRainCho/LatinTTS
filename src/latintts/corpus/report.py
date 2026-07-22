from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Any, cast

from latintts.corpus.audio import DerivedAudio, _raw_source, _validate_source_hash
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState, IssueCode
from latintts.corpus.locking import corpus_mutation_lease
from latintts.corpus.manifest import (
    _decode_manifest_attestations,
    _load_rights,
    _ManifestAttestation,
    _validate_manifest_terminal_event,
    _validate_transcript_sources,
)
from latintts.corpus.pairing import pairing_from_dict
from latintts.corpus.paths import CorpusPaths, require_canonical_descendant
from latintts.corpus.records import (
    RecordingRecord,
    ReviewEvent,
    SegmentRecord,
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
_REPORT_INTENT_NAME = ".report-output-transaction.json"
_REPORT_TARGET_NAMES = ("report.json", "report.md")
_REPORT_INTENT_FIELDS = frozenset({"schema_version", "transaction_directory", "outputs"})
_REPORT_INTENT_OUTPUT_FIELDS = frozenset(
    {
        "target_name",
        "prepared_name",
        "backup_name",
        "old_sha256",
        "old_size",
        "new_sha256",
        "new_size",
    }
)


class ScaleDecision(str, Enum):
    SCALABLE = "scalable"
    OPTIMIZE = "optimize"
    NOT_READY = "not_ready"


class ReportOutputRecoveryError(OSError):
    """A durable report transaction needs operator-assisted recovery."""

    code = "REPORT_OUTPUT_RECOVERY_REQUIRED"

    def __init__(self, recovery_paths: tuple[Path, ...], *, manifests_root: Path) -> None:
        self.recovery_paths = tuple(Path(os.path.abspath(path)) for path in recovery_paths)
        root = manifests_root.resolve()
        relative_locations: list[str] = []
        for path in self.recovery_paths:
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = "<invalid-recovery-path>"
            relative_locations.append(f"manifests/{relative}")
        locations = ", ".join(relative_locations) or "manifests/<none>"
        super().__init__(
            f"report output rollback failed; recovery required; preserved backups: {locations}"
        )


class _ReportPublicationVerificationError(OSError):
    """Publication returned but ownership of the resulting path cannot be proven."""


@dataclass(frozen=True, slots=True)
class _ReportOutputSnapshot:
    device: int
    inode: int
    mode: int
    size: int
    sha256: str
    content: bytes = dataclass_field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class _ReportTransactionOutput:
    target_name: str
    prepared_name: str
    backup_name: str | None
    old_sha256: str | None
    old_size: int | None
    new_sha256: str
    new_size: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ReportTransactionIntent:
    schema_version: str
    transaction_directory: str
    outputs: tuple[_ReportTransactionOutput, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transaction_directory": self.transaction_directory,
            "outputs": [output.to_dict() for output in self.outputs],
        }


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
    attestations: tuple[_ManifestAttestation, ...],
) -> None:
    events = validate_processing_events(processing_rows)
    attestation_by_terminal_id = {item.terminal_event_id: item for item in attestations}
    consumed_attestation_ids: set[str] = set()
    for recording in recordings:
        terminal_events = tuple(
            event
            for event in events
            if event.recording_id == recording.recording_id
            and event.previous_state is CorpusState.REVIEWED
            and event.target_state is recording.state
        )
        if len(terminal_events) != 1:
            raise ValueError("processing history lacks one terminal event for a selected recording")
        v2_inputs = (
            recording.sha256,
            inventory_sha256,
            rights_sha256,
            transcripts_sha256,
            pairing_sha256s[recording.recording_id],
            alignment_sha256s[recording.recording_id],
            review_sha256,
            segments_sha256,
        )
        v1_inputs = (
            recording.sha256,
            rights_sha256,
            transcripts_sha256,
            pairing_sha256s[recording.recording_id],
            alignment_sha256s[recording.recording_id],
            review_sha256,
            segments_sha256,
        )
        legacy_inputs = (
            recording.sha256,
            pairing_sha256s[recording.recording_id],
            alignment_sha256s[recording.recording_id],
            review_sha256,
            segments_sha256,
        )
        terminal_event = terminal_events[0]
        version = _validate_manifest_terminal_event(
            terminal_event,
            recording=recording,
            target=recording.state,
            config_sha256=config_sha256,
            v2_inputs=v2_inputs,
            v1_inputs=v1_inputs,
            legacy_inputs=legacy_inputs,
        )
        if version == "v2":
            continue
        attestation = attestation_by_terminal_id.get(terminal_event.event_id)
        if attestation is None:
            raise ValueError(
                "legacy terminal evidence lacks a v2 manifest attestation; rebuild the manifest"
            )
        try:
            attestation_version = _validate_manifest_terminal_event(
                attestation.evidence,
                recording=recording,
                target=recording.state,
                config_sha256=config_sha256,
                v2_inputs=v2_inputs,
                v1_inputs=v1_inputs,
                legacy_inputs=legacy_inputs,
            )
        except ValueError as error:
            raise ValueError(
                "manifest attestation evidence does not bind current evidence"
            ) from error
        if attestation_version != "v2":
            raise ValueError("manifest attestation evidence must be v2")
        consumed_attestation_ids.add(terminal_event.event_id)
    if set(attestation_by_terminal_id) != consumed_attestation_ids:
        raise ValueError("extraneous manifest attestation does not bind an old terminal event")


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
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise
    return temporary


def _report_output_snapshot(path: Path) -> _ReportOutputSnapshot | None:
    if not (path.exists() or path.is_symlink()):
        return None
    canonical = require_canonical_descendant(
        path.parent,
        path,
        kind="report output",
        require_file=True,
    )
    with canonical.open("rb") as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        content = bytearray()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            content.extend(chunk)
        after = os.fstat(handle.fileno())
    current = os.lstat(canonical)
    required_before = (before.st_dev, before.st_ino, before.st_mode, before.st_size)
    required_after = (after.st_dev, after.st_ino, after.st_mode, after.st_size)
    required_current = (current.st_dev, current.st_ino, current.st_mode, current.st_size)
    if (
        not stat.S_ISREG(before.st_mode)
        or required_before != required_after
        or required_after != required_current
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or len(content) != current.st_size
    ):
        raise OSError(f"report output changed while snapshotting: {canonical}")
    return _ReportOutputSnapshot(
        device=current.st_dev,
        inode=current.st_ino,
        mode=current.st_mode,
        size=current.st_size,
        sha256=digest.hexdigest(),
        content=bytes(content),
    )


def _require_report_output_snapshot(
    path: Path,
    expected: _ReportOutputSnapshot | None,
    *,
    operation: str,
) -> None:
    if _report_output_snapshot(path) != expected:
        raise OSError(f"report output changed before {operation}: {path}")


def _prepared_report_snapshot(path: Path) -> _ReportOutputSnapshot:
    snapshot = _report_output_snapshot(path)
    if snapshot is None:
        raise OSError(f"prepared report output is missing: {path}")
    return snapshot


def _validate_derived_audio_file(
    paths: CorpusPaths,
    relative_path: str,
    expected_sha256: str,
    *,
    root: Path,
    kind: str,
) -> None:
    lexical_path = paths.local_data.joinpath(*relative_path.split("/"))
    try:
        canonical = require_canonical_descendant(
            root,
            lexical_path,
            kind=kind,
            require_file=True,
        )
        _validate_source_hash(canonical, expected_sha256, derived=True)
    except CorpusFailure:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise CorpusFailure("CACHE_ARTIFACT_INVALID", f"{kind} is invalid") from error


def _canonical_intent_bytes(intent: _ReportTransactionIntent) -> bytes:
    return (
        json.dumps(
            intent.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _snapshot_matches(
    snapshot: _ReportOutputSnapshot | None,
    sha256: str | None,
    size: int | None,
) -> bool:
    if sha256 is None or size is None:
        return snapshot is None
    return snapshot is not None and snapshot.sha256 == sha256 and snapshot.size == size


def _sync_report_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _transaction_directory(manifests: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name.startswith(".report-output-transaction.")
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("report transaction directory name is invalid")
    directory = require_canonical_descendant(
        manifests,
        manifests / name,
        kind="report transaction directory",
    )
    metadata = os.lstat(directory)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("report transaction directory must be a directory")
    return directory


def _validate_transaction_private_namespace(directory: Path) -> None:
    exact_names = {
        "report.json.displaced",
        "report.md.displaced",
        "report.json.rollback-current",
        "report.md.rollback-current",
        "intent.completed",
    }
    restore_prefixes = (".restore.report.json.", ".restore.report.md.")
    for child in directory.iterdir():
        if child.name not in exact_names and not child.name.startswith(restore_prefixes):
            raise ValueError("report transaction private path is not registered")
        require_canonical_descendant(
            directory,
            child,
            kind="report transaction private path",
            require_file=True,
        )


def _decode_report_intent(raw: object) -> _ReportTransactionIntent:
    if type(raw) is not dict or set(raw) != _REPORT_INTENT_FIELDS:
        raise ValueError("report transaction intent must contain exact fields")
    if raw.get("schema_version") != "1" or type(raw.get("schema_version")) is not str:
        raise ValueError("report transaction intent schema_version must be '1'")
    transaction_directory = raw.get("transaction_directory")
    if not isinstance(transaction_directory, str):
        raise TypeError("report transaction directory must be a string")
    output_rows = raw.get("outputs")
    if type(output_rows) is not list or len(output_rows) != 2:
        raise ValueError("report transaction intent must contain exactly two outputs")
    outputs: list[_ReportTransactionOutput] = []
    for row in output_rows:
        if type(row) is not dict or set(row) != _REPORT_INTENT_OUTPUT_FIELDS:
            raise ValueError("report transaction output must contain exact fields")
        target_name = row.get("target_name")
        prepared_name = row.get("prepared_name")
        backup_name = row.get("backup_name")
        old_sha256 = row.get("old_sha256")
        old_size = row.get("old_size")
        new_sha256 = row.get("new_sha256")
        new_size = row.get("new_size")
        if target_name not in _REPORT_TARGET_NAMES or type(target_name) is not str:
            raise ValueError("report transaction target is invalid")
        if (
            not isinstance(prepared_name, str)
            or Path(prepared_name).name != prepared_name
            or not prepared_name.startswith(f".{target_name}.")
        ):
            raise ValueError("report transaction prepared path is invalid")
        if backup_name is not None and (
            not isinstance(backup_name, str)
            or Path(backup_name).name != backup_name
            or not backup_name.startswith(f".recovery.{target_name}.")
        ):
            raise ValueError("report transaction backup path is invalid")
        old_absent = backup_name is None and old_sha256 is None and old_size is None
        old_present = (
            isinstance(backup_name, str)
            and _is_sha256(old_sha256)
            and type(old_size) is int
            and old_size >= 0
        )
        if not old_absent and not old_present:
            raise ValueError("report transaction old output identity is invalid")
        if not _is_sha256(new_sha256) or type(new_size) is not int or new_size < 0:
            raise ValueError("report transaction new output identity is invalid")
        outputs.append(
            _ReportTransactionOutput(
                target_name=target_name,
                prepared_name=prepared_name,
                backup_name=backup_name,
                old_sha256=old_sha256,
                old_size=old_size,
                new_sha256=cast(str, new_sha256),
                new_size=new_size,
            )
        )
    if tuple(output.target_name for output in outputs) != _REPORT_TARGET_NAMES:
        raise ValueError("report transaction output order is invalid")
    return _ReportTransactionIntent("1", transaction_directory, tuple(outputs))


def _load_report_intent(
    manifests: Path,
) -> tuple[_ReportTransactionIntent, Path, Path, _ReportOutputSnapshot] | None:
    intent_path = manifests / _REPORT_INTENT_NAME
    if not (intent_path.exists() or intent_path.is_symlink()):
        return None
    canonical = require_canonical_descendant(
        manifests,
        intent_path,
        kind="report transaction intent",
        require_file=True,
    )
    raw_bytes = canonical.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("report transaction intent is not valid JSON") from error
    intent = _decode_report_intent(raw)
    if raw_bytes != _canonical_intent_bytes(intent):
        raise ValueError("report transaction intent is not canonical")
    directory = _transaction_directory(manifests, intent.transaction_directory)
    _validate_transaction_private_namespace(directory)
    marker_snapshot = _prepared_report_snapshot(canonical)
    for output in intent.outputs:
        prepared_path = require_canonical_descendant(
            manifests,
            manifests / output.prepared_name,
            kind="report transaction prepared output",
        )
        if prepared_path.exists() or prepared_path.is_symlink():
            prepared = _prepared_report_snapshot(prepared_path)
            if not _snapshot_matches(prepared, output.new_sha256, output.new_size):
                raise ValueError("report transaction prepared output is invalid")
        if output.backup_name is not None:
            backup_path = require_canonical_descendant(
                manifests,
                manifests / output.backup_name,
                kind="report transaction backup",
                require_file=True,
            )
            backup = _prepared_report_snapshot(backup_path)
            if not _snapshot_matches(backup, output.old_sha256, output.old_size):
                raise ValueError("report transaction backup is invalid")
    return intent, canonical, directory, marker_snapshot


def _recovery_paths(
    manifests: Path,
    loaded: tuple[_ReportTransactionIntent, Path, Path, _ReportOutputSnapshot] | None,
) -> tuple[Path, ...]:
    candidates: list[Path] = [manifests / _REPORT_INTENT_NAME]
    candidates.extend(manifests.glob(".recovery.report.*"))
    candidates.extend(manifests.glob(".report-output-transaction.*"))
    if loaded is not None:
        intent, _marker, directory, _snapshot = loaded
        candidates.extend(
            manifests / output.backup_name
            for output in intent.outputs
            if output.backup_name is not None
        )
        candidates.append(directory)
        with suppress(OSError):
            candidates.extend(path for path in directory.iterdir())
    unique: list[Path] = []
    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate))
        if (absolute.exists() or absolute.is_symlink()) and absolute not in unique:
            unique.append(absolute)
    return tuple(unique)


def _claim_report_output(
    target: Path,
    claimed: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    operation: str,
) -> _ReportOutputSnapshot:
    if claimed.exists() or claimed.is_symlink():
        raise OSError(f"report output claim already exists during {operation}")
    os.rename(target, claimed)
    _sync_report_directory(target.parent)
    snapshot = _prepared_report_snapshot(claimed)
    if not _snapshot_matches(snapshot, expected_sha256, expected_size):
        with suppress(OSError):
            os.link(claimed, target)
            _sync_report_directory(target.parent)
        raise _ReportPublicationVerificationError(
            f"report output changed before {operation}: {target.name}"
        )
    return snapshot


def _publish_report_output(
    temporary: Path,
    target: Path,
    *,
    old: _ReportOutputSnapshot | None,
    prepared: _ReportOutputSnapshot,
) -> _ReportOutputSnapshot:
    loaded = _load_report_intent(target.parent)
    if loaded is None:
        raise OSError("report transaction intent is missing before publication")
    intent, _marker, transaction_directory, _marker_snapshot = loaded
    output = next(item for item in intent.outputs if item.target_name == target.name)
    if temporary.name != output.prepared_name:
        raise OSError("prepared report output does not match transaction intent")
    current_prepared = _prepared_report_snapshot(temporary)
    if current_prepared != prepared or not _snapshot_matches(
        prepared, output.new_sha256, output.new_size
    ):
        raise OSError("prepared report output changed before publication")
    if old is None:
        if output.old_sha256 is not None or output.old_size is not None:
            raise OSError("report transaction old output identity is inconsistent")
    else:
        if not _snapshot_matches(old, output.old_sha256, output.old_size):
            raise OSError("report transaction old output identity is inconsistent")
        _claim_report_output(
            target,
            transaction_directory / f"{target.name}.displaced",
            expected_sha256=old.sha256,
            expected_size=old.size,
            operation="publication",
        )
    os.link(temporary, target)
    _sync_report_directory(target.parent)
    try:
        published = _prepared_report_snapshot(target)
    except Exception as error:
        raise _ReportPublicationVerificationError(
            f"published report output cannot be verified: {target.name}"
        ) from error
    if published != prepared:
        raise _ReportPublicationVerificationError(
            f"published report output differs from prepared file: {target.name}"
        )
    return published


def _restore_old_report_output(
    manifests: Path,
    transaction_directory: Path,
    output: _ReportTransactionOutput,
) -> None:
    target = manifests / output.target_name
    current = _report_output_snapshot(target)
    if _snapshot_matches(current, output.old_sha256, output.old_size):
        return
    if output.old_sha256 is None or output.old_size is None or output.backup_name is None:
        if current is None:
            return
        if not _snapshot_matches(current, output.new_sha256, output.new_size):
            raise OSError(f"foreign report output blocks recovery: {target.name}")
        _claim_report_output(
            target,
            transaction_directory / f"{target.name}.rollback-current",
            expected_sha256=output.new_sha256,
            expected_size=output.new_size,
            operation="rollback",
        )
        return
    if current is not None:
        if not _snapshot_matches(current, output.new_sha256, output.new_size):
            raise OSError(f"foreign report output blocks recovery: {target.name}")
        _claim_report_output(
            target,
            transaction_directory / f"{target.name}.rollback-current",
            expected_sha256=output.new_sha256,
            expected_size=output.new_size,
            operation="rollback",
        )
    backup = _prepared_report_snapshot(manifests / output.backup_name)
    if not _snapshot_matches(backup, output.old_sha256, output.old_size):
        raise OSError(f"report output backup is invalid before rollback: {target.name}")
    restore = _prepare_atomic(
        transaction_directory / f"restore.{target.name}",
        backup.content,
    )
    restore_snapshot = _prepared_report_snapshot(restore)
    if not _snapshot_matches(restore_snapshot, output.old_sha256, output.old_size):
        raise OSError(f"report output restore copy is invalid: {target.name}")
    os.link(restore, target)
    _sync_report_directory(target.parent)
    restored = _prepared_report_snapshot(target)
    if not _snapshot_matches(restored, output.old_sha256, output.old_size):
        raise OSError(f"restored report output differs from old content: {target.name}")


def _claim_completed_intent(
    intent_path: Path,
    transaction_directory: Path,
    marker_snapshot: _ReportOutputSnapshot,
) -> None:
    completed = transaction_directory / "intent.completed"
    _claim_report_output(
        intent_path,
        completed,
        expected_sha256=marker_snapshot.sha256,
        expected_size=marker_snapshot.size,
        operation="transaction completion",
    )


def _cleanup_report_transaction(
    manifests: Path,
    intent: _ReportTransactionIntent,
    transaction_directory: Path,
) -> None:
    for output in intent.outputs:
        for name in (output.prepared_name, output.backup_name):
            if name is None:
                continue
            path = manifests / name
            if path.exists() or path.is_symlink():
                canonical = require_canonical_descendant(
                    manifests,
                    path,
                    kind="report transaction cleanup file",
                    require_file=True,
                )
                canonical.unlink()
    for child in tuple(transaction_directory.iterdir()):
        canonical = require_canonical_descendant(
            transaction_directory,
            child,
            kind="report transaction private file",
            require_file=True,
        )
        canonical.unlink()
    transaction_directory.rmdir()
    _sync_report_directory(manifests)


def _recover_report_transaction(manifests: Path) -> None:
    loaded: tuple[_ReportTransactionIntent, Path, Path, _ReportOutputSnapshot] | None = None
    try:
        loaded = _load_report_intent(manifests)
        if loaded is None:
            return
        intent, marker, transaction_directory, marker_snapshot = loaded
        recovery_errors: list[Exception] = []
        for output in intent.outputs:
            try:
                _restore_old_report_output(manifests, transaction_directory, output)
            except Exception as error:
                recovery_errors.append(error)
        if recovery_errors:
            raise recovery_errors[0]
        for output in intent.outputs:
            current = _report_output_snapshot(manifests / output.target_name)
            if not _snapshot_matches(current, output.old_sha256, output.old_size):
                raise OSError("restored report output pair is inconsistent")
        _claim_completed_intent(marker, transaction_directory, marker_snapshot)
        _cleanup_report_transaction(manifests, intent, transaction_directory)
    except ReportOutputRecoveryError:
        raise
    except Exception as error:
        raise ReportOutputRecoveryError(
            _recovery_paths(manifests, loaded),
            manifests_root=manifests,
        ) from error


def _publish_report_intent(manifests: Path, intent: _ReportTransactionIntent) -> Path:
    intent_path = manifests / _REPORT_INTENT_NAME
    temporary = _prepare_atomic(intent_path, _canonical_intent_bytes(intent))
    try:
        os.link(temporary, intent_path)
        _sync_report_directory(manifests)
    finally:
        temporary.unlink(missing_ok=True)
    return intent_path


def _cleanup_unpublished_report_transaction(
    transaction_directory: Path | None,
    artifacts: tuple[Path, ...],
) -> None:
    for artifact in artifacts:
        artifact.unlink(missing_ok=True)
    if transaction_directory is not None:
        for child in tuple(transaction_directory.iterdir()):
            child.unlink(missing_ok=True)
        transaction_directory.rmdir()


def _write_outputs(paths: CorpusPaths, report: PilotReport) -> None:
    manifests = paths.manifests
    json_path = manifests / "report.json"
    markdown_path = manifests / "report.md"
    old_snapshots = (
        _report_output_snapshot(json_path),
        _report_output_snapshot(markdown_path),
    )
    transaction_directory: Path | None = None
    artifacts: list[Path] = []
    intent_published = False
    try:
        transaction_directory = Path(
            tempfile.mkdtemp(dir=manifests, prefix=".report-output-transaction.")
        )
        prepared_items: list[Path] = []
        prepared_items.append(_prepare_atomic(json_path, _canonical_report_bytes(report)))
        artifacts.append(prepared_items[-1])
        prepared_items.append(_prepare_atomic(markdown_path, _render_markdown(report)))
        artifacts.append(prepared_items[-1])
        prepared_paths = tuple(prepared_items)
        prepared_snapshots = tuple(_prepared_report_snapshot(path) for path in prepared_paths)
        backup_paths: list[Path | None] = []
        for target, old in zip((json_path, markdown_path), old_snapshots, strict=True):
            backup = (
                _prepare_atomic(target.with_name(f"recovery.{target.name}"), old.content)
                if old is not None
                else None
            )
            backup_paths.append(backup)
            if backup is not None:
                artifacts.append(backup)
        intent = _ReportTransactionIntent(
            schema_version="1",
            transaction_directory=transaction_directory.name,
            outputs=tuple(
                _ReportTransactionOutput(
                    target_name=target.name,
                    prepared_name=prepared.name,
                    backup_name=backup.name if backup is not None else None,
                    old_sha256=old.sha256 if old is not None else None,
                    old_size=old.size if old is not None else None,
                    new_sha256=prepared_snapshot.sha256,
                    new_size=prepared_snapshot.size,
                )
                for target, prepared, backup, old, prepared_snapshot in zip(
                    (json_path, markdown_path),
                    prepared_paths,
                    backup_paths,
                    old_snapshots,
                    prepared_snapshots,
                    strict=True,
                )
            ),
        )
        _publish_report_intent(manifests, intent)
        intent_published = True
        published = tuple(
            _publish_report_output(prepared, target, old=old, prepared=prepared_snapshot)
            for prepared, target, old, prepared_snapshot in zip(
                prepared_paths,
                (json_path, markdown_path),
                old_snapshots,
                prepared_snapshots,
                strict=True,
            )
        )
        for target, expected in zip((json_path, markdown_path), published, strict=True):
            _require_report_output_snapshot(target, expected, operation="pair commit")
        loaded = _load_report_intent(manifests)
        if loaded is None:
            raise OSError("report transaction intent disappeared before pair commit")
        committed_intent, marker, committed_directory, marker_snapshot = loaded
        _claim_completed_intent(marker, committed_directory, marker_snapshot)
        intent_published = False
        _cleanup_report_transaction(manifests, committed_intent, committed_directory)
    except Exception:
        marker_exists = (manifests / _REPORT_INTENT_NAME).exists() or (
            manifests / _REPORT_INTENT_NAME
        ).is_symlink()
        if intent_published or marker_exists:
            _recover_report_transaction(manifests)
        else:
            with suppress(OSError):
                _cleanup_unpublished_report_transaction(
                    transaction_directory,
                    tuple(artifacts),
                )
        raise


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
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    processing_rows = read_jsonl(processing_path)
    attestation_path = processing_path.with_name("manifest-attestations.jsonl")
    if attestation_path.exists() or attestation_path.is_symlink():
        require_canonical_descendant(
            processing_path.parent,
            attestation_path,
            kind="manifest attestation journal",
            require_file=True,
        )
        attestations = _decode_manifest_attestations(read_jsonl(attestation_path))
    else:
        attestations = ()
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
        _validate_derived_audio_file(
            paths,
            analysis_audio.relative_path,
            analysis_audio.sha256,
            root=paths.normalized,
            kind="analysis audio artifact",
        )
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
        _validate_derived_audio_file(
            paths,
            row.derived_audio_relative_path,
            row.derived_audio_sha256,
            root=paths.segments / "lossless",
            kind="lossless segment artifact",
        )

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
        attestations=attestations,
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
        _recover_report_transaction(paths.manifests)
        return _build_report_locked(paths, config)
