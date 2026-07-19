from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from subprocess import CompletedProcess

from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.intake import load_rights, read_intake
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import (
    AudioMetadata,
    IntakeRow,
    ProcessingEvent,
    RecordingRecord,
    RightsRecord,
    advance_recording,
)
from latintts.corpus.store import persist_recording_transition, read_jsonl, write_jsonl_atomic

RunCommand = Callable[..., CompletedProcess[str]]
SUPPORTED_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"})
_INTAKE_FIELDS = (
    "relative_path",
    "title_or_citation",
    "speaker_id",
    "rights_id",
    "notes",
)
_INVENTORY_CONFIG_SHA256 = hashlib.sha256(b"latintts-corpus-inventory-v1").hexdigest()
_PROCESSING_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "recording_id",
        "previous_state",
        "target_state",
        "input_sha256s",
        "config_sha256",
        "tool_versions",
        "started_at",
        "finished_at",
        "result",
    }
)


class InventoryInputError(ValueError):
    def __init__(self, message: str, code: str = "MANIFEST_SCHEMA_MISMATCH") -> None:
        super().__init__(message)
        self.code = code


def _has_inventory_events(events_path: Path, records: tuple[RecordingRecord, ...]) -> bool:
    if not records:
        return not events_path.exists() or not read_jsonl(events_path)
    if not events_path.exists():
        return False
    raw_events = read_jsonl(events_path)
    if len(raw_events) != len(records):
        return False
    events: list[ProcessingEvent] = []
    for raw in raw_events:
        if (
            set(raw) != _PROCESSING_EVENT_FIELDS
            or type(raw.get("input_sha256s")) is not list
            or type(raw.get("tool_versions")) is not list
        ):
            return False
        try:
            events.append(
                ProcessingEvent(
                    schema_version=raw["schema_version"],
                    event_id=raw["event_id"],
                    recording_id=raw["recording_id"],
                    previous_state=CorpusState(raw["previous_state"]),
                    target_state=CorpusState(raw["target_state"]),
                    input_sha256s=tuple(raw["input_sha256s"]),
                    config_sha256=raw["config_sha256"],
                    tool_versions=tuple(raw["tool_versions"]),
                    started_at=raw["started_at"],
                    finished_at=raw["finished_at"],
                    result=raw["result"],
                )
            )
        except (TypeError, ValueError):
            return False
    if len({event.event_id for event in events}) != len(events):
        return False
    return all(
        any(
            event.recording_id == record.recording_id
            and event.previous_state is CorpusState.DISCOVERED
            and event.target_state is CorpusState.INVENTORIED
            and event.input_sha256s == (record.sha256,)
            and event.config_sha256 == _INVENTORY_CONFIG_SHA256
            and event.tool_versions == ("ffprobe",)
            and event.result == "success"
            for event in events
        )
        for record in records
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_audio_source(paths: CorpusPaths, relative_path: str) -> tuple[Path, str]:
    parts = relative_path.split("/")
    if "\\" in relative_path or any(part in ("", ".", "..") for part in parts):
        raise ValueError("intake audio path must be canonical POSIX")
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or posix_path.as_posix() != relative_path:
        raise ValueError("intake audio path must be canonical POSIX")
    try:
        source = paths.resolve_local(posix_path)
        canonical = paths.relative_local(source)
    except ValueError as error:
        raise ValueError("intake audio path must be canonical POSIX") from error
    if canonical != relative_path:
        raise ValueError("intake audio path must be canonical POSIX")
    return source, canonical


def probe_audio(path: Path, run_command: RunCommand = subprocess.run) -> AudioMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=codec_type,codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = run_command(
            command,
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "ffprobe not found in PATH") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
        message = stderr or f"ffprobe exited with status {error.returncode}"
        raise CorpusFailure("INVENTORY_UNSUPPORTED_FORMAT", message) from error
    try:
        raw = json.loads(completed.stdout)
    except (TypeError, ValueError) as error:
        raise CorpusFailure(
            "INVENTORY_UNSUPPORTED_FORMAT", f"invalid ffprobe output for {path}"
        ) from error
    if type(raw) is not dict:
        raise CorpusFailure("INVENTORY_UNSUPPORTED_FORMAT", f"invalid ffprobe output for {path}")
    format_data = raw.get("format")
    streams = raw.get("streams")
    if type(format_data) is not dict or type(streams) is not list:
        raise CorpusFailure("INVENTORY_UNSUPPORTED_FORMAT", f"invalid ffprobe output for {path}")
    audio_streams: list[dict[str, object]] = []
    for item in streams:
        if type(item) is not dict or type(item.get("codec_type")) is not str:
            raise CorpusFailure(
                "INVENTORY_UNSUPPORTED_FORMAT", f"invalid ffprobe output for {path}"
            )
        if item["codec_type"] == "audio":
            audio_streams.append(item)
    if not audio_streams:
        raise CorpusFailure("INVENTORY_UNSUPPORTED_FORMAT", f"no audio stream: {path}")
    stream = audio_streams[0]
    duration = format_data.get("duration")
    bit_rate = format_data.get("bit_rate")
    sample_rate = stream.get("sample_rate")
    channels = stream.get("channels")
    codec = stream.get("codec_name")
    if (
        type(duration) is not str
        or not (bit_rate is None or type(bit_rate) is str)
        or type(sample_rate) is not str
        or type(channels) is not int
        or type(codec) is not str
    ):
        raise CorpusFailure("INVENTORY_UNSUPPORTED_FORMAT", f"invalid audio metadata for {path}")
    try:
        return AudioMetadata(
            duration_seconds=float(duration),
            sample_rate=int(sample_rate),
            channels=channels,
            codec=codec,
            bit_rate=int(bit_rate) if bit_rate is not None else None,
        )
    except ValueError as error:
        raise CorpusFailure(
            "INVENTORY_UNSUPPORTED_FORMAT", f"invalid audio metadata for {path}"
        ) from error


def inventory_row(
    row: IntakeRow,
    rights: RightsRecord,
    paths: CorpusPaths,
    run_command: RunCommand = subprocess.run,
) -> RecordingRecord:
    source, canonical_path = _canonical_audio_source(paths, row.relative_path)
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise CorpusFailure(
            "INVENTORY_UNSUPPORTED_FORMAT",
            f"unsupported audio suffix: {source.suffix}",
        )
    prefix = PurePosixPath(canonical_path).parts[:2]
    if prefix not in (("raw", "spoken"), ("raw", "sung")):
        raise ValueError("intake audio must be under raw/spoken or raw/sung")
    if rights.rights_id != row.rights_id:
        raise ValueError(f"rights mismatch for {row.relative_path}")
    if rights.speaker_id != row.speaker_id:
        raise ValueError(f"speaker mismatch for {row.relative_path}")
    digest = sha256_file(source)
    return RecordingRecord(
        schema_version="1",
        recording_id=f"rec-{digest[:12]}",
        relative_path=canonical_path,
        sha256=digest,
        content_type="sung" if prefix == ("raw", "sung") else "spoken",
        title_or_citation=row.title_or_citation,
        speaker_id=row.speaker_id,
        rights_id=row.rights_id,
        notes=row.notes,
        metadata=probe_audio(source, run_command),
        state=CorpusState.INVENTORIED,
    )


def write_intake_skeleton(paths: CorpusPaths, output: Path) -> bool:
    if output.exists():
        return False
    discovered = sorted(
        paths.relative_local(path)
        for root in (paths.raw_spoken, paths.raw_sung)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_INTAKE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for relative_path in discovered:
            writer.writerow(
                {
                    "relative_path": relative_path,
                    "title_or_citation": "",
                    "speaker_id": "",
                    "rights_id": "",
                    "notes": "",
                }
            )
    return True


def inventory_from_manifests(
    paths: CorpusPaths,
    *,
    run_command: RunCommand = subprocess.run,
    run_directory: Path | None = None,
    timestamp: str | None = None,
) -> tuple[RecordingRecord, ...]:
    intake_path = paths.manifests / "intake.csv"
    rights_path = paths.manifests / "rights.jsonl"
    try:
        intake_rows = read_intake(intake_path)
    except FileNotFoundError as error:
        raise InventoryInputError(f"missing required manifest: {intake_path.name}") from error
    except (TypeError, ValueError) as error:
        raise InventoryInputError(str(error)) from error
    try:
        rights_by_id = load_rights(rights_path)
    except FileNotFoundError as error:
        raise InventoryInputError(f"missing required manifest: {rights_path.name}") from error
    except (TypeError, ValueError) as error:
        raise InventoryInputError(str(error)) from error
    missing_rights = sorted({row.rights_id for row in intake_rows} - rights_by_id.keys())
    if missing_rights:
        raise InventoryInputError(
            f"unknown rights_id: {', '.join(missing_rights)}",
            code="RIGHTS_SCOPE_UNCONFIRMED",
        )

    try:
        final_records = tuple(
            inventory_row(row, rights_by_id[row.rights_id], paths, run_command)
            for row in sorted(intake_rows, key=lambda item: item.relative_path)
        )
    except (FileNotFoundError, ValueError) as error:
        raise InventoryInputError(str(error)) from error
    recording_ids = [record.recording_id for record in final_records]
    if len(set(recording_ids)) != len(recording_ids):
        raise InventoryInputError("duplicate recording_id derived from identical audio content")

    recordings_path = paths.manifests / "recordings.jsonl"
    event_directory = run_directory or paths.alignments / "runs" / "inventory"
    events_path = event_directory / "processing-events.jsonl"
    final_rows = tuple(record.to_dict() for record in final_records)
    if recordings_path.exists():
        try:
            is_complete = read_jsonl(recordings_path) == final_rows and _has_inventory_events(
                events_path, final_records
            )
        except (OSError, ValueError) as error:
            raise InventoryInputError(
                "existing inventory is inconsistent; manual recovery required"
            ) from error
        if is_complete:
            return final_records
        raise InventoryInputError("existing inventory is inconsistent; manual recovery required")
    if events_path.exists():
        raise InventoryInputError("existing inventory is inconsistent; manual recovery required")

    current_records = tuple(
        replace(record, state=CorpusState.DISCOVERED) for record in final_records
    )
    write_jsonl_atomic(recordings_path, (record.to_dict() for record in current_records))

    event_timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    for index, record in enumerate(current_records):
        updated, event = advance_recording(
            record,
            CorpusState.INVENTORIED,
            input_sha256s=(record.sha256,),
            config_sha256=_INVENTORY_CONFIG_SHA256,
            tool_versions=("ffprobe",),
            started_at=event_timestamp,
            finished_at=event_timestamp,
            result="success",
        )
        target_records = (*current_records[:index], updated, *current_records[index + 1 :])
        persist_recording_transition(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=target_records,
            event=event,
        )
        current_records = target_records
    return current_records


inventory = inventory_from_manifests
