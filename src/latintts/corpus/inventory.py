from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from subprocess import CompletedProcess

from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.intake import load_rights, read_intake
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import (
    AudioMetadata,
    IntakeRow,
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


def _has_inventory_events(events_path: Path, records: tuple[RecordingRecord, ...]) -> bool:
    if not records:
        return True
    if not events_path.exists():
        return False
    events = read_jsonl(events_path)
    return all(
        any(
            event.get("recording_id") == record.recording_id
            and event.get("previous_state") == CorpusState.DISCOVERED.value
            and event.get("target_state") == CorpusState.INVENTORIED.value
            and event.get("input_sha256s") == [record.sha256]
            and event.get("config_sha256") == _INVENTORY_CONFIG_SHA256
            and event.get("result") == "success"
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
        audio_streams = [item for item in raw["streams"] if item.get("codec_type") == "audio"]
    except (KeyError, TypeError, ValueError) as error:
        raise CorpusFailure(
            "INVENTORY_UNSUPPORTED_FORMAT", f"invalid ffprobe output for {path}"
        ) from error
    if not audio_streams:
        raise CorpusFailure("INVENTORY_UNSUPPORTED_FORMAT", f"no audio stream: {path}")
    try:
        stream = audio_streams[0]
        bit_rate = raw["format"].get("bit_rate")
        return AudioMetadata(
            duration_seconds=float(raw["format"]["duration"]),
            sample_rate=int(stream["sample_rate"]),
            channels=int(stream["channels"]),
            codec=str(stream["codec_name"]),
            bit_rate=int(bit_rate) if bit_rate is not None else None,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CorpusFailure(
            "INVENTORY_UNSUPPORTED_FORMAT", f"invalid audio metadata for {path}"
        ) from error


def inventory_row(
    row: IntakeRow,
    rights: RightsRecord,
    paths: CorpusPaths,
    run_command: RunCommand = subprocess.run,
) -> RecordingRecord:
    source = paths.resolve_local(row.relative_path)
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise CorpusFailure(
            "INVENTORY_UNSUPPORTED_FORMAT",
            f"unsupported audio suffix: {source.suffix}",
        )
    prefix = Path(row.relative_path).parts[:2]
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
        relative_path=Path(row.relative_path).as_posix(),
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
    intake_rows = read_intake(paths.manifests / "intake.csv")
    rights_by_id = load_rights(paths.manifests / "rights.jsonl")
    missing_rights = sorted({row.rights_id for row in intake_rows} - rights_by_id.keys())
    if missing_rights:
        raise ValueError(f"unknown rights_id: {', '.join(missing_rights)}")

    final_records = tuple(
        inventory_row(row, rights_by_id[row.rights_id], paths, run_command)
        for row in sorted(intake_rows, key=lambda item: item.relative_path)
    )
    recording_ids = [record.recording_id for record in final_records]
    if len(set(recording_ids)) != len(recording_ids):
        raise ValueError("duplicate recording_id derived from identical audio content")

    recordings_path = paths.manifests / "recordings.jsonl"
    event_directory = run_directory or paths.alignments / "runs" / "inventory"
    events_path = event_directory / "processing-events.jsonl"
    final_rows = tuple(record.to_dict() for record in final_records)
    if (
        recordings_path.exists()
        and read_jsonl(recordings_path) == final_rows
        and _has_inventory_events(events_path, final_records)
    ):
        return final_records

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
