from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.inventory import (
    InventoryInputError,
    inventory_from_manifests,
    write_intake_skeleton,
)
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import AudioMetadata, RecordingRecord, require_exact_fields
from latintts.corpus.selection import PilotSelection, select_pilot
from latintts.corpus.store import read_jsonl, write_jsonl_atomic


def _doctor(project_root: Path) -> int:
    CorpusPaths.from_project_root(project_root).ensure_layout()
    failures: list[str] = []
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            failures.append(f"ALIGNER_UNAVAILABLE: {executable} not found in PATH")
    for package in ("silero_vad", "ctc_forced_aligner"):
        if importlib.util.find_spec(package) is None:
            failures.append(f"ALIGNER_UNAVAILABLE: optional package {package} is not installed")
    for message in failures:
        print(message)
    return 1 if failures else 0


def _decode_recording(raw: dict[str, Any]) -> RecordingRecord:
    try:
        require_exact_fields(
            raw,
            frozenset(field.name for field in fields(RecordingRecord)),
            "recording row",
        )
        metadata = raw["metadata"]
        if type(metadata) is not dict:
            raise TypeError("recording metadata must be an object")
        require_exact_fields(
            metadata,
            frozenset(field.name for field in fields(AudioMetadata)),
            "recording metadata",
        )
        return RecordingRecord(
            schema_version=raw["schema_version"],
            recording_id=raw["recording_id"],
            relative_path=raw["relative_path"],
            sha256=raw["sha256"],
            content_type=raw["content_type"],
            title_or_citation=raw["title_or_citation"],
            speaker_id=raw["speaker_id"],
            rights_id=raw["rights_id"],
            notes=raw["notes"],
            metadata=AudioMetadata(**metadata),
            state=CorpusState(raw["state"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise InventoryInputError(f"recording row is invalid: {error}") from error


def _load_recordings(paths: CorpusPaths) -> tuple[RecordingRecord, ...]:
    recordings_path = paths.manifests / "recordings.jsonl"
    if not recordings_path.exists():
        raise InventoryInputError("missing required manifest: recordings.jsonl")
    try:
        records = tuple(_decode_recording(raw) for raw in read_jsonl(recordings_path))
    except ValueError as error:
        raise InventoryInputError(str(error)) from error
    ids = tuple(record.recording_id for record in records)
    if len(ids) != len(set(ids)):
        raise InventoryInputError("recordings.jsonl contains duplicate recording_id")
    return records


def _load_selection(path: Path) -> PilotSelection:
    try:
        rows = read_jsonl(path)
        if len(rows) != 1:
            raise ValueError("pilot-selection.json must contain exactly one row")
        raw = rows[0]
        require_exact_fields(
            raw,
            frozenset(field.name for field in fields(PilotSelection)),
            "pilot selection row",
        )
        if type(raw["recording_ids"]) is not list or type(raw["inventory_hashes"]) is not list:
            raise TypeError("pilot selection IDs and hashes must be arrays")
        return PilotSelection(
            schema_version=raw["schema_version"],
            strategy=raw["strategy"],
            recording_ids=tuple(raw["recording_ids"]),
            inventory_hashes=tuple(raw["inventory_hashes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise InventoryInputError(f"pilot-selection.json is invalid: {error}") from error


def _select_pilot(paths: CorpusPaths, explicit_ids: tuple[str, ...], replace: bool) -> None:
    selection = select_pilot(_load_recordings(paths), explicit_ids=explicit_ids)
    selection_path = paths.manifests / "pilot-selection.json"
    if selection_path.exists():
        existing = _load_selection(selection_path)
        if existing.inventory_hashes != selection.inventory_hashes and not replace:
            raise InventoryInputError(
                "existing pilot selection has different inventory hashes; use --replace"
            )
    write_jsonl_atomic(selection_path, (selection.to_dict(),))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the local LatinTTS corpus")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--init-intake", action="store_true")
    selection_parser = subparsers.add_parser("select-pilot")
    selection_parser.add_argument("--recording-id", action="append", default=[])
    selection_parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.project_root)
    if args.command == "inventory":
        paths = CorpusPaths.from_project_root(args.project_root)
        paths.ensure_layout()
        if args.init_intake:
            write_intake_skeleton(paths, paths.manifests / "intake.csv")
        else:
            try:
                inventory_from_manifests(paths)
            except CorpusFailure as error:
                print(f"{error.code}: {error}", file=sys.stderr)
                return 1
            except InventoryInputError as error:
                print(f"{error.code}: {error}", file=sys.stderr)
                return 2
        return 0
    if args.command == "select-pilot":
        paths = CorpusPaths.from_project_root(args.project_root)
        paths.ensure_layout()
        try:
            _select_pilot(paths, tuple(args.recording_id), args.replace)
        except InventoryInputError as error:
            print(f"{error.code}: {error}", file=sys.stderr)
            return 2
        except ValueError as error:
            print(f"MANIFEST_SCHEMA_MISMATCH: {error}", file=sys.stderr)
            return 2
        return 0
    raise AssertionError(args.command)
