from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.inventory import (
    InventoryInputError,
    inventory_from_manifests,
    write_intake_skeleton,
)
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import (
    AudioMetadata,
    ProcessingEvent,
    RecordingRecord,
    SourceCandidateIntake,
    TranscriptIntakeRow,
    advance_recording,
    require_exact_fields,
)
from latintts.corpus.selection import PilotSelection, select_pilot
from latintts.corpus.store import (
    persist_recording_transition,
    processing_event_exists,
    read_jsonl,
    write_jsonl_atomic,
)
from latintts.corpus.transcripts import (
    TextCandidate,
    TranscriptInputError,
    TranscriptRecord,
    build_text_candidate,
    build_transcript,
    compare_asr_observation,
)
from latintts.domain import PronunciationOverride

_PREPARE_TEXT_CONFIG_SHA256 = hashlib.sha256(b"latintts-prepare-text-v1").hexdigest()
_OVERRIDE_FIELDS = frozenset({"stress_index", "ipa", "model_phonemes"})


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


def _transcript_intake_skeleton(recording_id: str) -> dict[str, Any]:
    return TranscriptIntakeRow(
        recording_id=recording_id,
        source_candidates=(SourceCandidateIntake("", "", "", "", "", True),),
        spoken_units_file="",
        pronunciation_overrides_file="",
        confirmed=False,
        asr_hypothesis_file="",
    ).to_dict()


def _init_transcript_intake(paths: CorpusPaths, *, replace_existing: bool) -> None:
    selection_path = paths.manifests / "pilot-selection.json"
    if not selection_path.exists():
        raise TranscriptInputError("missing required manifest: pilot-selection.json")
    selection = _load_selection(selection_path)
    intake_path = paths.manifests / "transcript-intake.jsonl"
    expected_rows = tuple(
        _transcript_intake_skeleton(recording_id) for recording_id in selection.recording_ids
    )
    if intake_path.exists():
        try:
            existing_rows = read_jsonl(intake_path)
        except ValueError as error:
            raise TranscriptInputError(
                f"existing transcript-intake.jsonl is invalid: {error}"
            ) from error
        if existing_rows == expected_rows:
            return
        if not replace_existing:
            raise TranscriptInputError("existing transcript-intake.jsonl differs; use --replace")
    write_jsonl_atomic(
        intake_path,
        expected_rows,
    )


def _load_transcript_intake(path: Path) -> tuple[TranscriptIntakeRow, ...]:
    if not path.exists():
        raise TranscriptInputError("missing required manifest: transcript-intake.jsonl")
    try:
        rows = read_jsonl(path)
        intake = tuple(TranscriptIntakeRow.from_dict(raw) for raw in rows)
    except (KeyError, TypeError, ValueError) as error:
        raise TranscriptInputError(f"transcript-intake.jsonl is invalid: {error}") from error
    recording_ids = tuple(row.recording_id for row in intake)
    if len(recording_ids) != len(set(recording_ids)):
        raise TranscriptInputError("transcript intake contains duplicate recording_id")
    return intake


def _read_local_text(paths: CorpusPaths, relative_path: str, field: str) -> str:
    if not relative_path.strip():
        raise TranscriptInputError(f"{field} must be a non-empty local-data path")
    try:
        path = paths.resolve_local(relative_path)
    except ValueError as error:
        raise TranscriptInputError(f"{field}: {error}") from error
    if not path.is_file():
        raise TranscriptInputError(f"{field} does not identify a file beneath local-data")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise TranscriptInputError(f"unable to read {field} as UTF-8: {error}") from error


def _pronunciation_review_path(paths: CorpusPaths, recording_id: str) -> Path:
    slug = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()
    manifests = paths.manifests.resolve()
    destination = (paths.manifests / f"pronunciation-review-{slug}.json").resolve()
    try:
        destination.relative_to(manifests)
    except ValueError as error:
        raise TranscriptInputError("pronunciation review path escapes manifests") from error
    if destination.parent != manifests:
        raise TranscriptInputError("pronunciation review path must be directly beneath manifests")
    return destination


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _read_json_object(paths: CorpusPaths, relative_path: str, field: str) -> dict[str, Any]:
    text = _read_local_text(paths, relative_path, field)
    try:
        raw = json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except ValueError as error:
        raise TranscriptInputError(f"{field} is invalid JSON: {error}") from error
    if type(raw) is not dict:
        raise TranscriptInputError(f"{field} must contain a JSON object")
    return raw


def _parse_pronunciation_overrides(
    paths: CorpusPaths,
    relative_path: str,
    spoken_text: str,
) -> dict[int, PronunciationOverride]:
    raw = _read_json_object(paths, relative_path, "pronunciation_overrides_file")
    require_exact_fields(
        raw,
        frozenset({"spoken_text_sha256", "overrides"}),
        "pronunciation override file",
    )
    expected_hash = hashlib.sha256(spoken_text.encode("utf-8")).hexdigest()
    if raw["spoken_text_sha256"] != expected_hash:
        raise TranscriptInputError("pronunciation override spoken-text hash does not match")
    if type(raw["overrides"]) is not dict:
        raise TranscriptInputError("pronunciation overrides must be an object")
    parsed: dict[int, PronunciationOverride] = {}
    for raw_index, value in raw["overrides"].items():
        if (
            not isinstance(raw_index, str)
            or not raw_index.isdecimal()
            or str(int(raw_index)) != raw_index
        ):
            raise TranscriptInputError("pronunciation override keys must be token indexes")
        if type(value) is not dict:
            raise TranscriptInputError("pronunciation override values must be objects")
        if not value:
            raise TranscriptInputError("pronunciation override objects must not be empty")
        try:
            require_exact_fields(value, _OVERRIDE_FIELDS, "pronunciation override")
        except ValueError as error:
            raise TranscriptInputError(str(error)) from error
        stress_index = value["stress_index"]
        ipa = value["ipa"]
        phonemes = value["model_phonemes"]
        if stress_index is not None and type(stress_index) is not int:
            raise TranscriptInputError("override stress_index must be an integer or null")
        if ipa is not None and not isinstance(ipa, str):
            raise TranscriptInputError("override ipa must be a string or null")
        if phonemes is not None:
            if type(phonemes) is not list or any(not isinstance(item, str) for item in phonemes):
                raise TranscriptInputError(
                    "override model_phonemes must be an array of strings or null"
                )
            model_phonemes = tuple(phonemes)
        else:
            model_phonemes = None
        parsed[int(raw_index)] = PronunciationOverride(stress_index, ipa, model_phonemes)
    return parsed


def _validate_source_candidates(
    paths: CorpusPaths,
    row: TranscriptIntakeRow,
) -> tuple[tuple[TextCandidate, ...], str]:
    if not row.source_candidates:
        raise TranscriptInputError("source_candidates must contain at least one candidate")
    selected = tuple(candidate for candidate in row.source_candidates if candidate.selected)
    if len(selected) != 1:
        raise TranscriptInputError("source_candidates must contain exactly one selected=true")
    built = []
    selected_candidate_id = ""
    for candidate in row.source_candidates:
        source_text = _read_local_text(paths, candidate.source_file, "source_file")
        try:
            text_candidate = build_text_candidate(
                source_id=candidate.source_id,
                source_url=candidate.source_url,
                source_version=candidate.source_version,
                accessed_at=candidate.accessed_at,
                source_text=source_text,
            )
        except (AttributeError, ValueError) as error:
            raise TranscriptInputError(f"source candidate is invalid: {error}") from error
        built.append(text_candidate)
        if candidate.selected:
            selected_candidate_id = text_candidate.candidate_id
    return tuple(built), selected_candidate_id


def _build_transcript_from_intake(
    paths: CorpusPaths,
    row: TranscriptIntakeRow,
) -> TranscriptRecord:
    candidates, selected_candidate_id = _validate_source_candidates(paths, row)
    units_source = _read_local_text(paths, row.spoken_units_file, "spoken_units_file")
    units = tuple(line.strip() for line in units_source.splitlines() if line.strip())
    spoken_text = "\n".join(units)
    overrides = None
    if row.pronunciation_overrides_file:
        overrides = _parse_pronunciation_overrides(
            paths,
            row.pronunciation_overrides_file,
            spoken_text,
        )
    try:
        transcript = build_transcript(
            recording_id=row.recording_id,
            source_candidates=candidates,
            selected_candidate_id=selected_candidate_id,
            spoken_unit_lines=units,
            pronunciation_overrides=overrides,
        )
    except ValueError as error:
        raise TranscriptInputError(f"transcript is invalid: {error}") from error
    if row.asr_hypothesis_file:
        hypothesis = _read_local_text(
            paths,
            row.asr_hypothesis_file,
            "asr_hypothesis_file",
        ).strip()
        if not hypothesis:
            raise TranscriptInputError("asr_hypothesis_file must not be empty")
        transcript = replace(
            transcript,
            asr_observation=compare_asr_observation(transcript.spoken_text, hypothesis),
        )
    return transcript


def _advance_transcript_recordings(
    paths: CorpusPaths,
    recordings: tuple[RecordingRecord, ...],
    transcripts: tuple[TranscriptRecord, ...],
) -> None:
    current = recordings
    events_path = paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    recordings_path = paths.manifests / "recordings.jsonl"
    for transcript in transcripts:
        index = next(
            index
            for index, record in enumerate(current)
            if record.recording_id == transcript.recording_id
        )
        record = current[index]
        candidate_hashes = tuple(
            candidate.source_sha256 for candidate in transcript.source_candidates
        )
        spoken_hash = hashlib.sha256(transcript.spoken_text.encode("utf-8")).hexdigest()
        timestamp = datetime.now(timezone.utc).isoformat()
        if record.state is CorpusState.INVENTORIED:
            ready_record, ready_event = advance_recording(
                record,
                CorpusState.TEXT_CANDIDATES_READY,
                input_sha256s=candidate_hashes,
                config_sha256=_PREPARE_TEXT_CONFIG_SHA256,
                tool_versions=("latintts-prepare-text-v1",),
                started_at=timestamp,
                finished_at=timestamp,
                result="success",
            )
            ready_records = (*current[:index], ready_record, *current[index + 1 :])
            persist_recording_transition(
                recordings_path=recordings_path,
                events_path=events_path,
                recordings=ready_records,
                event=ready_event,
            )
            current = ready_records
            record = ready_record
        if record.state is CorpusState.TRANSCRIPT_CONFIRMED:
            continue
        confirmed_record, confirmed_event = advance_recording(
            record,
            CorpusState.TRANSCRIPT_CONFIRMED,
            input_sha256s=(*candidate_hashes, spoken_hash),
            config_sha256=_PREPARE_TEXT_CONFIG_SHA256,
            tool_versions=("latintts-prepare-text-v1", "ecclesiastical-roman-v1"),
            started_at=timestamp,
            finished_at=timestamp,
            result="success",
        )
        confirmed_records = (
            *current[:index],
            confirmed_record,
            *current[index + 1 :],
        )
        persist_recording_transition(
            recordings_path=recordings_path,
            events_path=events_path,
            recordings=confirmed_records,
            event=confirmed_event,
        )
        current = confirmed_records


def _expected_transcript_events(
    record: RecordingRecord,
    transcript: TranscriptRecord,
) -> tuple[ProcessingEvent, ProcessingEvent]:
    candidate_hashes = tuple(candidate.source_sha256 for candidate in transcript.source_candidates)
    spoken_hash = hashlib.sha256(transcript.spoken_text.encode("utf-8")).hexdigest()
    timestamp = "1970-01-01T00:00:00+00:00"
    inventoried = replace(record, state=CorpusState.INVENTORIED)
    ready, ready_event = advance_recording(
        inventoried,
        CorpusState.TEXT_CANDIDATES_READY,
        input_sha256s=candidate_hashes,
        config_sha256=_PREPARE_TEXT_CONFIG_SHA256,
        tool_versions=("latintts-prepare-text-v1",),
        started_at=timestamp,
        finished_at=timestamp,
        result="success",
    )
    _, confirmed_event = advance_recording(
        ready,
        CorpusState.TRANSCRIPT_CONFIRMED,
        input_sha256s=(*candidate_hashes, spoken_hash),
        config_sha256=_PREPARE_TEXT_CONFIG_SHA256,
        tool_versions=("latintts-prepare-text-v1", "ecclesiastical-roman-v1"),
        started_at=timestamp,
        finished_at=timestamp,
        result="success",
    )
    return ready_event, confirmed_event


def _preflight_transcript_events(
    paths: CorpusPaths,
    records: tuple[RecordingRecord, ...],
    transcripts: tuple[TranscriptRecord, ...],
) -> None:
    events_path = paths.alignments / "runs" / "prepare-text" / "processing-events.jsonl"
    existing_events = read_jsonl(events_path) if events_path.exists() else ()
    for record, transcript in zip(records, transcripts, strict=True):
        ready_event, confirmed_event = _expected_transcript_events(record, transcript)
        ready_exists = processing_event_exists(existing_events, ready_event)
        confirmed_exists = processing_event_exists(existing_events, confirmed_event)
        if confirmed_exists and not ready_exists:
            raise TranscriptInputError(
                f"recording {record.recording_id} confirmed event lacks ready predecessor"
            )
        if (
            record.state
            in {
                CorpusState.TEXT_CANDIDATES_READY,
                CorpusState.TRANSCRIPT_CONFIRMED,
            }
            and not ready_exists
        ):
            raise TranscriptInputError(
                f"recording {record.recording_id} state lacks durable ready event"
            )
        if record.state is CorpusState.TRANSCRIPT_CONFIRMED and not confirmed_exists:
            raise TranscriptInputError(
                f"recording {record.recording_id} state lacks durable confirmed event"
            )


def _is_controlled_pronunciation_resolution(
    existing: dict[str, Any],
    proposed: dict[str, Any],
) -> bool:
    existing_facts = {key: value for key, value in existing.items() if key != "pronunciation_plan"}
    proposed_facts = {key: value for key, value in proposed.items() if key != "pronunciation_plan"}
    if proposed_facts != existing_facts:
        return False
    existing_plan = existing["pronunciation_plan"]
    proposed_plan = proposed["pronunciation_plan"]
    for key in ("schema_version", "rule_version", "original_text", "normalized_text"):
        if proposed_plan[key] != existing_plan[key]:
            return False
    existing_warnings = Counter(existing_plan["warning_codes"])
    proposed_warnings = Counter(proposed_plan["warning_codes"])
    if not proposed_warnings < existing_warnings:
        return False
    existing_tokens = existing_plan["tokens"]
    proposed_tokens = proposed_plan["tokens"]
    if len(proposed_tokens) != len(existing_tokens):
        return False
    for existing_token, proposed_token in zip(
        existing_tokens,
        proposed_tokens,
        strict=True,
    ):
        token_warnings = Counter(existing_token["warning_codes"])
        if not token_warnings and proposed_token != existing_token:
            return False
        if Counter(proposed_token["warning_codes"]) - token_warnings:
            return False
    return True


def _prepare_text(paths: CorpusPaths) -> None:
    selection_path = paths.manifests / "pilot-selection.json"
    if not selection_path.exists():
        raise TranscriptInputError("missing required manifest: pilot-selection.json")
    selection = _load_selection(selection_path)
    intake = _load_transcript_intake(paths.manifests / "transcript-intake.jsonl")
    intake_ids = tuple(row.recording_id for row in intake)
    if intake_ids != selection.recording_ids:
        raise TranscriptInputError(
            "transcript intake recording IDs must match pilot selection order"
        )
    if any(not row.confirmed for row in intake):
        raise TranscriptInputError("every transcript intake row must set confirmed=true")
    recordings = _load_recordings(paths)
    by_id = {record.recording_id: record for record in recordings}
    pilot_records: list[RecordingRecord] = []
    for recording_id, inventory_hash in zip(
        selection.recording_ids,
        selection.inventory_hashes,
        strict=True,
    ):
        record = by_id.get(recording_id)
        if record is None or record.sha256 != inventory_hash:
            raise TranscriptInputError("pilot selection does not match recordings.jsonl")
        pilot_records.append(record)
    allowed_states = {
        CorpusState.INVENTORIED,
        CorpusState.TEXT_CANDIDATES_READY,
        CorpusState.TRANSCRIPT_CONFIRMED,
    }
    if any(record.state not in allowed_states for record in pilot_records):
        raise TranscriptInputError(
            "pilot recordings must be INVENTORIED, TEXT_CANDIDATES_READY, or TRANSCRIPT_CONFIRMED"
        )
    transcripts = tuple(_build_transcript_from_intake(paths, row) for row in intake)
    transcript_rows = tuple(
        json.loads(json.dumps(transcript.to_dict(), ensure_ascii=False))
        for transcript in transcripts
    )
    _preflight_transcript_events(paths, tuple(pilot_records), transcripts)
    review_recording_ids = [
        transcript.recording_id
        for transcript in transcripts
        if transcript.pronunciation_plan["warning_codes"]
    ]
    transcripts_path = paths.manifests / "transcripts.jsonl"
    resolved_recording_ids: list[str] = []
    if transcripts_path.exists():
        existing_rows = read_jsonl(transcripts_path)
        if len(existing_rows) != len(transcript_rows):
            raise TranscriptInputError("transcript inputs differ from transcripts.jsonl")
        for row, record, existing, proposed in zip(
            intake,
            pilot_records,
            existing_rows,
            transcript_rows,
            strict=True,
        ):
            if proposed == existing:
                continue
            review_path = _pronunciation_review_path(paths, row.recording_id)
            if (
                record.state is not CorpusState.TRANSCRIPT_CONFIRMED
                or not row.pronunciation_overrides_file
                or not review_path.is_file()
                or not _is_controlled_pronunciation_resolution(existing, proposed)
            ):
                raise TranscriptInputError("transcript inputs differ from transcripts.jsonl")
            review = read_jsonl(review_path)
            expected_hash = hashlib.sha256(proposed["spoken_text"].encode("utf-8")).hexdigest()
            if (
                len(review) != 1
                or review[0].get("recording_id") != row.recording_id
                or review[0].get("spoken_text_sha256") != expected_hash
            ):
                raise TranscriptInputError("pronunciation review template does not match")
            resolved_recording_ids.append(row.recording_id)
        if resolved_recording_ids:
            write_jsonl_atomic(transcripts_path, transcript_rows)
            for recording_id in resolved_recording_ids:
                review_path = _pronunciation_review_path(paths, recording_id)
                review_path.unlink()
    else:
        if any(record.state is not CorpusState.INVENTORIED for record in pilot_records):
            raise TranscriptInputError(
                "resumable recording state requires existing transcripts.jsonl"
            )
        write_jsonl_atomic(transcripts_path, transcript_rows)
    for transcript in transcripts:
        review_tokens = [
            {
                "token_index": token_index,
                "surface": token["surface"],
                "candidate_stress_index": token["stress_index"],
                "warning_codes": token["warning_codes"],
            }
            for token_index, token in enumerate(transcript.pronunciation_plan["tokens"])
            if token["warning_codes"]
        ]
        if not review_tokens:
            continue
        write_jsonl_atomic(
            _pronunciation_review_path(paths, transcript.recording_id),
            (
                {
                    "recording_id": transcript.recording_id,
                    "spoken_text_sha256": hashlib.sha256(
                        transcript.spoken_text.encode("utf-8")
                    ).hexdigest(),
                    "review_tokens": review_tokens,
                    "override_template": {
                        str(item["token_index"]): {
                            "stress_index": None,
                            "ipa": None,
                            "model_phonemes": None,
                        }
                        for item in review_tokens
                    },
                },
            ),
        )
    _advance_transcript_recordings(paths, recordings, transcripts)
    if review_recording_ids:
        raise CorpusFailure(
            "PRONUNCIATION_NEEDS_REVIEW",
            "pronunciation review required for recording IDs: " + ", ".join(review_recording_ids),
        )


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
    prepare_text_parser = subparsers.add_parser("prepare-text")
    prepare_text_parser.add_argument("--init", action="store_true")
    prepare_text_parser.add_argument("--replace", action="store_true")
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
    if args.command == "prepare-text":
        paths = CorpusPaths.from_project_root(args.project_root)
        paths.ensure_layout()
        try:
            if args.init:
                _init_transcript_intake(paths, replace_existing=args.replace)
            else:
                if args.replace:
                    raise TranscriptInputError("--replace is only valid with --init")
                _prepare_text(paths)
        except TranscriptInputError as error:
            print(f"{error.code}: {error}", file=sys.stderr)
            return 2
        except CorpusFailure as error:
            print(f"{error.code}: {error}", file=sys.stderr)
            return 1
        except InventoryInputError as error:
            print(f"{error.code}: {error}", file=sys.stderr)
            return 2
        except ValueError as error:
            print(f"MANIFEST_SCHEMA_MISMATCH: {error}", file=sys.stderr)
            return 2
        return 0
    raise AssertionError(args.command)
