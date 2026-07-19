from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latintts.corpus.alignment import result_to_dict
from latintts.corpus.audio import DerivedAudio, RunCommand, derive_analysis_audio
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.inventory import (
    InventoryInputError,
    inventory_from_manifests,
    write_intake_skeleton,
)
from latintts.corpus.mms_alignment import create_alignment_backend
from latintts.corpus.pairing import (
    PairingBackend,
    PairingParameters,
    PairingRecording,
    load_selected_alignments,
    map_text_units,
    pair_recording,
    pairing_from_dict,
    pairing_run_directory,
    require_safe_pairing_id,
)
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.pauses import PauseAnalysis, PauseInterval, classify_pauses
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
    SpokenUnit,
    TextCandidate,
    TranscriptInputError,
    TranscriptRecord,
    build_text_candidate,
    build_transcript,
    compare_asr_observation,
)
from latintts.corpus.vad import (
    SileroVadBackend,
    SpeechInterval,
    VadBackend,
    VadFrame,
    VadResult,
)
from latintts.domain import PronunciationOverride
from latintts.normalization import normalize_phrase, normalize_word, tokenize_words

_PREPARE_TEXT_CONFIG_SHA256 = hashlib.sha256(b"latintts-prepare-text-v1").hexdigest()
_OVERRIDE_FIELDS = frozenset({"stress_index", "ipa", "model_phonemes"})
_SILERO_VERSION = "6.2.1"
_CONFIG_FIELDS = frozenset(
    {"schema_version", "corpus_version", "analysis", "vad", "pause", "pairing", "alignment"}
)
_VAD_FIELDS = frozenset(
    {
        "backend",
        "model_version",
        "threshold",
        "neg_threshold",
        "min_speech_duration_ms",
        "min_silence_duration_ms",
        "max_speech_duration_s",
        "speech_pad_ms",
        "min_silence_at_max_speech",
        "use_max_poss_sil_at_max_speech",
        "sample_rate",
        "window_samples",
    }
)
_PAUSE_FIELDS = frozenset(
    {
        "profile_id",
        "minimum_gap_ms",
        "minimum_gap_count",
        "separation_ratio",
        "maximum_iterations",
        "minimum_cluster_size",
        "maximum_cluster_imbalance_ratio",
    }
)
_PAIRING_FIELDS = frozenset(
    {"minimum_duration_ratio", "maximum_duration_ratio", "minimum_score_margin"}
)
_SEGMENT_FIELDS = frozenset(
    {
        "schema_version",
        "recording_id",
        "status",
        "issue_code",
        "config_sha256",
        "cache_key",
        "input_sha256s",
        "analysis_audio",
        "vad",
        "pause",
    }
)


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


def _canonical_digest(raw: object) -> str:
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _segmentation_cache_key(
    recording_id: str,
    config_sha256: str,
    input_sha256s: list[str],
) -> str:
    return _canonical_digest(
        {
            "schema_version": "1",
            "recording_id": recording_id,
            "config_sha256": config_sha256,
            "input_sha256s": input_sha256s,
        }
    )


def _validate_segmentation_config(config: CorpusConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(config.raw) is not dict or set(config.raw) != _CONFIG_FIELDS:
        raise ValueError("corpus config must contain exact top-level fields")
    if _canonical_digest(config.raw) != config.digest:
        raise ValueError("corpus config digest does not match its parameters")
    analysis = config.raw["analysis"]
    vad = config.raw["vad"]
    pause = config.raw["pause"]
    if analysis != {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}:
        raise ValueError("segment analysis config must be 16 kHz mono PCM16")
    if type(vad) is not dict or set(vad) != _VAD_FIELDS:
        raise ValueError("VAD config must contain exact fields")
    if vad["backend"] != "silero-vad" or vad["model_version"] != _SILERO_VERSION:
        raise ValueError("VAD config must pin silero-vad 6.2.1")
    SileroVadBackend(
        threshold=vad["threshold"],
        neg_threshold=vad["neg_threshold"],
        min_speech_duration_ms=vad["min_speech_duration_ms"],
        min_silence_duration_ms=vad["min_silence_duration_ms"],
        max_speech_duration_s=vad["max_speech_duration_s"],
        speech_pad_ms=vad["speech_pad_ms"],
        min_silence_at_max_speech=vad["min_silence_at_max_speech"],
        use_max_poss_sil_at_max_speech=vad["use_max_poss_sil_at_max_speech"],
        sample_rate=vad["sample_rate"],
        window_samples=vad["window_samples"],
    )
    if type(pause) is not dict or set(pause) != _PAUSE_FIELDS:
        raise ValueError("pause config must contain exact fields")
    if pause["profile_id"] != "pause-profile-v1":
        raise ValueError("pause config must pin pause-profile-v1")
    # Exercise the same parameter validator without requiring eligible gaps.
    with suppress(CorpusFailure):
        classify_pauses(
            (),
            sample_rate=16000,
            minimum_gap_ms=pause["minimum_gap_ms"],
            minimum_gap_count=pause["minimum_gap_count"],
            separation_ratio=pause["separation_ratio"],
            maximum_iterations=pause["maximum_iterations"],
            minimum_cluster_size=pause["minimum_cluster_size"],
            maximum_cluster_imbalance_ratio=pause["maximum_cluster_imbalance_ratio"],
        )
    return vad, pause


def _load_segment_inputs(
    paths: CorpusPaths,
) -> tuple[tuple[RecordingRecord, ...], dict[str, str]]:
    selection = _load_selection(paths.manifests / "pilot-selection.json")
    recordings = _load_recordings(paths)
    by_id = {record.recording_id: record for record in recordings}
    pilot: list[RecordingRecord] = []
    for recording_id, expected_hash in zip(
        selection.recording_ids, selection.inventory_hashes, strict=True
    ):
        record = by_id.get(recording_id)
        if record is None or record.sha256 != expected_hash:
            raise ValueError("pilot selection does not match recordings.jsonl")
        if record.content_type != "spoken":
            raise ValueError("segment accepts only spoken pilot recordings")
        if record.state not in {CorpusState.TRANSCRIPT_CONFIRMED, CorpusState.SEGMENTED}:
            raise ValueError("pilot recording must be TRANSCRIPT_CONFIRMED or SEGMENTED")
        if (
            not recording_id
            or recording_id in {".", ".."}
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in recording_id
            )
        ):
            raise ValueError("recording_id is unsafe for the segmentation run path")
        pilot.append(record)
    transcripts_path = paths.manifests / "transcripts.jsonl"
    if not transcripts_path.exists():
        raise ValueError("missing required manifest: transcripts.jsonl")
    transcript_rows = read_jsonl(transcripts_path)
    transcript_by_id: dict[str, str] = {}
    for row in transcript_rows:
        transcript_recording_id = row.get("recording_id")
        spoken_text = row.get("spoken_text")
        if (
            not isinstance(transcript_recording_id, str)
            or not isinstance(spoken_text, str)
            or not spoken_text
            or row.get("state") != "TRANSCRIPT_CONFIRMED"
        ):
            raise ValueError("transcripts.jsonl contains an invalid confirmed transcript")
        if transcript_recording_id in transcript_by_id:
            raise ValueError("transcripts.jsonl contains duplicate recording_id")
        transcript_by_id[transcript_recording_id] = spoken_text
    if set(transcript_by_id) != set(selection.recording_ids):
        raise ValueError("transcripts must exactly match pilot selection recording IDs")
    return recordings, transcript_by_id


def _validate_vad_result(
    result: VadResult,
    *,
    sample_count: int,
    window_samples: int,
    model_sha256: str,
) -> None:
    if type(result) is not VadResult:
        raise TypeError("VAD backend must return VadResult")
    if type(result.frames) is not tuple or type(result.speech_intervals) is not tuple:
        raise TypeError("VAD frames and speech intervals must be tuples")
    if (
        result.backend != "silero-vad"
        or result.model_version != _SILERO_VERSION
        or result.model_sha256 != model_sha256
        or result.sample_rate != 16000
    ):
        raise ValueError("VAD result backend, model identity, or sample rate does not match")
    if any(type(frame) is not VadFrame for frame in result.frames):
        raise TypeError("VAD frames must contain VadFrame values")
    expected_bounds = tuple(
        (start, min(start + window_samples, sample_count))
        for start in range(0, sample_count, window_samples)
    )
    if tuple((frame.start_sample, frame.end_sample) for frame in result.frames) != expected_bounds:
        raise ValueError("VAD frames do not cover consecutive analysis sample windows")
    previous_end = 0
    for interval in result.speech_intervals:
        if type(interval) is not SpeechInterval:
            raise TypeError("VAD speech intervals must contain SpeechInterval values")
        if interval.end_sample > sample_count:
            raise ValueError("VAD speech interval exceeds analysis sample bounds")
        if interval.start_sample < previous_end:
            raise ValueError("VAD speech intervals must be ordered and non-overlapping")
        previous_end = interval.end_sample


def _pause_analysis(result: VadResult, pause: dict[str, Any]) -> PauseAnalysis:
    return classify_pauses(
        result.speech_intervals,
        sample_rate=result.sample_rate,
        minimum_gap_ms=pause["minimum_gap_ms"],
        minimum_gap_count=pause["minimum_gap_count"],
        separation_ratio=pause["separation_ratio"],
        maximum_iterations=pause["maximum_iterations"],
        minimum_cluster_size=pause["minimum_cluster_size"],
        maximum_cluster_imbalance_ratio=pause["maximum_cluster_imbalance_ratio"],
    )


def _segmentation_row(
    *,
    record: RecordingRecord,
    config: CorpusConfig,
    analysis: DerivedAudio,
    transcript_sha256: str,
    vad_result: VadResult,
    vad_parameters: dict[str, Any],
    pause_parameters: dict[str, Any],
    pause_analysis: PauseAnalysis | None,
) -> dict[str, Any]:
    failure = pause_analysis is None
    input_sha256s = [
        record.sha256,
        analysis.sha256,
        transcript_sha256,
        vad_result.model_sha256,
    ]
    return {
        "schema_version": "1",
        "recording_id": record.recording_id,
        "status": "failure" if failure else "success",
        "issue_code": "PAUSE_CLASSES_AMBIGUOUS" if failure else None,
        "config_sha256": config.digest,
        "cache_key": _segmentation_cache_key(record.recording_id, config.digest, input_sha256s),
        "input_sha256s": input_sha256s,
        "analysis_audio": analysis.to_dict(),
        "vad": {
            "backend": vad_result.backend,
            "model_version": vad_result.model_version,
            "model_sha256": vad_result.model_sha256,
            "sample_rate": vad_result.sample_rate,
            "parameters": vad_parameters,
            "parameters_sha256": _canonical_digest(vad_parameters),
            "frames": [asdict(frame) for frame in vad_result.frames],
            "speech_intervals": [asdict(interval) for interval in vad_result.speech_intervals],
        },
        "pause": {
            "profile_id": pause_parameters["profile_id"],
            "parameters": pause_parameters,
            "parameters_sha256": _canonical_digest(pause_parameters),
            "intervals": []
            if pause_analysis is None
            else [asdict(interval) for interval in pause_analysis.pauses],
            "threshold_seconds": None
            if pause_analysis is None
            else pause_analysis.threshold_seconds,
            "short_median_seconds": None
            if pause_analysis is None
            else pause_analysis.short_median_seconds,
            "long_median_seconds": None
            if pause_analysis is None
            else pause_analysis.long_median_seconds,
        },
    }


def _decode_cached_vad(raw: object) -> VadResult:
    if type(raw) is not dict or set(raw) != {
        "backend",
        "model_version",
        "model_sha256",
        "sample_rate",
        "parameters",
        "parameters_sha256",
        "frames",
        "speech_intervals",
    }:
        raise ValueError("cached VAD result must contain exact fields")
    if type(raw["frames"]) is not list or type(raw["speech_intervals"]) is not list:
        raise TypeError("cached VAD arrays are invalid")
    return VadResult(
        raw["backend"],
        raw["model_version"],
        raw["model_sha256"],
        raw["sample_rate"],
        tuple(VadFrame(**frame) for frame in raw["frames"]),
        tuple(SpeechInterval(**interval) for interval in raw["speech_intervals"]),
    )


def _validate_result_row(
    raw: dict[str, Any],
    *,
    record: RecordingRecord,
    config: CorpusConfig,
    analysis: DerivedAudio,
    transcript_sha256: str,
    model_sha256: str,
    vad_parameters: dict[str, Any],
    pause_parameters: dict[str, Any],
) -> bool:
    if set(raw) != _SEGMENT_FIELDS:
        raise ValueError("segmentation result must contain exact fields")
    expected_input_sha256s = [record.sha256, analysis.sha256, transcript_sha256, model_sha256]
    if (
        raw["schema_version"] != "1"
        or raw["recording_id"] != record.recording_id
        or raw["config_sha256"] != config.digest
        or raw["cache_key"]
        != _segmentation_cache_key(record.recording_id, config.digest, expected_input_sha256s)
        or raw["input_sha256s"] != expected_input_sha256s
        or raw["analysis_audio"] != analysis.to_dict()
    ):
        raise ValueError("segmentation result identity or provenance does not match")
    vad_raw = raw["vad"]
    vad_result = _decode_cached_vad(vad_raw)
    if vad_raw["parameters"] != vad_parameters or vad_raw["parameters_sha256"] != _canonical_digest(
        vad_parameters
    ):
        raise ValueError("cached VAD parameters do not match")
    assert analysis.metrics is not None
    _validate_vad_result(
        vad_result,
        sample_count=analysis.metrics.sample_count,
        window_samples=vad_parameters["window_samples"],
        model_sha256=model_sha256,
    )
    pause_raw = raw["pause"]
    if type(pause_raw) is not dict or set(pause_raw) != {
        "profile_id",
        "parameters",
        "parameters_sha256",
        "intervals",
        "threshold_seconds",
        "short_median_seconds",
        "long_median_seconds",
    }:
        raise ValueError("cached pause result must contain exact fields")
    if (
        pause_raw["profile_id"] != "pause-profile-v1"
        or pause_raw["parameters"] != pause_parameters
        or pause_raw["parameters_sha256"] != _canonical_digest(pause_parameters)
    ):
        raise ValueError("cached pause parameters do not match")
    try:
        expected_pause = _pause_analysis(vad_result, pause_parameters)
    except CorpusFailure as error:
        if (
            error.code != "PAUSE_CLASSES_AMBIGUOUS"
            or raw["status"] != "failure"
            or raw["issue_code"] != error.code
            or pause_raw["intervals"] != []
            or any(
                pause_raw[field] is not None
                for field in (
                    "threshold_seconds",
                    "short_median_seconds",
                    "long_median_seconds",
                )
            )
        ):
            raise ValueError("cached pause failure does not match VAD evidence") from error
        return False
    expected = _segmentation_row(
        record=record,
        config=config,
        analysis=analysis,
        transcript_sha256=transcript_sha256,
        vad_result=vad_result,
        vad_parameters=vad_parameters,
        pause_parameters=pause_parameters,
        pause_analysis=expected_pause,
    )
    if raw != expected:
        raise ValueError("cached successful segmentation does not match VAD evidence")
    return True


def _transition_segmented(
    paths: CorpusPaths,
    config: CorpusConfig,
    recordings: tuple[RecordingRecord, ...],
    index: int,
    input_sha256s: tuple[str, ...],
) -> tuple[RecordingRecord, ...]:
    record = recordings[index]
    timestamp = datetime.now(timezone.utc).isoformat()
    segmented, event = advance_recording(
        record,
        CorpusState.SEGMENTED,
        input_sha256s=input_sha256s,
        config_sha256=config.digest,
        tool_versions=("silero-vad==6.2.1", "pause-profile-v1"),
        started_at=timestamp,
        finished_at=timestamp,
        result="success",
    )
    updated = (*recordings[:index], segmented, *recordings[index + 1 :])
    persist_recording_transition(
        recordings_path=paths.manifests / "recordings.jsonl",
        events_path=paths.alignments / "runs" / config.digest / "processing-events.jsonl",
        recordings=updated,
        event=event,
    )
    return updated


def segment_corpus(
    paths: CorpusPaths,
    config: CorpusConfig,
    backend: VadBackend,
    *,
    ffmpeg_version: str,
    run_command: RunCommand = subprocess.run,
) -> bool:
    vad_parameters, pause_parameters = _validate_segmentation_config(config)
    recordings, transcripts = _load_segment_inputs(paths)
    model_sha256 = backend.model_sha256
    if (
        not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in model_sha256)
    ):
        raise ValueError("VAD backend model_sha256 must be a lowercase SHA-256 digest")
    current = recordings
    all_successful = True
    for recording_id in _load_selection(paths.manifests / "pilot-selection.json").recording_ids:
        index = next(
            item for item, record in enumerate(current) if record.recording_id == recording_id
        )
        record = current[index]
        transcript_sha256 = hashlib.sha256(
            transcripts[record.recording_id].encode("utf-8")
        ).hexdigest()
        result_path = (
            paths.alignments / "runs" / config.digest / record.recording_id / "segmentation.json"
        )
        if result_path.exists():
            try:
                rows = read_jsonl(result_path)
                if len(rows) != 1:
                    raise ValueError("segmentation cache must contain exactly one result")
                cached_analysis_raw = rows[0].get("analysis_audio")
                if type(cached_analysis_raw) is not dict:
                    raise ValueError("segmentation cache lacks analysis provenance")
                cached_analysis = DerivedAudio.from_dict(cached_analysis_raw)
                analysis = derive_analysis_audio(
                    record,
                    paths,
                    config,
                    ffmpeg_version=ffmpeg_version,
                    cached=cached_analysis,
                    run_command=run_command,
                )
                successful = _validate_result_row(
                    rows[0],
                    record=record,
                    config=config,
                    analysis=analysis,
                    transcript_sha256=transcript_sha256,
                    model_sha256=model_sha256,
                    vad_parameters=vad_parameters,
                    pause_parameters=pause_parameters,
                )
            except (KeyError, TypeError, ValueError, CorpusFailure) as error:
                if isinstance(error, CorpusFailure) and error.code == "ALIGNER_UNAVAILABLE":
                    raise
                raise CorpusFailure(
                    "CACHE_ARTIFACT_INVALID", "cached segmentation result is invalid"
                ) from error
            if successful:
                if record.state is CorpusState.TRANSCRIPT_CONFIRMED:
                    current = _transition_segmented(
                        paths,
                        config,
                        current,
                        index,
                        (record.sha256, analysis.sha256, transcript_sha256, model_sha256),
                    )
                elif record.state is CorpusState.SEGMENTED:
                    timestamp = "1970-01-01T00:00:00+00:00"
                    prior = replace(record, state=CorpusState.TRANSCRIPT_CONFIRMED)
                    _, expected_event = advance_recording(
                        prior,
                        CorpusState.SEGMENTED,
                        input_sha256s=(
                            record.sha256,
                            analysis.sha256,
                            transcript_sha256,
                            model_sha256,
                        ),
                        config_sha256=config.digest,
                        tool_versions=("silero-vad==6.2.1", "pause-profile-v1"),
                        started_at=timestamp,
                        finished_at=timestamp,
                        result="success",
                    )
                    events_path = (
                        paths.alignments / "runs" / config.digest / "processing-events.jsonl"
                    )
                    events = read_jsonl(events_path) if events_path.exists() else ()
                    if not processing_event_exists(events, expected_event):
                        raise CorpusFailure(
                            "CACHE_ARTIFACT_INVALID",
                            "SEGMENTED recording lacks its durable transition event",
                        )
            else:
                if record.state is not CorpusState.TRANSCRIPT_CONFIRMED:
                    raise CorpusFailure(
                        "CACHE_ARTIFACT_INVALID",
                        "failed segmentation cannot accompany SEGMENTED state",
                    )
                all_successful = False
            continue
        if record.state is CorpusState.SEGMENTED:
            raise CorpusFailure(
                "CACHE_ARTIFACT_INVALID", "SEGMENTED recording lacks segmentation result"
            )
        analysis = derive_analysis_audio(
            record,
            paths,
            config,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
        )
        assert analysis.metrics is not None
        vad_result = backend.analyze(paths.local_data / Path(*analysis.relative_path.split("/")))
        _validate_vad_result(
            vad_result,
            sample_count=analysis.metrics.sample_count,
            window_samples=vad_parameters["window_samples"],
            model_sha256=model_sha256,
        )
        try:
            pauses = _pause_analysis(vad_result, pause_parameters)
        except CorpusFailure as error:
            if error.code != "PAUSE_CLASSES_AMBIGUOUS":
                raise
            pauses = None
        row = _segmentation_row(
            record=record,
            config=config,
            analysis=analysis,
            transcript_sha256=transcript_sha256,
            vad_result=vad_result,
            vad_parameters=vad_parameters,
            pause_parameters=pause_parameters,
            pause_analysis=pauses,
        )
        _validate_result_row(
            row,
            record=record,
            config=config,
            analysis=analysis,
            transcript_sha256=transcript_sha256,
            model_sha256=model_sha256,
            vad_parameters=vad_parameters,
            pause_parameters=pause_parameters,
        )
        write_jsonl_atomic(result_path, (row,))
        if pauses is None:
            all_successful = False
            continue
        current = _transition_segmented(
            paths,
            config,
            current,
            index,
            (record.sha256, analysis.sha256, transcript_sha256, model_sha256),
        )
    return all_successful


def _validate_pairing_config(config: CorpusConfig) -> PairingParameters:
    _validate_segmentation_config(config)
    raw = config.raw["pairing"]
    if type(raw) is not dict or set(raw) != _PAIRING_FIELDS:
        raise ValueError("pairing config must contain exact fields")
    return PairingParameters(
        raw["minimum_duration_ratio"],
        raw["maximum_duration_ratio"],
        raw["minimum_score_margin"],
    )


def _decode_spoken_units(raw: dict[str, Any], recording_id: str) -> tuple[SpokenUnit, ...]:
    require_exact_fields(
        raw,
        frozenset(field.name for field in fields(TranscriptRecord)),
        "transcript row",
    )
    if (
        raw["schema_version"] != "1"
        or raw["recording_id"] != recording_id
        or raw["state"] != "TRANSCRIPT_CONFIRMED"
        or type(raw["spoken_text"]) is not str
        or not raw["spoken_text"]
    ):
        raise ValueError("transcript identity or state is invalid for pairing")
    units_raw = raw["spoken_units"]
    if type(units_raw) is not list or not units_raw:
        raise TypeError("transcript spoken_units must be a non-empty array")
    expected = frozenset(field.name for field in fields(SpokenUnit))
    units: list[SpokenUnit] = []
    unit_ids: set[str] = set()
    for unit_raw in units_raw:
        if type(unit_raw) is not dict:
            raise TypeError("spoken unit must be an object")
        require_exact_fields(unit_raw, expected, "spoken unit")
        unit = SpokenUnit(**unit_raw)
        require_safe_pairing_id(unit.unit_id, "unit_id")
        if unit.unit_id in unit_ids:
            raise ValueError("spoken unit_id values must be unique")
        unit_ids.add(unit.unit_id)
        units.append(unit)
    if "\n".join(unit.text for unit in units) != raw["spoken_text"]:
        raise ValueError("spoken units do not exactly reconstruct spoken_text")
    plan = raw["pronunciation_plan"]
    if type(plan) is not dict:
        raise TypeError("transcript pronunciation plan is invalid")
    require_exact_fields(
        plan,
        frozenset(
            {
                "schema_version",
                "rule_version",
                "original_text",
                "normalized_text",
                "tokens",
                "phrase_phonemes",
                "warning_codes",
            }
        ),
        "pronunciation plan",
    )
    tokens_raw = plan["tokens"]
    if type(tokens_raw) is not list:
        raise TypeError("transcript pronunciation plan tokens must be an array")
    if (
        plan["schema_version"] != "1"
        or plan["rule_version"] != "ecclesiastical-roman-v1"
        or plan["original_text"] != raw["spoken_text"]
        or plan["normalized_text"] != raw["normalized_text"]
        or plan["normalized_text"] != normalize_phrase(raw["spoken_text"])
    ):
        raise ValueError("pronunciation plan identity or text does not match transcript")
    for field in ("phrase_phonemes", "warning_codes"):
        if type(plan[field]) is not list or any(type(item) is not str for item in plan[field]):
            raise TypeError(f"pronunciation plan {field} must be a string array")
    token_fields = frozenset(
        {
            "surface",
            "normalized",
            "source_span",
            "syllables",
            "stress_index",
            "ipa",
            "model_phonemes",
            "resolution_method",
            "source_ids",
            "warning_codes",
        }
    )
    spans = tokenize_words(raw["spoken_text"])
    if len(tokens_raw) != len(spans):
        raise ValueError("pronunciation plan token count does not match spoken_text")
    for token_raw, span in zip(tokens_raw, spans, strict=True):
        if type(token_raw) is not dict:
            raise TypeError("pronunciation plan token must be an object")
        require_exact_fields(token_raw, token_fields, "pronunciation plan token")
        if (
            token_raw["surface"] != span.surface
            or token_raw["normalized"] != normalize_word(span.surface).normalized
            or token_raw["source_span"] != [span.start, span.end]
        ):
            raise ValueError("pronunciation plan token content does not match spoken_text")
        for field in ("syllables", "model_phonemes", "source_ids", "warning_codes"):
            if type(token_raw[field]) is not list or any(
                type(item) is not str for item in token_raw[field]
            ):
                raise TypeError(f"pronunciation token {field} must be a string array")
        if (
            not token_raw["syllables"]
            or type(token_raw["stress_index"]) is not int
            or not 0 <= token_raw["stress_index"] < len(token_raw["syllables"])
            or type(token_raw["ipa"]) is not str
            or type(token_raw["resolution_method"]) is not str
        ):
            raise ValueError("pronunciation token linguistic content is invalid")
    expected_phrase_phonemes: list[str] = []
    expected_warning_codes: list[str] = []
    for token_raw in tokens_raw:
        if expected_phrase_phonemes:
            expected_phrase_phonemes.append("|")
        expected_phrase_phonemes.extend(token_raw["model_phonemes"])
        expected_warning_codes.extend(token_raw["warning_codes"])
    if (
        plan["phrase_phonemes"] != expected_phrase_phonemes
        or plan["warning_codes"] != expected_warning_codes
    ):
        raise ValueError("pronunciation plan aggregate content does not match tokens")
    token_cursor = 0
    for ordinal, unit in enumerate(units, 1):
        unit_surfaces = tuple(span.surface for span in tokenize_words(unit.text))
        planned_surfaces = tuple(
            token["surface"] for token in tokens_raw[unit.token_start_index : unit.token_end_index]
        )
        if (
            unit.ordinal != ordinal
            or unit.token_start_index != token_cursor
            or unit.token_end_index <= unit.token_start_index
            or planned_surfaces != unit_surfaces
        ):
            raise ValueError("spoken unit token ranges do not match unit text")
        token_cursor = unit.token_end_index
    if token_cursor != len(tokens_raw):
        raise ValueError("spoken unit token ranges do not cover pronunciation plan")
    return tuple(units)


def _load_pairing_segmentation(
    paths: CorpusPaths,
    config: CorpusConfig,
    record: RecordingRecord,
    transcript_sha256: str,
    *,
    ffmpeg_version: str,
    run_command: RunCommand,
) -> tuple[DerivedAudio, tuple[SpeechInterval, ...], PauseAnalysis, str]:
    result_path = pairing_run_directory(paths, config.digest, record.recording_id) / (
        "segmentation.json"
    )
    rows = read_jsonl(result_path)
    if len(rows) != 1:
        raise ValueError("segmentation cache must contain exactly one result")
    row = rows[0]
    if type(row.get("analysis_audio")) is not dict:
        raise TypeError("segmentation analysis_audio must be an object")
    analysis = DerivedAudio.from_dict(row["analysis_audio"])
    cached_analysis = derive_analysis_audio(
        record,
        paths,
        config,
        ffmpeg_version=ffmpeg_version,
        cached=analysis,
        run_command=run_command,
    )
    vad_parameters, pause_parameters = _validate_segmentation_config(config)
    vad_raw = row.get("vad")
    if type(vad_raw) is not dict or type(vad_raw.get("model_sha256")) is not str:
        raise TypeError("segmentation VAD provenance is invalid")
    if not _validate_result_row(
        row,
        record=record,
        config=config,
        analysis=cached_analysis,
        transcript_sha256=transcript_sha256,
        model_sha256=vad_raw["model_sha256"],
        vad_parameters=vad_parameters,
        pause_parameters=pause_parameters,
    ):
        raise ValueError("pairing requires successful pause classification")
    vad_result = _decode_cached_vad(vad_raw)
    pause_raw = row["pause"]
    intervals_raw = pause_raw["intervals"]
    if type(intervals_raw) is not list:
        raise TypeError("segmentation pause intervals must be an array")
    pauses = tuple(PauseInterval(**item) for item in intervals_raw)
    analysis_result = PauseAnalysis(
        pauses,
        pause_raw["threshold_seconds"],
        pause_raw["short_median_seconds"],
        pause_raw["long_median_seconds"],
    )
    return (
        cached_analysis,
        vad_result.speech_intervals,
        analysis_result,
        hashlib.sha256(result_path.read_bytes()).hexdigest(),
    )


def _expected_cached_transition(
    record: RecordingRecord,
    previous_state: CorpusState,
    target_state: CorpusState,
    *,
    input_sha256s: tuple[str, ...],
    config_sha256: str,
    tool_versions: tuple[str, ...],
) -> ProcessingEvent:
    prior = replace(record, state=previous_state)
    _, event = advance_recording(
        prior,
        target_state,
        input_sha256s=input_sha256s,
        config_sha256=config_sha256,
        tool_versions=tool_versions,
        started_at="1970-01-01T00:00:00+00:00",
        finished_at="1970-01-01T00:00:00+00:00",
        result="success",
    )
    return event


def _persist_stage_transition(
    paths: CorpusPaths,
    config: CorpusConfig,
    recordings: tuple[RecordingRecord, ...],
    index: int,
    target_state: CorpusState,
    *,
    input_sha256s: tuple[str, ...],
    tool_versions: tuple[str, ...],
) -> tuple[RecordingRecord, ...]:
    timestamp = datetime.now(timezone.utc).isoformat()
    advanced, event = advance_recording(
        recordings[index],
        target_state,
        input_sha256s=input_sha256s,
        config_sha256=config.digest,
        tool_versions=tool_versions,
        started_at=timestamp,
        finished_at=timestamp,
        result="success",
    )
    updated = (*recordings[:index], advanced, *recordings[index + 1 :])
    persist_recording_transition(
        recordings_path=paths.manifests / "recordings.jsonl",
        events_path=(
            pairing_run_directory(paths, config.digest, recordings[index].recording_id).parent
            / "processing-events.jsonl"
        ),
        recordings=updated,
        event=event,
    )
    return updated


def _pairing_tool_versions(pairing: PairingRecording, stage: str) -> tuple[str, ...]:
    return (
        stage,
        pairing.ffmpeg_version,
        *(f"alignment-runtime-sha256={digest}" for digest in pairing.alignment_runtime_sha256s),
    )


def pair_corpus(
    paths: CorpusPaths,
    config: CorpusConfig,
    backend: PairingBackend,
    *,
    ffmpeg_version: str,
    run_command: RunCommand = subprocess.run,
) -> bool:
    parameters = _validate_pairing_config(config)
    selection = _load_selection(paths.manifests / "pilot-selection.json")
    recordings = _load_recordings(paths)
    by_id = {record.recording_id: (index, record) for index, record in enumerate(recordings)}
    transcript_rows = read_jsonl(paths.manifests / "transcripts.jsonl")
    transcript_by_id = {row.get("recording_id"): row for row in transcript_rows}
    if len(transcript_by_id) != len(transcript_rows) or set(transcript_by_id) != set(
        selection.recording_ids
    ):
        raise ValueError("transcripts must exactly match pilot selection")
    current = recordings
    all_successful = True
    for recording_id, inventory_hash in zip(
        selection.recording_ids, selection.inventory_hashes, strict=True
    ):
        item = by_id.get(recording_id)
        if item is None:
            raise ValueError("pilot selection recording is missing")
        index, original = item
        record = current[index]
        if record.sha256 != inventory_hash:
            raise ValueError("pilot selection hash does not match recording")
        if record.content_type != "spoken":
            raise ValueError("pair accepts only spoken pilot recordings")
        if record.state not in {CorpusState.SEGMENTED, CorpusState.PAIRED}:
            raise ValueError("pair requires SEGMENTED or PAIRED recording state")
        run_directory = pairing_run_directory(paths, config.digest, recording_id)
        transcript_raw = transcript_by_id[recording_id]
        assert isinstance(transcript_raw, dict)
        units = _decode_spoken_units(transcript_raw, recording_id)
        transcript_sha256 = hashlib.sha256(transcript_raw["spoken_text"].encode()).hexdigest()
        analysis, speech, pause_analysis, segmentation_sha256 = _load_pairing_segmentation(
            paths,
            config,
            original,
            transcript_sha256,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
        )
        windows = map_text_units(
            units,
            speech,
            pause_analysis,
            preserve_missing_take_candidates=True,
        )
        pairing = pair_recording(
            recording_id,
            windows,
            analysis,
            paths,
            backend,
            segmentation_artifact_sha256=segmentation_sha256,
            config_sha256=config.digest,
            pairing_parameters=parameters,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
        )
        pairing_path = run_directory / "pairing.json"
        pairing_sha256 = hashlib.sha256(pairing_path.read_bytes()).hexdigest()
        inputs = (record.sha256, segmentation_sha256, pairing_sha256)
        tools = _pairing_tool_versions(pairing, "two-take-pairing-v1")
        fully_selected = all(
            outcome.status == "selected" and outcome.group is not None for outcome in pairing.groups
        )
        if not fully_selected:
            if record.state is CorpusState.PAIRED:
                raise CorpusFailure(
                    "CACHE_ARTIFACT_INVALID", "PAIRED recording contains review outcomes"
                )
            all_successful = False
            continue
        if record.state is CorpusState.SEGMENTED:
            current = _persist_stage_transition(
                paths,
                config,
                current,
                index,
                CorpusState.PAIRED,
                input_sha256s=inputs,
                tool_versions=tools,
            )
        else:
            event = _expected_cached_transition(
                record,
                CorpusState.SEGMENTED,
                CorpusState.PAIRED,
                input_sha256s=inputs,
                config_sha256=config.digest,
                tool_versions=tools,
            )
            events = read_jsonl(run_directory.parent / "processing-events.jsonl")
            if not processing_event_exists(events, event):
                raise CorpusFailure(
                    "CACHE_ARTIFACT_INVALID", "PAIRED recording lacks durable transition event"
                )
        if len(pairing.groups) != len(windows):
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "pairing does not cover every unit")
    return all_successful


def _alignment_row(
    pairing_path: Path,
    pairing: Any,
    paths: CorpusPaths,
    backend: PairingBackend,
) -> dict[str, Any]:
    if any(outcome.status != "selected" or outcome.group is None for outcome in pairing.groups):
        raise ValueError("alignment requires selected outcomes for every spoken unit")
    selected = load_selected_alignments(pairing, paths, backend)
    issues: list[dict[str, Any]] = []
    pairing_sha256 = hashlib.sha256(pairing_path.read_bytes()).hexdigest()
    takes = [
        {
            "repetition_group_id": item.repetition_group_id,
            "unit_id": item.unit_id,
            "take_index": item.take.take_index,
            "audio_sha256": item.take.audio_sha256,
            "alignment_cache_key": item.take.alignment_cache_key,
            "alignment_result": result_to_dict(item.result),
        }
        for item in selected
    ]
    take_identity = [
        (item.take.alignment_cache_key, item.result.integrity_sha256) for item in selected
    ]
    return {
        "schema_version": "1",
        "recording_id": pairing.recording_id,
        "status": "success",
        "config_sha256": pairing.config_sha256,
        "pairing_artifact_sha256": pairing_sha256,
        "pairing_cache_key": pairing.cache_key,
        "takes": takes,
        "issues": issues,
        "cache_key": _canonical_digest(
            {
                "schema_version": "1",
                "recording_id": pairing.recording_id,
                "config_sha256": pairing.config_sha256,
                "pairing_artifact_sha256": pairing_sha256,
                "takes": take_identity,
                "issues": issues,
            }
        ),
    }


def align_corpus(paths: CorpusPaths, config: CorpusConfig, backend: PairingBackend) -> bool:
    _validate_pairing_config(config)
    selection = _load_selection(paths.manifests / "pilot-selection.json")
    recordings = _load_recordings(paths)
    by_id = {record.recording_id: (index, record) for index, record in enumerate(recordings)}
    current = recordings
    for recording_id in selection.recording_ids:
        item = by_id.get(recording_id)
        if item is None:
            raise ValueError("pilot selection recording is missing")
        index, _ = item
        record = current[index]
        if record.content_type != "spoken":
            raise ValueError("align accepts only spoken pilot recordings")
        if record.state not in {CorpusState.PAIRED, CorpusState.ALIGNED}:
            raise ValueError("align requires PAIRED or ALIGNED recording state")
        run_directory = pairing_run_directory(paths, config.digest, recording_id)
        pairing_path = run_directory / "pairing.json"
        try:
            pairing_rows = read_jsonl(pairing_path)
            if len(pairing_rows) != 1:
                raise ValueError("pairing cache must contain exactly one recording")
            pairing = pairing_from_dict(pairing_rows[0])
            if pairing.recording_id != recording_id or pairing.config_sha256 != config.digest:
                raise ValueError("pairing identity does not match align request")
            pairing_sha256 = hashlib.sha256(pairing_path.read_bytes()).hexdigest()
            pair_tools = _pairing_tool_versions(pairing, "two-take-pairing-v1")
            paired_event = _expected_cached_transition(
                record,
                CorpusState.SEGMENTED,
                CorpusState.PAIRED,
                input_sha256s=(
                    record.sha256,
                    pairing.segmentation_artifact_sha256,
                    pairing_sha256,
                ),
                config_sha256=config.digest,
                tool_versions=pair_tools,
            )
            events_path = run_directory.parent / "processing-events.jsonl"
            events = read_jsonl(events_path)
            if not processing_event_exists(events, paired_event):
                raise ValueError("pairing artifact is not bound to its PAIRED event")
            expected = _alignment_row(pairing_path, pairing, paths, backend)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise CorpusFailure(
                "CACHE_ARTIFACT_INVALID", "selected alignment cache is invalid"
            ) from error
        alignment_path = run_directory / "alignment.json"
        if alignment_path.exists():
            rows = read_jsonl(alignment_path)
            if len(rows) != 1 or rows[0] != expected:
                raise CorpusFailure("CACHE_ARTIFACT_INVALID", "alignment artifact is invalid")
        else:
            write_jsonl_atomic(alignment_path, (expected,))
        alignment_sha256 = hashlib.sha256(alignment_path.read_bytes()).hexdigest()
        inputs = (record.sha256, pairing_sha256, alignment_sha256)
        tools = _pairing_tool_versions(pairing, "cached-word-alignment-v1")
        if record.state is CorpusState.PAIRED:
            current = _persist_stage_transition(
                paths,
                config,
                current,
                index,
                CorpusState.ALIGNED,
                input_sha256s=inputs,
                tool_versions=tools,
            )
        else:
            event = _expected_cached_transition(
                record,
                CorpusState.PAIRED,
                CorpusState.ALIGNED,
                input_sha256s=inputs,
                config_sha256=config.digest,
                tool_versions=tools,
            )
            events = read_jsonl(run_directory.parent / "processing-events.jsonl")
            if not processing_event_exists(events, event):
                raise CorpusFailure(
                    "CACHE_ARTIFACT_INVALID", "ALIGNED recording lacks durable transition event"
                )
    return True


def _ffmpeg_version() -> str:
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            encoding="utf-8",
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "ffmpeg version is unavailable") from error
    first_line = completed.stdout.splitlines()[0] if completed.stdout.splitlines() else ""
    if not first_line.strip():
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "ffmpeg version output is empty")
    return first_line.strip()


class _EnvironmentCommandFailure(Exception):
    def __init__(self, artifact_field: str, failure_type: str) -> None:
        super().__init__(failure_type)
        self.artifact_field = artifact_field
        self.evidence = f"unavailable: {failure_type}"


def _capture_environment_command(
    command: list[str],
    run_command: RunCommand,
    *,
    artifact_field: str,
    cwd: Path | None = None,
) -> str:
    try:
        options: dict[str, object] = {
            "capture_output": True,
            "check": True,
            "encoding": "utf-8",
            "text": True,
            "timeout": 30,
        }
        if cwd is not None:
            options["cwd"] = cwd
        completed = run_command(command, **options)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeError,
    ) as error:
        raise _EnvironmentCommandFailure(artifact_field, type(error).__name__) from error
    return completed.stdout.strip()


def _sanitize_pip_freeze(output: str) -> str:
    sanitized: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        package, separator, _ = line.partition(" @ ")
        if (
            separator
            and package
            and all(character.isalnum() or character in "-_." for character in package)
        ):
            sanitized.append(f"{package} @ <redacted>")
        elif "://" in line or "\\" in line or "/" in line or line.startswith("-e "):
            sanitized.append("<direct-reference-redacted>")
        else:
            sanitized.append(line)
    return "\n".join(sanitized)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def alignment_smoke_test(
    paths: CorpusPaths,
    config: CorpusConfig,
    *,
    cache_dir: Path | None = None,
    local_files_only: bool = True,
    module_loader: Callable[[str], Any] | None = None,
    tool_commit_resolver: Callable[[], str] | None = None,
    dependency_version_resolver: Callable[[str], str] | None = None,
    run_command: RunCommand = subprocess.run,
    now: Callable[[], datetime] | None = None,
) -> int:
    """Load the pinned backend and persist a reproducible local runtime audit."""
    timestamp = (now or (lambda: datetime.now(timezone.utc)))().isoformat()
    alignment = config.raw.get("alignment")
    model_revision = alignment.get("model_revision") if type(alignment) is dict else None
    model_license = alignment.get("license") if type(alignment) is dict else None
    run_id = hashlib.sha256(f"{timestamp}\0{config.digest}\0{model_revision}".encode()).hexdigest()[
        :24
    ]
    runtime_directory = paths.alignments / "runtime"
    runtime_directory.mkdir(parents=True, exist_ok=True)
    environment_values = {
        "pip_freeze": "not collected",
        "nvidia_smi": "not collected",
        "code_commit": "not collected",
    }
    success = False
    failure_code: str | None = None
    message = ""
    runtime: Any | None = None
    try:
        environment_values["pip_freeze"] = _sanitize_pip_freeze(
            _capture_environment_command(
                [sys.executable, "-m", "pip", "freeze"],
                run_command,
                artifact_field="pip_freeze",
            )
        )
        try:
            environment_values["nvidia_smi"] = _capture_environment_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                run_command,
                artifact_field="nvidia_smi",
            )
        except _EnvironmentCommandFailure as error:
            environment_values[error.artifact_field] = error.evidence
        environment_values["code_commit"] = _capture_environment_command(
            ["git", "rev-parse", "HEAD"],
            run_command,
            artifact_field="code_commit",
            cwd=paths.project_root,
        )
        backend = create_alignment_backend(
            config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            module_loader=module_loader,
            tool_commit_resolver=tool_commit_resolver,
            dependency_version_resolver=dependency_version_resolver,
        )
        runtime = backend.runtime_info()
        success = True
    except _EnvironmentCommandFailure as error:
        environment_values[error.artifact_field] = error.evidence
        failure_code = "ALIGNER_UNAVAILABLE"
        message = "environment evidence unavailable"
    except CorpusFailure as error:
        failure_code = error.code
        message = "pinned alignment backend unavailable"
    except (OSError, ValueError):
        failure_code = "MANIFEST_SCHEMA_MISMATCH"
        message = "invalid alignment smoke configuration"
    smoke = {
        "schema_version": "1",
        "success": success,
        "failure_code": failure_code,
        "message": message,
        "timestamp": timestamp,
        "python_version": sys.version,
        "pip_freeze": environment_values["pip_freeze"],
        "nvidia_smi": environment_values["nvidia_smi"],
        "code_commit": environment_values["code_commit"],
        "backend_version": runtime.aligner_version if runtime is not None else None,
        "aligner_commit": runtime.aligner_commit if runtime is not None else None,
        "torch_version": runtime.torch_version if runtime is not None else None,
        "transformers_version": runtime.transformers_version if runtime is not None else None,
        "uroman_version": runtime.uroman_version if runtime is not None else None,
        "cuda_available": runtime.cuda_available if runtime is not None else None,
        "requested_device": runtime.requested_device if runtime is not None else None,
        "requested_dtype": runtime.requested_dtype if runtime is not None else None,
        "resolved_device": runtime.resolved_device if runtime is not None else None,
        "resolved_dtype": runtime.resolved_dtype if runtime is not None else None,
        "device_name": runtime.device_name if runtime is not None else None,
        "peak_memory_bytes": runtime.peak_memory_bytes if runtime is not None else None,
        "measurement_scope": runtime.measurement_scope if runtime is not None else None,
        "model_revision": model_revision,
        "resolved_model_revision": (
            runtime.resolved_model_revision if runtime is not None else None
        ),
        "model_weights_sha256": runtime.model_weights_sha256 if runtime is not None else None,
        "model_weight_manifest": (
            [item.to_dict() for item in runtime.model_weight_manifest]
            if runtime is not None
            else None
        ),
        "model_license": model_license,
    }
    environment = "\n".join(
        (
            f"Run ID: {run_id}",
            f"Python version: {sys.version}",
            f"Timestamp: {timestamp}",
            f"Code commit: {environment_values['code_commit']}",
            f"Model revision: {model_revision}",
            f"Model license: {model_license}",
            f"Torch version: {smoke['torch_version']}",
            f"Transformers version: {smoke['transformers_version']}",
            f"CUDA available: {smoke['cuda_available']}",
            f"ctc-forced-aligner commit: {smoke['aligner_commit']}",
            "",
            "pip freeze:",
            environment_values["pip_freeze"],
            "",
            "nvidia-smi:",
            environment_values["nvidia_smi"],
            "",
        )
    )
    smoke["run_id"] = run_id
    smoke["environment_sha256"] = hashlib.sha256(environment.encode("utf-8")).hexdigest()
    _write_text_atomic(runtime_directory / "environment.txt", environment)
    write_jsonl_atomic(runtime_directory / "smoke.json", (smoke,))
    return 0 if success else 1


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
    segment_parser = subparsers.add_parser("segment")
    segment_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/corpus/pilot-v1.json"),
    )
    pair_parser = subparsers.add_parser("pair")
    pair_parser.add_argument("--allow-download", action="store_true")
    pair_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/corpus/pilot-v1.json"),
    )
    align_parser = subparsers.add_parser("align")
    align_parser.add_argument("--smoke-test", action="store_true")
    align_parser.add_argument("--allow-download", action="store_true")
    align_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/corpus/pilot-v1.json"),
    )
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
    if args.command == "align":
        paths = CorpusPaths.from_project_root(args.project_root)
        config_path = args.config
        if not config_path.is_absolute():
            config_path = args.project_root / config_path
        try:
            config = CorpusConfig.load(config_path)
            paths.ensure_layout()
            if args.smoke_test:
                return alignment_smoke_test(
                    paths,
                    config,
                    cache_dir=paths.local_data / "cache" / "huggingface",
                    local_files_only=not args.allow_download,
                )
            alignment_backend = create_alignment_backend(
                config,
                cache_dir=paths.local_data / "cache" / "huggingface",
                local_files_only=not args.allow_download,
            )
            return 0 if align_corpus(paths, config, alignment_backend) else 1
        except CorpusFailure as error:
            print(f"{error.code}: {error}", file=sys.stderr)
            return 1
        except (OSError, ValueError) as error:
            print(f"MANIFEST_SCHEMA_MISMATCH: {error}", file=sys.stderr)
            return 2
    if args.command == "pair":
        paths = CorpusPaths.from_project_root(args.project_root)
        config_path = args.config
        if not config_path.is_absolute():
            config_path = args.project_root / config_path
        try:
            config = CorpusConfig.load(config_path)
            paths.ensure_layout()
            pair_backend = create_alignment_backend(
                config,
                cache_dir=paths.local_data / "cache" / "huggingface",
                local_files_only=not args.allow_download,
            )
            successful = pair_corpus(
                paths,
                config,
                pair_backend,
                ffmpeg_version=_ffmpeg_version(),
            )
        except CorpusFailure as error:
            print(f"{error.code}: {error}", file=sys.stderr)
            return 1
        except (InventoryInputError, OSError, TypeError, ValueError) as error:
            print(f"MANIFEST_SCHEMA_MISMATCH: {error}", file=sys.stderr)
            return 2
        return 0 if successful else 1
    if args.command == "segment":
        paths = CorpusPaths.from_project_root(args.project_root)
        config_path = args.config
        if not config_path.is_absolute():
            config_path = args.project_root / config_path
        try:
            config = CorpusConfig.load(config_path)
            vad, _ = _validate_segmentation_config(config)
            paths.ensure_layout()
            vad_backend = SileroVadBackend(
                threshold=vad["threshold"],
                neg_threshold=vad["neg_threshold"],
                min_speech_duration_ms=vad["min_speech_duration_ms"],
                min_silence_duration_ms=vad["min_silence_duration_ms"],
                max_speech_duration_s=vad["max_speech_duration_s"],
                speech_pad_ms=vad["speech_pad_ms"],
                min_silence_at_max_speech=vad["min_silence_at_max_speech"],
                use_max_poss_sil_at_max_speech=vad["use_max_poss_sil_at_max_speech"],
                sample_rate=vad["sample_rate"],
                window_samples=vad["window_samples"],
            )
            successful = segment_corpus(
                paths,
                config,
                vad_backend,
                ffmpeg_version=_ffmpeg_version(),
            )
        except CorpusFailure as error:
            print(f"{error.code}: {error}", file=sys.stderr)
            return 1
        except (InventoryInputError, OSError, TypeError, ValueError) as error:
            print(f"MANIFEST_SCHEMA_MISMATCH: {error}", file=sys.stderr)
            return 2
        return 0 if successful else 1
    raise AssertionError(args.command)
