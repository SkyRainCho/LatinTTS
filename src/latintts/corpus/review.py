from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latintts.corpus.alignment import AlignmentResult, WordSpan, result_from_dict
from latintts.corpus.audio import DerivedAudio, RunCommand, extract_review_wav
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusState
from latintts.corpus.pairing import (
    PairingRecording,
    RepetitionGroup,
    TakeCandidate,
    pairing_from_dict,
    pairing_run_directory,
    require_safe_pairing_id,
)
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import (
    AudioMetadata,
    RecordingRecord,
    ReviewEvent,
    advance_recording,
    require_exact_fields,
)
from latintts.corpus.selection import PilotSelection
from latintts.corpus.store import persist_recording_transition, read_jsonl, write_jsonl_atomic
from latintts.corpus.transcripts import TranscriptRecord


@dataclass(frozen=True, slots=True)
class ReviewedWordSpan:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class ReviewedBoundaries:
    duration_seconds: float
    take_start: float
    take_end: float
    words: tuple[ReviewedWordSpan, ...]


def _number(value: float) -> str:
    return str(value)


def _quoted(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def write_textgrid(
    path: Path,
    *,
    duration_seconds: float,
    take_start: float,
    take_end: float,
    words: tuple[WordSpan | ReviewedWordSpan, ...],
) -> None:
    duration = _finite_number(duration_seconds, "duration")
    start = _finite_number(take_start, "take start")
    end = _finite_number(take_end, "take end")
    if duration <= 0:
        raise ValueError("TextGrid duration must be positive")
    if start < 0 or end <= start or end > duration:
        raise ValueError("take bounds must be a positive range inside the duration")
    intervals: list[tuple[float, float, str]] = []
    cursor = start
    for word in words:
        word_start = _finite_number(word.start_seconds, "word interval start")
        word_end = _finite_number(word.end_seconds, "word interval end")
        if not isinstance(word.text, str) or not word.text:
            raise ValueError("word interval text must be non-empty")
        if word_start < cursor or word_end <= word_start or word_end > end:
            raise ValueError("word intervals must be ordered inside the take")
        if word_start > cursor:
            intervals.append((cursor, word_start, ""))
        intervals.append((word_start, word_end, word.text))
        cursor = word_end
    if cursor < end:
        intervals.append((cursor, end, ""))
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {_number(duration)}",
        "tiers? <exists>",
        "size = 2",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "take"',
        f"        xmin = {_number(start)}",
        f"        xmax = {_number(end)}",
        "        intervals: size = 1",
        "        intervals [1]:",
        f"            xmin = {_number(start)}",
        f"            xmax = {_number(end)}",
        '            text = "take"',
        "    item [2]:",
        '        class = "IntervalTier"',
        '        name = "words"',
        f"        xmin = {_number(start)}",
        f"        xmax = {_number(end)}",
        f"        intervals: size = {len(intervals)}",
    ]
    for index, (start, end, text) in enumerate(intervals, 1):
        lines.extend(
            (
                f"        intervals [{index}]:",
                f"            xmin = {_number(start)}",
                f"            xmax = {_number(end)}",
                f"            text = {_quoted(text)}",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


_ASSIGNMENT = re.compile(r'^\s*([A-Za-z?]+) = (?:"((?:[^"]|"")*)"|([^\s]+))$')


def _finite_number(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _assignment(lines: list[str], index: int, name: str) -> tuple[str, int]:
    if index >= len(lines):
        raise ValueError(f"missing TextGrid {name}")
    match = _ASSIGNMENT.fullmatch(lines[index])
    if match is None or match.group(1) != name:
        raise ValueError(f"invalid TextGrid {name}")
    quoted = match.group(2)
    return ((quoted.replace('""', '"') if quoted is not None else match.group(3)), index + 1)


def _literal(lines: list[str], index: int, expected: str) -> int:
    if index >= len(lines) or lines[index].strip() != expected:
        raise ValueError(f"invalid TextGrid structure: expected {expected}")
    return index + 1


def _parsed_number(raw: str, field: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"TextGrid {field} must be numeric") from error
    return _finite_number(value, f"TextGrid {field}")


def read_textgrid(path: Path) -> ReviewedBoundaries:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[:3] != ['File type = "ooTextFile"', 'Object class = "TextGrid"', ""]:
        raise ValueError("invalid TextGrid header")
    index = 3
    global_start_raw, index = _assignment(lines, index, "xmin")
    duration_raw, index = _assignment(lines, index, "xmax")
    index = _literal(lines, index, "tiers? <exists>")
    index = _literal(lines, index, "size = 2")
    index = _literal(lines, index, "item []:")
    index = _literal(lines, index, "item [1]:")
    take_class, index = _assignment(lines, index, "class")
    take_name, index = _assignment(lines, index, "name")
    take_start_raw, index = _assignment(lines, index, "xmin")
    take_end_raw, index = _assignment(lines, index, "xmax")
    index = _literal(lines, index, "intervals: size = 1")
    index = _literal(lines, index, "intervals [1]:")
    interval_start_raw, index = _assignment(lines, index, "xmin")
    interval_end_raw, index = _assignment(lines, index, "xmax")
    take_label, index = _assignment(lines, index, "text")
    index = _literal(lines, index, "item [2]:")
    words_class, index = _assignment(lines, index, "class")
    words_name, index = _assignment(lines, index, "name")
    words_start_raw, index = _assignment(lines, index, "xmin")
    words_end_raw, index = _assignment(lines, index, "xmax")
    if index >= len(lines):
        raise ValueError("missing TextGrid interval count")
    size_match = re.fullmatch(r"\s*intervals: size = (\d+)", lines[index])
    if size_match is None:
        raise ValueError("invalid TextGrid interval count")
    index += 1
    intervals: list[tuple[float, float, str]] = []
    for interval_index in range(1, int(size_match.group(1)) + 1):
        if index >= len(lines):
            raise ValueError("missing TextGrid interval entry")
        if lines[index].strip() != f"intervals [{interval_index}]:":
            raise ValueError("invalid TextGrid interval order")
        index += 1
        start_raw, index = _assignment(lines, index, "xmin")
        end_raw, index = _assignment(lines, index, "xmax")
        text, index = _assignment(lines, index, "text")
        intervals.append(
            (
                _parsed_number(start_raw, "interval start"),
                _parsed_number(end_raw, "interval end"),
                text,
            )
        )
    if index != len(lines):
        raise ValueError("unexpected TextGrid content")
    if (
        take_class != "IntervalTier"
        or words_class != "IntervalTier"
        or take_name != "take"
        or words_name != "words"
        or take_label != "take"
    ):
        raise ValueError("invalid TextGrid tiers")
    global_start = _parsed_number(global_start_raw, "global start")
    duration = _parsed_number(duration_raw, "duration")
    take_start = _parsed_number(take_start_raw, "take start")
    take_end = _parsed_number(take_end_raw, "take end")
    if global_start != 0 or duration <= 0 or take_start < 0 or take_end <= take_start:
        raise ValueError("invalid TextGrid duration or take bounds")
    if take_end > duration:
        raise ValueError("take bounds exceed TextGrid duration")
    if (
        _parsed_number(interval_start_raw, "take interval start") != take_start
        or _parsed_number(interval_end_raw, "take interval end") != take_end
        or _parsed_number(words_start_raw, "words tier start") != take_start
        or _parsed_number(words_end_raw, "words tier end") != take_end
    ):
        raise ValueError("take interval differs from tier bounds")
    cursor = take_start
    words: list[ReviewedWordSpan] = []
    for start, end, text in intervals:
        if start < 0 or end <= start:
            raise ValueError("TextGrid interval must be a non-negative positive range")
        if start < cursor:
            raise ValueError("TextGrid intervals overlap")
        if start > cursor:
            raise ValueError("TextGrid intervals must exactly cover the words tier")
        if end > take_end:
            raise ValueError("TextGrid interval exceeds take bounds")
        if text:
            words.append(ReviewedWordSpan(text, start, end))
        cursor = end
    if cursor != take_end:
        raise ValueError("TextGrid intervals must exactly cover the words tier")
    return ReviewedBoundaries(duration, take_start, take_end, tuple(words))


def replay_review_events(
    automatic: dict[str, Any], events: tuple[ReviewEvent, ...]
) -> dict[str, Any]:
    if type(automatic) is not dict:
        raise TypeError("automatic review values must be an object")
    if type(events) is not tuple or any(type(event) is not ReviewEvent for event in events):
        raise TypeError("review events must be a tuple of ReviewEvent values")
    event_ids = tuple(event.review_event_id for event in events)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("review event history contains duplicate review_event_id")
    entity_ids = {event.entity_id for event in events}
    if len(entity_ids) > 1:
        raise ValueError("review event history must contain exactly one entity")
    effective = deepcopy(automatic)
    for event in events:
        before = _review_field_value(effective, event.field)
        if before != event.before:
            raise ValueError("review event before value does not match replay state")
        if event.field == "review_decision" and (
            event.before not in {"unreviewed", "approved", "rejected"}
            or event.after
            not in {
                "approved",
                "rejected",
            }
        ):
            raise ValueError("review_decision event must use the decision enum")
        _set_review_field(effective, event.field, event.after)
    return effective


_WORD_FIELD = re.compile(r"word:(\d+):(text|start_seconds|end_seconds)")


def _review_field_value(values: dict[str, Any], field: str) -> object:
    match = _WORD_FIELD.fullmatch(field)
    if match is None:
        if field not in values:
            raise ValueError(f"review event field is absent from automatic values: {field}")
        return values[field]
    words = values.get("words")
    index = int(match.group(1))
    if type(words) is not list or index >= len(words) or type(words[index]) is not dict:
        raise ValueError(f"review event word field is absent: {field}")
    key = match.group(2)
    if key not in words[index]:
        raise ValueError(f"review event word field is absent: {field}")
    return words[index][key]


def _set_review_field(values: dict[str, Any], field: str, value: object) -> None:
    match = _WORD_FIELD.fullmatch(field)
    if match is None:
        values[field] = deepcopy(value)
        return
    words = values["words"]
    assert isinstance(words, list)
    word = words[int(match.group(1))]
    assert isinstance(word, dict)
    word[match.group(2)] = deepcopy(value)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_text_atomic(path, _canonical_json(value) + "\n")


def _decode_recording(raw: dict[str, Any]) -> RecordingRecord:
    require_exact_fields(
        raw, frozenset(field.name for field in fields(RecordingRecord)), "recording"
    )
    metadata = raw["metadata"]
    if type(metadata) is not dict:
        raise TypeError("recording metadata must be an object")
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


def _load_selection(paths: CorpusPaths) -> PilotSelection:
    rows = read_jsonl(paths.manifests / "pilot-selection.json")
    if len(rows) != 1:
        raise ValueError("pilot-selection.json must contain exactly one row")
    raw = rows[0]
    require_exact_fields(
        raw, frozenset(field.name for field in fields(PilotSelection)), "selection"
    )
    if type(raw["recording_ids"]) is not list or type(raw["inventory_hashes"]) is not list:
        raise TypeError("selection IDs and hashes must be arrays")
    return PilotSelection(
        raw["schema_version"],
        raw["strategy"],
        tuple(raw["recording_ids"]),
        tuple(raw["inventory_hashes"]),
    )


def _load_transcript_layers(
    paths: CorpusPaths, selection: PilotSelection
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(paths.manifests / "transcripts.jsonl")
    by_id: dict[str, dict[str, Any]] = {}
    expected_fields = frozenset(field.name for field in fields(TranscriptRecord))
    for raw in rows:
        require_exact_fields(raw, expected_fields, "transcript")
        recording_id = raw["recording_id"]
        if not isinstance(recording_id, str) or recording_id in by_id:
            raise ValueError("transcripts must contain unique recording_id values")
        for field in ("source_text", "spoken_text", "normalized_text"):
            if not isinstance(raw[field], str) or not raw[field]:
                raise ValueError(f"transcript {field} must be a non-empty string")
        by_id[recording_id] = raw
    if set(by_id) != set(selection.recording_ids):
        raise ValueError("transcripts must exactly match pilot selection")
    return by_id


def _load_pairing_and_alignment(
    paths: CorpusPaths, config: CorpusConfig, recording: RecordingRecord
) -> tuple[Path, PairingRecording, dict[str, Any]]:
    run_directory = pairing_run_directory(paths, config.digest, recording.recording_id)
    pairing_path = run_directory / "pairing.json"
    pairing_rows = read_jsonl(pairing_path)
    if len(pairing_rows) != 1:
        raise ValueError("pairing artifact must contain exactly one row")
    pairing = pairing_from_dict(pairing_rows[0])
    alignment_rows = read_jsonl(run_directory / "alignment.json")
    if len(alignment_rows) != 1:
        raise ValueError("alignment artifact must contain exactly one row")
    alignment = alignment_rows[0]
    require_exact_fields(
        alignment,
        frozenset(
            {
                "schema_version",
                "recording_id",
                "status",
                "config_sha256",
                "pairing_artifact_sha256",
                "pairing_cache_key",
                "takes",
                "issues",
                "cache_key",
            }
        ),
        "alignment artifact",
    )
    if (
        alignment["schema_version"] != "1"
        or alignment["status"] != "success"
        or alignment["recording_id"] != recording.recording_id
        or alignment["config_sha256"] != config.digest
        or alignment["pairing_artifact_sha256"] != _digest(pairing_path)
        or alignment["pairing_cache_key"] != pairing.cache_key
        or alignment["issues"] != []
    ):
        raise ValueError("alignment artifact identity does not match review export")
    if type(alignment["takes"]) is not list:
        raise TypeError("alignment takes must be an array")
    return run_directory, pairing, alignment


def _alignment_by_take(
    pairing: PairingRecording, alignment: dict[str, Any], paths: CorpusPaths
) -> dict[tuple[str, int], dict[str, Any]]:
    decoded: dict[tuple[str, int], dict[str, Any]] = {}
    take_fields = frozenset(
        {
            "repetition_group_id",
            "unit_id",
            "take_index",
            "audio_sha256",
            "alignment_cache_key",
            "alignment_result",
        }
    )
    for raw in alignment["takes"]:
        if type(raw) is not dict:
            raise TypeError("alignment take must be an object")
        require_exact_fields(raw, take_fields, "alignment take")
        result = result_from_dict(raw["alignment_result"])
        key = (raw["repetition_group_id"], raw["take_index"])
        if key in decoded:
            raise ValueError("alignment artifact contains duplicate take identity")
        decoded[key] = raw | {"alignment_result": result}
    expected = {
        (outcome.group.repetition_group_id, take.take_index)
        for outcome in pairing.groups
        if outcome.group is not None
        for take in outcome.group.takes
    }
    if any(outcome.status != "selected" or outcome.group is None for outcome in pairing.groups):
        raise ValueError("review export requires selected pairing outcomes")
    if set(decoded) != expected:
        raise ValueError("alignment artifact must exactly cover selected pairing takes")
    for outcome in pairing.groups:
        assert outcome.group is not None
        for take in outcome.group.takes:
            row = decoded[(outcome.group.repetition_group_id, take.take_index)]
            result = row["alignment_result"]
            if (
                row["unit_id"] != outcome.group.unit_id
                or row["audio_sha256"] != take.audio_sha256
                or row["alignment_cache_key"] != take.alignment_cache_key
                or result.integrity_sha256 != take.alignment_integrity_sha256
            ):
                raise ValueError("alignment take does not match selected pairing evidence")
            candidate = paths.resolve_local(take.audio_relative_path)
            if not candidate.is_file() or _digest(candidate) != take.audio_sha256:
                raise ValueError("selected candidate audio reference or hash is invalid")
    return decoded


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}."
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _export_review_audio(
    recording: RecordingRecord,
    paths: CorpusPaths,
    destination: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    ffmpeg_version: str,
    run_command: RunCommand,
) -> DerivedAudio:
    derived = extract_review_wav(
        recording,
        paths,
        start_seconds,
        end_seconds,
        ffmpeg_version=ffmpeg_version,
        run_command=run_command,
    )
    _copy_file_atomic(paths.resolve_local(derived.relative_path), destination)
    return derived


def _require_canonical_descendant(
    root: Path,
    target: Path,
    *,
    kind: str,
    require_file: bool = False,
) -> Path:
    root_absolute = Path(os.path.abspath(root))
    target_absolute = Path(os.path.abspath(target))
    try:
        relative = target_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError(f"{kind} is outside its canonical root") from error
    current = root_absolute
    candidates = [current]
    for component in relative.parts:
        current /= component
        candidates.append(current)
    for current in candidates:
        if not current.exists() and not current.is_symlink():
            continue
        metadata = os.lstat(current)
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"{kind} path contains an alias or reparse point")
        if current.resolve() != current:
            raise ValueError(f"{kind} path is not canonical")
    if root_absolute.resolve(strict=False) != root_absolute:
        raise ValueError(f"{kind} root is not canonical")
    resolved_target = target_absolute.resolve(strict=False)
    try:
        resolved_target.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError(f"{kind} resolves outside its canonical root") from error
    if require_file and not target_absolute.is_file():
        raise ValueError(f"{kind} must be a regular file")
    return target_absolute


def _group_directory(run_directory: Path, group_id: str) -> Path:
    safe_id = require_safe_pairing_id(group_id, "repetition_group_id")
    review_root = _require_canonical_descendant(
        run_directory, run_directory / "review", kind="review root"
    )
    return _require_canonical_descendant(review_root, review_root / safe_id, kind="review group")


_BUNDLE_FILES = frozenset(
    {
        "index.html",
        "automatic.json",
        "decision.json",
        "take-1.wav",
        "take-1.TextGrid",
        "take-2.wav",
        "take-2.TextGrid",
    }
)


def _export_group(
    directory: Path,
    recording: RecordingRecord,
    group: RepetitionGroup,
    alignment: dict[tuple[str, int], dict[str, Any]],
    transcript: dict[str, Any],
    paths: CorpusPaths,
    *,
    ffmpeg_version: str,
    run_command: RunCommand,
) -> None:
    _require_canonical_descendant(directory.parent, directory, kind="review group")
    directory.mkdir(parents=True, exist_ok=True)
    existing_files = {item.name for item in directory.iterdir()}
    if existing_files:
        if existing_files != _BUNDLE_FILES or any(
            not _require_canonical_descendant(
                directory.parent,
                directory / name,
                kind="review bundle file",
                require_file=True,
            )
            for name in _BUNDLE_FILES
        ):
            raise ValueError("existing review bundle does not contain the exact file schema")
        return
    takes: list[dict[str, Any]] = []
    decisions: list[dict[str, str]] = []
    for take in group.takes:
        row = alignment[(group.repetition_group_id, take.take_index)]
        result = row["alignment_result"]
        audio_name = f"take-{take.take_index}.wav"
        textgrid_name = f"take-{take.take_index}.TextGrid"
        start_seconds = take.start_sample / 16_000
        end_seconds = take.end_sample / 16_000
        derived = _export_review_audio(
            recording,
            paths,
            directory / audio_name,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
        )
        start_frame = round(start_seconds * recording.metadata.sample_rate)
        end_frame = round(end_seconds * recording.metadata.sample_rate)
        duration = (end_frame - start_frame) / recording.metadata.sample_rate
        write_textgrid(
            directory / textgrid_name,
            duration_seconds=duration,
            take_start=0.0,
            take_end=duration,
            words=result.words,
        )
        entity_id = f"{group.repetition_group_id}-take-{take.take_index}"
        automatic_values = {
            "segment_start": 0.0,
            "segment_end": duration,
            "words": [
                {
                    "text": word.text,
                    "start_seconds": word.start_seconds,
                    "end_seconds": word.end_seconds,
                }
                for word in result.words
            ],
            "review_decision": "unreviewed",
        }
        takes.append(
            {
                "entity_id": entity_id,
                "take_index": take.take_index,
                "source_start_sample": start_frame,
                "source_end_sample": end_frame,
                "audio_file": audio_name,
                "audio_sha256": derived.sha256,
                "audio_provenance": derived.to_dict(),
                "textgrid_file": textgrid_name,
                "textgrid_sha256": _digest(directory / textgrid_name),
                "candidate_audio_sha256": take.audio_sha256,
                "candidate_scores": {
                    "alignment_score": take.alignment_score,
                    "coverage": take.coverage,
                },
                "quality_metrics": {
                    "coverage": result.coverage,
                    "mean_score": result.mean_score,
                },
                "warnings": list(result.warnings),
                "alignment_versions": {
                    "backend": result.backend,
                    "backend_version": result.backend_version,
                    "model_id": result.model_id,
                    "model_revision": result.model_revision,
                    "model_license": result.model_license,
                    "alignment_transform_version": result.alignment_transform_version,
                },
                "automatic_values": automatic_values,
            }
        )
        decisions.append(
            {
                "entity_id": entity_id,
                "decision": "unreviewed",
                "reason": "",
                "reviewer": "",
                "reviewed_at": "",
            }
        )
    automatic = {
        "schema_version": "1",
        "recording_id": recording.recording_id,
        "repetition_group_id": group.repetition_group_id,
        "unit_id": group.unit_id,
        "source_audio": {"relative_path": recording.relative_path, "sha256": recording.sha256},
        "text_layers": {
            "title_or_citation": recording.title_or_citation,
            "source_text": transcript["source_text"],
            "spoken_text": transcript["spoken_text"],
            "normalized_text": transcript["normalized_text"],
            "unit_spoken_text": group.text,
            "alignment_texts": [
                alignment[(group.repetition_group_id, index)]["alignment_result"].alignment_text
                for index in (1, 2)
            ],
        },
        "takes": takes,
    }
    _write_json(directory / "automatic.json", automatic)
    decision_path = directory / "decision.json"
    if not decision_path.exists():
        _write_json(
            decision_path,
            {
                "schema_version": "1",
                "recording_id": recording.recording_id,
                "repetition_group_id": group.repetition_group_id,
                "decisions": decisions,
            },
        )
    title = html.escape(recording.title_or_citation, quote=True)
    text = html.escape(group.text, quote=True)
    items = "\n".join(
        f'<section><h2>Take {index}</h2><audio controls src="take-{index}.wav"></audio></section>'
        for index in (1, 2)
    )
    _write_text_atomic(
        directory / "index.html",
        '<!doctype html>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<h1>{title}</h1>\n<p>{text}</p>\n{items}\n",
    )
    for name in _BUNDLE_FILES:
        _require_canonical_descendant(
            directory.parent,
            directory / name,
            kind="review bundle file",
            require_file=True,
        )


def export_review_bundle(
    paths: CorpusPaths,
    config: CorpusConfig,
    *,
    ffmpeg_version: str | None = None,
    run_command: RunCommand | None = None,
) -> tuple[Path, ...]:
    if type(paths) is not CorpusPaths or type(config) is not CorpusConfig:
        raise TypeError("export_review_bundle requires CorpusPaths and CorpusConfig")
    if not isinstance(ffmpeg_version, str) or not ffmpeg_version.strip():
        raise ValueError("export_review_bundle requires ffmpeg_version")
    command = subprocess.run if run_command is None else run_command
    selection = _load_selection(paths)
    transcripts = _load_transcript_layers(paths, selection)
    recording_rows = read_jsonl(paths.manifests / "recordings.jsonl")
    recordings = tuple(_decode_recording(row) for row in recording_rows)
    by_id = {recording.recording_id: recording for recording in recordings}
    if len(by_id) != len(recordings):
        raise ValueError("recordings contain duplicate recording_id")
    exported: list[Path] = []
    for recording_id, expected_hash in zip(
        selection.recording_ids, selection.inventory_hashes, strict=True
    ):
        recording = by_id.get(recording_id)
        if recording is None or recording.sha256 != expected_hash:
            raise ValueError("selection does not match recordings")
        if recording.state not in {CorpusState.ALIGNED, CorpusState.REVIEWED}:
            raise ValueError("review export requires ALIGNED or REVIEWED recording state")
        source = paths.resolve_local(recording.relative_path)
        if not source.is_file() or _digest(source) != recording.sha256:
            raise ValueError("source audio reference or hash is invalid")
        run_directory, pairing, artifact = _load_pairing_and_alignment(paths, config, recording)
        alignment = _alignment_by_take(pairing, artifact, paths)
        for outcome in pairing.groups:
            if outcome.group is None:
                raise ValueError("review export requires selected repetition groups")
            directory = _group_directory(run_directory, outcome.group.repetition_group_id)
            _export_group(
                directory,
                recording,
                outcome.group,
                alignment,
                transcripts[recording_id],
                paths,
                ffmpeg_version=ffmpeg_version,
                run_command=command,
            )
            exported.append(directory)
    return tuple(exported)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"invalid review JSON: {path.name}") from error
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


_AUTOMATIC_FIELDS = frozenset(
    {
        "schema_version",
        "recording_id",
        "repetition_group_id",
        "unit_id",
        "source_audio",
        "text_layers",
        "takes",
    }
)
_AUTOMATIC_TAKE_FIELDS = frozenset(
    {
        "entity_id",
        "take_index",
        "source_start_sample",
        "source_end_sample",
        "audio_file",
        "audio_sha256",
        "audio_provenance",
        "textgrid_file",
        "textgrid_sha256",
        "candidate_audio_sha256",
        "candidate_scores",
        "quality_metrics",
        "warnings",
        "alignment_versions",
        "automatic_values",
    }
)
_DECISION_FIELDS = frozenset({"schema_version", "recording_id", "repetition_group_id", "decisions"})
_DECISION_ITEM_FIELDS = frozenset({"entity_id", "decision", "reason", "reviewer", "reviewed_at"})
_AUTOMATIC_VALUE_FIELDS = frozenset({"segment_start", "segment_end", "words", "review_decision"})
_WORD_VALUE_FIELDS = frozenset({"text", "start_seconds", "end_seconds"})
_CANDIDATE_SCORE_FIELDS = frozenset({"alignment_score", "coverage"})
_QUALITY_FIELDS = frozenset({"coverage", "mean_score"})
_ALIGNMENT_VERSION_FIELDS = frozenset(
    {
        "backend",
        "backend_version",
        "model_id",
        "model_revision",
        "model_license",
        "alignment_transform_version",
    }
)


def _bundle_file(
    directory: Path,
    name: object,
    expected: str,
    kind: str,
    *,
    review_root: Path | None = None,
) -> Path:
    if name != expected or type(name) is not str:
        raise ValueError(f"review {kind} reference must be {expected}")
    canonical_root = directory if review_root is None else review_root
    return _require_canonical_descendant(
        canonical_root,
        directory / name,
        kind=f"review {kind}",
        require_file=True,
    )


def _validated_automatic_values(raw: object) -> dict[str, Any]:
    if type(raw) is not dict:
        raise TypeError("automatic_values must be an object")
    require_exact_fields(raw, _AUTOMATIC_VALUE_FIELDS, "automatic values")
    if raw["review_decision"] != "unreviewed":
        raise ValueError("automatic review_decision must remain unreviewed")
    start = _finite_number(raw["segment_start"], "automatic segment_start")
    end = _finite_number(raw["segment_end"], "automatic segment_end")
    if start < 0 or end <= start:
        raise ValueError("automatic segment bounds are invalid")
    words_raw = raw["words"]
    if type(words_raw) is not list:
        raise TypeError("automatic words must be an array")
    words: list[dict[str, Any]] = []
    cursor = start
    for word in words_raw:
        if type(word) is not dict:
            raise TypeError("automatic word must be an object")
        require_exact_fields(word, _WORD_VALUE_FIELDS, "automatic word")
        if not isinstance(word["text"], str) or not word["text"]:
            raise ValueError("automatic word text must be non-empty")
        word_start = _finite_number(word["start_seconds"], "automatic word start")
        word_end = _finite_number(word["end_seconds"], "automatic word end")
        if word_start < cursor or word_end <= word_start or word_end > end:
            raise ValueError("automatic words must be ordered inside the take")
        cursor = word_end
        words.append({"text": word["text"], "start_seconds": word_start, "end_seconds": word_end})
    return {
        "segment_start": start,
        "segment_end": end,
        "words": words,
        "review_decision": "unreviewed",
    }


def _event_identity(event: ReviewEvent) -> str:
    payload = event.to_dict()
    del payload["review_event_id"]
    return "review-" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _new_review_event(
    entity_id: str,
    field: str,
    before: object,
    after: object,
    decision: dict[str, Any],
) -> ReviewEvent:
    provisional = ReviewEvent(
        "1",
        "pending",
        entity_id,
        field,
        deepcopy(before),
        deepcopy(after),
        decision["reason"],
        decision["reviewer"],
        decision["reviewed_at"],
    )
    return ReviewEvent(
        provisional.schema_version,
        _event_identity(provisional),
        provisional.entity_id,
        provisional.field,
        provisional.before,
        provisional.after,
        provisional.reason,
        provisional.reviewer,
        provisional.reviewed_at,
    )


def _load_existing_review_events(path: Path) -> tuple[ReviewEvent, ...]:
    events = tuple(ReviewEvent.from_dict(row) for row in read_jsonl(path)) if path.exists() else ()
    event_ids = tuple(event.review_event_id for event in events)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("review history contains duplicate review_event_id")
    for event in events:
        if event.review_event_id != _event_identity(event):
            raise ValueError("review history contains a non-deterministic event ID")
        if event.before == event.after:
            raise ValueError("review history contains a no-op event")
    return events


def _event_fields(
    effective: dict[str, Any], target: dict[str, Any]
) -> tuple[tuple[str, object, object], ...]:
    changes: list[tuple[str, object, object]] = []
    for field in ("segment_start", "segment_end"):
        if effective[field] != target[field]:
            changes.append((field, effective[field], target[field]))
    effective_words = effective["words"]
    target_words = target["words"]
    if type(effective_words) is not list or type(target_words) is not list:
        raise ValueError("review word values must be arrays")
    if len(effective_words) != len(target_words):
        raise ValueError("review cannot change the automatic word count")
    for index, (before_word, after_word) in enumerate(
        zip(effective_words, target_words, strict=True)
    ):
        if type(before_word) is not dict or type(after_word) is not dict:
            raise ValueError("review word values must be objects")
        for key in ("text", "start_seconds", "end_seconds"):
            if before_word[key] != after_word[key]:
                changes.append((f"word:{index}:{key}", before_word[key], after_word[key]))
    if effective["review_decision"] != target["review_decision"]:
        changes.append(
            (
                "review_decision",
                effective["review_decision"],
                target["review_decision"],
            )
        )
    return tuple(changes)


def _validate_decision_item(raw: object) -> dict[str, Any]:
    if type(raw) is not dict:
        raise TypeError("review decision must be an object")
    require_exact_fields(raw, _DECISION_ITEM_FIELDS, "review decision")
    decision = raw["decision"]
    if decision not in {"unreviewed", "approved", "rejected"} or type(decision) is not str:
        raise ValueError("review decision must use the decision enum")
    if decision == "unreviewed":
        if any(raw[field] != "" for field in ("reason", "reviewer", "reviewed_at")):
            raise ValueError("unreviewed decision metadata must be empty")
    else:
        for field in ("reason", "reviewer", "reviewed_at"):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise ValueError(f"review {field} must be non-empty")
        ReviewEvent(
            "1",
            "timestamp-validation",
            raw["entity_id"],
            "review_decision",
            "unreviewed",
            decision,
            raw["reason"],
            raw["reviewer"],
            raw["reviewed_at"],
        )
    return raw


def _target_from_textgrid(
    automatic: dict[str, Any], boundaries: ReviewedBoundaries, decision: str
) -> dict[str, Any]:
    if len(boundaries.words) != len(automatic["words"]):
        raise ValueError("review TextGrid cannot change the automatic word count")
    return {
        "segment_start": boundaries.take_start,
        "segment_end": boundaries.take_end,
        "words": [
            {
                "text": word.text,
                "start_seconds": word.start_seconds,
                "end_seconds": word.end_seconds,
            }
            for word in boundaries.words
        ],
        "review_decision": decision,
    }


def _validate_take_summary(
    raw: dict[str, Any],
    take: TakeCandidate,
    result: AlignmentResult,
    recording: RecordingRecord,
) -> dict[str, Any]:
    candidate_scores = raw["candidate_scores"]
    quality = raw["quality_metrics"]
    versions = raw["alignment_versions"]
    if (
        type(candidate_scores) is not dict
        or type(quality) is not dict
        or type(versions) is not dict
    ):
        raise TypeError("review take metric and version summaries must be objects")
    require_exact_fields(candidate_scores, _CANDIDATE_SCORE_FIELDS, "candidate scores")
    require_exact_fields(quality, _QUALITY_FIELDS, "quality metrics")
    require_exact_fields(versions, _ALIGNMENT_VERSION_FIELDS, "alignment versions")
    if candidate_scores != {
        "alignment_score": take.alignment_score,
        "coverage": take.coverage,
    }:
        raise ValueError("review candidate scores do not match pairing")
    if quality != {"coverage": result.coverage, "mean_score": result.mean_score}:
        raise ValueError("review quality metrics do not match alignment")
    expected_versions = {
        "backend": result.backend,
        "backend_version": result.backend_version,
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "model_license": result.model_license,
        "alignment_transform_version": result.alignment_transform_version,
    }
    if versions != expected_versions:
        raise ValueError("review alignment versions do not match alignment")
    warnings = raw["warnings"]
    if type(warnings) is not list or warnings != list(result.warnings):
        raise ValueError("review warnings do not match alignment")
    if raw["candidate_audio_sha256"] != take.audio_sha256:
        raise ValueError("review candidate audio hash does not match pairing")
    frame_rate = recording.metadata.sample_rate
    expected_start = round(take.start_sample / 16_000 * frame_rate)
    expected_end = round(take.end_sample / 16_000 * frame_rate)
    if raw["source_start_sample"] != expected_start or raw["source_end_sample"] != expected_end:
        raise ValueError("review source sample bounds do not match pairing")
    automatic = _validated_automatic_values(raw["automatic_values"])
    expected_automatic = {
        "segment_start": 0.0,
        "segment_end": (expected_end - expected_start) / frame_rate,
        "words": [
            {
                "text": word.text,
                "start_seconds": word.start_seconds,
                "end_seconds": word.end_seconds,
            }
            for word in result.words
        ],
        "review_decision": "unreviewed",
    }
    if automatic != expected_automatic:
        raise ValueError("review automatic values do not match alignment and source audio")
    return automatic


def _validate_trusted_review_audio(
    raw: dict[str, Any],
    audio_path: Path,
    recording: RecordingRecord,
    take: TakeCandidate,
    paths: CorpusPaths,
    *,
    ffmpeg_version: str,
    run_command: RunCommand,
) -> None:
    provenance_raw = raw["audio_provenance"]
    if type(provenance_raw) is not dict:
        raise TypeError("review audio_provenance must be an object")
    provenance = DerivedAudio.from_dict(provenance_raw)
    trusted = extract_review_wav(
        recording,
        paths,
        take.start_sample / 16_000,
        take.end_sample / 16_000,
        ffmpeg_version=ffmpeg_version,
        cached=provenance,
        run_command=run_command,
    )
    if raw["audio_sha256"] != trusted.sha256 or _digest(audio_path) != trusted.sha256:
        raise ValueError("review audio does not match the trusted source-derived clip")


def import_review_bundle(
    paths: CorpusPaths,
    config: CorpusConfig,
    *,
    ffmpeg_version: str | None = None,
    run_command: RunCommand | None = None,
) -> bool:
    if type(paths) is not CorpusPaths or type(config) is not CorpusConfig:
        raise TypeError("import_review_bundle requires CorpusPaths and CorpusConfig")
    if not isinstance(ffmpeg_version, str) or not ffmpeg_version.strip():
        raise ValueError("import_review_bundle requires ffmpeg_version")
    command = subprocess.run if run_command is None else run_command
    selection = _load_selection(paths)
    transcripts = _load_transcript_layers(paths, selection)
    recordings_path = paths.manifests / "recordings.jsonl"
    recording_rows = read_jsonl(recordings_path)
    recordings = tuple(_decode_recording(row) for row in recording_rows)
    by_id = {
        recording.recording_id: (index, recording) for index, recording in enumerate(recordings)
    }
    if len(by_id) != len(recordings):
        raise ValueError("recordings contain duplicate recording_id")
    review_path = paths.manifests / "review.jsonl"
    existing = _load_existing_review_events(review_path)
    events_by_entity: dict[str, list[ReviewEvent]] = {}
    for event in existing:
        events_by_entity.setdefault(event.entity_id, []).append(event)
    automatic_by_entity: dict[str, dict[str, Any]] = {}
    decision_rows: dict[str, dict[str, Any]] = {}
    expected_entities: set[str] = set()
    alignment_paths: dict[str, Path] = {}
    run_directories: dict[str, Path] = {}
    for recording_id, expected_hash in zip(
        selection.recording_ids, selection.inventory_hashes, strict=True
    ):
        item = by_id.get(recording_id)
        if item is None or item[1].sha256 != expected_hash:
            raise ValueError("selection does not match recordings")
        recording = item[1]
        if recording.state not in {CorpusState.ALIGNED, CorpusState.REVIEWED}:
            raise ValueError("review import requires ALIGNED or REVIEWED recording state")
        source = paths.resolve_local(recording.relative_path)
        if not source.is_file() or _digest(source) != recording.sha256:
            raise ValueError("source audio reference or hash is invalid")
        run_directory, pairing, artifact = _load_pairing_and_alignment(paths, config, recording)
        alignment_paths[recording_id] = run_directory / "alignment.json"
        run_directories[recording_id] = run_directory
        decoded_alignment = _alignment_by_take(pairing, artifact, paths)
        for outcome in pairing.groups:
            group = outcome.group
            if outcome.status != "selected" or group is None:
                raise ValueError("review import requires selected repetition groups")
            directory = _group_directory(run_directory, group.repetition_group_id)
            if (
                not directory.is_dir()
                or {item.name for item in directory.iterdir()} != _BUNDLE_FILES
            ):
                raise ValueError("review bundle does not contain the exact file schema")
            bundle_paths = {
                name: _require_canonical_descendant(
                    directory.parent,
                    directory / name,
                    kind="review bundle file",
                    require_file=True,
                )
                for name in _BUNDLE_FILES
            }
            automatic_raw = _read_json(bundle_paths["automatic.json"])
            require_exact_fields(automatic_raw, _AUTOMATIC_FIELDS, "review automatic")
            if (
                automatic_raw["schema_version"] != "1"
                or automatic_raw["recording_id"] != recording_id
                or automatic_raw["repetition_group_id"] != group.repetition_group_id
                or automatic_raw["unit_id"] != group.unit_id
            ):
                raise ValueError("review automatic identity does not match pairing")
            source_raw = automatic_raw["source_audio"]
            if type(source_raw) is not dict:
                raise TypeError("review source_audio must be an object")
            require_exact_fields(
                source_raw, frozenset({"relative_path", "sha256"}), "review source audio"
            )
            if source_raw != {
                "relative_path": recording.relative_path,
                "sha256": recording.sha256,
            }:
                raise ValueError("review source audio identity does not match recording")
            text_layers = automatic_raw["text_layers"]
            if type(text_layers) is not dict:
                raise TypeError("review text_layers must be an object")
            require_exact_fields(
                text_layers,
                frozenset(
                    {
                        "title_or_citation",
                        "source_text",
                        "spoken_text",
                        "normalized_text",
                        "unit_spoken_text",
                        "alignment_texts",
                    }
                ),
                "review text layers",
            )
            transcript = transcripts[recording_id]
            if (
                text_layers["title_or_citation"] != recording.title_or_citation
                or text_layers["source_text"] != transcript["source_text"]
                or text_layers["spoken_text"] != transcript["spoken_text"]
                or text_layers["normalized_text"] != transcript["normalized_text"]
                or text_layers["unit_spoken_text"] != group.text
                or type(text_layers["alignment_texts"]) is not list
            ):
                raise ValueError("review text layers do not match pairing")
            expected_alignment_texts = [
                decoded_alignment[(group.repetition_group_id, index)][
                    "alignment_result"
                ].alignment_text
                for index in (1, 2)
            ]
            if text_layers["alignment_texts"] != expected_alignment_texts:
                raise ValueError("review alignment text layers do not match alignment")
            takes_raw = automatic_raw["takes"]
            if type(takes_raw) is not list or len(takes_raw) != 2:
                raise ValueError("review automatic must contain exactly two takes")
            takes_by_index: dict[int, dict[str, Any]] = {}
            for take_raw in takes_raw:
                if type(take_raw) is not dict:
                    raise TypeError("review automatic take must be an object")
                require_exact_fields(take_raw, _AUTOMATIC_TAKE_FIELDS, "review automatic take")
                take_index = take_raw["take_index"]
                if (
                    take_index not in (1, 2)
                    or type(take_index) is not int
                    or take_index in takes_by_index
                ):
                    raise ValueError("review automatic take identity must be unique")
                takes_by_index[take_index] = take_raw
            decision_raw = _read_json(bundle_paths["decision.json"])
            require_exact_fields(decision_raw, _DECISION_FIELDS, "review decision file")
            if (
                decision_raw["schema_version"] != "1"
                or decision_raw["recording_id"] != recording_id
                or decision_raw["repetition_group_id"] != group.repetition_group_id
            ):
                raise ValueError("review decision identity does not match bundle")
            decisions_raw = decision_raw["decisions"]
            if type(decisions_raw) is not list or len(decisions_raw) != 2:
                raise ValueError("review decision file must contain exactly two decisions")
            decisions = tuple(_validate_decision_item(item) for item in decisions_raw)
            decision_by_entity = {item["entity_id"]: item for item in decisions}
            if len(decision_by_entity) != 2:
                raise ValueError("review decision entity_id must be unique")
            for take in group.takes:
                take_raw = takes_by_index[take.take_index]
                aligned = decoded_alignment[(group.repetition_group_id, take.take_index)]
                entity_id = f"{group.repetition_group_id}-take-{take.take_index}"
                expected_entities.add(entity_id)
                if take_raw["entity_id"] != entity_id or set(decision_by_entity) != {
                    f"{group.repetition_group_id}-take-1",
                    f"{group.repetition_group_id}-take-2",
                }:
                    raise ValueError("review bundle entity identity does not match pairing")
                audio_path = _bundle_file(
                    directory,
                    take_raw["audio_file"],
                    f"take-{take.take_index}.wav",
                    "audio",
                    review_root=directory.parent,
                )
                if take_raw["audio_sha256"] != _digest(audio_path):
                    raise ValueError("review audio hash does not match automatic.json")
                _validate_trusted_review_audio(
                    take_raw,
                    audio_path,
                    recording,
                    take,
                    paths,
                    ffmpeg_version=ffmpeg_version,
                    run_command=command,
                )
                textgrid_path = _bundle_file(
                    directory,
                    take_raw["textgrid_file"],
                    f"take-{take.take_index}.TextGrid",
                    "TextGrid",
                    review_root=directory.parent,
                )
                result = aligned["alignment_result"]
                automatic_values = _validate_take_summary(take_raw, take, result, recording)
                boundaries = read_textgrid(textgrid_path)
                if boundaries.duration_seconds != automatic_values["segment_end"]:
                    raise ValueError("review TextGrid duration differs from automatic audio")
                decision = decision_by_entity[entity_id]
                automatic_by_entity[entity_id] = automatic_values
                decision_rows[entity_id] = decision
                if decision["decision"] != "unreviewed":
                    decision_rows[entity_id] = decision | {
                        "target": _target_from_textgrid(
                            automatic_values, boundaries, decision["decision"]
                        )
                    }
    unknown_entities = set(events_by_entity) - expected_entities
    if unknown_entities:
        raise ValueError("review history contains unknown entity_id")
    new_events: list[ReviewEvent] = []
    effective_by_entity: dict[str, dict[str, Any]] = {}
    for entity_id in sorted(expected_entities):
        automatic = automatic_by_entity[entity_id]
        prior = tuple(events_by_entity.get(entity_id, ()))
        effective = replay_review_events(automatic, prior)
        decision = decision_rows[entity_id]
        if decision["decision"] == "unreviewed":
            effective_by_entity[entity_id] = effective
            continue
        target = decision["target"]
        changes = _event_fields(effective, target)
        if not changes:
            if not prior:
                raise ValueError("final review must append a human decision event")
            latest = prior[-1]
            metadata = (latest.reason, latest.reviewer, latest.reviewed_at)
            submitted = (decision["reason"], decision["reviewer"], decision["reviewed_at"])
            if metadata != submitted:
                raise ValueError("review submission conflicts with existing effective values")
        for field, before, after in changes:
            event = _new_review_event(entity_id, field, before, after, decision)
            new_events.append(event)
            _set_review_field(effective, field, after)
        effective_by_entity[entity_id] = effective
    if new_events:
        combined = (*existing, *new_events)
        ids = tuple(event.review_event_id for event in combined)
        if len(ids) != len(set(ids)):
            raise ValueError("review submission conflicts with existing event IDs")
        write_jsonl_atomic(review_path, (event.to_dict() for event in combined))
    all_complete = True
    for entity_id in expected_entities:
        entity_events = (
            *events_by_entity.get(entity_id, ()),
            *(event for event in new_events if event.entity_id == entity_id),
        )
        if effective_by_entity[entity_id]["review_decision"] not in {
            "approved",
            "rejected",
        } or not any(event.field == "review_decision" for event in entity_events):
            all_complete = False
            break
    if not all_complete:
        return False
    current = recordings
    for recording_id in selection.recording_ids:
        index, _ = by_id[recording_id]
        record = current[index]
        if record.state is CorpusState.REVIEWED:
            continue
        timestamp = datetime.now(timezone.utc).isoformat()
        advanced, processing_event = advance_recording(
            record,
            CorpusState.REVIEWED,
            input_sha256s=(
                record.sha256,
                _digest(alignment_paths[recording_id]),
                _digest(review_path),
            ),
            config_sha256=config.digest,
            tool_versions=("human-review-v1",),
            started_at=timestamp,
            finished_at=timestamp,
            result="success",
        )
        current = (*current[:index], advanced, *current[index + 1 :])
        persist_recording_transition(
            recordings_path=recordings_path,
            events_path=run_directories[recording_id].parent / "processing-events.jsonl",
            recordings=current,
            event=processing_event,
        )
    return True
