from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from latintts.corpus.alignment import AlignmentResult, WordSpan
from latintts.corpus.audio import (
    DerivedAudio,
    RunCommand,
    _raw_source,
    _validate_source_hash,
    extract_analysis_segment,
    extract_lossless_segment,
)
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.pairing import PairingRecording
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import (
    ProcessingEvent,
    RecordingRecord,
    ReviewEvent,
    RightsRecord,
    SegmentRecord,
    advance_recording,
    require_exact_fields,
)
from latintts.corpus.review import (
    _alignment_by_take,
    _decode_recording,
    _digest,
    _event_identity,
    _load_existing_review_events,
    _load_pairing_and_alignment,
    _load_selection,
    _load_transcript_layers,
    _pairing_entity_id,
    _require_canonical_descendant,
    _review_entity_id,
    _validate_existing_pairing_corrections,
    import_review_bundle,
    replay_review_events,
)
from latintts.corpus.store import (
    persist_recording_transition,
    processing_event_exists,
    read_jsonl,
    write_jsonl_atomic,
)
from latintts.corpus.transcripts import SpokenUnit
from latintts.normalization import tokenize_words

LosslessExtractor = Callable[..., DerivedAudio]
AnalysisExtractor = Callable[..., DerivedAudio]


@dataclass(frozen=True, slots=True)
class _ManifestContext:
    recording_index: int
    recording: RecordingRecord
    transcript: dict[str, Any]
    units: tuple[SpokenUnit, ...]
    pairing: PairingRecording
    alignments: dict[tuple[str, int], dict[str, Any]]
    analysis: DerivedAudio
    pairing_sha256: str
    alignment_sha256: str
    run_directory: Path


def require_approval(
    *,
    review_decision: str | None,
    pronunciation_warning_codes: tuple[str, ...],
) -> None:
    if review_decision != "approved":
        raise CorpusFailure("REVIEW_REQUIRED", "take lacks an approved human review event")
    if pronunciation_warning_codes:
        raise CorpusFailure(
            "PRONUNCIATION_NEEDS_REVIEW",
            ", ".join(pronunciation_warning_codes),
        )


def _sample_position(seconds: float, sample_rate: int) -> int:
    """Map seconds to the nearest sample with Python's deterministic half-even rounding."""
    return round(seconds * sample_rate)


def _pronunciation_slice(
    transcript: dict[str, Any], unit: SpokenUnit
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], tuple[str, ...], str]:
    plan = transcript["pronunciation_plan"]
    if type(plan) is not dict:
        raise TypeError("transcript pronunciation_plan must be an object")
    if plan.get("schema_version") != "1" or plan.get("rule_version") != ("ecclesiastical-roman-v1"):
        raise CorpusFailure("MANIFEST_SCHEMA_MISMATCH", "pronunciation plan version is invalid")
    tokens = plan.get("tokens")
    if type(tokens) is not list:
        raise TypeError("pronunciation plan tokens must be an array")
    start = unit.token_start_index
    end = unit.token_end_index
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or end > len(tokens)
    ):
        raise ValueError("spoken unit pronunciation token slice is out of range or empty")
    selected = tokens[start:end]
    words = tokenize_words(unit.text)
    if len(selected) != len(words):
        raise ValueError("spoken unit token slice count differs from unit words")
    ipa: list[str] = []
    phonemes: list[tuple[str, ...]] = []
    warnings: list[str] = []
    normalized: list[str] = []
    for raw, word in zip(selected, words, strict=True):
        if type(raw) is not dict or raw.get("surface") != word.surface:
            raise ValueError("spoken unit token slice does not match unit text")
        ipa_value = raw.get("ipa")
        phoneme_values = raw.get("model_phonemes")
        warning_values = raw.get("warning_codes")
        normalized_value = raw.get("normalized")
        if not isinstance(ipa_value, str) or not ipa_value:
            raise ValueError("pronunciation token IPA must be non-empty")
        if (
            type(phoneme_values) is not list
            or not phoneme_values
            or any(not isinstance(value, str) or not value for value in phoneme_values)
        ):
            raise ValueError("pronunciation token model phonemes must be non-empty")
        if type(warning_values) is not list or any(
            not isinstance(value, str) or not value for value in warning_values
        ):
            raise TypeError("pronunciation warning codes must be a string array")
        if not isinstance(normalized_value, str) or not normalized_value:
            raise ValueError("pronunciation token normalized text must be non-empty")
        ipa.append(ipa_value)
        phonemes.append(tuple(phoneme_values))
        warnings.extend(warning_values)
        normalized.append(normalized_value)
    return tuple(ipa), tuple(phonemes), tuple(warnings), " ".join(normalized)


def _automatic_values(
    recording: RecordingRecord,
    take_start_sample: int,
    take_end_sample: int,
    result: AlignmentResult,
) -> dict[str, Any]:
    duration = (
        _sample_position(
            (take_end_sample - take_start_sample) / 16_000,
            recording.metadata.sample_rate,
        )
        / recording.metadata.sample_rate
    )
    return {
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


def _effective_word_spans(
    effective: dict[str, Any], result: AlignmentResult
) -> tuple[WordSpan, ...]:
    words = effective["words"]
    if type(words) is not list or len(words) != len(result.words):
        raise ValueError("effective review words do not cover the alignment")
    segment_start = effective["segment_start"]
    if type(segment_start) not in (int, float):
        raise TypeError("effective segment_start must be numeric")
    output: list[WordSpan] = []
    for index, (raw, original) in enumerate(zip(words, result.words, strict=True)):
        if type(raw) is not dict:
            raise TypeError("effective review word must be an object")
        output.append(
            WordSpan(
                raw["text"],
                raw["start_seconds"] - segment_start,
                raw["end_seconds"] - segment_start,
                original.score,
                index,
            )
        )
    return tuple(output)


def _segment_identity(recording_id: str, group_id: str, take_index: int) -> str:
    payload = f"{recording_id}\0{group_id}\0{take_index}".encode()
    return "segment-" + hashlib.sha256(payload).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class _ApprovedProposal:
    ordinal: int
    unit_id: str
    spoken_text: str
    repetition_group_id: str
    take_index: int
    source_start: int
    source_end: int
    absolute_start: float
    absolute_end: float
    effective: dict[str, Any]
    result: AlignmentResult
    ipa: tuple[str, ...]
    phonemes: tuple[tuple[str, ...], ...]
    normalized_text: str
    review_event_ids: tuple[str, ...]
    cached_quality: DerivedAudio | None


def build_approved_segments(
    *,
    recording: RecordingRecord,
    transcript: dict[str, Any],
    units: tuple[SpokenUnit, ...],
    rights: Mapping[str, RightsRecord],
    pairing: PairingRecording,
    alignments: Mapping[tuple[str, int], AlignmentResult],
    review_events: tuple[ReviewEvent, ...],
    analysis_audio: DerivedAudio,
    paths: CorpusPaths,
    config: CorpusConfig,
    ffmpeg_version: str,
    run_command: RunCommand = subprocess.run,
    probe_command: RunCommand = subprocess.run,
    decode_command: RunCommand = subprocess.run,
    extract_lossless: LosslessExtractor = extract_lossless_segment,
    extract_analysis: AnalysisExtractor = extract_analysis_segment,
    _validate_only: bool = False,
) -> tuple[SegmentRecord, ...]:
    if recording.rights_id not in rights:
        raise ValueError("recording rights_id does not exist")
    right = rights[recording.rights_id]
    if not right.allow_local_processing or not right.allow_model_training:
        raise CorpusFailure("RIGHTS_SCOPE_UNCONFIRMED", "rights do not permit corpus training")
    if pairing.recording_id != recording.recording_id or pairing.config_sha256 != config.digest:
        raise ValueError("pairing provenance does not match recording and config")
    if analysis_audio.mode != "analysis" or (
        analysis_audio.relative_path != pairing.analysis_audio_relative_path
        or analysis_audio.sha256 != pairing.analysis_audio_sha256
    ):
        raise ValueError("analysis audio provenance does not match pairing")
    units_by_id = {unit.unit_id: unit for unit in units}
    if len(units_by_id) != len(units):
        raise ValueError("spoken units contain duplicate unit_id")
    event_ids = tuple(event.review_event_id for event in review_events)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("review history contains duplicate review_event_id")
    for event in review_events:
        if event.review_event_id != _event_identity(event):
            raise ValueError("review history contains a non-deterministic event ID")
    proposals: list[_ApprovedProposal] = []
    decided_entities: set[str] = set()
    expected_entities: set[str] = set()
    source_ranges: list[tuple[int, int, str]] = []
    for outcome in pairing.groups:
        group = outcome.group
        if outcome.status != "selected" or group is None:
            raise ValueError("manifest requires selected pairing outcomes")
        unit = units_by_id.get(group.unit_id)
        if unit is None or group.text != unit.text:
            raise ValueError("pairing unit does not match transcript")
        ipa, phonemes, warning_codes, normalized_text = _pronunciation_slice(transcript, unit)
        pairing_entity = _pairing_entity_id(recording.recording_id, unit.unit_id)
        for take in group.takes:
            key = (group.repetition_group_id, take.take_index)
            result = alignments.get(key)
            if result is None:
                raise ValueError("alignment does not cover selected pairing take")
            entity_id = _review_entity_id(
                recording.recording_id, group.repetition_group_id, take.take_index
            )
            expected_entities.add(entity_id)
            take_events = tuple(
                event
                for event in review_events
                if event.entity_id == entity_id and event.field != "pairing_selected_split"
            )
            automatic = _automatic_values(recording, take.start_sample, take.end_sample, result)
            effective = replay_review_events(automatic, take_events, expected_entity_id=entity_id)
            decision = effective["review_decision"]
            if decision in {"approved", "rejected"} and any(
                event.field == "review_decision" for event in take_events
            ):
                decided_entities.add(entity_id)
            if decision == "rejected":
                continue
            require_approval(
                review_decision=decision,
                pronunciation_warning_codes=warning_codes,
            )
            local_start = effective["segment_start"]
            local_end = effective["segment_end"]
            if type(local_start) not in (int, float) or type(local_end) not in (int, float):
                raise TypeError("effective review bounds must be numeric")
            absolute_start = take.start_sample / 16_000 + local_start
            absolute_end = take.start_sample / 16_000 + local_end
            source_start = _sample_position(absolute_start, recording.metadata.sample_rate)
            source_end = _sample_position(absolute_end, recording.metadata.sample_rate)
            if source_start < 0 or source_end <= source_start:
                raise ValueError("approved source sample range is empty after rounding")
            source_ranges.append((source_start, source_end, entity_id))
            evidence_audio = (
                group.selected_evidence.first_audio
                if take.take_index == 1
                else group.selected_evidence.second_audio
            )
            cached_quality = None
            if (
                evidence_audio is not None
                and evidence_audio.start_seconds == absolute_start
                and evidence_audio.end_seconds == absolute_end
            ):
                cached_quality = evidence_audio
            consumed_entities = {pairing_entity, entity_id}
            consumed_events = tuple(
                event.review_event_id
                for event in review_events
                if event.entity_id in consumed_entities
            )
            proposals.append(
                _ApprovedProposal(
                    unit.ordinal,
                    unit.unit_id,
                    unit.text,
                    group.repetition_group_id,
                    take.take_index,
                    source_start,
                    source_end,
                    absolute_start,
                    absolute_end,
                    effective,
                    result,
                    ipa,
                    phonemes,
                    normalized_text,
                    consumed_events,
                    cached_quality,
                )
            )
    if decided_entities != expected_entities:
        raise CorpusFailure("REVIEW_REQUIRED", "recording still contains unreviewed takes")
    ordered_ranges = sorted(source_ranges)
    for left_range, right_range in pairwise(ordered_ranges):
        if left_range[1] > right_range[0]:
            raise ValueError("approved source sample ranges overlap after boundary rounding")
    if _validate_only:
        return ()
    rows_with_order: list[tuple[int, SegmentRecord]] = []
    for proposal in proposals:
        lossless = extract_lossless(
            recording,
            paths,
            proposal.source_start / recording.metadata.sample_rate,
            proposal.source_end / recording.metadata.sample_rate,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
            probe_command=probe_command,
            decode_command=decode_command,
        )
        quality = extract_analysis(
            analysis_audio,
            paths,
            proposal.absolute_start,
            proposal.absolute_end,
            ffmpeg_version=ffmpeg_version,
            cached=proposal.cached_quality,
            run_command=run_command,
        )
        if lossless.mode != "lossless" or lossless.source_sha256 != recording.sha256:
            raise ValueError("lossless derived clip provenance does not match raw recording")
        if quality.mode != "candidate" or quality.metrics is None:
            raise ValueError("quality metric clip must be a 16 kHz analysis segment")
        row = SegmentRecord(
            schema_version="1",
            corpus_version="corpus-v1",
            config_sha256=config.digest,
            segment_id=_segment_identity(
                recording.recording_id,
                proposal.repetition_group_id,
                proposal.take_index,
            ),
            recording_id=recording.recording_id,
            text_unit_id=proposal.unit_id,
            repetition_group_id=proposal.repetition_group_id,
            take_index=proposal.take_index,
            source_audio_sha256=recording.sha256,
            source_start_sample=proposal.source_start,
            source_end_sample=proposal.source_end,
            derived_audio_relative_path=lossless.relative_path,
            derived_audio_sha256=lossless.sha256,
            source_text_id=transcript["selected_candidate_id"],
            spoken_text=proposal.spoken_text,
            normalized_text=proposal.normalized_text,
            pronunciation_schema_version=transcript["pronunciation_plan"]["schema_version"],
            rule_version=transcript["pronunciation_plan"]["rule_version"],
            ipa_by_token=proposal.ipa,
            model_phonemes_by_token=proposal.phonemes,
            word_spans=_effective_word_spans(proposal.effective, proposal.result),
            alignment_backend=proposal.result.backend,
            alignment_model_id=proposal.result.model_id,
            alignment_model_revision=proposal.result.model_revision,
            alignment_model_license=proposal.result.model_license,
            alignment_level=proposal.result.alignment_level,
            phoneme_timing_status=proposal.result.phoneme_timing_status,
            quality_metrics=quality.metrics,
            quality_metric_audio_sha256=quality.sha256,
            quality_labels=(),
            review_event_ids=proposal.review_event_ids,
            rights_id=recording.rights_id,
            split="unassigned",
        )
        rows_with_order.append((proposal.ordinal, row))
    return tuple(
        row
        for _, row in sorted(
            rows_with_order,
            key=lambda item: (recording.recording_id, item[0], item[1].take_index),
        )
    )


def _load_rights(paths: CorpusPaths) -> dict[str, RightsRecord]:
    rows = read_jsonl(paths.manifests / "rights.jsonl")
    records = tuple(RightsRecord.from_dict(row) for row in rows)
    by_id = {record.rights_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("rights.jsonl contains duplicate rights_id")
    return by_id


def _validate_transcript_sources(transcript: dict[str, Any], recording_id: str) -> None:
    if (
        transcript.get("schema_version") != "1"
        or transcript.get("recording_id") != recording_id
        or transcript.get("state") != "TRANSCRIPT_CONFIRMED"
    ):
        raise ValueError("transcript identity or schema is invalid")
    candidates = transcript.get("source_candidates")
    if type(candidates) is not list or not candidates:
        raise ValueError("transcript source_candidates must be a non-empty array")
    fields = frozenset(
        {
            "candidate_id",
            "source_id",
            "source_url",
            "source_version",
            "accessed_at",
            "source_sha256",
            "source_text",
        }
    )
    by_id: dict[str, dict[str, Any]] = {}
    for raw in candidates:
        if type(raw) is not dict:
            raise TypeError("transcript source candidate must be an object")
        require_exact_fields(raw, fields, "transcript source candidate")
        candidate_id = raw["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in by_id:
            raise ValueError("transcript source candidate identities must be unique")
        source_text = raw["source_text"]
        if not isinstance(source_text, str) or not source_text:
            raise ValueError("transcript source candidate text must be non-empty")
        if raw["source_sha256"] != hashlib.sha256(source_text.encode()).hexdigest():
            raise ValueError("transcript source candidate hash does not match text")
        by_id[candidate_id] = raw
    selected_candidate_id = transcript.get("selected_candidate_id")
    if not isinstance(selected_candidate_id, str):
        raise ValueError("selected source candidate identity must be a string")
    selected = by_id.get(selected_candidate_id)
    if selected is None or selected["source_text"] != transcript.get("source_text"):
        raise ValueError("selected source candidate does not bind transcript source_text")


def _validate_processing_history(
    rows: tuple[dict[str, Any], ...],
    context: _ManifestContext,
    review_sha256: str,
    config: CorpusConfig,
    paths: CorpusPaths,
) -> None:
    from latintts.corpus.cli import (
        _expected_cached_transition,
        _human_pairing_transition_exists,
        _pairing_tool_versions,
    )

    events = tuple(ProcessingEvent.from_dict(row) for row in rows)
    ids = tuple(event.event_id for event in events)
    semantics = tuple(
        (
            event.recording_id,
            event.previous_state,
            event.target_state,
            event.input_sha256s,
            event.config_sha256,
            event.tool_versions,
            event.result,
        )
        for event in events
    )
    if len(ids) != len(set(ids)) or len(semantics) != len(set(semantics)):
        raise ValueError("processing history contains duplicate identities")
    recording_events = tuple(
        event for event in events if event.recording_id == context.recording.recording_id
    )
    if not recording_events or any(
        left.target_state is not right.previous_state for left, right in pairwise(recording_events)
    ):
        raise ValueError("processing history contains a stale state chain")
    if recording_events[-1].target_state is not context.recording.state:
        raise ValueError("recording state is not bound to processing history")
    paired = _expected_cached_transition(
        context.recording,
        CorpusState.SEGMENTED,
        CorpusState.PAIRED,
        input_sha256s=(
            context.recording.sha256,
            context.pairing.segmentation_artifact_sha256,
            context.pairing_sha256,
        ),
        config_sha256=config.digest,
        tool_versions=_pairing_tool_versions(context.pairing, "two-take-pairing-v1"),
    )
    automatic_pairing = context.run_directory / "pairing-automatic.json"
    if automatic_pairing.exists() or automatic_pairing.is_symlink():
        paired_ok = _human_pairing_transition_exists(
            rows,
            context.recording,
            context.pairing,
            context.pairing_sha256,
            config,
            paths,
            context.run_directory,
        )
    else:
        paired_ok = processing_event_exists(rows, paired)
    if not paired_ok:
        raise ValueError("pairing artifact is not bound to processing history")
    aligned = _expected_cached_transition(
        context.recording,
        CorpusState.PAIRED,
        CorpusState.ALIGNED,
        input_sha256s=(
            context.recording.sha256,
            context.pairing_sha256,
            context.alignment_sha256,
        ),
        config_sha256=config.digest,
        tool_versions=_pairing_tool_versions(context.pairing, "cached-word-alignment-v1"),
    )
    reviewed = _expected_cached_transition(
        context.recording,
        CorpusState.ALIGNED,
        CorpusState.REVIEWED,
        input_sha256s=(
            context.recording.sha256,
            context.alignment_sha256,
            review_sha256,
        ),
        config_sha256=config.digest,
        tool_versions=("human-review-v1",),
    )
    if not processing_event_exists(rows, aligned) or not processing_event_exists(rows, reviewed):
        raise ValueError("alignment or review artifact is not bound to processing history")


def _validate_existing_manifest(paths: CorpusPaths) -> tuple[SegmentRecord, ...]:
    manifest = paths.manifests / "segments.jsonl"
    if not manifest.exists():
        raise ValueError("terminal recording states require an existing segments.jsonl")
    rows = tuple(SegmentRecord.from_dict(row) for row in read_jsonl(manifest))
    identities = tuple(row.segment_id for row in rows)
    if len(identities) != len(set(identities)):
        raise ValueError("segments.jsonl contains duplicate segment_id")
    for row in rows:
        path = _require_canonical_descendant(
            paths.segments / "lossless",
            paths.resolve_local(row.derived_audio_relative_path),
            kind="manifest lossless clip",
            require_file=True,
        )
        if _digest(path) != row.derived_audio_sha256:
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "derived manifest clip is invalid")
    return rows


def _validate_terminal_history(
    paths: CorpusPaths,
    config: CorpusConfig,
    recordings: tuple[RecordingRecord, ...],
    segments: tuple[SegmentRecord, ...],
    ffmpeg_version: str,
) -> None:
    review_path = paths.manifests / "review.jsonl"
    review_events = _load_existing_review_events(review_path)
    review_by_id = {event.review_event_id: event for event in review_events}
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    processing = tuple(ProcessingEvent.from_dict(row) for row in read_jsonl(processing_path))
    ids = tuple(event.event_id for event in processing)
    semantics = tuple(
        (
            event.recording_id,
            event.previous_state,
            event.target_state,
            event.input_sha256s,
            event.config_sha256,
            event.tool_versions,
            event.result,
        )
        for event in processing
    )
    if len(ids) != len(set(ids)) or len(semantics) != len(set(semantics)):
        raise ValueError("terminal processing history contains duplicate identities")
    segment_ids = tuple(segment.segment_id for segment in segments)
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("terminal segment identities are duplicated")
    manifest_sha256 = _digest(paths.manifests / "segments.jsonl")
    review_sha256 = _digest(review_path)
    by_recording: dict[str, list[SegmentRecord]] = {}
    for segment in segments:
        by_recording.setdefault(segment.recording_id, []).append(segment)
    if set(by_recording) - {recording.recording_id for recording in recordings}:
        raise ValueError("terminal manifest contains an unknown recording_id")
    for recording in recordings:
        recording_events = tuple(
            event for event in processing if event.recording_id == recording.recording_id
        )
        if not recording_events or any(
            left.target_state is not right.previous_state
            for left, right in pairwise(recording_events)
        ):
            raise ValueError("terminal processing history contains a stale state chain")
        final = recording_events[-1]
        run_directory = paths.alignments / "runs" / config.digest / recording.recording_id
        expected_inputs = (
            recording.sha256,
            _digest(run_directory / "pairing.json"),
            _digest(run_directory / "alignment.json"),
            review_sha256,
            manifest_sha256,
        )
        if (
            final.previous_state is not CorpusState.REVIEWED
            or final.target_state is not recording.state
            or final.input_sha256s != expected_inputs
            or final.config_sha256 != config.digest
            or final.tool_versions != ("approved-manifest-v1", ffmpeg_version)
        ):
            raise ValueError("terminal recording is not bound to the approved manifest event")
        recording_segments = by_recording.get(recording.recording_id, [])
        if (recording.state is CorpusState.APPROVED) != bool(recording_segments):
            raise ValueError("terminal state does not match approved segment count")
        for segment in recording_segments:
            if (
                segment.config_sha256 != config.digest
                or segment.source_audio_sha256 != recording.sha256
                or segment.rights_id != recording.rights_id
            ):
                raise ValueError("terminal segment provenance does not match recording")
            try:
                consumed = tuple(review_by_id[event_id] for event_id in segment.review_event_ids)
            except KeyError as error:
                raise ValueError("terminal segment references a missing review event") from error
            if not any(
                event.field == "review_decision" and event.after == "approved" for event in consumed
            ):
                raise ValueError("terminal segment lacks an approved human decision")


def build_manifest_corpus(
    paths: CorpusPaths,
    config: CorpusConfig,
    *,
    ffmpeg_version: str,
    run_command: RunCommand = subprocess.run,
    probe_command: RunCommand = subprocess.run,
    decode_command: RunCommand = subprocess.run,
) -> bool:
    if type(paths) is not CorpusPaths or type(config) is not CorpusConfig:
        raise TypeError("build_manifest_corpus requires CorpusPaths and CorpusConfig")
    if config.raw.get("schema_version") != "1" or config.raw.get("corpus_version") != "corpus-v1":
        raise CorpusFailure("MANIFEST_SCHEMA_MISMATCH", "corpus configuration version is invalid")
    if not isinstance(ffmpeg_version, str) or not ffmpeg_version.strip():
        raise ValueError("build_manifest_corpus requires ffmpeg_version")
    from latintts.corpus.cli import _decode_spoken_units, _load_pairing_segmentation

    selection = _load_selection(paths)
    transcript_by_id = _load_transcript_layers(paths, selection)
    recording_rows = read_jsonl(paths.manifests / "recordings.jsonl")
    recordings = tuple(_decode_recording(row) for row in recording_rows)
    by_id = {
        recording.recording_id: (index, recording) for index, recording in enumerate(recordings)
    }
    if len(by_id) != len(recordings) or set(by_id) != set(selection.recording_ids):
        raise ValueError("recordings must uniquely and exactly match pilot selection")
    rights = _load_rights(paths)
    for recording_id, expected_hash in zip(
        selection.recording_ids, selection.inventory_hashes, strict=True
    ):
        item = by_id.get(recording_id)
        if item is None or item[1].sha256 != expected_hash:
            raise ValueError("pilot selection does not match recording inventory")
        _validate_source_hash(_raw_source(item[1], paths), item[1].sha256, derived=False)
        if item[1].rights_id not in rights:
            raise ValueError("recording rights_id does not exist")
    terminal = {CorpusState.APPROVED, CorpusState.REJECTED}
    if all(record.state in terminal for _, record in by_id.values()):
        existing_segments = _validate_existing_manifest(paths)
        _validate_terminal_history(paths, config, recordings, existing_segments, ffmpeg_version)
        return True
    if any(record.state is not CorpusState.REVIEWED for _, record in by_id.values()):
        raise CorpusFailure("REVIEW_REQUIRED", "manifest build requires REVIEWED recordings")
    if not import_review_bundle(
        paths,
        config,
        ffmpeg_version=ffmpeg_version,
        run_command=run_command,
    ):
        raise CorpusFailure("REVIEW_REQUIRED", "review import is incomplete")
    review_path = paths.manifests / "review.jsonl"
    review_events = _load_existing_review_events(review_path)
    review_sha256 = _digest(review_path)
    legal_pairing_event_ids = _validate_existing_pairing_corrections(
        paths, config, selection, recordings, review_events
    )
    contexts: list[_ManifestContext] = []
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    processing_rows = read_jsonl(processing_path)
    expected_review_entities: set[str] = set()
    for recording_id in selection.recording_ids:
        index, recording = by_id[recording_id]
        transcript = transcript_by_id[recording_id]
        _validate_transcript_sources(transcript, recording_id)
        units = _decode_spoken_units(transcript, recording_id)
        transcript_sha256 = hashlib.sha256(transcript["spoken_text"].encode("utf-8")).hexdigest()
        analysis, _, _, segmentation_sha256 = _load_pairing_segmentation(
            paths,
            config,
            recording,
            transcript_sha256,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
        )
        run_directory, pairing, alignment_artifact = _load_pairing_and_alignment(
            paths, config, recording
        )
        if segmentation_sha256 != pairing.segmentation_artifact_sha256:
            raise ValueError("pairing references a stale segmentation artifact")
        decoded = _alignment_by_take(pairing, alignment_artifact, paths)
        context = _ManifestContext(
            index,
            recording,
            transcript,
            units,
            pairing,
            decoded,
            analysis,
            _digest(run_directory / "pairing.json"),
            _digest(run_directory / "alignment.json"),
            run_directory,
        )
        _validate_processing_history(processing_rows, context, review_sha256, config, paths)
        contexts.append(context)
        expected_review_entities.update(
            _review_entity_id(recording_id, outcome.group.repetition_group_id, take.take_index)
            for outcome in pairing.groups
            if outcome.group is not None
            for take in outcome.group.takes
        )
    unknown = {
        event.entity_id
        for event in review_events
        if event.field != "pairing_selected_split"
        and event.entity_id not in expected_review_entities
    }
    illegal_pairing = {
        event.review_event_id
        for event in review_events
        if event.field == "pairing_selected_split"
        and event.review_event_id not in legal_pairing_event_ids
    }
    if unknown or illegal_pairing:
        raise ValueError("review history contains an unknown or illegal entity")
    for context in contexts:
        build_approved_segments(
            recording=context.recording,
            transcript=context.transcript,
            units=context.units,
            rights=rights,
            pairing=context.pairing,
            alignments={
                key: value["alignment_result"] for key, value in context.alignments.items()
            },
            review_events=review_events,
            analysis_audio=context.analysis,
            paths=paths,
            config=config,
            ffmpeg_version=ffmpeg_version,
            _validate_only=True,
        )
    all_rows: list[tuple[str, int, int, SegmentRecord]] = []
    decisions_by_recording: dict[str, tuple[int, int]] = {}
    for context in contexts:
        alignments = {key: value["alignment_result"] for key, value in context.alignments.items()}
        rows = build_approved_segments(
            recording=context.recording,
            transcript=context.transcript,
            units=context.units,
            rights=rights,
            pairing=context.pairing,
            alignments=alignments,
            review_events=review_events,
            analysis_audio=context.analysis,
            paths=paths,
            config=config,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
            probe_command=probe_command,
            decode_command=decode_command,
        )
        ordinal_by_id = {unit.unit_id: unit.ordinal for unit in context.units}
        all_rows.extend(
            (row.recording_id, ordinal_by_id[row.text_unit_id], row.take_index, row) for row in rows
        )
        total_takes = sum(
            len(outcome.group.takes)
            for outcome in context.pairing.groups
            if outcome.group is not None
        )
        decisions_by_recording[context.recording.recording_id] = (len(rows), total_takes)
    sorted_rows = tuple(item[3] for item in sorted(all_rows, key=lambda item: item[:3]))
    identities = tuple(row.segment_id for row in sorted_rows)
    if len(identities) != len(set(identities)):
        raise ValueError("approved segments contain duplicate segment_id")
    manifest_path = paths.manifests / "segments.jsonl"
    write_jsonl_atomic(manifest_path, (row.to_dict() for row in sorted_rows))
    manifest_sha256 = _digest(manifest_path)
    current = recordings
    for context in contexts:
        approved_count, total_takes = decisions_by_recording[context.recording.recording_id]
        target = CorpusState.APPROVED if approved_count else CorpusState.REJECTED
        if approved_count > total_takes:
            raise AssertionError("approved count exceeds reviewed take count")
        timestamp = "1970-01-01T00:00:00+00:00"
        advanced, event = advance_recording(
            current[context.recording_index],
            target,
            input_sha256s=(
                context.recording.sha256,
                context.pairing_sha256,
                context.alignment_sha256,
                review_sha256,
                manifest_sha256,
            ),
            config_sha256=config.digest,
            tool_versions=("approved-manifest-v1", ffmpeg_version),
            started_at=timestamp,
            finished_at=timestamp,
            result="success",
        )
        current = (
            *current[: context.recording_index],
            advanced,
            *current[context.recording_index + 1 :],
        )
        persist_recording_transition(
            recordings_path=paths.manifests / "recordings.jsonl",
            events_path=processing_path,
            recordings=current,
            event=event,
        )
    return True
