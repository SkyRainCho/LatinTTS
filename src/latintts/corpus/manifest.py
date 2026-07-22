from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from latintts.corpus.alignment import AlignmentResult, WordSpan
from latintts.corpus.audio import (
    DerivedAudio,
    RunCommand,
    _artifact_relative_path,
    _cache_identity,
    _canonical_output_config,
    _raw_source,
    _validate_source_hash,
    extract_analysis_segment,
    extract_lossless_segment,
    measure_pcm16,
)
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.locking import CorpusMutationLease, corpus_mutation_lease
from latintts.corpus.pairing import PairingRecording
from latintts.corpus.paths import CorpusPaths, require_canonical_descendant
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
    validate_review_bundle,
)
from latintts.corpus.store import (
    jsonl_sha256,
    persist_recording_transitions,
    processing_event_exists,
    read_jsonl,
    validate_processing_events,
    write_jsonl_atomic,
)
from latintts.corpus.transcripts import SpokenUnit
from latintts.normalization import tokenize_words

LosslessExtractor = Callable[..., DerivedAudio]
AnalysisExtractor = Callable[..., DerivedAudio]
BeforePublishLink = Callable[[Path], None]


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


def _expected_lossless_audio(
    recording: RecordingRecord,
    proposal: _ApprovedProposal,
    ffmpeg_version: str,
    sha256: str,
) -> DerivedAudio:
    output_config = _canonical_output_config(
        {
            "codec": "flac",
            "compression_level": 5,
            "channels": recording.metadata.channels,
            "sample_rate": recording.metadata.sample_rate,
        }
    )
    start = proposal.source_start / recording.metadata.sample_rate
    end = proposal.source_end / recording.metadata.sample_rate
    cache_key = _cache_identity(
        mode="lossless",
        source_relative_path=recording.relative_path,
        source_sha256=recording.sha256,
        config_sha256=output_config,
        output_config_sha256=output_config,
        ffmpeg_version=ffmpeg_version,
        start_seconds=start,
        end_seconds=end,
    )
    return DerivedAudio(
        "1",
        "lossless",
        _artifact_relative_path("lossless", cache_key),
        sha256,
        recording.relative_path,
        recording.sha256,
        output_config,
        output_config,
        ffmpeg_version,
        cache_key,
        start,
        end,
        None,
    )


def _expected_quality_audio(
    analysis: DerivedAudio,
    proposal: _ApprovedProposal,
    ffmpeg_version: str,
    sha256: str,
    metrics: Any,
) -> DerivedAudio:
    output_config = _canonical_output_config(
        {"codec": "pcm_s16le", "channels": 1, "sample_rate": 16000}
    )
    cache_key = _cache_identity(
        mode="candidate",
        source_relative_path=analysis.relative_path,
        source_sha256=analysis.sha256,
        config_sha256=analysis.config_sha256,
        output_config_sha256=output_config,
        ffmpeg_version=ffmpeg_version,
        start_seconds=proposal.absolute_start,
        end_seconds=proposal.absolute_end,
    )
    return DerivedAudio(
        "1",
        "candidate",
        _artifact_relative_path("candidate", cache_key),
        sha256,
        analysis.relative_path,
        analysis.sha256,
        analysis.config_sha256,
        output_config,
        ffmpeg_version,
        cache_key,
        proposal.absolute_start,
        proposal.absolute_end,
        metrics,
    )


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
    _staging_root: Path | None = None,
    _artifacts: list[DerivedAudio] | None = None,
    _existing_segments: Mapping[tuple[str, int], SegmentRecord] | None = None,
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
            effective_words = effective.get("words")
            expected_words = tuple(word.surface for word in tokenize_words(unit.text))
            reviewed_words = (
                tuple(word.get("text") if type(word) is dict else None for word in effective_words)
                if type(effective_words) is list
                else ()
            )
            if len(reviewed_words) != len(expected_words) or any(
                not isinstance(actual, str) or actual.casefold() != expected.casefold()
                for actual, expected in zip(reviewed_words, expected_words, strict=True)
            ):
                raise ValueError("reviewed word text is stale against transcript and pronunciation")
            if len(result.words) != len(expected_words) or any(
                word.text.casefold() != expected.casefold()
                for word, expected in zip(result.words, expected_words, strict=True)
            ):
                raise ValueError("alignment word text is stale against transcript")
            assert isinstance(effective_words, list)
            for reviewed_word, expected_word in zip(effective_words, expected_words, strict=True):
                assert type(reviewed_word) is dict
                reviewed_word["text"] = expected_word
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
        existing = (
            _existing_segments.get((proposal.repetition_group_id, proposal.take_index))
            if _existing_segments is not None
            else None
        )
        cached_lossless = (
            _expected_lossless_audio(
                recording, proposal, ffmpeg_version, existing.derived_audio_sha256
            )
            if existing is not None
            else None
        )
        cached_quality = (
            _expected_quality_audio(
                analysis_audio,
                proposal,
                ffmpeg_version,
                existing.quality_metric_audio_sha256,
                existing.quality_metrics,
            )
            if existing is not None
            else (None if _staging_root is not None else proposal.cached_quality)
        )
        if existing is not None:
            assert cached_lossless is not None and cached_quality is not None
            lossless_path = _require_canonical_descendant(
                paths.segments / "lossless",
                paths.resolve_local(cached_lossless.relative_path),
                kind="manifest lossless clip",
                require_file=True,
            )
            quality_path = _require_canonical_descendant(
                paths.segments / "candidates",
                paths.resolve_local(cached_quality.relative_path),
                kind="manifest quality clip",
                require_file=True,
            )
            if _digest(lossless_path) != cached_lossless.sha256 or (
                _digest(quality_path) != cached_quality.sha256
                or measure_pcm16(quality_path) != cached_quality.metrics
            ):
                raise CorpusFailure("CACHE_ARTIFACT_INVALID", "terminal audio cache is invalid")
            lossless = cached_lossless
            quality = cached_quality
        else:
            lossless = extract_lossless(
                recording,
                paths,
                proposal.source_start / recording.metadata.sample_rate,
                proposal.source_end / recording.metadata.sample_rate,
                ffmpeg_version=ffmpeg_version,
                cached=cached_lossless,
                run_command=run_command,
                probe_command=probe_command,
                decode_command=decode_command,
                _staging_root=_staging_root,
            )
            quality = extract_analysis(
                analysis_audio,
                paths,
                proposal.absolute_start,
                proposal.absolute_end,
                ffmpeg_version=ffmpeg_version,
                cached=cached_quality,
                run_command=run_command,
                _staging_root=_staging_root,
            )
        expected_lossless = _expected_lossless_audio(
            recording, proposal, ffmpeg_version, lossless.sha256
        )
        if lossless != expected_lossless:
            raise ValueError("lossless derived clip provenance does not match raw recording")
        expected_quality = _expected_quality_audio(
            analysis_audio, proposal, ffmpeg_version, quality.sha256, quality.metrics
        )
        if quality != expected_quality or quality.metrics is None:
            raise ValueError("quality metric clip must be a 16 kHz analysis segment")
        if _artifacts is not None:
            _artifacts.extend((lossless, quality))
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
    built = tuple(
        row
        for _, row in sorted(
            rows_with_order,
            key=lambda item: (recording.recording_id, item[0], item[1].take_index),
        )
    )
    if _existing_segments is not None:
        existing_rows = tuple(
            _existing_segments[(row.repetition_group_id, row.take_index)] for row in built
        )
        if len(_existing_segments) != len(built) or existing_rows != built:
            raise ValueError("existing manifest segments differ from deterministic expected rows")
    return built


def _windows_handle_identity(handle: object, *, kind: str) -> tuple[int, int, int]:
    import ctypes
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ByHandleFileInformation),
    ]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    information = ByHandleFileInformation()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
        raise ctypes.WinError(ctypes.get_last_error())
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if information.dwFileAttributes & reparse_flag:
        raise ValueError(f"{kind} handle is a reparse point")
    return (
        information.dwVolumeSerialNumber,
        information.nFileIndexHigh,
        information.nFileIndexLow,
    )


def _windows_open_handle(
    path: Path,
    *,
    desired_access: int,
    share_mode: int,
    flags: int,
    creation_disposition: int = 3,
) -> object:
    import ctypes
    from ctypes import wintypes

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    handle = kernel32.CreateFileW(
        str(path),
        desired_access,
        share_mode,
        None,
        creation_disposition,
        flags,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def _windows_close_handle(handle: object) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


@dataclass(slots=True)
class _PublicationGuard:
    root: Path
    handle: object
    identity: tuple[int, ...]
    windows: bool

    def validate(self) -> None:
        require_canonical_descendant(self.root, self.root, kind="publication mode root")
        identity: tuple[int, ...]
        if self.windows:
            current = _windows_open_handle(
                self.root,
                desired_access=0x80,
                share_mode=0x1 | 0x2,
                flags=0x02000000 | 0x00200000,
            )
            try:
                identity = _windows_handle_identity(current, kind="publication directory")
            finally:
                _windows_close_handle(current)
        else:  # pragma: no cover - exercised on POSIX hosts
            metadata = os.stat(self.root, follow_symlinks=False)
            identity = (metadata.st_dev, metadata.st_ino)
        if identity != self.identity:
            raise ValueError("publication directory identity changed")

    def close(self) -> None:
        if self.windows:
            _windows_close_handle(self.handle)
        else:  # pragma: no cover - exercised on POSIX hosts
            os.close(self.handle)  # type: ignore[arg-type]


def _windows_digest_handle(handle: object) -> str:
    import ctypes
    from ctypes import wintypes

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    kernel32.SetFilePointerEx.restype = wintypes.BOOL
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    if not kernel32.SetFilePointerEx(handle, 0, None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    digest = hashlib.sha256()
    buffer = ctypes.create_string_buffer(1024 * 1024)
    read = wintypes.DWORD()
    try:
        while True:
            if not kernel32.ReadFile(handle, buffer, len(buffer), ctypes.byref(read), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if read.value == 0:
                return digest.hexdigest()
            digest.update(buffer.raw[: read.value])
    finally:
        if not kernel32.SetFilePointerEx(handle, 0, None, 0):
            raise ctypes.WinError(ctypes.get_last_error())


def _posix_digest_handle(descriptor: int) -> str:  # pragma: no cover - exercised on POSIX hosts
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)


@dataclass(slots=True)
class _ArtifactGuard:
    path: Path
    handle: object
    identity: tuple[int, ...]
    sha256: str
    directory: _PublicationGuard
    windows: bool

    def validate(self) -> None:
        self.directory.validate()
        require_canonical_descendant(
            self.directory.root,
            self.path,
            kind="published audio artifact",
            require_file=True,
        )
        identity: tuple[int, ...]
        if self.windows:
            current = _windows_open_handle(
                self.path,
                desired_access=0x80,
                share_mode=0x1,
                flags=0x00200000,
            )
            try:
                identity = _windows_handle_identity(current, kind="published artifact")
            finally:
                _windows_close_handle(current)
            actual_sha256 = _windows_digest_handle(self.handle)
        else:  # pragma: no cover - exercised on POSIX hosts
            descriptor = cast(int, self.handle)
            pinned = os.fstat(descriptor)
            current = os.stat(
                self.path.name,
                dir_fd=cast(int, self.directory.handle),
                follow_symlinks=False,
            )
            if not stat.S_ISREG(pinned.st_mode) or not stat.S_ISREG(current.st_mode):
                raise ValueError("published artifact handle must identify a regular file")
            identity = (current.st_dev, current.st_ino)
            actual_sha256 = _posix_digest_handle(descriptor)
        if identity != self.identity:
            raise ValueError("published artifact identity changed")
        if actual_sha256 != self.sha256:
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "published audio artifact is invalid")

    def close(self) -> None:
        if self.windows:
            _windows_close_handle(self.handle)
        else:  # pragma: no cover - exercised on POSIX hosts
            os.close(cast(int, self.handle))


@dataclass(slots=True)
class _FileSnapshot:
    path: Path
    handle: object
    identity: tuple[int, ...]
    sha256: str
    windows: bool
    directory: _PublicationGuard | None = None
    closed: bool = False

    def validate(self) -> None:
        if self.closed:
            raise ValueError("file snapshot is already closed")
        if self.directory is not None:
            self.directory.validate()
        require_canonical_descendant(
            self.path.parent,
            self.path,
            kind="transaction snapshot file",
            require_file=True,
        )
        identity: tuple[int, ...]
        if self.windows:
            current = _windows_open_handle(
                self.path,
                desired_access=0x80000000,
                share_mode=0x1,
                flags=0x00200000,
            )
            try:
                identity = _windows_handle_identity(current, kind="transaction snapshot file")
            finally:
                _windows_close_handle(current)
            digest = _windows_digest_handle(self.handle)
        else:  # pragma: no cover - exercised on POSIX hosts
            descriptor = cast(int, self.handle)
            pinned = os.fstat(descriptor)
            current_metadata = os.stat(self.path, follow_symlinks=False)
            if not stat.S_ISREG(pinned.st_mode) or not stat.S_ISREG(current_metadata.st_mode):
                raise ValueError("transaction snapshot must identify a regular file")
            identity = (current_metadata.st_dev, current_metadata.st_ino)
            digest = _posix_digest_handle(descriptor)
        if identity != self.identity or digest != self.sha256:
            raise ValueError(f"transaction input snapshot changed: {self.path}")

    def close(self) -> None:
        if self.closed:
            return
        if self.windows:
            _windows_close_handle(self.handle)
        else:  # pragma: no cover - exercised on POSIX hosts
            os.close(cast(int, self.handle))
        self.closed = True


def _open_file_snapshot(
    path: Path,
    *,
    directory: _PublicationGuard | None = None,
    expected_sha256: str | None = None,
) -> _FileSnapshot:
    path = Path(os.path.abspath(path))
    require_canonical_descendant(
        path.parent,
        path,
        kind="transaction snapshot file",
        require_file=True,
    )
    if directory is not None:
        directory.validate()
    if os.name == "nt":
        handle = _windows_open_handle(
            path,
            desired_access=0x80000000,
            share_mode=0x1,
            flags=0x00200000,
        )
        try:
            identity = _windows_handle_identity(handle, kind="transaction snapshot file")
            digest = _windows_digest_handle(handle)
            if expected_sha256 is not None and digest != expected_sha256:
                raise ValueError("durable file bytes differ from expected transaction output")
        except Exception:
            _windows_close_handle(handle)
            raise
        return _FileSnapshot(path, handle, identity, digest, True, directory)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)  # pragma: no cover
    if directory is not None and path.parent == directory.root:  # pragma: no cover
        descriptor = os.open(path.name, flags, dir_fd=cast(int, directory.handle))
    else:  # pragma: no cover - exercised on POSIX hosts
        descriptor = os.open(path, flags)
    try:  # pragma: no cover - exercised on POSIX hosts
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("transaction snapshot handle is not a regular file")
        digest = _posix_digest_handle(descriptor)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("durable file bytes differ from expected transaction output")
    except Exception:  # pragma: no cover - exercised on POSIX hosts
        os.close(descriptor)
        raise
    return _FileSnapshot(  # pragma: no cover - exercised on POSIX hosts
        path,
        descriptor,
        (metadata.st_dev, metadata.st_ino),
        digest,
        False,
        directory,
    )


def _close_file_snapshots(guards: Mapping[str, _FileSnapshot]) -> None:
    failure: OSError | None = None
    for guard in reversed(tuple(guards.values())):
        try:
            guard.close()
        except OSError as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def _validate_file_snapshots(guards: Mapping[str, _FileSnapshot]) -> None:
    for guard in guards.values():
        guard.validate()


def _add_file_snapshot(
    guards: dict[str, _FileSnapshot],
    path: Path,
    *,
    directory: _PublicationGuard | None = None,
    expected_sha256: str | None = None,
) -> _FileSnapshot:
    key = str(Path(os.path.abspath(path)))
    existing = guards.get(key)
    if existing is not None:
        existing.validate()
        if expected_sha256 is not None and existing.sha256 != expected_sha256:
            raise ValueError("transaction snapshot differs from expected bytes")
        return existing
    guard = _open_file_snapshot(
        path,
        directory=directory,
        expected_sha256=expected_sha256,
    )
    guards[key] = guard
    return guard


def _release_file_snapshot(guards: dict[str, _FileSnapshot], path: Path) -> None:
    key = str(Path(os.path.abspath(path)))
    guard = guards[key]
    guard.validate()
    guard.close()
    del guards[key]


def _open_artifact_guard(
    path: Path,
    sha256: str,
    directory: _PublicationGuard,
) -> _ArtifactGuard:
    if os.name == "nt":
        handle = _windows_open_handle(
            path,
            desired_access=0x80000000,  # GENERIC_READ
            share_mode=0x1,  # deliberately deny write and delete sharing
            flags=0x00200000,
        )
        try:
            identity = _windows_handle_identity(handle, kind="published artifact")
            if _windows_digest_handle(handle) != sha256:
                raise CorpusFailure("CACHE_ARTIFACT_INVALID", "published audio artifact is invalid")
        except Exception:
            _windows_close_handle(handle)
            raise
        return _ArtifactGuard(path, handle, identity, sha256, directory, True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)  # pragma: no cover
    descriptor = os.open(  # pragma: no cover - exercised on POSIX hosts
        path.name,
        flags,
        dir_fd=cast(int, directory.handle),
    )
    try:  # pragma: no cover - exercised on POSIX hosts
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("published artifact handle is not a regular file")
        if _posix_digest_handle(descriptor) != sha256:
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "published audio artifact is invalid")
    except Exception:  # pragma: no cover - exercised on POSIX hosts
        os.close(descriptor)
        raise
    return _ArtifactGuard(  # pragma: no cover - exercised on POSIX hosts
        path,
        descriptor,
        (metadata.st_dev, metadata.st_ino),
        sha256,
        directory,
        False,
    )


def _open_directory_guard(root: Path, *, writable: bool) -> _PublicationGuard:
    if os.name == "nt":
        handle = _windows_open_handle(
            root,
            desired_access=0x80 | (0x2 if writable else 0),
            share_mode=0x1 | 0x2,  # deliberately deny FILE_SHARE_DELETE
            flags=0x02000000 | 0x00200000,
        )
        try:
            identity = _windows_handle_identity(handle, kind="publication directory")
        except Exception:
            _windows_close_handle(handle)
            raise
        return _PublicationGuard(root, handle, identity, True)
    else:  # pragma: no cover - exercised on POSIX hosts
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(root, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(descriptor)
            raise ValueError("publication root handle is not a directory")
        return _PublicationGuard(root, descriptor, (metadata.st_dev, metadata.st_ino), False)


def _open_publication_guard(root: Path) -> _PublicationGuard:
    return _open_directory_guard(root, writable=True)


@dataclass(slots=True)
class _OutputNamespaceGuard:
    directories: dict[Path, _PublicationGuard]
    manifests: _PublicationGuard
    processing: _PublicationGuard
    mutation_lease: CorpusMutationLease
    windows: bool

    def validate(self) -> None:
        for guard in self.directories.values():
            guard.validate()

    def directory_fd(self, guard: _PublicationGuard) -> int | None:
        return None if self.windows else cast(int, guard.handle)

    def close(self) -> None:
        failure: OSError | None = None
        try:
            self.mutation_lease.close()
        except OSError as error:
            failure = error
        for guard in reversed(tuple(self.directories.values())):
            try:
                guard.close()
            except OSError as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure


def _directory_chain(anchor: Path, target: Path) -> tuple[Path, ...]:
    anchor = Path(os.path.abspath(anchor))
    target = require_canonical_descendant(anchor, target, kind="durable output directory")
    relative = target.relative_to(anchor)
    chain = [anchor]
    current = anchor
    for part in relative.parts:
        current /= part
        chain.append(current)
    return tuple(chain)


def _open_output_namespace_guard(
    paths: CorpusPaths,
    processing_path: Path,
    publication_guards: dict[str, _PublicationGuard],
) -> _OutputNamespaceGuard:
    project_root = Path(os.path.abspath(paths.project_root))
    manifests = Path(os.path.abspath(paths.manifests))
    processing = Path(os.path.abspath(processing_path.parent))
    segments = Path(os.path.abspath(paths.segments))
    targets = [manifests, processing, segments]
    targets.extend(guard.root.parent for guard in publication_guards.values())
    ordered_paths = sorted(
        {directory for target in targets for directory in _directory_chain(project_root, target)},
        key=lambda path: (len(path.parts), str(path).casefold()),
    )
    directories: dict[Path, _PublicationGuard] = {}
    mutation_lease: CorpusMutationLease | None = None
    try:
        for directory in ordered_paths:
            directories[directory] = _open_directory_guard(
                directory,
                writable=directory in {manifests, processing},
            )
        manifest_guard = directories[manifests]
        processing_guard = directories[processing]
        mutation_lease = corpus_mutation_lease(
            paths.manifests / "segments.jsonl",
            processing_path,
        )
        guard = _OutputNamespaceGuard(
            directories,
            manifest_guard,
            processing_guard,
            mutation_lease,
            os.name == "nt",
        )
        guard.validate()
        return guard
    except Exception:
        if mutation_lease is not None:
            with suppress(OSError):
                mutation_lease.close()
        for directory_guard in reversed(tuple(directories.values())):
            with suppress(OSError):
                directory_guard.close()
        raise


def _digest_in_guarded_directory(
    path: Path,
    namespace: _OutputNamespaceGuard,
    directory: _PublicationGuard,
) -> str:
    directory_fd = namespace.directory_fd(directory)
    if directory_fd is None:
        return _digest(path)
    descriptor = os.open(  # pragma: no cover - exercised on POSIX hosts
        path.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:  # pragma: no cover - exercised on POSIX hosts
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("durable manifest input must be a regular file")
        return _posix_digest_handle(descriptor)
    finally:  # pragma: no cover - exercised on POSIX hosts
        os.close(descriptor)


def _windows_link_from_pinned_directory(
    staged: Path,
    final: Path,
    guard: _PublicationGuard,
    expected_sha256: str | None = None,
) -> _ArtifactGuard:
    import ctypes
    from ctypes import wintypes

    source_handle = _windows_open_handle(
        staged,
        desired_access=0x00010000 | 0x80000000,  # DELETE | GENERIC_READ
        share_mode=0x1 | 0x4,  # allow link creation while denying writers
        flags=0x00200000,
    )
    try:
        source_identity = _windows_handle_identity(source_handle, kind="staged artifact")
        source_sha256 = _windows_digest_handle(source_handle)
        if expected_sha256 is not None and source_sha256 != expected_sha256:
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "staged audio artifact is invalid")

        class FileLinkInformation(ctypes.Structure):
            _fields_ = [
                ("ReplaceIfExists", wintypes.BOOLEAN),
                ("RootDirectory", wintypes.HANDLE),
                ("FileNameLength", wintypes.DWORD),
                ("FileName", wintypes.WCHAR * 1),
            ]

        filename = final.name.encode("utf-16-le")
        filename_offset = FileLinkInformation.FileName.offset
        buffer = ctypes.create_string_buffer(filename_offset + len(filename))
        information = FileLinkInformation.from_buffer(buffer)
        information.ReplaceIfExists = False
        information.RootDirectory = guard.handle
        information.FileNameLength = len(filename)
        ctypes.memmove(ctypes.addressof(buffer) + filename_offset, filename, len(filename))

        class IoStatusBlock(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

        io_status = IoStatusBlock()
        ntdll: Any = ctypes.WinDLL("ntdll")
        ntdll.NtSetInformationFile.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(IoStatusBlock),
            wintypes.LPVOID,
            wintypes.ULONG,
            ctypes.c_int,
        ]
        ntdll.NtSetInformationFile.restype = ctypes.c_long
        status = ntdll.NtSetInformationFile(
            source_handle,
            ctypes.byref(io_status),
            buffer,
            len(buffer),
            11,  # FileLinkInformation
        )
        if status < 0 and status & 0xFFFFFFFF != 0xC0000035:  # STATUS_OBJECT_NAME_COLLISION
            ntdll.RtlNtStatusToDosError.argtypes = [ctypes.c_long]
            ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG
            failure = ctypes.WinError(ntdll.RtlNtStatusToDosError(status))
            raise ValueError("publication hard-link creation failed closed") from failure
        target_handle = _windows_open_handle(
            final,
            desired_access=0x80,
            share_mode=0x1 | 0x4,
            flags=0x00200000,
        )
        try:
            target_identity = _windows_handle_identity(target_handle, kind="published artifact")
        finally:
            _windows_close_handle(target_handle)
        if target_identity != source_identity:
            raise ValueError("published artifact identity differs from staged source")
    finally:
        _windows_close_handle(source_handle)
    return _open_artifact_guard(final, source_sha256, guard)


def _link_from_pinned_directory(
    staged: Path,
    final: Path,
    guard: _PublicationGuard,
    before_link: BeforePublishLink | None,
    expected_sha256: str,
) -> _ArtifactGuard:
    if before_link is not None:
        try:
            before_link(guard.root)
        except OSError as error:
            raise ValueError("publication directory changed during hard-link creation") from error
    guard.validate()
    if os.name == "nt":
        artifact_guard = _windows_link_from_pinned_directory(
            staged,
            final,
            guard,
            expected_sha256,
        )
    else:  # pragma: no cover - exercised on POSIX hosts
        with suppress(FileExistsError):
            os.link(
                staged,
                final.name,
                dst_dir_fd=guard.handle,  # type: ignore[arg-type]
                follow_symlinks=False,
            )
        source = os.stat(staged, follow_symlinks=False)
        target = os.stat(final, follow_symlinks=False)
        if (source.st_dev, source.st_ino) != (target.st_dev, target.st_ino):
            raise ValueError("published artifact identity differs from staged source")
        artifact_guard = _open_artifact_guard(final, expected_sha256, guard)
    guard.validate()
    return artifact_guard


def _publication_guards(
    paths: CorpusPaths,
    artifacts: tuple[DerivedAudio, ...],
) -> dict[str, _PublicationGuard]:
    local_root = Path(os.path.abspath(paths.local_data))
    guards: dict[str, _PublicationGuard] = {}
    try:
        for mode in sorted({artifact.mode for artifact in artifacts}):
            if mode not in {"candidate", "lossless"}:
                raise ValueError("manifest staging contains an unsupported audio mode")
            mode_root = require_canonical_descendant(
                local_root,
                Path(
                    os.path.abspath(
                        paths.segments / {"candidate": "candidates", "lossless": "lossless"}[mode]
                    )
                ),
                kind=f"{mode} mode root",
            )
            mode_root.mkdir(parents=True, exist_ok=True)
            require_canonical_descendant(local_root, mode_root, kind=f"{mode} mode root")
            guards[mode] = _open_publication_guard(mode_root)
        return guards
    except Exception:
        with suppress(OSError):
            _close_publication_guards(guards)
        raise


def _close_publication_guards(guards: dict[str, _PublicationGuard]) -> None:
    failure: OSError | None = None
    for guard in reversed(tuple(guards.values())):
        try:
            guard.close()
        except OSError as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def _validate_publication_guards(guards: dict[str, _PublicationGuard]) -> None:
    for guard in guards.values():
        guard.validate()


def _close_artifact_guards(guards: dict[str, _ArtifactGuard]) -> None:
    failure: OSError | None = None
    for guard in reversed(tuple(guards.values())):
        try:
            guard.close()
        except OSError as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def _validate_artifact_guards(guards: dict[str, _ArtifactGuard]) -> None:
    for guard in guards.values():
        guard.validate()


def _pin_existing_artifacts(
    paths: CorpusPaths,
    artifacts: tuple[DerivedAudio, ...],
    publication_guards: dict[str, _PublicationGuard],
) -> dict[str, _ArtifactGuard]:
    local_root = Path(os.path.abspath(paths.local_data))
    guards: dict[str, _ArtifactGuard] = {}
    try:
        for artifact in artifacts:
            directory = publication_guards[artifact.mode]
            final = require_canonical_descendant(
                directory.root,
                local_root / Path(*artifact.relative_path.split("/")),
                kind="final audio artifact",
                require_file=True,
            )
            key = str(final)
            if key in guards:
                if guards[key].sha256 != artifact.sha256:
                    raise CorpusFailure(
                        "CACHE_ARTIFACT_INVALID", "duplicate final audio artifact conflicts"
                    )
                guards[key].validate()
                continue
            guards[key] = _open_artifact_guard(final, artifact.sha256, directory)
        return guards
    except Exception:
        with suppress(OSError):
            _close_artifact_guards(guards)
        raise


def _publish_staged_audio(
    paths: CorpusPaths,
    staging_root: Path,
    artifacts: tuple[DerivedAudio, ...],
    guards: dict[str, _PublicationGuard],
    *,
    before_link: BeforePublishLink | None = None,
) -> dict[str, _ArtifactGuard]:
    local_root = Path(os.path.abspath(paths.local_data))
    segments_root = require_canonical_descendant(
        local_root,
        Path(os.path.abspath(paths.segments)),
        kind="fixed segments root",
    )
    staging_root = require_canonical_descendant(
        segments_root,
        Path(os.path.abspath(staging_root)),
        kind="audio staging root",
    )
    pending_hook = before_link
    artifact_guards: dict[str, _ArtifactGuard] = {}
    try:
        for artifact in artifacts:
            if artifact.mode not in {"candidate", "lossless"}:
                raise ValueError("manifest staging contains an unsupported audio mode")
            guard = guards[artifact.mode]
            mode_root = guard.root
            staged = require_canonical_descendant(
                staging_root,
                staging_root / Path(*artifact.relative_path.split("/")),
                kind="staged audio artifact",
                require_file=True,
            )
            final = require_canonical_descendant(
                mode_root,
                local_root / Path(*artifact.relative_path.split("/")),
                kind="final audio artifact",
            )
            if _digest(staged) != artifact.sha256:
                raise CorpusFailure("CACHE_ARTIFACT_INVALID", "staged audio artifact is invalid")
            key = str(final)
            if key in artifact_guards:
                if artifact_guards[key].sha256 != artifact.sha256:
                    raise CorpusFailure(
                        "CACHE_ARTIFACT_INVALID", "duplicate final audio artifact conflicts"
                    )
                artifact_guards[key].validate()
                continue
            if final.exists():
                require_canonical_descendant(
                    mode_root,
                    final,
                    kind="final audio artifact",
                    require_file=True,
                )
                artifact_guards[key] = _open_artifact_guard(final, artifact.sha256, guard)
                continue
            require_canonical_descendant(
                staging_root,
                staged,
                kind="staged audio artifact",
                require_file=True,
            )
            require_canonical_descendant(
                mode_root,
                final,
                kind="final audio artifact",
            )
            artifact_guards[key] = _link_from_pinned_directory(
                staged,
                final,
                guard,
                pending_hook,
                artifact.sha256,
            )
            pending_hook = None
            artifact_guards[key].validate()
        return artifact_guards
    except Exception:
        with suppress(OSError):
            _close_artifact_guards(artifact_guards)
        raise


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
    review_path: Path,
    review_events: tuple[ReviewEvent, ...],
    config: CorpusConfig,
    paths: CorpusPaths,
) -> tuple[ReviewEvent, ...]:
    from latintts.corpus.cli import (
        _expected_cached_transition,
        _human_pairing_transition_exists,
        _pairing_tool_versions,
    )

    events = validate_processing_events(rows)
    recording_events = tuple(
        event for event in events if event.recording_id == context.recording.recording_id
    )
    if not recording_events:
        raise ValueError("processing history contains a stale state chain")
    final_state = recording_events[-1]
    recoverable_audit_ahead = (
        context.recording.state is CorpusState.REVIEWED
        and final_state.previous_state is CorpusState.REVIEWED
        and final_state.target_state in {CorpusState.APPROVED, CorpusState.REJECTED}
    )
    if final_state.target_state is not context.recording.state and not recoverable_audit_ahead:
        raise ValueError("recording state is not bound to processing history")
    # Context creation has already required exactly one segmentation row and
    # validated its input/config/model provenance against the pinned artifact.
    segmentation_row = read_jsonl(context.run_directory / "segmentation.json")[0]
    segmentation_vad = cast(dict[str, Any], segmentation_row["vad"])
    segmentation_model_sha256 = cast(str, segmentation_vad["model_sha256"])
    transcript_sha256 = hashlib.sha256(
        context.transcript["spoken_text"].encode("utf-8")
    ).hexdigest()
    segmented = _expected_cached_transition(
        context.recording,
        CorpusState.TRANSCRIPT_CONFIRMED,
        CorpusState.SEGMENTED,
        input_sha256s=(
            context.recording.sha256,
            context.analysis.sha256,
            transcript_sha256,
            segmentation_model_sha256,
        ),
        config_sha256=config.digest,
        tool_versions=("silero-vad==6.2.1", "pause-profile-v1"),
    )
    try:
        segmented_exists = processing_event_exists(rows, segmented)
    except ValueError as error:
        raise ValueError("SEGMENTED artifact conflicts with processing history") from error
    if not segmented_exists:
        raise ValueError("SEGMENTED artifact is not bound to processing history")
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
    reviewed_candidates = tuple(
        event
        for event in recording_events
        if event.previous_state is CorpusState.ALIGNED
        and event.target_state is CorpusState.REVIEWED
        and len(event.input_sha256s) == 3
        and event.input_sha256s[:2] == (context.recording.sha256, context.alignment_sha256)
        and event.config_sha256 == config.digest
        and event.tool_versions == ("human-review-v1",)
        and event.result == "success"
    )
    if len(reviewed_candidates) != 1:
        raise ValueError("processing history lacks one bound human review transition")
    review_prefix_sha256 = reviewed_candidates[0].input_sha256s[2]
    prefix_events = _review_event_prefix(review_path, review_events, review_prefix_sha256)
    reviewed = _expected_cached_transition(
        context.recording,
        CorpusState.ALIGNED,
        CorpusState.REVIEWED,
        input_sha256s=(
            context.recording.sha256,
            context.alignment_sha256,
            review_prefix_sha256,
        ),
        config_sha256=config.digest,
        tool_versions=("human-review-v1",),
    )
    if not processing_event_exists(rows, aligned) or not processing_event_exists(rows, reviewed):
        raise ValueError("alignment or review artifact is not bound to processing history")
    return prefix_events


def _review_event_prefix(
    path: Path,
    events: tuple[ReviewEvent, ...],
    expected_sha256: str,
) -> tuple[ReviewEvent, ...]:
    data = path.read_bytes()
    consumed = bytearray()
    for count, line in enumerate(data.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            raise ValueError("review history is not strict newline-terminated JSONL")
        consumed.extend(line)
        if hashlib.sha256(consumed).hexdigest() == expected_sha256:
            return events[:count]
    raise ValueError("review transition digest is not an exact current JSONL prefix")


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


def build_manifest_corpus(
    paths: CorpusPaths,
    config: CorpusConfig,
    *,
    ffmpeg_version: str,
    run_command: RunCommand = subprocess.run,
    probe_command: RunCommand = subprocess.run,
    decode_command: RunCommand = subprocess.run,
    _before_publish_link: BeforePublishLink | None = None,
) -> bool:
    if type(paths) is not CorpusPaths or type(config) is not CorpusConfig:
        raise TypeError("build_manifest_corpus requires CorpusPaths and CorpusConfig")
    if config.raw.get("schema_version") != "1" or config.raw.get("corpus_version") != "corpus-v1":
        raise CorpusFailure("MANIFEST_SCHEMA_MISMATCH", "corpus configuration version is invalid")
    if not isinstance(ffmpeg_version, str) or not ffmpeg_version.strip():
        raise ValueError("build_manifest_corpus requires ffmpeg_version")
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    namespace_guard = _open_output_namespace_guard(paths, processing_path, {})
    input_snapshots: dict[str, _FileSnapshot] = {}
    durable_snapshots: dict[str, _FileSnapshot] = {}
    try:
        return _build_manifest_corpus_locked(
            paths,
            config,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
            probe_command=probe_command,
            decode_command=decode_command,
            _before_publish_link=_before_publish_link,
            _namespace_guard=namespace_guard,
            _input_snapshots=input_snapshots,
            _durable_snapshots=durable_snapshots,
        )
    finally:
        active_failure = sys.exc_info()[0] is not None
        cleanup_failure: OSError | None = None
        cleanup_functions: tuple[Callable[[], None], ...] = (
            lambda: _close_file_snapshots(durable_snapshots),
            lambda: _close_file_snapshots(input_snapshots),
            namespace_guard.close,
        )
        for cleanup in cleanup_functions:
            try:
                cleanup()
            except OSError as error:
                if cleanup_failure is None:
                    cleanup_failure = error
        if cleanup_failure is not None and not active_failure:
            raise cleanup_failure


def _build_manifest_corpus_locked(
    paths: CorpusPaths,
    config: CorpusConfig,
    *,
    ffmpeg_version: str,
    run_command: RunCommand,
    probe_command: RunCommand,
    decode_command: RunCommand,
    _before_publish_link: BeforePublishLink | None,
    _namespace_guard: _OutputNamespaceGuard,
    _input_snapshots: dict[str, _FileSnapshot],
    _durable_snapshots: dict[str, _FileSnapshot],
) -> bool:
    from latintts.corpus.cli import _decode_spoken_units, _load_pairing_segmentation

    selection = _load_selection(paths)
    recording_rows = read_jsonl(paths.manifests / "recordings.jsonl")
    recordings = tuple(_decode_recording(row) for row in recording_rows)
    by_id = {
        recording.recording_id: (index, recording) for index, recording in enumerate(recordings)
    }
    selection_ids = set(selection.recording_ids)
    if len(by_id) != len(recordings) or not selection_ids <= set(by_id):
        raise ValueError("recordings must be unique and contain every pilot selection")
    for recording_id, expected_hash in zip(
        selection.recording_ids, selection.inventory_hashes, strict=True
    ):
        preflight_recording = by_id[recording_id][1]
        if preflight_recording.sha256 != expected_hash:
            raise ValueError("pilot selection does not match recording inventory")
        _validate_source_hash(
            _raw_source(preflight_recording, paths),
            preflight_recording.sha256,
            derived=False,
        )
    terminal = {CorpusState.APPROVED, CorpusState.REJECTED}
    pilot_records = tuple(by_id[recording_id][1] for recording_id in selection.recording_ids)
    if any(record.state not in {CorpusState.REVIEWED, *terminal} for record in pilot_records):
        raise CorpusFailure("REVIEW_REQUIRED", "manifest build requires REVIEWED recordings")
    all_reviewed = all(record.state is CorpusState.REVIEWED for record in pilot_records)
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    processing_rows = read_jsonl(processing_path)
    decoded_processing = validate_processing_events(processing_rows)
    audit_ahead = any(
        event.recording_id in selection_ids
        and event.previous_state is CorpusState.REVIEWED
        and event.target_state in terminal
        for event in decoded_processing
    )
    recovery_only = audit_ahead or not all_reviewed
    if audit_ahead:
        if not validate_review_bundle(
            paths,
            config,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
        ):
            raise CorpusFailure("REVIEW_REQUIRED", "durable review replay is incomplete")
    elif all_reviewed:
        if not import_review_bundle(
            paths,
            config,
            ffmpeg_version=ffmpeg_version,
            run_command=run_command,
        ):
            raise CorpusFailure("REVIEW_REQUIRED", "review import is incomplete")
    elif not validate_review_bundle(
        paths,
        config,
        ffmpeg_version=ffmpeg_version,
        run_command=run_command,
    ):
        raise CorpusFailure("REVIEW_REQUIRED", "terminal review replay is incomplete")

    # Phase A above may write review/state artifacts. Discard every Phase A decode and
    # establish one authoritative, handle-pinned snapshot before consuming any evidence.
    selection_path = paths.manifests / "pilot-selection.json"
    recordings_path = paths.manifests / "recordings.jsonl"
    rights_path = paths.manifests / "rights.jsonl"
    transcripts_path = paths.manifests / "transcripts.jsonl"
    review_path = paths.manifests / "review.jsonl"
    _add_file_snapshot(
        _input_snapshots,
        selection_path,
        directory=_namespace_guard.manifests,
    )
    recordings_snapshot = _add_file_snapshot(
        _input_snapshots,
        recordings_path,
        directory=_namespace_guard.manifests,
    )
    rights_snapshot = _add_file_snapshot(
        _input_snapshots,
        rights_path,
        directory=_namespace_guard.manifests,
    )
    transcripts_snapshot = _add_file_snapshot(
        _input_snapshots,
        transcripts_path,
        directory=_namespace_guard.manifests,
    )
    review_snapshot = _add_file_snapshot(
        _input_snapshots,
        review_path,
        directory=_namespace_guard.manifests,
    )
    processing_snapshot = _add_file_snapshot(
        _input_snapshots,
        processing_path,
        directory=_namespace_guard.processing,
    )

    selection = _load_selection(paths)
    transcript_by_id = _load_transcript_layers(paths, selection)
    recording_rows = read_jsonl(recordings_path)
    recordings = tuple(_decode_recording(row) for row in recording_rows)
    by_id = {
        recording.recording_id: (index, recording) for index, recording in enumerate(recordings)
    }
    selection_ids = set(selection.recording_ids)
    if len(by_id) != len(recordings) or not selection_ids <= set(by_id):
        raise ValueError("recordings must be unique and contain every pilot selection")
    rights = _load_rights(paths)
    processing_rows = read_jsonl(processing_path)
    decoded_processing = validate_processing_events(processing_rows)
    audit_ahead = any(
        event.recording_id in selection_ids
        and event.previous_state is CorpusState.REVIEWED
        and event.target_state in terminal
        for event in decoded_processing
    )
    pilot_records = tuple(by_id[recording_id][1] for recording_id in selection.recording_ids)
    all_reviewed = all(record.state is CorpusState.REVIEWED for record in pilot_records)
    recovery_only = audit_ahead or not all_reviewed
    if any(record.state not in {CorpusState.REVIEWED, *terminal} for record in pilot_records):
        raise CorpusFailure("REVIEW_REQUIRED", "manifest build requires REVIEWED recordings")
    for recording_id, expected_hash in zip(
        selection.recording_ids, selection.inventory_hashes, strict=True
    ):
        item = by_id.get(recording_id)
        if item is None or item[1].sha256 != expected_hash:
            raise ValueError("pilot selection does not match recording inventory")
        raw_path = _raw_source(item[1], paths)
        _add_file_snapshot(_input_snapshots, raw_path)
        _validate_source_hash(raw_path, item[1].sha256, derived=False)
        if item[1].rights_id not in rights:
            raise ValueError("recording rights_id does not exist")
        run_directory = paths.alignments / "runs" / config.digest / recording_id
        for name in ("segmentation.json", "pairing.json", "alignment.json"):
            _add_file_snapshot(_input_snapshots, run_directory / name)
        automatic_pairing = run_directory / "pairing-automatic.json"
        if automatic_pairing.exists() or automatic_pairing.is_symlink():
            _add_file_snapshot(_input_snapshots, automatic_pairing)
    _validate_file_snapshots(_input_snapshots)

    review_events = _load_existing_review_events(review_path)
    review_sha256 = review_snapshot.sha256
    legal_pairing_event_ids = _validate_existing_pairing_corrections(
        paths, config, selection, recordings, review_events
    )
    _validate_file_snapshots(_input_snapshots)
    contexts: list[_ManifestContext] = []
    review_prefixes: dict[str, tuple[ReviewEvent, ...]] = {}
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
        segmentation_snapshot = _add_file_snapshot(
            _input_snapshots,
            run_directory / "segmentation.json",
        )
        pairing_snapshot = _add_file_snapshot(
            _input_snapshots,
            run_directory / "pairing.json",
        )
        alignment_snapshot = _add_file_snapshot(
            _input_snapshots,
            run_directory / "alignment.json",
        )
        analysis_path = paths.resolve_local(analysis.relative_path)
        analysis_snapshot = _add_file_snapshot(_input_snapshots, analysis_path)
        if analysis_snapshot.sha256 != analysis.sha256:
            raise CorpusFailure("CACHE_ARTIFACT_INVALID", "analysis audio snapshot is invalid")
        if segmentation_snapshot.sha256 != segmentation_sha256:
            raise ValueError("segmentation artifact changed while loading")
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
            pairing_snapshot.sha256,
            alignment_snapshot.sha256,
            run_directory,
        )
        review_prefixes[recording_id] = _validate_processing_history(
            processing_rows,
            context,
            review_path,
            review_events,
            config,
            paths,
        )
        contexts.append(context)
        _validate_file_snapshots(_input_snapshots)
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
            review_events=review_prefixes[context.recording.recording_id],
            analysis_audio=context.analysis,
            paths=paths,
            config=config,
            ffmpeg_version=ffmpeg_version,
            _validate_only=True,
        )
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
    existing_segments: tuple[SegmentRecord, ...] | None = None
    existing_by_recording: dict[str, dict[tuple[str, int], SegmentRecord]] = {}
    manifest_path = paths.manifests / "segments.jsonl"
    if recovery_only:
        _add_file_snapshot(
            _durable_snapshots,
            manifest_path,
            directory=_namespace_guard.manifests,
        )
        existing_segments = _validate_existing_manifest(paths)
        for segment in existing_segments:
            existing_by_recording.setdefault(segment.recording_id, {})[
                (segment.repetition_group_id, segment.take_index)
            ] = segment
    all_rows: list[tuple[str, int, int, SegmentRecord]] = []
    decisions_by_recording: dict[str, tuple[int, int]] = {}
    staging_identity = hashlib.sha256(
        "\0".join(
            (
                config.digest,
                review_sha256,
                rights_snapshot.sha256,
                transcripts_snapshot.sha256,
                *(
                    value
                    for context in contexts
                    for value in (
                        context.pairing_sha256,
                        context.alignment_sha256,
                    )
                ),
            )
        ).encode("utf-8")
    ).hexdigest()
    staging_root = paths.segments / ".manifest-staging" / staging_identity
    artifacts: list[DerivedAudio] = []
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
            _staging_root=staging_root if not recovery_only else None,
            _artifacts=artifacts,
            _existing_segments=(
                existing_by_recording.get(context.recording.recording_id, {})
                if recovery_only
                else None
            ),
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
    if existing_segments is not None and sorted_rows != existing_segments:
        raise ValueError("existing manifest row order differs from deterministic expected order")
    publication_guards: dict[str, _PublicationGuard] = {}
    artifact_guards: dict[str, _ArtifactGuard] = {}
    namespace_guard = _namespace_guard
    try:
        _validate_file_snapshots(_input_snapshots)
        publication_guards = _publication_guards(paths, tuple(artifacts))
        if not recovery_only:
            artifact_guards = _publish_staged_audio(
                paths,
                staging_root,
                tuple(artifacts),
                publication_guards,
                before_link=_before_publish_link,
            )
        else:
            artifact_guards = _pin_existing_artifacts(
                paths,
                tuple(artifacts),
                publication_guards,
            )
        _validate_file_snapshots(_input_snapshots)
        _validate_publication_guards(publication_guards)
        _validate_artifact_guards(artifact_guards)
        namespace_guard.validate()
        if not recovery_only:
            manifest_directory_fd = namespace_guard.directory_fd(namespace_guard.manifests)
            if manifest_directory_fd is None:
                write_jsonl_atomic(manifest_path, (row.to_dict() for row in sorted_rows))
            else:  # pragma: no cover - exercised on POSIX hosts
                write_jsonl_atomic(
                    manifest_path,
                    (row.to_dict() for row in sorted_rows),
                    directory_fd=manifest_directory_fd,
                )
            _add_file_snapshot(
                _durable_snapshots,
                manifest_path,
                directory=namespace_guard.manifests,
                expected_sha256=jsonl_sha256(row.to_dict() for row in sorted_rows),
            )
            namespace_guard.validate()
        _validate_file_snapshots(_input_snapshots)
        _validate_file_snapshots(_durable_snapshots)
        manifest_sha256 = _durable_snapshots[str(Path(os.path.abspath(manifest_path)))].sha256
        rights_sha256 = rights_snapshot.sha256
        transcripts_sha256 = transcripts_snapshot.sha256
        inventory_sha256 = jsonl_sha256(
            (
                replace(recording, state=CorpusState.REVIEWED)
                if recording.recording_id in selection_ids
                else recording
            ).to_dict()
            for recording in recordings
        )
        current = recordings
        terminal_events: list[ProcessingEvent] = []
        for context in contexts:
            approved_count, total_takes = decisions_by_recording[context.recording.recording_id]
            target = CorpusState.APPROVED if approved_count else CorpusState.REJECTED
            if approved_count > total_takes:
                raise AssertionError("approved count exceeds reviewed take count")
            timestamp = "1970-01-01T00:00:00+00:00"
            base = replace(current[context.recording_index], state=CorpusState.REVIEWED)
            advanced, event = advance_recording(
                base,
                target,
                input_sha256s=(
                    context.recording.sha256,
                    inventory_sha256,
                    rights_sha256,
                    transcripts_sha256,
                    context.pairing_sha256,
                    context.alignment_sha256,
                    review_sha256,
                    manifest_sha256,
                ),
                config_sha256=config.digest,
                tool_versions=("approved-manifest-v2", ffmpeg_version),
                started_at=timestamp,
                finished_at=timestamp,
                result="success",
            )
            _, v1_event = advance_recording(
                base,
                target,
                input_sha256s=(
                    context.recording.sha256,
                    rights_sha256,
                    transcripts_sha256,
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
            _, legacy_event = advance_recording(
                base,
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
            processing_by_id = {
                item.event_id: item
                for item in (ProcessingEvent.from_dict(row) for row in processing_rows)
            }
            if event.event_id in processing_by_id:
                if processing_by_id[event.event_id] != event:
                    raise ValueError("existing terminal event conflicts with deterministic event")
            elif v1_event.event_id in processing_by_id:
                if processing_by_id[v1_event.event_id] != v1_event:
                    raise ValueError("v1 terminal event conflicts with deterministic evidence")
                event = v1_event
            elif legacy_event.event_id in processing_by_id:
                if processing_by_id[legacy_event.event_id] != legacy_event:
                    raise ValueError("legacy terminal event conflicts with deterministic evidence")
                event = legacy_event
            current = (
                *current[: context.recording_index],
                advanced,
                *current[context.recording_index + 1 :],
            )
            terminal_events.append(event)
        _validate_publication_guards(publication_guards)
        _validate_artifact_guards(artifact_guards)
        namespace_guard.validate()

        def validate_transaction() -> None:
            namespace_guard.validate()
            _validate_file_snapshots(_input_snapshots)
            _validate_file_snapshots(_durable_snapshots)
            _validate_publication_guards(publication_guards)
            _validate_artifact_guards(artifact_guards)

        def before_events_write() -> None:
            validate_transaction()
            _release_file_snapshot(_input_snapshots, processing_path)

        def after_events_write(expected_sha256: str) -> None:
            _add_file_snapshot(
                _durable_snapshots,
                processing_path,
                directory=namespace_guard.processing,
                expected_sha256=expected_sha256,
            )

        def before_recordings_write() -> None:
            validate_transaction()
            _release_file_snapshot(_input_snapshots, recordings_path)

        def after_recordings_write(expected_sha256: str) -> None:
            _add_file_snapshot(
                _durable_snapshots,
                recordings_path,
                directory=namespace_guard.manifests,
                expected_sha256=expected_sha256,
            )

        validate_transaction()
        persist_recording_transitions(
            recordings_path=paths.manifests / "recordings.jsonl",
            events_path=processing_path,
            recordings=current,
            events=terminal_events,
            _recordings_directory_fd=namespace_guard.directory_fd(namespace_guard.manifests),
            _events_directory_fd=namespace_guard.directory_fd(namespace_guard.processing),
            _validate_durable_namespace=validate_transaction,
            _expected_recordings_sha256=recordings_snapshot.sha256,
            _expected_events_sha256=processing_snapshot.sha256,
            _before_events_write=before_events_write,
            _after_events_write=after_events_write,
            _before_recordings_write=before_recordings_write,
            _after_recordings_write=after_recordings_write,
        )
        validate_transaction()
        return True
    finally:
        active_failure = sys.exc_info()[0] is not None
        cleanup_failure: OSError | None = None
        cleanup_functions: tuple[Callable[[], None], ...] = (
            lambda: _close_artifact_guards(artifact_guards),
            lambda: _close_publication_guards(publication_guards),
        )
        for cleanup in cleanup_functions:
            try:
                cleanup()
            except OSError as error:
                if cleanup_failure is None:
                    cleanup_failure = error
        if cleanup_failure is not None and not active_failure:
            raise cleanup_failure
