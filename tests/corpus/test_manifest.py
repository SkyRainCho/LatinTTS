from __future__ import annotations

import hashlib

import pytest

from latintts.corpus import manifest as manifest_module
from latintts.corpus.alignment import WordSpan
from latintts.corpus.audio import PcmMetrics
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.manifest import require_approval
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import RightsRecord, SegmentRecord
from latintts.corpus.store import write_jsonl_atomic
from latintts.corpus.transcripts import SpokenUnit


def test_manifest_rejects_take_without_human_approval() -> None:
    with pytest.raises(CorpusFailure) as error:
        require_approval(review_decision=None, pronunciation_warning_codes=())
    assert error.value.code == "REVIEW_REQUIRED"


def test_manifest_rejects_unresolved_pronunciation_warning() -> None:
    with pytest.raises(CorpusFailure) as error:
        require_approval(
            review_decision="approved",
            pronunciation_warning_codes=("PRONUNCIATION_NEEDS_REVIEW",),
        )
    assert error.value.code == "PRONUNCIATION_NEEDS_REVIEW"


def test_manifest_accepts_human_approved_warning_free_take() -> None:
    assert (
        require_approval(
            review_decision="approved",
            pronunciation_warning_codes=(),
        )
        is None
    )


def _segment(**changes: object) -> SegmentRecord:
    values: dict[str, object] = {
        "schema_version": "1",
        "corpus_version": "corpus-v1",
        "config_sha256": "a" * 64,
        "segment_id": "rec-1-unit-0001-take-1",
        "recording_id": "rec-1",
        "text_unit_id": "rec-1-unit-0001",
        "repetition_group_id": "rec-1-unit-0001",
        "take_index": 1,
        "source_audio_sha256": "b" * 64,
        "source_start_sample": 4_800,
        "source_end_sample": 48_000,
        "derived_audio_relative_path": (
            "derived/corpus-v1/segments/lossless/lossless-" + "c" * 64 + ".flac"
        ),
        "derived_audio_sha256": "d" * 64,
        "source_text_id": "text-source-1",
        "spoken_text": "gratia plena",
        "normalized_text": "gratia plena",
        "pronunciation_schema_version": "1",
        "rule_version": "ecclesiastical-roman-v1",
        "ipa_by_token": ("grat͡sia", "plena"),
        "model_phonemes_by_token": (("g", "r", "a"), ("p", "l", "e")),
        "word_spans": (
            WordSpan("gratia", 0.0, 0.4, 0.9, 0),
            WordSpan("plena", 0.4, 0.9, 0.8, 1),
        ),
        "alignment_backend": "fake",
        "alignment_model_id": "fake/model",
        "alignment_model_revision": "e" * 40,
        "alignment_model_license": "MIT",
        "alignment_level": "word",
        "phoneme_timing_status": "not_estimated",
        "quality_metrics": PcmMetrics(14_400, 0.5, 0.2, 0.0, 0.1),
        "quality_metric_audio_sha256": "f" * 64,
        "quality_labels": ("reverb",),
        "review_event_ids": ("review-" + "1" * 64,),
        "rights_id": "rights-1",
        "split": "unassigned",
    }
    values.update(changes)
    return SegmentRecord(**values)  # type: ignore[arg-type]


def test_segment_record_round_trips_exact_provenance() -> None:
    record = _segment()
    assert SegmentRecord.from_dict(record.to_dict()) == record
    assert record.phoneme_timing_status == "not_estimated"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"take_index": 3}, "take_index"),
        ({"source_end_sample": 4_800}, "sample"),
        ({"model_phonemes_by_token": (("g",), ())}, "model phonemes"),
        ({"corpus_version": "corpus-v2"}, "corpus_version"),
        ({"rule_version": "classical-v1"}, "rule_version"),
        ({"alignment_level": "phoneme"}, "word-level"),
        ({"review_event_ids": ()}, "review_event_ids"),
    ],
)
def test_segment_record_rejects_invalid_manifest_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _segment(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"derived_audio_relative_path": "derived/corpus-v1/segments/../bad.flac"},
        {"pronunciation_schema_version": "2"},
        {"ipa_by_token": ()},
        {"word_spans": ()},
        {
            "word_spans": (
                WordSpan("gratia", 0.0, 0.6, 0.9, 0),
                WordSpan("plena", 0.5, 0.9, 0.8, 1),
            )
        },
        {"quality_metrics": object()},
        {"quality_labels": ("",)},
        {"quality_labels": ("reverb", "reverb")},
        {"review_event_ids": ("not-review",)},
        {"split": "train"},
    ],
)
def test_segment_record_rejects_additional_contract_drift(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _segment(**changes)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ipa_by_token", ("gratia", "plena")),
        ("model_phonemes_by_token", [("g",), ["p"]]),
        ("word_spans", ["bad"]),
        ("quality_metrics", "bad"),
    ],
)
def test_segment_record_from_dict_rejects_non_json_shape(field: str, value: object) -> None:
    raw = _segment().to_dict()
    raw[field] = value
    with pytest.raises((TypeError, ValueError)):
        SegmentRecord.from_dict(raw)


def _plan(token: object = None) -> dict[str, object]:
    if token is None:
        token = {
            "surface": "Pater",
            "ipa": "pater",
            "model_phonemes": ["p", "a"],
            "warning_codes": [],
            "normalized": "pater",
        }
    return {
        "schema_version": "1",
        "rule_version": "ecclesiastical-roman-v1",
        "tokens": [token],
    }


@pytest.mark.parametrize(
    ("plan", "unit"),
    [
        (None, SpokenUnit("unit", 1, "Pater", 0, 1)),
        ({"schema_version": "2", "rule_version": "bad"}, SpokenUnit("unit", 1, "Pater", 0, 1)),
        (
            {"schema_version": "1", "rule_version": "ecclesiastical-roman-v1", "tokens": {}},
            SpokenUnit("unit", 1, "Pater", 0, 1),
        ),
        (_plan(), SpokenUnit("unit", 1, "Pater", 1, 2)),
        (_plan(), SpokenUnit("unit", 1, "Pater noster", 0, 1)),
        (_plan("bad"), SpokenUnit("unit", 1, "Pater", 0, 1)),
        (
            _plan(
                {
                    "surface": "Pater",
                    "ipa": "",
                    "model_phonemes": ["p"],
                    "warning_codes": [],
                    "normalized": "pater",
                }
            ),
            SpokenUnit("unit", 1, "Pater", 0, 1),
        ),
        (
            _plan(
                {
                    "surface": "Pater",
                    "ipa": "p",
                    "model_phonemes": [],
                    "warning_codes": [],
                    "normalized": "pater",
                }
            ),
            SpokenUnit("unit", 1, "Pater", 0, 1),
        ),
        (
            _plan(
                {
                    "surface": "Pater",
                    "ipa": "p",
                    "model_phonemes": ["p"],
                    "warning_codes": {},
                    "normalized": "pater",
                }
            ),
            SpokenUnit("unit", 1, "Pater", 0, 1),
        ),
        (
            _plan(
                {
                    "surface": "Pater",
                    "ipa": "p",
                    "model_phonemes": ["p"],
                    "warning_codes": [],
                    "normalized": "",
                }
            ),
            SpokenUnit("unit", 1, "Pater", 0, 1),
        ),
    ],
)
def test_pronunciation_slice_rejects_schema_and_token_drift(plan: object, unit: SpokenUnit) -> None:
    with pytest.raises((CorpusFailure, TypeError, ValueError)):
        manifest_module._pronunciation_slice({"pronunciation_plan": plan}, unit)


@pytest.mark.parametrize(
    "effective",
    [
        {"words": [], "segment_start": 0.0},
        {"words": [{"text": "gratia"}], "segment_start": "bad"},
        {"words": ["bad"], "segment_start": 0.0},
    ],
)
def test_effective_word_spans_rejects_invalid_review_values(
    effective: dict[str, object],
) -> None:
    result = type("Result", (), {"words": (WordSpan("gratia", 0.0, 0.4, 0.9, 0),)})()
    with pytest.raises((TypeError, ValueError)):
        manifest_module._effective_word_spans(effective, result)  # type: ignore[arg-type]


def _transcript_sources() -> dict[str, object]:
    text = "Pater"
    return {
        "schema_version": "1",
        "recording_id": "rec-1",
        "state": "TRANSCRIPT_CONFIRMED",
        "selected_candidate_id": "candidate-1",
        "source_text": text,
        "source_candidates": [
            {
                "candidate_id": "candidate-1",
                "source_id": "source-1",
                "source_url": "https://example.invalid",
                "source_version": "1",
                "accessed_at": "2026-07-19",
                "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_text": text,
            }
        ],
    }


@pytest.mark.parametrize(
    "case", ("identity", "empty", "object", "hash", "selected-type", "selected")
)
def test_transcript_source_validation_rejects_provenance_drift(case: str) -> None:
    raw = _transcript_sources()
    if case == "identity":
        raw["recording_id"] = "other"
    elif case == "empty":
        raw["source_candidates"] = []
    elif case == "object":
        raw["source_candidates"] = [None]
    elif case == "hash":
        raw["source_candidates"][0]["source_sha256"] = "0" * 64  # type: ignore[index]
    elif case == "selected-type":
        raw["selected_candidate_id"] = None
    else:
        raw["selected_candidate_id"] = "missing"
    with pytest.raises(ValueError):
        manifest_module._validate_transcript_sources(raw, "rec-1")  # type: ignore[arg-type]


def test_manifest_rights_loader_converts_external_shape_type_error(tmp_path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    row = RightsRecord(
        "rights-1",
        "owner-1",
        "speaker-1",
        True,
        True,
        True,
        "unknown",
        "unknown",
        "unknown",
        "2026-07-19",
        "consent",
        "",
    ).to_dict()
    row["allow_model_training"] = "yes"
    write_jsonl_atomic(paths.manifests / "rights.jsonl", (row,))

    with pytest.raises(ValueError, match="rights"):
        manifest_module._load_rights(paths)


def test_existing_manifest_loader_converts_external_shape_type_error(tmp_path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    row = _segment().to_dict()
    row["quality_metrics"] = []
    write_jsonl_atomic(paths.manifests / "segments.jsonl", (row,))

    with pytest.raises(ValueError, match="segment"):
        manifest_module._validate_existing_manifest(paths)
