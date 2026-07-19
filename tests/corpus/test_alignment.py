from __future__ import annotations

import json
from pathlib import Path

import pytest

from latintts.corpus.alignment import (
    AlignmentRequest,
    AlignmentResult,
    AlignmentToken,
    WordSpan,
    alignment_cache_key,
    result_from_dict,
    result_to_dict,
    transform_latin_for_alignment,
    validate_alignment,
)


def _request(tmp_path: Path, **changes: object) -> AlignmentRequest:
    values: dict[str, object] = {
        "audio_path": tmp_path / "take.wav",
        "audio_sha256": "a" * 64,
        "spoken_text": "Grātiā plena",
        "segmentation_artifact_sha256": "b" * 64,
        "backend": "fake",
        "backend_version": "1.0",
        "model_id": "fake/model",
        "model_revision": "c" * 40,
        "model_license": "test-only",
        "config_sha256": "d" * 64,
        "effective_parameters": {"language": "lat"},
    }
    values.update(changes)
    return AlignmentRequest(**values)  # type: ignore[arg-type]


def _result(request: AlignmentRequest, **changes: object) -> AlignmentResult:
    values: dict[str, object] = {
        "backend": request.backend,
        "backend_version": request.backend_version,
        "model_id": request.model_id,
        "model_revision": request.model_revision,
        "model_license": request.model_license,
        "alignment_text": request.alignment_text,
        "tokens": (
            AlignmentToken(0, "gratia", 0, "Grātiā"),
            AlignmentToken(1, "plena", 1, "plena"),
        ),
        "words": (
            WordSpan("gratia", 0.0, 0.7, 0.9, 0),
            WordSpan("plena", 0.7, 1.4, 0.8, 1),
        ),
        "coverage": 1.0,
        "mean_score": 0.85,
        "alignment_level": "word",
        "phoneme_timing_status": "not_estimated",
        "warnings": (),
        "raw_output": {"backend": {"segments": 2}},
    }
    values.update(changes)
    return AlignmentResult(**values)  # type: ignore[arg-type]


def test_alignment_request_separates_authoritative_and_backend_text(tmp_path: Path) -> None:
    request = _request(tmp_path)

    assert request.alignment_text == "gratia plena"
    assert request.text_sha256 != request.alignment_text_sha256
    assert request.spoken_text == "Grātiā plena"


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"audio_sha256": "A" * 64}, "audio_sha256"),
        ({"segmentation_artifact_sha256": "short"}, "segmentation_artifact_sha256"),
        ({"model_revision": "revision-1"}, "model_revision"),
        ({"effective_parameters": {}}, "effective_parameters"),
        ({"effective_parameters": {"set": {"not-json"}}}, "effective_parameters"),
    ),
)
def test_alignment_request_rejects_invalid_provenance(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _request(tmp_path, **changes)


@pytest.mark.parametrize(
    "span",
    (
        ("", 0.0, 0.1, 0.5, 0),
        ("word", -0.1, 0.1, 0.5, 0),
        ("word", 0.1, 0.1, 0.5, 0),
        ("word", 0.0, float("inf"), 0.5, 0),
        ("word", 0.0, 0.1, 1.1, 0),
        ("word", 0.0, 0.1, 0.5, -1),
    ),
)
def test_word_span_rejects_empty_or_non_half_open_values(span: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        WordSpan(*span)  # type: ignore[arg-type]


def test_alignment_rejects_overlapping_word_spans(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="non-overlapping"):
        _result(
            request,
            words=(WordSpan("gratia", 0.0, 1.0, 0.9, 0), WordSpan("plena", 0.9, 1.5, 0.8, 1)),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"alignment_level": "phoneme"}, "word-level"),
        ({"phoneme_timing_status": "estimated"}, "word-level"),
        ({"tokens": []}, "tokens"),
        ({"raw_output": []}, "raw_output"),
    ),
)
def test_alignment_result_rejects_invalid_contract_values(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _result(_request(tmp_path), **changes)


def test_validate_alignment_accepts_exact_complete_word_evidence(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = _result(request)

    validate_alignment(result, request, audio_duration_seconds=1.4)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"backend": "other"}, "backend"),
        ({"coverage": 0.5}, "coverage"),
        ({"mean_score": 0.8}, "mean_score"),
        (
            {"words": (WordSpan("gratia", 0.0, 0.7, 0.9, 0), WordSpan("plena", 0.7, 1.5, 0.8, 1))},
            "audio duration",
        ),
    ),
)
def test_validate_alignment_rejects_provenance_summaries_and_audio_bounds(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match=message):
        validate_alignment(_result(request, **changes), request, audio_duration_seconds=1.4)


def test_result_json_round_trip_is_exact_and_defensive(tmp_path: Path) -> None:
    result = _result(_request(tmp_path), warnings=("LOW_SCORE",), raw_output={"nested": [1, True]})

    raw = result_to_dict(result)

    assert json.loads(json.dumps(raw)) == raw
    assert result_from_dict(raw) == result
    assert raw["integrity_sha256"] == result.integrity_sha256


@pytest.mark.parametrize(
    "change",
    (
        {"warnings": ["forged"]},
        {
            "words": [
                {
                    "text": "gratia",
                    "start_seconds": 0.0,
                    "end_seconds": 0.7,
                    "score": 0.1,
                    "spoken_token_index": 0,
                },
                {
                    "text": "plena",
                    "start_seconds": 0.7,
                    "end_seconds": 1.4,
                    "score": 0.8,
                    "spoken_token_index": 1,
                },
            ]
        },
        {"raw_output": {"forged": True}},
    ),
)
def test_result_from_dict_rejects_integrity_tampering(
    tmp_path: Path, change: dict[str, object]
) -> None:
    raw = result_to_dict(_result(_request(tmp_path))) | change

    with pytest.raises(ValueError, match="integrity"):
        result_from_dict(raw)


@pytest.mark.parametrize(
    "raw",
    (
        {},
        [],
        {"unknown": True},
    ),
)
def test_result_from_dict_requires_exact_schema(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        result_from_dict(raw)  # type: ignore[arg-type]


def test_transform_rejects_empty_tokens_and_unknown_versions() -> None:
    with pytest.raises(ValueError):
        transform_latin_for_alignment("—")
    with pytest.raises(ValueError, match="version"):
        transform_latin_for_alignment("Gratia", version="unknown")


@pytest.mark.parametrize(
    "constructor",
    (
        lambda tmp_path: _request(tmp_path, audio_path="take.wav"),
        lambda tmp_path: _request(tmp_path, effective_parameters={1: "invalid"}),
        lambda _tmp_path: AlignmentToken(-1, "gratia", 0, "Gratia"),
        lambda _tmp_path: AlignmentToken(0, "gratia plena", 0, "Gratia"),
        lambda _tmp_path: WordSpan("gratia plena", 0.0, 0.1, 0.5, 0),
        lambda _tmp_path: WordSpan("gratia", "bad", 0.1, 0.5, 0),
    ),
)
def test_contract_values_reject_impostors(tmp_path: Path, constructor: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        constructor(tmp_path)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"tokens": ("not-token", "not-token")}, "tokens"),
        ({"tokens": (AlignmentToken(0, "gratia", 0, "Grātiā"),)}, "one form"),
        ({"words": ()}, "words"),
        ({"words": ("not-word",)}, "words"),
        ({"warnings": ["mutable"]}, "warnings"),
        ({"alignment_transform_version": "unknown"}, "transform"),
    ),
)
def test_result_rejects_noncanonical_members(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _result(_request(tmp_path), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"alignment_text": "plena gratia"}, "alignment_text"),
        (
            {
                "tokens": (
                    AlignmentToken(1, "gratia", 0, "Grātiā"),
                    AlignmentToken(0, "plena", 1, "plena"),
                )
            },
            "token indexes",
        ),
        (
            {
                "tokens": (
                    AlignmentToken(0, "wrong", 0, "Grātiā"),
                    AlignmentToken(1, "plena", 1, "plena"),
                )
            },
            "mapping",
        ),
        (
            {
                "words": (
                    WordSpan("wrong", 0.0, 0.7, 0.9, 0),
                    WordSpan("plena", 0.7, 1.4, 0.8, 1),
                )
            },
            "word span",
        ),
    ),
)
def test_validate_alignment_rejects_transform_and_mapping_mismatches(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match=message):
        validate_alignment(_result(request, **changes), request, audio_duration_seconds=1.4)


def test_validate_alignment_and_cache_key_reject_impostor_objects(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="request"):
        alignment_cache_key(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="result"):
        validate_alignment(object(), _request(tmp_path), audio_duration_seconds=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="request"):
        validate_alignment(_result(_request(tmp_path)), object(), audio_duration_seconds=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        validate_alignment(
            _result(_request(tmp_path)), _request(tmp_path), audio_duration_seconds=0.0
        )


@pytest.mark.parametrize(
    "change",
    (
        {"tokens": ()},
        {"tokens": [{"alignment_token_index": 0}]},
        {"words": ()},
        {"words": [{"text": "gratia"}]},
        {"warnings": ()},
        {"warnings": [1]},
        {"raw_output": []},
    ),
)
def test_result_from_dict_rejects_wrong_json_member_types(
    tmp_path: Path, change: dict[str, object]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        result_from_dict(result_to_dict(_result(_request(tmp_path))) | change)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_license", "forged"),
        ("warnings", ("FORGED",)),
        ("raw_output", {"forged": True}),
    ),
)
def test_result_to_dict_rejects_direct_frozen_object_tampering(
    tmp_path: Path, field: str, value: object
) -> None:
    result = _result(_request(tmp_path))
    object.__setattr__(result, field, value)

    with pytest.raises(ValueError, match="integrity"):
        result_to_dict(result)
