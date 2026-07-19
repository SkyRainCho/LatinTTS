from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from latintts.corpus.alignment import (
    AlignmentRequest,
    AlignmentResult,
    WordSpan,
    alignment_cache_key,
    result_from_dict,
    result_to_dict,
    validate_alignment,
)


def _request(tmp_path: Path, **changes: object) -> AlignmentRequest:
    values: dict[str, object] = {
        "audio_path": tmp_path / "take.wav",
        "audio_sha256": "a" * 64,
        "spoken_text": "Grātiā, plēna!",
        "config_sha256": "b" * 64,
    }
    values.update(changes)
    return AlignmentRequest(**values)  # type: ignore[arg-type]


def _result(**changes: object) -> AlignmentResult:
    values: dict[str, object] = {
        "backend": "fake",
        "backend_version": "1.0",
        "model_id": "fake/model",
        "model_revision": "revision-1",
        "model_license": "test-only",
        "alignment_text": "gratia plena",
        "words": (
            WordSpan("gratia", 0.0, 0.7, 0.9),
            WordSpan("plena", 0.7, 1.4, 0.8),
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


def test_alignment_request_hash_separates_text_from_backend_text(tmp_path: Path) -> None:
    request = _request(tmp_path)

    assert request.text_sha256 == hashlib.sha256("Grātiā, plēna!".encode()).hexdigest()
    assert request.text_sha256 != request.audio_sha256


def test_alignment_cache_key_changes_for_each_content_addressed_input(tmp_path: Path) -> None:
    request = _request(tmp_path)

    assert request.cache_key == alignment_cache_key(
        request.audio_sha256, request.text_sha256, request.config_sha256
    )
    assert request.cache_key != _request(tmp_path, audio_sha256="c" * 64).cache_key
    assert request.cache_key != _request(tmp_path, spoken_text="Gratia plena").cache_key
    assert request.cache_key != _request(tmp_path, config_sha256="d" * 64).cache_key


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"audio_sha256": "A" * 64}, "audio_sha256"),
        ({"config_sha256": "short"}, "config_sha256"),
        ({"spoken_text": "  , —  "}, "spoken_text"),
    ),
)
def test_alignment_request_rejects_invalid_provenance(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(tmp_path, **changes)


@pytest.mark.parametrize(
    "span",
    (
        ("", 0.0, 0.1, 0.5),
        ("word", -0.1, 0.1, 0.5),
        ("word", 0.1, 0.1, 0.5),
        ("word", 0.0, float("inf"), 0.5),
        ("word", 0.0, 0.1, float("nan")),
    ),
)
def test_word_span_rejects_empty_or_non_half_open_values(span: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        WordSpan(*span)  # type: ignore[arg-type]


@pytest.mark.parametrize("span", (("word", 0.0, 0.1, 1.1), ("word", 0.0, 0.1, "bad")))
def test_word_span_rejects_invalid_score(span: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="score"):
        WordSpan(*span)  # type: ignore[arg-type]


def test_alignment_rejects_overlapping_word_spans() -> None:
    with pytest.raises(ValueError, match="non-overlapping"):
        _result(
            words=(
                WordSpan("gratia", 0.0, 1.0, 0.9),
                WordSpan("plena", 0.9, 1.5, 0.8),
            )
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"backend": ""}, "backend"),
        ({"model_license": "  "}, "model_license"),
        ({"alignment_text": ""}, "alignment_text"),
        ({"words": ()}, "at least one"),
        ({"warnings": ["warning"]}, "warnings"),
        ({"raw_output": []}, "raw_output"),
        ({"alignment_level": "phoneme"}, "word-level"),
        ({"phoneme_timing_status": "estimated"}, "word-level"),
    ),
)
def test_alignment_result_rejects_invalid_contract_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _result(**changes)


@pytest.mark.parametrize(
    ("constructor", "message"),
    (
        (lambda tmp_path: _request(tmp_path, audio_path="take.wav"), "audio_path"),
        (lambda _tmp_path: _result(words=("not-a-span",)), "WordSpan"),
        (lambda _tmp_path: _result(raw_output={1: "not-json"}), "keys"),
        (lambda _tmp_path: _result(raw_output={"not-json": ("tuple",)}), "JSON"),
    ),
)
def test_contracts_reject_impostor_values(
    tmp_path: Path, constructor: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        constructor(tmp_path)  # type: ignore[operator]


def test_validate_alignment_maps_unicode_and_punctuation_without_mutating_spoken_text(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    result = _result()

    validate_alignment(result, request, audio_duration_seconds=1.4)

    assert request.spoken_text == "Grātiā, plēna!"
    assert result.alignment_text == "gratia plena"


@pytest.mark.parametrize(
    ("result", "message"),
    (
        (_result(alignment_text="gratia"), "alignment_text"),
        (
            _result(
                alignment_text="gratia error",
                words=(WordSpan("gratia", 0.0, 0.7, 0.9), WordSpan("error", 0.7, 1.4, 0.8)),
            ),
            "spoken_text",
        ),
        (
            _result(
                alignment_text="gratia", words=(WordSpan("gratia", 0.0, 0.7, 0.9),), coverage=1.0
            ),
            "coverage",
        ),
        (
            _result(
                words=(WordSpan("gratia", 0.0, 0.7, 0.9),), alignment_text="gratia", coverage=0.5
            ),
            "mean_score",
        ),
        (
            _result(words=(WordSpan("gratia", 0.0, 0.7, 0.9), WordSpan("plena", 0.7, 1.5, 0.8))),
            "audio duration",
        ),
    ),
)
def test_validate_alignment_rejects_bad_mapping_summaries_and_audio_bounds(
    tmp_path: Path, result: AlignmentResult, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_alignment(result, _request(tmp_path), audio_duration_seconds=1.4)


@pytest.mark.parametrize(
    ("result", "request_value", "duration", "message"),
    (
        (object(), None, 1.0, "result"),
        (None, object(), 1.0, "request"),
        (None, None, 0.0, "positive"),
    ),
)
def test_validate_alignment_rejects_invalid_objects_and_duration(
    tmp_path: Path, result: object, request_value: object, duration: float, message: str
) -> None:
    actual_result = _result() if result is None else result
    actual_request = _request(tmp_path) if request_value is None else request_value

    with pytest.raises((TypeError, ValueError), match=message):
        validate_alignment(  # type: ignore[arg-type]
            actual_result, actual_request, audio_duration_seconds=duration
        )


def test_alignment_result_json_round_trip_preserves_order_and_exact_schema() -> None:
    result = _result(warnings=("LOW_SCORE",), raw_output={"nested": [1, True, None]})

    raw = result_to_dict(result)

    assert json.loads(json.dumps(raw)) == raw
    assert result_from_dict(raw) == result
    assert [word["text"] for word in raw["words"]] == ["gratia", "plena"]


@pytest.mark.parametrize(
    "raw",
    (
        {"unknown": True},
        _result().raw_output,
        result_to_dict(_result()) | {"words": tuple(result_to_dict(_result())["words"])},
        result_to_dict(_result()) | {"warnings": [""]},
        result_to_dict(_result()) | {"raw_output": {"not_json": float("nan")}},
    ),
)
def test_alignment_result_from_dict_rejects_noncanonical_json(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        result_from_dict(raw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw",
    (
        [],
        result_to_dict(_result()) | {"words": [{"text": "gratia"}]},
        result_to_dict(_result()) | {"warnings": ()},
        result_to_dict(_result()) | {"warnings": [1]},
        result_to_dict(_result()) | {"raw_output": []},
    ),
)
def test_alignment_result_from_dict_rejects_wrong_json_types(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        result_from_dict(raw)  # type: ignore[arg-type]


def test_result_to_dict_rejects_non_result() -> None:
    with pytest.raises(TypeError, match="result"):
        result_to_dict(object())  # type: ignore[arg-type]


def test_result_to_dict_revalidates_a_tampered_frozen_result() -> None:
    result = _result()
    object.__setattr__(result, "model_license", "")

    with pytest.raises(ValueError, match="model_license"):
        result_to_dict(result)
