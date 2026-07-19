from __future__ import annotations

from pathlib import Path

import pytest

from latintts.corpus.alignment import (
    AlignmentRequest,
    AlignmentResult,
    AlignmentToken,
    WordSpan,
    result_from_dict,
    result_to_dict,
    validate_alignment,
)


def _request(tmp_path: Path, **changes: object) -> AlignmentRequest:
    values: dict[str, object] = {
        "audio_path": tmp_path / "take.wav",
        "audio_sha256": "a" * 64,
        "spoken_text": "Veni, veni Iesus",
        "segmentation_artifact_sha256": "b" * 64,
        "backend": "fake",
        "backend_version": "1.0",
        "model_id": "fake/model",
        "model_revision": "c" * 40,
        "model_license": "test-only",
        "config_sha256": "d" * 64,
        "effective_parameters": {"language": "lat", "romanize": True},
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
        "alignment_transform_version": request.alignment_transform_version,
        "alignment_text": request.alignment_text,
        "tokens": (
            AlignmentToken(0, "ueni", 0, "Veni"),
            AlignmentToken(1, "ueni", 1, "veni"),
            AlignmentToken(2, "iesus", 2, "Iesus"),
        ),
        "words": (
            WordSpan("ueni", 0.0, 0.4, 0.8, 0),
            WordSpan("ueni", 0.4, 0.8, 0.8, 1),
            WordSpan("iesus", 0.8, 1.2, 0.8, 2),
        ),
        "coverage": 1.0,
        "mean_score": 0.8,
        "alignment_level": "word",
        "phoneme_timing_status": "not_estimated",
        "warnings": ("LOW_SCORE",),
        "raw_output": {"segments": [{"score": 0.8}]},
    }
    values.update(changes)
    return AlignmentResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    (
        {"audio_sha256": "e" * 64},
        {"spoken_text": "Veni Iesus"},
        {"segmentation_artifact_sha256": "f" * 64},
        {"backend": "other"},
        {"backend_version": "2.0"},
        {"model_id": "other/model"},
        {"model_revision": "e" * 40},
        {"model_license": "other-license"},
        {"config_sha256": "f" * 64},
        {"effective_parameters": {"language": "lat", "romanize": False}},
    ),
)
def test_cache_identity_binds_every_alignment_provenance_field(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    request = _request(tmp_path)

    assert request.cache_key != _request(tmp_path, **changes).cache_key


def test_request_transform_is_versioned_uroman_compatible_and_non_authoritative(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, spoken_text="Jūlius vīvō æquus œconomus")

    assert request.alignment_transform_version == "latin-uroman-v1"
    assert request.alignment_text == "iulius uiuo aequus oeconomus"
    assert request.spoken_text == "Jūlius vīvō æquus œconomus"


def test_request_rejects_non_json_or_non_finite_effective_parameters(tmp_path: Path) -> None:
    with pytest.raises((TypeError, ValueError), match="effective_parameters"):
        _request(tmp_path, effective_parameters={"window": float("nan")})


@pytest.mark.parametrize(
    ("tokens", "words", "message"),
    (
        (
            (AlignmentToken(0, "ueni", 0, "Veni"), AlignmentToken(1, "ueni", 1, "veni")),
            (WordSpan("ueni", 0.0, 0.4, 0.8, 0), WordSpan("ueni", 0.4, 0.8, 0.8, 1)),
            "cover every",
        ),
        (
            (
                AlignmentToken(0, "ueni", 0, "Veni"),
                AlignmentToken(1, "ueni", 0, "veni"),
                AlignmentToken(2, "iesus", 2, "Iesus"),
            ),
            (
                WordSpan("ueni", 0.0, 0.4, 0.8, 0),
                WordSpan("ueni", 0.4, 0.8, 0.8, 0),
                WordSpan("iesus", 0.8, 1.2, 0.8, 2),
            ),
            "unique",
        ),
        (
            (
                AlignmentToken(0, "ueni", 0, "Veni"),
                AlignmentToken(1, "ueni", 1, "veni"),
                AlignmentToken(2, "iesus", 2, "Iesus"),
            ),
            (
                WordSpan("ueni", 0.0, 0.4, 0.8, 1),
                WordSpan("ueni", 0.4, 0.8, 0.8, 0),
                WordSpan("iesus", 0.8, 1.2, 0.8, 2),
            ),
            "index order",
        ),
    ),
)
def test_validate_alignment_requires_complete_unique_ordered_spoken_token_mapping(
    tmp_path: Path,
    tokens: tuple[AlignmentToken, ...],
    words: tuple[WordSpan, ...],
    message: str,
) -> None:
    request = _request(tmp_path)
    result = _result(
        request,
        tokens=tokens,
        words=words,
        alignment_text=" ".join(token.alignment_form for token in tokens),
    )

    with pytest.raises(ValueError, match=message):
        validate_alignment(result, request, audio_duration_seconds=1.2)


def test_duplicate_surfaces_are_mapped_by_explicit_indexes(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = _result(request)

    validate_alignment(result, request, audio_duration_seconds=1.2)

    assert [word.spoken_token_index for word in result.words] == [0, 1, 2]


def test_result_integrity_detects_tampering_and_raw_output_is_deeply_isolated(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    source_raw = {"segments": [{"score": 0.8}]}
    result = _result(request, raw_output=source_raw)
    source_raw["segments"][0]["score"] = 0.1
    serialized = result_to_dict(result)

    assert serialized["raw_output"]["segments"][0]["score"] == 0.8
    serialized["raw_output"]["segments"][0]["score"] = 0.2
    assert result_to_dict(result)["raw_output"]["segments"][0]["score"] == 0.8

    serialized["model_license"] = "forged"
    with pytest.raises(ValueError, match="integrity"):
        result_from_dict(serialized)
