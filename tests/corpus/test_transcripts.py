from dataclasses import replace

import pytest

from latintts.corpus.transcripts import (
    build_text_candidate,
    build_transcript,
    compare_asr_observation,
)
from latintts.pipeline import Pronouncer


def test_build_transcript_preserves_three_text_layers_and_units() -> None:
    candidate = build_text_candidate(
        source_id="vulgate-source",
        source_url="https://example.invalid/source",
        source_version="edition-1",
        accessed_at="2026-07-19",
        source_text="Pater noster, qui es in caelis.",
    )
    transcript = build_transcript(
        recording_id="rec-1",
        source_candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
        spoken_unit_lines=("Pater noster", "qui es in caelis"),
    )

    assert transcript.source_text == "Pater noster, qui es in caelis."
    assert transcript.spoken_text == "Pater noster\nqui es in caelis"
    assert [unit.text for unit in transcript.spoken_units] == [
        "Pater noster",
        "qui es in caelis",
    ]
    assert [
        (unit.token_start_index, unit.token_end_index)
        for unit in transcript.spoken_units
    ] == [(0, 2), (2, 6)]
    assert transcript.normalized_text == "Pater noster qui es in caelis"
    assert transcript.pronunciation_plan["rule_version"] == "ecclesiastical-roman-v1"
    assert "alignment_text" not in transcript.to_dict()


def test_asr_observation_never_mutates_spoken_text() -> None:
    observation = compare_asr_observation("pater noster", "pater poster")

    assert observation.hypothesis == "pater poster"
    assert observation.differences
    assert observation.confirmed_text is None


def test_build_transcript_rejects_noncanonical_pronunciation_rule_version() -> None:
    candidate = build_text_candidate(
        source_id="source-1",
        source_url="https://example.invalid/source",
        source_version="edition-1",
        accessed_at="2026-07-19",
        source_text="Pater noster",
    )
    pronouncer = replace(Pronouncer.default(), rule_version="experimental-v2")

    with pytest.raises(ValueError, match="ecclesiastical-roman-v1"):
        build_transcript(
            recording_id="rec-1",
            source_candidates=(candidate,),
            selected_candidate_id=candidate.candidate_id,
            spoken_unit_lines=("Pater noster",),
            pronouncer=pronouncer,
        )
