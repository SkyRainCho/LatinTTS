import json
from pathlib import Path

import pytest

from latintts.domain import ResolutionMethod
from latintts.pipeline import Pronouncer

GOLD_FIXTURE = Path("tests/fixtures/gold_pronunciations.jsonl")
ROWS: list[dict[str, object]] = [
    json.loads(line)
    for line in GOLD_FIXTURE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


@pytest.mark.parametrize(
    ("word", "source_ids"),
    [
        (
            "gloria",
            {
                "perseus-lewis-short",
                "allen-greenough-accents",
                "liber-usualis-1962",
            },
        ),
        ("kyrie", {"liber-usualis-1962"}),
    ],
)
def test_source_backed_gold_stress_is_not_a_candidate(
    word: str,
    source_ids: set[str],
) -> None:
    token = Pronouncer.default().analyze(word).tokens[0]

    assert token.resolution_method is ResolutionMethod.LEXICON
    assert token.stress_index == 0
    assert token.warnings == ()
    assert set(token.source_ids) == source_ids


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["word"])
def test_gold_pronunciation(row: dict[str, object]) -> None:
    plan = Pronouncer.default().analyze(str(row["word"]))
    token = plan.tokens[0]

    assert len(plan.tokens) == 1
    assert token.normalized == row["normalized"]
    assert list(token.syllables) == row["syllables"]
    assert token.stress_index == row["stress_index"]
    assert token.ipa == row["ipa"]
    assert list(token.applied_rule_ids) == row["rule_ids"]
    assert list(token.source_ids) == row["source_ids"]
