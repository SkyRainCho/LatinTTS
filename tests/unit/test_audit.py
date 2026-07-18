from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from latintts.audit import (
    AuditError,
    AuditPolicy,
    AuditReport,
    GoldEntry,
    audit_gold_file,
    main,
)
from latintts.domain import ResolutionMethod
from latintts.normalization import normalize_word
from latintts.pipeline import Pronouncer

GOLD_FIXTURE = Path("tests/fixtures/gold_pronunciations.jsonl")
GOLD_FIELDS = (
    "word",
    "normalized",
    "syllables",
    "stress_index",
    "ipa",
    "category",
    "rule_ids",
    "source_ids",
    "review_state",
)
LIGATURE_PAIRS = (
    ("aetas", "ætas"),
    ("laetum", "lætum"),
    ("praesens", "præsens"),
    ("poenas", "pœnas"),
    ("foetus", "fœtus"),
)


def _valid_row() -> dict[str, Any]:
    return {
        "word": "ave",
        "normalized": "ave",
        "syllables": ["a", "ve"],
        "stress_index": 0,
        "ipa": "ˈa.ve",
        "category": "stress",
        "rule_ids": ["disyllable-stress", "simple-a", "simple-v", "simple-e"],
        "source_ids": ["allen-greenough-accents", "liber-usualis-1962"],
        "review_state": "approved",
    }


def _write_rows(path: Path, *rows: dict[str, Any]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )


def _error_codes(path: Path) -> list[str]:
    report = audit_gold_file(path, AuditPolicy(minimum_total=0, category_minimums={}))
    return [error.code for error in report.errors]


def _file_read_error(path: Path) -> AuditReport:
    return AuditReport(
        total=0,
        category_counts={},
        errors=(AuditError("FILE_READ_ERROR", f"cannot read gold file: {path}"),),
    )


def _gold_rows_by_word() -> dict[str, dict[str, Any]]:
    rows = (
        json.loads(line)
        for line in GOLD_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {row["word"]: row for row in rows}


def test_gold_entry_has_the_fixed_exact_schema() -> None:
    assert tuple(field.name for field in fields(GoldEntry)) == GOLD_FIELDS


def test_missing_file_returns_a_stable_file_read_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    assert audit_gold_file(missing) == _file_read_error(missing)


def test_invalid_utf8_returns_a_stable_file_read_error(tmp_path: Path) -> None:
    fixture = tmp_path / "invalid-utf8.jsonl"
    fixture.write_bytes(b"\xff\xfe")

    assert audit_gold_file(fixture) == _file_read_error(fixture)


def test_gold_fixture_has_unique_approved_source_backed_entries() -> None:
    report = audit_gold_file(
        GOLD_FIXTURE,
        AuditPolicy(
            minimum_total=270,
            category_minimums={
                "vowels": 20,
                "diphthongs": 20,
                "consonants": 100,
                "syllabification": 40,
                "stress": 60,
                "orthographic_variants": 30,
            },
        ),
    )

    assert report.errors == ()
    assert report.total == 300
    assert report.category_counts == {
        "vowels": 25,
        "diphthongs": 25,
        "consonants": 110,
        "syllabification": 40,
        "stress": 65,
        "liturgical": 5,
        "orthographic_variants": 30,
    }


def test_ligature_pairs_share_lookup_and_pronunciation_but_keep_surface() -> None:
    rows = _gold_rows_by_word()
    pronouncer = Pronouncer.default()

    for digraph, ligature in LIGATURE_PAIRS:
        digraph_row = rows[digraph]
        ligature_row = rows[ligature]
        digraph_word = normalize_word(digraph)
        ligature_word = normalize_word(ligature)
        digraph_token = pronouncer.analyze(digraph).tokens[0]
        ligature_token = pronouncer.analyze(ligature).tokens[0]

        assert digraph != ligature
        assert digraph_word.lookup_key == ligature_word.lookup_key
        assert "expand-ae-ligature" not in digraph_word.transformations
        assert "expand-oe-ligature" not in digraph_word.transformations
        assert any(
            transform in ligature_word.transformations
            for transform in ("expand-ae-ligature", "expand-oe-ligature")
        )
        assert (
            digraph_token.normalized,
            digraph_token.syllables,
            digraph_token.stress_index,
            digraph_token.ipa,
        ) == (
            ligature_token.normalized,
            ligature_token.syllables,
            ligature_token.stress_index,
            ligature_token.ipa,
        )
        assert digraph_row["word"] == digraph
        assert ligature_row["word"] == ligature


def test_j_i_contexts_unify_lookup_without_rewriting_canonical_spelling() -> None:
    pronouncer = Pronouncer.default()
    for j_form, i_form in (
        ("Joseph", "Ioseph"),
        ("Joannes", "Ioannes"),
        ("judex", "iudex"),
        ("ejus", "eius"),
        ("cujus", "cuius"),
    ):
        j_word = normalize_word(j_form)
        i_word = normalize_word(i_form)
        j_token = pronouncer.analyze(j_form).tokens[0]
        i_token = pronouncer.analyze(i_form).tokens[0]

        assert j_word.normalized != i_word.normalized
        assert j_word.lookup_key == i_word.lookup_key
        assert j_token.ipa == i_token.ipa
        assert "j-consonantal" in j_token.applied_rule_ids
        assert "i-consonantal" in i_token.applied_rule_ids

    troia = pronouncer.analyze("Troia").tokens[0]
    finis = pronouncer.analyze("finis").tokens[0]
    assert "i-consonantal" in troia.applied_rule_ids
    assert "i-consonantal" not in finis.applied_rule_ids
    assert "simple-i" in finis.applied_rule_ids


def test_u_v_lookup_unification_preserves_pronunciation_roles_and_h_boundaries() -> None:
    pronouncer = Pronouncer.default()
    for word in ("servus", "avus", "vivus", "vox"):
        normalized = normalize_word(word)
        token = pronouncer.analyze(word).tokens[0]
        assert "v" in normalized.normalized
        assert "v" not in normalized.lookup_key
        assert "simple-v" in token.applied_rule_ids

    for word in ("unus", "umbra"):
        normalized = normalize_word(word)
        token = pronouncer.analyze(word).tokens[0]
        assert normalized.normalized == normalized.lookup_key
        assert "simple-u" in token.applied_rule_ids
        assert "simple-v" not in token.applied_rule_ids

    for word in ("nihildum", "nihilne"):
        token = pronouncer.analyze(word).tokens[0]
        assert token.resolution_method is ResolutionMethod.EXCEPTION
        assert token.applied_rule_ids[-1] == "h-mihi-nihil"
        assert "h-muted" not in token.applied_rule_ids

    for word in ("traho", "honor"):
        token = pronouncer.analyze(word).tokens[0]
        assert "h-muted" in token.applied_rule_ids
        assert "h-mihi-nihil" not in token.applied_rule_ids


def test_every_gold_row_exactly_matches_an_approved_runtime_token() -> None:
    pronouncer = Pronouncer.default()
    for row in _gold_rows_by_word().values():
        token = pronouncer.analyze(row["word"]).tokens[0]

        assert row["review_state"] == "approved"
        assert token.resolution_method is not ResolutionMethod.CANDIDATE
        assert token.warnings == ()
        assert token.normalized == row["normalized"]
        assert list(token.syllables) == row["syllables"]
        assert token.stress_index == row["stress_index"]
        assert token.ipa == row["ipa"]
        assert list(token.applied_rule_ids) == row["rule_ids"]
        assert list(token.source_ids) == row["source_ids"]


def test_audit_rejects_duplicate_word_and_unknown_source(tmp_path: Path) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = _valid_row()
    row["source_ids"] = ["missing"]
    _write_rows(fixture, row, row)

    report = audit_gold_file(
        fixture,
        AuditPolicy(minimum_total=1, category_minimums={}),
    )

    assert {error.code for error in report.errors} == {
        "DUPLICATE_WORD",
        "UNKNOWN_SOURCE",
    }


@pytest.mark.parametrize(
    "contents",
    [
        "{not-json}\n",
        "[]\n",
        "null\n",
        '"a string"\n',
    ],
    ids=("malformed-json", "array", "null", "string"),
)
def test_malformed_json_values_become_invalid_row(
    tmp_path: Path,
    contents: str,
) -> None:
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text(contents, encoding="utf-8")

    assert _error_codes(fixture) == ["INVALID_ROW"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("word", 1),
        ("normalized", ["ave"]),
        ("ipa", None),
        ("category", False),
        ("review_state", 1.0),
        ("syllables", "ave"),
        ("syllables", ["a", 1]),
        ("rule_ids", "simple-a"),
        ("rule_ids", ["simple-a", None]),
        ("source_ids", "liber-usualis-1962"),
        ("source_ids", [False]),
        ("stress_index", False),
        ("stress_index", 0.0),
        ("stress_index", "0"),
    ],
)
def test_wrong_field_types_become_invalid_row(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = _valid_row()
    row[field] = value
    _write_rows(fixture, row)

    assert _error_codes(fixture) == ["INVALID_ROW"]


@pytest.mark.parametrize("extra_or_missing", ["extra", "missing"])
def test_non_exact_fields_become_invalid_row(
    tmp_path: Path,
    extra_or_missing: str,
) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = _valid_row()
    if extra_or_missing == "extra":
        row["note"] = "not part of the fixed schema"
    else:
        del row["ipa"]
    _write_rows(fixture, row)

    assert _error_codes(fixture) == ["INVALID_ROW"]


@pytest.mark.parametrize("field", ["word", "normalized", "ipa", "category", "review_state"])
def test_blank_string_fields_are_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = _valid_row()
    row[field] = " "
    _write_rows(fixture, row)

    assert "EMPTY_FIELD" in _error_codes(fixture)


@pytest.mark.parametrize("field", ["syllables", "rule_ids", "source_ids"])
def test_empty_list_fields_are_rejected(tmp_path: Path, field: str) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = _valid_row()
    row[field] = []
    _write_rows(fixture, row)

    assert "EMPTY_FIELD" in _error_codes(fixture)


@pytest.mark.parametrize("field", ["syllables", "rule_ids", "source_ids"])
def test_blank_list_items_are_rejected(tmp_path: Path, field: str) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = _valid_row()
    row[field] = [" "]
    _write_rows(fixture, row)

    assert "EMPTY_FIELD" in _error_codes(fixture)


@pytest.mark.parametrize("field", ["syllables", "rule_ids", "source_ids"])
def test_duplicate_list_items_become_invalid_row(tmp_path: Path, field: str) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = _valid_row()
    item = row[field][0]
    row[field] = [item, item]
    _write_rows(fixture, row)

    assert _error_codes(fixture) == ["INVALID_ROW"]


@pytest.mark.parametrize("stress_index", [-1, 2])
def test_audit_rejects_out_of_range_stress(
    tmp_path: Path,
    stress_index: int,
) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = _valid_row()
    row["stress_index"] = stress_index
    _write_rows(fixture, row)

    assert _error_codes(fixture) == ["INVALID_STRESS"]


def test_audit_rejects_unapproved_and_missing_provenance(tmp_path: Path) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = _valid_row()
    row["review_state"] = "candidate"
    row["rule_ids"] = []
    row["source_ids"] = []
    _write_rows(fixture, row)

    assert set(_error_codes(fixture)) == {
        "EMPTY_FIELD",
        "MISSING_PROVENANCE",
        "UNAPPROVED_ENTRY",
    }


def test_audit_enforces_total_and_category_minimums(tmp_path: Path) -> None:
    fixture = tmp_path / "small.jsonl"
    _write_rows(fixture, _valid_row())

    report = audit_gold_file(
        fixture,
        AuditPolicy(
            minimum_total=2,
            category_minimums={"stress": 2, "vowels": 1},
        ),
    )

    assert [error.code for error in report.errors] == [
        "TOTAL_MINIMUM_NOT_MET",
        "CATEGORY_MINIMUM_NOT_MET",
        "CATEGORY_MINIMUM_NOT_MET",
    ]
    assert [error.message for error in report.errors[1:]] == [
        "stress: 1 < 2",
        "vowels: 0 < 1",
    ]


def test_audit_policy_rejects_negative_minimum_total() -> None:
    with pytest.raises(ValueError, match="minimum_total must be non-negative"):
        AuditPolicy(minimum_total=-1, category_minimums={})


def test_audit_policy_rejects_negative_category_minimum() -> None:
    with pytest.raises(ValueError, match="category minimums must be non-negative"):
        AuditPolicy(minimum_total=0, category_minimums={"stress": -1})


def test_cli_success_returns_zero_and_prints_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([str(GOLD_FIXTURE)])

    assert exit_code == 0
    assert capsys.readouterr().out == "gold-audit: PASS total=300 errors=0\n"


def test_cli_failure_returns_one_and_prints_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text("{not-json}\n", encoding="utf-8")

    exit_code = main([str(fixture)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "INVALID_ROW: line 1:" in output
    assert "TOTAL_MINIMUM_NOT_MET: 1" in output


def test_cli_missing_file_returns_one_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.jsonl"

    exit_code = main([str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == f"FILE_READ_ERROR: cannot read gold file: {missing}\n"
    assert captured.err == ""


def test_cli_invalid_utf8_returns_one_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = tmp_path / "invalid-utf8.jsonl"
    fixture.write_bytes(b"\xff\xfe")

    exit_code = main([str(fixture)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == f"FILE_READ_ERROR: cannot read gold file: {fixture}\n"
    assert captured.err == ""
