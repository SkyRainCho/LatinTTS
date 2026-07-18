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
            minimum_total=30,
            category_minimums={
                "vowels": 5,
                "diphthongs": 5,
                "consonants": 10,
                "stress": 5,
                "liturgical": 5,
            },
        ),
    )

    assert report.errors == ()
    assert report.total == 30
    assert report.category_counts == {
        "vowels": 5,
        "diphthongs": 5,
        "consonants": 10,
        "stress": 5,
        "liturgical": 5,
    }


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
    assert capsys.readouterr().out == "gold-audit: PASS total=30 errors=0\n"


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
