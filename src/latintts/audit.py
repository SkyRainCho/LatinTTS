from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from latintts.sources import load_source_registry


@dataclass(frozen=True, slots=True)
class GoldEntry:
    word: str
    normalized: str
    syllables: tuple[str, ...]
    stress_index: int
    ipa: str
    category: str
    rule_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    review_state: str


@dataclass(frozen=True, slots=True)
class AuditPolicy:
    minimum_total: int
    category_minimums: Mapping[str, int]

    def __post_init__(self) -> None:
        if type(self.minimum_total) is not int or self.minimum_total < 0:
            raise ValueError("minimum_total must be non-negative")
        if any(
            type(minimum) is not int or minimum < 0 for minimum in self.category_minimums.values()
        ):
            raise ValueError("category minimums must be non-negative")


@dataclass(frozen=True, slots=True)
class AuditError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AuditReport:
    total: int
    category_counts: Mapping[str, int]
    errors: tuple[AuditError, ...]


DEFAULT_POLICY = AuditPolicy(minimum_total=30, category_minimums={})

_GOLD_FIELDS = frozenset(
    {
        "word",
        "normalized",
        "syllables",
        "stress_index",
        "ipa",
        "category",
        "rule_ids",
        "source_ids",
        "review_state",
    }
)


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise TypeError(f"{field} must be a list of strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicate items")
    return result


def _parse_entry(raw: object) -> GoldEntry:
    if type(raw) is not dict or set(raw) != _GOLD_FIELDS:
        raise ValueError("row must be an object with the exact gold fields")

    text_fields = ("word", "normalized", "ipa", "category", "review_state")
    if any(type(raw[field]) is not str for field in text_fields):
        raise TypeError("text fields must be strings")
    if type(raw["stress_index"]) is not int:
        raise TypeError("stress_index must be an integer")

    return GoldEntry(
        word=raw["word"],
        normalized=raw["normalized"],
        syllables=_string_tuple(raw["syllables"], field="syllables"),
        stress_index=raw["stress_index"],
        ipa=raw["ipa"],
        category=raw["category"],
        rule_ids=_string_tuple(raw["rule_ids"], field="rule_ids"),
        source_ids=_string_tuple(raw["source_ids"], field="source_ids"),
        review_state=raw["review_state"],
    )


def audit_gold_file(path: Path, policy: AuditPolicy = DEFAULT_POLICY) -> AuditReport:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return AuditReport(
            total=0,
            category_counts={},
            errors=(AuditError("FILE_READ_ERROR", f"cannot read gold file: {path}"),),
        )

    known_sources = set(load_source_registry())
    seen_words: set[str] = set()
    counts: Counter[str] = Counter()
    errors: list[AuditError] = []
    total = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        total += 1
        try:
            entry = _parse_entry(json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(AuditError("INVALID_ROW", f"line {line_number}: {error}"))
            continue

        if entry.word in seen_words:
            errors.append(AuditError("DUPLICATE_WORD", entry.word))
        seen_words.add(entry.word)

        text_values = (
            entry.word,
            entry.normalized,
            entry.ipa,
            entry.category,
            entry.review_state,
        )
        list_values = (entry.syllables, entry.rule_ids, entry.source_ids)
        if any(not value.strip() for value in text_values) or any(
            not values or any(not value.strip() for value in values) for values in list_values
        ):
            errors.append(AuditError("EMPTY_FIELD", f"line {line_number}"))
        if not 0 <= entry.stress_index < len(entry.syllables):
            errors.append(AuditError("INVALID_STRESS", entry.word))
        if entry.review_state != "approved":
            errors.append(AuditError("UNAPPROVED_ENTRY", entry.word))
        for source_id in sorted(set(entry.source_ids) - known_sources):
            errors.append(AuditError("UNKNOWN_SOURCE", f"{entry.word}: {source_id}"))
        if not entry.rule_ids or not entry.source_ids:
            errors.append(AuditError("MISSING_PROVENANCE", entry.word))
        counts[entry.category] += 1

    if total < policy.minimum_total:
        errors.append(AuditError("TOTAL_MINIMUM_NOT_MET", str(total)))
    for category, minimum in policy.category_minimums.items():
        if counts[category] < minimum:
            errors.append(
                AuditError(
                    "CATEGORY_MINIMUM_NOT_MET",
                    f"{category}: {counts[category]} < {minimum}",
                )
            )
    return AuditReport(total, dict(counts), tuple(errors))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the LatinTTS pronunciation gold set")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    report = audit_gold_file(args.path)
    if report.errors:
        for error in report.errors:
            print(f"{error.code}: {error.message}")
        return 1
    print(f"gold-audit: PASS total={report.total} errors=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
