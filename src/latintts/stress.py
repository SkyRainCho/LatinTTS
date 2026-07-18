from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files

from latintts.domain import Diagnostic, ResolutionMethod, Severity
from latintts.sources import load_source_registry

MACRON_VOWELS = frozenset("āēīōūȳ")
PLAIN_VOWELS = frozenset("aeiouy")
STRESS_SOURCE = ("allen-greenough-accents",)
ENCLITICS = ("que", "ne", "ve")


@dataclass(frozen=True, slots=True)
class StressLexiconEntry:
    lookup_key: str
    syllables: tuple[str, ...]
    stress_index: int
    source_ids: tuple[str, ...]
    note: str
    is_exception: bool


@dataclass(frozen=True, slots=True)
class StressDecision:
    stress_index: int
    method: ResolutionMethod
    source_ids: tuple[str, ...]
    warnings: tuple[Diagnostic, ...] = ()
    applied_rule_ids: tuple[str, ...] = ()


def load_stress_lexicon() -> dict[str, StressLexiconEntry]:
    text = files("latintts.resources").joinpath("stress_lexicon.jsonl").read_text(
        encoding="utf-8"
    )
    known_sources = set(load_source_registry())
    result: dict[str, StressLexiconEntry] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        entry = StressLexiconEntry(
            lookup_key=raw["lookup_key"],
            syllables=tuple(raw["syllables"]),
            stress_index=raw["stress_index"],
            source_ids=tuple(raw["source_ids"]),
            note=raw["note"],
            is_exception=raw["is_exception"],
        )
        if entry.lookup_key in result:
            raise ValueError(f"duplicate stress key at line {line_number}: {entry.lookup_key}")
        if not 0 <= entry.stress_index < len(entry.syllables):
            raise ValueError(f"invalid stress index at line {line_number}")
        unknown_sources = set(entry.source_ids) - known_sources
        if unknown_sources:
            raise ValueError(
                f"unknown stress sources at line {line_number}: {sorted(unknown_sources)}"
            )
        if not entry.source_ids or not entry.note.strip():
            raise ValueError(f"incomplete stress entry at line {line_number}")
        result[entry.lookup_key] = entry
    return result


def _is_heavy(syllable: str) -> bool:
    folded = "".join(unicodedata.normalize("NFD", char)[0] for char in syllable)
    has_macron = any(char in MACRON_VOWELS for char in syllable)
    has_diphthong = any(pair in folded for pair in ("ae", "oe", "au", "eu", "ay"))
    closes_with_consonant = folded[-1] not in PLAIN_VOWELS
    return has_macron or has_diphthong or closes_with_consonant


def resolve_stress(
    lookup_key: str,
    syllables: tuple[str, ...],
    lexicon: Mapping[str, StressLexiconEntry],
    *,
    explicit_stress_index: int | None = None,
    override_index: int | None = None,
) -> StressDecision:
    def checked(index: int) -> int:
        if not 0 <= index < len(syllables):
            raise ValueError("stress_index must point to an existing syllable")
        return index

    if override_index is not None:
        return StressDecision(
            checked(override_index),
            ResolutionMethod.OVERRIDE,
            (),
            applied_rule_ids=("override-stress",),
        )
    if explicit_stress_index is not None:
        return StressDecision(
            checked(explicit_stress_index),
            ResolutionMethod.OVERRIDE,
            (),
            applied_rule_ids=("explicit-stress",),
        )
    entry = lexicon.get(lookup_key)
    if entry is not None:
        if len(entry.syllables) != len(syllables):
            raise ValueError(f"stress lexicon syllable mismatch: {lookup_key}")
        method = ResolutionMethod.EXCEPTION if entry.is_exception else ResolutionMethod.LEXICON
        rule_id = "stress-exception" if entry.is_exception else "stress-lexicon"
        return StressDecision(
            entry.stress_index,
            method,
            entry.source_ids,
            applied_rule_ids=(rule_id,),
        )
    enclitic_suffix = next(
        (
            suffix
            for suffix in ENCLITICS
            if lookup_key.endswith(suffix) and lookup_key[: -len(suffix)] in lexicon
        ),
        None,
    )
    if len(syllables) >= 2 and enclitic_suffix is not None:
        base_entry = lexicon[lookup_key[: -len(enclitic_suffix)]]
        return StressDecision(
            len(syllables) - 2,
            ResolutionMethod.RULE,
            tuple(dict.fromkeys((*STRESS_SOURCE, *base_entry.source_ids))),
            applied_rule_ids=("enclitic-stress",),
        )
    if len(syllables) == 1:
        return StressDecision(
            0,
            ResolutionMethod.RULE,
            STRESS_SOURCE,
            applied_rule_ids=("monosyllable-stress",),
        )
    if len(syllables) == 2:
        return StressDecision(
            0,
            ResolutionMethod.RULE,
            STRESS_SOURCE,
            applied_rule_ids=("disyllable-stress",),
        )
    penult = len(syllables) - 2
    if _is_heavy(syllables[penult]):
        return StressDecision(
            penult,
            ResolutionMethod.RULE,
            STRESS_SOURCE,
            applied_rule_ids=("heavy-penult-stress",),
        )
    warning = Diagnostic(
        code="PRONUNCIATION_NEEDS_REVIEW",
        message="Open penult quantity is not established by the stress lexicon",
        severity=Severity.WARNING,
    )
    return StressDecision(
        len(syllables) - 3,
        ResolutionMethod.CANDIDATE,
        STRESS_SOURCE,
        (warning,),
        ("candidate-antepenult-stress",),
    )
