# ruff: noqa: RUF001

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files

from latintts.domain import Diagnostic
from latintts.sources import load_source_registry


@dataclass(frozen=True, slots=True)
class G2PExceptionEntry:
    lookup_key: str
    phonemes_by_syllable: tuple[tuple[str, ...], ...]
    rule_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    note: str


@dataclass(frozen=True, slots=True)
class G2PResult:
    ipa: str
    phonemes: tuple[str, ...]
    applied_rule_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    warnings: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    pattern: re.Pattern[str]
    phonemes: tuple[str, ...]
    # One placement anchor per emitted phoneme, not a list of every consumed
    # grapheme. Anchors decide which syllable receives a fused phoneme.
    source_offsets: tuple[int, ...]
    source_ids: tuple[str, ...] = ("liber-usualis-1962",)


FRONT = r"(?:ae|oe|e|i|y)"
VOWEL = r"(?:ae|oe|[aeiouy])"
RULES = (
    Rule("xc-before-front-vowel", re.compile(rf"xc(?={FRONT})"), ("k", "ʃ"), (0, 1)),
    Rule("cc-before-front-vowel", re.compile(rf"cc(?={FRONT})"), ("t", "t͡ʃ"), (0, 1)),
    # Anchor fused /ʃ/ to c so a split such as nes-cio places it in the onset
    # of the second syllable; offset 0 would incorrectly attach it to nes-.
    Rule("sc-before-front-vowel", re.compile(rf"sc(?={FRONT})"), ("ʃ",), (1,)),
    Rule("gn-palatal", re.compile(r"gn"), ("ɲ",), (0,)),
    Rule("ti-before-vowel", re.compile(rf"(?<![sxt])ti(?={VOWEL})"), ("t͡s", "i"), (0, 1)),
    Rule("ch-hard", re.compile(r"ch"), ("k",), (0,)),
    Rule(
        "ph-f",
        re.compile(r"ph"),
        ("f",),
        (0,),
        ("iveson-roman-pronunciation-1964",),
    ),
    Rule("th-t", re.compile(r"th"), ("t",), (0,)),
    Rule("qu-before-vowel", re.compile(rf"qu(?={VOWEL})"), ("k", "w"), (0, 1)),
    Rule("ngu-before-vowel", re.compile(rf"ngu(?={VOWEL})"), ("ŋ", "ɡ", "w"), (0, 1, 2)),
    Rule("ae-e", re.compile(r"ae"), ("e",), (0,)),
    Rule("oe-e", re.compile(r"oe"), ("e",), (0,)),
    Rule("au-diphthong", re.compile(r"au"), ("a", "u̯"), (0, 1)),
    Rule("eu-diphthong", re.compile(r"eu"), ("e", "u̯"), (0, 1)),
    Rule("ay-diphthong", re.compile(r"ay"), ("a", "i̯"), (0, 1)),
    Rule(
        "i-consonantal",
        re.compile(r"(?<![^aeiouy])i(?=[aeiouy])"),
        ("j",),
        (0,),
    ),
    Rule("c-before-front-vowel", re.compile(rf"c(?={FRONT})"), ("t͡ʃ",), (0,)),
    Rule("g-before-front-vowel", re.compile(rf"g(?={FRONT})"), ("d͡ʒ",), (0,)),
)

DIAERESIS_BREAK_RULE_IDS = frozenset(
    {
        "ae-e",
        "oe-e",
        "au-diphthong",
        "eu-diphthong",
        "ay-diphthong",
        "i-consonantal",
        "qu-before-vowel",
        "ngu-before-vowel",
    }
)

SIMPLE_PHONEMES: dict[str, tuple[str, ...]] = {
    "a": ("a",),
    "b": ("b",),
    "c": ("k",),
    "d": ("d",),
    "e": ("e",),
    "f": ("f",),
    "g": ("ɡ",),
    "h": (),
    "i": ("i",),
    "j": ("j",),
    "k": ("k",),
    "l": ("l",),
    "m": ("m",),
    "n": ("n",),
    "o": ("o",),
    "p": ("p",),
    "q": ("k",),
    "r": ("r",),
    "s": ("s",),
    "t": ("t",),
    "u": ("u",),
    "v": ("v",),
    "x": ("k", "s"),
    "y": ("i",),
    "z": ("d͡z",),
}

SIMPLE_RULE_IDS = {
    "a": "simple-a",
    "b": "simple-b",
    "c": "c-hard",
    "d": "simple-d",
    "e": "simple-e",
    "f": "simple-f",
    "g": "g-hard",
    "h": "h-muted",
    "i": "simple-i",
    "j": "j-consonantal",
    "k": "simple-k",
    "l": "simple-l",
    "m": "simple-m",
    "n": "simple-n",
    "o": "simple-o",
    "p": "simple-p",
    "q": "q-hard",
    "r": "simple-r",
    "s": "simple-s",
    "t": "simple-t",
    "u": "simple-u",
    "v": "simple-v",
    "x": "x-ks",
    "y": "y-as-i",
    "z": "z-dz",
}

EXCEPTION_RULE_IDS = frozenset({"h-mihi-nihil", "hei-ei-diphthong"})

IMPLEMENTED_RULE_IDS = frozenset(
    {rule.rule_id for rule in RULES} | set(SIMPLE_RULE_IDS.values()) | EXCEPTION_RULE_IDS
)

_RULE_SOURCE_IDS = {
    **{rule.rule_id: rule.source_ids for rule in RULES},
    **{rule_id: ("liber-usualis-1962",) for rule_id in SIMPLE_RULE_IDS.values()},
}

PHONEME_INVENTORY = frozenset(
    {
        "a",
        "b",
        "d",
        "d͡ʒ",
        "d͡z",
        "e",
        "f",
        "ɡ",
        "i",
        "i̯",
        "j",
        "k",
        "l",
        "m",
        "n",
        "ɲ",
        "ŋ",
        "o",
        "p",
        "r",
        "s",
        "ʃ",
        "t",
        "t͡ʃ",
        "t͡s",
        "u",
        "u̯",
        "v",
        "w",
    }
)

_EXCEPTION_FIELDS = frozenset(
    {"lookup_key", "phonemes_by_syllable", "rule_ids", "source_ids", "note"}
)


def _parse_required_unique_strings(
    value: object,
    *,
    field: str,
    line_number: int,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"invalid {field} at line {line_number}")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {field} at line {line_number}")
    return result


def _exception_key_is_valid(lookup_key: str) -> bool:
    decomposed = unicodedata.normalize("NFD", lookup_key)
    return (
        bool(lookup_key)
        and lookup_key == lookup_key.casefold()
        and "\u0301" not in decomposed
        and "j" not in decomposed
        and "v" not in decomposed
        and "æ" not in lookup_key
        and "œ" not in lookup_key
        and all(char.isalpha() or unicodedata.category(char).startswith("M") for char in lookup_key)
    )


def load_g2p_exceptions() -> dict[str, G2PExceptionEntry]:
    text = files("latintts.resources").joinpath("g2p_exceptions.jsonl").read_text(encoding="utf-8")
    known_sources = set(load_source_registry())
    result: dict[str, G2PExceptionEntry] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or set(raw) != _EXCEPTION_FIELDS:
            raise ValueError(f"invalid G2P exception fields at line {line_number}")

        lookup_key = raw["lookup_key"]
        note = raw["note"]
        if not isinstance(lookup_key, str) or not isinstance(note, str):
            raise ValueError(f"invalid G2P exception fields at line {line_number}")
        if lookup_key != unicodedata.normalize("NFC", lookup_key):
            raise ValueError(f"non-canonical G2P exception key at line {line_number}")
        if not _exception_key_is_valid(lookup_key):
            raise ValueError(f"invalid lookup G2P exception key at line {line_number}")

        raw_phonemes = raw["phonemes_by_syllable"]
        if not isinstance(raw_phonemes, list) or not raw_phonemes:
            raise ValueError(f"invalid phonemes_by_syllable at line {line_number}")
        phonemes_by_syllable: list[tuple[str, ...]] = []
        for part in raw_phonemes:
            if (
                not isinstance(part, list)
                or not part
                or any(not isinstance(phoneme, str) or not phoneme for phoneme in part)
            ):
                raise ValueError(f"invalid phonemes_by_syllable at line {line_number}")
            phonemes_by_syllable.append(tuple(part))

        rule_ids = _parse_required_unique_strings(
            raw["rule_ids"], field="rule_ids", line_number=line_number
        )
        source_ids = _parse_required_unique_strings(
            raw["source_ids"], field="source_ids", line_number=line_number
        )
        entry = G2PExceptionEntry(
            lookup_key=lookup_key,
            phonemes_by_syllable=tuple(phonemes_by_syllable),
            rule_ids=rule_ids,
            source_ids=source_ids,
            note=note,
        )
        if entry.lookup_key in result:
            raise ValueError(
                f"duplicate G2P exception key at line {line_number}: {entry.lookup_key}"
            )
        unknown_sources = set(entry.source_ids) - known_sources
        if unknown_sources:
            raise ValueError(
                f"unknown G2P exception sources at line {line_number}: {sorted(unknown_sources)}"
            )
        if not entry.rule_ids or not entry.source_ids or not entry.note.strip():
            raise ValueError(f"incomplete G2P exception at line {line_number}")
        unknown_rules = set(entry.rule_ids) - IMPLEMENTED_RULE_IDS
        if unknown_rules:
            raise ValueError(
                f"unknown G2P exception rules at line {line_number}: {sorted(unknown_rules)}"
            )
        unknown_phonemes = {
            phoneme
            for part in entry.phonemes_by_syllable
            for phoneme in part
            if phoneme not in PHONEME_INVENTORY
        }
        if unknown_phonemes:
            raise ValueError(f"unknown phoneme at line {line_number}: {sorted(unknown_phonemes)}")
        result[entry.lookup_key] = entry
    return result


def _is_after_glide_u(
    word: str,
    index: int,
    diaeresis_indices: set[int],
) -> bool:
    if index == 0 or word[index - 1] != "u" or index - 1 in diaeresis_indices:
        return False
    return word[max(0, index - 2) : index - 1] == "q" or word[max(0, index - 3) : index - 1] == "ng"


def _scan(word: str) -> tuple[tuple[tuple[int, str], ...], tuple[str, ...]]:
    folded = "".join(unicodedata.normalize("NFD", char)[0] for char in word)
    diaeresis_indices = {
        index for index, char in enumerate(word) if "\u0308" in unicodedata.normalize("NFD", char)
    }
    emitted: list[tuple[int, str]] = []
    applied: list[str] = []
    index = 0
    while index < len(folded):
        matched = False
        for rule in RULES:
            if rule.rule_id == "i-consonantal" and _is_after_glide_u(
                folded, index, diaeresis_indices
            ):
                continue
            result = rule.pattern.match(folded, index)
            if result is None:
                continue
            if rule.rule_id in DIAERESIS_BREAK_RULE_IDS and any(
                position in diaeresis_indices for position in range(index, result.end())
            ):
                continue
            emitted.extend(
                (index + offset, phoneme)
                for offset, phoneme in zip(rule.source_offsets, rule.phonemes, strict=True)
            )
            applied.append(rule.rule_id)
            index = result.end()
            matched = True
            break
        if matched:
            continue
        phonemes = SIMPLE_PHONEMES.get(folded[index])
        if phonemes is None:
            raise ValueError(f"unsupported grapheme at index {index}: {folded[index]!r}")
        emitted.extend((index, phoneme) for phoneme in phonemes)
        applied.append(SIMPLE_RULE_IDS[folded[index]])
        index += 1
    return tuple(emitted), tuple(applied)


def _syllable_starts(syllables: tuple[str, ...]) -> tuple[tuple[int, ...], int]:
    starts: list[int] = []
    cursor = 0
    for syllable in syllables:
        starts.append(cursor)
        cursor += len(syllable)
    return tuple(starts), cursor


def _render_ipa(
    syllables: tuple[str, ...],
    stress_index: int,
    emitted: tuple[tuple[int, str], ...],
) -> str:
    starts, cursor = _syllable_starts(syllables)
    phonemes_by_syllable: list[tuple[str, ...]] = []
    for syllable_index, start in enumerate(starts):
        end = starts[syllable_index + 1] if syllable_index + 1 < len(starts) else cursor
        phonemes_by_syllable.append(
            tuple(phoneme for source_index, phoneme in emitted if start <= source_index < end)
        )
    return _render_ipa_syllables(tuple(phonemes_by_syllable), stress_index)


def _render_ipa_syllables(
    phonemes_by_syllable: tuple[tuple[str, ...], ...],
    stress_index: int,
) -> str:
    rendered: list[str] = []
    for index, syllable_phonemes in enumerate(phonemes_by_syllable):
        if index and index != stress_index:
            rendered.append(".")
        if index == stress_index:
            rendered.append("ˈ")
        rendered.extend(syllable_phonemes)
    return "".join(rendered)


def _phonemes_by_syllable(
    syllables: tuple[str, ...],
    emitted: tuple[tuple[int, str], ...],
) -> tuple[tuple[str, ...], ...]:
    starts, cursor = _syllable_starts(syllables)
    result: list[tuple[str, ...]] = []
    for syllable_index, start in enumerate(starts):
        end = starts[syllable_index + 1] if syllable_index + 1 < len(starts) else cursor
        result.append(
            tuple(phoneme for source_index, phoneme in emitted if start <= source_index < end)
        )
    return tuple(result)


def _render_model_phonemes(
    phonemes_by_syllable: tuple[tuple[str, ...], ...],
    stress_index: int,
) -> tuple[str, ...]:
    rendered: list[str] = []
    for index, syllable_phonemes in enumerate(phonemes_by_syllable):
        if index:
            rendered.append(".")
        if index == stress_index:
            rendered.append("ˈ")
        rendered.extend(syllable_phonemes)
    return tuple(rendered)


def ecclesiastical_g2p(
    word: str,
    syllables: tuple[str, ...],
    stress_index: int,
    *,
    lookup_key: str | None = None,
    exceptions: Mapping[str, G2PExceptionEntry] | None = None,
) -> G2PResult:
    """Convert one canonical word to G2P output.

    Passing ``exceptions`` avoids resource I/O for repeated direct calls.
    ``Pronouncer`` loads the exception mapping once per instance and always
    supplies it here.
    """
    if not 0 <= stress_index < len(syllables):
        raise ValueError("stress_index must point to an existing syllable")
    if word != unicodedata.normalize("NFC", word):
        raise ValueError("word must be NFC normalized")
    if not all(syllables) or "".join(syllables) != word:
        raise ValueError("syllables must reconstruct word")

    exception_key = lookup_key if lookup_key is not None else word
    exception = (exceptions if exceptions is not None else load_g2p_exceptions()).get(exception_key)
    if exception is not None:
        if len(exception.phonemes_by_syllable) != len(syllables):
            raise ValueError(f"G2P exception syllable mismatch: {word}")
        model_phonemes = _render_model_phonemes(
            exception.phonemes_by_syllable,
            stress_index,
        )
        return G2PResult(
            ipa=_render_ipa_syllables(exception.phonemes_by_syllable, stress_index),
            phonemes=model_phonemes,
            applied_rule_ids=exception.rule_ids,
            source_ids=exception.source_ids,
        )

    emitted, applied = _scan(word)
    phonemes_by_syllable = _phonemes_by_syllable(syllables, emitted)
    model_phonemes = _render_model_phonemes(phonemes_by_syllable, stress_index)
    source_ids = tuple(
        dict.fromkeys(source_id for rule_id in applied for source_id in _RULE_SOURCE_IDS[rule_id])
    )
    return G2PResult(
        ipa=_render_ipa(syllables, stress_index, emitted),
        phonemes=model_phonemes,
        applied_rule_ids=tuple(dict.fromkeys(applied)),
        source_ids=source_ids,
    )
