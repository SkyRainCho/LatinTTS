from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from itertools import pairwise
from types import MappingProxyType
from typing import Final

VOWELS = frozenset("aeiouyāēīōūȳ")
DIPHTHONGS = frozenset({"ae", "oe", "au", "eu", "ay"})
SyllableRanges = tuple[tuple[int, int], ...]
SOURCE_BACKED_SYLLABLE_RANGE_EXCEPTIONS: Final[Mapping[str, SyllableRanges]] = MappingProxyType(
    {
        # Liber Usualis PDF lines 1281-1291 limits one-syllable ei to this interjection.
        "hei": ((0, 3),),
    }
)
ONSET_CLUSTERS = frozenset(
    {
        "bl",
        "br",
        "cl",
        "cr",
        "dr",
        "fl",
        "fr",
        "gl",
        "gr",
        "pl",
        "pr",
        "tr",
        "qu",
        "gu",
        "ch",
        "ph",
        "th",
        "gn",
    }
)


def _base_letter(char: str) -> str:
    return unicodedata.normalize("NFD", char)[0]


def _has_diaeresis(char: str) -> bool:
    return "\u0308" in unicodedata.normalize("NFD", char)


def _is_vowel_at(word: str, index: int) -> bool:
    return 0 <= index < len(word) and _base_letter(word[index]) in VOWELS


def _is_consonantal_i(word: str, index: int) -> bool:
    if (
        _base_letter(word[index]) != "i"
        or _has_diaeresis(word[index])
        or not _is_vowel_at(word, index + 1)
    ):
        return False
    previous_is_vowel_nucleus = _is_vowel_at(word, index - 1) and not _is_glide_u(word, index - 1)
    return index == 0 or previous_is_vowel_nucleus


def _is_glide_u(word: str, index: int) -> bool:
    if (
        _base_letter(word[index]) != "u"
        or _has_diaeresis(word[index])
        or not _is_vowel_at(word, index + 1)
    ):
        return False
    return word[max(0, index - 2) : index] == "ng" or word[index - 1 : index] == "q"


def _is_diphthong_at(word: str, index: int) -> bool:
    if index + 1 >= len(word):
        return False
    first, second = word[index], word[index + 1]
    if _has_diaeresis(first) or _has_diaeresis(second):
        return False
    return _base_letter(first) + _base_letter(second) in DIPHTHONGS


def _find_nuclei(word: str) -> tuple[tuple[int, int], ...]:
    nuclei: list[tuple[int, int]] = []
    index = 0
    while index < len(word):
        if _is_diphthong_at(word, index):
            nuclei.append((index, index + 2))
            index += 2
            continue
        if (
            _is_vowel_at(word, index)
            and not _is_consonantal_i(word, index)
            and not _is_glide_u(word, index)
        ):
            nuclei.append((index, index + 1))
        index += 1
    return tuple(nuclei)


def _onset_length(cluster: str) -> int:
    for size in (3, 2):
        if len(cluster) >= size and cluster[-size:] in ONSET_CLUSTERS:
            return size
    return 1


def syllable_ranges(word: str) -> tuple[tuple[int, int], ...]:
    exception = SOURCE_BACKED_SYLLABLE_RANGE_EXCEPTIONS.get(word)
    if exception is not None:
        return exception

    nuclei = _find_nuclei(word)
    if not nuclei:
        raise ValueError("word contains no vowel nucleus")
    starts = [0]
    for previous, current in pairwise(nuclei):
        cluster = word[previous[1] : current[0]]
        boundary = current[0] if not cluster else current[0] - _onset_length(cluster)
        starts.append(max(previous[1], boundary))
    ends = [*starts[1:], len(word)]
    return tuple(zip(starts, ends, strict=True))


def syllabify(word: str) -> tuple[str, ...]:
    return tuple(word[start:end] for start, end in syllable_ranges(word))
