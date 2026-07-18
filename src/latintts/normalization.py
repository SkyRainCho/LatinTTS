from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WordSpan:
    surface: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NormalizedWord:
    surface: str
    normalized: str
    lookup_key: str
    marked_vowel_index: int | None
    transformations: tuple[str, ...]


COMBINING_ACUTE = "\u0301"


def tokenize_words(text: str) -> tuple[WordSpan, ...]:
    result: list[WordSpan] = []
    start: int | None = None
    for index, char in enumerate(text):
        is_mark = unicodedata.category(char).startswith("M")
        is_word_char = char.isalpha() or (start is not None and is_mark)
        if is_word_char and start is None:
            start = index
        elif not is_word_char and start is not None:
            result.append(WordSpan(text[start:index], start, index))
            start = None
    if start is not None:
        result.append(WordSpan(text[start:], start, len(text)))
    return tuple(result)


def normalize_phrase(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _lookup_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    mapped = "".join(
        char.replace("j", "i").replace("v", "u")
        for char in decomposed
        if char != COMBINING_ACUTE
    )
    return unicodedata.normalize("NFC", mapped)


def _describe_transformations(surface: str, normalized: str) -> tuple[str, ...]:
    result: list[str] = []
    nfc_surface = unicodedata.normalize("NFC", surface)
    folded_surface = nfc_surface.casefold()
    decomposed_surface = unicodedata.normalize("NFD", folded_surface)
    decomposed_normalized = unicodedata.normalize("NFD", normalized)
    if nfc_surface != surface:
        result.append("unicode-nfc")
    if folded_surface != nfc_surface:
        result.append("casefold")
    if "æ" in decomposed_surface:
        result.append("expand-ae-ligature")
    if "œ" in decomposed_surface:
        result.append("expand-oe-ligature")
    if COMBINING_ACUTE in decomposed_surface:
        result.append("remove-acute-stress-mark")
    if "j" in decomposed_normalized:
        result.append("lookup-j-to-i")
    if "v" in decomposed_normalized:
        result.append("lookup-v-to-u")
    return tuple(result)


def normalize_word(surface: str) -> NormalizedWord:
    folded = unicodedata.normalize("NFC", surface).casefold()
    decomposed = unicodedata.normalize("NFD", folded)
    output: list[str] = []
    base_index = -1
    marked_vowel_index: int | None = None
    for char in decomposed:
        if unicodedata.category(char).startswith("M"):
            if char == COMBINING_ACUTE:
                marked_vowel_index = base_index
            else:
                output.append(char)
            continue
        expanded = char.replace("æ", "ae").replace("œ", "oe")
        base_index += len(expanded)
        output.extend(expanded)
    normalized = unicodedata.normalize("NFC", "".join(output))
    return NormalizedWord(
        surface=surface,
        normalized=normalized,
        lookup_key=_lookup_key(normalized),
        marked_vowel_index=marked_vowel_index,
        transformations=_describe_transformations(surface, normalized),
    )
