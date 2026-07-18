import pytest

from latintts import syllables as syllables_module
from latintts.normalization import normalize_word
from latintts.syllables import syllabify, syllable_ranges


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("ave", ("a", "ve")),
        ("gratia", ("gra", "ti", "a")),
        ("ecce", ("ec", "ce")),
        ("sanctus", ("sanc", "tus")),
        ("patris", ("pa", "tris")),
        ("caelum", ("cae", "lum")),
        ("alleluia", ("al", "le", "lu", "ia")),
        ("cui", ("cu", "i")),
        ("qui", ("qui",)),
        ("poëta", ("po", "ë", "ta")),
    ],
)
def test_syllabify_source_backed_examples(word: str, expected: tuple[str, ...]) -> None:
    assert syllabify(word) == expected


def test_syllable_ranges_are_canonical_code_point_half_open_ranges() -> None:
    assert syllable_ranges("poëta") == ((0, 2), (2, 3), (3, 5))


def test_diaeresis_breaks_u_glide() -> None:
    assert syllabify("qüi") == ("qü", "i")


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("hei", ("hei",)),
        ("mei", ("me", "i")),
        ("heï", ("he", "ï")),
    ],
)
def test_only_canonical_hei_has_the_source_backed_ei_syllable_exception(
    word: str,
    expected: tuple[str, ...],
) -> None:
    assert syllabify(word) == expected


def test_source_backed_syllable_exception_has_an_explicit_named_registry() -> None:
    exceptions = getattr(
        syllables_module,
        "SOURCE_BACKED_SYLLABLE_RANGE_EXCEPTIONS",
        {},
    )

    assert exceptions == {"hei": ((0, 3),)}
    assert "ei" not in syllables_module.DIPHTHONGS


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("qui", ("qui",)),
        ("cui", ("cu", "i")),
        ("quia", ("qui", "a")),
        ("eius", ("e", "ius")),
        ("sequi", ("se", "qui")),
        ("equus", ("e", "quus")),
        ("qüia", ("qü", "ia")),
    ],
)
def test_qu_glide_does_not_turn_a_following_i_into_a_consonant(
    word: str,
    expected: tuple[str, ...],
) -> None:
    assert syllabify(word) == expected


@pytest.mark.parametrize("surface", ["ǣlum", "æ\u0304lum"])
def test_ligature_surface_forms_have_equivalent_canonical_ranges(surface: str) -> None:
    canonical = normalize_word(surface).normalized

    assert canonical == "aēlum"
    assert syllable_ranges(canonical) == ((0, 2), (2, 5))
    assert syllabify(canonical) == ("aē", "lum")


def test_syllable_ranges_rejects_words_without_a_vowel_nucleus() -> None:
    with pytest.raises(ValueError, match="word contains no vowel nucleus"):
        syllable_ranges("brrr")
