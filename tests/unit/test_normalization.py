from latintts.normalization import normalize_phrase, normalize_word, tokenize_words


def test_ligatures_expand_without_losing_original_span() -> None:
    text = "Cælum et cœli"
    spans = tokenize_words(text)
    assert [(item.surface, item.start, item.end) for item in spans] == [
        ("Cælum", 0, 5),
        ("et", 6, 8),
        ("cœli", 9, 13),
    ]
    assert normalize_word(spans[0].surface).normalized == "caelum"
    assert normalize_word(spans[2].surface).normalized == "coeli"


def test_lookup_key_unifies_variants_but_normalized_text_preserves_roles() -> None:
    assert normalize_word("jam").normalized == "jam"
    assert normalize_word("iam").normalized == "iam"
    assert normalize_word("jam").lookup_key == normalize_word("iam").lookup_key
    assert normalize_word("servus").lookup_key == "seruus"


def test_normalization_records_stable_transformation_ids() -> None:
    assert normalize_word("Cælum").transformations == (
        "casefold",
        "expand-ae-ligature",
    )
    assert "lookup-j-to-i" in normalize_word("jam").transformations
    assert "lookup-v-to-u" in normalize_word("servus").transformations


def test_acute_mark_becomes_explicit_stress_hint() -> None:
    word = normalize_word("Dóminus")
    assert word.normalized == "dominus"
    assert word.marked_vowel_index == 1


def test_phrase_normalization_is_stable_and_keeps_punctuation() -> None:
    assert normalize_phrase("  Ave\tMaria,\n  gratia. ") == "Ave Maria, gratia."
