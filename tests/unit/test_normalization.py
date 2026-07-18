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


def test_lookup_key_preserves_non_acute_marks() -> None:
    assert normalize_word("māter").lookup_key == "māter"
    assert normalize_word("aër").lookup_key == "aër"
    assert normalize_word("ae\u0308r").lookup_key == "aër"


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


def test_marked_ligature_expansion_preserves_trailing_non_acute_marks() -> None:
    precomposed = normalize_word("ǣ")
    decomposed = normalize_word("æ\u0304")
    spans = tokenize_words("ǣ æ\u0304")

    assert [(item.surface, item.start, item.end) for item in spans] == [
        ("ǣ", 0, 1),
        ("æ\u0304", 2, 4),
    ]
    assert precomposed.surface == "ǣ"
    assert decomposed.surface == "æ\u0304"
    assert precomposed.normalized == decomposed.normalized == "aē"
    assert precomposed.lookup_key == decomposed.lookup_key == "aē"
    assert precomposed.marked_vowel_index is decomposed.marked_vowel_index is None
    assert "expand-ae-ligature" in precomposed.transformations
    assert "expand-ae-ligature" in decomposed.transformations

    oe_with_diaeresis = normalize_word("œ\u0308")
    assert oe_with_diaeresis.normalized == "oë"
    assert oe_with_diaeresis.lookup_key == "oë"
    assert "expand-oe-ligature" in oe_with_diaeresis.transformations


def test_acute_marked_ligature_expansion_targets_trailing_letter() -> None:
    precomposed = normalize_word("ǽ")
    decomposed = normalize_word("æ\u0301")

    assert precomposed.normalized == decomposed.normalized == "ae"
    assert precomposed.lookup_key == decomposed.lookup_key == "ae"
    assert precomposed.marked_vowel_index == decomposed.marked_vowel_index == 1
    assert "expand-ae-ligature" in precomposed.transformations
    assert "expand-ae-ligature" in decomposed.transformations
    assert "remove-acute-stress-mark" in precomposed.transformations
    assert "remove-acute-stress-mark" in decomposed.transformations


def test_phrase_normalization_is_stable_and_keeps_punctuation() -> None:
    assert normalize_phrase("  Ave\tMaria,\n  gratia. ") == "Ave Maria, gratia."
