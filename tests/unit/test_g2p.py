# ruff: noqa: RUF001

import json
from pathlib import Path

import pytest

from latintts import g2p as g2p_module
from latintts.g2p import (
    G2PExceptionEntry,
    ecclesiastical_g2p,
    load_g2p_exceptions,
)
from latintts.syllables import syllabify


@pytest.mark.parametrize(
    ("word", "syllables", "stress", "ipa", "rule_id"),
    [
        ("caelum", ("cae", "lum"), 0, "ˈt͡ʃe.lum", "c-before-front-vowel"),
        ("ecce", ("ec", "ce"), 0, "ˈet.t͡ʃe", "cc-before-front-vowel"),
        ("descendit", ("de", "scen", "dit"), 1, "deˈʃen.dit", "sc-before-front-vowel"),
        ("regina", ("re", "gi", "na"), 1, "reˈd͡ʒi.na", "g-before-front-vowel"),
        ("regnum", ("re", "gnum"), 0, "ˈre.ɲum", "gn-palatal"),
        ("mihi", ("mi", "hi"), 0, "ˈmi.ki", "h-mihi-nihil"),
        ("gratia", ("gra", "ti", "a"), 0, "ˈɡra.t͡si.a", "ti-before-vowel"),
        ("excelsis", ("ex", "cel", "sis"), 1, "ekˈʃel.sis", "xc-before-front-vowel"),
        ("poëta", ("po", "ë", "ta"), 1, "poˈe.ta", "simple-e"),
    ],
)
def test_source_backed_g2p_rules(
    word: str,
    syllables: tuple[str, ...],
    stress: int,
    ipa: str,
    rule_id: str,
) -> None:
    result = ecclesiastical_g2p(word, syllables, stress)

    assert result.ipa == ipa
    assert rule_id in result.applied_rule_ids
    assert "liber-usualis-1962" in result.source_ids


def test_source_indices_keep_cross_syllable_consonants_on_their_own_side() -> None:
    ecce = ecclesiastical_g2p("ecce", ("ec", "ce"), 0)
    excelsis = ecclesiastical_g2p("excelsis", ("ex", "cel", "sis"), 1)

    assert ecce.phonemes == ("ˈ", "e", "t", ".", "t͡ʃ", "e")
    assert excelsis.phonemes == ("e", "k", ".", "ˈ", "ʃ", "e", "l", ".", "s", "i", "s")


def test_regular_g2p_projects_emitted_phonemes_to_syllables_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = g2p_module._syllable_starts
    call_count = 0

    def counting_syllable_starts(
        syllables: tuple[str, ...],
    ) -> tuple[tuple[int, ...], int]:
        nonlocal call_count
        call_count += 1
        return original(syllables)

    monkeypatch.setattr(g2p_module, "_syllable_starts", counting_syllable_starts)

    result = ecclesiastical_g2p("descendit", ("de", "scen", "dit"), 1, exceptions={})

    assert result.ipa == "deˈʃen.dit"
    assert call_count == 1


@pytest.mark.parametrize(
    ("word", "ipa"),
    [
        ("descendit", "deˈʃen.dit"),
        ("ascendit", "aˈʃen.dit"),
    ],
)
def test_sc_phoneme_follows_the_c_source_index_across_real_syllable_boundaries(
    word: str,
    ipa: str,
) -> None:
    result = ecclesiastical_g2p(word, syllabify(word), 1)

    assert result.ipa == ipa


@pytest.mark.parametrize(
    ("word", "syllables", "ipa", "blocked_rule_id"),
    [
        ("poëta", ("po", "ë", "ta"), "ˈpo.e.ta", "oe-e"),
        ("aüla", ("a", "ü", "la"), "ˈa.u.la", "au-diphthong"),
        ("qüi", ("qü", "i"), "ˈku.i", "qu-before-vowel"),
        ("sangüis", ("san", "gü", "is"), "ˈsan.ɡu.is", "ngu-before-vowel"),
        ("aïa", ("a", "ï", "a"), "ˈa.i.a", "i-consonantal"),
    ],
)
def test_diaeresis_blocks_multigraph_rules(
    word: str,
    syllables: tuple[str, ...],
    ipa: str,
    blocked_rule_id: str,
) -> None:
    result = ecclesiastical_g2p(word, syllables, 0)

    assert result.ipa == ipa
    assert blocked_rule_id not in result.applied_rule_ids


@pytest.mark.parametrize(
    ("word", "syllables", "ipa"),
    [
        ("hostia", ("hos", "ti", "a"), "ˈos.ti.a"),
        ("mixtio", ("mix", "ti", "o"), "ˈmiks.ti.o"),
        ("attia", ("at", "ti", "a"), "ˈat.ti.a"),
    ],
)
def test_ti_rule_is_blocked_after_s_x_or_t(
    word: str,
    syllables: tuple[str, ...],
    ipa: str,
) -> None:
    result = ecclesiastical_g2p(word, syllables, 0)

    assert result.ipa == ipa
    assert "ti-before-vowel" not in result.applied_rule_ids


@pytest.mark.parametrize(
    ("word", "syllables", "ipa", "rule_id"),
    [
        ("chorus", ("cho", "rus"), "ˈko.rus", "ch-hard"),
        ("pharus", ("pha", "rus"), "ˈfa.rus", "ph-f"),
        ("thoma", ("tho", "ma"), "ˈto.ma", "th-t"),
        ("qui", ("qui",), "ˈkwi", "qu-before-vowel"),
        ("sanguis", ("san", "guis"), "ˈsaŋ.ɡwis", "ngu-before-vowel"),
        ("poena", ("poe", "na"), "ˈpe.na", "oe-e"),
        ("lauda", ("lau", "da"), "ˈlau̯.da", "au-diphthong"),
        ("euge", ("eu", "ge"), "ˈeu̯.d͡ʒe", "eu-diphthong"),
        ("ay", ("ay",), "ˈai̯", "ay-diphthong"),
    ],
)
def test_longest_match_multigraph_rules(
    word: str,
    syllables: tuple[str, ...],
    ipa: str,
    rule_id: str,
) -> None:
    result = ecclesiastical_g2p(word, syllables, 0)

    assert result.ipa == ipa
    assert rule_id in result.applied_rule_ids


def test_bare_q_is_a_synthetic_fallback_for_nonstandard_or_incomplete_input() -> None:
    result = ecclesiastical_g2p("q", ("q",), 0)

    assert result.ipa == "ˈk"
    assert result.phonemes == ("ˈ", "k")
    assert result.applied_rule_ids == ("q-hard",)
    assert result.source_ids == ("liber-usualis-1962",)


@pytest.mark.parametrize(
    ("word", "syllables", "ipa", "has_consonantal_i"),
    [
        ("qui", ("qui",), "ˈkwi", False),
        ("cui", ("cu", "i"), "ˈku.i", False),
        ("quia", ("qui", "a"), "ˈkwi.a", False),
        ("eius", ("e", "ius"), "ˈe.jus", True),
        ("sequi", ("se", "qui"), "ˈse.kwi", False),
        ("equus", ("e", "quus"), "ˈe.kwus", False),
        ("qüia", ("qü", "ia"), "ˈku.ja", True),
    ],
)
def test_qu_glide_does_not_trigger_consonantal_i(
    word: str,
    syllables: tuple[str, ...],
    ipa: str,
    has_consonantal_i: bool,
) -> None:
    result = ecclesiastical_g2p(word, syllables, 0)

    assert result.ipa == ipa
    assert ("i-consonantal" in result.applied_rule_ids) is has_consonantal_i


def test_ph_rule_aggregates_its_direct_source_with_simple_rule_sources() -> None:
    result = ecclesiastical_g2p("pharus", ("pha", "rus"), 0)

    assert result.source_ids == (
        "iveson-roman-pronunciation-1964",
        "liber-usualis-1962",
    )


def test_h_is_silent_outside_a_source_backed_exception() -> None:
    result = ecclesiastical_g2p("hora", ("ho", "ra"), 0)

    assert result.ipa == "ˈo.ra"
    assert "h-muted" in result.applied_rule_ids
    assert "h-mihi-nihil" not in result.applied_rule_ids


def test_lookup_key_selects_a_packaged_exception_without_rewriting_the_word() -> None:
    result = ecclesiastical_g2p("mihī", ("mi", "hī"), 0, lookup_key="mihi")

    assert result.ipa == "ˈmi.ki"
    assert result.phonemes == ("ˈ", "m", "i", ".", "k", "i")
    assert result.applied_rule_ids == ("h-mihi-nihil",)


@pytest.mark.parametrize(
    ("word", "syllables", "ipa", "phonemes"),
    [
        (
            "nihildum",
            ("ni", "hil", "dum"),
            "niˈkil.dum",
            ("n", "i", ".", "ˈ", "k", "i", "l", ".", "d", "u", "m"),
        ),
    ],
)
def test_nihil_compounds_use_the_source_backed_h_exception(
    word: str,
    syllables: tuple[str, ...],
    ipa: str,
    phonemes: tuple[str, ...],
) -> None:
    result = ecclesiastical_g2p(word, syllables, 1)

    assert result.ipa == ipa
    assert result.phonemes == phonemes
    assert result.applied_rule_ids == ("h-mihi-nihil",)
    assert result.source_ids == (
        "liber-usualis-1962",
        "perseus-lewis-short",
    )


@pytest.mark.parametrize("word", ["traho", "honor", "herba"])
def test_unrelated_h_words_remain_muted(word: str) -> None:
    syllables = syllabify(word)
    result = ecclesiastical_g2p(word, syllables, 0)

    assert "h-muted" in result.applied_rule_ids
    assert "h-mihi-nihil" not in result.applied_rule_ids


def test_load_g2p_exceptions_reads_the_source_backed_entries() -> None:
    exceptions = load_g2p_exceptions()

    assert set(exceptions) == {"hei", "mihi", "nihil", "nihildum"}
    assert exceptions["hei"] == G2PExceptionEntry(
        lookup_key="hei",
        phonemes_by_syllable=(("e", "i̯"),),
        rule_ids=("hei-ei-diphthong",),
        source_ids=("liber-usualis-1962",),
        note=(
            "Liber Usualis PDF lines 1281-1291: ei is treated as one syllable only "
            "in the interjection hei; both vowels are heard and the first has principal emphasis"
        ),
    )
    assert exceptions["mihi"] == G2PExceptionEntry(
        lookup_key="mihi",
        phonemes_by_syllable=(("m", "i"), ("k", "i")),
        rule_ids=("h-mihi-nihil",),
        source_ids=("liber-usualis-1962",),
        note="Liber Usualis pronunciation table, h pronounced k in mihi",
    )
    assert exceptions["nihildum"] == G2PExceptionEntry(
        lookup_key="nihildum",
        phonemes_by_syllable=(("n", "i"), ("k", "i", "l"), ("d", "u", "m")),
        rule_ids=("h-mihi-nihil",),
        source_ids=("liber-usualis-1962", "perseus-lewis-short"),
        note=(
            "Liber Usualis PDF lines 1319-1321: mihi/nihil and their compounds; "
            "Lewis and Short entryFree id=n30955, key=nihildum"
        ),
    )
    assert exceptions["nihil"] == G2PExceptionEntry(
        lookup_key="nihil",
        phonemes_by_syllable=(("n", "i"), ("k", "i", "l")),
        rule_ids=("h-mihi-nihil",),
        source_ids=("liber-usualis-1962",),
        note="Liber Usualis pronunciation table, h pronounced k in nihil",
    )


def test_hei_exception_emits_audible_e_and_nonsyllabic_i() -> None:
    result = ecclesiastical_g2p("hei", ("hei",), 0)

    assert result.ipa == "ˈei̯"
    assert result.phonemes == ("ˈ", "e", "i̯")
    assert result.applied_rule_ids == ("hei-ei-diphthong",)
    assert result.source_ids == ("liber-usualis-1962",)


def _install_exception_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rows: list[dict[str, object]],
) -> None:
    resource = tmp_path / "g2p_exceptions.jsonl"
    resource.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(g2p_module, "files", lambda _package: tmp_path)
    monkeypatch.setattr(
        g2p_module,
        "load_source_registry",
        lambda: {"liber-usualis-1962": object()},
    )


def _valid_exception_row() -> dict[str, object]:
    return {
        "lookup_key": "mihi",
        "phonemes_by_syllable": [["m", "i"], ["k", "i"]],
        "rule_ids": ["h-mihi-nihil"],
        "source_ids": ["liber-usualis-1962"],
        "note": "source-backed exception",
    }


def test_exception_loader_rejects_duplicate_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _valid_exception_row()
    _install_exception_rows(monkeypatch, tmp_path, [row, row])

    with pytest.raises(ValueError, match="duplicate G2P exception key at line 2: mihi"):
        load_g2p_exceptions()


def test_exception_loader_rejects_unknown_source_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _valid_exception_row()
    row["source_ids"] = ["invented-source"]
    _install_exception_rows(monkeypatch, tmp_path, [row])

    with pytest.raises(
        ValueError,
        match=r"unknown G2P exception sources at line 1: \['invented-source'\]",
    ):
        load_g2p_exceptions()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"note": ""}, "incomplete G2P exception at line 1"),
        ({"phonemes_by_syllable": [["m", "i"], []]}, "invalid phonemes_by_syllable at line 1"),
        ({"phonemes_by_syllable": [["m", "?"], ["k", "i"]]}, "unknown phoneme at line 1"),
        ({"rule_ids": ["invented-rule"]}, "unknown G2P exception rules at line 1"),
        ({"lookup_key": "mi\u0304hi"}, "non-canonical G2P exception key at line 1"),
        ({"lookup_key": "mihiv"}, "invalid lookup G2P exception key at line 1"),
        ({"extra": "field"}, "invalid G2P exception fields at line 1"),
    ],
)
def test_exception_loader_rejects_invalid_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    row = _valid_exception_row()
    row.update(mutation)
    _install_exception_rows(monkeypatch, tmp_path, [row])

    with pytest.raises(ValueError, match=message):
        load_g2p_exceptions()


def test_exception_syllable_count_must_match_current_syllabification() -> None:
    exception = G2PExceptionEntry(
        lookup_key="mihi",
        phonemes_by_syllable=(("m", "i", "k", "i"),),
        rule_ids=("h-mihi-nihil",),
        source_ids=("liber-usualis-1962",),
        note="invalid for this syllabification",
    )

    with pytest.raises(ValueError, match="G2P exception syllable mismatch: mihi"):
        ecclesiastical_g2p(
            "mihi",
            ("mi", "hi"),
            0,
            exceptions={"mihi": exception},
        )


@pytest.mark.parametrize("stress_index", [-1, 2])
def test_stress_index_must_point_to_an_existing_syllable(stress_index: int) -> None:
    with pytest.raises(ValueError, match="stress_index must point to an existing syllable"):
        ecclesiastical_g2p("ave", ("a", "ve"), stress_index)


def test_unsupported_grapheme_reports_its_source_index() -> None:
    with pytest.raises(ValueError, match=r"unsupported grapheme at index 1: 'β'"):
        ecclesiastical_g2p("aβ", ("aβ",), 0, exceptions={})


def test_word_must_be_canonical_nfc_and_match_its_syllables() -> None:
    with pytest.raises(ValueError, match="word must be NFC normalized"):
        ecclesiastical_g2p("poe\u0308ta", ("po", "e\u0308", "ta"), 1, exceptions={})

    with pytest.raises(ValueError, match="syllables must reconstruct word"):
        ecclesiastical_g2p("ave", ("a", "va"), 0, exceptions={})
