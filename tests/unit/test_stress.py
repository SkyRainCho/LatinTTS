import json
from pathlib import Path

import pytest

from latintts import stress as stress_module
from latintts.domain import ResolutionMethod, Severity
from latintts.normalization import normalize_word
from latintts.stress import StressLexiconEntry, load_stress_lexicon, resolve_stress
from latintts.syllables import syllabify

PRONUNCIATION_DOCUMENT = (
    Path(__file__).parents[2] / "docs" / "pronunciation" / "roman-ecclesiastical.md"
)
SeedExpectation = tuple[tuple[str, ...], int, tuple[str, ...], tuple[str, ...]]
SEED_EXPECTATIONS: dict[str, SeedExpectation] = {
    "dominus": (
        ("do", "mi", "nus"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n14699",
            "key=dominus",
            "orth=dŏmĭnus",
            "Allen and Greenough Section 12",
        ),
    ),
    "regina": (
        ("re", "gi", "na"),
        1,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n40899",
            "key=regina",
            "orth=rēgīna",
            "Allen and Greenough Section 12",
        ),
    ),
    "maria": (
        ("ma", "ri", "a"),
        1,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n28037",
            "key=Maria1",
            "orth=Mărī^a",
            "sense I.1 Mary",
            "Allen and Greenough Section 12",
        ),
    ),
    "gratia": (
        ("gra", "ti", "a"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n19896",
            "key=gratia",
            "orth=grātĭa",
            "Allen and Greenough Section 12",
        ),
    ),
    "gloria": (
        ("glo", "ri", "a"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n19675",
            "key=gloria",
            "orth=glōrĭa",
            "Allen and Greenough Section 12",
            "light penult -> antepenult",
        ),
    ),
    "kyrie": (
        ("ky", "ri", "e"),
        0,
        ("liber-usualis-1962",),
        (
            "PDF page 32 (printed xxxviii), R example",
            "prints Kýrie",
        ),
    ),
    "caelum": (
        ("cae", "lum"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n6042",
            "key=caelum2",
            "orth=caelum",
            "Allen and Greenough Section 12, disyllables stress the first syllable",
        ),
    ),
    "alleluia": (
        ("al", "le", "lu", "ia"),
        2,
        (
            "perseus-lewis-short",
            "liber-usualis-1962",
            "allen-greenough-accents",
        ),
        (
            "entryFree id=n1926",
            "key=alleluja",
            "orth=allēlūja",
            "PDF page 32 (printed xxxviii), J example",
            "Latin form is unaccented alleluia",
            "pronunciation approximation allelóoya supplies the acute stress evidence",
            "Allen and Greenough Section 12",
        ),
    ),
    "magnificat": (
        ("ma", "gni", "fi", "cat"),
        1,
        (
            "perseus-lewis-short",
            "liber-usualis-1962",
            "allen-greenough-accents",
        ),
        (
            "entryFree id=n27636",
            "key=magnifico",
            "orth=magnĭfĭco",
            "PDF page 32 (printed xxxviii), GN example",
            "prints Magníficat = Mah-nyee-fee-caht",
            "Allen and Greenough Section 12",
        ),
    ),
    "misericordia": (
        ("mi", "se", "ri", "cor", "di", "a"),
        3,
        (
            "perseus-lewis-short",
            "liber-usualis-1962",
            "allen-greenough-accents",
        ),
        (
            "entryFree id=n29266",
            "key=misericordia",
            "orth=mĭsĕrĭcordĭa",
            "PDF page 32 (printed xxxviii), S example",
            "prints misericórdia",
            "Allen and Greenough Section 12",
        ),
    ),
    "benedictus": (
        ("be", "ne", "dic", "tus"),
        2,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n5170",
            "key=benedico",
            "orth=bĕnĕdīco with principal part ctum",
            "supports the inflected participle benedictus",
            "Allen and Greenough Section 12 rule inference",
            "closed penult dic ends in a consonant",
        ),
    ),
    "excelsis": (
        ("ex", "cel", "sis"),
        1,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n16651",
            "key=excelsus",
            "orth=excelsus with inflection a, um",
            "supports the inflected form excelsis",
            "Allen and Greenough Section 12 rule inference",
            "closed penult cel ends in a consonant",
        ),
    ),
    "anima": (
        ("a", "ni", "ma"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n2612",
            "key=anima",
            "orth=ănĭma",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "animus": (
        ("a", "ni", "mus"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n2636",
            "key=animus",
            "orth=ănĭmus",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "spiritus": (
        ("spi", "ri", "tus"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n45053",
            "key=spiritus",
            "orth=spīrĭtus",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "oculus": (
        ("o", "cu", "lus"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n32239",
            "key=oculus",
            "orth=ŏcŭlus",
            "Allen and Greenough Section 12",
            "light penult ŭ -> antepenult",
        ),
    ),
    "saeculum": (
        ("sae", "cu", "lum"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n42210",
            "key=saeculum",
            "orth=saecŭlum",
            "Allen and Greenough Section 12",
            "light penult ŭ -> antepenult",
        ),
    ),
    "discipulus": (
        ("dis", "ci", "pu", "lus"),
        1,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n14173",
            "key=discipulus",
            "orth=discĭpŭlus",
            "Allen and Greenough Section 12",
            "light penult ŭ -> antepenult",
        ),
    ),
    "angelus": (
        ("an", "ge", "lus"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n2554",
            "key=angelus",
            "orth=angĕlus",
            "Allen and Greenough Section 12",
            "light penult ĕ -> antepenult",
        ),
    ),
    "opera": (
        ("o", "pe", "ra"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n32660",
            "key=opera",
            "orth=ŏpĕra",
            "Allen and Greenough Section 12",
            "light penult ĕ -> antepenult",
        ),
    ),
    "familia": (
        ("fa", "mi", "li", "a"),
        1,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n17652",
            "key=familia",
            "orth=fămĭlĭa",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "femina": (
        ("fe", "mi", "na"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n17904",
            "key=femina",
            "orth=fēmĭna",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "formula": (
        ("for", "mu", "la"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n18602",
            "key=formula",
            "orth=formŭla",
            "Allen and Greenough Section 12",
            "light penult ŭ -> antepenult",
        ),
    ),
    "tabula": (
        ("ta", "bu", "la"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n47346",
            "key=tabula",
            "orth=tăbŭla",
            "Allen and Greenough Section 12",
            "light penult ŭ -> antepenult",
        ),
    ),
    "epistula": (
        ("e", "pis", "tu", "la"),
        1,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n15995",
            "key=epistula",
            "orth=ĕpistŭla",
            "Allen and Greenough Section 12",
            "light penult ŭ -> antepenult",
        ),
    ),
    "caritas": (
        ("ca", "ri", "tas"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n6810",
            "key=caritas",
            "orth=cārĭtas",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "ueritas": (
        ("ve", "ri", "tas"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n50557",
            "key=veritas",
            "orth=vērĭtas",
            "lookup v -> u",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "unitas": (
        ("u", "ni", "tas"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n49852",
            "key=unitas",
            "orth=ūnĭtas",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "uictima": (
        ("vic", "ti", "ma"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n50861",
            "key=victima",
            "orth=victĭma",
            "lookup v -> u",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "maximus": (
        ("ma", "xi", "mus"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n28269",
            "key=maximus",
            "orth=maxĭmus",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "optimus": (
        ("op", "ti", "mus"),
        0,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n32847",
            "key=optimus",
            "orth=optĭmus",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "humilitas": (
        ("hu", "mi", "li", "tas"),
        1,
        ("perseus-lewis-short", "allen-greenough-accents"),
        (
            "entryFree id=n21059",
            "key=humilitas",
            "orth=hŭmĭlĭtas",
            "Allen and Greenough Section 12",
            "light penult ĭ -> antepenult",
        ),
    ),
    "eleison": (
        ("e", "le", "i", "son"),
        1,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 2 / PDF p. 112",
            "Ordinary of the Mass",
            "prints eléison",
        ),
    ),
    "laudamus": (
        ("lau", "da", "mus"),
        1,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 2 / PDF p. 112",
            "Gloria",
            "prints Laudámus",
        ),
    ),
    "adoramus": (
        ("a", "do", "ra", "mus"),
        2,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 2 / PDF p. 112",
            "Gloria",
            "prints Adorámus",
        ),
    ),
    "omnipotens": (
        ("om", "ni", "po", "tens"),
        1,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 2 / PDF p. 112",
            "Gloria and Credo",
            "prints omnípotens",
        ),
    ),
    "peccata": (
        ("pec", "ca", "ta"),
        1,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 2 / PDF p. 112",
            "Gloria",
            "prints peccáta",
        ),
    ),
    "suscipe": (
        ("sus", "ci", "pe"),
        0,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 2 / PDF p. 112",
            "Gloria",
            "prints súscipe",
        ),
    ),
    "mulieribus": (
        ("mu", "li", "e", "ri", "bus"),
        2,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 1861 / PDF p. 2105",
            "Ave Maria",
            "prints muliéribus",
        ),
    ),
    "peccatoribus": (
        ("pec", "ca", "to", "ri", "bus"),
        2,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 1861 / PDF p. 2105",
            "Ave Maria",
            "prints peccatóribus",
        ),
    ),
    "sanctificetur": (
        ("sanc", "ti", "fi", "ce", "tur"),
        3,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 6 / PDF p. 116",
            "Pater noster",
            "prints Sanctificétur",
        ),
    ),
    "quotidianum": (
        ("quo", "ti", "di", "a", "num"),
        3,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 6 / PDF p. 116",
            "Pater noster",
            "prints quotidiánum",
        ),
    ),
    "spiritu": (
        ("spi", "ri", "tu"),
        0,
        ("liber-usualis-1961-full-scan",),
        (
            "unnumbered first Ordinary page / PDF p. 111",
            "Et cum spíritu tuo",
        ),
    ),
    "oremus": (
        ("o", "re", "mus"),
        1,
        ("liber-usualis-1961-full-scan",),
        (
            "printed p. 2 / PDF p. 112",
            "prints Orémus",
        ),
    ),
    "confiteor": (
        ("con", "fi", "te", "or"),
        1,
        ("liber-usualis-1961-full-scan",),
        (
            "unnumbered first Ordinary page / PDF p. 111",
            "Confiteor prayer",
            "prints Confíteor",
        ),
    ),
}

LOOKUP_SURFACES = {
    "ueritas": "veritas",
    "uictima": "victima",
}


class _FakeResource:
    def __init__(self, text: str) -> None:
        self._text = text

    def joinpath(self, name: str) -> "_FakeResource":
        assert name == "stress_lexicon.jsonl"
        return self

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        return self._text


def _entry(
    lookup_key: str,
    syllables: tuple[str, ...],
    stress_index: int,
    *,
    is_exception: bool = False,
) -> StressLexiconEntry:
    return StressLexiconEntry(
        lookup_key=lookup_key,
        syllables=syllables,
        stress_index=stress_index,
        source_ids=("perseus-lewis-short",),
        note="Perseus Lewis and Short XML entryFree id=n-test, key=test",
        is_exception=is_exception,
    )


def _load_rows(monkeypatch: pytest.MonkeyPatch, *rows: dict[str, object]) -> None:
    text = "\n".join(json.dumps(row) for row in rows)
    monkeypatch.setattr(stress_module, "files", lambda package: _FakeResource(text))
    load_stress_lexicon()


def test_seed_lexicon_has_exact_source_backed_entries() -> None:
    lexicon = load_stress_lexicon()

    assert set(lexicon) == set(SEED_EXPECTATIONS)
    for lookup_key, expected in SEED_EXPECTATIONS.items():
        syllables, stress_index, source_ids, note_fragments = expected
        entry = lexicon[lookup_key]
        surface = LOOKUP_SURFACES.get(lookup_key, lookup_key)
        normalized = normalize_word(surface)
        assert normalized.lookup_key == lookup_key
        assert entry.syllables == syllables == syllabify(normalized.normalized)
        assert entry.stress_index == stress_index
        assert entry.source_ids == source_ids
        assert all(fragment in entry.note for fragment in note_fragments)
        assert not entry.is_exception

    assert "prints allelúia" not in lexicon["alleluia"].note


def test_override_wins_over_explicit_stress_and_lexicon() -> None:
    decision = resolve_stress(
        "dominus",
        ("do", "mi", "nus"),
        load_stress_lexicon(),
        override_index=1,
        explicit_stress_index=2,
    )

    assert decision.stress_index == 1
    assert decision.method is ResolutionMethod.OVERRIDE
    assert decision.applied_rule_ids == ("override-stress",)


def test_explicit_acute_stress_wins_over_lexicon() -> None:
    decision = resolve_stress(
        "dominus",
        ("do", "mi", "nus"),
        load_stress_lexicon(),
        explicit_stress_index=1,
    )

    assert decision.stress_index == 1
    assert decision.method is ResolutionMethod.OVERRIDE
    assert decision.applied_rule_ids == ("explicit-stress",)


def test_exception_entry_resolves_before_general_rules() -> None:
    exception = _entry("fabula", ("fa", "bu", "la"), 1, is_exception=True)

    decision = resolve_stress("fabula", exception.syllables, {"fabula": exception})

    assert decision.stress_index == 1
    assert decision.method is ResolutionMethod.EXCEPTION
    assert decision.applied_rule_ids == ("stress-exception",)


def test_lexicon_resolves_open_penult_quantity() -> None:
    decision = resolve_stress(
        "dominus",
        ("do", "mi", "nus"),
        load_stress_lexicon(),
    )

    assert decision.stress_index == 0
    assert decision.method is ResolutionMethod.LEXICON
    assert "perseus-lewis-short" in decision.source_ids


def test_whole_word_lexicon_entry_wins_over_enclitic_analysis() -> None:
    base = _entry("dominus", ("do", "mi", "nus"), 0)
    whole_word = _entry("dominusque", ("do", "mi", "nus", "que"), 1)

    decision = resolve_stress(
        "dominusque",
        whole_word.syllables,
        {"dominus": base, "dominusque": whole_word},
    )

    assert decision.stress_index == 1
    assert decision.method is ResolutionMethod.LEXICON


def test_enclitic_stresses_syllable_before_suffix_for_known_base() -> None:
    decision = resolve_stress(
        "dominusque",
        ("do", "mi", "nus", "que"),
        load_stress_lexicon(),
    )

    assert decision.stress_index == 2
    assert decision.method is ResolutionMethod.RULE
    assert decision.applied_rule_ids == ("enclitic-stress",)
    assert "allen-greenough-accents" in decision.source_ids
    assert "perseus-lewis-short" in decision.source_ids


def test_unknown_enclitic_looking_word_is_not_silently_resolved() -> None:
    decision = resolve_stress("fabulaque", ("fa", "bu", "la", "que"), {})

    assert decision.stress_index == 1
    assert decision.method is ResolutionMethod.CANDIDATE
    assert decision.warnings[0].code == "PRONUNCIATION_NEEDS_REVIEW"


def test_monosyllable_uses_only_syllable() -> None:
    decision = resolve_stress("pax", ("pax",), {})

    assert decision.stress_index == 0
    assert decision.method is ResolutionMethod.RULE
    assert decision.applied_rule_ids == ("monosyllable-stress",)


def test_disyllable_uses_first_syllable_rule() -> None:
    decision = resolve_stress("sanctus", ("sanc", "tus"), {})

    assert decision.stress_index == 0
    assert decision.method is ResolutionMethod.RULE
    assert decision.applied_rule_ids == ("disyllable-stress",)


@pytest.mark.parametrize("penult", ["mā", "aē", "dic"])
def test_provably_heavy_penult_receives_stress(penult: str) -> None:
    decision = resolve_stress("unlisted", ("a", penult, "re"), {})

    assert decision.stress_index == 1
    assert decision.method is ResolutionMethod.RULE
    assert decision.applied_rule_ids == ("heavy-penult-stress",)


def test_diaeresis_breaks_diphthong_for_penult_weight() -> None:
    decision = resolve_stress("unlisted", ("a", "aë", "re"), {})

    assert decision.stress_index == 0
    assert decision.method is ResolutionMethod.CANDIDATE
    assert decision.warnings[0].code == "PRONUNCIATION_NEEDS_REVIEW"


def test_unknown_open_penult_returns_review_warning() -> None:
    decision = resolve_stress("fabula", ("fa", "bu", "la"), {})

    assert decision.stress_index == 0
    assert decision.method is ResolutionMethod.CANDIDATE
    assert decision.source_ids == ("allen-greenough-accents",)
    assert decision.warnings[0].code == "PRONUNCIATION_NEEDS_REVIEW"
    assert decision.warnings[0].severity is Severity.WARNING


@pytest.mark.parametrize("keyword", ["override_index", "explicit_stress_index"])
def test_explicit_stress_indices_must_point_to_a_syllable(keyword: str) -> None:
    with pytest.raises(ValueError, match="stress_index must point to an existing syllable"):
        resolve_stress("pax", ("pax",), {}, **{keyword: 1})


def test_lexicon_syllables_must_match_request() -> None:
    with pytest.raises(ValueError, match="stress lexicon syllable mismatch: dominus"):
        resolve_stress("dominus", ("do", "minus"), load_stress_lexicon())


def test_loader_rejects_duplicate_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "lookup_key": "dominus",
        "syllables": ["do", "mi", "nus"],
        "stress_index": 0,
        "source_ids": ["perseus-lewis-short"],
        "note": "Perseus XML entryFree id=n14699",
        "is_exception": False,
    }

    with pytest.raises(ValueError, match="duplicate stress key at line 2: dominus"):
        _load_rows(monkeypatch, row, row)


def test_loader_rejects_invalid_stress_index(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "lookup_key": "dominus",
        "syllables": ["do", "mi", "nus"],
        "stress_index": 3,
        "source_ids": ["perseus-lewis-short"],
        "note": "Perseus XML entryFree id=n14699",
        "is_exception": False,
    }

    with pytest.raises(ValueError, match="invalid stress index at line 1"):
        _load_rows(monkeypatch, row)


def test_loader_rejects_unknown_source(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "lookup_key": "dominus",
        "syllables": ["do", "mi", "nus"],
        "stress_index": 0,
        "source_ids": ["unknown-source"],
        "note": "specific locator",
        "is_exception": False,
    }

    with pytest.raises(ValueError, match="unknown stress sources at line 1"):
        _load_rows(monkeypatch, row)


@pytest.mark.parametrize(
    ("source_ids", "note"),
    [([], "specific locator"), (["perseus-lewis-short"], "   ")],
)
def test_loader_rejects_incomplete_entry(
    monkeypatch: pytest.MonkeyPatch,
    source_ids: list[str],
    note: str,
) -> None:
    row = {
        "lookup_key": "dominus",
        "syllables": ["do", "mi", "nus"],
        "stress_index": 0,
        "source_ids": source_ids,
        "note": note,
        "is_exception": False,
    }

    with pytest.raises(ValueError, match="incomplete stress entry at line 1"):
        _load_rows(monkeypatch, row)


def test_document_describes_stress_priority_and_candidate_policy() -> None:
    document = PRONUNCIATION_DOCUMENT.read_text(encoding="utf-8")
    section = document.split("## 重音", maxsplit=1)[1].split("## 双辅音", maxsplit=1)[0]

    assert "请求覆盖 > acute" in section
    assert "已知 base lexicon" in section
    assert "PRONUNCIATION_NEEDS_REVIEW" in section
    assert "不得把未知开放 penult" in section
