import json
from pathlib import Path

import pytest

from latintts import stress as stress_module
from latintts.domain import ResolutionMethod, Severity
from latintts.stress import StressLexiconEntry, load_stress_lexicon, resolve_stress
from latintts.syllables import syllabify

PRONUNCIATION_DOCUMENT = (
    Path(__file__).parents[2] / "docs" / "pronunciation" / "roman-ecclesiastical.md"
)
SEED_STRESSES = {
    "dominus": 0,
    "regina": 1,
    "maria": 1,
    "gratia": 0,
    "caelum": 0,
    "alleluia": 2,
    "magnificat": 1,
    "misericordia": 3,
    "benedictus": 2,
    "excelsis": 1,
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

    assert set(lexicon) == set(SEED_STRESSES)
    for lookup_key, stress_index in SEED_STRESSES.items():
        entry = lexicon[lookup_key]
        assert entry.syllables == syllabify(lookup_key)
        assert entry.stress_index == stress_index
        assert entry.source_ids
        assert entry.note.strip()
        assert not entry.is_exception

    assert "entryFree id=n14699" in lexicon["dominus"].note
    assert "entryFree id=n6042" in lexicon["caelum"].note
    assert "PDF page 32 (printed xxxviii)" in lexicon["alleluia"].note
    assert "PDF page 32 (printed xxxviii)" in lexicon["magnificat"].note
    assert "closed penult" in lexicon["benedictus"].note
    assert "closed penult" in lexicon["excelsis"].note


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
