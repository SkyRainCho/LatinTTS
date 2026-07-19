import json
from pathlib import Path

import pytest

from latintts import sources as sources_module
from latintts.sources import load_source_registry

SOURCE_IDS = {
    "allen-greenough-accents",
    "ewtn-ecclesiastical-latin",
    "iveson-roman-pronunciation-1964",
    "liber-usualis-1961-full-scan",
    "liber-usualis-1962",
    "librivox-public-domain",
    "perseus-lewis-short",
    "wikimedia-ecclesiastical-pronunciation",
}
PRONUNCIATION_DOCUMENT = (
    Path(__file__).parents[2] / "docs" / "pronunciation" / "roman-ecclesiastical.md"
)
GOLD_FIXTURE_DOCUMENT = Path(__file__).parents[1] / "fixtures" / "README.md"


def _load_pronunciation_document() -> str:
    return PRONUNCIATION_DOCUMENT.read_text(encoding="utf-8")


class _FakeResource:
    def __init__(self, text: str) -> None:
        self._text = text

    def joinpath(self, name: str) -> "_FakeResource":
        assert name == "pronunciation_sources.json"
        return self

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        return self._text


def _source_row(source_id: str = "test-source", authority_rank: int = 1) -> dict[str, object]:
    return {
        "source_id": source_id,
        "title": "Test source",
        "url": "https://example.test/source",
        "source_kind": "test",
        "authority_rank": authority_rank,
        "accessed_on": "2026-07-18",
        "locator": "test locator",
        "license_or_terms": "test terms",
        "usage_note": "test use only",
    }


def _install_source_rows(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
) -> None:
    text = json.dumps(rows)
    monkeypatch.setattr(sources_module, "files", lambda package: _FakeResource(text))


def test_registry_contains_normative_and_evaluation_sources() -> None:
    sources = load_source_registry()
    assert set(sources) == SOURCE_IDS
    assert sources["liber-usualis-1962"].authority_rank == 1
    assert sources["iveson-roman-pronunciation-1964"].accessed_on == "2026-07-18"
    accessed_on_2026_07_18 = {
        "iveson-roman-pronunciation-1964",
        "liber-usualis-1961-full-scan",
    }
    assert all(
        sources[source_id].accessed_on == "2026-07-18" for source_id in accessed_on_2026_07_18
    )
    assert all(
        record.accessed_on == "2026-07-17"
        for source_id, record in sources.items()
        if source_id not in accessed_on_2026_07_18
    )


def test_registry_has_unique_ids_and_nonempty_usage_terms() -> None:
    sources = load_source_registry()
    assert len(sources) == 8
    assert all(record.license_or_terms.strip() for record in sources.values())
    assert all(record.usage_note.strip() for record in sources.values())


def test_registry_rejects_duplicate_source_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _source_row()
    _install_source_rows(monkeypatch, [row, row])

    with pytest.raises(ValueError, match="pronunciation source IDs must be unique"):
        load_source_registry()


def test_registry_rejects_nonpositive_authority_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_source_rows(monkeypatch, [_source_row(authority_rank=0)])

    with pytest.raises(ValueError, match="authority_rank must be positive"):
        load_source_registry()


def test_perseus_terms_use_current_license_version() -> None:
    terms = load_source_registry()["perseus-lewis-short"].license_or_terms
    assert "CC BY-SA 4.0" in terms
    assert "3.0" not in terms


def test_iveson_source_has_the_direct_ph_rule_locator_and_restricted_terms() -> None:
    source = load_source_registry()["iveson-roman-pronunciation-1964"]

    assert source.authority_rank == 2
    assert source.locator == "PDF page 1 (printed p. 14), lines 44-46"
    assert source.license_or_terms == (
        "Copyrighted journal article; reference use only; do not redistribute page content"
    )


def test_full_liber_scan_is_registered_only_for_text_and_printed_stress_evidence() -> None:
    source = load_source_registry()["liber-usualis-1961-full-scan"]

    assert source.url == ("https://propria.org/wp-content/uploads/2019/10/liber-usualis-1961.pdf")
    assert source.source_kind == "liturgical_text_scan"
    assert source.locator == (
        "Complete 2340-page PDF scan; cite each occurrence by printed page, PDF page, "
        "section or prayer, and verse where applicable"
    )
    assert "text occurrence and printed acute stress evidence only" in source.usage_note
    assert "not a pronunciation-rule source" in source.usage_note
    assert "verify copyright and local jurisdiction before redistribution or training use" in (
        source.license_or_terms
    )


def test_document_defines_canonical_token_and_lookup_layers() -> None:
    document = _load_pronunciation_document()
    section = document.split("## Unicode 与拼写规范化", maxsplit=1)[1].split("## 元音", maxsplit=1)[
        0
    ]
    required_contracts = (
        "`surface`",
        "`source_span`",
        "canonical normalized token",
        "`æ -> ae`",
        "`œ -> oe`",
        "`expand-ae-ligature`",
        "`expand-oe-ligature`",
        "长度变化绝不改写原文 span",
        "lookup key",
        "`j -> i`",
        "`v -> u`",
        "`lookup-j-to-i`",
        "`lookup-v-to-u`",
        "不得覆盖 canonical normalized token",
        "禁止对原始短语做无条件全局替换",
    )
    assert all(contract in section for contract in required_contracts)


def test_ligatures_reach_rules_as_expanded_canonical_tokens() -> None:
    document = _load_pronunciation_document()
    section = document.split("## 双元音与相邻元音", maxsplit=1)[1].split("## 辅音", maxsplit=1)[0]
    assert "G2P/音节规则消费展开后的 canonical normalized token" in section
    assert "`æ` 已展开为 `ae`" in section
    assert "`œ` 已展开为 `oe`" in section


def test_document_requires_review_when_penult_weight_is_unknown() -> None:
    document = _load_pronunciation_document()
    required_contracts = (
        "普通拼写",
        "penult",
        "只能生成候选",
        "PRONUNCIATION_NEEDS_REVIEW",
        "不得静默宣称确定",
    )
    assert all(contract in document for contract in required_contracts)


def test_document_defines_version_and_phoneme_rendering_contracts() -> None:
    document = _load_pronunciation_document()
    required_contracts = (
        '`schema_version == "1"`',
        '`rule_version == "ecclesiastical-roman-v1"`',
        "消费前必须同时校验",
        "不得静默消费不匹配的版本",
        "每个音节边界都保留 `.`",
        "重读音节前插入独立 token `ˈ`",
        "重读音节边界用 `ˈ` 取代 `.`",
    )
    assert all(contract in document for contract in required_contracts)


def test_document_defines_gold_as_a_human_reviewed_regression_anchor() -> None:
    document = _load_pronunciation_document()
    required_contracts = (
        "gold 集成测试是回归锚点和变更检测器",
        "不是发音正确性的独立证明",
        "禁止从当前 `Pronouncer` 输出自动重写",
        "人工对照受影响规则的规范来源",
        "重新审核所有受影响词条",
    )
    assert all(contract in document for contract in required_contracts)


def test_gold_fixture_document_marks_ipa_as_human_verified() -> None:
    assert GOLD_FIXTURE_DOCUMENT.is_file()
    document = GOLD_FIXTURE_DOCUMENT.read_text(encoding="utf-8")

    assert "`ipa`: human-verified, not pipeline-derived" in document
    assert '`review_state="approved"`' in document
    assert "不能用 `Pronouncer` 当前输出自动生成" in document


def test_document_resolves_the_equal_rank_liber_source_scopes() -> None:
    document = _load_pronunciation_document()

    assert "相同的 `authority_rank=1` 不构成可互相覆盖的平局" in document
    assert "1962 发音规则节选决定规则语义" in document
    assert "1961 全扫描只决定正文出现与印刷 acute 证据" in document


def test_canonical_phoneme_rows_cite_registered_sources() -> None:
    document = _load_pronunciation_document()
    section = document.split("## IPA 与规范音素表", maxsplit=1)[1].split(
        "## 规则覆盖矩阵", maxsplit=1
    )[0]
    table_rows = [line for line in section.splitlines() if line.startswith("| ")][2:]
    assert "有来源的工程归一化" in section
    assert table_rows
    assert all(any(f"`{source_id}`" in row for source_id in SOURCE_IDS) for row in table_rows)
