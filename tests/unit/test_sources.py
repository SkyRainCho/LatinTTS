from pathlib import Path

from latintts.sources import load_source_registry

SOURCE_IDS = {
    "allen-greenough-accents",
    "ewtn-ecclesiastical-latin",
    "liber-usualis-1962",
    "librivox-public-domain",
    "perseus-lewis-short",
    "wikimedia-ecclesiastical-pronunciation",
}
PRONUNCIATION_DOCUMENT = (
    Path(__file__).parents[2] / "docs" / "pronunciation" / "roman-ecclesiastical.md"
)


def _load_pronunciation_document() -> str:
    return PRONUNCIATION_DOCUMENT.read_text(encoding="utf-8")


def test_registry_contains_normative_and_evaluation_sources() -> None:
    sources = load_source_registry()
    assert set(sources) == SOURCE_IDS
    assert sources["liber-usualis-1962"].authority_rank == 1
    assert all(record.accessed_on == "2026-07-17" for record in sources.values())


def test_registry_has_unique_ids_and_nonempty_usage_terms() -> None:
    sources = load_source_registry()
    assert len(sources) == 6
    assert all(record.license_or_terms.strip() for record in sources.values())
    assert all(record.usage_note.strip() for record in sources.values())


def test_perseus_terms_use_current_license_version() -> None:
    terms = load_source_registry()["perseus-lewis-short"].license_or_terms
    assert "CC BY-SA 4.0" in terms
    assert "3.0" not in terms


def test_document_defines_canonical_token_and_lookup_layers() -> None:
    document = _load_pronunciation_document()
    section = document.split("## Unicode 与拼写规范化", maxsplit=1)[1].split(
        "## 元音", maxsplit=1
    )[0]
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
    section = document.split("## 双元音与相邻元音", maxsplit=1)[1].split(
        "## 辅音", maxsplit=1
    )[0]
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


def test_canonical_phoneme_rows_cite_registered_sources() -> None:
    document = _load_pronunciation_document()
    section = document.split("## IPA 与规范音素表", maxsplit=1)[1].split(
        "## 规则覆盖矩阵", maxsplit=1
    )[0]
    table_rows = [line for line in section.splitlines() if line.startswith("| ")][2:]
    assert "有来源的工程归一化" in section
    assert table_rows
    assert all(any(f"`{source_id}`" in row for source_id in SOURCE_IDS) for row in table_rows)
