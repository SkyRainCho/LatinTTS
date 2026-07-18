import json
from pathlib import Path

from latintts.g2p import IMPLEMENTED_RULE_IDS

GOLD_FIXTURE = Path("tests/fixtures/gold_pronunciations.jsonl")
RULE_DOCUMENT = Path("docs/pronunciation/roman-ecclesiastical.md")
REAL_WORD_GOLD_EXEMPT_RULE_IDS = {"q-hard"}
RULE_MATRIX_START = "<!-- g2p-rule-matrix:start -->"
RULE_MATRIX_END = "<!-- g2p-rule-matrix:end -->"


def test_every_g2p_rule_has_real_word_gold_coverage_except_q_fallback() -> None:
    covered: set[str] = set()
    for line in GOLD_FIXTURE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            covered.update(json.loads(line)["rule_ids"])

    assert IMPLEMENTED_RULE_IDS - covered == REAL_WORD_GOLD_EXEMPT_RULE_IDS


def test_rule_document_has_one_complete_manual_matrix_row_per_implemented_rule() -> None:
    document = RULE_DOCUMENT.read_text(encoding="utf-8")
    gold_rules_by_word = {
        row["word"]: set(row["rule_ids"])
        for row in (
            json.loads(line)
            for line in GOLD_FIXTURE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    assert RULE_MATRIX_START in document
    assert RULE_MATRIX_END in document
    matrix = document.split(RULE_MATRIX_START, maxsplit=1)[1].split(
        RULE_MATRIX_END,
        maxsplit=1,
    )[0]

    rows: dict[str, tuple[str, ...]] = {}
    for line in matrix.splitlines():
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if len(cells) != 7 or not cells[0].startswith("`") or not cells[0].endswith("`"):
            continue
        rule_id = cells[0][1:-1]
        assert rule_id not in rows
        rows[rule_id] = cells[1:]

    assert set(rows) == IMPLEMENTED_RULE_IDS
    assert all(all(cell for cell in cells) for cells in rows.values())
    assert rows["q-hard"][-1] == "无真实 gold; synthetic fallback 单元测试"
    for rule_id, cells in rows.items():
        assert "liber-usualis-1962" in cells[-2] or "iveson-roman-pronunciation-1964" in cells[-2]
        if rule_id == "q-hard":
            continue
        gold_cell = cells[-1]
        assert gold_cell.startswith("`") and gold_cell.endswith("`")
        gold_word = gold_cell[1:-1]
        assert rule_id in gold_rules_by_word[gold_word]
