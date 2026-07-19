from pathlib import Path

import pytest

from latintts.corpus.store import append_jsonl_event, read_jsonl, write_jsonl_atomic


def test_atomic_jsonl_round_trip_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl_atomic(path, ({"text": "cælum", "ordinal": 1},))
    assert read_jsonl(path) == ({"ordinal": 1, "text": "cælum"},)


def test_read_jsonl_rejects_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"one"}\n\n', encoding="utf-8")
    with pytest.raises(ValueError, match="blank line 2"):
        read_jsonl(path)


def test_append_event_keeps_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    append_jsonl_event(path, {"id": "one"})
    append_jsonl_event(path, {"id": "two"})
    assert [row["id"] for row in read_jsonl(path)] == ["one", "two"]
