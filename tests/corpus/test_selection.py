from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from latintts.corpus.cli import main
from latintts.corpus.domain import CorpusState
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.selection import select_pilot
from latintts.corpus.store import write_jsonl_atomic
from tests.corpus.factories import recording


def test_select_pilot_chooses_short_median_and_long_spoken() -> None:
    records = (
        recording("short", 10.0),
        recording("middle-low", 20.0),
        recording("middle", 30.0),
        recording("long", 60.0),
        recording("chant", 120.0, content_type="sung"),
    )

    selection = select_pilot(records)

    assert selection.recording_ids == ("short", "middle-low", "long")
    assert selection.strategy == "duration-short-median-long-v1"
    assert selection.inventory_hashes == tuple(
        record.sha256 for record in (records[0], records[1], records[3])
    )


def test_explicit_selection_requires_two_or_three_spoken_ids() -> None:
    records = (recording("one", 10.0), recording("two", 20.0))

    selection = select_pilot(records, explicit_ids=("two", "one"))

    assert selection.recording_ids == ("two", "one")
    assert selection.strategy == "explicit-v1"


@pytest.mark.parametrize(
    "explicit_ids",
    (("one",), ("one", "one"), ("one", "two", "three", "four")),
)
def test_explicit_selection_requires_two_or_three_unique_ids(
    explicit_ids: tuple[str, ...],
) -> None:
    records = (
        recording("one", 10.0),
        recording("two", 20.0),
        recording("three", 30.0),
        recording("four", 40.0),
    )

    with pytest.raises(ValueError, match="two or three unique IDs"):
        select_pilot(records, explicit_ids=explicit_ids)


def test_selection_hard_excludes_sung_and_non_inventoried_records() -> None:
    records = (
        recording("eligible-one", 10.0),
        recording("eligible-two", 20.0),
        recording("chant", 30.0, content_type="sung"),
        recording("rejected", 40.0, state=CorpusState.REJECTED),
    )

    assert select_pilot(records).recording_ids == ("eligible-one", "eligible-two")
    with pytest.raises(ValueError, match="missing or not eligible"):
        select_pilot(records, explicit_ids=("eligible-one", "chant"))
    with pytest.raises(ValueError, match="missing or not eligible"):
        select_pilot(records, explicit_ids=("eligible-one", "rejected"))


def test_cli_writes_selection_and_refuses_different_inventory_without_replace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    records = (recording("one", 10.0), recording("two", 20.0))
    write_jsonl_atomic(
        paths.manifests / "recordings.jsonl", (record.to_dict() for record in records)
    )

    assert main(["--project-root", str(tmp_path), "select-pilot"]) == 0

    selection_path = paths.manifests / "pilot-selection.json"
    assert json.loads(selection_path.read_text(encoding="utf-8")) == {
        "inventory_hashes": [records[0].sha256, records[1].sha256],
        "recording_ids": ["one", "two"],
        "schema_version": "1",
        "strategy": "duration-short-median-long-v1",
    }
    changed = replace(records[1], sha256="a" * 64)
    write_jsonl_atomic(
        paths.manifests / "recordings.jsonl", (records[0].to_dict(), changed.to_dict())
    )

    assert main(["--project-root", str(tmp_path), "select-pilot"]) == 2
    assert capsys.readouterr().err == (
        "MANIFEST_SCHEMA_MISMATCH: existing pilot selection has different inventory hashes; "
        "use --replace\n"
    )
    assert main(["--project-root", str(tmp_path), "select-pilot", "--replace"]) == 0


def test_cli_rejects_nonexact_recording_manifest_without_writing_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    row = recording("one", 10.0).to_dict()
    row["unexpected"] = "field"
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (row,))

    assert main(["--project-root", str(tmp_path), "select-pilot"]) == 2

    assert capsys.readouterr().err.startswith("MANIFEST_SCHEMA_MISMATCH: recording row")
    assert not (paths.manifests / "pilot-selection.json").exists()
