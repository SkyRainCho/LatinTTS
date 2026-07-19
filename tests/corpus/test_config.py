import json
from pathlib import Path

import pytest

from latintts.corpus.config import CorpusConfig


def test_config_digest_is_independent_of_json_key_order(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "corpus_version": "corpus-v1",
                "analysis": {},
                "vad": {},
                "pause": {},
                "pairing": {},
                "alignment": {},
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "alignment": {},
                "pairing": {},
                "pause": {},
                "vad": {},
                "analysis": {},
                "corpus_version": "corpus-v1",
                "schema_version": "1",
            }
        )
    )
    assert CorpusConfig.load(first).digest == CorpusConfig.load(second).digest


def test_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "corpus_version": "corpus-v1",
                "analysis": {},
                "vad": {},
                "pause": {},
                "pairing": {},
                "alignment": {},
                "extra": True,
            }
        )
    )
    with pytest.raises(ValueError, match="exact top-level fields"):
        CorpusConfig.load(path)
