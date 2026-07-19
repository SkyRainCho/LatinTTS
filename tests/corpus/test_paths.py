from pathlib import Path

import pytest

from latintts.corpus.paths import CorpusPaths


def test_resolve_local_rejects_absolute_and_parent_escape(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    with pytest.raises(ValueError, match="relative"):
        paths.resolve_local(Path("C:/outside.wav"))
    with pytest.raises(ValueError, match="escapes local-data"):
        paths.resolve_local(Path("../outside.wav"))


def test_ensure_layout_creates_only_local_data_directories(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    assert paths.raw_spoken == tmp_path / "local-data" / "raw" / "spoken"
    assert paths.raw_spoken.is_dir()
    assert paths.alignments.is_dir()
