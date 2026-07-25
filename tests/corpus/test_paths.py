from pathlib import Path
from types import SimpleNamespace

import pytest

from latintts.corpus import paths as paths_module
from latintts.corpus.paths import CorpusPaths, require_canonical_descendant


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


def test_require_canonical_descendant_rejects_reparse_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    component = root / "component"
    target = component / "artifact.flac"
    component.mkdir(parents=True)
    target.write_bytes(b"audio")
    real_lstat = paths_module.os.lstat

    def reparse_lstat(path: Path) -> object:
        metadata = real_lstat(path)
        if Path(path) == component:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=getattr(
                    paths_module.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
                ),
            )
        return metadata

    monkeypatch.setattr(paths_module.os, "lstat", reparse_lstat)
    with pytest.raises(ValueError, match="alias or reparse"):
        require_canonical_descendant(root, target, kind="test artifact", require_file=True)


def test_require_canonical_descendant_rejects_lexical_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="outside its canonical root"):
        require_canonical_descendant(root, tmp_path / "outside", kind="test artifact")


def test_require_canonical_descendant_requires_existing_regular_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        require_canonical_descendant(
            root,
            root / "missing.flac",
            kind="test artifact",
            require_file=True,
        )
