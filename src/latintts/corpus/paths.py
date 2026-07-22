from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def require_canonical_descendant(
    root: Path,
    target: Path,
    *,
    kind: str,
    require_file: bool = False,
) -> Path:
    """Reject lexical escapes and every existing alias/reparse path component."""
    root_absolute = Path(os.path.abspath(root))
    target_absolute = Path(os.path.abspath(target))
    try:
        relative = target_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError(f"{kind} is outside its canonical root") from error
    current = root_absolute
    candidates = [current]
    for component in relative.parts:
        current /= component
        candidates.append(current)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for candidate in candidates:
        if not candidate.exists() and not candidate.is_symlink():
            continue
        metadata = os.lstat(candidate)
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or attributes & reparse_flag:
            raise ValueError(f"{kind} path contains an alias or reparse point")
        if candidate.resolve() != candidate:
            raise ValueError(f"{kind} path is not canonical")
    if root_absolute.resolve(strict=False) != root_absolute:
        raise ValueError(f"{kind} root is not canonical")
    resolved_target = target_absolute.resolve(strict=False)
    try:
        resolved_target.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError(f"{kind} resolves outside its canonical root") from error
    if require_file and not target_absolute.is_file():
        raise ValueError(f"{kind} must be a regular file")
    return target_absolute


@dataclass(frozen=True, slots=True)
class CorpusPaths:
    project_root: Path
    local_data: Path
    raw_spoken: Path
    raw_sung: Path
    normalized: Path
    segments: Path
    alignments: Path
    manifests: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> CorpusPaths:
        return cls._from_absolute_root(project_root.resolve())

    @classmethod
    def from_project_root_lexical(cls, project_root: Path) -> CorpusPaths:
        """Build the fixed layout without querying filesystem aliases or existence."""
        return cls._from_absolute_root(Path(os.path.abspath(project_root)))

    @classmethod
    def _from_absolute_root(cls, root: Path) -> CorpusPaths:
        local = root / "local-data"
        derived = local / "derived" / "corpus-v1"
        return cls(
            project_root=root,
            local_data=local,
            raw_spoken=local / "raw" / "spoken",
            raw_sung=local / "raw" / "sung",
            normalized=derived / "normalized",
            segments=derived / "segments",
            alignments=derived / "alignments",
            manifests=local / "manifests",
        )

    def ensure_layout(self) -> None:
        for path in (
            self.raw_spoken,
            self.raw_sung,
            self.normalized,
            self.segments,
            self.alignments,
            self.manifests,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve_local(self, relative_path: Path | PurePosixPath | str) -> Path:
        candidate = Path(str(relative_path))
        if candidate.is_absolute():
            raise ValueError("local corpus path must be relative")
        resolved = (self.local_data / candidate).resolve()
        try:
            resolved.relative_to(self.local_data.resolve())
        except ValueError as error:
            raise ValueError("local corpus path escapes local-data") from error
        return resolved

    def relative_local(self, path: Path) -> str:
        return path.resolve().relative_to(self.local_data.resolve()).as_posix()
