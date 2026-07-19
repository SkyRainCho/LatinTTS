from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


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
        root = project_root.resolve()
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
