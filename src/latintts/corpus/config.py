from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FIELDS = {
    "schema_version",
    "corpus_version",
    "analysis",
    "vad",
    "pause",
    "pairing",
    "alignment",
}


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    raw: dict[str, Any]
    digest: str

    @classmethod
    def load(cls, path: Path) -> CorpusConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if type(raw) is not dict or set(raw) != _FIELDS:
            raise ValueError("config must contain the exact top-level fields")
        if raw["schema_version"] != "1" or raw["corpus_version"] != "corpus-v1":
            raise ValueError("unsupported corpus configuration version")
        canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return cls(
            raw=raw,
            digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
