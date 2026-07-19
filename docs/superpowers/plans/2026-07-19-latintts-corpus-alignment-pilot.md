# LatinTTS Corpus Alignment Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a reproducible local pipeline that turns 2–3 full-length spoken Ecclesiastical Latin recordings with two valid takes per text unit into human-approved clips, word timestamps, pronunciation plans, JSONL manifests, and a full-corpus effort report.

**Architecture:** Keep the existing dependency-free pronunciation core unchanged and add a focused latintts.corpus package. Pure domain, manifest, pause, pairing, review, and reporting logic remains standard-library-only; FFmpeg, Silero VAD, PyTorch, and the MMS CTC aligner are lazy optional adapters whose versions and raw outputs are recorded. Local audio and manifests live under ignored local-data, while tests use generated WAV files and fake backends.

**Tech Stack:** Python 3.10, frozen dataclasses, pathlib, csv/json/wave/subprocess, FFmpeg/ffprobe, Silero VAD 6.2.1, PyTorch CUDA, MahmoudAshraf97/ctc-forced-aligner at commit 11855d1de76af2b490dd2e8e2db2661805ae90a0, MahmoudAshraf/mms-300m-1130-forced-aligner at revision f37ba6bf1673872e07519fb951866cb2a32a6d7f, pytest, Ruff, mypy.

**Execution:** Subagent-Driven development in an isolated worktree, with a fresh implementer plus specification and code-quality review for every task.

## Global Constraints

- Python remains >=3.10,<3.11; the verified local interpreter is Python 3.10.6.
- The existing project default dependencies remain an empty list; corpus dependencies are installed from a separate requirements file and imported lazily.
- The existing rule version ecclesiastical-roman-v1 and PronunciationPlan schema version 1 remain the pronunciation facts.
- The pilot processes only ordinary spoken recordings; sung or chanted recordings are inventoried but never selected or segmented.
- local-data is wholly Git-ignored; no real audio, real transcript snapshot, local manifest, model weight, or model cache enters Git.
- Raw recordings are immutable. Analysis copies and clips are derived files whose source hash and configuration hash are recorded.
- source_text, spoken_text, normalized_text, and backend-only alignment_text never overwrite one another.
- ASR observations may create diagnostics but can never confirm or mutate spoken_text.
- Every approved take requires an append-only human review event; there is no automatic approval path.
- Every successful corpus-state transition appends a ProcessingEvent with previous/target states, input hashes, config hash, tool versions, timestamps, and result before recordings.jsonl is atomically updated.
- Alignment output is word-level. phoneme_timing_status is always not_estimated in this pilot.
- Both valid takes remain independent candidates linked by repetition_group_id.
- The MMS alignment model is CC BY-NC 4.0 and this pilot is local non-commercial research; every alignment record stores the license.
- The base test suite, Ruff, Ruff format check, strict mypy, at least 95% coverage, and the gold pronunciation audit must continue to pass.
- FFmpeg and ffprobe are explicit prerequisites. They are currently absent from PATH on the verified Windows machine, so doctor must fail clearly until they are installed.
- GPU smoke testing targets NVIDIA GeForce RTX 4080 Laptop GPU with 12282 MiB reported memory and driver 560.70; CPU-only unit tests must remain possible.

---

## File Structure

The implementation creates or modifies these files:

| Path | Responsibility |
| --- | --- |
| .gitignore | Ignore local-data and the corpus virtual environment. |
| config/corpus/pilot-v1.json | Versioned analysis, VAD, pause, pairing, and alignment settings. |
| requirements/corpus.txt | Pin Silero and the CTC aligner source revision outside default runtime dependencies. |
| src/latintts/corpus/__init__.py | Public corpus types and package version constants. |
| src/latintts/corpus/__main__.py | python -m latintts.corpus entry point. |
| src/latintts/corpus/cli.py | Subcommand registration, stable exit codes, and command dispatch. |
| src/latintts/corpus/domain.py | State machine, issue codes, failures, review status, and common value objects. |
| src/latintts/corpus/paths.py | Local-data layout, path confinement, and directory creation. |
| src/latintts/corpus/config.py | Strict configuration loading and canonical configuration hashing. |
| src/latintts/corpus/store.py | Strict JSONL reads, atomic JSONL writes, and append-only events. |
| src/latintts/corpus/records.py | Rights, intake, recording, transcript, segment, processing-event, and review record dataclasses. |
| src/latintts/corpus/intake.py | CSV intake parsing and rights validation. |
| src/latintts/corpus/inventory.py | SHA-256, ffprobe parsing, discovery, inventory, and pilot intake scaffolding. |
| src/latintts/corpus/selection.py | Deterministic short/median/long spoken pilot selection. |
| src/latintts/corpus/transcripts.py | Source snapshots, spoken units, text diffs, pronunciation serialization, and ASR diagnostics. |
| src/latintts/corpus/audio.py | FFmpeg derivation, PCM metrics, candidate extraction, hashes, and atomic output. |
| src/latintts/corpus/vad.py | VAD interface, Silero adapter, probability frames, and speech intervals. |
| src/latintts/corpus/pauses.py | Deterministic per-recording gap analysis and two-class pause clustering. |
| src/latintts/corpus/alignment.py | Alignment request/result contracts, validation, caching keys, and backend protocol. |
| src/latintts/corpus/mms_alignment.py | Lazy MMS CTC backend and raw result normalization. |
| src/latintts/corpus/pairing.py | Long-gap text-unit mapping, candidate splits, evidence scoring, and take pairing. |
| src/latintts/corpus/review.py | Review bundles, TextGrid read/write, append-only decisions, and effective values. |
| src/latintts/corpus/manifest.py | Approved lossless clip extraction and segments.jsonl construction. |
| src/latintts/corpus/report.py | Pilot metrics, scale decision bands, projections, and Markdown/JSON reports. |
| docs/corpus/alignment-pilot-operator-guide.md | Exact local pilot workflow and recovery instructions. |
| README.md | Link to the corpus pilot operator guide. |
| tests/corpus/ | Unit and integration tests using tmp_path, synthetic WAV, and fake adapters. |

No existing pronunciation module is reorganized. Corpus code imports Pronouncer and PronunciationPlan but never modifies their semantics.

## Expected Schedule

| Work block | Tasks | Expected time |
| --- | --- | ---: |
| Contracts, local boundary, inventory, selection, text | 1–5 | 1–2 days |
| Audio derivation, VAD, pause classification | 6–7 | 1–1.5 days |
| Alignment contract, MMS backend, repetition pairing | 8–10 | 1–2 days |
| Review, manifest, report, full gates | 11–13 | 1–1.5 days |
| Private 2–3-file run and human review | Post-implementation run | 0.5–1 day |

Total remains 4–7 working days. FFmpeg/Windows or transcript-version problems consume the contingency already included in that range.

---

### Task 1: Corpus Foundation, Configuration, Paths, and Doctor

**Files:**
- Modify: .gitignore
- Create: config/corpus/pilot-v1.json
- Create: requirements/corpus.txt
- Create: src/latintts/corpus/__init__.py
- Create: src/latintts/corpus/__main__.py
- Create: src/latintts/corpus/cli.py
- Create: src/latintts/corpus/domain.py
- Create: src/latintts/corpus/paths.py
- Create: src/latintts/corpus/config.py
- Create: tests/corpus/__init__.py
- Test: tests/corpus/test_domain.py
- Test: tests/corpus/test_paths.py
- Test: tests/corpus/test_config.py
- Test: tests/corpus/test_cli.py

**Interfaces:**
- Consumes: pathlib.Path and the repository root.
- Produces: CorpusState, ReviewDecision, CorpusIssue, CorpusFailure, CorpusPaths.from_project_root(), CorpusPaths.resolve_local(), CorpusConfig.load(), CorpusConfig.digest, and CLI command doctor.

- [ ] **Step 1: Write failing state and path tests**

~~~python
# tests/corpus/test_domain.py
import pytest

from latintts.corpus.domain import CorpusState, require_transition


def test_state_machine_accepts_next_state_only() -> None:
    assert require_transition(CorpusState.DISCOVERED, CorpusState.INVENTORIED) is None


def test_state_machine_rejects_skipped_state() -> None:
    with pytest.raises(ValueError, match="illegal corpus state transition"):
        require_transition(CorpusState.DISCOVERED, CorpusState.SEGMENTED)
~~~

~~~python
# tests/corpus/test_paths.py
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
~~~

- [ ] **Step 2: Run the new tests and verify import failure**

Run:

~~~powershell
pytest tests/corpus/test_domain.py tests/corpus/test_paths.py -v
~~~

Expected: collection fails with ModuleNotFoundError for latintts.corpus.

- [ ] **Step 3: Add domain and confined-path implementation**

~~~python
# src/latintts/corpus/domain.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CorpusState(str, Enum):
    DISCOVERED = "DISCOVERED"
    INVENTORIED = "INVENTORIED"
    TEXT_CANDIDATES_READY = "TEXT_CANDIDATES_READY"
    TRANSCRIPT_CONFIRMED = "TRANSCRIPT_CONFIRMED"
    SEGMENTED = "SEGMENTED"
    PAIRED = "PAIRED"
    ALIGNED = "ALIGNED"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class IssueCode(str, Enum):
    INVENTORY_UNSUPPORTED_FORMAT = "INVENTORY_UNSUPPORTED_FORMAT"
    INVENTORY_HASH_MISMATCH = "INVENTORY_HASH_MISMATCH"
    RIGHTS_SCOPE_UNCONFIRMED = "RIGHTS_SCOPE_UNCONFIRMED"
    TRANSCRIPT_SOURCE_NOT_FOUND = "TRANSCRIPT_SOURCE_NOT_FOUND"
    TRANSCRIPT_AMBIGUOUS = "TRANSCRIPT_AMBIGUOUS"
    TRANSCRIPT_SPOKEN_MISMATCH = "TRANSCRIPT_SPOKEN_MISMATCH"
    PAUSE_CLASSES_AMBIGUOUS = "PAUSE_CLASSES_AMBIGUOUS"
    TAKE_COUNT_MISMATCH = "TAKE_COUNT_MISMATCH"
    TAKE_DURATION_MISMATCH = "TAKE_DURATION_MISMATCH"
    TAKE_TEXT_MISMATCH = "TAKE_TEXT_MISMATCH"
    ALIGNER_UNAVAILABLE = "ALIGNER_UNAVAILABLE"
    ALIGNMENT_TEXT_UNSUPPORTED = "ALIGNMENT_TEXT_UNSUPPORTED"
    ALIGNMENT_LOW_CONFIDENCE = "ALIGNMENT_LOW_CONFIDENCE"
    AUDIO_QUALITY_REJECTED = "AUDIO_QUALITY_REJECTED"
    MANIFEST_SCHEMA_MISMATCH = "MANIFEST_SCHEMA_MISMATCH"
    CACHE_ARTIFACT_INVALID = "CACHE_ARTIFACT_INVALID"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    PRONUNCIATION_NEEDS_REVIEW = "PRONUNCIATION_NEEDS_REVIEW"


_NEXT_STATES = {
    CorpusState.DISCOVERED: frozenset({CorpusState.INVENTORIED}),
    CorpusState.INVENTORIED: frozenset({CorpusState.TEXT_CANDIDATES_READY}),
    CorpusState.TEXT_CANDIDATES_READY: frozenset({CorpusState.TRANSCRIPT_CONFIRMED}),
    CorpusState.TRANSCRIPT_CONFIRMED: frozenset({CorpusState.SEGMENTED}),
    CorpusState.SEGMENTED: frozenset({CorpusState.PAIRED}),
    CorpusState.PAIRED: frozenset({CorpusState.ALIGNED}),
    CorpusState.ALIGNED: frozenset({CorpusState.REVIEWED}),
    CorpusState.REVIEWED: frozenset({CorpusState.APPROVED, CorpusState.REJECTED}),
    CorpusState.APPROVED: frozenset(),
    CorpusState.REJECTED: frozenset(),
}


def require_transition(current: CorpusState, target: CorpusState) -> None:
    if target not in _NEXT_STATES[current]:
        raise ValueError(f"illegal corpus state transition: {current.value} -> {target.value}")


@dataclass(frozen=True, slots=True)
class CorpusIssue:
    code: IssueCode
    message: str
    entity_id: str | None = None


class CorpusFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = IssueCode(code).value
~~~

~~~python
# src/latintts/corpus/paths.py
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
    def from_project_root(cls, project_root: Path) -> "CorpusPaths":
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
~~~

- [ ] **Step 4: Write failing configuration and doctor tests**

~~~python
# tests/corpus/test_config.py
import json
from pathlib import Path

import pytest

from latintts.corpus.config import CorpusConfig


def test_config_digest_is_independent_of_json_key_order(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"schema_version": "1", "corpus_version": "corpus-v1"}))
    second.write_text(json.dumps({"corpus_version": "corpus-v1", "schema_version": "1"}))
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
~~~

~~~python
# tests/corpus/test_cli.py
from pathlib import Path

from latintts.corpus.cli import main


def test_doctor_returns_failure_when_ffmpeg_is_missing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("latintts.corpus.cli.shutil.which", lambda name: None)
    exit_code = main(["--project-root", str(tmp_path), "doctor"])
    assert exit_code == 1
    assert "ALIGNER_UNAVAILABLE: ffmpeg not found in PATH" in capsys.readouterr().out
~~~

- [ ] **Step 5: Add strict config, doctor, package entry point, and pinned files**

Add config/corpus/pilot-v1.json with this exact content:

~~~json
{
  "schema_version": "1",
  "corpus_version": "corpus-v1",
  "analysis": {
    "sample_rate": 16000,
    "channels": 1,
    "codec": "pcm_s16le"
  },
  "vad": {
    "backend": "silero-vad",
    "model_version": "6.2.1",
    "threshold": 0.5,
    "min_speech_ms": 250,
    "min_silence_ms": 100,
    "window_samples": 512
  },
  "pause": {
    "profile_id": "pause-profile-v1",
    "minimum_gap_ms": 100,
    "minimum_gap_count": 4,
    "separation_ratio": 1.8,
    "maximum_iterations": 50
  },
  "pairing": {
    "minimum_duration_ratio": 0.65,
    "maximum_duration_ratio": 1.35,
    "minimum_score_margin": 0.02
  },
  "alignment": {
    "backend": "mms-ctc",
    "model_id": "MahmoudAshraf/mms-300m-1130-forced-aligner",
    "model_revision": "f37ba6bf1673872e07519fb951866cb2a32a6d7f",
    "language": "lat",
    "romanize": true,
    "split_size": "word",
    "star_frequency": "edges",
    "window_seconds": 30,
    "context_seconds": 2,
    "batch_size": 4,
    "dtype": "float16",
    "device": "cuda",
    "license": "CC-BY-NC-4.0"
  }
}
~~~

~~~python
# src/latintts/corpus/config.py
from __future__ import annotations

import hashlib
from typing import Literal
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
    def load(cls, path: Path) -> "CorpusConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if type(raw) is not dict or set(raw) != _FIELDS:
            raise ValueError("config must contain the exact top-level fields")
        if raw["schema_version"] != "1" or raw["corpus_version"] != "corpus-v1":
            raise ValueError("unsupported corpus configuration version")
        canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return cls(raw=raw, digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest())
~~~

~~~python
# src/latintts/corpus/cli.py
from __future__ import annotations

import argparse
import importlib.util
import shutil
from collections.abc import Sequence
from pathlib import Path

from latintts.corpus.paths import CorpusPaths


def _doctor(project_root: Path) -> int:
    CorpusPaths.from_project_root(project_root).ensure_layout()
    failures: list[str] = []
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            failures.append(f"ALIGNER_UNAVAILABLE: {executable} not found in PATH")
    for package in ("silero_vad", "ctc_forced_aligner"):
        if importlib.util.find_spec(package) is None:
            failures.append(f"ALIGNER_UNAVAILABLE: optional package {package} is not installed")
    for message in failures:
        print(message)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the local LatinTTS corpus")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.project_root)
    raise AssertionError(args.command)
~~~

~~~python
# src/latintts/corpus/__main__.py
from latintts.corpus.cli import main

raise SystemExit(main())
~~~

Add these exact lines to .gitignore:

~~~gitignore
.venv-corpus/
local-data/
~~~

Create requirements/corpus.txt:

~~~text
silero-vad[onnx-cpu]==6.2.1
ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git@11855d1de76af2b490dd2e8e2db2661805ae90a0
~~~

- [ ] **Step 6: Run foundation tests and static checks**

Run:

~~~powershell
pytest tests/corpus/test_domain.py tests/corpus/test_paths.py tests/corpus/test_config.py tests/corpus/test_cli.py -v
ruff check src/latintts/corpus tests/corpus
ruff format --check src/latintts/corpus tests/corpus
mypy src
git check-ignore -v local-data/raw/spoken/example.wav
~~~

Expected: all tests pass; Ruff and mypy exit 0; git check-ignore prints the local-data rule.

- [ ] **Step 7: Commit the foundation**

~~~powershell
git add .gitignore config/corpus/pilot-v1.json requirements/corpus.txt src/latintts/corpus tests/corpus
git commit -m "feat: add corpus pipeline foundation"
~~~

---

### Task 2: Strict Intake, Rights, and JSONL Stores

**Files:**
- Create: src/latintts/corpus/records.py
- Create: src/latintts/corpus/store.py
- Create: src/latintts/corpus/intake.py
- Test: tests/corpus/test_store.py
- Test: tests/corpus/test_intake.py
- Test: tests/corpus/test_records.py

**Interfaces:**
- Consumes: CorpusState, ReviewDecision, and local manifest paths.
- Produces: RightsRecord.from_dict(), IntakeRow, AudioMetadata, RecordingRecord, ProcessingEvent, ReviewEvent, advance_recording(), read_jsonl(), write_jsonl_atomic(), append_jsonl_event(), read_intake(), and load_rights().

- [ ] **Step 1: Write failing store and intake tests**

~~~python
# tests/corpus/test_store.py
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
~~~

~~~python
# tests/corpus/test_intake.py
from pathlib import Path

import pytest

from latintts.corpus.intake import load_rights, read_intake


def test_intake_requires_exact_header(tmp_path: Path) -> None:
    path = tmp_path / "intake.csv"
    path.write_text("relative_path,title_or_citation\nraw/spoken/a.wav,Pater Noster\n")
    with pytest.raises(ValueError, match="exact header"):
        read_intake(path)


def test_rights_require_local_processing_and_training_authority(tmp_path: Path) -> None:
    path = tmp_path / "rights.jsonl"
    path.write_text(
        '{"rights_id":"r1","owner_id":"o1","speaker_id":"s1",'
        '"allow_local_processing":true,"allow_model_training":false,'
        '"allow_internal_evaluation":true,"allow_raw_release":"unknown",'
        '"allow_segment_release":"unknown","allow_model_release":"unknown",'
        '"authorized_at":"2026-07-19","basis":"owner","notes":""}\n'
    )
    with pytest.raises(ValueError, match="training authorization"):
        load_rights(path)
~~~

- [ ] **Step 2: Run tests and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_store.py tests/corpus/test_intake.py -v
~~~

Expected: collection fails because store and intake do not exist.

- [ ] **Step 3: Implement strict records and parsers**

~~~python
# src/latintts/corpus/records.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

from latintts.corpus.domain import CorpusState, require_transition

UnknownBool = bool | Literal["unknown"]


def require_exact_fields(raw: dict[str, Any], fields: frozenset[str], record: str) -> None:
    if set(raw) != fields:
        raise ValueError(f"{record} must contain exact fields; got {sorted(raw)}")


@dataclass(frozen=True, slots=True)
class RightsRecord:
    rights_id: str
    owner_id: str
    speaker_id: str
    allow_local_processing: bool
    allow_model_training: bool
    allow_internal_evaluation: bool
    allow_raw_release: UnknownBool
    allow_segment_release: UnknownBool
    allow_model_release: UnknownBool
    authorized_at: str
    basis: str
    notes: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RightsRecord":
        require_exact_fields(raw, frozenset(field.name for field in fields(cls)), "rights row")
        boolean_fields = (
            "allow_local_processing",
            "allow_model_training",
            "allow_internal_evaluation",
        )
        if any(type(raw[name]) is not bool for name in boolean_fields):
            raise TypeError("rights authorization fields must be booleans")
        release_fields = ("allow_raw_release", "allow_segment_release", "allow_model_release")
        if any(raw[name] not in (True, False, "unknown") for name in release_fields):
            raise TypeError("rights release fields must be booleans or unknown")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntakeRow:
    relative_path: str
    title_or_citation: str
    speaker_id: str
    rights_id: str
    notes: str


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str
    bit_rate: int | None


@dataclass(frozen=True, slots=True)
class RecordingRecord:
    schema_version: str
    recording_id: str
    relative_path: str
    sha256: str
    content_type: Literal["spoken", "sung"]
    title_or_citation: str
    speaker_id: str
    rights_id: str
    notes: str
    metadata: AudioMetadata
    state: CorpusState

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["state"] = self.state.value
        return raw


@dataclass(frozen=True, slots=True)
class ProcessingEvent:
    schema_version: str
    event_id: str
    recording_id: str
    previous_state: CorpusState
    target_state: CorpusState
    input_sha256s: tuple[str, ...]
    config_sha256: str
    tool_versions: tuple[str, ...]
    started_at: str
    finished_at: str
    result: str

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["previous_state"] = self.previous_state.value
        raw["target_state"] = self.target_state.value
        raw["input_sha256s"] = list(self.input_sha256s)
        raw["tool_versions"] = list(self.tool_versions)
        return raw


def advance_recording(
    record: RecordingRecord,
    target: CorpusState,
    *,
    input_sha256s: tuple[str, ...],
    config_sha256: str,
    tool_versions: tuple[str, ...],
    started_at: str,
    finished_at: str,
    result: str,
) -> tuple[RecordingRecord, ProcessingEvent]:
    require_transition(record.state, target)
    identity = {
        "recording_id": record.recording_id,
        "previous_state": record.state.value,
        "target_state": target.value,
        "inputs": input_sha256s,
        "config": config_sha256,
        "tools": tool_versions,
        "started_at": started_at,
        "finished_at": finished_at,
        "result": result,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    event = ProcessingEvent(
        "1",
        f"state-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}",
        record.recording_id,
        record.state,
        target,
        input_sha256s,
        config_sha256,
        tool_versions,
        started_at,
        finished_at,
        result,
    )
    return replace(record, state=target), event


@dataclass(frozen=True, slots=True)
class ReviewEvent:
    schema_version: str
    review_event_id: str
    entity_id: str
    field: str
    before: object
    after: object
    reason: str
    reviewer: str
    reviewed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
~~~

~~~python
# src/latintts/corpus/store.py
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank line {line_number} in {path}")
        raw = json.loads(line)
        if type(raw) is not dict:
            raise ValueError(f"line {line_number} must be a JSON object")
        rows.append(raw)
    return tuple(rows)


def _encode(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_encode(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl_event(path: Path, row: dict[str, Any]) -> None:
    existing = read_jsonl(path) if path.exists() else ()
    write_jsonl_atomic(path, (*existing, row))
~~~

~~~python
# src/latintts/corpus/intake.py
from __future__ import annotations

import csv
from pathlib import Path

from latintts.corpus.records import IntakeRow, RightsRecord
from latintts.corpus.store import read_jsonl

_INTAKE_FIELDS = (
    "relative_path",
    "title_or_citation",
    "speaker_id",
    "rights_id",
    "notes",
)


def read_intake(path: Path) -> tuple[IntakeRow, ...]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _INTAKE_FIELDS:
            raise ValueError("intake.csv must use the exact header")
        rows = tuple(IntakeRow(**row) for row in reader)
    if len({row.relative_path for row in rows}) != len(rows):
        raise ValueError("intake.csv contains duplicate relative_path values")
    return rows


def load_rights(path: Path) -> dict[str, RightsRecord]:
    records: dict[str, RightsRecord] = {}
    for raw in read_jsonl(path):
        record = RightsRecord.from_dict(raw)
        if not record.allow_local_processing or not record.allow_model_training:
            raise ValueError(f"{record.rights_id}: missing local or training authorization")
        if record.rights_id in records:
            raise ValueError(f"duplicate rights_id: {record.rights_id}")
        records[record.rights_id] = record
    return records
~~~

- [ ] **Step 4: Add record invariant tests**

~~~python
# tests/corpus/test_records.py
from latintts.corpus.domain import CorpusState
from latintts.corpus.records import AudioMetadata, RecordingRecord, advance_recording


def _recording(state: CorpusState) -> RecordingRecord:
    return RecordingRecord(
        schema_version="1",
        recording_id="rec-abc",
        relative_path="raw/spoken/a.wav",
        sha256="a" * 64,
        content_type="spoken",
        title_or_citation="Pater Noster",
        speaker_id="speaker-1",
        rights_id="rights-1",
        notes="",
        metadata=AudioMetadata(10.0, 48000, 1, "pcm_s16le", None),
        state=state,
    )


def test_recording_serialization_uses_state_value() -> None:
    record = _recording(CorpusState.INVENTORIED)
    assert record.to_dict()["state"] == "INVENTORIED"


def test_advance_recording_returns_event_and_new_immutable_record() -> None:
    record = _recording(CorpusState.DISCOVERED)
    updated, event = advance_recording(
        record,
        CorpusState.INVENTORIED,
        input_sha256s=("a" * 64,),
        config_sha256="b" * 64,
        tool_versions=("ffprobe=test",),
        started_at="2026-07-19T10:00:00+08:00",
        finished_at="2026-07-19T10:00:01+08:00",
        result="success",
    )
    assert record.state is CorpusState.DISCOVERED
    assert updated.state is CorpusState.INVENTORIED
    assert event.previous_state is CorpusState.DISCOVERED
    assert event.target_state is CorpusState.INVENTORIED
~~~

- [ ] **Step 5: Run manifest tests**

Run:

~~~powershell
pytest tests/corpus/test_store.py tests/corpus/test_intake.py tests/corpus/test_records.py -v
ruff check src/latintts/corpus tests/corpus
mypy src
~~~

Expected: all tests pass and static checks exit 0.

- [ ] **Step 6: Commit strict input records**

~~~powershell
git add src/latintts/corpus/records.py src/latintts/corpus/store.py src/latintts/corpus/intake.py tests/corpus
git commit -m "feat: add strict corpus input manifests"
~~~

---

### Task 3: Audio Discovery and Inventory

**Files:**
- Create: src/latintts/corpus/inventory.py
- Modify: src/latintts/corpus/cli.py
- Test: tests/corpus/test_inventory.py
- Test: tests/corpus/test_inventory_cli.py

**Interfaces:**
- Consumes: CorpusPaths, IntakeRow, RightsRecord, AudioMetadata, and subprocess.run-compatible injection.
- Produces: sha256_file(), probe_audio(), write_intake_skeleton(), inventory(), recordings.jsonl, and CLI inventory with --init-intake.

- [ ] **Step 1: Write failing ffprobe and inventory tests**

~~~python
# tests/corpus/test_inventory.py
import json
from pathlib import Path
from subprocess import CompletedProcess

from latintts.corpus.inventory import probe_audio, sha256_file


def test_probe_audio_uses_argument_array_and_parses_first_audio_stream(tmp_path: Path) -> None:
    audio = tmp_path / "name with spaces.wav"
    audio.write_bytes(b"RIFF")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(command)
        payload = {
            "format": {"duration": "12.5", "bit_rate": "768000"},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "sample_rate": "48000",
                    "channels": 1,
                }
            ],
        }
        return CompletedProcess(command, 0, json.dumps(payload), "")

    metadata = probe_audio(audio, run_command=fake_run)
    assert metadata.duration_seconds == 12.5
    assert metadata.sample_rate == 48000
    assert calls[0][-1] == str(audio)
    assert "-of" in calls[0]


def test_sha256_file_hashes_original_bytes(tmp_path: Path) -> None:
    path = tmp_path / "audio.bin"
    path.write_bytes(b"LatinTTS")
    assert sha256_file(path) == "405aa96624fbf5da83c15b35bdf3c2281312a87e47ef53a2fec298b751de601b"
~~~

- [ ] **Step 2: Run test and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_inventory.py -v
~~~

Expected: collection fails because inventory does not exist.

- [ ] **Step 3: Implement hashing, ffprobe parsing, and deterministic records**

~~~python
# src/latintts/corpus/inventory.py
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess

from latintts.corpus.domain import CorpusFailure, CorpusState
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import AudioMetadata, IntakeRow, RecordingRecord, RightsRecord

RunCommand = Callable[..., CompletedProcess[str]]
SUPPORTED_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_audio(path: Path, run_command: RunCommand = subprocess.run) -> AudioMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=codec_type,codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = run_command(
            command,
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise CorpusFailure("ALIGNER_UNAVAILABLE", "ffprobe not found in PATH") from error
    except subprocess.CalledProcessError as error:
        raise CorpusFailure("INVENTORY_UNSUPPORTED_FORMAT", error.stderr.strip()) from error
    raw = json.loads(completed.stdout)
    audio_streams = [item for item in raw["streams"] if item.get("codec_type") == "audio"]
    if not audio_streams:
        raise CorpusFailure("INVENTORY_UNSUPPORTED_FORMAT", f"no audio stream: {path}")
    stream = audio_streams[0]
    bit_rate = raw["format"].get("bit_rate")
    return AudioMetadata(
        duration_seconds=float(raw["format"]["duration"]),
        sample_rate=int(stream["sample_rate"]),
        channels=int(stream["channels"]),
        codec=str(stream["codec_name"]),
        bit_rate=int(bit_rate) if bit_rate is not None else None,
    )


def inventory_row(
    row: IntakeRow,
    rights: RightsRecord,
    paths: CorpusPaths,
    run_command: RunCommand = subprocess.run,
) -> RecordingRecord:
    source = paths.resolve_local(row.relative_path)
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise CorpusFailure(
            "INVENTORY_UNSUPPORTED_FORMAT",
            f"unsupported audio suffix: {source.suffix}",
        )
    digest = sha256_file(source)
    prefix = Path(row.relative_path).parts[:2]
    if prefix not in (("raw", "spoken"), ("raw", "sung")):
        raise ValueError("intake audio must be under raw/spoken or raw/sung")
    content_type = "sung" if prefix == ("raw", "sung") else "spoken"
    if rights.speaker_id != row.speaker_id:
        raise ValueError(f"speaker mismatch for {row.relative_path}")
    return RecordingRecord(
        schema_version="1",
        recording_id=f"rec-{digest[:12]}",
        relative_path=Path(row.relative_path).as_posix(),
        sha256=digest,
        content_type=content_type,
        title_or_citation=row.title_or_citation,
        speaker_id=row.speaker_id,
        rights_id=row.rights_id,
        notes=row.notes,
        metadata=probe_audio(source, run_command),
        state=CorpusState.INVENTORIED,
    )
~~~

- [ ] **Step 4: Add intake skeleton and CLI behavior tests**

~~~python
# tests/corpus/test_inventory_cli.py
from pathlib import Path

from latintts.corpus.inventory import write_intake_skeleton
from latintts.corpus.paths import CorpusPaths


def test_intake_skeleton_lists_spoken_and_sung_without_overwriting(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    (paths.raw_spoken / "spoken.wav").write_bytes(b"a")
    (paths.raw_sung / "chant.flac").write_bytes(b"b")
    output = paths.manifests / "intake.csv"
    write_intake_skeleton(paths, output)
    text = output.read_text(encoding="utf-8")
    assert "raw/spoken/spoken.wav" in text
    assert "raw/sung/chant.flac" in text
    assert write_intake_skeleton(paths, output) is False
~~~

Implement write_intake_skeleton() so it sorts POSIX relative paths, writes the exact intake header, leaves human fields empty, and returns False without modifying an existing file. Add inventory_from_manifests() that validates every rights_id, creates and appends the DISCOVERED → INVENTORIED ProcessingEvent for each file, writes recordings.jsonl atomically, writes the events beneath the run directory, and rejects duplicate recording IDs.

Register these commands in cli.py:

~~~text
python -m latintts.corpus inventory --init-intake
python -m latintts.corpus inventory
~~~

The first creates only intake.csv. The second requires intake.csv and rights.jsonl and writes recordings.jsonl.

- [ ] **Step 5: Run inventory tests**

Run:

~~~powershell
pytest tests/corpus/test_inventory.py tests/corpus/test_inventory_cli.py -v
ruff check src/latintts/corpus tests/corpus
mypy src
~~~

Expected: all tests pass.

- [ ] **Step 6: Commit inventory**

~~~powershell
git add src/latintts/corpus/inventory.py src/latintts/corpus/cli.py tests/corpus
git commit -m "feat: inventory local corpus audio"
~~~

---

### Task 4: Representative Pilot Selection

**Files:**
- Create: src/latintts/corpus/selection.py
- Modify: src/latintts/corpus/cli.py
- Test: tests/corpus/test_selection.py

**Interfaces:**
- Consumes: tuple of RecordingRecord and optional explicit IDs.
- Produces: PilotSelection, select_pilot(), pilot-selection.json, and CLI select-pilot.

- [ ] **Step 1: Write failing selection tests**

~~~python
# tests/corpus/test_selection.py
from latintts.corpus.selection import select_pilot

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


def test_explicit_selection_requires_two_or_three_spoken_ids() -> None:
    records = (recording("one", 10.0), recording("two", 20.0))
    selection = select_pilot(records, explicit_ids=("two", "one"))
    assert selection.recording_ids == ("two", "one")
~~~

Create tests/corpus/factories.py with this reusable complete record factory:

~~~python
from __future__ import annotations

import hashlib

from latintts.corpus.domain import CorpusState
from latintts.corpus.records import AudioMetadata, RecordingRecord


def recording(
    recording_id: str,
    duration_seconds: float,
    *,
    content_type: Literal["spoken", "sung"] = "spoken",
) -> RecordingRecord:
    digest = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()
    return RecordingRecord(
        schema_version="1",
        recording_id=recording_id,
        relative_path=f"raw/{content_type}/{recording_id}.wav",
        sha256=digest,
        content_type=content_type,
        title_or_citation=recording_id,
        speaker_id="speaker-1",
        rights_id="rights-1",
        notes="",
        metadata=AudioMetadata(duration_seconds, 48000, 1, "pcm_s16le", None),
        state=CorpusState.INVENTORIED,
    )
~~~

- [ ] **Step 2: Run test and verify failure**

Run:

~~~powershell
pytest tests/corpus/test_selection.py -v
~~~

Expected: collection fails because selection does not exist.

- [ ] **Step 3: Implement deterministic selection**

~~~python
# src/latintts/corpus/selection.py
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from latintts.corpus.records import RecordingRecord


@dataclass(frozen=True, slots=True)
class PilotSelection:
    schema_version: str
    strategy: str
    recording_ids: tuple[str, ...]
    inventory_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["recording_ids"] = list(self.recording_ids)
        raw["inventory_hashes"] = list(self.inventory_hashes)
        return raw


def select_pilot(
    records: tuple[RecordingRecord, ...],
    explicit_ids: tuple[str, ...] = (),
) -> PilotSelection:
    spoken = {record.recording_id: record for record in records if record.content_type == "spoken"}
    if explicit_ids:
        if len(explicit_ids) not in (2, 3) or len(set(explicit_ids)) != len(explicit_ids):
            raise ValueError("explicit pilot selection must contain two or three unique IDs")
        missing = set(explicit_ids) - set(spoken)
        if missing:
            raise ValueError(f"pilot IDs are missing or not spoken: {sorted(missing)}")
        chosen = tuple(spoken[item] for item in explicit_ids)
        strategy = "explicit-v1"
    else:
        ordered = sorted(spoken.values(), key=lambda item: (item.metadata.duration_seconds, item.recording_id))
        if len(ordered) < 2:
            raise ValueError("at least two spoken recordings are required")
        if len(ordered) <= 3:
            chosen = tuple(ordered)
        else:
            median = statistics.median(item.metadata.duration_seconds for item in ordered)
            middle = min(
                ordered[1:-1],
                key=lambda item: (
                    abs(item.metadata.duration_seconds - median),
                    item.metadata.duration_seconds,
                    item.recording_id,
                ),
            )
            chosen = (ordered[0], middle, ordered[-1])
        strategy = "duration-short-median-long-v1"
    return PilotSelection(
        schema_version="1",
        strategy=strategy,
        recording_ids=tuple(item.recording_id for item in chosen),
        inventory_hashes=tuple(item.sha256 for item in chosen),
    )
~~~

- [ ] **Step 4: Wire CLI and atomic selection output**

Add select-pilot with repeatable --recording-id. Load recordings.jsonl through strict RecordingRecord decoding, call select_pilot(), and atomically write a one-row pilot-selection.json file under manifests. Refuse to overwrite a selection whose inventory hashes differ unless --replace is supplied.

Run:

~~~powershell
pytest tests/corpus/test_selection.py -v
python -m latintts.corpus --help
~~~

Expected: tests pass and help lists select-pilot.

- [ ] **Step 5: Commit selection**

~~~powershell
git add src/latintts/corpus/selection.py src/latintts/corpus/cli.py tests/corpus
git commit -m "feat: select representative pilot recordings"
~~~

---

### Task 5: Source Text, Spoken Units, Pronunciation Plans, and ASR Diagnostics

**Files:**
- Create: src/latintts/corpus/transcripts.py
- Modify: src/latintts/corpus/records.py
- Modify: src/latintts/corpus/cli.py
- Test: tests/corpus/test_transcripts.py
- Test: tests/corpus/test_transcript_cli.py

**Interfaces:**
- Consumes: selected recording IDs, source snapshot files, line-delimited spoken unit files, Pronouncer.default(), and optional ASR hypothesis text.
- Produces: TextCandidate, SpokenUnit, TextDifference, AsrObservation, TranscriptRecord, build_transcript(), pronunciation_plan_to_dict(), transcript-intake.jsonl, transcripts.jsonl, and CLI prepare-text.

- [ ] **Step 1: Write failing transcript invariants**

~~~python
# tests/corpus/test_transcripts.py
from latintts.corpus.transcripts import (
    build_text_candidate,
    build_transcript,
    compare_asr_observation,
)


def test_build_transcript_preserves_three_text_layers_and_units() -> None:
    candidate = build_text_candidate(
        source_id="vulgate-source",
        source_url="https://example.invalid/source",
        source_version="edition-1",
        accessed_at="2026-07-19",
        source_text="Pater noster, qui es in caelis.",
    )
    transcript = build_transcript(
        recording_id="rec-1",
        source_candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
        spoken_unit_lines=("Pater noster", "qui es in caelis"),
    )
    assert transcript.source_text == "Pater noster, qui es in caelis."
    assert transcript.spoken_text == "Pater noster\nqui es in caelis"
    assert [unit.text for unit in transcript.spoken_units] == [
        "Pater noster",
        "qui es in caelis",
    ]
    assert [
        (unit.token_start_index, unit.token_end_index)
        for unit in transcript.spoken_units
    ] == [(0, 2), (2, 6)]
    assert transcript.normalized_text == "Pater noster qui es in caelis"
    assert transcript.pronunciation_plan["rule_version"] == "ecclesiastical-roman-v1"


def test_asr_observation_never_mutates_spoken_text() -> None:
    observation = compare_asr_observation("pater noster", "pater poster")
    assert observation.hypothesis == "pater poster"
    assert observation.differences
    assert observation.confirmed_text is None
~~~

- [ ] **Step 2: Run test and verify missing API failure**

Run:

~~~powershell
pytest tests/corpus/test_transcripts.py -v
~~~

Expected: collection fails because transcript APIs do not exist.

- [ ] **Step 3: Add transcript records and exact pronunciation serialization**

~~~python
# src/latintts/corpus/transcripts.py
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from collections.abc import Mapping
from typing import Any

from latintts.domain import PronunciationOverride, PronunciationPlan
from latintts.normalization import tokenize_words
from latintts.pipeline import Pronouncer


@dataclass(frozen=True, slots=True)
class TextCandidate:
    candidate_id: str
    source_id: str
    source_url: str
    source_version: str
    accessed_at: str
    source_sha256: str
    source_text: str


@dataclass(frozen=True, slots=True)
class SpokenUnit:
    unit_id: str
    ordinal: int
    text: str
    token_start_index: int
    token_end_index: int


@dataclass(frozen=True, slots=True)
class TextDifference:
    operation: str
    source_tokens: tuple[str, ...]
    spoken_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    candidate_id: str
    differences_from_selected: tuple[TextDifference, ...]


@dataclass(frozen=True, slots=True)
class AsrObservation:
    hypothesis: str
    differences: tuple[TextDifference, ...]
    confirmed_text: None = None


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    schema_version: str
    recording_id: str
    source_candidates: tuple[TextCandidate, ...]
    selected_candidate_id: str
    source_text: str
    spoken_text: str
    spoken_units: tuple[SpokenUnit, ...]
    normalized_text: str
    pronunciation_plan: dict[str, Any]
    differences: tuple[TextDifference, ...]
    candidate_comparisons: tuple[CandidateComparison, ...]
    state: str


def pronunciation_plan_to_dict(plan: PronunciationPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "rule_version": plan.rule_version,
        "original_text": plan.original_text,
        "normalized_text": plan.normalized_text,
        "tokens": [
            {
                "surface": token.surface,
                "normalized": token.normalized,
                "source_span": list(token.source_span),
                "syllables": list(token.syllables),
                "stress_index": token.stress_index,
                "ipa": token.ipa,
                "model_phonemes": list(token.model_phonemes),
                "resolution_method": token.resolution_method.value,
                "source_ids": list(token.source_ids),
                "warning_codes": [warning.code for warning in token.warnings],
            }
            for token in plan.tokens
        ],
        "phrase_phonemes": list(plan.phrase_phonemes),
        "warning_codes": [warning.code for warning in plan.warnings],
    }


def _differences(source: str, spoken: str) -> tuple[TextDifference, ...]:
    source_tokens = source.split()
    spoken_tokens = spoken.split()
    matcher = SequenceMatcher(a=source_tokens, b=spoken_tokens, autojunk=False)
    return tuple(
        TextDifference(tag, tuple(source_tokens[i1:i2]), tuple(spoken_tokens[j1:j2]))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def build_text_candidate(
    *,
    source_id: str,
    source_url: str,
    source_version: str,
    accessed_at: str,
    source_text: str,
) -> TextCandidate:
    if not all(value.strip() for value in (source_id, source_url, source_version, accessed_at, source_text)):
        raise ValueError("text candidate fields must not be empty")
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    identity = hashlib.sha256(
        "\0".join((source_id, source_version, digest)).encode("utf-8")
    ).hexdigest()[:16]
    return TextCandidate(
        f"text-{identity}",
        source_id,
        source_url,
        source_version,
        accessed_at,
        digest,
        source_text,
    )


def build_transcript(
    *,
    recording_id: str,
    source_candidates: tuple[TextCandidate, ...],
    selected_candidate_id: str,
    spoken_unit_lines: tuple[str, ...],
    pronunciation_overrides: Mapping[int, PronunciationOverride] | None = None,
    pronouncer: Pronouncer | None = None,
) -> TranscriptRecord:
    candidates = {candidate.candidate_id: candidate for candidate in source_candidates}
    if len(candidates) != len(source_candidates) or selected_candidate_id not in candidates:
        raise ValueError("text candidates must be unique and contain the selected candidate")
    selected = candidates[selected_candidate_id]
    units_text = tuple(line.strip() for line in spoken_unit_lines if line.strip())
    if not units_text:
        raise ValueError("spoken transcript must contain at least one non-empty unit")
    spoken_text = "\n".join(units_text)
    plan = (pronouncer or Pronouncer.default()).analyze(
        spoken_text,
        overrides=pronunciation_overrides,
    )
    units_list: list[SpokenUnit] = []
    token_cursor = 0
    for index, text in enumerate(units_text, 1):
        token_count = len(tokenize_words(text))
        units_list.append(
            SpokenUnit(
                f"{recording_id}-unit-{index:04d}",
                index,
                text,
                token_cursor,
                token_cursor + token_count,
            )
        )
        token_cursor += token_count
    units = tuple(units_list)
    if token_cursor != len(plan.tokens):
        raise ValueError("spoken unit token ranges do not cover PronunciationPlan tokens")
    return TranscriptRecord(
        schema_version="1",
        recording_id=recording_id,
        source_candidates=source_candidates,
        selected_candidate_id=selected_candidate_id,
        source_text=selected.source_text,
        spoken_text=spoken_text,
        spoken_units=units,
        normalized_text=plan.normalized_text,
        pronunciation_plan=pronunciation_plan_to_dict(plan),
        differences=_differences(selected.source_text, spoken_text),
        candidate_comparisons=tuple(
            CandidateComparison(
                candidate.candidate_id,
                _differences(selected.source_text, candidate.source_text),
            )
            for candidate in source_candidates
            if candidate.candidate_id != selected_candidate_id
        ),
        state="TRANSCRIPT_CONFIRMED",
    )


def compare_asr_observation(spoken_text: str, hypothesis: str) -> AsrObservation:
    return AsrObservation(hypothesis, _differences(spoken_text, hypothesis))
~~~

- [ ] **Step 4: Add prepare-text intake and explicit confirmation flow**

prepare-text --init reads pilot-selection.json and creates transcript-intake.jsonl entries with these exact fields:

~~~json
{
  "recording_id": "rec-000000000000",
  "source_candidates": [
    {
      "source_id": "",
      "source_url": "",
      "source_version": "",
      "accessed_at": "",
      "source_file": "",
      "selected": true
    }
  ],
  "spoken_units_file": "",
  "pronunciation_overrides_file": "",
  "confirmed": false,
  "asr_hypothesis_file": ""
}
~~~

The normal prepare-text command must:

1. Reject confirmed=false.
2. Require one or more source candidates and exactly one selected=true candidate.
3. Resolve all non-empty files beneath local-data.
4. Read every source_file as a full UTF-8 snapshot and retain every candidate in transcripts.jsonl.
5. Read spoken_units_file as one intended text unit per non-empty line.
6. When pronunciation_overrides_file is non-empty, parse a strict JSON object keyed by token index into PronunciationOverride values and pass it to Pronouncer.analyze().
7. Store ASR differences separately when asr_hypothesis_file is present.
8. Write transcripts.jsonl atomically.
9. Never copy an ASR hypothesis into spoken_text.
10. Append INVENTORIED → TEXT_CANDIDATES_READY after all candidate snapshots validate.
11. Append TEXT_CANDIDATES_READY → TRANSCRIPT_CONFIRMED only for confirmed=true, then atomically update recordings.jsonl.

The override file uses exact token indexes from the current spoken_text:

~~~json
{
  "spoken_text_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "overrides": {
    "3": {
      "stress_index": 1,
      "ipa": null,
      "model_phonemes": null
    }
  }
}
~~~

Reject non-integer keys, empty override objects, unknown fields, and an override file whose stored spoken-text hash no longer matches.
When Pronouncer returns warnings and no matching override file resolves them, prepare-text writes a pronunciation-review template containing the current hash, token indexes, surfaces, candidate stress, and warning codes, then leaves the transcript confirmed but ineligible for final approval.

Add tests/corpus/test_transcript_cli.py that proves unconfirmed input fails, path escape fails, and confirmed input produces TRANSCRIPT_CONFIRMED.

- [ ] **Step 5: Run transcript and existing pronunciation tests**

Run:

~~~powershell
pytest tests/corpus/test_transcripts.py tests/corpus/test_transcript_cli.py tests/unit/test_pipeline.py -v
ruff check src/latintts/corpus tests/corpus
mypy src
~~~

Expected: all tests pass.

- [ ] **Step 6: Commit transcript registry**

~~~powershell
git add src/latintts/corpus/transcripts.py src/latintts/corpus/records.py src/latintts/corpus/cli.py tests/corpus
git commit -m "feat: register source and spoken transcripts"
~~~

---

### Task 6: Analysis Audio, Candidate Clips, and PCM Quality Metrics

**Files:**
- Create: src/latintts/corpus/audio.py
- Test: tests/corpus/test_audio.py

**Interfaces:**
- Consumes: RecordingRecord, CorpusPaths, CorpusConfig, and subprocess.run-compatible injection.
- Produces: PcmMetrics, DerivedAudio, derive_analysis_audio(), measure_pcm16(), extract_analysis_segment(), extract_review_wav(), and extract_lossless_segment().

- [ ] **Step 1: Write failing audio command and metric tests**

~~~python
# tests/corpus/test_audio.py
import struct
import wave
from pathlib import Path
from subprocess import CompletedProcess

from latintts.corpus.audio import measure_pcm16


def _write_pcm16(path: Path, samples: tuple[int, ...], sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def test_measure_pcm16_reports_peak_rms_clipping_and_silence(tmp_path: Path) -> None:
    path = tmp_path / "metrics.wav"
    _write_pcm16(path, (0, 0, 1000, -1000, 32767, -32768))
    metrics = measure_pcm16(path)
    assert metrics.sample_count == 6
    assert metrics.peak == 1.0
    assert metrics.clipped_sample_ratio == 2 / 6
    assert metrics.silent_sample_ratio == 2 / 6
    assert 0.57 < metrics.rms < 0.59
~~~

Add a derive_analysis_audio test with a fake command runner. Assert the exact argument sequence contains -nostdin, -ac 1, -ar 16000, -c:a pcm_s16le, uses no shell string, writes a temporary output, and atomically publishes only after the fake runner creates a valid WAV.

- [ ] **Step 2: Run test and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_audio.py -v
~~~

Expected: collection fails because audio does not exist.

- [ ] **Step 3: Implement PCM metrics**

~~~python
# src/latintts/corpus/audio.py
from __future__ import annotations

import array
import hashlib
import math
import os
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

from latintts.corpus.domain import CorpusFailure


@dataclass(frozen=True, slots=True)
class PcmMetrics:
    sample_count: int
    peak: float
    rms: float
    clipped_sample_ratio: float
    silent_sample_ratio: float


@dataclass(frozen=True, slots=True)
class DerivedAudio:
    relative_path: str
    sha256: str
    source_sha256: str
    config_sha256: str
    metrics: PcmMetrics


def measure_pcm16(path: Path, silence_threshold: int = 327) -> PcmMetrics:
    total = squared = clipped = silent = peak = 0
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("analysis WAV must be mono PCM16")
        while frames := handle.readframes(65536):
            samples = array.array("h")
            samples.frombytes(frames)
            if sys.byteorder != "little":
                samples.byteswap()
            for sample in samples:
                magnitude = abs(sample)
                total += 1
                squared += sample * sample
                peak = max(peak, magnitude)
                clipped += magnitude >= 32760
                silent += magnitude <= silence_threshold
    if total == 0:
        raise ValueError("analysis WAV contains no samples")
    return PcmMetrics(
        sample_count=total,
        peak=min(peak / 32768.0, 1.0),
        rms=math.sqrt(squared / total) / 32768.0,
        clipped_sample_ratio=clipped / total,
        silent_sample_ratio=silent / total,
    )
~~~

- [ ] **Step 4: Implement atomic FFmpeg derivation and three explicit extraction modes**

derive_analysis_audio() must use:

~~~python
command = [
    "ffmpeg",
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    str(source),
    "-vn",
    "-ac",
    "1",
    "-ar",
    "16000",
    "-c:a",
    "pcm_s16le",
    str(temporary),
]
~~~

All extraction functions place -ss and -to after -i for accurate decoding, reject end_seconds <= start_seconds, and publish atomically:

- extract_analysis_segment() reads the already-derived analysis WAV and writes 16 kHz mono pcm_s16le for candidate pairing.
- extract_review_wav() reads the raw recording and writes WAV, preserving sample rate and channels and using pcm_s24le for human listening.
- extract_lossless_segment() reads the raw recording and writes FLAC, preserving sample rate and channels with compression_level 5.

None uses loudnorm, denoise, dynamic compression, dereverb, or silence trimming. On cache hit, recompute the output hash and raise CACHE_ARTIFACT_INVALID when the recorded hash differs.

- [ ] **Step 5: Run audio tests and format checks**

Run:

~~~powershell
pytest tests/corpus/test_audio.py -v
ruff check src/latintts/corpus/audio.py tests/corpus/test_audio.py
ruff format --check src/latintts/corpus/audio.py tests/corpus/test_audio.py
mypy src
~~~

Expected: all commands exit 0.

- [ ] **Step 6: Commit audio derivation**

~~~powershell
git add src/latintts/corpus/audio.py tests/corpus/test_audio.py
git commit -m "feat: derive corpus analysis audio"
~~~

---

### Task 7: Silero VAD and Per-Recording Pause Classification

**Files:**
- Create: src/latintts/corpus/vad.py
- Create: src/latintts/corpus/pauses.py
- Modify: src/latintts/corpus/cli.py
- Test: tests/corpus/test_vad.py
- Test: tests/corpus/test_pauses.py
- Test: tests/corpus/test_segment_cli.py

**Interfaces:**
- Consumes: 16 kHz mono PCM WAV, VAD config, pause config, and a VadBackend.
- Produces: VadFrame, SpeechInterval, VadResult, SileroVadBackend.analyze(), PauseInterval, PauseAnalysis, classify_pauses(), segmentation run JSON, and CLI segment.

- [ ] **Step 1: Write failing pure pause tests**

~~~python
# tests/corpus/test_pauses.py
import pytest

from latintts.corpus.domain import CorpusFailure
from latintts.corpus.pauses import classify_pauses
from latintts.corpus.vad import SpeechInterval


def test_classify_pauses_separates_short_and_long_per_recording() -> None:
    speech = (
        SpeechInterval(0, 1000),
        SpeechInterval(1200, 2000),
        SpeechInterval(4000, 5000),
        SpeechInterval(5250, 6000),
        SpeechInterval(7800, 8500),
    )
    result = classify_pauses(
        speech,
        sample_rate=1000,
        minimum_gap_ms=100,
        minimum_gap_count=4,
        separation_ratio=1.8,
        maximum_iterations=50,
    )
    assert [pause.kind for pause in result.pauses] == ["short", "long", "short", "long"]
    assert result.long_median_seconds > result.short_median_seconds


def test_classify_pauses_reports_ambiguous_distribution() -> None:
    speech = tuple(SpeechInterval(index * 1200, index * 1200 + 1000) for index in range(5))
    with pytest.raises(CorpusFailure) as error:
        classify_pauses(
            speech,
            sample_rate=1000,
            minimum_gap_ms=100,
            minimum_gap_count=4,
            separation_ratio=1.8,
            maximum_iterations=50,
        )
    assert error.value.code == "PAUSE_CLASSES_AMBIGUOUS"
~~~

- [ ] **Step 2: Run pause test and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_pauses.py -v
~~~

Expected: collection fails because vad and pauses do not exist.

- [ ] **Step 3: Implement VAD contracts and deterministic log-space two-means**

~~~python
# src/latintts/corpus/vad.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VadFrame:
    start_sample: int
    probability: float


@dataclass(frozen=True, slots=True)
class SpeechInterval:
    start_sample: int
    end_sample: int

    def __post_init__(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("speech interval must be a positive half-open range")


@dataclass(frozen=True, slots=True)
class VadResult:
    backend: str
    model_version: str
    sample_rate: int
    frames: tuple[VadFrame, ...]
    speech_intervals: tuple[SpeechInterval, ...]


class VadBackend(Protocol):
    def analyze(self, audio_path: Path) -> VadResult: ...
~~~

~~~python
# src/latintts/corpus/pauses.py
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from latintts.corpus.domain import CorpusFailure
from latintts.corpus.vad import SpeechInterval


@dataclass(frozen=True, slots=True)
class PauseInterval:
    start_sample: int
    end_sample: int
    duration_seconds: float
    kind: str


@dataclass(frozen=True, slots=True)
class PauseAnalysis:
    pauses: tuple[PauseInterval, ...]
    threshold_seconds: float
    short_median_seconds: float
    long_median_seconds: float


def classify_pauses(
    speech: tuple[SpeechInterval, ...],
    *,
    sample_rate: int,
    minimum_gap_ms: int,
    minimum_gap_count: int,
    separation_ratio: float,
    maximum_iterations: int,
) -> PauseAnalysis:
    gaps = [
        (left.end_sample, right.start_sample)
        for left, right in zip(speech, speech[1:])
        if (right.start_sample - left.end_sample) * 1000 / sample_rate >= minimum_gap_ms
    ]
    if len(gaps) < minimum_gap_count:
        raise CorpusFailure("PAUSE_CLASSES_AMBIGUOUS", "too few eligible internal pauses")
    durations = [(end - start) / sample_rate for start, end in gaps]
    values = [math.log(value) for value in durations]
    ordered = sorted(values)
    centers = [ordered[len(ordered) // 4], ordered[(3 * len(ordered)) // 4]]
    labels = [0] * len(values)
    for _ in range(maximum_iterations):
        next_labels = [min(range(2), key=lambda item: abs(value - centers[item])) for value in values]
        groups = [[value for value, label in zip(values, next_labels) if label == item] for item in range(2)]
        if any(not group for group in groups):
            raise CorpusFailure("PAUSE_CLASSES_AMBIGUOUS", "pause cluster is empty")
        next_centers = [statistics.fmean(group) for group in groups]
        if next_labels == labels and next_centers == centers:
            break
        labels, centers = next_labels, next_centers
    medians = [statistics.median(duration for duration, label in zip(durations, labels) if label == item) for item in range(2)]
    short_label = 0 if medians[0] < medians[1] else 1
    short_median = medians[short_label]
    long_median = medians[1 - short_label]
    if long_median / short_median < separation_ratio:
        raise CorpusFailure("PAUSE_CLASSES_AMBIGUOUS", "pause clusters are not separated")
    threshold = math.sqrt(short_median * long_median)
    pauses = tuple(
        PauseInterval(start, end, duration, "short" if duration < threshold else "long")
        for (start, end), duration in zip(gaps, durations)
    )
    return PauseAnalysis(pauses, threshold, short_median, long_median)
~~~

- [ ] **Step 4: Add lazy Silero adapter tests and implementation**

Use monkeypatch to supply fake silero_vad functions. Prove analyze() resets model state, reads 16 kHz audio, evaluates consecutive 512-sample windows, records every probability, resets again, and converts get_speech_timestamps sample dictionaries into SpeechInterval.

The adapter imports silero_vad only inside _load_dependencies(). Missing packages raise CorpusFailure with code ALIGNER_UNAVAILABLE. It reports backend silero-vad and model_version 6.2.1.

- [ ] **Step 5: Wire segment command**

segment must, for each locked pilot recording:

1. Validate TRANSCRIPT_CONFIRMED.
2. Build or validate the analysis WAV.
3. Run the injected VAD backend.
4. Classify pauses using pause-profile-v1.
5. Atomically write alignments/runs/<config digest>/<recording id>/segmentation.json.
6. Preserve VAD frames, speech intervals, pause intervals, model version, input hashes, and config hash.
7. Leave the recording at its last successful state and record PAUSE_CLASSES_AMBIGUOUS when classification fails.
8. Append TRANSCRIPT_CONFIRMED → SEGMENTED only after segmentation.json validates, then atomically update recordings.jsonl.

Add an integration test with a fake VAD backend and generated WAV; assert a second identical segment run reuses the hash-valid result.

- [ ] **Step 6: Run segmentation tests**

Run:

~~~powershell
pytest tests/corpus/test_vad.py tests/corpus/test_pauses.py tests/corpus/test_segment_cli.py -v
ruff check src/latintts/corpus tests/corpus
mypy src
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit VAD and pauses**

~~~powershell
git add src/latintts/corpus/vad.py src/latintts/corpus/pauses.py src/latintts/corpus/cli.py tests/corpus
git commit -m "feat: segment corpus speech and pauses"
~~~

---

### Task 8: Word Alignment Contracts and Cache Keys

**Files:**
- Create: src/latintts/corpus/alignment.py
- Test: tests/corpus/test_alignment.py

**Interfaces:**
- Consumes: audio Path, audio SHA-256, spoken text, configuration, and a backend.
- Produces: AlignmentRequest, WordSpan, AlignmentResult, AlignmentBackend protocol, alignment_cache_key(), and validate_alignment().

- [ ] **Step 1: Write failing alignment invariant tests**

~~~python
# tests/corpus/test_alignment.py
from pathlib import Path

import pytest

from latintts.corpus.alignment import AlignmentRequest, AlignmentResult, WordSpan


def test_alignment_request_hash_separates_text_from_backend_text(tmp_path: Path) -> None:
    request = AlignmentRequest(
        audio_path=tmp_path / "take.wav",
        audio_sha256="a" * 64,
        spoken_text="Gratia plena",
        config_sha256="b" * 64,
    )
    assert request.text_sha256 != request.audio_sha256


def test_alignment_rejects_overlapping_word_spans() -> None:
    with pytest.raises(ValueError, match="non-overlapping"):
        AlignmentResult(
            backend="fake",
            backend_version="1",
            model_id="fake/model",
            model_revision="rev",
            model_license="test-only",
            alignment_text="gratia plena",
            words=(
                WordSpan("gratia", 0.0, 1.0, 0.9),
                WordSpan("plena", 0.9, 1.5, 0.8),
            ),
            coverage=1.0,
            mean_score=0.85,
            alignment_level="word",
            phoneme_timing_status="not_estimated",
            warnings=(),
            raw_output={},
        )
~~~

- [ ] **Step 2: Run test and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_alignment.py -v
~~~

Expected: collection fails because alignment does not exist.

- [ ] **Step 3: Implement exact contracts**

~~~python
# src/latintts/corpus/alignment.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class AlignmentRequest:
    audio_path: Path
    audio_sha256: str
    spoken_text: str
    config_sha256: str

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.spoken_text.encode("utf-8")).hexdigest()

    @property
    def cache_key(self) -> str:
        payload = "\0".join(
            (self.audio_sha256, self.text_sha256, self.config_sha256)
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class WordSpan:
    text: str
    start_seconds: float
    end_seconds: float
    score: float

    def __post_init__(self) -> None:
        if not self.text.strip() or self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("word span must contain text and a positive time range")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("word score must be between zero and one")


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    backend: str
    backend_version: str
    model_id: str
    model_revision: str
    model_license: str
    alignment_text: str
    words: tuple[WordSpan, ...]
    coverage: float
    mean_score: float
    alignment_level: Literal["word"]
    phoneme_timing_status: Literal["not_estimated"]
    warnings: tuple[str, ...]
    raw_output: dict[str, Any]

    def __post_init__(self) -> None:
        if any(left.end_seconds > right.start_seconds for left, right in zip(self.words, self.words[1:])):
            raise ValueError("word spans must be ordered and non-overlapping")
        if not 0.0 <= self.coverage <= 1.0 or not 0.0 <= self.mean_score <= 1.0:
            raise ValueError("alignment scores must be between zero and one")
        if self.alignment_level != "word" or self.phoneme_timing_status != "not_estimated":
            raise ValueError("pilot alignment must be word-level without phoneme timing")


class AlignmentBackend(Protocol):
    def align(self, request: AlignmentRequest) -> AlignmentResult: ...
~~~

- [ ] **Step 4: Add cache serialization tests**

Add result_to_dict() and result_from_dict() with exact fields and strict validation. A round trip must preserve tuple ordering. A changed audio hash, text, or config hash must change AlignmentRequest.cache_key.

- [ ] **Step 5: Run alignment tests**

Run:

~~~powershell
pytest tests/corpus/test_alignment.py -v
ruff check src/latintts/corpus/alignment.py tests/corpus/test_alignment.py
mypy src
~~~

Expected: all commands exit 0.

- [ ] **Step 6: Commit alignment contracts**

~~~powershell
git add src/latintts/corpus/alignment.py tests/corpus/test_alignment.py
git commit -m "feat: define corpus word alignment contracts"
~~~

---

### Task 9: Pinned MMS CTC Alignment Backend

**Files:**
- Create: src/latintts/corpus/mms_alignment.py
- Modify: src/latintts/corpus/cli.py
- Test: tests/corpus/test_mms_alignment.py
- Test: tests/corpus/test_alignment_smoke_cli.py

**Interfaces:**
- Consumes: AlignmentRequest and the alignment section of CorpusConfig.
- Produces: MmsCtcAligner, BackendRuntimeInfo, normalize_mms_output(), create_alignment_backend(), cached raw alignment JSON, and CLI align --smoke-test.

- [ ] **Step 1: Write failing adapter test with fake third-party modules**

~~~python
# tests/corpus/test_mms_alignment.py
from latintts.corpus.mms_alignment import normalize_mms_output


def test_mms_output_keeps_alignment_text_non_authoritative() -> None:
    result = normalize_mms_output(
        spoken_text="Grátia plena.",
        text_starred=("", "Grátia", "plena.", ""),
        tokens_starred=("", "g r a t i a", "p l e n a", ""),
        raw_results=(
            {"start": 0.1, "end": 0.7, "text": "Grátia", "score": -0.1},
            {"start": 0.7, "end": 1.4, "text": "plena.", "score": -0.2},
        ),
        backend_version="0.3.0",
        model_id="MahmoudAshraf/mms-300m-1130-forced-aligner",
        model_revision="f37ba6bf1673872e07519fb951866cb2a32a6d7f",
    )
    assert result.alignment_text == "gratia plena"
    assert [word.text for word in result.words] == ["Grátia", "plena."]
    assert result.phoneme_timing_status == "not_estimated"
    assert result.model_license == "CC-BY-NC-4.0"
~~~

- [ ] **Step 2: Run test and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_mms_alignment.py -v
~~~

Expected: collection fails because mms_alignment does not exist.

- [ ] **Step 3: Implement lazy pinned model loading**

~~~python
# src/latintts/corpus/mms_alignment.py
from __future__ import annotations

import importlib
import math
import statistics
from dataclasses import dataclass
from typing import Any

from latintts.corpus.alignment import AlignmentRequest, AlignmentResult, WordSpan
from latintts.corpus.domain import CorpusFailure


@dataclass(frozen=True, slots=True)
class BackendRuntimeInfo:
    torch_version: str
    transformers_version: str
    device_name: str
    peak_memory_bytes: int | None


class MmsCtcAligner:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        device: str,
        dtype: str,
        window_seconds: int,
        context_seconds: int,
        batch_size: int,
        dependencies: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self.dtype_name = dtype
        self.window_seconds = window_seconds
        self.context_seconds = context_seconds
        self.batch_size = batch_size
        self.dependencies = dependencies
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def _load(self) -> Any:
        if self.dependencies is not None:
            return self.dependencies
        try:
            torch = importlib.import_module("torch")
            hub = importlib.import_module("huggingface_hub")
            aligner = importlib.import_module("ctc_forced_aligner")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise CorpusFailure("ALIGNER_UNAVAILABLE", str(error)) from error
        model_path = hub.snapshot_download(
            repo_id=self.model_id,
            revision=self.model_revision,
        )
        dtype = getattr(torch, self.dtype_name)
        model = transformers.AutoModelForCTC.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(self.device).eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            word_delimiter_token=None,
        )
        return type(
            "MmsDependencies",
            (),
            {
                "torch": torch,
                "aligner": aligner,
                "model": model,
                "tokenizer": tokenizer,
            },
        )()
~~~

- [ ] **Step 4: Implement alignment call and normalized output**

align() must call, in order:

1. load_audio(request.audio_path, model.dtype, model.device)
2. generate_emissions(model, waveform, 30, 2, 4)
3. preprocess_text(request.spoken_text, True, "lat", "word", "edges")
4. get_alignments()
5. get_spans()
6. postprocess_results()

Convert backend scores from negative log values or scalar tensors to a stable 0–1 value using max(0.0, min(1.0, exp(score))) when score is negative, otherwise clamp directly. Preserve text_starred display tokens in WordSpan.text. Build alignment_text as " ".join(token.replace(" ", "") for each non-empty tokens_starred item), so the backend representation remains visible but cannot overwrite spoken_text. Coverage is returned non-empty display words divided by expected spoken words. Convert every tensor/NumPy scalar in raw backend segments to Python int/float/string before storing raw_output. If no words are returned, raise ALIGNMENT_LOW_CONFIDENCE.

Use this normalization boundary:

~~~python
def _probability(value: object) -> float:
    number = float(value.item()) if hasattr(value, "item") else float(value)
    return max(0.0, min(1.0, math.exp(number) if number < 0 else number))


def normalize_mms_output(
    *,
    spoken_text: str,
    text_starred: tuple[str, ...],
    tokens_starred: tuple[str, ...],
    raw_results: tuple[dict[str, object], ...],
    backend_version: str,
    model_id: str,
    model_revision: str,
) -> AlignmentResult:
    if not raw_results:
        raise CorpusFailure("ALIGNMENT_LOW_CONFIDENCE", "MMS returned no word spans")
    words = tuple(
        WordSpan(
            text=str(item["text"]),
            start_seconds=float(item["start"]),
            end_seconds=float(item["end"]),
            score=_probability(item["score"]),
        )
        for item in raw_results
    )
    expected = tuple(item for item in text_starred if item)
    alignment_text = " ".join(item.replace(" ", "") for item in tokens_starred if item)
    raw_output = {
        "spoken_text": spoken_text,
        "segments": [
            {
                "text": word.text,
                "start": word.start_seconds,
                "end": word.end_seconds,
                "raw_score": (
                    float(item["score"].item())
                    if hasattr(item["score"], "item")
                    else float(item["score"])
                ),
                "normalized_score": word.score,
            }
            for item, word in zip(raw_results, words)
        ],
    }
    return AlignmentResult(
        backend="mms-ctc",
        backend_version=backend_version,
        model_id=model_id,
        model_revision=model_revision,
        model_license="CC-BY-NC-4.0",
        alignment_text=alignment_text,
        words=words,
        coverage=min(1.0, len(words) / max(1, len(expected))),
        mean_score=statistics.fmean(word.score for word in words),
        alignment_level="word",
        phoneme_timing_status="not_estimated",
        warnings=(),
        raw_output=raw_output,
    )
~~~

MmsCtcAligner.align() delegates without changing spoken_text:

~~~python
def align(self, request: AlignmentRequest) -> AlignmentResult:
    dependencies = self._load()
    waveform = dependencies.aligner.load_audio(
        str(request.audio_path),
        dependencies.model.dtype,
        dependencies.model.device,
    )
    emissions, stride = dependencies.aligner.generate_emissions(
        dependencies.model,
        waveform,
        self.window_seconds,
        self.context_seconds,
        self.batch_size,
    )
    tokens_starred, text_starred = dependencies.aligner.preprocess_text(
        request.spoken_text,
        True,
        "lat",
        "word",
        "edges",
    )
    segments, scores, blank = dependencies.aligner.get_alignments(
        emissions,
        tokens_starred,
        dependencies.tokenizer,
    )
    spans = dependencies.aligner.get_spans(tokens_starred, segments, blank)
    raw_results = dependencies.aligner.postprocess_results(
        text_starred,
        spans,
        stride,
        scores,
    )
    return normalize_mms_output(
        spoken_text=request.spoken_text,
        text_starred=tuple(text_starred),
        tokens_starred=tuple(tokens_starred),
        raw_results=tuple(raw_results),
        backend_version="0.3.0",
        model_id=self.model_id,
        model_revision=self.model_revision,
    )
~~~

- [ ] **Step 5: Add smoke-test command**

align --smoke-test loads the pinned model without real user audio, reports CUDA availability, device name, peak allocated memory, code package version, and model revision. It writes environment.txt and smoke.json under local-data/derived/corpus-v1/alignments/runtime/ with:

- Python version
- pip freeze output
- nvidia-smi query output
- torch and transformers versions
- code commit and model revision
- model license
- timestamp and success/failure code

External commands use argument arrays. The test injects fake module loaders and fake nvidia-smi output; it must not download a model in CI.

- [ ] **Step 6: Run adapter tests**

Run:

~~~powershell
pytest tests/corpus/test_mms_alignment.py tests/corpus/test_alignment_smoke_cli.py -v
ruff check src/latintts/corpus tests/corpus
mypy src
~~~

Expected: all tests pass without importing torch in tests that do not request the backend.

- [ ] **Step 7: Commit MMS adapter**

~~~powershell
git add src/latintts/corpus/mms_alignment.py src/latintts/corpus/cli.py tests/corpus
git commit -m "feat: add pinned MMS corpus aligner"
~~~

---

### Task 10: Text-Unit Mapping and Two-Take Pairing

**Files:**
- Create: src/latintts/corpus/pairing.py
- Modify: src/latintts/corpus/cli.py
- Test: tests/corpus/test_pairing.py
- Test: tests/corpus/test_pair_cli.py

**Interfaces:**
- Consumes: SpokenUnit sequence, SpeechInterval sequence, PauseAnalysis, extracted candidate WAVs, and AlignmentBackend evidence.
- Produces: TextUnitWindow, SplitEvidence, TakeCandidate, RepetitionGroup, map_text_units(), choose_split(), pair_recording(), cached pairing.json, and CLI pair.

- [ ] **Step 1: Write failing pairing tests**

~~~python
# tests/corpus/test_pairing.py
import pytest

from latintts.corpus.domain import CorpusFailure
from latintts.corpus.pairing import SplitEvidence, TextUnitWindow, choose_split


def test_choose_split_selects_unique_two_take_candidate() -> None:
    unit = TextUnitWindow("unit-1", "Pater noster", 0, 10000, ((4500, 5000),))
    evidence = (
        SplitEvidence(4750, 0.92, 0.90, True, 0.95),
    )
    group = choose_split(
        "rec-1",
        unit,
        evidence,
        sample_rate=1000,
        minimum_duration_ratio=0.65,
        maximum_duration_ratio=1.35,
        minimum_score_margin=0.02,
    )
    assert group.repetition_group_id == "rec-1-unit-1"
    assert [take.take_index for take in group.takes] == [1, 2]
    assert group.takes[0].end_sample == 4750
    assert group.takes[1].start_sample == 4750


def test_choose_split_rejects_text_mismatch() -> None:
    unit = TextUnitWindow("unit-1", "Pater noster", 0, 10000, ((4500, 5000),))
    with pytest.raises(CorpusFailure) as error:
        choose_split(
            "rec-1",
            unit,
            (SplitEvidence(4750, 0.9, 0.9, False, 0.95),),
            sample_rate=1000,
            minimum_duration_ratio=0.65,
            maximum_duration_ratio=1.35,
            minimum_score_margin=0.02,
        )
    assert error.value.code == "TAKE_TEXT_MISMATCH"
~~~

- [ ] **Step 2: Run test and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_pairing.py -v
~~~

Expected: collection fails because pairing does not exist.

- [ ] **Step 3: Implement pairing domain and pure selection**

~~~python
# src/latintts/corpus/pairing.py
from __future__ import annotations

from dataclasses import dataclass

from latintts.corpus.domain import CorpusFailure


@dataclass(frozen=True, slots=True)
class TextUnitWindow:
    unit_id: str
    text: str
    start_sample: int
    end_sample: int
    short_pause_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class SplitEvidence:
    split_sample: int
    first_alignment_score: float
    second_alignment_score: float
    word_order_same: bool
    coverage: float

    @property
    def rank(self) -> tuple[float, float]:
        return (min(self.first_alignment_score, self.second_alignment_score), self.coverage)


@dataclass(frozen=True, slots=True)
class TakeCandidate:
    take_index: int
    start_sample: int
    end_sample: int


@dataclass(frozen=True, slots=True)
class RepetitionGroup:
    repetition_group_id: str
    recording_id: str
    unit_id: str
    text: str
    takes: tuple[TakeCandidate, TakeCandidate]
    selected_evidence: SplitEvidence


def choose_split(
    recording_id: str,
    unit: TextUnitWindow,
    evidence: tuple[SplitEvidence, ...],
    *,
    sample_rate: int,
    minimum_duration_ratio: float,
    maximum_duration_ratio: float,
    minimum_score_margin: float,
) -> RepetitionGroup:
    valid = tuple(item for item in evidence if item.word_order_same)
    if not valid:
        raise CorpusFailure("TAKE_TEXT_MISMATCH", f"no matching split for {unit.unit_id}")
    ordered = sorted(valid, key=lambda item: item.rank, reverse=True)
    if len(ordered) > 1 and ordered[0].rank[0] - ordered[1].rank[0] < minimum_score_margin:
        raise CorpusFailure("TAKE_COUNT_MISMATCH", f"ambiguous split for {unit.unit_id}")
    selected = ordered[0]
    first_duration = selected.split_sample - unit.start_sample
    second_duration = unit.end_sample - selected.split_sample
    ratio = first_duration / second_duration
    if not minimum_duration_ratio <= ratio <= maximum_duration_ratio:
        raise CorpusFailure("TAKE_DURATION_MISMATCH", f"take duration ratio {ratio:.3f}")
    takes = (
        TakeCandidate(1, unit.start_sample, selected.split_sample),
        TakeCandidate(2, selected.split_sample, unit.end_sample),
    )
    return RepetitionGroup(
        f"{recording_id}-{unit.unit_id}",
        recording_id,
        unit.unit_id,
        unit.text,
        takes,
        selected,
    )
~~~

- [ ] **Step 4: Map long-gap windows to spoken units**

Implement map_text_units() so:

- The first unit starts at the first speech sample.
- Each long pause midpoint separates adjacent text units.
- The final unit ends at the final speech sample.
- Short pause ranges remain candidates inside their containing unit.
- The number of windows must equal the number of SpokenUnit records; mismatch raises TRANSCRIPT_SPOKEN_MISMATCH.
- A text unit with no short-pause candidate raises TAKE_COUNT_MISMATCH.
- No chosen pair crosses a long pause.

Add explicit tests for each rule.

- [ ] **Step 5: Implement evidence scoring and pair command**

pair_recording() must extract each candidate side from the 16 kHz analysis copy, create two AlignmentRequest values using the same spoken unit text, run the injected backend, compare word order to the unit tokens, and cache results by request.cache_key. It passes SplitEvidence objects to choose_split() and writes pairing.json under the locked run directory. Failed groups remain in pairing.json with their stable issue code and every candidate score.

After every group has a selected pair or an explicitly preserved issue, pair appends SEGMENTED → PAIRED and atomically updates recordings.jsonl.

The normal align command reads selected groups, reuses the cached selected AlignmentResult values, and writes alignment.json. It never recomputes a hash-valid cache. After every selected take has a valid result or explicit alignment issue, align appends PAIRED → ALIGNED and atomically updates recordings.jsonl.

- [ ] **Step 6: Run pairing tests**

Run:

~~~powershell
pytest tests/corpus/test_pairing.py tests/corpus/test_pair_cli.py -v
ruff check src/latintts/corpus tests/corpus
mypy src
~~~

Expected: normal, duration mismatch, text mismatch, multiple candidate, and count mismatch tests all pass.

- [ ] **Step 7: Commit repetition pairing**

~~~powershell
git add src/latintts/corpus/pairing.py src/latintts/corpus/cli.py tests/corpus
git commit -m "feat: pair repeated Latin readings"
~~~

---

### Task 11: Human Review Bundles and Append-Only Corrections

**Files:**
- Create: src/latintts/corpus/review.py
- Modify: src/latintts/corpus/records.py
- Modify: src/latintts/corpus/cli.py
- Test: tests/corpus/test_review.py
- Test: tests/corpus/test_review_cli.py

**Interfaces:**
- Consumes: paired takes, AlignmentResult, source audio, and review-decision JSON.
- Produces: write_textgrid(), read_textgrid(), export_review_bundle(), import_review_bundle(), replay_review_events(), review.jsonl, and CLI export-review/import-review.

- [ ] **Step 1: Write failing TextGrid round-trip test**

~~~python
# tests/corpus/test_review.py
from pathlib import Path

from latintts.corpus.alignment import WordSpan
from latintts.corpus.review import read_textgrid, write_textgrid


def test_textgrid_round_trip_preserves_take_and_word_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "take.TextGrid"
    words = (
        WordSpan("Pater", 0.1, 0.7, 0.9),
        WordSpan("noster", 0.7, 1.4, 0.8),
    )
    write_textgrid(path, duration_seconds=1.5, take_start=0.0, take_end=1.5, words=words)
    result = read_textgrid(path)
    assert result.take_start == 0.0
    assert result.take_end == 1.5
    assert [(word.text, word.start_seconds, word.end_seconds) for word in result.words] == [
        ("Pater", 0.1, 0.7),
        ("noster", 0.7, 1.4),
    ]
~~~

- [ ] **Step 2: Run test and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_review.py -v
~~~

Expected: collection fails because review does not exist.

- [ ] **Step 3: Implement deterministic long TextGrid writer and strict reader**

write_textgrid() creates a Praat long-text TextGrid with two IntervalTier values:

- take: one interval spanning the effective clip boundary and labeled take.
- words: ordered intervals for each word, with empty intervals inserted for gaps.

Escape embedded double quotes by doubling them. read_textgrid() accepts only the generated two-tier structure, validates non-negative ordered intervals, and rejects changed tier names, duplicate word intervals, overlapping words, or take bounds outside the WAV duration.

The returned ReviewedBoundaries contains take_start, take_end, and word spans without alignment scores; original automatic scores stay in AlignmentResult.

- [ ] **Step 4: Write failing append-only review replay test**

~~~python
from latintts.corpus.records import ReviewEvent
from latintts.corpus.review import replay_review_events


def test_review_replay_preserves_automatic_values_and_applies_events() -> None:
    automatic = {
        "segment_start": 0.0,
        "segment_end": 1.4,
        "review_decision": "unreviewed",
    }
    events = (
        ReviewEvent(
            "1",
            "event-boundary",
            "take-1",
            "segment_end",
            1.4,
            1.5,
            "final consonant retained",
            "owner",
            "2026-07-19T12:00:00+08:00",
        ),
        ReviewEvent(
            "1",
            "event-decision",
            "take-1",
            "review_decision",
            "unreviewed",
            "approved",
            "listened in full",
            "owner",
            "2026-07-19T12:00:00+08:00",
        ),
    )
    effective = replay_review_events(automatic, events)
    assert automatic["segment_end"] == 1.4
    assert effective["segment_end"] == 1.5
    assert effective["review_decision"] == "approved"
~~~

- [ ] **Step 5: Implement export/import and event replay**

export-review must create one directory per repetition group containing:

- take-1.wav and take-2.wav extracted from the original source without loudness processing.
- take-1.TextGrid and take-2.TextGrid.
- automatic.json with text layers, hashes, quality metrics, warnings, alignment versions, and candidate scores.
- decision.json initialized with decision set to unreviewed, empty reason, reviewer, and reviewed_at.

import-review rejects unreviewed, requires a non-empty reason/reviewer/timestamp, compares TextGrid values with automatic.json, and appends one ReviewEvent per changed field plus one review_decision event. Re-importing the identical decision is idempotent by a deterministic event hash. replay_review_events() returns the effective boundary and decision without mutating the automatic record.

When every exported take has an approved or rejected human decision, import-review appends ALIGNED → REVIEWED and atomically updates recordings.jsonl. Partial review leaves the recording in ALIGNED.

- [ ] **Step 6: Run review tests**

Run:

~~~powershell
pytest tests/corpus/test_review.py tests/corpus/test_review_cli.py -v
ruff check src/latintts/corpus tests/corpus
mypy src
~~~

Expected: round-trip, boundary edit, rejection, invalid tier, missing reason, and idempotent import tests all pass.

- [ ] **Step 7: Commit review workflow**

~~~powershell
git add src/latintts/corpus/review.py src/latintts/corpus/records.py src/latintts/corpus/cli.py tests/corpus
git commit -m "feat: add corpus human review workflow"
~~~

---

### Task 12: Approved Clip Extraction and Master Segment Manifest

**Files:**
- Create: src/latintts/corpus/manifest.py
- Modify: src/latintts/corpus/records.py
- Modify: src/latintts/corpus/cli.py
- Test: tests/corpus/test_manifest.py
- Test: tests/corpus/test_manifest_cli.py

**Interfaces:**
- Consumes: automatic pair/alignment records, effective ReviewEvent values, TranscriptRecord pronunciation plans, rights, raw recording, and CorpusConfig.
- Produces: SegmentRecord, require_approval(), build_approved_segments(), lossless derived clips, segments.jsonl, and CLI build-manifest.

- [ ] **Step 1: Write failing approval-gate tests**

~~~python
import pytest

from latintts.corpus.domain import CorpusFailure
from latintts.corpus.manifest import require_approval


def test_manifest_rejects_take_without_human_approval() -> None:
    with pytest.raises(CorpusFailure) as error:
        require_approval(review_decision=None, pronunciation_warning_codes=())
    assert error.value.code == "REVIEW_REQUIRED"


def test_manifest_rejects_unresolved_pronunciation_warning() -> None:
    with pytest.raises(CorpusFailure) as error:
        require_approval(
            review_decision="approved",
            pronunciation_warning_codes=("PRONUNCIATION_NEEDS_REVIEW",),
        )
    assert error.value.code == "PRONUNCIATION_NEEDS_REVIEW"


def test_manifest_accepts_human_approved_warning_free_take() -> None:
    assert (
        require_approval(
            review_decision="approved",
            pronunciation_warning_codes=(),
        )
        is None
    )
~~~

- [ ] **Step 2: Run tests and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_manifest.py -v
~~~

Expected: collection fails because manifest does not exist.

- [ ] **Step 3: Add SegmentRecord with exact provenance**

SegmentRecord must include:

~~~python
@dataclass(frozen=True, slots=True)
class SegmentRecord:
    schema_version: str
    corpus_version: str
    config_sha256: str
    segment_id: str
    recording_id: str
    text_unit_id: str
    repetition_group_id: str
    take_index: int
    source_audio_sha256: str
    source_start_sample: int
    source_end_sample: int
    derived_audio_relative_path: str
    derived_audio_sha256: str
    source_text_id: str
    spoken_text: str
    normalized_text: str
    pronunciation_schema_version: str
    rule_version: str
    ipa_by_token: tuple[str, ...]
    model_phonemes_by_token: tuple[tuple[str, ...], ...]
    word_spans: tuple[WordSpan, ...]
    alignment_backend: str
    alignment_model_id: str
    alignment_model_revision: str
    alignment_model_license: str
    alignment_level: Literal["word"]
    phoneme_timing_status: Literal["not_estimated"]
    quality_metrics: PcmMetrics
    quality_metric_audio_sha256: str
    quality_labels: tuple[str, ...]
    review_event_ids: tuple[str, ...]
    rights_id: str
    split: Literal["unassigned"]
~~~

Validate take_index in (1, 2), positive non-overlapping source samples, non-empty model phonemes, approved human event, known rights, no unresolved pronunciation warnings, and compatible schema/rule versions.

For each segment, locate its SpokenUnit and slice the full pronunciation-plan token list with token_start_index:token_end_index. ipa_by_token and model_phonemes_by_token come only from that slice. Reject an empty slice, an out-of-range index, or a token count that differs from the words in the unit text.

Implement the approval gate exactly:

~~~python
def require_approval(
    *,
    review_decision: str | None,
    pronunciation_warning_codes: tuple[str, ...],
) -> None:
    if review_decision != "approved":
        raise CorpusFailure("REVIEW_REQUIRED", "take lacks an approved human review event")
    if pronunciation_warning_codes:
        raise CorpusFailure(
            "PRONUNCIATION_NEEDS_REVIEW",
            ", ".join(pronunciation_warning_codes),
        )
~~~

- [ ] **Step 4: Implement final extraction and atomic manifest build**

For every approved take:

1. Replay review events.
2. Convert effective seconds to source sample positions using original sample rate.
3. Call extract_lossless_segment() on the raw recording, never the 16 kHz analysis copy.
4. Re-probe and hash the derived lossless clip.
5. Measure quality on the matching 16 kHz PCM analysis interval and record that metric source hash.
6. Construct SegmentRecord.
7. Sort by recording ID, text-unit ordinal, and take index.
8. Atomically replace manifests/segments.jsonl only after every row validates.
9. Append REVIEWED → APPROVED when at least one take is approved and every take is decided; append REVIEWED → REJECTED only when every take is rejected.

Rejected and unreviewed entities remain in review and run artifacts but do not enter segments.jsonl. No file is deleted.

- [ ] **Step 5: Wire build-manifest CLI and tests**

The command must validate schema_version=1, corpus-v1, and ecclesiastical-roman-v1 before consuming records. Add a test that corrupts a source audio file after inventory and expects INVENTORY_HASH_MISMATCH before any clip is written.

Run:

~~~powershell
pytest tests/corpus/test_manifest.py tests/corpus/test_manifest_cli.py -v
ruff check src/latintts/corpus tests/corpus
mypy src
~~~

Expected: all tests pass.

- [ ] **Step 6: Commit master manifest**

~~~powershell
git add src/latintts/corpus/manifest.py src/latintts/corpus/records.py src/latintts/corpus/cli.py tests/corpus
git commit -m "feat: build approved corpus manifest"
~~~

---

### Task 13: Pilot Report, End-to-End Fake Pipeline, and Operator Guide

**Files:**
- Create: src/latintts/corpus/report.py
- Modify: src/latintts/corpus/cli.py
- Create: tests/corpus/test_report.py
- Create: tests/integration/test_corpus_pilot_pipeline.py
- Create: docs/corpus/alignment-pilot-operator-guide.md
- Modify: README.md

**Interfaces:**
- Consumes: recordings, VAD artifacts, pairing artifacts, review events, segments, runtime telemetry, and full-corpus spoken inventory duration.
- Produces: PilotMetrics, ScaleDecision, build_report(), report.json, report.md, CLI report, and an exact operator workflow.

- [ ] **Step 1: Write failing threshold tests**

~~~python
# tests/corpus/test_report.py
from latintts.corpus.report import PilotMetrics, classify_scale_readiness


def test_scale_readiness_is_scalable_at_all_green_boundaries() -> None:
    metrics = PilotMetrics(
        input_duration_seconds=600.0,
        speech_duration_seconds=500.0,
        auto_pairing_correct_ratio=0.95,
        boundary_unchanged_ratio=0.85,
        review_minutes_per_audio_minute=3.0,
        approved_speech_ratio=0.80,
        gpu_realtime_factor=0.2,
        peak_gpu_memory_bytes=4_000_000_000,
    )
    assert classify_scale_readiness(metrics).value == "scalable"


def test_any_red_metric_makes_batch_processing_not_ready() -> None:
    metrics = PilotMetrics(
        input_duration_seconds=600.0,
        speech_duration_seconds=500.0,
        auto_pairing_correct_ratio=0.84,
        boundary_unchanged_ratio=0.90,
        review_minutes_per_audio_minute=2.0,
        approved_speech_ratio=0.90,
        gpu_realtime_factor=0.2,
        peak_gpu_memory_bytes=4_000_000_000,
    )
    assert classify_scale_readiness(metrics).value == "not_ready"
~~~

- [ ] **Step 2: Run test and verify missing-module failure**

Run:

~~~powershell
pytest tests/corpus/test_report.py -v
~~~

Expected: collection fails because report does not exist.

- [ ] **Step 3: Implement metrics and decision bands**

~~~python
# src/latintts/corpus/report.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class ScaleDecision(str, Enum):
    SCALABLE = "scalable"
    OPTIMIZE = "optimize"
    NOT_READY = "not_ready"


@dataclass(frozen=True, slots=True)
class PilotMetrics:
    input_duration_seconds: float
    speech_duration_seconds: float
    auto_pairing_correct_ratio: float
    boundary_unchanged_ratio: float
    review_minutes_per_audio_minute: float
    approved_speech_ratio: float
    gpu_realtime_factor: float
    peak_gpu_memory_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_scale_readiness(metrics: PilotMetrics) -> ScaleDecision:
    red = (
        metrics.auto_pairing_correct_ratio < 0.85
        or metrics.boundary_unchanged_ratio < 0.70
        or metrics.review_minutes_per_audio_minute > 6.0
        or metrics.approved_speech_ratio < 0.60
    )
    if red:
        return ScaleDecision.NOT_READY
    green = (
        metrics.auto_pairing_correct_ratio >= 0.95
        and metrics.boundary_unchanged_ratio >= 0.85
        and metrics.review_minutes_per_audio_minute <= 3.0
        and metrics.approved_speech_ratio >= 0.80
    )
    return ScaleDecision.SCALABLE if green else ScaleDecision.OPTIMIZE
~~~

build_report() must also include counts by issue code and rejection reason, approved/rejected/unreviewed durations, text preparation time, review time, GPU retry count, projected full-corpus person-hours, projected GPU hours, and projected storage bytes. Divide only after explicit non-zero denominator checks.

- [ ] **Step 4: Add full fake-backend integration test**

~~~python
# tests/integration/test_corpus_pilot_pipeline.py
def test_synthetic_two_take_pipeline_is_reproducible(tmp_path, fake_tools) -> None:
    project = fake_tools.create_project(tmp_path)
    fake_tools.create_two_take_recording(project, title="Pater Noster")
    assert fake_tools.run(project, "inventory", "--init-intake") == 0
    fake_tools.complete_intake_and_rights(project)
    assert fake_tools.run(project, "inventory") == 0
    assert fake_tools.run(project, "select-pilot", "--recording-id", "rec-one", "--recording-id", "rec-two") == 0
    fake_tools.complete_transcript_intake(project)
    assert fake_tools.run(project, "prepare-text") == 0
    assert fake_tools.run(project, "segment") == 0
    assert fake_tools.run(project, "pair") == 0
    assert fake_tools.run(project, "align") == 0
    assert fake_tools.run(project, "export-review") == 0
    fake_tools.approve_all_reviews(project)
    assert fake_tools.run(project, "import-review") == 0
    assert fake_tools.run(project, "build-manifest") == 0
    assert fake_tools.run(project, "report") == 0
    first = fake_tools.hash_outputs(project)
    assert fake_tools.run(project, "report") == 0
    assert fake_tools.hash_outputs(project) == first
~~~

Implement fake_tools with generated PCM16 WAV, fake ffprobe/ffmpeg adapters, fake VAD, and fake alignment backend. It must never import Silero, torch, or Transformers.

- [ ] **Step 5: Write the exact operator guide**

docs/corpus/alignment-pilot-operator-guide.md must contain:

1. Install or expose ffmpeg and ffprobe, then verify doctor.
2. Create .venv-corpus using Python 3.10.
3. Install requirements/corpus.txt and the CUDA-compatible PyTorch wheel selected for the machine.
4. Run align --smoke-test and inspect runtime/smoke.json.
5. Copy audio into local-data/raw/spoken and local-data/raw/sung.
6. Generate and complete intake.csv and rights.jsonl.
7. Run inventory and select-pilot.
8. Generate transcript-intake.jsonl, save source snapshots, and mark one spoken unit per line.
9. Run prepare-text, segment, pair, align, and export-review.
10. Listen to both takes, edit TextGrid boundaries, fill decision.json, and run import-review.
11. Run build-manifest and report.
12. Verify local-data is ignored and no real files are staged.
13. Recover from every stable error code without deleting raw data.

Include the exact command sequence:

~~~powershell
python -m latintts.corpus doctor
python -m latintts.corpus inventory --init-intake
python -m latintts.corpus inventory
python -m latintts.corpus select-pilot
python -m latintts.corpus prepare-text --init
python -m latintts.corpus prepare-text
python -m latintts.corpus segment
python -m latintts.corpus pair
python -m latintts.corpus align
python -m latintts.corpus export-review
python -m latintts.corpus import-review
python -m latintts.corpus build-manifest
python -m latintts.corpus report
~~~

Link this guide from README.md.

- [ ] **Step 6: Run the complete verification gate**

Run:

~~~powershell
pytest -q
pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95
ruff check .
ruff format --check .
mypy src
python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
git check-ignore -v local-data/raw/spoken/example.wav
git status --short
~~~

Expected:

- All tests pass.
- Coverage is at least 95%.
- Ruff, format check, and mypy exit 0.
- Gold audit prints gold-audit: PASS with zero errors.
- git check-ignore identifies the local-data rule.
- git status shows only intentional tracked implementation changes before commit.

- [ ] **Step 7: Commit reporting and documentation**

~~~powershell
git add src/latintts/corpus/report.py src/latintts/corpus/cli.py tests/corpus tests/integration/test_corpus_pilot_pipeline.py docs/corpus/alignment-pilot-operator-guide.md README.md
git commit -m "feat: report corpus alignment pilot quality"
~~~

---

## Spec Coverage Map

| Approved specification area | Implemented by |
| --- | --- |
| Goals, 2–3-file boundary, spoken-only scope | Tasks 3–5 and Post-Implementation Local Pilot Run |
| Immutable raw data, ignored local-data, relative paths | Tasks 1, 3, 6, 12 |
| Rights scope and release unknowns | Task 2 |
| Ordered state machine, no silent skips, transition provenance | Tasks 1–3, 5, 7, 10–12 |
| Short/median/long pilot selection | Task 4 |
| source_text, spoken_text, normalized_text, alignment_text separation | Tasks 5 and 9 |
| ASR diagnostic-only rule | Task 5 |
| 16 kHz analysis copy and raw-derived lossless final clips | Tasks 6 and 12 |
| PCM quality metrics without denoise or loudness processing | Task 6 |
| Silero VAD 6.2.1 probability frames and per-recording pause classes | Task 7 |
| Two valid takes, duration ratio, mismatch preservation | Task 10 |
| Backend-neutral word alignment contract | Task 8 |
| Pinned MMS CTC backend, CUDA telemetry, CC BY-NC recording | Task 9 |
| Word-level boundary claim and no phoneme timestamps | Tasks 8, 9, and 12 |
| Praat TextGrid review, before/after history, no auto approval | Task 11 |
| Versioned segments.jsonl and unassigned split | Task 12 |
| Stable error codes and resumable content-addressed artifacts | Tasks 1–12 |
| Synthetic/fake CI and local-only real backend smoke test | Tasks 6–9 and 13 |
| Green/yellow/red metrics and full-corpus estimate | Task 13 |
| Intermediate human checkpoints and exact command workflow | Task 13 and Post-Implementation Local Pilot Run |
| 4–7-day pilot outcome without TTS training | Entire plan; completion stops after the local report |

---

## Post-Implementation Local Pilot Run

This run uses private data and therefore creates no Git commit.

- [ ] Confirm the implementation branch is clean and local-data is ignored.
- [ ] Install FFmpeg/ffprobe only after confirming the chosen Windows installation method with the user; rerun doctor until both executables and optional packages pass.
- [ ] Run align --smoke-test on the RTX 4080 and retain smoke.json, environment.txt, model revision, real-time factor, and peak memory under local-data.
- [ ] Copy, do not move, the original files into raw/spoken and raw/sung while retaining the external backup.
- [ ] Complete intake and rights records; keep release permissions unknown unless explicitly granted.
- [ ] Run inventory and inspect the proposed short/median/long pilot selection before locking it.
- [ ] Save each selected source snapshot and one-line-per-unit spoken transcript, then explicitly confirm transcript intake.
- [ ] Run segment and listen to the proposed long-gap units before running pair.
- [ ] Run pair and inspect every mismatch; do not delete abnormal repetition groups.
- [ ] Run align and export-review; listen to every take and edit boundaries in Praat.
- [ ] Import all review decisions, then build segments.jsonl.
- [ ] Run report and compare every metric with the approved green/yellow/red table.
- [ ] Report the approved duration, issue distribution, human minutes per audio minute, GPU performance, projected full-corpus effort, and recommendation to the user.
- [ ] Stop after the report. Do not start full-corpus processing, MMS-TTS baseline work, or TTS training without a new approved plan.

## Plan Completion Criteria

The plan is complete only when:

- The 13 implementation task commits exist in order on an isolated codex/ branch.
- The full verification gate passes.
- The private pilot produces human-approved takes and a schema-valid segments.jsonl.
- report.json and report.md contain every required metric and a deterministic scale decision.
- No file beneath local-data is tracked or staged.
- The user receives the pilot outcome and full-corpus estimate before any model-training work begins.
