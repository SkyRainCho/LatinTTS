# Task 11 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every Critical and Important finding from the Task 11 independent review while preserving append-before-materialized-state durability and existing selected-pair review behavior.

**Architecture:** Keep `review.jsonl` as the immutable audit stream and treat review bundles and corrected pairing files as rebuildable materialized state. Reuse `audio.extract_review_wav()` as the only review decoder, bind its complete `DerivedAudio` provenance plus upstream pairing/alignment hashes into every bundle, and validate all path components from the canonical `local-data` root without following aliases. Add a controlled pairing-correction materialization that selects only saved `SplitEvidence` and becomes durable only after its human `ReviewEvent`.

**Tech Stack:** Python 3.10, frozen dataclasses, canonical JSON/JSONL, FFmpeg through the existing injected `RunCommand`, Praat long-text TextGrid, pytest, Ruff, strict mypy.

## Global Constraints

- Strict RED -> GREEN: every production change is preceded by a focused failing test and observed expected failure.
- Never add real corpus audio, decisions, runtime artifacts, or model files to Git; tests use generated media and controlled runners.
- `review.jsonl` is persisted before corrected pairing, processing events, or `recordings.jsonl` state.
- No unreviewed or unconfirmed take may contribute to `REVIEWED` or any later state.
- Final gates: focused review tests, full pytest with coverage >=95%, Ruff check/format, strict mypy, gold audit, diff check, clean status.

---

### Task 1: Canonical review paths and bounded TextGrid parsing

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Test: `tests/corpus/test_review.py`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- Produces: `_canonical_review_root(paths, config_sha256, recording_id)` and `_review_file(root, group_id, filename)` that validate every existing ancestor with `lstat`, reject reparse/symlink/junction/alias components, and require resolved paths beneath the reconstructed canonical local-data review root.
- Preserves: `read_textgrid(path) -> ReviewedBoundaries` but converts every truncation/index failure into `ValueError`.

- [ ] Write tests for review-root, group-directory, and file alias attacks, plus truncation after every long-TextGrid structural marker.
- [ ] Run those tests and verify alias cases currently succeed or fail for the wrong reason and truncations leak `IndexError`.
- [ ] Implement one component-walking canonical-path validator and cursor-based TextGrid reads with explicit EOF checks.
- [ ] Run the focused tests and verify GREEN.

### Task 2: Trusted review audio derivation and codec support

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Modify: `src/latintts/corpus/cli.py`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- `export_review_bundle(..., ffmpeg_version: str, run_command: RunCommand = subprocess.run)` calls `extract_review_wav()` for each take and stores exact `DerivedAudio.to_dict()` under `audio_provenance`.
- `import_review_bundle(..., run_command: RunCommand = subprocess.run)` reconstructs the expected derivation from trusted raw source, exact seconds, record layout, fixed PCM24 output config, cached provenance, and validates the bundle WAV bytes/format/sample count against that trusted artifact.

- [ ] Add a failing test that edits both bundle WAV bytes and editable automatic hash, proving import currently accepts the forgery.
- [ ] Add failing FLAC and lossy-input tests with a fake FFmpeg runner that emits exact PCM24 review WAV output.
- [ ] Replace the local `wave` clipper with `extract_review_wav()`, copy only the verified derived WAV into the bundle, and persist complete provenance.
- [ ] Rebuild/validate expected derived audio on import and compare bytes, PCM width/rate/channels/sample count/duration and provenance exactly.
- [ ] Run focused export/import/codec tests and verify GREEN.

### Task 3: Strict review-event replay invariants

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Test: `tests/corpus/test_review.py`

**Interfaces:**
- `replay_review_events(automatic, events, *, entity_id: str | None = None)` permits only `segment_start`, `segment_end`, `word:N:{text,start_seconds,end_seconds}`, and `review_decision`.
- Every event validates before/after runtime types, finite numbers, exact entity identity, and the complete post-event boundary model: fixed word count, ordered non-overlapping positive spans inside the take.

- [ ] Add parameterized failing histories for unknown fields, booleans/NaN/strings in numeric fields, invalid decisions, wrong entities, reversed/overlapping/out-of-clip spans, and malformed word indexes.
- [ ] Implement allowed-field parsing and call one full-boundary invariant validator before and after every event application.
- [ ] Run replay/import tests and verify malformed histories cannot advance state.

### Task 4: Immutable artifact identity and collision-free entities

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- Bundle automatic schema records exact `pairing_artifact_sha256`, `pairing_cache_key`, `pairing_integrity_sha256`, `alignment_artifact_sha256`, `alignment_cache_key`, and per-take result/cache/audio identities.
- `review_entity_id(recording_id, unit_id, take_index)` uses canonical length-prefixed UTF-8 components and a full SHA-256 digest; import globally rejects duplicate automatic entities and any duplicate identity across recordings/groups.

- [ ] Add failing stale-bundle tests by replacing pairing/alignment artifacts after export and collision tests using ambiguous hyphenated IDs.
- [ ] Add the exact immutable identity block to export, regenerate the bundle identity digest, and compare it against current strict artifacts on import.
- [ ] Replace concatenated entity IDs with structured digest identities and reject duplicates before reading decisions/history.
- [ ] Run focused stale/collision tests and verify GREEN.

### Task 5: Review-required pairing correction

**Files:**
- Modify: `src/latintts/corpus/pairing.py`
- Modify: `src/latintts/corpus/review.py`
- Modify: `src/latintts/corpus/cli.py`
- Test: `tests/corpus/test_review_cli.py`
- Test: `tests/corpus/test_pairing.py`

**Interfaces:**
- `materialize_reviewed_pairing(pairing, selections)` accepts exact `unit_id -> split_sample` choices, requires the split in saved candidate evidence, and returns a valid all-selected `PairingRecording` without recomputation or acoustic auto-selection.
- Version-2 `decision.json` adds exact `pairing_selections` rows with `unit_id`, `split_sample`, `reason`, `reviewer`, and timezone-aware `reviewed_at`.
- Import appends one deterministic `pairing_selected_split` ReviewEvent per correction, writes the original bytes once as `pairing-automatic.json`, writes corrected `pairing.json` only after events, and durably advances only corrected recordings from `SEGMENTED` to `PAIRED`. A later normal `align` run produces the bound alignment/review bundle.

- [ ] Add failing pure pairing tests for valid saved selection and rejection of absent/tampered candidates.
- [ ] Add a failing CLI flow for a review outcome: export correction template, choose a saved split with human metadata, import, verify audit-first writes and `PAIRED`, then align/export normal take review.
- [ ] Implement the pure pairing materializer, strict v2 decision schema, pairing correction event, original-artifact preservation, corrected artifact integrity, and state transition.
- [ ] Run correction and existing selected-pair review tests and verify no unconfirmed outcome advances.

### Task 6: Per-recording completion, verification, and report

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Modify: `.superpowers/sdd/task-11-report.md`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- `import_review_bundle()` computes completeness and `ALIGNED -> REVIEWED` independently per selected recording and returns true only when the complete selection is reviewed.

- [ ] Add a failing two-recording test where one recording is complete and one partial; assert the first becomes `REVIEWED` while the second remains `ALIGNED`.
- [ ] Refactor import to maintain per-recording entity/effective maps and persist each eligible transition independently after all new review events are durable.
- [ ] Run focused review tests with coverage and verify GREEN.
- [ ] Run the full pytest coverage gate, Ruff check, Ruff format check, strict mypy, gold audit, `git diff --check`, and inspect status.
- [ ] Update the Task 11 report with follow-up SHA(s), exact results, coverage, and residual risks; commit with focused remediation subjects and do not push.

## Self-Review

- Spec coverage: Tasks 1-5 cover Critical 1-6 and Important 8-9; Task 6 covers Important 7 and every required gate. Minor naming/exit-code cleanup is permitted only if a regression test demonstrates ambiguity.
- Placeholder scan: every production behavior above names its exact interface, validation, persistence order, and test command category; no deferred implementation item remains.
- Type consistency: review export/import receive `CorpusPaths`, `CorpusConfig`, explicit FFmpeg identity and injected `RunCommand`; pairing correction returns the existing strict `PairingRecording`; replay continues returning an effective plain JSON object.
