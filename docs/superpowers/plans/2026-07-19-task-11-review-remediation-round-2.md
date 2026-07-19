# Task 11 Review Remediation Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make human pairing correction crash-recoverable and repeatable, bind alignment to deterministic immutable review history, unify TextGrid duration with trusted WAV frames, and prove literal coverage of at least 95.00%.

**Architecture:** Treat the original pairing artifact, deterministic correction events, corrected pairing bytes, and processing event as a recoverable transaction whose durable checkpoints are validated before advancing state. Reconstruct legal correction entities from the preserved original pairing and validate the current review log as an append-only extension of the processing-event-bound snapshot. Derive review duration only from the validated PCM24 WAV frame count.

**Tech Stack:** Python 3.10, frozen dataclasses, canonical JSON/JSONL, pytest failure injection, coverage.py precision configuration, Ruff, strict mypy.

## Global Constraints

- Strict RED -> GREEN for every production behavior.
- `review.jsonl` remains durable before corrected pairing, processing transition, and `recordings.jsonl`.
- Retry may resume only when every earlier checkpoint exactly matches deterministic expected bytes and identities.
- No reviewed selection may be accepted without saved candidate evidence and a deterministic correction ReviewEvent.
- Coverage must use the raw percentage with precision at least two and be at least 95.00%, not an integer-rounded display.

---

### Task 1: Recover interrupted pairing corrections

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- Produces a correction recovery state reconstructed from `pairing-automatic.json`, deterministic `pairing_selected_split` events, current corrected `pairing.json`, and any existing processing event.
- Retry writes only the missing downstream checkpoint and rejects any byte or identity drift.

- [ ] Add failure-injection tests for an exception after corrected `pairing.json` replacement and after processing-event replacement but before `recordings.jsonl` replacement.
- [ ] Run both tests and verify SEGMENTED retry currently rejects the corrected artifact as stale.
- [ ] Implement strict checkpoint reconstruction and resume the missing processing/state writes.
- [ ] Run correction tests and verify both retries reach exactly one PAIRED transition without rewriting review history.
- [ ] Commit the recovery fix.

### Task 2: Idempotent repeated pair and immutable review snapshot binding

**Files:**
- Modify: `src/latintts/corpus/pairing.py`
- Modify: `src/latintts/corpus/cli.py`
- Test: `tests/corpus/test_pair_cli.py`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- Repeated `pair_corpus()` accepts a human-reviewed selection only after validating the original pairing, deterministic correction events, corrected pairing, and PAIRED processing event.
- Alignment validates the correction event hash and requires the processing event's fourth input digest to equal an immutable byte prefix of current `review.jsonl`; later valid appended events remain allowed.

- [ ] Add failing repeated-pair, event-ID tamper, prefix tamper, and valid-suffix tests.
- [ ] Add shared deterministic correction-event and review-prefix validators.
- [ ] Permit reviewed selection in pairing cache validation only through the validated correction context.
- [ ] Run pair/align tests and commit.

### Task 3: Validate pairing correction history during normal review import

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- Normal import reconstructs the exact legal pairing entity/split map from selected recordings' `pairing-automatic.json` artifacts and rejects orphan, duplicate, unsaved, or mismatched correction events.

- [ ] Add parameterized failing histories for orphan identity, duplicate correction, unsaved split, and corrected-pair mismatch.
- [ ] Implement one strict correction-history validator used before take-event replay.
- [ ] Run focused tests and commit.

### Task 4: Frame-derived review duration

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Test: `tests/corpus/test_review.py`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- `_export_review_audio()` returns validated provenance plus duration computed as actual WAV frames divided by trusted sample rate.
- Export and import compare TextGrid duration to this exact frame-derived value.

- [ ] Add a failing 44.1 kHz test whose 16 kHz boundary conversion rounds by one frame.
- [ ] Read the trusted WAV header after derivation and use its frame count as the sole duration source.
- [ ] Run audio/review tests and commit.

### Task 5: Literal coverage, report, and final gates

**Files:**
- Modify: `pyproject.toml`
- Modify: `.superpowers/sdd/task-11-report.md`
- Test: focused negative branches in `tests/corpus/test_review.py` and `tests/corpus/test_review_cli.py`

**Interfaces:**
- Coverage configuration sets `precision = 2` and `fail_under = 95.00`.

- [ ] Add valuable negative/recovery tests until exact covered statements divided by total statements is at least 95.00%.
- [ ] Run focused pairing/review/CLI tests.
- [ ] Run full pytest coverage, Ruff check, Ruff format check, strict mypy, gold audit, `git diff --check`, and clean-status checks.
- [ ] Update the report with exact two-decimal-or-better coverage and follow-up SHAs; correct entity-ID wording.
- [ ] Commit documentation/configuration and preserve the worktree without push or merge.

## Self-Review

- Spec coverage: Tasks 1-4 cover the Critical and five Important findings; Task 5 enforces the literal coverage requirement and all requested gates.
- Placeholder scan: every task names exact failure tests, checkpoint semantics, files, and commands; no deferred behavior remains.
- Type consistency: correction recovery continues using existing `PairingRecording`, `ReviewEvent`, and `ProcessingEvent` schemas; no new persisted schema is introduced.
