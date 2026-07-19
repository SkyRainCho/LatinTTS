# Task 11 Review Remediation Round 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make correction recovery validate every existing processing-event-bound checkpoint in a pure read-only preflight before writing any missing correction stage.

**Architecture:** Split recovery into a read-only planning phase and a write phase. The preflight computes expected automatic/corrected pairing bytes and prospective review journal bytes, validates existing artifact identities, locates any existing human PAIRED event, and validates that event against its already-durable prefix before `review.jsonl` or pairing artifacts can be changed.

**Tech Stack:** Python 3.10, canonical JSON/JSONL, SHA-256, frozen dataclasses, failure-injection pytest, Ruff, strict mypy, coverage.py.

## Global Constraints

- Strict RED -> GREEN for the mutation-order defect.
- No review journal, automatic pairing, corrected pairing, processing event, or recording state write may occur before preflight succeeds.
- An existing processing event is validated against current durable bytes, never prospective newly appended correction events.
- Invalid old-prefix recovery leaves every relevant file byte-for-byte unchanged.
- All previously supported legal crash-recovery windows remain green.
- Exact coverage remains at least 95.00% with two-decimal reporting.

---

### Task 1: Reproduce preflight mutation defect

**Files:**
- Modify: `tests/corpus/test_review_cli.py`

**Interfaces:**
- Uses the event-before-recordings crash fixture.
- Produces a regression where current `review.jsonl` contains only a valid unrelated old-prefix event, the correction event is absent and would become `new_events`, while the existing processing event binds that old prefix.

- [x] Add the distinct recovery regression and snapshot `review.jsonl`, `pairing-automatic.json`, `pairing.json`, `processing-events.jsonl`, and `recordings.jsonl` before retry.
- [x] Run it and verify the current code raises only after changing `review.jsonl`, so the byte-for-byte assertion fails.

### Task 2: Pure read-only recovery preflight

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- Produces an immutable preflight result per `_PairingCorrectionSubmission` containing the chosen review snapshot hash and validated prefix correction events.
- Existing processing events are validated before the write phase; missing processing events validate the prospective full review journal without writing it.

- [x] Build canonical prospective journal bytes from `existing + new_events` in memory.
- [x] For every submission, validate current automatic/current pairing identities and any existing matching processing event against current durable review bytes.
- [x] If no processing event exists, validate the prospective journal bytes through the same strict snapshot logic without publishing them.
- [x] Move all `review.jsonl` and pairing artifact writes after all submissions pass preflight.
- [x] Run the new regression and all legal recovery-window tests; confirm invalid recovery changes no bytes and legal recovery remains idempotent.
- [x] Commit the functional fix.

### Task 3: Report statistics and final gates

**Files:**
- Modify: `.superpowers/sdd/task-11-report.md`
- Modify: this plan

**Interfaces:**
- Reports focused review results as separate passed/skipped counts, explicitly identifying Windows alias skips.

- [x] Run focused review tests and record exact passed/skipped totals.
- [ ] Run full pytest with exact coverage, Ruff check, Ruff format check, strict mypy, gold audit, `git diff --check`, and clean-status checks.
- [ ] Update abbreviated remediation commit IDs, exact coverage, and focused pass/skip wording.
- [ ] Commit documentation and preserve the worktree without push or merge.

## Self-Review

- Spec coverage: Task 1 proves the pre-write mutation; Task 2 makes all validation read-only and retains legal recovery; Task 3 corrects statistics and all gates.
- Placeholder scan: each step names exact durable files, validation order, and expected failure/pass evidence; no deferred behavior remains.
- Type consistency: the preflight result reuses `ReviewEvent`, `_PairingCorrectionSubmission`, and existing hash strings; persisted schemas remain unchanged.
