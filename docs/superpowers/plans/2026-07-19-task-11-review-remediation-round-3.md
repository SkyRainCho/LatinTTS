# Task 11 Review Remediation Round 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that the deterministic pairing correction events are contained inside the exact review-journal byte prefix bound by the human PAIRED processing event.

**Architecture:** Add one shared strict snapshot decoder in `review.py` that locates the exact newline-complete byte prefix by SHA-256, parses only those bytes as strict JSONL, and validates the resulting prefix events against the preserved automatic and corrected pairing artifacts. Both repeated pair/align validation and crash recovery call this helper; journal suffixes remain allowed but cannot supply missing correction evidence.

**Tech Stack:** Python 3.10, canonical JSONL, SHA-256, frozen dataclasses, pytest failure injection, Ruff, strict mypy, coverage.py.

## Global Constraints

- Strict RED -> GREEN for production behavior.
- The processing event's fourth input must bind a newline-complete exact prefix of `review.jsonl`.
- Required correction events must exist and validate inside that bound prefix, not merely somewhere in the current journal.
- Invalid prefix recovery must not modify pairing artifacts, processing events, review history, or recording state.
- `pairing-automatic.json` reads in repeated pair/align use component-wise canonical/reparse validation.
- Exact total coverage remains at least 95.00% with two-decimal reporting.

---

### Task 1: Reproduce stale-prefix acceptance

**Files:**
- Modify: `tests/corpus/test_review_cli.py`

**Interfaces:**
- Uses the existing corrected-pairing fixture and crash-after-processing-event fixture.
- Produces two regressions whose processing event binds a valid old journal prefix that excludes the later correction event.

- [x] Add an align test that prepends a valid unrelated event, rewrites the processing event's fourth hash and deterministic event ID to bind only that old prefix, appends the required correction event afterward, and asserts `align_corpus()` rejects it without changing artifacts or state.
- [x] Run the align regression and verify it fails because current validation accepts the old prefix.
- [x] Add a recovery test that creates the same old-prefix binding after an event-before-recordings crash and asserts retry rejects without changing any artifact or state.
- [x] Run the recovery regression and verify it fails because current recovery accepts the old prefix.

### Task 2: Shared strict bound-prefix validation

**Files:**
- Modify: `src/latintts/corpus/review.py`
- Modify: `src/latintts/corpus/cli.py`
- Test: `tests/corpus/test_review_cli.py`

**Interfaces:**
- Produces `validate_pairing_review_snapshot(review_path, snapshot_sha256, original, corrected) -> tuple[ReviewEvent, ...]`.
- The helper returns correction events validated from the exact bound prefix; malformed JSONL, partial-line prefixes, missing/orphan/duplicate/unsaved correction events, and non-deterministic IDs raise `ValueError`.

- [x] Implement exact prefix-byte discovery and strict JSONL decoding without reading correction evidence from the suffix.
- [x] Replace `_human_pairing_transition_exists()` full-journal correction validation plus boolean prefix search with the shared helper.
- [x] Replace recovery's `_review_snapshot_is_prefix()` check with the same helper and derive the deterministic processing-event timestamp from prefix correction events.
- [x] Run both new regressions and all pairing/review focused tests; verify no invalid retry writes occur.
- [x] Commit the functional fix.

### Task 3: Canonical automatic artifact reads and final verification

**Files:**
- Modify: `src/latintts/corpus/cli.py`
- Modify: `tests/corpus/test_review_cli.py`
- Modify: `.superpowers/sdd/task-11-report.md`
- Modify: this plan

**Interfaces:**
- Repeated pair and align reject `pairing-automatic.json` when any path component is an alias/reparse point or the file is not canonical and regular.

- [x] Add a failing canonical-path regression using a mocked reparse attribute on the automatic artifact path.
- [x] Validate `pairing-automatic.json` with `_require_canonical_descendant()` before every repeated-pair/align read.
- [ ] Change report wording from “original WAV” to “original source audio” and record abbreviated remediation commit IDs plus exact final coverage.
- [ ] Run full pytest coverage, Ruff check, Ruff format check, strict mypy, gold audit, `git diff --check`, and clean-status checks.
- [ ] Commit documentation and preserve the worktree without push or merge.

## Self-Review

- Spec coverage: Task 1 covers both required negative paths and no-write assertions; Task 2 enforces one shared bound-prefix proof; Task 3 covers canonical artifact reads, wording, coverage, and all gates.
- Placeholder scan: every step names an exact artifact, behavior, command family, and expected outcome; no deferred implementation remains.
- Type consistency: both consumers use the same `tuple[ReviewEvent, ...]` result, and no persisted schema changes.
