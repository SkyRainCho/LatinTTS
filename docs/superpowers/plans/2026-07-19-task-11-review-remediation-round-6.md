# Task 11 Review Remediation Round 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append pairing review events without changing any byte of the existing durable `review.jsonl` prefix.

**Architecture:** Read and strictly validate the current journal as one complete newline-terminated byte snapshot, preserving an absent or empty journal as the only empty-prefix cases. Build the plan as `durable_raw_bytes + canonical_new_event_bytes`, preflight that exact plan, atomically replace the journal with those exact bytes, and re-read it before persisting any pairing artifact or transition.

**Tech Stack:** Python 3.10+, pytest, Ruff, mypy, coverage.py

## Global Constraints

- Follow strict RED → GREEN TDD.
- Preserve deterministic duplicate, conflict, and idempotency semantics.
- Preserve the single-recording whitespace recovery, invalid-old-prefix zero-write behavior, and all legal recovery windows.
- A non-empty durable journal must end with `LF`; absent and zero-byte journals are legal empty prefixes.
- Full coverage must remain at least `95.00%`.
- Do not merge, push, or create a PR.

---

### Task 1: Preserve durable prefixes while appending corrections

**Files:**
- Modify: `tests/corpus/test_review_cli.py`
- Modify: `src/latintts/corpus/review.py`

**Interfaces:**
- Consumes: `_load_existing_review_events(...)`, `_review_snapshot_rows_from_bytes(...)`, `_review_events_bytes(...)`, `_write_bytes_atomic(...)`.
- Produces: an exact plan equal to the validated durable prefix plus canonical encodings of `new_events` only.

- [x] **Step 1: Write the failing two-recording test**

```python
def test_later_pairing_correction_preserves_whitespace_bound_review_prefix(...):
    # Recover recording A from non-canonical durable review bytes and retain its bound hash.
    # Confirm recording B later so it contributes new review events.
    # Assert the final journal starts byte-for-byte with A's original prefix and both
    # ProcessingEvent snapshot hashes pass validate_pairing_review_snapshot().
```

- [x] **Step 2: Run the new test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/corpus/test_review_cli.py::test_later_pairing_correction_preserves_whitespace_bound_review_prefix -q`

Expected: FAIL because the final journal no longer starts with A's non-canonical durable prefix, and A's bound hash is no longer a complete-line prefix.

- [x] **Step 3: Implement strict raw-prefix planning**

```python
durable_review_bytes = _complete_review_journal_bytes(review_path, existing)
prospective_review_bytes = durable_review_bytes + _review_events_bytes(tuple(new_events))
```

`_complete_review_journal_bytes` returns `b""` for an absent or zero-byte journal. For non-empty bytes it validates the full journal by hashing the whole byte string through `_review_snapshot_rows_from_bytes`, thereby requiring a complete final `LF`-terminated line, then verifies the decoded `ReviewEvent` tuple equals `existing`.

- [x] **Step 4: Atomically publish and verify exact planned bytes**

```python
if new_events:
    _write_bytes_atomic(review_path, prospective_review_bytes)
if review_path.read_bytes() != prospective_review_bytes:
    raise ValueError("review journal differs from recovery preflight")
```

Keep this re-read before automatic/corrected pairing writes and `persist_recording_transition`.

- [x] **Step 5: Verify GREEN and all recovery windows**

Run the new test, the single-recording whitespace test, invalid-old-prefix zero-write test, and all pairing correction retry/preflight tests. Expected: PASS.

- [x] **Step 6: Commit the implementation and tests**

```powershell
git add src/latintts/corpus/review.py tests/corpus/test_review_cli.py docs/superpowers/plans/2026-07-19-task-11-review-remediation-round-6.md
git commit -m "fix: preserve review journal byte prefixes"
```

### Task 2: Verify and report the sixth remediation

**Files:**
- Modify: `.superpowers/sdd/task-11-report.md`

**Interfaces:**
- Consumes: fresh focused/full test, coverage, and quality-gate outputs.
- Produces: the actual Task 11 final statistics and sixth remediation history.

- [ ] **Step 1: Run focused review tests and full coverage suite**

Run review-focused pytest, then full pytest with `--cov-fail-under=95`; record exact pass/skip and coverage totals.

- [ ] **Step 2: Run quality gates**

Run Ruff check/format, strict mypy, gold audit, and `git diff --check`.

- [ ] **Step 3: Update and commit the report**

Record the RED failure, preserved-prefix behavior, actual statistics, skip reasons, coverage, and abbreviated remediation commit.

- [ ] **Step 4: Verify the final committed branch state**

```powershell
.venv\Scripts\python.exe -m pytest -q
git status --short
```

Expected: full pytest passes and the linked worktree is clean.
