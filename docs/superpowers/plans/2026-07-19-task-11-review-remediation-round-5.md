# Task 11 Review Remediation Round 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure a recovered pairing transition always binds the exact durable `review.jsonl` bytes that were validated, including semantically valid non-canonical JSON whitespace.

**Architecture:** Keep the existing all-submission read-only preflight. When no review events need publication, use the current journal's raw bytes as the prospective snapshot; when events do need publication, retain the canonical planned bytes. Before any transition is persisted, verify that the durable journal exactly equals those preflighted bytes.

**Tech Stack:** Python 3.11+, pytest, Ruff, mypy, coverage.py

## Global Constraints

- Follow strict RED → GREEN TDD.
- Preserve the previous invalid-old-prefix zero-write guarantee.
- Preserve every legal pairing-correction recovery window.
- Full coverage must remain at least `95.00%`.
- Do not merge, push, or create a PR.

---

### Task 1: Bind missing-transition recovery to durable review bytes

**Files:**
- Modify: `tests/corpus/test_review_cli.py`
- Modify: `src/latintts/corpus/review.py`

**Interfaces:**
- Consumes: `_import_pairing_corrections(...)`, `_pairing_review_snapshot_sha256(...)`, `validate_pairing_review_snapshot(...)`.
- Produces: a recovered `ProcessingEvent.input_sha256s[3]` equal to `sha256(review_path.read_bytes())` when `new_events == 0`.

- [x] **Step 1: Write the failing whitespace-preserving recovery test**

```python
def test_pairing_correction_retry_binds_new_transition_to_durable_review_whitespace(...):
    # Fail before persist_recording_transition so journal and pairing artifacts are durable,
    # but no PAIRED ProcessingEvent or recording-state update exists.
    # Rewrite each semantic review JSON row with non-canonical whitespace.
    # Retry, then assert the new event's fourth hash equals sha256(raw durable bytes)
    # and validate_pairing_review_snapshot accepts that exact snapshot.
```

- [x] **Step 2: Run the new test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/corpus/test_review_cli.py::test_pairing_correction_retry_binds_new_transition_to_durable_review_whitespace -q`

Expected: FAIL because the fourth input hash equals the canonical re-encoding digest instead of the durable whitespace-preserving journal digest.

- [x] **Step 3: Implement the minimal durable-bytes selection and guard**

```python
prospective_review_bytes = (
    _review_events_bytes(prospective_events)
    if new_events
    else review_path.read_bytes()
)

if review_path.read_bytes() != prospective_review_bytes:
    raise ValueError("review journal differs from recovery preflight")
```

Use the selected bytes for the existing strict prefix/correction preflight. Place the equality guard after any planned journal publication and before artifact/transition persistence.

- [x] **Step 4: Verify GREEN and recovery regressions**

Run the new test, then all pairing correction retry/preflight tests. Expected: PASS, including invalid-old-prefix byte-for-byte immutability.

- [x] **Step 5: Commit the code and regression test**

```powershell
git add src/latintts/corpus/review.py tests/corpus/test_review_cli.py docs/superpowers/plans/2026-07-19-task-11-review-remediation-round-5.md
git commit -m "fix: bind recovery to durable review bytes"
```

### Task 2: Verify and report the fifth remediation

**Files:**
- Modify: `.superpowers/sdd/task-11-report.md`

**Interfaces:**
- Consumes: fresh focused/full test and quality-gate outputs.
- Produces: actual Task 11 verification statistics and remediation history.

- [ ] **Step 1: Run focused review tests and full coverage suite**

Run review-focused pytest, then full pytest with `--cov-fail-under=95`. Record exact pass/skip counts and coverage.

- [ ] **Step 2: Run quality gates**

Run Ruff check/format, strict mypy, gold audit, and `git diff --check`.

- [ ] **Step 3: Update the report with actual results**

Record the fifth remediation, RED symptom, GREEN behavior, all test counts, skip reasons, and coverage values.

- [ ] **Step 4: Commit and verify final branch state**

```powershell
git add .superpowers/sdd/task-11-report.md docs/superpowers/plans/2026-07-19-task-11-review-remediation-round-5.md
git commit -m "docs: report durable review recovery"
.venv\Scripts\python.exe -m pytest -q
git status --short
```

Expected: final pytest passes; working tree is clean.
