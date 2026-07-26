# Minimal Functional CI Design

## Goal

Restore useful CI quickly with minimal changes, prioritizing TTS, pronunciation, and MMS alignment
correctness over full cross-platform hardening of the historical corpus transaction suite.

## Scope

The change is limited to:

- `.github/workflows/ci.yml`
- `tests/corpus/test_audio_review.py`

No production TTS, G2P, stress, normalization, alignment, cache, or corpus implementation code is
changed.

## CI Architecture

The existing three check names remain:

- `quality`
- `Tests (windows-latest)`
- `Tests (ubuntu-latest)`

`quality` moves from Ubuntu to Windows and continues to run:

```text
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
git diff --check "$BASE_SHA...$HEAD_SHA"
```

It retains full checkout history, immutable action SHAs, `contents: read`, and the existing
event-aware diff range.

The Windows test job remains the release-confidence gate:

```text
python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q
```

The Ubuntu test job becomes a portable core-functionality gate:

```text
python -m pytest tests/unit tests/integration \
  tests/corpus/test_mms_alignment.py \
  tests/corpus/test_alignment_smoke_cli.py -q
```

The Ubuntu job does not apply a repository-wide coverage threshold because the historical corpus
suite and coverage annotations contain Windows-specific assumptions that require a separate
hardening project.

## Flaky Test Stabilization

`test_concurrent_same_key_has_one_producer_and_no_overwrite` currently releases the first producer
immediately after starting the second thread. A slow CI scheduler may not run the second thread
until after publication and lock removal, causing the strict cache validator to correctly reject an
unrecorded existing artifact.

The test will observe the second thread receiving `FileExistsError` from the real exclusive lock
attempt and release the producer only after that contention occurs. Production cache validation,
locking, and no-overwrite behavior remain unchanged.

## Functional Verification

Before pushing:

1. Run the stabilized concurrency test repeatedly.
2. Run the complete `tests/corpus/test_audio_review.py` module.
3. Run the exact Ubuntu portable core suite locally.
4. Run Ruff format, Ruff lint, mypy, gold audit, pip check, and `git diff --check`.
5. Run `latintts.corpus doctor`.
6. Run the real cached MMS alignment smoke test when the existing local corpus environment is
   available.

The remote Windows job supplies the final full-suite and 95% coverage evidence.

## Deferred Work

This change intentionally defers:

- Linux mypy support for Windows-only `ctypes` helpers.
- Full Ubuntu execution of the corpus transaction and file-handle suites.
- Independent repository-wide 95% coverage on Ubuntu.
- Platform-specific coverage accounting.
- The POSIX handling of Windows-drive absolute path input discovered during the first CI run.

These items do not change the current Latin pronunciation, G2P, stress, or MMS model revision
behavior and should be addressed in a separate cross-platform hardening change.

## Acceptance Criteria

- Only the workflow, the flaky test, and this specification change.
- Core Ubuntu tests pass.
- Windows full pytest reaches at least 95% coverage.
- `quality`, `Tests (windows-latest)`, and `Tests (ubuntu-latest)` all pass on PR #2.
- The PR remains Draft and Open.
- No production TTS or corpus implementation file changes.
