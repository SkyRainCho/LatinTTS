# Minimal Functional CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a reliable PR gate quickly by keeping full Windows coverage, running portable core functionality on Ubuntu, and removing the scheduler race from one concurrency test without changing production TTS code.

**Architecture:** Keep the existing three GitHub check names and the existing two-OS test matrix. Move static quality checks to Windows, select the full coverage command on Windows and the portable TTS/MMS subset on Ubuntu, and make the concurrency test wait until the second worker has observed the real exclusive-lock collision before releasing the producer.

**Tech Stack:** GitHub Actions, PowerShell, Python 3.10, pytest, pytest-cov, Ruff, mypy

## Global Constraints

- Production files under `src/` must not change.
- The only implementation files modified are `.github/workflows/ci.yml` and `tests/corpus/test_audio_review.py`.
- Keep the check names `quality`, `Tests (windows-latest)`, and `Tests (ubuntu-latest)`.
- Windows remains the full-suite release-confidence gate with `--cov-fail-under=95`.
- Ubuntu runs `tests/unit`, `tests/integration`, `tests/corpus/test_mms_alignment.py`, and `tests/corpus/test_alignment_smoke_cli.py` without a repository-wide coverage threshold.
- Keep immutable action SHAs, `permissions: contents: read`, full history in `quality`, and the event-aware diff range.
- Do not change production cache validation, locking, no-overwrite behavior, TTS, G2P, stress, normalization, or MMS alignment code.
- PR #2 must remain Draft and Open.

---

## File Map

- `tests/corpus/test_audio_review.py`: owns the deterministic concurrency regression test; no production behavior is added here.
- `.github/workflows/ci.yml`: owns platform selection and the commands run by the three existing CI checks.

### Task 1: Stabilize the exclusive-cache-lock concurrency test

**Files:**
- Modify: `tests/corpus/test_audio_review.py:637`
- Test: `tests/corpus/test_audio_review.py::test_concurrent_same_key_has_one_producer_and_no_overwrite`

**Interfaces:**
- Consumes: `latintts.corpus.audio.os.open`, the real `_exclusive_key_lock()` implementation, and pytest's `monkeypatch` fixture.
- Produces: a test-only `lock_contended: threading.Event` signal proving that the second worker received `FileExistsError` while the first worker still held the destination lock.

- [ ] **Step 1: Record the current scheduler-sensitive behavior**

Run the existing test repeatedly before editing:

```powershell
1..20 | ForEach-Object {
    & 'D:\LatinTTS\.venv-corpus\Scripts\python.exe' -m pytest `
        'tests/corpus/test_audio_review.py::test_concurrent_same_key_has_one_producer_and_no_overwrite' -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

Expected: the test may pass on a fast local scheduler; the already-recorded CI failure is `CACHE_ARTIFACT_INVALID: cache artifact lacks metadata`, which occurs when the second worker starts after publication and lock removal.

- [ ] **Step 2: Add a test-only observation of the real lock collision**

Change the test signature and setup to preserve the real `os.open` call while signaling only after a lock-file collision:

```python
def test_concurrent_same_key_has_one_producer_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, record, config, _ = _fixture(tmp_path)
    producer_entered = threading.Event()
    release_producer = threading.Event()
    lock_contended = threading.Event()
    real_open = audio_module.os.open
    calls = 0
    results: list[DerivedAudio] = []
    errors: list[BaseException] = []

    def observing_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        try:
            return real_open(path, flags, mode, dir_fd=dir_fd)
        except FileExistsError:
            if Path(path).suffix == ".lock":
                lock_contended.set()
            raise

    monkeypatch.setattr(audio_module.os, "open", observing_open)
```

Keep the existing `blocking_runner()` and worker logic. After `second.start()`, replace the immediate producer release with:

```python
    assert lock_contended.wait(timeout=2)
    release_producer.set()
```

This ordering requires the second thread to attempt the real exclusive lock while the first producer is blocked. Do not patch `_exclusive_key_lock`, `_materialize`, cache validation, or publication.

- [ ] **Step 3: Run the targeted test repeatedly**

```powershell
1..20 | ForEach-Object {
    & 'D:\LatinTTS\.venv-corpus\Scripts\python.exe' -m pytest `
        'tests/corpus/test_audio_review.py::test_concurrent_same_key_has_one_producer_and_no_overwrite' -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

Expected: all 20 invocations pass; every invocation still asserts one producer call, two equal results, no worker errors, and no remaining `.lock` file.

- [ ] **Step 4: Run the complete audio review module**

```powershell
& 'D:\LatinTTS\.venv-corpus\Scripts\python.exe' -m pytest tests/corpus/test_audio_review.py -q
```

Expected: PASS with no failures.

- [ ] **Step 5: Review and commit the isolated test change**

```powershell
git diff --check
git diff -- tests/corpus/test_audio_review.py
git add tests/corpus/test_audio_review.py
git commit -m "test: stabilize concurrent audio cache coverage"
```

Expected: only `tests/corpus/test_audio_review.py` is committed.

### Task 2: Split CI responsibilities by platform

**Files:**
- Modify: `.github/workflows/ci.yml:17`
- Modify: `.github/workflows/ci.yml:44`
- Test: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the existing `quality` job, the `tests` matrix, `runner.os`, existing checkout/setup action pins, and the current diff-range environment variables.
- Produces: the unchanged check names `quality`, `Tests (windows-latest)`, and `Tests (ubuntu-latest)` with platform-specific test commands.

- [ ] **Step 1: Move quality checks to Windows**

In `.github/workflows/ci.yml`, change only the `quality` runner:

```yaml
  quality:
    runs-on: windows-latest
```

Keep all quality commands, `fetch-depth: 0`, action SHAs, permissions, and the explicit Bash shell for `git diff --check` unchanged.

- [ ] **Step 2: Make the full coverage step Windows-only**

Replace the current unconditional coverage step with:

```yaml
      - name: Run full coverage suite
        if: runner.os == 'Windows'
        run: python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q
```

- [ ] **Step 3: Add the Ubuntu portable core-functionality step**

Immediately after the Windows step, add:

```yaml
      - name: Run portable core suite
        if: runner.os == 'Linux'
        run: >-
          python -m pytest
          tests/unit
          tests/integration
          tests/corpus/test_mms_alignment.py
          tests/corpus/test_alignment_smoke_cli.py
          -q
```

Do not remove either OS from the matrix and do not add a coverage threshold to the Linux command.

- [ ] **Step 4: Inspect the workflow diff and whitespace**

```powershell
git diff --check
git diff -- .github/workflows/ci.yml
```

Expected: the diff contains one runner change, one Windows condition, and one Ubuntu test step; action pins, permissions, matrix names, and diff-range handling remain unchanged.

- [ ] **Step 5: Commit the isolated workflow change**

```powershell
git add .github/workflows/ci.yml
git commit -m "ci: prioritize portable functional checks"
```

Expected: only `.github/workflows/ci.yml` is committed.

### Task 3: Verify functionality and the PR gate

**Files:**
- Verify: `tests/corpus/test_audio_review.py`
- Verify: `.github/workflows/ci.yml`
- Verify unchanged: `src/`

**Interfaces:**
- Consumes: the Task 1 concurrency signal and the Task 2 platform-specific workflow commands.
- Produces: local evidence for core functionality and remote evidence from all three PR checks.

- [ ] **Step 1: Run the exact portable core suite**

```powershell
& 'D:\LatinTTS\.venv-corpus\Scripts\python.exe' -m pytest `
    tests/unit `
    tests/integration `
    tests/corpus/test_mms_alignment.py `
    tests/corpus/test_alignment_smoke_cli.py `
    -q
```

Expected: PASS with no failures.

- [ ] **Step 2: Run all local quality gates**

```powershell
& 'D:\LatinTTS\.venv-corpus\Scripts\python.exe' -m ruff format --check .
& 'D:\LatinTTS\.venv-corpus\Scripts\python.exe' -m ruff check .
& 'D:\LatinTTS\.venv-corpus\Scripts\python.exe' -m mypy src
& 'D:\LatinTTS\.venv-corpus\Scripts\python.exe' -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
& 'D:\LatinTTS\.venv-corpus\Scripts\python.exe' -m pip check
git diff --check
```

Expected: every command exits zero and the pronunciation audit reports no failures.

- [ ] **Step 3: Confirm that production code is unchanged**

```powershell
git diff e6323760d21e60d58ad50a9b0b3168fa05ee49c9...HEAD -- src
git status --short
```

Expected: the `src` diff is empty and the worktree has no uncommitted files.

- [ ] **Step 4: Run the corpus environment doctor from the cache-owning checkout**

From `D:\LatinTTS`:

```powershell
$ErrorActionPreference = 'Stop'
Get-Command ffmpeg -ErrorAction Stop
Get-Command ffprobe -ErrorAction Stop
& '.\.venv-corpus\Scripts\python.exe' -m latintts.corpus doctor
if ($LASTEXITCODE -ne 0) { throw 'latintts.corpus doctor failed' }
```

Expected: ffmpeg and ffprobe are found and doctor exits zero.

- [ ] **Step 5: Run the real cached MMS smoke test**

From `D:\LatinTTS`:

```powershell
& '.\.venv-corpus\Scripts\python.exe' -m latintts.corpus align --smoke-test
```

Expected: the smoke test uses the existing local MMS cache, completes an alignment, and exits zero without downloading.

- [ ] **Step 6: Push the implementation branch**

From `D:\LatinTTS\.worktrees\github-actions-ci`:

```powershell
git push origin codex/ci-workflow-implementation
```

Expected: remote branch `codex/ci-workflow-implementation` advances to the local `HEAD`.

- [ ] **Step 7: Verify PR state and wait for all checks**

```powershell
gh pr view 2 --repo SkyRainCho/LatinTTS --json state,isDraft,headRefName,url
gh pr checks 2 --repo SkyRainCho/LatinTTS --watch
```

Expected: PR #2 is `OPEN`, `isDraft` is `true`, its head is `codex/ci-workflow-implementation`, and `quality`, `Tests (windows-latest)`, and `Tests (ubuntu-latest)` all pass.

- [ ] **Step 8: Report deferred failures separately**

If a remote check still fails, capture the exact failing job and test:

```powershell
gh pr checks 2 --repo SkyRainCho/LatinTTS
gh run view <run-id> --repo SkyRainCho/LatinTTS --log-failed
```

Expected: any remaining issue is classified as either a regression in the two changed files or one of the explicitly deferred Linux hardening items; do not broaden this implementation into production code changes.
