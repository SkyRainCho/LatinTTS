# GitHub Actions Dual-Platform CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a least-privilege GitHub Actions workflow that enforces LatinTTS quality gates and ≥95% full-suite coverage on Windows and Ubuntu while preserving historical plan documents byte-for-byte.

**Architecture:** Add one formatter-only exclusion in `pyproject.toml`, then add one `CI` workflow with an Ubuntu quality job and a Windows/Ubuntu test matrix. Static checks run once, full coverage runs once per platform, and GitHub concurrency cancels superseded runs from the same branch.

**Tech Stack:** Python 3.10, pytest 8, pytest-cov 6, Ruff 0.16-compatible configuration, mypy 1.x strict mode, GitHub Actions, `actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2`, `actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405 # v6.2.0`, PowerShell, GitHub CLI

## Global Constraints

- Python remains `>=3.10,<3.11`; CI uses Python `3.10`.
- CI uses `windows-latest` and `ubuntu-latest`.
- Every platform independently runs the full suite with `--cov-fail-under=95`.
- Do not run the real MMS GPU smoke test or install `requirements/corpus.txt` in CI.
- Do not upload `local-data/`, model caches, coverage artifacts, or releases.
- Workflow permissions remain exactly `contents: read`.
- Pin all action references to immutable SHAs and retain their release-version comments exactly.
- The `quality` checkout uses `fetch-depth: 0`; its whitespace check uses the event-aware PR/push range.
- Use worktree-local `.venv` for local commands. Install `PyYAML==6.0.3` only transiently before local YAML parsing; do not add it to `pyproject.toml` or CI dependencies.
- Preserve every file under `docs/superpowers/plans/` byte-for-byte.
- Exclude `docs/superpowers/plans/**` from Ruff formatting only; retain Ruff lint coverage.
- Keep PR #2 as a Draft; do not modify its review state.

---

## File Structure

- Modify: `pyproject.toml`
  - Owns the formatter-only exclusion for historical plan documents.
- Create: `.github/workflows/ci.yml`
  - Owns GitHub event triggers, permissions, concurrency, quality checks, and the dual-platform test matrix.
- Reference only: `docs/superpowers/specs/2026-07-25-github-actions-ci-design.md`
  - Provides the approved decisions and acceptance criteria.

### Task 1: Exclude Historical Plans from Ruff Formatting

**Files:**
- Modify: `pyproject.toml:35-43`
- Test: existing repository-wide Ruff format and lint commands

**Interfaces:**
- Consumes: Ruff configuration rooted at `[tool.ruff]`.
- Produces: `[tool.ruff.format].exclude: list[str]` containing only `docs/superpowers/plans/**`.

- [ ] **Step 1: Capture the historical plan baseline**

Run:

```powershell
git status -sb
git diff -- docs/superpowers/plans
```

Expected: the branch is clean except for the already committed design/plan history, and the second command prints no diff.

- [ ] **Step 2: Run the formatter gate to verify the current failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
```

Expected: exit 1 and exactly these historical files are reported as requiring formatting:

```text
docs/superpowers/plans/2026-07-17-latintts-pronunciation-foundation.md
docs/superpowers/plans/2026-07-19-latintts-corpus-alignment-pilot.md
docs/superpowers/plans/2026-07-19-task-11-review-remediation-round-5.md
```

- [ ] **Step 3: Add the formatter-only exclusion**

Add immediately after the existing `[tool.ruff]` table and before `[tool.ruff.lint]`:

```toml
[tool.ruff.format]
exclude = ["docs/superpowers/plans/**"]
```

The resulting Ruff section must be:

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.format]
exclude = ["docs/superpowers/plans/**"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
allowed-confusables = ["ˈ"]
```

- [ ] **Step 4: Run the format and lint gates**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
```

Expected:

```text
All checks passed!
```

The formatter must report no unformatted files. Its discovered-file count is not an assertion because
the new directory exclusion changes that count; both commands must exit 0.

- [ ] **Step 5: Prove historical plans remain byte-for-byte unchanged**

Run:

```powershell
git diff --exit-code -- docs/superpowers/plans
git diff --check
```

Expected: both commands exit 0 and print no historical-plan diff.

- [ ] **Step 6: Commit the Ruff boundary**

Run:

```powershell
git add -- pyproject.toml
git commit -m "chore: exclude historical plans from Ruff format"
```

Expected: one commit containing only `pyproject.toml`.

### Task 2: Add the Dual-Platform CI Workflow

**Files:**
- Create: `.github/workflows/ci.yml`
- Test: inline YAML syntax and workflow-contract assertions

**Interfaces:**
- Consumes: `pyproject.toml`, the `dev` extra, `tests/fixtures/gold_pronunciations.jsonl`, and Git refs; the local YAML assertion additionally uses transient `PyYAML==6.0.3` from worktree `.venv` only.
- Produces: GitHub checks named `quality`, `Tests (windows-latest)`, and `Tests (ubuntu-latest)`; `quality` has full history and checks the event-aware `BASE_SHA...HEAD_SHA` range.

- [ ] **Step 1: Run a contract assertion to verify the workflow is absent**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; path = Path('.github/workflows/ci.yml'); assert path.is_file(), f'{path} is missing'"
```

Expected: exit 1 with `AssertionError: .github\workflows\ci.yml is missing`.

- [ ] **Step 2: Create the workflow with the approved contract**

Create `.github/workflows/ci.yml` with exactly:

```yaml
name: CI

on:
  pull_request:
  push:
    branches:
      - main

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.head_ref || github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          fetch-depth: 0
      - name: Set up Python
        uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405 # v6.2.0
        with:
          python-version: "3.10"
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Upgrade pip
        run: python -m pip install --upgrade pip
      - name: Install project
        run: python -m pip install -e ".[dev]"
      - name: Check formatting
        run: python -m ruff format --check .
      - name: Lint
        run: python -m ruff check .
      - name: Type check
        run: python -m mypy src
      - name: Audit gold pronunciations
        run: python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
      - name: Check whitespace
        shell: bash
        env:
          BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.before }}
          HEAD_SHA: ${{ github.event.pull_request.head.sha || github.sha }}
        run: git diff --check "$BASE_SHA...$HEAD_SHA"

  tests:
    name: Tests (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os:
          - windows-latest
          - ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
      - name: Set up Python
        uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405 # v6.2.0
        with:
          python-version: "3.10"
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Upgrade pip
        run: python -m pip install --upgrade pip
      - name: Install project
        run: python -m pip install -e ".[dev]"
      - name: Run full coverage suite
        run: python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q
```

- [ ] **Step 3: Parse the YAML and assert the security/test contract**

Run (the installation is local and transient; do not add PyYAML to `pyproject.toml` or CI dependencies):

```powershell
.\.venv\Scripts\python.exe -m pip install PyYAML==6.0.3
.\.venv\Scripts\python.exe -c "from pathlib import Path; import yaml; data = yaml.load(Path('.github/workflows/ci.yml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader); assert data['name'] == 'CI'; assert data['permissions'] == {'contents': 'read'}; assert data['concurrency']['cancel-in-progress'] == 'true'; assert data['jobs']['quality']['runs-on'] == 'ubuntu-latest'; assert data['jobs']['quality']['steps'][0]['with']['fetch-depth'] == '0'; assert data['jobs']['tests']['strategy']['fail-fast'] == 'false'; assert data['jobs']['tests']['strategy']['matrix']['os'] == ['windows-latest', 'ubuntu-latest']; assert data['jobs']['quality']['steps'][-1]['env']['BASE_SHA'] == '${{ github.event.pull_request.base.sha || github.event.before }}'; assert data['jobs']['quality']['steps'][-1]['env']['HEAD_SHA'] == '${{ github.event.pull_request.head.sha || github.sha }}'; steps = data['jobs']['tests']['steps']; assert steps[-1]['run'] == 'python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q'; print('workflow-contract: PASS')"
```

Expected:

```text
Successfully installed PyYAML-6.0.3
workflow-contract: PASS
```

- [ ] **Step 4: Assert the workflow does not add forbidden behavior**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; text = Path('.github/workflows/ci.yml').read_text(encoding='utf-8'); forbidden = ('requirements/corpus.txt', 'smoke-test', 'local-data/', 'upload-artifact', 'release'); found = [item for item in forbidden if item in text]; assert not found, found; print('workflow-scope: PASS')"
```

Expected:

```text
workflow-scope: PASS
```

- [ ] **Step 5: Run local quality gates**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
git diff --check
```

Expected: every command exits 0; the audit reports `gold-audit: PASS total=351 errors=0`.

- [ ] **Step 6: Commit the workflow**

Run:

```powershell
git add -- .github/workflows/ci.yml
git commit -m "ci: add dual-platform validation workflow"
```

Expected: one commit containing only `.github/workflows/ci.yml`.

### Task 3: Run the Full Gate and Verify GitHub Actions

**Files:**
- No source file changes expected.
- Verify: `pyproject.toml`, `.github/workflows/ci.yml`, PR #2 checks.

**Interfaces:**
- Consumes: committed Task 1 and Task 2 changes on `codex/fix-mms-model-revision`.
- Produces: a pushed branch whose PR has three successful GitHub Actions checks.

- [ ] **Step 1: Run the complete local coverage gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q
```

Expected: `1850 passed`, only platform-specific skips, total coverage at least `95.00%`, and exit 0. Test counts may increase only if implementation adds tests.

- [ ] **Step 2: Re-run all non-test gates**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
.\.venv\Scripts\python.exe -m pip check
git diff --check
git status -sb
```

Expected: every gate exits 0; status shows a clean `codex/fix-mms-model-revision` branch ahead of its upstream only by the new commits.

- [ ] **Step 3: Push the implementation commits**

Run:

```powershell
git push origin codex/fix-mms-model-revision
```

Expected: the remote branch advances to the local `HEAD`.

- [ ] **Step 4: Confirm GitHub recognizes the workflow**

Run:

```powershell
gh workflow view CI --repo SkyRainCho/LatinTTS --yaml
```

Expected: exit 0 and output containing `quality`, `windows-latest`, and `ubuntu-latest`.

- [ ] **Step 5: Locate and watch the PR workflow run**

Run:

```powershell
$headSha = git rev-parse HEAD
$run = gh run list --repo SkyRainCho/LatinTTS --workflow CI --branch codex/fix-mms-model-revision --event pull_request --limit 10 --json databaseId,headSha,status,conclusion,url |
  ConvertFrom-Json |
  Where-Object { $_.headSha -eq $headSha } |
  Select-Object -First 1
if ($null -eq $run) { throw "No CI run found for $headSha" }
gh run watch $run.databaseId --repo SkyRainCho/LatinTTS --exit-status
```

Expected: `gh run watch` exits 0 after all three jobs finish.

- [ ] **Step 6: Verify the three required checks and remote identity**

Run:

```powershell
$localHead = git rev-parse HEAD
$remoteHead = git rev-parse origin/codex/fix-mms-model-revision
$checks = gh pr checks 2 --repo SkyRainCho/LatinTTS --json name,state,bucket,workflow | ConvertFrom-Json
$required = @("quality", "Tests (windows-latest)", "Tests (ubuntu-latest)")
if ($localHead -ne $remoteHead) { throw "Local and remote heads differ" }
foreach ($name in $required) {
  $check = $checks | Where-Object { $_.name -eq $name }
  if ($null -eq $check -or $check.bucket -ne "pass") {
    throw "Required check did not pass: $name"
  }
}
git status -sb
```

Expected: no exception; local and remote `HEAD` match; all three named checks have `bucket: pass`; the worktree is clean.

- [ ] **Step 7: Preserve Draft state and report completion**

Run:

```powershell
gh pr view 2 --repo SkyRainCho/LatinTTS --json isDraft,state,url
```

Expected: `isDraft` is `true`, `state` is `OPEN`, and no command changes the PR review state.
