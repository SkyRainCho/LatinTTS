# Task 11 Report: Human Review Bundles and Append-Only Corrections

## Status

Complete against baseline commit `33afda0` (abbreviated Git object ID).

Commit subject: `feat: add corpus human review workflow`

## Implemented

- Added deterministic Praat long-text `TextGrid` output with `take` and `words` interval tiers,
  explicit empty gap intervals, quote escaping, and a strict reader that rejects schema drift,
  invalid/overlapping/out-of-range intervals, duplicate interval indexes, and tier changes.
- Added review bundle export beneath each recording's canonical alignment run. Every repetition
  group receives two clips decoded directly from the original source audio without loudness
  processing,
  two TextGrids, `automatic.json`, initialized per-take `decision.json`, and an HTML listening page
  with escaped human-controlled text.
- Bound bundle summaries back to the strict Task 10 pairing/alignment artifacts, original and
  candidate audio hashes, source frame ranges, three transcript text layers, alignment versions,
  candidate/quality scores, warnings, and exact entity identities. Fixed filenames and resolved
  containment prevent bundle and candidate references from escaping their roots.
- Added strict decision import for `unreviewed|approved|rejected`, exact JSON fields, unique entities,
  non-empty human reason/reviewer/time, timezone-aware timestamps, unchanged audio hashes, and
  controlled TextGrid boundary/text edits. No automatic decision or approval path exists.
- Added deterministic append-only `ReviewEvent` identities and strict history replay. Every event
  carries entity, field, before/after, reason, reviewer, and timestamp. Corrections append from the
  current effective value; history is never rewritten. Exact resubmission is idempotent and
  same-effective-value submissions with conflicting metadata are rejected.
- Persisted `review.jsonl` before processing-state materialization. Partial review leaves a
  recording `ALIGNED`; only when every exported take has a durable approved/rejected human decision
  does the audit-first `ALIGNED -> REVIEWED` transition atomically replace `recordings.jsonl`.
  Review-required pairing outcomes advance only after a human selects exact preserved evidence;
  incomplete pairing corrections remain `SEGMENTED`.
- Added `export-review` and `import-review` CLI commands using the normal corpus config. Re-export
  preserves an existing exact bundle so edited TextGrids and decisions cannot be overwritten.

## TDD and verification

- Followed RED -> GREEN from the required missing-module failure through TextGrid round-trip and
  rejection cases, replay, export, import, CLI, audit ordering, correction, idempotency, conflict,
  containment/hash, nested schema, and transcript-layer regressions.
- Focused review suites: **138 passed, 3 skipped**. The three focused skips are the
  review root/group/file alias tests that require unavailable Windows symlink privilege.
- Full suite with literal `precision = 2` / `fail_under = 95.00`: **1,305 passed,
  9 skipped**, total coverage **95.01%** (`5,734` statements, `286` missed);
  `review.py` coverage is **95.21%**.
  All skips are existing Windows tests requiring unavailable file/directory symlink privilege.
- Ruff check: pass. Ruff format check: 69 files already formatted. Strict `mypy src`: pass.
- Gold audit: `gold-audit: PASS total=351 errors=0`.
- `git diff --check`: pass (Git only reports the repository's existing LF-to-CRLF checkout warning
  for modified tracked Python files).

## Self-review and residual risk

- Tests use generated PCM16 WAVs and synthetic pairing/alignment artifacts. No real corpus audio,
  decisions, model files, or runtime artifacts were added to Git.
- Review extraction now delegates WAV, FLAC, MP3, M4A, OGG, and Opus decoding to the pinned
  FFmpeg identity recorded in `DerivedAudio`; generated review clips are validated PCM24 WAV.
- This task stops at `REVIEWED`; later tasks decide whether reviewed takes become final `APPROVED`
  or `REJECTED` corpus records. Mixed per-take decisions are retained as human events and cannot
  produce an automatic `APPROVED` recording state.

## Independent review remediation

All Critical and Important follow-up findings were remediated in focused commits. Commit IDs
below are abbreviated Git object IDs:

- `ac8ee4e`: canonical component-walking review paths and bounded TextGrid parsing.
- `c390e22`: trusted PCM24 review derivation through `extract_review_wav()`, complete
  `DerivedAudio` provenance, and FLAC/MP3 source coverage.
- `2bb49a6`: strict event-field, type, finite-value, entity, and post-event invariants.
- `6ab89e2`: exact pairing/alignment artifact and per-take provenance binding plus
  collision-free structured entity identities.
- `300b19c`: human-only materialization of saved review-required pairing candidates,
  v2 correction decisions, append-before-pairing persistence, and correction-bound align.
- `a1e1167`: independent per-recording completion and state advancement.
- `340e2d3`: correction-schema and human-metadata regression coverage required by the
  repository-wide coverage gate.
- `04a0ed6`: crash recovery across review-journal, automatic-pairing, corrected-pairing,
  processing-event, and recording-manifest checkpoints.
- `d82c52b`: strict repeated-pair validation, deterministic correction-event and immutable-prefix
  binding, ordinary-import legal entity validation, and actual-WAV-frame review duration.
- `78a416d`: two-decimal coverage enforcement plus structural and media-boundary regression tests.
- `885000c`: exact processing-event-bound review prefix decoding, prefix-local deterministic
  correction validation, legal suffix validation, and canonical automatic-pairing reads.
- `2b4ac1c`: read-only correction-recovery preflight before journal or pairing writes, including
  prospective in-memory journal validation for missing processing transitions.
- `66c4c51`: whitespace-preserving recovery for a durable correction journal whose processing
  transition is still missing; the new transition binds and validates the exact durable raw bytes,
  while planned new journal bytes must be durable before transition persistence.

The final remediation suite collected 1,314 tests: **1,305 passed, 9 skipped**, with
**95.01% total coverage** (`5,734` statements, `286` missed). Ruff check and format check,
strict `mypy src`, the gold audit (`gold-audit: PASS total=351 errors=0`), and
`git diff --check` all pass. The nine skips are Windows-host tests requiring unavailable
file or directory symlink privilege; three of those are the new review root/group/file
alias cases.

Residual risk is limited to operational FFmpeg behavior with real-world containers and
Windows reparse-point variants that cannot be created by the unprivileged test runner.
The production code rejects every existing reparse component via `lstat` and validates
all review media against immutable raw-source and artifact provenance before state change.
