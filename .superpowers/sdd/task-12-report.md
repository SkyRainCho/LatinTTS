# Task 12 实现报告：Approved Clip Extraction and Master Segment Manifest

## 结果

- 实现提交：`e77693d352fd82ba5c4c181d886efa2510345e00`（`feat: build approved corpus manifest`）
- 实现 `SegmentRecord`、`require_approval()`、`build_approved_segments()`、`build_manifest_corpus()` 与 `build-manifest` CLI。
- 最终片段只从 immutable raw 调用 `extract_lossless_segment()`；审核后精确区间的质量指标通过 `extract_analysis_segment()` 在 16 kHz analysis 音频上重新验证/计算。
- `segments.jsonl` 在全部 pilot recordings、全部审批行和全部派生结果成功后一次 atomic replace；派生片段不删除。
- 全部 take 已决定后，至少一条 approved 执行 `REVIEWED -> APPROVED`，全部 rejected 执行 `REVIEWED -> REJECTED`；processing event 先于 recordings replace。
- 未加入真实语料、真实片段或本地 manifest。

## TDD 证据

### RED

1. 首个审批门测试：`python -m pytest tests/corpus/test_manifest.py -q`
   - 预期失败：`ModuleNotFoundError: No module named 'latintts.corpus.manifest'`。
2. `SegmentRecord` 测试：预期失败于 `ImportError: cannot import name 'SegmentRecord'`。
3. `build-manifest` CLI 分派测试：预期失败于 argparse `invalid choice: 'build-manifest'`。
4. 核心 reviewed fixture：预期失败于 `build_manifest_corpus()` 缺少提取依赖/尚未实现。
5. 后续逐项确认过 RED 的契约包括：editable `automatic.json` 漂移、human pairing correction replay、跨 recording 全局预检、terminal history tampering、duplicate provenance identities、pronunciation warning、raw tampering before extraction、extractor provenance drift。

### GREEN / REFACTOR

- 聚焦门禁：`python -m pytest tests/corpus/test_manifest.py tests/corpus/test_manifest_cli.py -q`
  - `70 passed`（最终全量执行也覆盖这些测试）。
- 将 manifest 构建收紧为两阶段：先对所有 pilot recordings 重放并验证 pairing/alignment/review/transcript/rights/boundary rows，再开始任何 final clip 提取。
- terminal rerun 会重新验证 raw、segments、review event IDs、processing chain、pairing/alignment/review/manifest hashes 和最终 state，不信任可编辑缓存摘要。

## 已落实的关键契约

- 只有 human `review_decision == "approved"` 且对应 SpokenUnit pronunciation token slice 无 warning 才能进入 manifest。
- 复用 Task 11 的 strict pairing/alignment/review loaders、deterministic event IDs、snapshot replay、canonical path 与 processing transition bindings。
- take-local review seconds 先加 candidate analysis 起点，再以原始 sample rate 统一使用 Python half-even `round()` 得到半开区间 `[source_start_sample, source_end_sample)`；批准区间必须正长度且同录音不重叠。
- raw inventory hash 在全部 final clip 写入前全量预检，并由 audio materialization 在读写前后再次校验。
- pronunciation plan 严格要求 `schema_version=1`、`corpus-v1`、`ecclesiastical-roman-v1`；IPA/model phonemes 只来自 SpokenUnit token slice。
- `phoneme_timing_status` 固定为 `not_estimated`，alignment 固定为 word-level。
- `review_event_ids` 按 review journal 原顺序记录该 take 实际消费的 pairing-correction（如有）与 take review 完整事件链。
- rights、recording、selected source candidate、transcript、pairing、alignment、analysis、review、lossless/quality derived artifacts 都做 identity/hash/schema 绑定。

## 最终门禁

- `python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q`
  - `1378 passed, 9 skipped`
  - raw coverage：`95.00%`（`6281` statements，`314` missing）
  - 9 个 skip 均为当前 Windows 主机无 symlink/reparse 创建权限的既有条件跳过。
- `python -m ruff check .`：PASS
- `python -m ruff format --check .`：PASS（72 files already formatted）
- `python -m mypy src`：PASS（31 source files）
- `python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl`：PASS（351/351）
- `git diff --check`：PASS

## 风险与后续边界

- 覆盖率原始值恰好为 `95.00%`，已满足仓库门禁，但后续增加未覆盖生产分支时必须同步增加测试。
- Windows 环境无法创建部分 symlink/reparse 测试夹具；canonical/alias 防护由现有可运行测试和无权限 skip 的专用测试共同覆盖。
- Task 12 不执行 Task 13 报告/成本估算，不创建真实 corpus 数据。
