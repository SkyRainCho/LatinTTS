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

## 2026-07-20 审查整改

### 整改提交

- `8256374`：新增严格 schema、append-only、一次 atomic replace 的批量 terminal event/state 持久化协议。
- `cf0ac63`：新增全局音频 staging、terminal 全量重建、三检查点恢复、review/text 与完整 extractor request provenance 校验。
- `a505c1c`：移除已被 deterministic rebuild 取代的旧 terminal fast-path validator。
- `dd59ec1`：补齐批量转换拒绝契约和精确 coverage 门禁。

### 新增 RED 证据

以下测试均先在旧实现或缺失接口上按预期失败，再实施最小修复：

1. 批量状态提交：两个测试失败于 `AttributeError: ... persist_recording_transitions`，证明旧 store 只有逐 recording commit。
2. terminal rights：撤销 `allow_model_training` 后 `DID NOT RAISE CorpusFailure`。
3. terminal manifest 精确重建：修改 `spoken_text` 时旧 fast path 错误放行。
4. terminal review snapshot：修改 `automatic.json` 边界后 `DID NOT RAISE ValueError`。
5. 人工 word text：追加 `word:0:text` 的 `Pater -> Mater` 事件后 `DID NOT RAISE ValueError`。
6. 全局 staging：第二个 take 提取失败后断言发现 final `lossless-*.flac` 残留。
7. manifest checkpoint 恢复：首次 manifest replace 故障后重跑失败于 `CACHE_ARTIFACT_INVALID: cache artifact lacks metadata`。
8. 文本一致性：首次构建行中 `spoken_text == "Pater ..."`，但 `word_spans[0].text == "pater"`，精确一致性断言失败。
9. coverage 第一次完整门禁为 `94.24%`，第二次为 `94.83%`；删除不可达旧 validator 并覆盖 batch 拒绝分支后达到精确 `95.00%`。

### Commit / recovery 协议

确定性顺序固定为 pilot selection recording ID、SpokenUnit ordinal、take index：

1. 纯读加载并严格复验 selection、recordings、rights、transcripts/pronunciation、raw hash、segmentation、pairing、alignment、review bundle/event replay 和 processing history。
2. 重建全部 approved proposals；rejected take 不生成行，unreviewed 阻止推进。
3. 所有 lossless 与 quality candidate 先写 deterministic transaction staging；后续 take、后续 recording 或 quality 任一点失败时不新增 final clip。
4. 全部 `DerivedAudio` 与 `SegmentRecord` 验证成功后才发布 canonical final cache；staging 路径从不进入 provenance。
5. 一次 atomic replace 发布 `segments.jsonl`，随后以其实际 SHA-256 构造所有 terminal `ProcessingEvent`。新事件 inputs 依次绑定 raw、rights.jsonl、transcripts/pronunciation、pairing、alignment、review、manifest digest。
6. 所有缺失 terminal events 按 selection 顺序合并后一次 atomic replace；严格拒绝 ID、语义、schema、state chain 或 input 冲突。
7. 最后只对完整 `recordings.jsonl` snapshot 做一次 atomic replace，不再逐 recording materialize。

可恢复窗口：

- manifest 已发布、events 未发布：重跑精确重建同一 manifest，再批量发布 events/state。
- events 全部 durable、recordings 未物化：接受 audit-ahead，验证全部 evidence 后只替换一次 recordings snapshot。
- mixed terminal/REVIEWED：接受新批量窗口及旧逐条提交窗口；已有 legacy event 必须与旧 deterministic inputs 完全一致，新建 event 必须包含 rights/transcript digests。
- manifest、events、recordings 三个 replace 检查点的故障注入重跑均收敛；event ID 数量保持唯一。
- terminal rerun 不信任现有 manifest：从 immutable/upstream inputs 重建 rows，逐行、逐序比较，并复验 lossless/quality hash 与 PCM metrics。

### 新增聚焦验证

- `tests/corpus/test_manifest.py tests/corpus/test_manifest_cli.py tests/corpus/test_store.py tests/corpus/test_review.py tests/corpus/test_review_cli.py`：`255 passed, 4 skipped`。
- 关键恢复/terminal/text/provenance 选择集：`10 passed`。
- staging 故障注入覆盖同 recording 后续 take、后续 recording、quality extraction：`3 passed`。
- batch commit checkpoints 覆盖 events replace、recordings replace、audit-ahead 与 mixed state：`2 passed`。
- 完整请求 provenance 覆盖 lossless/quality 的结构有效但 interval 错误结果：`2 passed`。

### 整改后完整门禁

- `python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q`
  - `1400 passed, 9 skipped`
  - raw coverage：`95.03%`（`6403` statements，`318` missing）
- `python -m ruff check .`：PASS。
- `python -m ruff format --check .`：PASS（72 files already formatted）。
- `python -m mypy src`：PASS（31 source files）。
- `python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl`：PASS（351/351）。
- `git diff --check`：PASS。

### 残余风险

- Windows 当前主机仍无 symlink/reparse 创建权限，9 个既有 alias 专项测试按条件 skip；可运行的 canonical/reparse/alias 链检查均通过。
- coverage raw value 精确为 `95.00%`，后续增加生产分支必须同步增加行为测试。
- final cache 使用 staging 后逐个无覆盖 hard-link 发布；跨文件中断可能留下未被 manifest 引用的完整内容寻址 cache，但不会发布 manifest/events/state。重跑会逐哈希验证并收敛，不覆盖或删除旧/真实数据。
