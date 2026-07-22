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
- 本轮整改前该快照的 coverage raw value 为 `95.03%`；后续新增生产分支仍必须同步增加行为测试。
- final cache 使用 staging 后逐个无覆盖 hard-link 发布；跨文件中断可能留下未被 manifest 引用的完整内容寻址 cache，但不会发布 manifest/events/state。重跑会逐哈希验证并收敛，不覆盖或删除旧/真实数据。

## 2026-07-20 第二轮审查整改

### 修复结果

- 在任何 review import 写入前严格解码完整 processing journal，并先分类 audit-ahead。若 terminal events 已 durable 而 recordings 仍为 `REVIEWED`，只执行 `validate_review_bundle()` 纯读复验；decision 或 transcript IPA 漂移均保持 review、manifest、events、recordings 与 clips 字节不变，恢复原输入后可无重复 event ID 收敛。
- `ALIGNED -> REVIEWED` 不再错误绑定当前完整 review journal。其第三个 input digest 必须等于当前 `review.jsonl` 的一个完整 LF 行字节前缀；该前缀自身必须包含当时完整、合法、可重放的人审决定。前缀之后的合法边界/decision 修订允许参与最终 manifest，非法或孤儿 suffix 在提取前拒绝且零写入。
- 新增共享 canonical descendant walker，逐组件 `lstat` 并拒绝 symlink、Windows junction/reparse point、非 canonical resolve 与根外逃逸。staging 父链、staged destination、candidate/lossless mode root、final destination 均在创建前后验证，并在 hard-link 前再次复检。
- `persist_recording_transitions()` 现在对 `existing + missing` prospective journal 做一次完整严格验证，再进行任何 replace；跨 batch 重复 `event_id`、重复语义与逐 recording state-chain gap 均零写入拒绝。

### 第二轮 RED -> GREEN 证据

1. audit-ahead decision 漂移：旧实现先改写 `review.jsonl`，零写断言 RED；修复后 decision 与 IPA 两类漂移及恢复路径同测 GREEN。
2. legal correction suffix：旧实现要求 review transition digest 等于当前完整 journal，合法 suffix 构建失败；改为 exact-byte prefix 后 GREEN，并断言最终行消费实体完整事件链及修订边界。
3. output alias：使用无需管理员权限的 Windows junction 构造 `.manifest-staging` 父组件与 `lossless` mode-root alias；旧实现两例均到达 FFmpeg runner，修复后均在 runner 前拒绝。另有运行中注入 mode-root junction 的发布竞态测试，确认 hard-link、manifest、events、state 均未发布。
4. prospective batch：重复 event ID 与 state-chain gap 两测在旧实现均 `DID NOT RAISE` 且会写入；共享完整 journal preflight 后均 GREEN。
5. coverage 新分支首次全量为 `94.97%`，补齐 lexical escape 与 required-file 行为测试后达到 raw `95.01773870121856%`。

### 第二轮最终门禁

- 受影响范围 `tests/corpus/test_manifest.py tests/corpus/test_manifest_cli.py tests/corpus/test_audio.py tests/corpus/test_audio_review.py tests/corpus/test_store.py tests/corpus/test_paths.py`：`203 passed, 2 skipped`。
- `python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q`：
  - `1412 passed, 9 skipped`
  - displayed coverage：`95.02%`
  - raw coverage：`95.01773870121856%`（`6483` statements，`323` missing）
- `python -m ruff check src tests`：PASS。
- `python -m ruff format --check src tests`：PASS（72 files already formatted）。
- `python -m mypy src`：PASS（31 source files）。
- `python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl`：PASS（351/351）。
- `git diff --check`：PASS。

### 第二轮残余边界

- 全量中的 9 个既有 symlink 用例仍因当前 Windows token 缺少 symlink privilege 而条件 skip；本轮新增的 Windows junction/reparse 攻击测试实际执行且通过，不依赖该 privilege。
- coverage raw value 为 `95.01773870121856%`，严格高于 `95%` 门禁；新增生产分支仍应同步增加行为覆盖。
- Task 12 仍不启动 Task 13，不写入真实 corpus，不执行 push/merge/PR。

## 2026-07-20 第三轮审查整改

### Mixed legacy recovery

- terminal recovery 改为逐 recording 证明，不再把全局 `audit_ahead` 误解为“所有 terminal events 必须已经存在”。
- 已有 terminal recording 必须精确匹配其当前 deterministic 7-input event 或受支持的 legacy 5-input event；legacy inputs 仍严格绑定 raw、pairing、alignment、review 和已持久化 manifest digest，且 recording state 与重建目标一致。
- 仍为 `REVIEWED` 且没有 terminal event 的 recording 使用当前 7-input 格式生成缺失 event，包含 rights/transcript digests；`persist_recording_transitions()` 对 existing legacy prefix 加 missing current suffix 做 prospective 全 journal 验证后，一次 replace events、一次 replace recordings。
- all-rejected 与 rec1 approved/rec2 rejected 两种 mixed fixture 均验证 legacy event 原 ID/顺序不变、只追加 rec2 current event、event IDs 唯一、rerun bytes 幂等。
- legacy input、已持久化 manifest row、terminal state 任一冲突均在写入前 fail closed；review、manifest、events、recordings 和 clips 保持逐字节不变。

### Handle-relative final publication

- Windows 不使用 `SetFileInformationByHandle(FileLinkInfo)`（本机实测返回 `WinError 87`），也没有退回裸 `os.link(full_path)`。实现使用标准库 `ctypes` 调用 `ntdll!NtSetInformationFile(FileLinkInformation)`，以已打开 canonical destination directory handle 作为 `RootDirectory`，只提交 final basename。
- `CreateFileW`、`GetFileInformationByHandle`、`CloseHandle`、`NtSetInformationFile`、`RtlNtStatusToDosError` 均显式声明 64-bit-safe `argtypes/restype`；unsupported/capability/syscall failure 一律 fail closed。
- mode-root handle 使用 `FILE_ADD_FILE | FILE_READ_ATTRIBUTES`、`FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT`，share mode 不含 `FILE_SHARE_DELETE`。PublicationGuard 持有 candidate/lossless roots，直到 manifest、events、recordings 全部发布并完成最终 identity 复验后才关闭。
- 每次新 link 前后都验证 lexical root 与 pinned volume/file-index identity；新 target 还要求 source/target file identity 相同、canonical regular file 与 SHA-256 相同。已有 target 只有 canonical 且 hash 精确匹配才幂等接受，否则拒绝。
- POSIX 分支使用 `O_DIRECTORY | O_NOFOLLOW` 打开根目录，并用 `os.link(..., dst_dir_fd=...)` 做 handle-relative 发布；Windows 全量门禁将该平台专属分支排除 coverage 统计。

### 第三轮 RED -> GREEN 证据

1. mixed all-rejected 与 approved/rejected 旧实现均失败于 `durable terminal event batch is incomplete or conflicts`；逐 recording recovery 后两例 GREEN。
2. 最后检查/link seam 首先因接口缺失 RED；实现 native handle link 后，正常发布 GREEN，且在最后检查后尝试 `rmdir + junction` 时 sharing violation 阻止替换、outside 为空、manifest/events/state 未发布。
3. terminal commit seam 在 `persist_recording_transitions()` 入口尝试 rename `lossless` root；PublicationGuard 仍持有 handle，rename 被 sharing violation 阻止，随后正常完成 state commit。
4. Windows guard 另覆盖 missing root、junction/reparse root、lexical handle identity drift、source/target identity mismatch、unsupported mode 与 guard-close failure。
5. 新 native 分支首次全量为 `94.52%` 且一个既有并发缓存测试在 coverage 压力下偶发超时；独立/expanded focused 均通过。明确排除不可在 Windows 执行的 POSIX 分支并补齐上述 Windows 防御行为后，最终全量无失败并达到 raw `95.05204404887616%`。

### 第三轮最终门禁

- expanded focused `tests/corpus/test_manifest.py tests/corpus/test_manifest_cli.py tests/corpus/test_audio.py tests/corpus/test_audio_review.py tests/corpus/test_store.py tests/corpus/test_paths.py`：`212 passed, 2 skipped`。
- `python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q`：
  - `1424 passed, 9 skipped`
  - displayed coverage：`95.05%`
  - raw coverage：`95.05204404887616%`（`6629` statements，`328` missing）
- `python -m ruff check src tests`：PASS。
- `python -m ruff format --check src tests`：PASS（72 files already formatted）。
- `python -m mypy src`：PASS（31 source files）。
- `python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl`：PASS（351/351）。
- `git diff --check`：PASS。

### 第三轮残余边界

- Windows native implementation 依赖 NTFS-compatible `FileLinkInformation`；不支持该能力的卷或运行时会明确 fail closed，不按可替换完整路径回退。
- 全量 9 个既有 skip 仍来自当前 Windows token 缺少 symlink privilege；本轮 junction/reparse、handle identity 与 sharing-violation 用例均实际执行。
- Task 12 仍不启动 Task 13，不写入真实 corpus，不执行 push/merge/PR。

## 2026-07-21 第四轮审查整改

### Final audio 生命周期

- 新建 hard-link 与复用 final artifact 都会取得独立 `_ArtifactGuard`，并持有到 `segments.jsonl`、terminal events 与 `recordings.jsonl` 全部提交完成且最终复验通过。
- Windows final handle 使用 `GENERIC_READ` 与仅 `FILE_SHARE_READ`，明确拒绝 write/delete sharing；文件 identity 与 SHA-256 均从持有 handle 复验。新 hard-link 仍使用既有 `NtSetInformationFile(FileLinkInformation=11)`、显式 64-bit ABI 与 pinned `RootDirectory`，没有回退到裸 `os.link(full_path)`。
- POSIX final artifact 使用 mode-root `dir_fd` 相对 `O_NOFOLLOW` 打开，保留 fd，并在每个提交点比较 pinned/public `(st_dev, st_ino)` 及 fd SHA-256。
- new 与 existing final 在 terminal seam 的 write/rename 均由 Windows sharing violation 阻止；预先持有 writer 时在任何 durable manifest/state 写入前 fail closed；terminal exception 后 final、manifests 与 processing directories 均可正常 rename，证明 handle 无泄漏。

### Durable output namespace

- 在首次写 `segments.jsonl` 前，从 resolved project root 到 manifests、processing-events parent、segments/mode-root parent 的每个现存目录组件都取得 identity-bound guard，并贯穿 segments → events → recordings 全部 durable writes。
- Windows directory handles 允许 read/write sharing、拒绝 delete sharing；实测 manifests leaf、processing run parent 与 `derived/corpus-v1` ancestor 在 persist 入口或 events replace 后均无法 rename，当前 namespace 保持 segments/event/state 完整一致。
- 该轮曾增加 build 私有 `.build-manifest.lock`；第六轮审查确认它未覆盖其他内置 writer 后，已由共享 `.corpus-mutation.lock` 完整替代。Windows 仍使用无共享 `CreateFileW(OPEN_ALWAYS)` handle，POSIX 仍使用 manifests `dir_fd` 相对 `O_NOFOLLOW` 打开并 `flock(LOCK_EX|LOCK_NB)`。
- POSIX `read_jsonl()` / `write_jsonl_atomic()` 新增内部 `directory_fd` 路径：temp create、fsync、`os.replace(src_dir_fd=..., dst_dir_fd=...)` 与 parent fsync 全部绑定同一 pinned directory inode；processing batch 在每次 read/replace 前后调用 namespace identity validator。manifest、rights 与 transcripts digest 也从 pinned manifests directory 读取。

### 第四轮 RED -> GREEN 证据

1. new/existing final terminal attack 两参数在旧实现均失败：write 或 rename 成功，断言 `final artifact was not pinned`；严格 file guard 后 `2 passed`。
2. manifests 在 persist 入口整体 rename/recreate、processing parent 在 terminal event replace 后 rename，两例在旧实现均 `build=True` 且产生 split namespace；完整目录链 guard 后两例均以 sharing violation 阻止漂移并正常完成。
3. ancestor-chain、single-writer、pre-existing writer fail-closed 与异常无 handle leak 四项专项均通过。
4. 首次全量功能为 `1432 passed, 9 skipped`，但新增安全分支使 coverage 降至 `94.78%`；补齐 handle identity/hash 漂移、invalid-open cleanup、post-lock acquisition cleanup、duplicate artifact cleanup 后，最终为 `1436 passed, 9 skipped`、raw `95.0577569820149%`。

### 第四轮最终门禁

- expanded focused `tests/corpus/test_manifest.py tests/corpus/test_manifest_cli.py tests/corpus/test_audio.py tests/corpus/test_audio_review.py tests/corpus/test_store.py tests/corpus/test_paths.py`：`229 passed, 2 skipped`。
- mixed legacy/checkpoint + new/existing final + manifests/processing namespace 定向组合：`11 passed`。
- `python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q`：
  - `1436 passed, 9 skipped`
  - displayed coverage：`95.06%`
  - raw coverage：`95.0577569820149%`（`6839` statements，`338` missing）
- `python -m ruff check src tests`：PASS。
- `python -m ruff format --check src tests`：PASS（72 files already formatted）。
- `python -m mypy src`：PASS（31 source files）。
- `python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl`：PASS（351/351）。
- `git diff --check`：PASS。

### 第四轮残余边界

- Windows 强制依赖 deny-sharing handle；不支持 `NtSetInformationFile(FileLinkInformation)` 的文件系统继续 fail closed。
- POSIX fd 可固定 inode、相对 replace 可固定写入目录，但标准 POSIX 不提供 Windows 式强制 deny-delete sharing。会修改 manifest build 所消费可变 durable evidence 的内置写路径遵守共享 `flock` 协议；内容寻址音频/缓存继续使用独立 key lock。手工或恶意进程若忽略协议，其 namespace ABA 不属于保证范围，也不宣称所有此类漂移都能被检测或回滚。
- 9 个 skip 仍全部来自当前 Windows token 缺少 symlink privilege；本轮 Windows junction、sharing-violation、identity/hash 与 lock tests 均实际执行。
- Task 12 仍不启动 Task 13，不写入真实 corpus，不执行 push/merge/PR。

## 2026-07-21 第五轮审查整改

### 锁生命周期与权威输入 snapshot

- build writer lease 与 output namespace guard 在第一次读取 `pilot-selection.json`、`recordings.jsonl`、`processing-events.jsonl` 以及任何 review import/validation 之前取得，并持有到完整 build 返回；第六轮起该 lease 使用所有内置 durable writer 共享的 `.corpus-mutation.lock`。第二 writer 在任何 loader、review import、clip extraction 或 `segments.jsonl` 写入前即失败。
- review import 作为可写 Phase A 在锁内完成；随后丢弃 Phase A 解码对象，重新加载 selection、recordings、rights、transcripts、review 与 processing 的权威 Phase B snapshot。
- Windows `_FileSnapshot` 使用 `GENERIC_READ`、仅 `FILE_SHARE_READ` 与 `FILE_FLAG_OPEN_REPARSE_POINT`，从持有 handle 取得 identity 与 SHA-256，拒绝后续 write/delete sharing；POSIX 使用 `O_NOFOLLOW` file descriptor、public inode identity 与 handle digest，并在所有 durable checkpoint 前后复验。
- raw recording、analysis audio、segmentation、pairing、alignment 与可选 `pairing-automatic.json` 同样被纳入 snapshot。event 的 rights、transcripts、pairing、alignment、review digest 与实际构行消费的同一份锁定 snapshot 完全一致。
- Windows rights 撤权、selection/recordings/transcripts/review/processing/segmentation/pairing/alignment 漂移都被强制 sharing violation 阻止；POSIX 上相同漂移在首次 `segments.jsonl` durable write 前由 identity/hash 复验拒绝。

### Durable file handoff

- fresh `segments.jsonl` atomic replace 后立即以 canonical expected JSONL digest 打开严格 file snapshot；recovery 先 pin 已有 `segments.jsonl` 再解析。两者均持有到 terminal events 与 recordings 完成且最终复验通过。
- `persist_recording_transitions()` 新增 exact-existing-digest 与 before/after write handoff：旧 processing snapshot 在最后复验后才关闭，events replace 后立即校验 expected bytes 并 pin；新 events guard 持有穿过 recordings replace。recordings 使用相同交接，写后立即 pin。
- events/recordings atomic replace 后、after-pin 前注入的字节漂移均被 expected digest 拒绝；recordings 写后失败继续使用既有 audit-first `RecordingTransitionPersistenceError` 恢复契约，不返回成功。
- cleanup 现在关闭全部 snapshot、artifact、publication、namespace handles，记录第一个 cleanup failure；已有业务异常优先，不再被后续 close failure 覆盖。release 失败的 snapshot 保留在 registry，最终清理会再次尝试。

### 第五轮 RED -> GREEN 证据

1. rights 在旧 `_open_output_namespace_guard` seam 被撤权：旧实现使用旧 rights 构行却把新 rights digest 写入 event，测试 `DID NOT RAISE`；锁前移后以 `RIGHTS_SCOPE_UNCONFIRMED` 拒绝且 segments/events/state 零写入。
2. 持有第一个 writer lock 后启动第二 build：旧实现先触发 `_load_selection`；修复后第二 build 在首次 loader 前以 sharing/lock failure 退出。
3. selection、recordings、rights、transcripts、review、processing、segmentation、pairing、alignment 九类 publish 前漂移在旧实现全部可继续；Phase B file snapshots 后九类均在 `segments.jsonl` 前 fail closed。
4. `segments.jsonl` digest 后、terminal persist 入口替换：旧实现仍返回 APPROVED；严格 segments snapshot 后 write/replace 被阻止或在 terminal state 前拒绝。
5. events/recordings 写后漂移、events guard 跨 recordings replace、event digest 精确绑定、cleanup 业务异常优先与 store handoff 顺序专项共 `17 passed`。
6. expanded focused 首跑发现 tampered-raw 错误类型从 `INVENTORY_HASH_MISMATCH` 退化为 review ValueError；恢复锁内 raw preflight 后既有契约与 Phase B snapshot 复验同时保留，复跑全绿。

### 第五轮最终门禁

- expanded focused `tests/corpus/test_manifest.py tests/corpus/test_manifest_cli.py tests/corpus/test_store.py tests/corpus/test_audio.py tests/corpus/test_audio_review.py tests/corpus/test_review.py tests/corpus/test_review_cli.py tests/corpus/test_paths.py`：`390 passed, 5 skipped`。
- `python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q`：
  - `1457 passed, 9 skipped`
  - displayed coverage：`95.00%`
  - raw coverage：`95.00142005112184%`（`7042` statements，`352` missing）
- `python -m ruff check src tests`：PASS。
- `python -m ruff format --check src tests`：PASS（72 files already formatted）。
- `python -m mypy src`：PASS（31 source files）。
- `python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl`：PASS（351/351）。
- `git diff --check`：PASS。

### 第五轮残余边界

- Windows 依靠 deny-sharing handles 提供强制 content/delete exclusion；snapshot 到 store replace 的有意 handoff 窗口仍受无共享 build lock、expected existing digest、write 后 expected bytes pin 三层约束。
- POSIX descriptor 能固定 inode，`dir_fd` 能固定 atomic replace parent，`flock` 只约束合作 writer。共享 lease 保证所有会改变 build 所消费可变 durable evidence 的内置写路径互斥；内容寻址音频/缓存由独立 key lock 管理。非合作 writer 无法被标准 POSIX 强制禁止，其手工/恶意 ABA 明确不在保证范围，报告不宣称 Windows 式强制排他或必然检测每次外部漂移。
- 全量 9 个 skip 仍全部来自当前 Windows token 缺少 symlink privilege；Windows sharing、snapshot identity/hash、post-write pin 与 lock tests 均实际执行。
- Task 12 仍不启动 Task 13，不写入真实 corpus，不执行 push/merge/PR。

## 2026-07-22 第六轮审查整改：共享 corpus mutation lease

### 根因与共享锁协议

- 第五轮 build 私有 `.build-manifest.lock` 只约束 `build-manifest`，`write_jsonl_atomic()`、review helper、inventory intake 与其他真实写路径并不取得同一把锁；POSIX 上 pinned fd/namespace 复验不能替代合作 writer 的跨事务互斥。
- 新增 `latintts.corpus.locking`，锁文件固定为 `local-data/manifests/.corpus-mutation.lock`，永久保留而不 unlink，避免合作进程锁住不同 inode。目标路径从最近的精确 `local-data` 祖先定位同一 corpus；混合 non-local/local 或不同 local-data root 的多路径事务 fail closed。
- Windows 同时持有 manifests directory deny-delete handle 与 lock file `CreateFileW(OPEN_ALWAYS, share_mode=0)` handle，拒绝 reparse/directory lock object；POSIX 从 `O_DIRECTORY | O_NOFOLLOW` pinned manifests dirfd 相对打开 regular lock file，比较 fd/public `(st_dev, st_ino)` 后执行 `flock(LOCK_EX | LOCK_NB)`。
- 进程内每 corpus 使用 nonblocking `RLock`、depth 与显式 `(owner_pid, owner_thread)`：同 PID 同线程嵌套可重入，其他线程立即返回 `CorpusMutationLockBusy`，其他进程由 Windows sharing 或 POSIX flock 拒绝。最后一层 close 才释放 OS lease；acquire/close 异常不会遗留 registry depth。
- fork child 在触碰可能由消失线程持有的旧 registry mutex 之前重置 registry；继承 backend 只 close child fd，绝不 `LOCK_UN` 父进程 flock。继承的 stale `CorpusMutationLease.close()` 在 PID 不同场景只标记自身 closed，不修改旧 Entry/RLock/backend；同 PID 跨线程 close 仍在任何状态变化前拒绝。

### 内置写路径接入

- `_OutputNamespaceGuard` 删除私有 `.build-manifest.lock`，改持共享 lease，从第一次 input/review 读取前一直贯穿 `segments.jsonl → processing-events.jsonl → recordings.jsonl`、最终 snapshot/artifact 复验与 build return。
- `store.write_jsonl_atomic()` 对 local-data 目标自动取得共享 lease；`append_jsonl_event()` 把 read+append+replace 放在一个外层 lease；`persist_recording_transition(s)()` 把 audit-first events→recordings 整个事务放在一个外层 lease。build 与高层事务内的 nested store write 通过同线程重入，不自阻塞。
- review 的 text/JSON/bytes/TextGrid/copy 原子 helper 全部接入共享 lease，`export_review_bundle()` 与 `import_review_bundle()` 还持有覆盖完整多文件流程的 outer lease。inventory `intake.csv` 独占创建与 CLI runtime text 原子写同样接入；inventory/transcript/pairing 的 durable JSON 写入由 store 自动覆盖。
- 协议范围精确限定为“会改变 manifest build 所消费可变 durable evidence 的内置写路径”。内容寻址 audio/cache 继续使用已有独立 key lock 与 build artifact/snapshot guards，不声称共享 corpus lease 覆盖所有派生缓存写入。

### 第六轮 RED → GREEN 证据

1. 首个共享锁测试在 collection 阶段按预期失败于 `ModuleNotFoundError: latintts.corpus.locking`；最小实现后同线程嵌套 store write、跨线程 fail-fast、review helper、persist 双文件 outer lease、nonregular lock 与 alias 拒绝转绿。
2. build-style outer lease 下，第二线程分别写 rights、transcripts、review、recordings、pairing 五类路径均返回 `CorpusMutationLockBusy`，原字节不变：`5 passed`。
3. 真实子进程争用共享 lock 返回明确 busy 且不改变 corpus bytes；release 后可重新写。backend acquire/close failure、nested refcount、junction manifests alias 与 lock directory 非 regular 专项均通过。
4. inventory direct `intake.csv` writer 与 CLI direct text helper 的 RED 均表现为第二线程可成功写入；接入共享 lease 后均 GREEN。
5. 旧测试用同一线程打开第二个 output guard，和新“同线程可重入”契约冲突；改用第二线程后，第二 build 在任何 `_load_selection` 前失败，guard release 后可重新取得。
6. 跨线程 close RED 精确暴露旧行为 `cannot release un-acquired lock`，并会先腐化 depth/backend；owner 校验前移后，错误线程零状态变化、原 owner 可正常 close、随后第三线程可取得。
7. PID fallback RED 证明旧 `_entry_for()` 丢 registry 时未调用 `close_after_fork()`；统一 child reset 后 mock inherited backend 已关闭。另有 POSIX-only真实 `fork()` 回归：child stale close 不释放 parent flock，child 新 acquire 仍 busy；当前 Windows 将该项标为平台 skip。

### 第六轮最终门禁

- locking 精确覆盖：`20 passed, 1 skipped`；`locking.py` 为 `186 statements / 8 missing = 95.70%`。
- expanded focused（locking、manifest、store、review、inventory、transcript、pairing、alignment smoke 共 11 个文件）：`530 passed, 6 skipped`；owner/fork 修复后相关专项再跑 `72 passed, 2 skipped`。
- `python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q`：
  - `1477 passed, 10 skipped`
  - displayed coverage：`95.01%`
  - raw coverage：`95.0096498483595%`（`7254` statements，`362` missing）
- `python -m ruff check src tests`：PASS。
- `python -m ruff format --check src tests`：PASS（74 files already formatted）。
- `python -m mypy src`：PASS（32 source files）。
- `python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl`：PASS（351/351）。
- `git diff --check`：PASS。

### 第六轮威胁边界

- Windows sharing handles 为合作与非合作 writer 提供更强的 open/write/delete exclusion；不支持所需 handle/API 的文件系统继续 fail closed。
- POSIX 保证所有会改变 build 所消费可变 durable evidence 的 LatinTTS 内置写路径合作互斥；`flock` 无法强制手工或恶意外部进程合作。外部 rename/replace namespace ABA 明确不在保证范围，不承诺每次都能检测、回滚或阻止。
- 10 个 skip 中 9 个仍来自当前 Windows token 缺少 symlink privilege；新增 1 个是只在 POSIX 执行的真实 fork/flock 回归。Windows 上跨线程、跨进程、junction、sharing 与模拟 PID fallback 均实际执行。
- Task 12 仍不启动 Task 13，不写入真实 corpus，不执行 push/merge/PR。
