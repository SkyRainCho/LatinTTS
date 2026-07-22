# LatinTTS 阶段 2：口语语料对齐试点设计

- 状态：已批准
- 设计批准日期：2026-07-19
- 规格批准日期：2026-07-19
- 适用项目版本：`v0.1.0` 之后
- 上游规则版本：`ecclesiastical-roman-v1`
- 试点范围：2–3 个现代罗马教会式拉丁语口语录音文件

## 1. 文档目的

本规格定义 LatinTTS 阶段 2 的第一个实施周期：从用户已有的整篇祷文或经文录音中，恢复准确文本，识别每个文本单元的两遍朗读，生成经人工批准的短音频、词级时间边界、发音计划和版本化语料清单。

本周期的目标不是立刻处理全部录音，也不是训练 TTS 模型，而是用 2–3 个有代表性的文件验证一条可重复、可审计、可估算成本的端到端流程。

本规格细化并在阶段 2 范围内覆盖 `2026-07-17-latintts-system-design.md`。尤其是：旧设计中“音素级强制对齐”的表述在本试点中修正为“人工批准的词级对齐，加上阶段 1 生成的正确音素序列”；本试点不声称得到精确音素时间戳。

## 2. 已确认的语料事实

- 一个音频文件对应一篇完整祷文，或一段已知书卷、章节和节数的经文。
- 当前没有逐句转写，但文件信息能够精确定位作品或经文范围。
- 每一句或每个由几个单词构成的小段通常连续朗读两遍。
- 两遍都是有效朗读，不表示“第一遍错误、第二遍纠正”。
- 两遍之间通常有较短停顿，不同文本单元之间通常有较长停顿。
- 不同录音的篇幅、停顿和文本差异较大，不能使用一个全局固定停顿阈值。
- 现有录音同时包含普通朗读和完整旋律的歌唱/圣咏；本试点只处理普通朗读。
- 用户拥有录音或已经取得朗读者用于模型训练的明确授权。
- 用户会把数据复制到项目的本地数据目录，并保留项目之外的原始备份。
- 本机有 NVIDIA GeForce RTX 4080，可用于强制对齐推理；实际显存和吞吐量由冒烟测试记录。

## 3. 目标与非目标

### 3.1 目标

1. 盘点本地口语与歌唱录音，验证原始文件完整性和授权信息。
2. 从口语录音中选择短、中、长或结构复杂的 2–3 个代表性文件并锁定试点集合。
3. 为每个试点录音登记有来源的原文、实际朗读文本和供阶段 1 使用的规范文本。
4. 使用语音活动检测和文件内停顿分类提出文本单元及重复朗读边界。
5. 将正常文本单元配对为两遍独立、可追溯的朗读片段。
6. 生成词级强制对齐候选、低置信度队列和可供人工修订的审核包。
7. 产出全部经过人工确认的片段、JSONL 主清单、质量报告和全量处理工时估算。
8. 验证 RTX 4080 上的对齐速度、显存占用和 Windows 兼容性。

### 3.2 非目标

- 不处理歌唱、额我略圣咏、旋律、音高或乐谱。
- 不处理全部语料。
- 不训练、微调或评测 TTS 模型。
- 不接入 MMS-TTS Latin 推理基线；本规格中的 MMS 仅指强制对齐声学模型。
- 不实现网页或云服务。
- 不把自动 ASR 输出当作规范文本。
- 不把真实录音、真实片段、本地 manifest 或模型缓存提交到 Git。
- 不锁定完整语料的 `train/dev/test` 划分；试点片段保持 `split: "unassigned"`。
- 不声称字符或音素时间戳具有人工语音学标注级精度。

## 4. 总体方法

采用“结构优先的半自动对齐”作为主线：

```mermaid
flowchart LR
    Raw["只读原始录音"] --> Inventory["清点、哈希、元数据"]
    Inventory --> Text["候选原文与实际朗读文本"]
    Text --> VAD["VAD 与文件内停顿分类"]
    VAD --> Pair["文本单元与两遍朗读配对"]
    Pair --> Align["MMS CTC 词级强制对齐"]
    Align --> Review["人工试听、修边界、确认文本"]
    Review --> Manifest["批准片段与版本化 JSONL"]
    Manifest --> Report["试点质量与全量成本报告"]
```

保留两个辅助策略：

- ASR 只用于发现漏词、插词、错读或文本版本异常，不能直接写入 `spoken_text`。
- Praat 等人工工具只处理低置信度、自动失败或边界需要调整的局部片段；人工修订结果必须重新导入主数据流。

不采用 ASR 优先方案，因为低资源拉丁语识别容易把重复朗读、停顿和非标准拼写解释成错误文本；也不采用全手工切分方案，因为它不能可靠估算和扩展到全量语料。

## 5. 本地数据布局与 Git 边界

```text
local-data/
├── raw/
│   ├── spoken/                  # 本试点输入，只读
│   └── sung/                    # 只登记，不处理
├── derived/
│   └── corpus-v1/
│       ├── normalized/          # 16 kHz mono PCM 分析副本
│       ├── segments/            # 从原始音频重新解码的审核/训练候选片段
│       └── alignments/          # 后端原始结果、统一结果、TextGrid、运行缓存
└── manifests/
    ├── intake.csv               # 用户填写的入口表
    ├── rights.jsonl             # 录音和朗读者授权范围
    ├── recordings.jsonl         # 原始文件清单与状态
    ├── transcripts.jsonl        # 三层文本、候选来源和版本差异
    ├── segments.jsonl           # 文本单元、两遍朗读、对齐与质量信息
    └── review.jsonl             # 追加式人工审核事件
```

约束如下：

- `local-data/` 整体加入 `.gitignore`，不使用 Git LFS，也不上传 GitHub。
- 原始文件只复制、不移动、不覆盖；任何派生操作只写 `derived/`。
- manifest 中保存相对于 `local-data/` 的 POSIX 风格相对路径，不保存机器专属绝对路径。
- 原始音频、文本快照、派生音频和审核结果都用 SHA-256 标识。
- Git 只跟踪代码、配置、schema、文档、测试、可公开分发的合成小音频和不含真实内容的汇总统计。
- 模型权重与下载缓存保存在 Git 忽略目录，并固定模型 revision 或文件哈希。

### 5.1 Intake 表

用户入口表至少包含：

```csv
relative_path,title_or_citation,speaker_id,rights_id,notes
raw/spoken/file001.wav,Pater Noster,speaker-001,rights-001,
raw/spoken/file002.wav,Ioannes 1:1-14,speaker-001,rights-001,
```

缺少标题、祷文名称或精确经文范围的文件可以进入清单，但不能进入 `TEXT_CANDIDATES_READY`。

### 5.2 权利记录

`rights.jsonl` 至少记录：

- 录音所有者和朗读者标识。
- 是否允许本地整理、模型训练和内部评测。
- 是否允许发布原始文件、切分片段和训练后模型；未知项必须保持 `unknown`，不能由训练授权推导。
- 授权日期、授权依据和备注。

本试点以用户已经确认的本地处理与训练授权为依据。任何公开数据、片段或模型发布都需要单独通过对应授权字段。

## 6. 数据状态机

每个试点录音按以下顺序推进：

```text
DISCOVERED
  → INVENTORIED
  → TEXT_CANDIDATES_READY
  → TRANSCRIPT_CONFIRMED
  → SEGMENTED
  → PAIRED
  → ALIGNED
  → REVIEWED
  → APPROVED | REJECTED
```

规则：

- 不允许静默跳过中间状态。
- 每次转换记录输入哈希、配置哈希、工具版本、开始/结束时间和结果摘要。
- 失败不会删除上一步产物；实体停留在最近的成功状态，并附加问题代码。
- `REJECTED` 是有原因的审查结果，不等于删除。
- 修订上游文本或边界时，下游状态失效并重新计算；历史版本保留。
- 只有人工审核事件可以产生 `APPROVED`。

## 7. 录音清点与试点选择

### 7.1 清点字段

`recordings.jsonl` 至少包含：

```json
{
  "schema_version": "1",
  "recording_id": "rec-0001",
  "relative_path": "raw/spoken/file001.wav",
  "sha256": "...",
  "content_type": "spoken",
  "title_or_citation": "Pater Noster",
  "speaker_id": "speaker-001",
  "rights_id": "rights-001",
  "duration_seconds": 123.45,
  "sample_rate": 48000,
  "channels": 1,
  "codec": "pcm_s24le",
  "state": "INVENTORIED"
}
```

元数据由 `ffprobe` 获取；哈希由项目代码对原始字节计算。重复内容用哈希识别，但不自动删除文件。

### 7.2 选择方法

清点工具提出三个候选：

1. 较短且停顿结构正常的录音。
2. 时长接近语料中位数的录音。
3. 较长、停顿密集或结构较复杂的录音。

最终试点 ID 写入版本化配置并锁定，后续参数比较始终使用同一集合。歌唱录音不进入候选。

## 8. 文本恢复与三层文本

每个录音保留三种语义明确、互不覆盖的文本：

### 8.1 `source_text`

公开或授权来源中的权威原文。每个候选版本保存：

- 来源 ID、标题、版本和稳定链接。
- 获取日期和本地快照哈希。
- 书卷、章节、节数或祷文版本说明。
- 与其他候选的规范化差异。
- 采用或拒绝理由。

### 8.2 `spoken_text`

录音中实际读出的文字。它允许记录相对于 `source_text` 的：

- 省略。
- 重复。
- 插入。
- 词序变化。
- 朗读错误或非标准变体。

ASR 可以提出异常位置，但不能直接修改该字段。人工必须边听边确认。

### 8.3 `normalized_text`

交给阶段 1 `Pronouncer` 的规范文本。它对应一个完整的 `PronunciationPlan`，并记录：

- `schema_version`。
- `rule_version`。
- 音节、重音、IPA 和 `model_phonemes`。
- `needs_review` 警告及人工覆盖。

### 8.4 `alignment_text`

强制对齐后端可能需要小写、去标点或 uroman token。`alignment_text` 是从 `spoken_text` 派生的非权威技术表示，不构成第四个语义文本层，不得反向覆盖前三层文本，也不得作为发音规则依据。

## 9. 音频派生与质量分析

### 9.1 分析副本

- 使用 FFmpeg 从原始文件生成 `16 kHz / mono / PCM WAV`。
- 文件名和缓存键包含原始哈希、转码配置哈希及 FFmpeg 版本。
- 分析副本只用于 VAD 和对齐，不作为最终训练片段的音频来源。

### 9.2 批准片段

- 根据批准的原始时间范围，从原始文件重新解码为无损 WAV 或 FLAC。
- 精确记录源文件、源起止采样位置、解码配置和输出哈希。
- 试点阶段不做降噪、响度标准化、动态压缩、去混响或自动静音裁除。
- 音质处理如在后续模型阶段需要，必须生成新的派生版本，不能覆盖 `corpus-v1`。

### 9.3 质量字段

每个录音和片段记录：

- 时长、采样率、声道数和编码。
- 峰值、RMS、削波样本比例和静音比例。
- 人工标签：背景噪声、混响、呼吸、咳嗽、外部人声、音乐、截断等。

质量指标只路由审核；未经人工确认不能据此自动删除片段。

## 10. VAD 与文件内停顿分类

### 10.1 VAD 后端

首选 Silero VAD `v6.2.1`，通过 ONNX 或其稳定 Python 接口运行：

- 输入为 16 kHz mono 分析副本。
- 保存逐窗口概率、语音区间和全部参数。
- 默认使用 CPU，保证 4080 可专用于 CTC 对齐；需要时可以单独基准 GPU。
- 固定软件版本、模型文件哈希和推理参数。

Silero VAD 只判断语音活动，不判断拉丁文本或重复关系。

### 10.2 停顿分类

对每个录音独立处理内部静音间隔：

1. 从相邻 VAD 语音区间计算停顿时长。
2. 在对数时长上进行确定性的两类聚类，得到候选短停顿和长停顿。
3. 使用配置化的最小样本量、类间分离度和持续时间边界验证聚类。
4. 长停顿提出新的文本单元边界；短停顿只是两遍朗读分割候选。
5. 停顿太少、两类重叠或结果不稳定时返回 `PAUSE_CLASSES_AMBIGUOUS`。

默认参数属于 `pause-profile-v1`，必须进入配置哈希。试点报告同时保存各文件的停顿分布，以便决定全量阶段是否需要多种 profile。

## 11. 重复朗读配对

### 11.1 正常结构

一个正常文本单元产生一个重复组：

```text
repetition_group_id = rec-0001-unit-0004
├── take_index = 1
└── take_index = 2
```

两遍是独立训练候选，共享文本单元和 `repetition_group_id`，但各自拥有音频路径、质量字段、对齐结果和审核状态。

### 11.2 配对方法

- 长停顿之间形成候选文本组，配对不能跨越长停顿。
- 组内所有短停顿先作为候选分割点，而不是直接当作两遍边界。
- 对每个分割候选比较两侧时长、相同文本的 CTC 覆盖率、词序和边界置信度。
- 选择唯一最佳且通过约束的分割；组内属于自然呼吸的停顿保留在对应 take 中。
- 如果多个候选近似相同、找不到有效候选或实际朗读次数异常，进入人工审核。

正常配对约束：

- 恰好两个 take。
- 两个 take 对应相同 `normalized_text`。
- 对齐词序一致。
- 初始时长比位于 `0.65–1.35`；超出只触发审核，不自动拒绝。
- 不跨文本单元长停顿。

异常代码：

- `TAKE_COUNT_MISMATCH`
- `TAKE_DURATION_MISMATCH`
- `TAKE_TEXT_MISMATCH`

任何异常组都保留原始候选、评分和失败原因。

## 12. 词级强制对齐

### 12.1 统一接口

项目定义后端无关接口：

```text
align(audio, spoken_text, config) -> AlignmentResult
```

`AlignmentResult` 至少包含：

- 后端、代码版本、模型 ID、模型 revision 和许可证。
- 输入音频哈希、`spoken_text` 哈希和 `alignment_text`。
- 词级起止时间。
- 字符或 token 分数、文本覆盖率和整体置信度摘要。
- 低置信度区间、警告和失败代码。
- 后端原始输出路径。
- `alignment_level: "word"`。
- `phoneme_timing_status: "not_estimated"`。

### 12.2 首选后端

试点首选：

- `MahmoudAshraf97/ctc-forced-aligner`
- `MahmoudAshraf/mms-300m-1130-forced-aligner`
- ISO 639-3 语言代码 `lat`
- RTX 4080 CUDA 推理，优先 FP16；不支持时回退到受控的 FP32 配置

实施首先进行一个不含真实语料的安装/加载冒烟测试，再用一个试点片段验证：

- 模型能够加载并在可用显存内运行。
- `lat` 文本预处理不会丢失必要字符。
- 长音频分块不会改变词序。
- 输出可以稳定映射回 `spoken_text`。

如果原生 Windows 依赖不稳定，对齐后端可以在 WSL2 的独立锁定环境中运行；项目 CLI、manifest schema 和相对路径约定保持不变。

### 12.3 不采用的当前主线

- 不直接依赖已归档 Fairseq 仓库中的旧 MMS 数据准备脚本。
- 不依赖已经弃用或从新版移除的 TorchAudio `forced_align` API。
- 暂不以 Montreal Forced Aligner 为主后端，因为官方预训练目录没有拉丁语声学模型。积累人工批准语料后，可以单独设计说话人/发音体系适配的 MFA 模型。

### 12.4 音素边界的可信度

MMS CTC 对齐使用字符或罗马化 token，不直接消费 `PronunciationPlan.model_phonemes`。因此试点输出必须区分：

```json
{
  "alignment_level": "word",
  "phoneme_timing_status": "not_estimated",
  "words": [
    {"text": "gratia", "start": 1.24, "end": 1.91}
  ],
  "pronunciation_plan": {
    "rule_version": "ecclesiastical-roman-v1",
    "model_phonemes": ["ɡ", "r", "a", "t͡s", "i", "a"]
  }
}
```

词边界经过人工修订后可以批准；音素序列由阶段 1 保证；不得用平均分配或字符时间戳伪造音素时间戳。后续如果声学模型明确需要音素持续时间，再基于批准语料设计专用对齐模型和独立验收集。

### 12.5 许可证

首选 MMS 对齐模型标注为 CC BY-NC 4.0。试点按个人研究和本地非商业处理执行，并在每个运行结果中记录模型许可证。时间戳输出和派生 manifest 的公开或商业使用不能由本规格自动授权；商业化前应更换兼容后端或完成单独许可证评估。

## 13. 人工复核闭环

### 13.1 审核包

每个候选组生成：

- 两个 take 的无损 WAV。
- 包含文本单元、take 和词边界的 Praat `TextGrid`。
- 包含来源、文本、质量指标、自动评分和警告的 JSON 摘要。
- 自动结果与对应原始录音时间范围。

不要求开发网页。Praat 只作为边界查看和局部修改工具，JSONL 主清单仍是系统事实来源。

### 13.2 允许的人工动作

- 接受候选结果。
- 移动片段或词级边界。
- 确认或修正 `spoken_text`。
- 确认两遍配对。
- 拒绝单个 take 或整个重复组。
- 添加质量标签和备注。

### 13.3 审核历史

`review.jsonl` 是追加式事件流，不覆盖自动结果：

```json
{
  "schema_version": "1",
  "review_event_id": "review-000123",
  "entity_id": "rec-0001-unit-0004-take-1",
  "field": "segment_end",
  "before": 14.82,
  "after": 15.07,
  "reason": "final consonant truncated",
  "reviewer": "human",
  "reviewed_at": "2026-07-19T12:00:00+08:00"
}
```

试点中的每个最终批准 take 都必须能追溯到人工审核事件。不存在自动批准路径。

## 14. 主清单

`segments.jsonl` 是试点事实来源。每个 take 至少记录：

- schema、语料和处理配置版本。
- `segment_id`、`recording_id`、`text_unit_id`、`repetition_group_id` 和 `take_index`。
- 原始文件哈希、原始起止采样位置、派生音频路径与哈希。
- `source_text`、`spoken_text`、`normalized_text` 的记录 ID 和版本。
- `PronunciationPlan.schema_version`、`rule_version`、IPA 和 `model_phonemes`。
- 词级对齐、后端版本、模型许可证和置信度摘要。
- `phoneme_timing_status`。
- 音质指标、人工标签、问题代码和审核状态。
- 权利记录 ID。
- `split: "unassigned"`。

任何训练器专用 CSV、LJSpeech metadata 或其他格式都必须从主清单生成，不能成为独立事实来源。

## 15. 缓存、重跑与依赖隔离

### 15.1 内容寻址

每一步缓存键至少包含：

- 上游音频或文本 SHA-256。
- 当前配置哈希。
- 工具、代码和模型版本。
- schema 版本。

缓存命中前必须校验产物哈希和必要字段。上游变化只使相关下游失效。

### 15.2 原子写入

- 所有产物先写临时文件。
- 完成格式与哈希校验后原子改名。
- 中断或异常留下可诊断日志，但不能把半成品标为成功。
- 重跑从最近成功状态继续。
- `report.json` / `report.md` 使用持久三态事务：`ACTIVE` 只回滚，恢复并联合校验旧报告对后
  原子移交到 `ROLLED_BACK`；`ROLLED_BACK` 只继续清理旧事务；`COMMITTED` 只 roll-forward。
- root outcome marker 在登记的 root payload 和私有事务目录之后最后删除。若中断发生在
  私有目录删除后，只有当前报告对的完整身份与 intent 一致且登记 payload 均已消失时，
  才能用 root outcome marker 完成终态恢复。
- 删除前核验登记的 device、inode、mode、size 与 SHA-256。该协议依赖
  `corpus_mutation_lease` 协作锁；忽略协作锁的同用户路径 ABA 竞争不属于自动恢复保证。

### 15.3 Python 依赖

- 阶段 1 发音核心继续保持 `dependencies = []`。
- 音频、VAD、PyTorch、Transformers 和对齐工具进入独立且锁定版本的 corpus 环境。
- corpus 模块对可选依赖使用延迟导入，并在缺失时返回清晰错误。
- FFmpeg/ffprobe 作为显式外部工具被检测并记录版本。

## 16. CLI 工作流

预期入口：

```powershell
python -m latintts.corpus inventory
python -m latintts.corpus select-pilot
python -m latintts.corpus prepare-text
python -m latintts.corpus segment
python -m latintts.corpus pair
python -m latintts.corpus align
python -m latintts.corpus export-review
python -m latintts.corpus import-review
python -m latintts.corpus build-manifest
python -m latintts.corpus report
```

共同要求：

- 默认读取项目内固定配置和 `local-data/`。
- 支持显式 `--recording-id` 或锁定试点配置。
- 支持 `--dry-run`，列出即将读取和写入的相对路径。
- 重复执行具有幂等性；相同输入与配置不重新计算。
- 错误使用稳定代码和非零退出码。
- 默认日志不输出完整敏感文本、绝对用户目录或音频内容。
- 所有用户提供的相对路径解析后必须仍位于 `local-data/` 内，拒绝绝对路径和 `..` 越界。
- FFmpeg、ffprobe 和后端命令使用参数数组调用，不把文件名拼接成 shell 命令。

## 17. 错误代码

至少定义：

| 代码 | 含义 | 默认处理 |
| --- | --- | --- |
| `INVENTORY_UNSUPPORTED_FORMAT` | 音频格式无法探测或解码 | 保留记录，停止该文件 |
| `INVENTORY_HASH_MISMATCH` | 原始文件与已登记哈希不同 | 阻止所有下游步骤 |
| `RIGHTS_SCOPE_UNCONFIRMED` | 本地处理或训练授权未确认 | 不进入试点 |
| `TRANSCRIPT_SOURCE_NOT_FOUND` | 找不到可靠候选文本 | 保持清点状态 |
| `TRANSCRIPT_AMBIGUOUS` | 多个文本版本无法确定 | 人工选择或隔离 |
| `TRANSCRIPT_SPOKEN_MISMATCH` | 实际朗读与候选文本不同 | 人工修订 `spoken_text` |
| `PAUSE_CLASSES_AMBIGUOUS` | 无法可靠区分短/长停顿 | 人工标边界 |
| `TAKE_COUNT_MISMATCH` | 文本单元不是恰好两遍 | 人工确认异常结构 |
| `TAKE_DURATION_MISMATCH` | 两遍时长差异过大 | 进入高优先级审核 |
| `TAKE_TEXT_MISMATCH` | 两遍词序或文本不同 | 分别确认或拒绝 |
| `ALIGNER_UNAVAILABLE` | 对齐模型、依赖或 GPU 不可用 | 停止对齐，保留上游结果 |
| `ALIGNMENT_TEXT_UNSUPPORTED` | 后端无法表示输入字符 | 人工调整技术表示或更换后端 |
| `ALIGNMENT_LOW_CONFIDENCE` | 词级对齐可信度不足 | 人工修订 |
| `AUDIO_QUALITY_REJECTED` | 人工确认音质不适用 | 保留原因并拒绝片段 |
| `MANIFEST_SCHEMA_MISMATCH` | 产物 schema 不兼容 | 阻止消费 |
| `CACHE_ARTIFACT_INVALID` | 缓存缺失、截断或哈希不符 | 删除该缓存产物并重算 |
| `REPORT_OUTPUT_RECOVERY_REQUIRED` | report 双输出事务无法安全自动恢复 | 保留 intent、claim 和备份，人工核对后重跑 |

错误记录包含实体 ID、阶段、稳定代码、可读说明、工具版本和可恢复建议，不依赖解析自由文本判断流程。

## 18. 测试策略

### 18.1 仓库内测试数据

- 不提交真实录音或真实祷文片段。
- 使用程序生成的静音、正弦波、噪声和公开可分发的小型语音夹具。
- 真实 MMS 对齐模型测试标记为本地 GPU 冒烟测试，不进入普通 CI。

### 18.2 单元测试

覆盖：

- intake 和相对路径校验。
- 哈希、ffprobe 结果解析和重复文件识别。
- 合法与非法状态转换。
- 文本三层不变量和版本差异。
- VAD 区间到停顿区间的转换。
- 确定性两类停顿分类及歧义分支。
- 正常两遍、少读、多读、时长异常和文本不一致的配对。
- `AlignmentResult` 归一化和后端错误映射。
- 审核事件的追加、重放和下游失效。
- manifest schema、版本和相对路径。
- 缓存命中、输入变更、损坏缓存和中断恢复。

### 18.3 集成测试

- 使用合成 WAV 和假对齐器完成 `inventory → manifest` 全流程。
- 验证重复执行不产生不同 ID 或重复审核事件。
- 验证修改 `spoken_text` 只使必要下游失效。
- 验证从 TextGrid 导入边界后保留自动值和人工值。
- 验证 `local-data/` 确实被 Git 忽略。
- 继续运行阶段 1 的完整测试、静态检查和黄金发音审计。

### 18.4 本地真实后端验证

记录：

- CUDA、驱动、PyTorch、Transformers 和模型 revision。
- 模型加载是否成功。
- 峰值显存、实时系数和分块配置。
- 相同输入重复运行的时间戳稳定性。
- Windows 原生与必要时 WSL2 的结果一致性。

## 19. 人工验收标准

每个批准片段必须满足：

- 没有截断首尾音素。
- 没有混入相邻文本单元。
- `spoken_text` 与实际朗读完全一致。
- `take_index`、重复组和来源录音明确。
- 不含破坏性咳嗽、外部人声或其他不可接受事件；轻微呼吸和混响以标签记录。
- 片段和词边界已经人工接受或调整。
- `PronunciationPlan` 使用规定的 schema 和规则版本，且不存在未解决的发音警告。

时长只作为审核提示：

- `2–12 秒`：优先训练候选。
- `0.8–2 秒` 或 `12–20 秒`：保留并提示复核。
- `<0.8 秒` 或 `>20 秒`：必须人工决定，不自动删除。

两遍朗读都可以批准；不预设第一遍或第二遍更好。后续训练阶段可以选择两遍都用、只用一遍、一遍训练一遍评测，或两遍都拒绝。

## 20. 试点报告与扩展判断

报告至少包含：

- 原始录音总时长和 VAD 语音时长。
- 文本单元、朗读次数、重复组和批准片段数。
- 自动正确配对比例。
- 无需人工移动的片段边界比例。
- 批准、拒绝和待复核的语音时长。
- 各问题代码和人工拒绝原因数量。
- 文本准备耗时与每录音分钟的人工审核分钟数。
- 4080 对齐速度、峰值显存和失败重试次数。
- 按完整口语语料规模推算的工时、GPU 时间和存储空间。

判断区间：

| 指标 | 可扩展 | 需要优化 | 暂不适合批处理 |
| --- | ---: | ---: | ---: |
| 自动配对正确率 | `≥95%` | `85–95%` | `<85%` |
| 边界无需修改率 | `≥85%` | `70–85%` | `<70%` |
| 人工复核/录音时长 | `≤3×` | `3–6×` | `>6×` |
| 可批准语音占比 | `≥80%` | `60–80%` | `<60%` |

这些指标决定是否扩展、调参或换方案，不降低最终数据质量要求。最终批准片段始终是 100% 人工确认。

## 21. 交付里程碑

### 里程碑 1：清点

- 原始录音哈希与元数据。
- 权利字段完整性报告。
- 口语/歌唱分类。
- 2–3 个试点候选及选择理由。

### 里程碑 2：停顿与试听

- 分析副本。
- VAD 和停顿分布。
- 候选文本单元与可试听区间。

### 里程碑 3：重复配对

- 每个文本单元的 take 1/take 2。
- 配对评分和异常队列。

### 里程碑 4：词级对齐

- `AlignmentResult`。
- WAV、TextGrid 和 JSON 审核包。
- GPU 性能记录。

### 里程碑 5：批准语料与报告

- 人工审核历史。
- 批准片段与 `segments.jsonl`。
- 质量指标、失败分类和全量处理估算。
- 扩展、调参或换方案的建议。

## 22. 实施顺序与估算

| 工作 | 预计时间 |
| --- | ---: |
| 数据契约、清点与目录保护 | 0.5–1 天 |
| 文本恢复与三层文本登记 | 0.5–1.5 天 |
| VAD、停顿分类和配对 | 1–1.5 天 |
| MMS 对齐器和 RTX 4080 验证 | 1–2 天 |
| 人工复核闭环与 manifest | 0.5–1 天 |
| 测试、报告和全量估算 | 0.5–1 天 |

总计预计 4–7 个工作日。若原生 Windows 对齐环境必须迁移到 WSL2，或实际朗读文本与公开版本差异较大，预计额外增加约 0.5–1 天。

实施时采用此前批准的 Subagent-Driven 流程：

- 使用独立 worktree 和 `codex/` 功能分支。
- 每个任务先写失败测试，再实现。
- 每个任务独立提交。
- 每个任务分别进行规格符合性和代码质量审查。
- 所有真实本地数据保持在 Git 边界之外。

## 23. 主要风险与缓解

| 风险 | 缓解 |
| --- | --- |
| 文件标题对应多个祷文或经文版本 | 保存所有候选快照和差异，人工确定 `source_text` 与 `spoken_text` |
| 停顿分布不是清晰的两类 | 每文件聚类；歧义时人工标边界，不强行套全局阈值 |
| 自然呼吸被误判为两遍分割 | 在长停顿文本组内联合比较时长、CTC 覆盖和词序 |
| 两遍实际存在漏词或不同读法 | 分别对齐，保留 `TAKE_TEXT_MISMATCH`，允许单独批准或拒绝 |
| MMS 字符对齐不能验证教会式音素 | 对齐文本和发音计划分离；不输出伪造音素时间戳 |
| Windows 安装或 CUDA 版本冲突 | 后端隔离、固定版本，WSL2 作为兼容回退 |
| CC BY-NC 模型影响未来商业路径 | 当前仅本地研究；记录许可证，商业化前更换后端或评估 |
| 自动指标掩盖错误 | 不设自动批准，试点内全部结果人工复核 |
| 本地真实数据误入 Git | 忽略整个 `local-data/`，测试 Git 忽略规则，manifest 只用相对路径 |
| 阶段 2 重依赖破坏阶段 1 零依赖核心 | 独立 corpus 环境、延迟导入、继续运行阶段 1 门禁 |

## 24. 参考实现与工具来源

- [Silero VAD 官方仓库](https://github.com/snakers4/silero-vad)
- [ctc-forced-aligner 仓库](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [MMS 1130 语言强制对齐模型](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)
- [Meta MMS 原始强制对齐数据准备说明](https://github.com/facebookresearch/fairseq/tree/main/examples/mms/data_prep)
- [TorchAudio `forced_align` 弃用说明](https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.forced_align.html)
- [Montreal Forced Aligner 官方模型目录](https://mfa-models.readthedocs.io/en/latest/acoustic/index.html)
- [FFmpeg Filters 文档](https://ffmpeg.org/ffmpeg-filters.html)
- [ffprobe 文档](https://ffmpeg.org/ffprobe.html)

这些工具是可替换实现。manifest、状态机、三层文本、原始文件不可变、人工批准和音素时间戳可信度边界是本规格的固定约束。
