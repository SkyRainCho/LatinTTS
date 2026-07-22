# 教会拉丁语语料对齐试点操作指南

本指南用于运行 2–3 条口语录音的端到端试点。流程止于人工批准的 `segments.jsonl` 和
扩展判断报告；它不会启动全量处理、MMS-TTS 基线、TTS 训练或歌唱语料对齐。原始录音必须
始终保留外部备份，并以“复制”方式进入项目。

## 1. 安装 FFmpeg，并运行环境检查

安装或暴露 `ffmpeg` 与 `ffprobe`，确认它们可以从当前 PowerShell 的 `PATH` 中调用。然后
运行：

```powershell
$env:PYTHONUTF8 = "1"
python -m latintts.corpus doctor
```

`doctor` 还会检查试点所需的可选 Python 包。任何 `ALIGNER_UNAVAILABLE` 都必须先解决，不能
通过跳过检查或手工伪造产物继续。

## 2. 创建独立的 Python 3.10 语料环境

```powershell
py -3.10 -m venv .venv-corpus
.venv-corpus\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

不要把阶段 2 的重依赖装进阶段 1 的零运行时依赖环境。

## 3. 安装语料依赖和本机 CUDA PyTorch

```powershell
python -m pip install -r requirements/corpus.txt
```

再按本机驱动和 CUDA 兼容性选择 PyTorch 官方 wheel；不要把机器相关的 wheel URL 固定进
仓库。安装后记录 `python --version`、PyTorch、CUDA、驱动和显卡版本。RTX 4080 的实际
CUDA 可用性以 smoke test 为准，而不是以包安装成功为准。

## 4. 运行对齐 smoke test

```powershell
python -m latintts.corpus align --smoke-test
# 仅在首次获取固定 revision 的模型时使用：
python -m latintts.corpus align --smoke-test --allow-download
```

首次需要拉取已固定版本的模型时，显式使用 `--allow-download`。检查
`local-data/derived/corpus-v1/alignments/runtime/smoke.json` 和同目录的
`environment.txt`：模型 revision、许可证、权重哈希、CUDA、实时因子和峰值显存都必须有
可追溯记录。失败时保留文件用于诊断，不要手工改成成功。

本试点固定的首选 MMS 对齐模型是
`MahmoudAshraf/mms-300m-1130-forced-aligner`，许可证为 `CC-BY-NC-4.0`，仅用于本地、
非商业研究试点。该模型许可不代表原始录音、文本、派生音频或 manifest 自动获得发布或商业
使用权；这些材料仍以各自的 `rights.jsonl` 证据为准。

## 5. 复制原始音频

把口语录音复制到 `local-data/raw/spoken/`，把歌唱或圣咏复制到
`local-data/raw/sung/`。不要移动或转码原件，不要删除外部备份。一个文件可以是一篇完整
祷文或经文；管线会在后续阶段识别文本单元及其两遍朗读。

## 6. 生成并完成清点与权利材料

先生成 `local-data/manifests/intake.csv`，填写每条录音的标题、说话人、`rights_id` 和说明。
再创建 `rights.jsonl`；只有明确允许本地处理、模型训练和内部评估的条目才能进入最终语料。
发布权限未知时保留 `unknown`，不得推定许可。

## 7. 清点并选择试点

运行正式清点，核对录音哈希、时长、格式和 `spoken`/`sung` 分类。然后选择 2–3 条口语
录音；默认策略按短、中、长覆盖，也可以用重复的 `--recording-id ID` 显式选择。提交选择后
不要替换源文件；源哈希变化必须回到清点阶段处理。

## 8. 准备三层文本和逐行朗读单元

运行 `prepare-text --init` 生成 `transcript-intake.jsonl`。为每条入选录音保存公开文本的本地
快照，登记 URL、版本和访问日期；把实际朗读内容写入 UTF-8 文件，每行一个文本单元。明确
区分 `source_text`、`spoken_text` 和 `normalized_text`，只在人工核对后设置
`confirmed=true`。ASR 只能作为差异提示，不能覆盖已确认文本。

## 9. 运行切分、重复配对、词级对齐并导出审核包

依次运行 `prepare-text`、`segment`、`pair`、`align` 和 `export-review`。在 `segment` 后先
试听长停顿形成的文本单元；在 `pair` 后检查每个 `TAKE_*` 问题。异常候选必须保留并进入
人工判断，不能静默删除。`align` 只声明词级边界，不生成音素时间戳。

## 10. 人工审核两遍朗读

逐个打开审核组，完整试听 `take-1.wav` 与 `take-2.wav`。必要时在 Praat 中编辑对应
TextGrid 的 take 和 word 边界，再在 `decision.json` 中分别填写 `approved` 或 `rejected`、
理由、审核人和带时区时间。两遍可以独立批准或拒绝。运行 `import-review` 后，所有修改都会
成为不可变的 before/after 事件；移动后又改回的边界仍算“曾人工修改”。

## 11. 构建 manifest 并生成报告

先运行 `build-manifest`，再填写唯一一行
`local-data/manifests/pilot-telemetry.json`。文件虽以 `.json` 结尾，内容遵循本项目的一行一
对象 JSONL 约定，例如：

```json
{"schema_version":"1","text_preparation_seconds":900.0,"review_seconds":1800.0,"alignment_wall_seconds":120.0,"gpu_retry_count":0,"peak_gpu_memory_bytes":4294967296,"pilot_storage_bytes":1073741824}
```

约束如下：文本准备和审核秒数必须是有限非负数；对齐墙钟秒数、峰值显存和试点存储必须
大于零；GPU 重试次数必须是非负整数。以下字段必须来自这一次正式试点的实测记录，不能猜测，
也不能拿 smoke test 的数值代替正式 `pair + align` 工作负载：

- `text_preparation_seconds`：从开始整理入选录音文本到三层文本和逐行单元确认完成，逐次记录
  实际操作会话的开始、结束时间并累加活跃墙钟时间；暂停和隔夜等待不计入。
- `review_seconds`：从文本确认完成后，逐会话累加所有人工对齐与复核活跃时间，包括停顿单元
  检查、pairing correction、两遍试听、TextGrid 边界编辑和 decision；暂停不计入，并且不得
  与 `text_preparation_seconds` 重叠。
- `alignment_wall_seconds`：记录正式 `pair` 与 `align` 的全部 forced-alignment 工作负载；
  用 `Measure-Command` 分别包住两个阶段，累加两阶段及失败尝试的 `TotalSeconds`。CTC 推理通常已在
  `pair` 发生，不能用仅命中 cache 的 `align` 冒充正式 GPU 工作负载，也不能用处理事件时间戳
  代替实际计时。
- `gpu_retry_count`：正式 `pair` 或 `align` 因 GPU/CUDA/OOM 等运行失败后重新执行的次数；
  没有重试才填 `0`。文本或审核返工不算 GPU 重试。
- `peak_gpu_memory_bytes`：正式 `pair` 与 `align` 期间在另一终端持续采样该 Python 进程的
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv -lms 200`，取峰值并换算为字节；
  不使用 `runtime/smoke.json` 的 model-load 峰值。
- `pilot_storage_bytes`：只统计能够归属于入选录音的 corpus 数据和派生产物，包括入选 raw
  副本、文本快照、分析/候选/审核/最终音频、录音专属对齐/配对产物；共享 JSONL 只累计属于
  入选录音的行（含换行）。排除未入选录音、模型权重与 Hugging Face cache、共享 runtime、
  lock、临时文件、`report.json` 和 `report.md`。记录纳入路径和逐项字节数，避免重复计数。

例如，可用下面的 PowerShell 片段分别实测两个正式阶段，再把两项 `TotalSeconds` 相加；命令
失败也要保留本次耗时并增加重试计数：

```powershell
$pairRun = Measure-Command { python -m latintts.corpus pair }
$alignRun = Measure-Command { python -m latintts.corpus align }
$pairRun.TotalSeconds + $alignRun.TotalSeconds
```

最后运行 `report`，得到确定性的 `report.json` 和 `report.md`。

报告公式和语义：

- 输入时长是入选口语录音的原始时长；VAD 时长是各入选录音有效语音区间之和。
- `auto_pairing_correct_ratio` 为“原始自动配对无需纠正而直接 accepted”的代理指标，不是
  发音或内容正确性的自动证明；合法人工配对后仍从 `pairing-automatic.json` 计数。
- 边界无需修改率按完整事件历史计算；出现过 `segment_start` 或 `segment_end` 事件即算改过。
- 可批准语音占比为批准 take 的有效时长除以批准、拒绝和待复核 take 的有效总时长，不以
  VAD 时长为分母。
- 人工复核倍率为 `review_seconds / selected_input_seconds`；GPU 实时因子为
  `alignment_wall_seconds / selected_input_seconds`。
- 全量缩放系数为“完整口语 inventory 时长 / 试点输入时长”。人时和 GPU 时按该系数线性
  外推，存储字节向上取整。未入选录音只参与全量时长，不参与试点计数。

扩展判断严格使用已批准区间：全部达到自动配对 `≥95%`、边界不改 `≥85%`、复核倍率
`≤3×`、批准占比 `≥80%` 才是 `scalable`；任一达到自动配对 `<85%`、边界不改 `<70%`、
复核倍率 `>6×`、批准占比 `<60%`，或仍有待复核时，为 `not_ready`；其余为 `optimize`。

## 12. 确认真实数据仍在 Git 边界外

```powershell
git check-ignore -v local-data/raw/spoken/example.wav
git status --short
git diff --cached --name-only
```

`git check-ignore` 必须指出项目的 `local-data/` 规则，另外两个命令不得出现真实录音、文本
快照、审核音频、模型权重或本机报告。不要使用 `git add -f local-data`。

## 13. 按稳定错误码恢复，不删除原始数据

所有恢复都应先修正输入或环境，再原样重跑失败命令。内容寻址缓存和处理事件负责安全复用；
不要通过删除 `local-data/raw/`、手工改状态或伪造 JSON 来“越过”错误。
如果 `report` 报错列出以 `.recovery.` 开头的备份，不要删除这些文件；按报错给出的绝对路径
核对并恢复旧的 `report.json` / `report.md` 文件对，再重新运行 `report`。

| 错误码 | 恢复动作 |
| --- | --- |
| `ALIGNER_UNAVAILABLE` | 修复 PATH、可选包、模型快照或 CUDA；重跑 `doctor`/smoke test。 |
| `INVENTORY_UNSUPPORTED_FORMAT` | 核对容器、音频流和 ffprobe 输出；保留原件，另行制作有来源记录的副本。 |
| `INVENTORY_HASH_MISMATCH` | 停止处理，核对复制来源与外部备份；不要覆盖已登记文件。 |
| `RIGHTS_SCOPE_UNCONFIRMED` | 补充明确授权；未知权限不得推断为允许。 |
| `TRANSCRIPT_SOURCE_NOT_FOUND` | 补齐可追溯的来源快照和来源登记，再重新准备文本。 |
| `TRANSCRIPT_AMBIGUOUS` | 人工比较候选版本并明确实际朗读来源，不让 ASR 自动裁决。 |
| `TRANSCRIPT_SPOKEN_MISMATCH` | 修正实际朗读文本或逐行单元，使其与录音和配对输入一致。 |
| `PRONUNCIATION_NEEDS_REVIEW` | 人工确认发音、重音和规范化结果后再继续。 |
| `PAUSE_CLASSES_AMBIGUOUS` | 人工试听停顿；保留证据后调整每文件切分策略。 |
| `TAKE_COUNT_MISMATCH` | 人工确认两遍朗读范围，并在 pairing correction 中保存合法 split。 |
| `TAKE_DURATION_MISMATCH` | 试听时长异常的两遍候选，选择或修正 split，不删除异常组。 |
| `TAKE_TEXT_MISMATCH` | 核对逐行文本与候选音频，在 pairing correction 中修正后重跑。 |
| `ALIGNMENT_TEXT_UNSUPPORTED` | 修正对齐层不支持的字符或空文本；保留 source/spoken 层原文。 |
| `ALIGNMENT_LOW_CONFIDENCE` | 复核文本和音频，检查模型/runtime；不得自动批准。 |
| `AUDIO_QUALITY_REJECTED` | 核对源媒体和转码环境；保留 raw，不做破坏性降噪覆盖。 |
| `CACHE_ARTIFACT_INVALID` | 停止并调查哈希、来源链或中断恢复；只清理已确认损坏的派生缓存。 |
| `REVIEW_REQUIRED` | 完成缺失的人工 decision/边界审核，再导入并重跑。 |
| `MANIFEST_SCHEMA_MISMATCH` | 按当前 schema 修正字段、版本或交叉引用；不要手改状态跳级。 |

## 完整命令顺序

以下顺序是试点的标准主流程：

```powershell
python -m latintts.corpus doctor
python -m latintts.corpus inventory --init-intake
python -m latintts.corpus inventory
python -m latintts.corpus select-pilot
python -m latintts.corpus prepare-text --init
python -m latintts.corpus prepare-text
python -m latintts.corpus segment
python -m latintts.corpus pair
python -m latintts.corpus align
python -m latintts.corpus export-review
python -m latintts.corpus import-review
python -m latintts.corpus build-manifest
python -m latintts.corpus report
```
