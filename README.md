# LatinTTS

LatinTTS 当前提供一个规则优先、可追溯的现代罗马/意大利式教会拉丁语发音核心。阶段 1
把单词和短语转换为规范文本、音节、词重音、宽式 IPA 与模型音素，并保留命中的规则、
来源和不确定性诊断。它还不是可直接生成音频的 TTS 应用。

## 阶段 1 范围

- 变体感知的 Unicode 与 `æ/ae`、`œ/oe`、`j/i`、`u/v` 规范化。
- 来源可追溯的音节划分、词重音解析和罗马教会式 G2P。
- `PronunciationPlan` Python API，以及人工重音/音素覆盖的输入验证。
- 350 条已审核单词黄金集和自动审计、精确匹配、规则覆盖发布门禁。

阶段 1 不包含录音文本恢复、音频切分或强制对齐，不包含 MMS-TTS、Piper/VITS 或
StyleTTS2 的推理与训练，不包含网页，也不处理句子、祷文或经文的长文本停顿、呼吸组和
韵律。黄金集只锁定单词级发音，不编码歌唱时值或圣咏旋律。

## 开发环境

项目要求 Python `>=3.10,<3.11`。在 Windows PowerShell 中创建环境并安装
`pyproject.toml` 声明的开发依赖。IPA 包含 GBK 无法编码的字符，因此必须在当前
PowerShell 会话调用 Python 之前启用 UTF-8 模式；该环境变量也适用于随后运行的示例和验证命令：

```powershell
$env:PYTHONUTF8 = "1"
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Python 使用示例

```python
from latintts import Pronouncer

plan = Pronouncer.default().analyze("Ave Maria")
for token in plan.tokens:
    print(token.surface, token.syllables, token.stress_index, token.ipa)
```

输出包含原词、音节、零基重音音节索引和宽式 IPA；`plan` 还保留原文跨度、规则版本、
实际规则 ID、来源 ID、模型音素和诊断。

## 完整验证

```powershell
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m mypy src
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q
.venv\Scripts\python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
git diff --check
```

审计使用最终发布策略：至少 320 条记录，并分别要求 `vowels >= 20`、
`diphthongs >= 20`、`consonants >= 100`、`syllabification >= 40`、`stress >= 60`、
`orthographic_variants >= 30`、`liturgical >= 50`。当前固定数据集为 350 条。

## 规则、来源与数据使用

- [现代罗马教会式拉丁语发音规范](docs/pronunciation/roman-ecclesiastical.md)
- [来源登记](src/latintts/resources/pronunciation_sources.json)
- [已批准的系统设计](docs/superpowers/specs/2026-07-17-latintts-system-design.md)

公开可访问不等于可以不受限制地再分发或用于模型训练。来源登记中的许可和用途说明只是
项目内的审慎记录；使用扫描件、词典、音频或派生数据前，仍须逐来源核对适用法域、署名、
共享方式及训练用途。候选公开录音不能自动成为规范证据或训练数据。

## 后续路线

后续继续以规则核心作为发音事实主线，同时用 MMS-TTS Latin 建立明确标记为
`text_only_baseline` 的基线。录音恢复与对齐稳定后，再训练可直接消费受控音素的
Piper/VITS 主线；只有发音规则、数据对齐和评测集稳定，而且自然度仍是主要短板时，才评估
StyleTTS2。
