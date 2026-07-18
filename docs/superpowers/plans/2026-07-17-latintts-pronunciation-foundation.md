# LatinTTS 阶段 1：发音规范基础 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可追溯、可测试的现代罗马教会式拉丁语文本规范化、音节划分、重音解析和 G2P 核心，并用至少 320 个来源明确的黄金词条锁定行为。

**Architecture:** 使用纯 Python `latin_core` 管线把原始文本转换为 `PronunciationPlan`。规则、来源、重音词典和黄金词表均版本化；每个阶段先写失败测试，再实现最小行为。该计划只覆盖系统设计中的阶段 1，不处理录音对齐、MMS、Piper、网页或长文本韵律。

**Tech Stack:** Windows PowerShell、Python 3.10.6、标准库 `dataclasses`/`enum`/`unicodedata`/`importlib.resources`、pytest、pytest-cov、Ruff、mypy、setuptools。

## Global Constraints

- 目标发音固定为现代罗马/意大利式教会拉丁语。
- 质量优先级固定为：发音正确、重音正确、清晰、自然、音色相似。
- 原始输入必须保留；规范化不得破坏来源定位。
- `æ/ae`、`œ/oe`、`j/i`、`u/v` 通过变体感知检索处理；禁止无条件全局替换 `j/i` 或 `u/v`。
- 普通拼写无法唯一确定重音时必须返回 `PRONUNCIATION_NEEDS_REVIEW`，不得静默宣称确定。
- `PronunciationPlan` 是后续模型层的发音事实来源。
- 每条规则、词典记录和黄金词条必须引用已登记的 `source_id`。
- Python 版本固定为 `>=3.10,<3.11`，与当前本机 Python 3.10.6 一致。
- 阶段 1 只使用标准库运行时依赖；测试和静态检查工具仅放入 `dev` 可选依赖。
- 所有任务遵循 TDD；每个任务单独提交，不混入 `.superpowers/`。

---

## File Map

```text
.gitignore                                      本地环境、缓存和可视化会话排除规则
pyproject.toml                                  Python 包、测试、Ruff、mypy 配置
README.md                                       阶段 1 使用说明和验证命令
docs/pronunciation/roman-ecclesiastical.md      完整发音规则、来源和决策记录
src/latintts/__init__.py                        稳定公共 API 导出
src/latintts/domain.py                          PronunciationPlan 领域类型和不变量
src/latintts/normalization.py                   词级规范化、检索键和原文跨度
src/latintts/syllables.py                       音节核识别和音节划分
src/latintts/stress.py                          重音词典加载和重音解析
src/latintts/g2p.py                             来源可追溯的罗马教会式 G2P
src/latintts/validation.py                      人工覆盖与规范音素一致性验证
src/latintts/pipeline.py                        各组件编排和 PronunciationPlan 构建
src/latintts/sources.py                         来源登记加载与校验
src/latintts/audit.py                           黄金词表审计 CLI
src/latintts/resources/__init__.py              包内资源命名空间
src/latintts/resources/pronunciation_sources.json  来源登记
src/latintts/resources/stress_lexicon.jsonl     带来源的重音词典
src/latintts/resources/g2p_exceptions.jsonl     非通用规则例外
tests/unit/test_domain.py                       领域类型不变量
tests/unit/test_sources.py                      来源登记校验
tests/unit/test_normalization.py                规范化和跨度
tests/unit/test_syllables.py                    音节划分
tests/unit/test_stress.py                       重音优先级和不确定性
tests/unit/test_g2p.py                          G2P 规则
tests/unit/test_validation.py                   覆盖值与未知音素验证
tests/unit/test_pipeline.py                     管线组合和人工覆盖
tests/unit/test_audit.py                        黄金词表审计
tests/integration/test_gold_pronunciations.py   320 词端到端精确匹配
tests/fixtures/gold_pronunciations.jsonl        黄金词表
```

文件边界固定如下：`domain.py` 不读取资源；`normalization.py` 不做音节或重音判断；`syllables.py` 不访问词典；`stress.py` 不生成 IPA；`g2p.py` 不决定重音；只有 `pipeline.py` 组合这些组件。

---

### Task 1: Python 包骨架与领域契约

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/latintts/__init__.py`
- Create: `src/latintts/domain.py`
- Test: `tests/unit/test_domain.py`

**Interfaces:**
- Consumes: 无。
- Produces: `Severity`、`ResolutionMethod`、`Diagnostic`、`PronunciationOverride`、`PronunciationToken`、`PronunciationPlan`。

- [ ] **Step 1: 创建项目元数据和本地环境配置**

`pyproject.toml` 使用以下完整配置：

```toml
[build-system]
requires = ["setuptools>=75,<82"]
build-backend = "setuptools.build_meta"

[project]
name = "latintts"
version = "0.1.0"
description = "Rule-first Ecclesiastical Latin pronunciation core"
readme = "README.md"
requires-python = ">=3.10,<3.11"
dependencies = []

[project.optional-dependencies]
dev = [
  "mypy>=1.13,<2",
  "pytest>=8,<9",
  "pytest-cov>=5,<7",
  "ruff>=0.11,<1",
]

[tool.setuptools]
package-dir = {"" = "src"}
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
latintts = ["resources/*.json", "resources/*.jsonl"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers --strict-config"
testpaths = ["tests"]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.mypy]
python_version = "3.10"
strict = true
packages = ["latintts"]
```

`.gitignore` 使用：

```gitignore
.venv/
.superpowers/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/
build/
dist/
*.egg-info/
```

运行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Expected: 安装成功，并显示 `Successfully installed latintts`。

- [ ] **Step 2: 写领域不变量失败测试**

`tests/unit/test_domain.py`：

```python
import pytest

from latintts.domain import (
    PronunciationPlan,
    PronunciationToken,
    ResolutionMethod,
)


def test_token_rejects_stress_outside_syllables() -> None:
    with pytest.raises(ValueError, match="stress_index"):
        PronunciationToken(
            surface="ave",
            normalized="ave",
            source_span=(0, 3),
            syllables=("a", "ve"),
            stress_index=2,
            ipa="ˈa.ve",
            model_phonemes=("ˈ", "a", "v", "e"),
            resolution_method=ResolutionMethod.RULE,
        )


def test_plan_rejects_token_span_outside_original_text() -> None:
    token = PronunciationToken(
        surface="ave",
        normalized="ave",
        source_span=(0, 3),
        syllables=("a", "ve"),
        stress_index=0,
        ipa="ˈa.ve",
        model_phonemes=("ˈ", "a", "v", "e"),
        resolution_method=ResolutionMethod.RULE,
    )
    with pytest.raises(ValueError, match="source_span"):
        PronunciationPlan(
            schema_version="1",
            rule_version="ecclesiastical-roman-v1",
            original_text="av",
            normalized_text="ave",
            tokens=(token,),
            phrase_phonemes=token.model_phonemes,
        )
```

- [ ] **Step 3: 运行测试确认失败**

Run:

```powershell
.venv\Scripts\python -m pytest tests/unit/test_domain.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError` or missing domain symbols.

- [ ] **Step 4: 实现领域类型**

`src/latintts/domain.py`：

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SourceSpan = tuple[int, int]


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ResolutionMethod(str, Enum):
    OVERRIDE = "override"
    EXCEPTION = "exception"
    LEXICON = "lexicon"
    RULE = "rule"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    message: str
    severity: Severity
    source_span: SourceSpan | None = None
    token_index: int | None = None


@dataclass(frozen=True, slots=True)
class PronunciationOverride:
    stress_index: int | None = None
    ipa: str | None = None
    model_phonemes: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class PronunciationToken:
    surface: str
    normalized: str
    source_span: SourceSpan
    syllables: tuple[str, ...]
    stress_index: int
    ipa: str
    model_phonemes: tuple[str, ...]
    resolution_method: ResolutionMethod
    applied_rule_ids: tuple[str, ...] = ()
    normalization_transforms: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    warnings: tuple[Diagnostic, ...] = ()
    override: PronunciationOverride | None = None

    def __post_init__(self) -> None:
        start, end = self.source_span
        if start < 0 or end <= start:
            raise ValueError("source_span must be a non-empty half-open range")
        if not self.syllables:
            raise ValueError("syllables must not be empty")
        if not 0 <= self.stress_index < len(self.syllables):
            raise ValueError("stress_index must point to an existing syllable")


@dataclass(frozen=True, slots=True)
class PronunciationPlan:
    schema_version: str
    rule_version: str
    original_text: str
    normalized_text: str
    tokens: tuple[PronunciationToken, ...]
    phrase_phonemes: tuple[str, ...]
    warnings: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for token in self.tokens:
            if token.source_span[1] > len(self.original_text):
                raise ValueError("token source_span exceeds original_text")
```

`src/latintts/__init__.py` 导出 `PronunciationPlan` 和 `PronunciationToken`，并设置 `__version__ = "0.1.0"`。

- [ ] **Step 5: 运行测试确认通过**

Run:

```powershell
.venv\Scripts\python -m pytest tests/unit/test_domain.py -q
```

Expected: `2 passed`。

- [ ] **Step 6: 运行静态检查并提交**

```powershell
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\python -m mypy src
git add .gitignore pyproject.toml src/latintts/__init__.py src/latintts/domain.py tests/unit/test_domain.py
git commit -m "feat: add pronunciation domain contracts"
```

Expected: Ruff 与 mypy exit 0；提交只包含本任务文件。

---

### Task 2: 来源登记与规范文档基线

**Files:**
- Create: `src/latintts/resources/__init__.py`
- Create: `src/latintts/resources/pronunciation_sources.json`
- Create: `src/latintts/sources.py`
- Create: `docs/pronunciation/roman-ecclesiastical.md`
- Test: `tests/unit/test_sources.py`

**Interfaces:**
- Consumes: Python 包资源配置。
- Produces: `SourceRecord`、`load_source_registry() -> dict[str, SourceRecord]`、稳定 `source_id` 集合。

- [ ] **Step 1: 写来源登记失败测试**

```python
from latintts.sources import load_source_registry


def test_registry_contains_normative_and_evaluation_sources() -> None:
    sources = load_source_registry()
    assert "liber-usualis-1962" in sources
    assert sources["liber-usualis-1962"].authority_rank == 1
    assert "wikimedia-ecclesiastical-pronunciation" in sources
    assert all(record.accessed_on == "2026-07-17" for record in sources.values())


def test_registry_has_unique_ids_and_nonempty_usage_terms() -> None:
    sources = load_source_registry()
    assert len(sources) == 6
    assert all(record.license_or_terms.strip() for record in sources.values())
    assert all(record.usage_note.strip() for record in sources.values())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/unit/test_sources.py -q`

Expected: FAIL because `latintts.sources` does not exist.

- [ ] **Step 3: 添加来源模型与加载器**

`src/latintts/sources.py`：

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    title: str
    url: str
    source_kind: str
    authority_rank: int
    accessed_on: str
    locator: str
    license_or_terms: str
    usage_note: str


def load_source_registry() -> dict[str, SourceRecord]:
    raw = files("latintts.resources").joinpath("pronunciation_sources.json").read_text(
        encoding="utf-8"
    )
    rows = json.loads(raw)
    records = [SourceRecord(**row) for row in rows]
    result = {record.source_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("pronunciation source IDs must be unique")
    if any(record.authority_rank < 1 for record in records):
        raise ValueError("authority_rank must be positive")
    return result
```

- [ ] **Step 4: 添加六个种子来源**

`pronunciation_sources.json` 写入六条完整记录：

```json
[
  {
    "source_id": "liber-usualis-1962",
    "title": "Liber Usualis 1962 - The reading and pronunciation of liturgical Latin",
    "url": "https://archive.ccwatershed.org/media/pdfs/12/07/06/17-07-32_0.pdf",
    "source_kind": "normative_traditional",
    "authority_rank": 1,
    "accessed_on": "2026-07-17",
    "locator": "PDF pages 29-32, lines 1229-1357 in the searchable copy",
    "license_or_terms": "Publicly accessible historical scan; verify jurisdiction before redistribution",
    "usage_note": "Primary rule source for Roman-style liturgical pronunciation"
  },
  {
    "source_id": "ewtn-ecclesiastical-latin",
    "title": "EWTN Ecclesiastical Latin pronunciation guide",
    "url": "https://www.ewtn.com/catholicism/answers/latin-ecclesiastical-24788",
    "source_kind": "secondary_guide",
    "authority_rank": 2,
    "accessed_on": "2026-07-17",
    "locator": "Pronunciation tables adapted from the Liber Usualis",
    "license_or_terms": "Reference use only; do not redistribute page content",
    "usage_note": "Cross-check for explanatory wording, not a replacement for the primary source"
  },
  {
    "source_id": "allen-greenough-accents",
    "title": "Allen and Greenough's New Latin Grammar - Accents",
    "url": "https://dcc.dickinson.edu/sv/grammar/latin/accents",
    "source_kind": "grammar_reference",
    "authority_rank": 2,
    "accessed_on": "2026-07-17",
    "locator": "Section 12, Accents",
    "license_or_terms": "Reference use with attribution; retain the suggested citation",
    "usage_note": "Source for disyllable, penult, antepenult, and enclitic stress rules"
  },
  {
    "source_id": "perseus-lewis-short",
    "title": "Lewis and Short, A Latin Dictionary - Perseus XML source",
    "url": "https://github.com/PerseusDL/lexica/tree/master/CTS_XML_TEI/perseus/pdllex/lat/ls",
    "source_kind": "quantity_dictionary",
    "authority_rank": 2,
    "accessed_on": "2026-07-17",
    "locator": "Per-entry XML lemma and orthography fields",
    "license_or_terms": "CC BY-SA 3.0 US; preserve Perseus attribution and share-alike terms",
    "usage_note": "Lexical evidence for vowel quantity and stress; record the exact entry locator per lexicon row"
  },
  {
    "source_id": "wikimedia-ecclesiastical-pronunciation",
    "title": "Wikimedia Commons category: Ecclesiastic Latin pronunciation",
    "url": "https://commons.wikimedia.org/wiki/Category:Ecclesiastic_Latin_pronunciation",
    "source_kind": "evaluation_audio_index",
    "authority_rank": 3,
    "accessed_on": "2026-07-17",
    "locator": "Per-file description and license pages",
    "license_or_terms": "Each audio file has its own license and must be checked individually",
    "usage_note": "Candidate evaluation audio; never auto-promote to normative or training data"
  },
  {
    "source_id": "librivox-public-domain",
    "title": "LibriVox public domain policy",
    "url": "https://librivox.org/pages/public-domain/",
    "source_kind": "candidate_audio_terms",
    "authority_rank": 4,
    "accessed_on": "2026-07-17",
    "locator": "Public Domain page",
    "license_or_terms": "Recordings are public domain in the United States; verify local jurisdiction",
    "usage_note": "Candidate corpus index only; pronunciation variety requires manual screening"
  }
]
```

- [ ] **Step 5: 编写规则文档基线**

`docs/pronunciation/roman-ecclesiastical.md` 必须包含以下已确定章节：目标与非目标、来源优先级、Unicode 与拼写规范化、元音、双元音与相邻元音、辅音、音节划分、重音、双辅音、词间处理、例外、IPA 与规范音素表、规则覆盖矩阵、冲突决策记录。

首批规则表必须逐行引用 `liber-usualis-1962`，并写入以下来源位置：

- 元音与双元音：PDF lines 1254-1297。
- `c/cc/sc/ch/g/gn/h/j`：PDF lines 1298-1324。
- `r/s/ti/th/x/xc/y/z` 和双辅音：PDF lines 1325-1354。
- 每个音节完整发音与重音区别：PDF lines 1231-1247。
- 双音节、重 penult、轻 penult、antepenult 和附着词重音：`allen-greenough-accents` Section 12。

规则文档明确采用以下工程政策：`s` 在元音间的完整音位输出先保持 `/s/` 并记录“轻微软化”为朗读实现注释；只有获得更精确且不冲突的来源后才改变音位。这样避免把来源中的音质描述未经证明地等同为 `/z/`。

- [ ] **Step 6: 运行测试、检查文档并提交**

```powershell
.venv\Scripts\python -m pytest tests/unit/test_sources.py -q
.venv\Scripts\python -m ruff check src tests
git diff --check
git add docs/pronunciation/roman-ecclesiastical.md src/latintts/resources src/latintts/sources.py tests/unit/test_sources.py
git commit -m "docs: establish ecclesiastical pronunciation sources"
```

Expected: 来源测试 `2 passed`；文档无空白错误；六个来源均可通过包资源加载。

---

### Task 3: 词级规范化与原文跨度

**Files:**
- Create: `src/latintts/normalization.py`
- Test: `tests/unit/test_normalization.py`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`

**Interfaces:**
- Consumes: 原始 Unicode 文本。
- Produces: `WordSpan(surface, start, end)`、`NormalizedWord(surface, normalized, lookup_key, marked_vowel_index, transformations)`、`tokenize_words()`、`normalize_word()`、`normalize_phrase()`。

- [ ] **Step 1: 写规范化失败测试**

```python
from latintts.normalization import normalize_phrase, normalize_word, tokenize_words


def test_ligatures_expand_without_losing_original_span() -> None:
    text = "Cælum et cœli"
    spans = tokenize_words(text)
    assert [(item.surface, item.start, item.end) for item in spans] == [
        ("Cælum", 0, 5),
        ("et", 6, 8),
        ("cœli", 9, 13),
    ]
    assert normalize_word(spans[0].surface).normalized == "caelum"
    assert normalize_word(spans[2].surface).normalized == "coeli"


def test_lookup_key_unifies_variants_but_normalized_text_preserves_roles() -> None:
    assert normalize_word("jam").normalized == "jam"
    assert normalize_word("iam").normalized == "iam"
    assert normalize_word("jam").lookup_key == normalize_word("iam").lookup_key
    assert normalize_word("servus").lookup_key == "seruus"


def test_normalization_records_stable_transformation_ids() -> None:
    assert normalize_word("Cælum").transformations == (
        "casefold",
        "expand-ae-ligature",
    )
    assert "lookup-j-to-i" in normalize_word("jam").transformations
    assert "lookup-v-to-u" in normalize_word("servus").transformations


def test_acute_mark_becomes_explicit_stress_hint() -> None:
    word = normalize_word("Dóminus")
    assert word.normalized == "dominus"
    assert word.marked_vowel_index == 1


def test_phrase_normalization_is_stable_and_keeps_punctuation() -> None:
    assert normalize_phrase("  Ave\tMaria,\n  gratia. ") == "Ave Maria, gratia."
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/unit/test_normalization.py -q`

Expected: FAIL because the module is missing.

- [ ] **Step 3: 实现跨度扫描和规范化**

使用不可变 `WordSpan` 和 `NormalizedWord` 数据类。`tokenize_words()` 逐字符扫描 `str.isalpha()` 和 Unicode combining mark；跨度始终指向原文的半开区间。`normalize_word()` 执行 NFC、`casefold()`、显式展开 `æ/œ`、移除 acute stress mark、保留 macron，并按 `j -> i`、`v -> u` 生成仅供词典检索的 `lookup_key`。

核心实现必须包含：

```python
from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WordSpan:
    surface: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NormalizedWord:
    surface: str
    normalized: str
    lookup_key: str
    marked_vowel_index: int | None
    transformations: tuple[str, ...]


COMBINING_ACUTE = "\u0301"


def tokenize_words(text: str) -> tuple[WordSpan, ...]:
    result: list[WordSpan] = []
    start: int | None = None
    for index, char in enumerate(text):
        is_mark = unicodedata.category(char).startswith("M")
        is_word_char = char.isalpha() or (start is not None and is_mark)
        if is_word_char and start is None:
            start = index
        elif not is_word_char and start is not None:
            result.append(WordSpan(text[start:index], start, index))
            start = None
    if start is not None:
        result.append(WordSpan(text[start:], start, len(text)))
    return tuple(result)


def normalize_phrase(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _lookup_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    without_marks = "".join(
        char for char in decomposed if not unicodedata.category(char).startswith("M")
    )
    return without_marks.replace("j", "i").replace("v", "u")


def _describe_transformations(surface: str, normalized: str) -> tuple[str, ...]:
    result: list[str] = []
    nfc_surface = unicodedata.normalize("NFC", surface)
    folded_surface = nfc_surface.casefold()
    if nfc_surface != surface:
        result.append("unicode-nfc")
    if folded_surface != nfc_surface:
        result.append("casefold")
    if "æ" in folded_surface:
        result.append("expand-ae-ligature")
    if "œ" in folded_surface:
        result.append("expand-oe-ligature")
    if COMBINING_ACUTE in unicodedata.normalize("NFD", folded_surface):
        result.append("remove-acute-stress-mark")
    if "j" in normalized:
        result.append("lookup-j-to-i")
    if "v" in normalized:
        result.append("lookup-v-to-u")
    return tuple(result)


def normalize_word(surface: str) -> NormalizedWord:
    expanded = unicodedata.normalize("NFC", surface).casefold()
    expanded = expanded.replace("æ", "ae").replace("œ", "oe")
    decomposed = unicodedata.normalize("NFD", expanded)
    output: list[str] = []
    base_index = -1
    marked_vowel_index: int | None = None
    for char in decomposed:
        if not unicodedata.category(char).startswith("M"):
            base_index += 1
            output.append(char)
        elif char == COMBINING_ACUTE:
            marked_vowel_index = base_index
        else:
            output.append(char)
    normalized = unicodedata.normalize("NFC", "".join(output))
    return NormalizedWord(
        surface=surface,
        normalized=normalized,
        lookup_key=_lookup_key(normalized),
        marked_vowel_index=marked_vowel_index,
        transformations=_describe_transformations(surface, normalized),
    )
```

`normalize_phrase()` 只规范 Unicode 和空白，不改变标点。

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/unit/test_normalization.py -q`

Expected: `4 passed`。

- [ ] **Step 5: 运行全量单元测试并提交**

```powershell
.venv\Scripts\python -m pytest tests/unit -q
.venv\Scripts\python -m ruff check src tests
git add src/latintts/normalization.py tests/unit/test_normalization.py docs/pronunciation/roman-ecclesiastical.md
git commit -m "feat: add variant-aware Latin normalization"
```

---

### Task 4: 音节核与音节划分

**Files:**
- Create: `src/latintts/syllables.py`
- Test: `tests/unit/test_syllables.py`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`

**Interfaces:**
- Consumes: `NormalizedWord.normalized`。
- Produces: `syllabify(word: str) -> tuple[str, ...]`、`syllable_ranges(word) -> tuple[tuple[int, int], ...]`。

- [ ] **Step 1: 写音节失败测试**

```python
import pytest

from latintts.syllables import syllabify


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("ave", ("a", "ve")),
        ("gratia", ("gra", "ti", "a")),
        ("ecce", ("ec", "ce")),
        ("sanctus", ("sanc", "tus")),
        ("patris", ("pa", "tris")),
        ("caelum", ("cae", "lum")),
        ("alleluia", ("al", "le", "lu", "ia")),
        ("cui", ("cu", "i")),
        ("qui", ("qui",)),
        ("poëta", ("po", "ë", "ta")),
    ],
)
def test_syllabify_source_backed_examples(word: str, expected: tuple[str, ...]) -> None:
    assert syllabify(word) == expected
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/unit/test_syllables.py -q`

Expected: FAIL because `latintts.syllables` is missing.

- [ ] **Step 3: 实现音节核检测**

使用以下固定集合，并在文档中逐项记录来源：

```python
from __future__ import annotations

import unicodedata


VOWELS = frozenset("aeiouyāēīōūȳ")
DIPHTHONGS = frozenset({"ae", "oe", "au", "eu", "ay"})
ONSET_CLUSTERS = frozenset(
    {
        "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr",
        "pl", "pr", "tr", "qu", "gu", "ch", "ph", "th", "gn",
    }
)
```

`i` 在词首接元音或位于两个元音之间时作为辅音；`u` 在 `q` 或 `ng` 后且后接元音时作为滑音。`cui` 按来源明确保留两音节。

- [ ] **Step 4: 实现边界算法**

实现顺序固定为：识别复合元音核；收集相邻音节核间的辅音；允许的起始辅音簇整体进入下一音节；双辅音从中间分开；其他多辅音簇仅把允许的最长词首簇移到下一音节。`syllable_ranges()` 返回半开区间，`syllabify()` 只负责切片。

核心音节核与边界代码：

```python
def _base_letter(char: str) -> str:
    return unicodedata.normalize("NFD", char)[0]


def _has_diaeresis(char: str) -> bool:
    return "\u0308" in unicodedata.normalize("NFD", char)


def _is_vowel_at(word: str, index: int) -> bool:
    return 0 <= index < len(word) and _base_letter(word[index]) in VOWELS


def _is_consonantal_i(word: str, index: int) -> bool:
    if (
        _base_letter(word[index]) != "i"
        or _has_diaeresis(word[index])
        or not _is_vowel_at(word, index + 1)
    ):
        return False
    return index == 0 or _is_vowel_at(word, index - 1)


def _is_glide_u(word: str, index: int) -> bool:
    if (
        _base_letter(word[index]) != "u"
        or _has_diaeresis(word[index])
        or not _is_vowel_at(word, index + 1)
    ):
        return False
    return word[max(0, index - 2) : index] == "ng" or word[index - 1 : index] == "q"


def _find_nuclei(word: str) -> tuple[tuple[int, int], ...]:
    nuclei: list[tuple[int, int]] = []
    index = 0
    while index < len(word):
        pair = word[index : index + 2]
        if pair in DIPHTHONGS:
            nuclei.append((index, index + 2))
            index += 2
            continue
        if _is_vowel_at(word, index):
            if not _is_consonantal_i(word, index) and not _is_glide_u(word, index):
                nuclei.append((index, index + 1))
        index += 1
    return tuple(nuclei)


def _onset_length(cluster: str) -> int:
    for size in (3, 2):
        if len(cluster) >= size and cluster[-size:] in ONSET_CLUSTERS:
            return size
    return 1


def syllable_ranges(word: str) -> tuple[tuple[int, int], ...]:
    nuclei = _find_nuclei(word)
    if not nuclei:
        raise ValueError("word contains no vowel nucleus")
    starts = [0]
    for previous, current in zip(nuclei, nuclei[1:]):
        cluster = word[previous[1] : current[0]]
        boundary = current[0] if not cluster else current[0] - _onset_length(cluster)
        starts.append(max(previous[1], boundary))
    ends = starts[1:] + [len(word)]
    return tuple(zip(starts, ends))


def syllabify(word: str) -> tuple[str, ...]:
    return tuple(word[start:end] for start, end in syllable_ranges(word))
```

- [ ] **Step 5: 运行测试、更新规则文档并提交**

```powershell
.venv\Scripts\python -m pytest tests/unit/test_syllables.py -q
.venv\Scripts\python -m pytest tests/unit -q
.venv\Scripts\python -m ruff check src tests
git add src/latintts/syllables.py tests/unit/test_syllables.py docs/pronunciation/roman-ecclesiastical.md
git commit -m "feat: add ecclesiastical Latin syllabification"
```

Expected: 参数化音节测试全部通过；文档中的每个规则示例与测试一致。

---

### Task 5: 重音词典与不确定性解析

**Files:**
- Create: `src/latintts/stress.py`
- Create: `src/latintts/resources/stress_lexicon.jsonl`
- Test: `tests/unit/test_stress.py`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`

**Interfaces:**
- Consumes: `lookup_key`、音节、显式重音提示、请求覆盖、来源登记。
- Produces: `StressLexiconEntry`、`StressDecision`、`load_stress_lexicon()`、`resolve_stress()`。

- [ ] **Step 1: 写重音优先级失败测试**

```python
from latintts.domain import ResolutionMethod
from latintts.stress import load_stress_lexicon, resolve_stress


def test_override_wins_over_lexicon() -> None:
    lexicon = load_stress_lexicon()
    decision = resolve_stress("dominus", ("do", "mi", "nus"), lexicon, override_index=1)
    assert decision.stress_index == 1
    assert decision.method is ResolutionMethod.OVERRIDE


def test_lexicon_resolves_open_penult_quantity() -> None:
    lexicon = load_stress_lexicon()
    decision = resolve_stress("dominus", ("do", "mi", "nus"), lexicon)
    assert decision.stress_index == 0
    assert decision.method is ResolutionMethod.LEXICON
    assert "perseus-lewis-short" in decision.source_ids


def test_unknown_open_penult_returns_review_warning() -> None:
    decision = resolve_stress("fabula", ("fa", "bu", "la"), {})
    assert decision.method is ResolutionMethod.CANDIDATE
    assert decision.warnings[0].code == "PRONUNCIATION_NEEDS_REVIEW"


def test_disyllable_uses_first_syllable_rule() -> None:
    decision = resolve_stress("sanctus", ("sanc", "tus"), {})
    assert decision.stress_index == 0
    assert decision.method is ResolutionMethod.RULE


def test_enclitic_stresses_syllable_before_suffix() -> None:
    decision = resolve_stress(
        "dominusque",
        ("do", "mi", "nus", "que"),
        load_stress_lexicon(),
    )
    assert decision.stress_index == 2
    assert decision.method is ResolutionMethod.RULE
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/unit/test_stress.py -q`

Expected: FAIL because the stress module and lexicon are missing.

- [ ] **Step 3: 实现词典加载与决策类型**

`StressLexiconEntry` 字段固定为 `lookup_key`、`syllables`、`stress_index`、`source_ids`、`note`、`is_exception`。加载器逐行解析 JSONL，拒绝重复键、非法重音索引和未知 `source_id`：

```python
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from importlib.resources import files
from typing import Mapping

from latintts.domain import Diagnostic, ResolutionMethod, Severity
from latintts.sources import load_source_registry


@dataclass(frozen=True, slots=True)
class StressLexiconEntry:
    lookup_key: str
    syllables: tuple[str, ...]
    stress_index: int
    source_ids: tuple[str, ...]
    note: str
    is_exception: bool


def load_stress_lexicon() -> dict[str, StressLexiconEntry]:
    text = files("latintts.resources").joinpath("stress_lexicon.jsonl").read_text(
        encoding="utf-8"
    )
    known_sources = set(load_source_registry())
    result: dict[str, StressLexiconEntry] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        entry = StressLexiconEntry(
            lookup_key=raw["lookup_key"],
            syllables=tuple(raw["syllables"]),
            stress_index=raw["stress_index"],
            source_ids=tuple(raw["source_ids"]),
            note=raw["note"],
            is_exception=raw["is_exception"],
        )
        if entry.lookup_key in result:
            raise ValueError(
                f"duplicate stress key at line {line_number}: {entry.lookup_key}"
            )
        if not 0 <= entry.stress_index < len(entry.syllables):
            raise ValueError(f"invalid stress index at line {line_number}")
        unknown_sources = set(entry.source_ids) - known_sources
        if unknown_sources:
            raise ValueError(
                f"unknown stress sources at line {line_number}: {sorted(unknown_sources)}"
            )
        if not entry.source_ids or not entry.note.strip():
            raise ValueError(f"incomplete stress entry at line {line_number}")
        result[entry.lookup_key] = entry
    return result
```

`StressDecision` 字段固定为：

```python
@dataclass(frozen=True, slots=True)
class StressDecision:
    stress_index: int
    method: ResolutionMethod
    source_ids: tuple[str, ...]
    warnings: tuple[Diagnostic, ...] = ()
    applied_rule_ids: tuple[str, ...] = ()
```

- [ ] **Step 4: 实现重音解析顺序**

`resolve_stress()` 严格执行：请求覆盖、文本 acute 提示、本地例外、词典、双音节规则、重音节 penult 规则、候选 antepenult 加警告。重音节判定只接受可证明条件：penult 含双元音、带 macron，或音节以辅音结尾。普通开放 penult 未登记长度时不得假设。

核心决策函数使用：

```python
MACRON_VOWELS = frozenset("āēīōūȳ")
PLAIN_VOWELS = frozenset("aeiouy")
STRESS_SOURCE = ("allen-greenough-accents",)
ENCLITICS = ("que", "ne", "ve")


def _is_heavy(syllable: str) -> bool:
    folded = "".join(unicodedata.normalize("NFD", char)[0] for char in syllable)
    has_macron = any(char in MACRON_VOWELS for char in syllable)
    has_diphthong = any(pair in folded for pair in ("ae", "oe", "au", "eu", "ay"))
    closes_with_consonant = folded[-1] not in PLAIN_VOWELS
    return has_macron or has_diphthong or closes_with_consonant


def resolve_stress(
    lookup_key: str,
    syllables: tuple[str, ...],
    lexicon: Mapping[str, StressLexiconEntry],
    *,
    explicit_stress_index: int | None = None,
    override_index: int | None = None,
) -> StressDecision:
    def checked(index: int) -> int:
        if not 0 <= index < len(syllables):
            raise ValueError("stress_index must point to an existing syllable")
        return index

    if override_index is not None:
        return StressDecision(
            checked(override_index),
            ResolutionMethod.OVERRIDE,
            (),
            applied_rule_ids=("override-stress",),
        )
    if explicit_stress_index is not None:
        return StressDecision(
            checked(explicit_stress_index),
            ResolutionMethod.OVERRIDE,
            (),
            applied_rule_ids=("explicit-stress",),
        )
    entry = lexicon.get(lookup_key)
    if entry is not None:
        if len(entry.syllables) != len(syllables):
            raise ValueError(f"stress lexicon syllable mismatch: {lookup_key}")
        method = ResolutionMethod.EXCEPTION if entry.is_exception else ResolutionMethod.LEXICON
        rule_id = "stress-exception" if entry.is_exception else "stress-lexicon"
        return StressDecision(
            entry.stress_index,
            method,
            entry.source_ids,
            applied_rule_ids=(rule_id,),
        )
    enclitic_suffix = next(
        (
            suffix
            for suffix in ENCLITICS
            if lookup_key.endswith(suffix) and lookup_key[: -len(suffix)] in lexicon
        ),
        None,
    )
    if len(syllables) >= 2 and enclitic_suffix is not None:
        base_entry = lexicon[lookup_key[: -len(enclitic_suffix)]]
        return StressDecision(
            len(syllables) - 2,
            ResolutionMethod.RULE,
            tuple(dict.fromkeys((*STRESS_SOURCE, *base_entry.source_ids))),
            applied_rule_ids=("enclitic-stress",),
        )
    if len(syllables) == 1:
        return StressDecision(
            0,
            ResolutionMethod.RULE,
            STRESS_SOURCE,
            applied_rule_ids=("monosyllable-stress",),
        )
    if len(syllables) == 2:
        return StressDecision(
            0,
            ResolutionMethod.RULE,
            STRESS_SOURCE,
            applied_rule_ids=("disyllable-stress",),
        )
    penult = len(syllables) - 2
    if _is_heavy(syllables[penult]):
        return StressDecision(
            penult,
            ResolutionMethod.RULE,
            STRESS_SOURCE,
            applied_rule_ids=("heavy-penult-stress",),
        )
    warning = Diagnostic(
        code="PRONUNCIATION_NEEDS_REVIEW",
        message="Open penult quantity is not established by the stress lexicon",
        severity=Severity.WARNING,
    )
    return StressDecision(
        len(syllables) - 3,
        ResolutionMethod.CANDIDATE,
        STRESS_SOURCE,
        (warning,),
        ("candidate-antepenult-stress",),
    )
```

种子词典固定包含 `dominus`、`regina`、`maria`、`gratia`、`caelum`、`alleluia`、`magnificat`、`misericordia`、`benedictus`、`excelsis`。普通词的元音长度引用 `perseus-lewis-short` 的具体 lemma/entry locator；礼仪或希腊来源形式同时引用 `liber-usualis-1962` 中实际带重音的词形位置。若两者不能证明同一词形的重音，该行保持未批准且不得进入包资源。

- [ ] **Step 5: 运行测试、检查词典并提交**

```powershell
.venv\Scripts\python -m pytest tests/unit/test_stress.py -q
.venv\Scripts\python -m pytest tests/unit -q
.venv\Scripts\python -m ruff check src tests
git add src/latintts/stress.py src/latintts/resources/stress_lexicon.jsonl tests/unit/test_stress.py docs/pronunciation/roman-ecclesiastical.md
git commit -m "feat: add source-backed stress resolution"
```

---

### Task 6: 罗马教会式 G2P

**Files:**
- Create: `src/latintts/g2p.py`
- Create: `src/latintts/resources/g2p_exceptions.jsonl`
- Test: `tests/unit/test_g2p.py`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`

**Interfaces:**
- Consumes: 规范词、音节和 `stress_index`。
- Produces: `G2PExceptionEntry`、`load_g2p_exceptions()`、`G2PResult(ipa, phonemes, applied_rule_ids, source_ids, warnings)`、`ecclesiastical_g2p()`。

- [ ] **Step 1: 写来源规则失败测试**

```python
import pytest

from latintts.g2p import ecclesiastical_g2p


@pytest.mark.parametrize(
    ("word", "syllables", "stress", "ipa", "rule_id"),
    [
        ("caelum", ("cae", "lum"), 0, "ˈt͡ʃe.lum", "c-before-front-vowel"),
        ("ecce", ("ec", "ce"), 0, "ˈet.t͡ʃe", "cc-before-front-vowel"),
        ("descendit", ("de", "scen", "dit"), 1, "deˈʃen.dit", "sc-before-front-vowel"),
        ("regina", ("re", "gi", "na"), 1, "reˈd͡ʒi.na", "g-before-front-vowel"),
        ("regnum", ("re", "gnum"), 0, "ˈre.ɲum", "gn-palatal"),
        ("mihi", ("mi", "hi"), 0, "ˈmi.ki", "h-mihi-nihil"),
        ("gratia", ("gra", "ti", "a"), 0, "ˈɡra.t͡si.a", "ti-before-vowel"),
        ("excelsis", ("ex", "cel", "sis"), 1, "ekˈʃel.sis", "xc-before-front-vowel"),
        ("poëta", ("po", "ë", "ta"), 1, "poˈe.ta", "simple-e"),
    ],
)
def test_source_backed_g2p_rules(
    word: str,
    syllables: tuple[str, ...],
    stress: int,
    ipa: str,
    rule_id: str,
) -> None:
    result = ecclesiastical_g2p(word, syllables, stress)
    assert result.ipa == ipa
    assert rule_id in result.applied_rule_ids
    assert "liber-usualis-1962" in result.source_ids
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/unit/test_g2p.py -q`

Expected: FAIL because `latintts.g2p` is missing.

- [ ] **Step 3: 定义最长匹配规则表**

规则表按优先级从特殊到一般排列：`h` 例外、`xc`、`cc`、`sc`、`gn`、`ti + vowel`、`ch/ph/th`、`qu`、`ngu + vowel`、`ae/oe`、`au/eu/ay`、软 `c/g`、单字符映射。每条规则保存稳定 `rule_id` 和 `source_ids=("liber-usualis-1962",)`。

`ti + vowel` 只有在 `ti` 前一字符不为 `s/x/t` 时输出 `/t͡s/ + /i/`；`h` 只在 `mihi`、`nihil` 及经例外表确认的派生形式中输出 `/k/`，其他位置不发音。

模块导入、例外类型、结果类型和加载器使用：

```python
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from importlib.resources import files
from typing import Mapping

from latintts.domain import Diagnostic
from latintts.sources import load_source_registry


@dataclass(frozen=True, slots=True)
class G2PExceptionEntry:
    lookup_key: str
    phonemes_by_syllable: tuple[tuple[str, ...], ...]
    rule_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    note: str


@dataclass(frozen=True, slots=True)
class G2PResult:
    ipa: str
    phonemes: tuple[str, ...]
    applied_rule_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    warnings: tuple[Diagnostic, ...] = ()


def load_g2p_exceptions() -> dict[str, G2PExceptionEntry]:
    text = files("latintts.resources").joinpath("g2p_exceptions.jsonl").read_text(
        encoding="utf-8"
    )
    known_sources = set(load_source_registry())
    result: dict[str, G2PExceptionEntry] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        entry = G2PExceptionEntry(
            lookup_key=raw["lookup_key"],
            phonemes_by_syllable=tuple(tuple(part) for part in raw["phonemes_by_syllable"]),
            rule_ids=tuple(raw["rule_ids"]),
            source_ids=tuple(raw["source_ids"]),
            note=raw["note"],
        )
        if entry.lookup_key in result:
            raise ValueError(
                f"duplicate G2P exception key at line {line_number}: {entry.lookup_key}"
            )
        unknown_sources = set(entry.source_ids) - known_sources
        if unknown_sources:
            raise ValueError(
                f"unknown G2P exception sources at line {line_number}: "
                f"{sorted(unknown_sources)}"
            )
        if not entry.rule_ids or not entry.source_ids or not entry.note.strip():
            raise ValueError(f"incomplete G2P exception at line {line_number}")
        result[entry.lookup_key] = entry
    return result
```

规则类型和首批规则表使用：

```python
@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    pattern: re.Pattern[str]
    phonemes: tuple[str, ...]
    source_offsets: tuple[int, ...]
    source_ids: tuple[str, ...] = ("liber-usualis-1962",)


FRONT = r"(?:ae|oe|e|i|y)"
VOWEL = r"(?:ae|oe|[aeiouy])"
RULES = (
    Rule("xc-before-front-vowel", re.compile(rf"xc(?={FRONT})"), ("k", "ʃ"), (0, 1)),
    Rule("cc-before-front-vowel", re.compile(rf"cc(?={FRONT})"), ("t", "t͡ʃ"), (0, 1)),
    Rule("sc-before-front-vowel", re.compile(rf"sc(?={FRONT})"), ("ʃ",), (0,)),
    Rule("ti-before-vowel", re.compile(rf"(?<![sxt])ti(?={VOWEL})"), ("t͡s", "i"), (0, 1)),
    Rule("gn-palatal", re.compile(r"gn"), ("ɲ",), (0,)),
    Rule("ngu-before-vowel", re.compile(rf"ngu(?={VOWEL})"), ("ŋ", "ɡ", "w"), (0, 1, 2)),
    Rule("qu-before-vowel", re.compile(rf"qu(?={VOWEL})"), ("k", "w"), (0, 1)),
    Rule("ch-hard", re.compile(r"ch"), ("k",), (0,)),
    Rule("ph-f", re.compile(r"ph"), ("f",), (0,)),
    Rule("th-t", re.compile(r"th"), ("t",), (0,)),
    Rule("ae-e", re.compile(r"ae"), ("e",), (0,)),
    Rule("oe-e", re.compile(r"oe"), ("e",), (0,)),
    Rule("au-diphthong", re.compile(r"au"), ("a", "u̯"), (0, 1)),
    Rule("eu-diphthong", re.compile(r"eu"), ("e", "u̯"), (0, 1)),
    Rule("ay-diphthong", re.compile(r"ay"), ("a", "i̯"), (0, 1)),
    Rule("i-consonantal", re.compile(r"(?<![^aeiouy])i(?=[aeiouy])"), ("j",), (0,)),
    Rule("c-before-front-vowel", re.compile(rf"c(?={FRONT})"), ("t͡ʃ",), (0,)),
    Rule("g-before-front-vowel", re.compile(rf"g(?={FRONT})"), ("d͡ʒ",), (0,)),
)

DIAERESIS_BREAK_RULE_IDS = frozenset(
    {
        "ae-e", "oe-e", "au-diphthong", "eu-diphthong", "ay-diphthong",
        "i-consonantal", "qu-before-vowel", "ngu-before-vowel",
    }
)
```

- [ ] **Step 4: 实现来源索引和音节边界映射**

扫描器返回 `(source_index, phoneme)`，而不是只返回字符串。这样 `cc` 可以把第一个输出 `/t/` 归到前一音节、第二个 `/t͡ʃ/` 归到后一音节。IPA 格式化器根据 `syllable_ranges` 插入 `.`，并在目标音节前插入 `ˈ`。

核心扫描器必须使用 `match(word, index)`，不能对整词做无来源位置的连续替换：

```python
def _scan(word: str) -> tuple[tuple[tuple[int, str], ...], tuple[str, ...]]:
    folded = "".join(unicodedata.normalize("NFD", char)[0] for char in word)
    diaeresis_indices = {
        index
        for index, char in enumerate(word)
        if "\u0308" in unicodedata.normalize("NFD", char)
    }
    emitted: list[tuple[int, str]] = []
    applied: list[str] = []
    index = 0
    while index < len(folded):
        matched = False
        for rule in RULES:
            result = rule.pattern.match(folded, index)
            if result is None:
                continue
            if rule.rule_id in DIAERESIS_BREAK_RULE_IDS and any(
                position in diaeresis_indices for position in range(index, result.end())
            ):
                continue
            emitted.extend(
                (index + offset, phoneme)
                for offset, phoneme in zip(rule.source_offsets, rule.phonemes)
            )
            applied.append(rule.rule_id)
            index = result.end()
            matched = True
            break
        if matched:
            continue
        phonemes = SIMPLE_PHONEMES.get(folded[index])
        if phonemes is None:
            raise ValueError(f"unsupported grapheme at index {index}: {folded[index]!r}")
        emitted.extend((index, phoneme) for phoneme in phonemes)
        applied.append(SIMPLE_RULE_IDS[folded[index]])
        index += 1
    return tuple(emitted), tuple(applied)


def _render_ipa(
    syllables: tuple[str, ...],
    stress_index: int,
    emitted: tuple[tuple[int, str], ...],
) -> str:
    starts: list[int] = []
    cursor = 0
    for syllable in syllables:
        starts.append(cursor)
        cursor += len(syllable)
    parts: list[str] = []
    for syllable_index, start in enumerate(starts):
        end = starts[syllable_index + 1] if syllable_index + 1 < len(starts) else cursor
        body = "".join(phoneme for source_index, phoneme in emitted if start <= source_index < end)
        prefix = "ˈ" if syllable_index == stress_index else ""
        parts.append(prefix + body)
    return ".".join(parts)
```

固定基础映射：

```python
SIMPLE_PHONEMES: dict[str, tuple[str, ...]] = {
    "a": ("a",),
    "b": ("b",),
    "c": ("k",),
    "d": ("d",),
    "e": ("e",),
    "f": ("f",),
    "g": ("ɡ",),
    "h": (),
    "i": ("i",),
    "j": ("j",),
    "k": ("k",),
    "l": ("l",),
    "m": ("m",),
    "n": ("n",),
    "o": ("o",),
    "p": ("p",),
    "q": ("k",),
    "r": ("r",),
    "s": ("s",),
    "t": ("t",),
    "u": ("u",),
    "v": ("v",),
    "x": ("k", "s"),
    "y": ("i",),
    "z": ("d͡z",),
}

SIMPLE_RULE_IDS = {
    "a": "simple-a", "b": "simple-b", "c": "c-hard", "d": "simple-d",
    "e": "simple-e", "f": "simple-f", "g": "g-hard", "h": "h-muted",
    "i": "simple-i", "j": "j-consonantal", "k": "simple-k", "l": "simple-l",
    "m": "simple-m", "n": "simple-n", "o": "simple-o", "p": "simple-p",
    "q": "q-hard", "r": "simple-r", "s": "simple-s", "t": "simple-t",
    "u": "simple-u", "v": "simple-v", "x": "x-ks", "y": "y-as-i",
    "z": "z-dz",
}

IMPLEMENTED_RULE_IDS = frozenset(
    {rule.rule_id for rule in RULES}
    | set(SIMPLE_RULE_IDS.values())
    | {"h-mihi-nihil"}
)

PHONEME_INVENTORY = frozenset(
    {
        "a", "b", "d", "d͡ʒ", "d͡z", "e", "f", "ɡ", "i", "i̯",
        "j", "k", "l", "m", "n", "ɲ", "ŋ", "o", "p", "r",
        "s", "ʃ", "t", "t͡ʃ", "t͡s", "u", "u̯", "v", "w",
    }
)
```

`g2p_exceptions.jsonl` 首批内容固定为：

```jsonl
{"lookup_key":"mihi","phonemes_by_syllable":[["m","i"],["k","i"]],"rule_ids":["h-mihi-nihil"],"source_ids":["liber-usualis-1962"],"note":"Liber Usualis pronunciation table, h pronounced k in mihi"}
{"lookup_key":"nihil","phonemes_by_syllable":[["n","i"],["k","i","l"]],"rule_ids":["h-mihi-nihil"],"source_ids":["liber-usualis-1962"],"note":"Liber Usualis pronunciation table, h pronounced k in nihil"}
```

加载器拒绝重复键和未知来源。`model_phonemes` 在阶段 1 使用拆分后的规范 IPA token；Piper 专用映射不在本计划中实现。

顶层函数必须接受可注入的例外映射；未注入时从包资源加载。例外记录中的音节数必须与当前音节划分一致，避免例外 IPA 与 `PronunciationPlan.syllables` 分叉。普通规则和例外共用相同的重音、音节分隔 token 格式：

```python
def _phonemes_by_syllable(
    syllables: tuple[str, ...],
    emitted: tuple[tuple[int, str], ...],
) -> tuple[tuple[str, ...], ...]:
    starts: list[int] = []
    cursor = 0
    for syllable in syllables:
        starts.append(cursor)
        cursor += len(syllable)
    result: list[tuple[str, ...]] = []
    for syllable_index, start in enumerate(starts):
        end = starts[syllable_index + 1] if syllable_index + 1 < len(starts) else cursor
        result.append(
            tuple(phoneme for source_index, phoneme in emitted if start <= source_index < end)
        )
    return tuple(result)


def _render_model_phonemes(
    phonemes_by_syllable: tuple[tuple[str, ...], ...],
    stress_index: int,
) -> tuple[str, ...]:
    rendered: list[str] = []
    for index, syllable_phonemes in enumerate(phonemes_by_syllable):
        if index:
            rendered.append(".")
        if index == stress_index:
            rendered.append("ˈ")
        rendered.extend(syllable_phonemes)
    return tuple(rendered)


def ecclesiastical_g2p(
    word: str,
    syllables: tuple[str, ...],
    stress_index: int,
    *,
    lookup_key: str | None = None,
    exceptions: Mapping[str, G2PExceptionEntry] | None = None,
) -> G2PResult:
    if not 0 <= stress_index < len(syllables):
        raise ValueError("stress_index must point to an existing syllable")
    exception_key = lookup_key if lookup_key is not None else word
    exception = (exceptions if exceptions is not None else load_g2p_exceptions()).get(exception_key)
    if exception is not None:
        if len(exception.phonemes_by_syllable) != len(syllables):
            raise ValueError(f"G2P exception syllable mismatch: {word}")
        model_phonemes = _render_model_phonemes(
            exception.phonemes_by_syllable,
            stress_index,
        )
        return G2PResult(
            ipa="".join(model_phonemes),
            phonemes=model_phonemes,
            applied_rule_ids=exception.rule_ids,
            source_ids=exception.source_ids,
        )

    emitted, applied = _scan(word)
    phonemes_by_syllable = _phonemes_by_syllable(syllables, emitted)
    model_phonemes = _render_model_phonemes(phonemes_by_syllable, stress_index)
    return G2PResult(
        ipa=_render_ipa(syllables, stress_index, emitted),
        phonemes=model_phonemes,
        applied_rule_ids=tuple(dict.fromkeys(applied)),
        source_ids=("liber-usualis-1962",),
    )
```

- [ ] **Step 5: 运行测试、核对文档规则矩阵并提交**

```powershell
.venv\Scripts\python -m pytest tests/unit/test_g2p.py -q
.venv\Scripts\python -m pytest tests/unit -q
.venv\Scripts\python -m ruff check src tests
git add src/latintts/g2p.py src/latintts/resources/g2p_exceptions.jsonl tests/unit/test_g2p.py docs/pronunciation/roman-ecclesiastical.md
git commit -m "feat: add Roman ecclesiastical Latin g2p"
```

---

### Task 7: PronunciationPlan 编排与人工覆盖

**Files:**
- Create: `src/latintts/validation.py`
- Create: `src/latintts/pipeline.py`
- Test: `tests/unit/test_validation.py`
- Test: `tests/unit/test_pipeline.py`
- Modify: `src/latintts/__init__.py`

**Interfaces:**
- Consumes: `PHONEME_INVENTORY`、`normalize_word()`、`tokenize_words()`、`syllabify()`、`resolve_stress()`、`load_g2p_exceptions()`、`ecclesiastical_g2p()`。
- Produces: `validate_override()`、`Pronouncer(rule_version, stress_lexicon, g2p_exceptions)`、`Pronouncer.analyze(text, overrides=None) -> PronunciationPlan`。

- [ ] **Step 1: 写非法覆盖失败测试**

```python
import pytest

from latintts.domain import PronunciationOverride
from latintts.validation import validate_override


def test_override_rejects_unknown_model_phoneme() -> None:
    override = PronunciationOverride(
        ipa="ˈaunknown",
        model_phonemes=("ˈ", "a", "unknown"),
    )
    with pytest.raises(ValueError, match="unknown model phonemes"):
        validate_override(override, syllable_count=1)


def test_override_rejects_ipa_without_matching_model_tokens() -> None:
    override = PronunciationOverride(ipa="ˈa.ve")
    with pytest.raises(ValueError, match="ipa and model_phonemes"):
        validate_override(override, syllable_count=2)


def test_override_accepts_stress_only() -> None:
    validate_override(PronunciationOverride(stress_index=1), syllable_count=3)


def test_override_rejects_stress_marker_on_different_syllable() -> None:
    override = PronunciationOverride(
        ipa="ˈdo.mi.nus",
        model_phonemes=("ˈ", "d", "o", ".", "m", "i", ".", "n", "u", "s"),
    )
    with pytest.raises(ValueError, match="stress token"):
        validate_override(override, syllable_count=3, resolved_stress_index=1)
```

- [ ] **Step 2: 运行验证测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/unit/test_validation.py -q`

Expected: FAIL because `latintts.validation` is missing.

- [ ] **Step 3: 实现覆盖验证器**

```python
from latintts.domain import PronunciationOverride
from latintts.g2p import PHONEME_INVENTORY

CONTROL_TOKENS = frozenset({"ˈ", "."})


def validate_override(
    override: PronunciationOverride,
    syllable_count: int,
    resolved_stress_index: int | None = None,
) -> None:
    if syllable_count < 1:
        raise ValueError("syllable_count must be positive")
    if (
        override.stress_index is None
        and override.ipa is None
        and override.model_phonemes is None
    ):
        raise ValueError("pronunciation override must change at least one field")
    if override.stress_index is not None and not 0 <= override.stress_index < syllable_count:
        raise ValueError("override stress_index must point to an existing syllable")
    if (override.ipa is None) != (override.model_phonemes is None):
        raise ValueError("ipa and model_phonemes must be supplied together")
    if override.model_phonemes is None:
        return
    unknown = set(override.model_phonemes) - PHONEME_INVENTORY - CONTROL_TOKENS
    if unknown:
        raise ValueError(f"unknown model phonemes: {sorted(unknown)}")
    if override.model_phonemes.count("ˈ") != 1:
        raise ValueError("model_phonemes must contain exactly one primary stress token")
    if override.model_phonemes.count(".") != syllable_count - 1:
        raise ValueError("model_phonemes syllable separators do not match syllables")
    if override.ipa != "".join(override.model_phonemes):
        raise ValueError("ipa and model_phonemes must describe the same token sequence")
    stress_token_index = 0
    for token in override.model_phonemes:
        if token == "ˈ":
            break
        if token == ".":
            stress_token_index += 1
    if resolved_stress_index is not None and stress_token_index != resolved_stress_index:
        raise ValueError("model_phonemes stress token does not match resolved stress_index")
```

- [ ] **Step 4: 写端到端失败测试**

```python
import pytest

from latintts.domain import PronunciationOverride, ResolutionMethod
from latintts.pipeline import Pronouncer


def test_analyze_preserves_source_spans_and_builds_plan() -> None:
    plan = Pronouncer.default().analyze("Ave, cælum!")
    assert plan.original_text == "Ave, cælum!"
    assert plan.normalized_text == "Ave, cælum!"
    assert [token.normalized for token in plan.tokens] == ["ave", "caelum"]
    assert [token.source_span for token in plan.tokens] == [(0, 3), (5, 10)]
    assert "expand-ae-ligature" in plan.tokens[1].normalization_transforms
    assert plan.tokens[1].ipa == "ˈt͡ʃe.lum"


def test_request_override_is_used_and_recorded() -> None:
    override = PronunciationOverride(stress_index=1)
    plan = Pronouncer.default().analyze("Dominus", overrides={0: override})
    assert plan.tokens[0].stress_index == 1
    assert plan.tokens[0].resolution_method is ResolutionMethod.OVERRIDE
    assert plan.tokens[0].override == override


def test_unknown_stress_warning_reaches_plan() -> None:
    plan = Pronouncer.default().analyze("Fabula")
    assert any(item.code == "PRONUNCIATION_NEEDS_REVIEW" for item in plan.warnings)


def test_g2p_exception_marks_token_resolution_method() -> None:
    token = Pronouncer.default().analyze("mihi").tokens[0]
    assert token.resolution_method is ResolutionMethod.EXCEPTION


def test_invalid_override_error_identifies_token() -> None:
    override = PronunciationOverride(
        ipa="ˈaunknown",
        model_phonemes=("ˈ", "a", "unknown"),
    )
    with pytest.raises(ValueError, match="token 0 'Ave'.*unknown model phonemes"):
        Pronouncer.default().analyze("Ave", overrides={0: override})
```

- [ ] **Step 5: 运行管线测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/unit/test_pipeline.py -q`

Expected: FAIL because the pipeline module is missing.

- [ ] **Step 6: 实现 `Pronouncer`**

`Pronouncer.default()` 从包资源加载重音词典和例外表。`analyze()` 依次：规范短语、扫描原文单词、词级规范化、音节划分、把 acute 字符位置映射到音节、解析重音、执行 G2P、创建 `PronunciationToken`、聚合诊断。词间在 `phrase_phonemes` 中插入单个 `"|"`。

覆盖映射以零基 token index 为键。非法 token index 或越界 `stress_index` 在合成计划前抛出 `ValueError`。覆盖只写入当前 `PronunciationPlan`，不修改词典文件。

核心编排代码使用：

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from latintts.domain import (
    Diagnostic,
    PronunciationOverride,
    PronunciationPlan,
    PronunciationToken,
    ResolutionMethod,
)
from latintts.g2p import (
    G2PExceptionEntry,
    ecclesiastical_g2p,
    load_g2p_exceptions,
)
from latintts.normalization import (
    NormalizedWord,
    normalize_phrase,
    normalize_word,
    tokenize_words,
)
from latintts.stress import StressLexiconEntry, load_stress_lexicon, resolve_stress
from latintts.syllables import syllabify, syllable_ranges
from latintts.validation import validate_override


def _marked_syllable_index(
    word: NormalizedWord,
    ranges: tuple[tuple[int, int], ...],
) -> int | None:
    if word.marked_vowel_index is None:
        return None
    for syllable_index, (start, end) in enumerate(ranges):
        if start <= word.marked_vowel_index < end:
            return syllable_index
    raise ValueError("marked stress position does not belong to a syllable")


@dataclass(frozen=True, slots=True)
class Pronouncer:
    rule_version: str
    stress_lexicon: Mapping[str, StressLexiconEntry]
    g2p_exceptions: Mapping[str, G2PExceptionEntry]

    @classmethod
    def default(cls) -> "Pronouncer":
        return cls(
            rule_version="ecclesiastical-roman-v1",
            stress_lexicon=load_stress_lexicon(),
            g2p_exceptions=load_g2p_exceptions(),
        )

    def analyze(
        self,
        text: str,
        overrides: Mapping[int, PronunciationOverride] | None = None,
    ) -> PronunciationPlan:
        override_map = dict(overrides or {})
        spans = tokenize_words(text)
        invalid_indexes = set(override_map) - set(range(len(spans)))
        if invalid_indexes:
            raise ValueError(f"override token indexes do not exist: {sorted(invalid_indexes)}")

        tokens: list[PronunciationToken] = []
        plan_warnings: list[Diagnostic] = []
        phrase_phonemes: list[str] = []
        for token_index, span in enumerate(spans):
            word = normalize_word(span.surface)
            syllables = syllabify(word.normalized)
            explicit_stress = _marked_syllable_index(word, syllable_ranges(word.normalized))
            override = override_map.get(token_index)
            stress = resolve_stress(
                word.lookup_key,
                syllables,
                self.stress_lexicon,
                explicit_stress_index=explicit_stress,
                override_index=override.stress_index if override else None,
            )
            if override is not None:
                try:
                    validate_override(override, len(syllables), stress.stress_index)
                except ValueError as error:
                    raise ValueError(
                        f"invalid override for token {token_index} {span.surface!r}: {error}"
                    ) from error
            g2p = ecclesiastical_g2p(
                word.normalized,
                syllables,
                stress.stress_index,
                lookup_key=word.lookup_key,
                exceptions=self.g2p_exceptions,
            )
            warnings = tuple(
                replace(item, source_span=(span.start, span.end), token_index=token_index)
                for item in (*stress.warnings, *g2p.warnings)
            )
            if override is not None:
                method = ResolutionMethod.OVERRIDE
            elif word.lookup_key in self.g2p_exceptions:
                method = ResolutionMethod.EXCEPTION
            else:
                method = stress.method
            ipa = override.ipa if override and override.ipa is not None else g2p.ipa
            phonemes = (
                override.model_phonemes
                if override and override.model_phonemes is not None
                else g2p.phonemes
            )
            token = PronunciationToken(
                surface=span.surface,
                normalized=word.normalized,
                source_span=(span.start, span.end),
                syllables=syllables,
                stress_index=stress.stress_index,
                ipa=ipa,
                model_phonemes=phonemes,
                resolution_method=method,
                applied_rule_ids=tuple(
                    dict.fromkeys((*stress.applied_rule_ids, *g2p.applied_rule_ids))
                ),
                normalization_transforms=word.transformations,
                source_ids=tuple(dict.fromkeys((*stress.source_ids, *g2p.source_ids))),
                warnings=warnings,
                override=override,
            )
            if phrase_phonemes:
                phrase_phonemes.append("|")
            phrase_phonemes.extend(token.model_phonemes)
            tokens.append(token)
            plan_warnings.extend(warnings)

        return PronunciationPlan(
            schema_version="1",
            rule_version=self.rule_version,
            original_text=text,
            normalized_text=normalize_phrase(text),
            tokens=tuple(tokens),
            phrase_phonemes=tuple(phrase_phonemes),
            warnings=tuple(plan_warnings),
        )
```

- [ ] **Step 7: 导出稳定公共 API**

`src/latintts/__init__.py` 只导出：

```python
from latintts.domain import PronunciationOverride, PronunciationPlan, PronunciationToken
from latintts.pipeline import Pronouncer

__all__ = [
    "Pronouncer",
    "PronunciationOverride",
    "PronunciationPlan",
    "PronunciationToken",
]
__version__ = "0.1.0"
```

- [ ] **Step 8: 运行单元测试、类型检查并提交**

```powershell
.venv\Scripts\python -m pytest tests/unit/test_pipeline.py -q
.venv\Scripts\python -m pytest tests/unit/test_validation.py -q
.venv\Scripts\python -m pytest tests/unit -q
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\python -m mypy src
git add src/latintts/validation.py src/latintts/pipeline.py src/latintts/__init__.py tests/unit/test_validation.py tests/unit/test_pipeline.py
git commit -m "feat: compose pronunciation analysis pipeline"
```

---

### Task 8: 黄金词表格式与审计工具

**Files:**
- Create: `src/latintts/audit.py`
- Create: `tests/fixtures/gold_pronunciations.jsonl`
- Test: `tests/unit/test_audit.py`
- Test: `tests/integration/test_gold_pronunciations.py`

**Interfaces:**
- Consumes: `Pronouncer`、来源登记、黄金 JSONL。
- Produces: `GoldEntry`、`AuditPolicy`、`AuditReport`、`audit_gold_file()`、`python -m latintts.audit <path>`。

- [ ] **Step 1: 写审计失败测试**

```python
from pathlib import Path

from latintts.audit import AuditPolicy, audit_gold_file


def test_gold_fixture_has_unique_approved_source_backed_entries() -> None:
    report = audit_gold_file(
        Path("tests/fixtures/gold_pronunciations.jsonl"),
        AuditPolicy(minimum_total=30, category_minimums={}),
    )
    assert report.errors == ()
    assert report.total >= 30


def test_audit_rejects_duplicate_word_and_unknown_source(tmp_path: Path) -> None:
    fixture = tmp_path / "bad.jsonl"
    row = (
        '{"word":"ave","normalized":"ave","syllables":["a","ve"],'
        '"stress_index":0,"ipa":"ˈa.ve","category":"stress",'
        '"rule_ids":["disyllable-stress"],"source_ids":["missing"],'
        '"review_state":"approved"}\n'
    )
    fixture.write_text(row + row, encoding="utf-8")
    report = audit_gold_file(fixture, AuditPolicy(minimum_total=1, category_minimums={}))
    assert {error.code for error in report.errors} == {"DUPLICATE_WORD", "UNKNOWN_SOURCE"}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/unit/test_audit.py -q`

Expected: FAIL because audit code and fixture do not exist.

- [ ] **Step 3: 实现严格 JSONL 审计**

每条 `GoldEntry` 固定字段：`word`、`normalized`、`syllables`、`stress_index`、`ipa`、`category`、`rule_ids`、`source_ids`、`review_state`。审计拒绝空字段、重复 `word`、非法重音、未知来源、非 `approved` 状态、低于总数门槛和低于分类门槛。

核心类型和审计循环使用：

```python
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from latintts.sources import load_source_registry


@dataclass(frozen=True, slots=True)
class GoldEntry:
    word: str
    normalized: str
    syllables: tuple[str, ...]
    stress_index: int
    ipa: str
    category: str
    rule_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    review_state: str


@dataclass(frozen=True, slots=True)
class AuditPolicy:
    minimum_total: int
    category_minimums: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class AuditError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AuditReport:
    total: int
    category_counts: Mapping[str, int]
    errors: tuple[AuditError, ...]


DEFAULT_POLICY = AuditPolicy(minimum_total=30, category_minimums={})


def audit_gold_file(path: Path, policy: AuditPolicy = DEFAULT_POLICY) -> AuditReport:
    known_sources = set(load_source_registry())
    seen_words: set[str] = set()
    counts: Counter[str] = Counter()
    errors: list[AuditError] = []
    total = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        total += 1
        try:
            raw = json.loads(line)
            entry = GoldEntry(
                word=raw["word"],
                normalized=raw["normalized"],
                syllables=tuple(raw["syllables"]),
                stress_index=raw["stress_index"],
                ipa=raw["ipa"],
                category=raw["category"],
                rule_ids=tuple(raw["rule_ids"]),
                source_ids=tuple(raw["source_ids"]),
                review_state=raw["review_state"],
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            errors.append(AuditError("INVALID_ROW", f"line {line_number}: {error}"))
            continue
        if entry.word in seen_words:
            errors.append(AuditError("DUPLICATE_WORD", entry.word))
        seen_words.add(entry.word)
        required_text = (entry.word, entry.normalized, entry.ipa, entry.category)
        if (
            any(not value.strip() for value in required_text)
            or not entry.syllables
            or any(not value.strip() for value in entry.syllables)
        ):
            errors.append(AuditError("EMPTY_FIELD", f"line {line_number}"))
        if not 0 <= entry.stress_index < len(entry.syllables):
            errors.append(AuditError("INVALID_STRESS", entry.word))
        if entry.review_state != "approved":
            errors.append(AuditError("UNAPPROVED_ENTRY", entry.word))
        for source_id in set(entry.source_ids) - known_sources:
            errors.append(AuditError("UNKNOWN_SOURCE", f"{entry.word}: {source_id}"))
        if not entry.rule_ids or not entry.source_ids:
            errors.append(AuditError("MISSING_PROVENANCE", entry.word))
        counts[entry.category] += 1
    if total < policy.minimum_total:
        errors.append(AuditError("TOTAL_MINIMUM_NOT_MET", str(total)))
    for category, minimum in policy.category_minimums.items():
        if counts[category] < minimum:
            errors.append(
                AuditError(
                    "CATEGORY_MINIMUM_NOT_MET",
                    f"{category}: {counts[category]} < {minimum}",
                )
            )
    return AuditReport(total, dict(counts), tuple(errors))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the LatinTTS pronunciation gold set")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    report = audit_gold_file(args.path)
    if report.errors:
        for error in report.errors:
            print(f"{error.code}: {error.message}")
        return 1
    print(f"gold-audit: PASS total={report.total} errors=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

CLI 在成功时输出一行：

```text
gold-audit: PASS total=30 errors=0
```

失败时每个错误输出 `code: message` 并以 exit code 1 退出。

- [ ] **Step 4: 添加 30 条冒烟黄金记录**

首批词和唯一分类固定如下，避免执行者临时选择只覆盖容易规则的样本：

| category | words |
|---|---|
| `vowels` | `pater`, `dominus`, `gloria`, `nobis`, `kyrie` |
| `diphthongs` | `caelum`, `coeli`, `lauda`, `euge`, `cui` |
| `consonants` | `ecce`, `descendit`, `regina`, `regnum`, `mihi`, `gratia`, `excelsis`, `chorus`, `christus`, `qui` |
| `stress` | `ave`, `maria`, `misericordia`, `benedictus`, `alleluia` |
| `liturgical` | `credo`, `hosanna`, `agnus`, `deo`, `magnificat` |

每条记录引用实际支持其 `rule_ids` 的来源。`rule_ids` 只能列入 `PronunciationToken.applied_rule_ids` 实际返回的 ID；所有 IPA 和重音必须由规则文档及已登记来源复核，不从现有模型输出反推。第一行以以下精确 JSONL 结构为基准，其余 29 行使用相同键顺序：

```jsonl
{"word":"ave","normalized":"ave","syllables":["a","ve"],"stress_index":0,"ipa":"ˈa.ve","category":"stress","rule_ids":["disyllable-stress","simple-a","simple-v","simple-e"],"source_ids":["allen-greenough-accents","liber-usualis-1962"],"review_state":"approved"}
```

- [ ] **Step 5: 添加黄金端到端测试**

```python
import json
from pathlib import Path

import pytest

from latintts.pipeline import Pronouncer


ROWS = [
    json.loads(line)
    for line in Path("tests/fixtures/gold_pronunciations.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["word"])
def test_gold_pronunciation(row: dict[str, object]) -> None:
    plan = Pronouncer.default().analyze(str(row["word"]))
    token = plan.tokens[0]
    assert token.normalized == row["normalized"]
    assert list(token.syllables) == row["syllables"]
    assert token.stress_index == row["stress_index"]
    assert token.ipa == row["ipa"]
    assert set(row["rule_ids"]) <= set(token.applied_rule_ids)
```

- [ ] **Step 6: 运行审计、集成测试并提交**

```powershell
.venv\Scripts\python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
.venv\Scripts\python -m pytest tests/unit/test_audit.py tests/integration/test_gold_pronunciations.py -q
git add src/latintts/audit.py tests/unit/test_audit.py tests/integration/test_gold_pronunciations.py tests/fixtures/gold_pronunciations.jsonl
git commit -m "test: add source-backed pronunciation gold set"
```

---

### Task 9: 元音与双元音黄金数据扩展

**Files:**
- Modify: `tests/fixtures/gold_pronunciations.jsonl`
- Modify: `tests/unit/test_audit.py`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`

**Interfaces:**
- Consumes: `AuditPolicy.category_minimums`。
- Produces: `vowels >= 20`、`diphthongs >= 20`，黄金总数至少 40。

- [ ] **Step 1: 把审计阈值提高到两个分类各 20 条**

将测试策略设置为：

```python
AuditPolicy(
    minimum_total=40,
    category_minimums={"vowels": 20, "diphthongs": 20},
)
```

- [ ] **Step 2: 运行审计测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/unit/test_audit.py -q`

Expected: FAIL with `CATEGORY_MINIMUM_NOT_MET` for one or both categories.

- [ ] **Step 3: 分四批各审核 10 条记录**

四批分别覆盖：短/长书写元音与 `y`；相邻元音分别成音节；`ae/oe`；`au/eu/ay` 及 `qui/sanguis/cui` 边界。每批提交前在规则文档覆盖矩阵中登记词、规则 ID、来源 locator 和评审者状态。

- [ ] **Step 4: 运行精确匹配和审计**

```powershell
.venv\Scripts\python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
.venv\Scripts\python -m pytest tests/integration/test_gold_pronunciations.py -q
```

Expected: 总数至少 40；两个分类均达到 20；所有精确匹配通过。

- [ ] **Step 5: 提交数据批次**

```powershell
git add tests/fixtures/gold_pronunciations.jsonl tests/unit/test_audit.py docs/pronunciation/roman-ecclesiastical.md
git commit -m "test: cover ecclesiastical Latin vowels"
```

---

### Task 10: 辅音规则黄金数据扩展

**Files:**
- Modify: `tests/fixtures/gold_pronunciations.jsonl`
- Modify: `tests/unit/test_audit.py`
- Modify: `tests/integration/test_gold_pronunciations.py`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`

**Interfaces:**
- Produces: `consonants >= 100`，累计黄金总数至少 140。

- [ ] **Step 1: 增加辅音分类门槛并确认失败**

设置 `minimum_total=140` 和 `category_minimums["consonants"]=100`，运行审计测试。

Expected: FAIL with total and consonant category不足。

- [ ] **Step 2: 审核 20 条 `c/cc/sc/ch` 数据**

覆盖前元音与非前元音、词首/词中、双辅音跨音节、`ch` 在 `e/i` 前仍为 `/k/`。每个 rule ID 至少两个正例和一个反例。

- [ ] **Step 3: 审核 20 条 `g/gn/h/j` 数据**

覆盖软硬 `g`、`gn`、`mihi/nihil`、普通静音 `h`、显式 `j` 和写作 `i` 的辅音值。

- [ ] **Step 4: 审核 20 条 `ti/th/x/xc/z` 数据**

覆盖 `ti + vowel` 正例、前接 `s/x/t` 的反例、`th`、普通 `x`、前元音前 `xc`、`z`。

- [ ] **Step 5: 审核 20 条 `qu/ngu/r/s` 数据**

覆盖 `qu`、`sanguis` 类、卷舌 `r` 的音位保持、词首/词中/元音间 `s`；按工程政策保持音位 `/s/`。

- [ ] **Step 6: 审核 20 条双辅音和基础辅音数据**

覆盖 `bb/cc/dd/ff/gg/ll/mm/nn/pp/rr/ss/tt` 中至少十类，以及 `b/d/f/k/l/m/n/p/v` 基础映射。标准普通拉丁正字法中的 `q` 由 `qu-before-vowel` 真实词例覆盖；`q-hard` 仅为非标准或残缺输入的内部退化路径，免除真实词黄金集正例要求，不计入至少 320 条 approved 黄金词。该决策由用户于 2026-07-18 选择方案 A 批准；Task 14 必须用明确标为 synthetic fallback 的单元测试覆盖 `q-hard`。

- [ ] **Step 7: 运行审计、精确匹配并提交**

```powershell
.venv\Scripts\python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
.venv\Scripts\python -m pytest tests/integration/test_gold_pronunciations.py -q
git add tests/fixtures/gold_pronunciations.jsonl tests/unit/test_audit.py tests/integration/test_gold_pronunciations.py docs/pronunciation/roman-ecclesiastical.md
git commit -m "test: cover ecclesiastical Latin consonants"
```

Expected: 总数至少 140，`consonants >= 100`，所有记录通过精确匹配。

---

### Task 11: 音节与重音黄金数据扩展

**Files:**
- Modify: `tests/fixtures/gold_pronunciations.jsonl`
- Modify: `tests/unit/test_audit.py`
- Modify: `src/latintts/audit.py`
- Modify: `src/latintts/resources/stress_lexicon.jsonl`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`

**Interfaces:**
- Produces: `syllabification >= 40`、`stress >= 60`，累计总数至少 240。

- [ ] **Step 1: 增加音节与重音门槛并确认失败**

设置 `minimum_total=240`、`syllabification=40`、`stress=60`，运行审计测试。

Expected: FAIL with the new category minima.

- [ ] **Step 2: 审核四批各 10 条音节数据**

四批依次覆盖：单辅音跨音节；允许词首辅音簇；双辅音；三辅音与滑音边界。每条记录同时验证 `syllables` 和 IPA 中的 `.`。

- [ ] **Step 3: 审核两批各 10 条单/双音节重音数据**

确认单音节 index 0、双音节首音节重读；避免把短语重音混入词重音。

- [ ] **Step 4: 审核两批各 10 条重 penult 数据**

覆盖双元音、macron 和闭音节三种可证明重 penult 情况。

- [ ] **Step 5: 审核两批各 10 条轻 penult 与词典数据**

开放 penult 的历史长短必须来自词典或印有重音的权威文本；每个新增词典行包含 `source_ids` 和 locator note。

- [ ] **Step 6: 运行词典加载、审计和端到端测试**

```powershell
.venv\Scripts\python -m pytest tests/unit/test_stress.py -q
.venv\Scripts\python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
.venv\Scripts\python -m pytest tests/integration/test_gold_pronunciations.py -q
```

Expected: 总数至少 240；音节和重音分类达到门槛；无未知来源。

- [ ] **Step 7: 提交**

```powershell
git add tests/fixtures/gold_pronunciations.jsonl tests/unit/test_audit.py src/latintts/resources/stress_lexicon.jsonl docs/pronunciation/roman-ecclesiastical.md
git commit -m "test: cover Latin syllables and stress"
```

---

### Task 12: 拼写变体与例外黄金数据扩展

**Files:**
- Modify: `tests/fixtures/gold_pronunciations.jsonl`
- Modify: `tests/unit/test_audit.py`
- Modify: `src/latintts/resources/g2p_exceptions.jsonl`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`

**Interfaces:**
- Produces: `orthographic_variants >= 30`，累计总数至少 270。

- [ ] **Step 1: 增加变体门槛并确认失败**

设置 `minimum_total=270` 和 `orthographic_variants=30`，运行审计测试。

Expected: FAIL with `CATEGORY_MINIMUM_NOT_MET`。

- [ ] **Step 2: 审核 10 条 `æ/ae` 与 `œ/oe` 成对记录**

每对必须得到相同 `lookup_key`、相同音节、相同重音和相同 IPA，同时保留不同原始 `word`。

- [ ] **Step 3: 审核 10 条 `j/i` 成对或上下文记录**

覆盖显式 `j`、词首辅音 `i`、元音间辅音 `i` 和保持元音的 `i`；禁止用全局替换生成预期 IPA。

- [ ] **Step 4: 审核 10 条 `u/v` 与例外记录**

覆盖词典检索键统一、实际辅音/元音角色保持、`mihi/nihil` 及至少两个经来源确认的派生或反例。

- [ ] **Step 5: 运行规范化、例外和黄金测试**

```powershell
.venv\Scripts\python -m pytest tests/unit/test_normalization.py tests/unit/test_g2p.py -q
.venv\Scripts\python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
.venv\Scripts\python -m pytest tests/integration/test_gold_pronunciations.py -q
```

- [ ] **Step 6: 提交**

```powershell
git add tests/fixtures/gold_pronunciations.jsonl tests/unit/test_audit.py src/latintts/resources/g2p_exceptions.jsonl docs/pronunciation/roman-ecclesiastical.md
git commit -m "test: cover Latin orthographic variants"
```

---

### Task 13: 常见礼仪词汇与最终 320 词门禁

**Files:**
- Modify: `tests/fixtures/gold_pronunciations.jsonl`
- Modify: `tests/unit/test_audit.py`
- Modify: `src/latintts/resources/stress_lexicon.jsonl`
- Modify: `src/latintts/resources/g2p_exceptions.jsonl`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`

**Interfaces:**
- Produces: `liturgical >= 50`、`minimum_total=320`，并保留此前全部分类门槛。

- [ ] **Step 1: 设置最终门禁并确认失败**

最终策略必须是：

```python
AuditPolicy(
    minimum_total=320,
    category_minimums={
        "vowels": 20,
        "diphthongs": 20,
        "consonants": 100,
        "syllabification": 40,
        "stress": 60,
        "orthographic_variants": 30,
        "liturgical": 50,
    },
)
```

同一个策略同时替换 `src/latintts/audit.py` 中的 `DEFAULT_POLICY`，保证 CLI 与测试使用相同最终门槛。

Run: `.venv\Scripts\python -m pytest tests/unit/test_audit.py -q`

Expected: FAIL because `liturgical` and total thresholds are not met.

- [ ] **Step 2: 审核 10 条弥撒常用词**

从已登记权威文本选择，覆盖 `Kyrie`、`Gloria`、`Credo`、`Sanctus`、`Agnus Dei` 相关高频词；每条保留文本 locator。

- [ ] **Step 3: 审核 10 条圣母经词汇**

覆盖 `Ave Maria` 相关词汇，并确保不把整句韵律写入单词预期。

- [ ] **Step 4: 审核 10 条天主经词汇**

覆盖 `Pater Noster` 相关词汇，优先选择能补足重音和辅音边界的词。

- [ ] **Step 5: 审核 10 条圣咏与经文高频词**

使用明确版本的拉丁文本；保存版本和 verse locator，不使用歌唱音高作为发音证据。

- [ ] **Step 6: 审核 10 条礼仪回应与祷文词汇**

覆盖常见回应和祷文，并优先补齐尚未达到两个正例一个反例的规则 ID。

- [ ] **Step 7: 运行最终数据门禁并提交**

```powershell
.venv\Scripts\python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
.venv\Scripts\python -m pytest tests/unit/test_audit.py tests/integration/test_gold_pronunciations.py -q
git add tests/fixtures/gold_pronunciations.jsonl tests/unit/test_audit.py src/latintts/audit.py src/latintts/resources/stress_lexicon.jsonl src/latintts/resources/g2p_exceptions.jsonl docs/pronunciation/roman-ecclesiastical.md
git commit -m "test: lock 320-word pronunciation gold set"
```

Expected: `gold-audit: PASS total=320 errors=0` 或更高总数；所有分类门槛通过。

---

### Task 14: 规则覆盖、README 与阶段 1 发布门禁

**Files:**
- Create: `tests/integration/test_rule_document_coverage.py`
- Modify: `tests/unit/test_g2p.py`
- Modify: `README.md`
- Modify: `docs/pronunciation/roman-ecclesiastical.md`
- Modify: `src/latintts/audit.py`

**Interfaces:**
- Consumes: G2P rule IDs、规则文档覆盖矩阵、黄金词条 `rule_ids`。
- Produces: 阶段 1 的完整验证命令和稳定 Python 使用示例。

- [ ] **Step 1: 写规则覆盖失败测试**

```python
import json
from pathlib import Path

from latintts.g2p import IMPLEMENTED_RULE_IDS

REAL_WORD_GOLD_EXEMPT_RULE_IDS = {"q-hard"}


def test_every_g2p_rule_has_gold_coverage() -> None:
    covered = set()
    for line in Path("tests/fixtures/gold_pronunciations.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            covered.update(json.loads(line)["rule_ids"])
    missing = IMPLEMENTED_RULE_IDS - covered
    assert missing == REAL_WORD_GOLD_EXEMPT_RULE_IDS


def test_rule_document_lists_every_implemented_rule() -> None:
    document = Path("docs/pronunciation/roman-ecclesiastical.md").read_text(
        encoding="utf-8"
    )
    missing = {rule_id for rule_id in IMPLEMENTED_RULE_IDS if f"`{rule_id}`" not in document}
    assert missing == set()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/integration/test_rule_document_coverage.py -q`

Expected: FAIL and list concrete missing rule IDs or documentation rows.

- [ ] **Step 3: 补齐规则覆盖矩阵**

对失败列表逐项添加黄金记录或文档矩阵行。每行包含：rule ID、拼写条件、IPA 输出、正例、反例、source ID、locator、对应黄金词。不得通过从 `RULES` 自动生成文档来掩盖缺少人工解释。

同时在 `tests/unit/test_g2p.py` 增加明确命名为 synthetic fallback 的裸 `q` 测试：输入 `q`，断言 IPA 为 `/k/`、rule ID 为 `q-hard`、source ID 为 `liber-usualis-1962`。该测试只验证非标准或残缺输入的内部容错实现，不得计入真实拉丁词黄金数据或 320 词门槛。

- [ ] **Step 4: 更新 README**

README 必须包含：项目范围、Python 3.10 环境创建、开发依赖安装、完整验证命令、来源文档链接、阶段 1 非目标，以及以下可运行示例：

```python
from latintts import Pronouncer

plan = Pronouncer.default().analyze("Ave Maria")
for token in plan.tokens:
    print(token.surface, token.syllables, token.stress_index, token.ipa)
```

- [ ] **Step 5: 运行完整验证**

```powershell
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m mypy src
.venv\Scripts\python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q
.venv\Scripts\python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
git diff --check
```

Expected:

- Ruff format/check exit 0。
- mypy exit 0。
- pytest 全部通过，覆盖率至少 95%。
- 黄金审计总数至少 320，错误数 0。
- `git diff --check` 无输出。

- [ ] **Step 6: 提交阶段 1 文档与发布门禁**

```powershell
git add README.md docs/pronunciation/roman-ecclesiastical.md src/latintts/audit.py tests/integration/test_rule_document_coverage.py tests/unit/test_g2p.py
git commit -m "docs: complete pronunciation foundation guide"
```

- [ ] **Step 7: 复核提交范围**

Run:

```powershell
git status --short
git log --oneline --decorate -14
```

Expected: 工作区没有阶段 1 文件的未提交变更；日志显示每个任务独立提交；`.superpowers/` 不在 Git 状态中。

---

## Spec Coverage Matrix

| 系统设计阶段 1 要求 | 实施任务 | 发布证据 |
|---|---|---|
| 来源登记、许可/条款、稳定 locator | Task 2 | `pronunciation_sources.json` 与 `test_sources.py` |
| 完整罗马教会式规则文档、正反例和冲突决策 | Task 2–6、Task 14 | `roman-ecclesiastical.md` 及规则文档覆盖测试 |
| Unicode、原文跨度、`æ/ae`、`œ/oe`、`j/i`、`u/v` 可追溯规范化 | Task 3、Task 7、Task 12 | 规范化测试、token `normalization_transforms`、变体黄金词 |
| 音节、辅音簇、双辅音、滑音和 hiatus | Task 4、Task 11 | 音节单元测试与 40 条音节黄金记录 |
| 覆盖、acute、例外词典、附着词、重/轻 penult、候选警告 | Task 5、Task 11 | 重音单元测试、60 条重音黄金记录、`PRONUNCIATION_NEEDS_REVIEW` |
| 来源可追溯的教会式 G2P 和稳定规范音素表 | Task 6、Task 10 | G2P 单元测试、100 条辅音黄金记录、规则 ID 覆盖门禁 |
| 非法覆盖和未知音素在模型调用前失败 | Task 7 | `test_validation.py` 与管线测试 |
| `PronunciationPlan` 保留原文、规范文本、跨度、规则版本、来源、警告和覆盖 | Task 1、Task 3、Task 7 | 领域不变量和端到端管线测试 |
| 300–500 词黄金范围、全部精确匹配、常见礼仪词 | Task 8–13 | 至少 320 条 approved JSONL、审计 CLI、集成测试 |
| 阶段 1 不混入录音、模型、网页或长文本 | Global Constraints、Task 14 | 文件范围复核和 README 非目标 |

阶段 2 的录音恢复/对齐、阶段 3 的 MMS 与 Piper、阶段 4 的网页，以及阶段 5 的长文本韵律各自需要独立实施计划；它们不作为本计划完成条件。

---

## Plan Completion Criteria

完成本计划必须同时满足：

1. 发音规范文档包含所有实现规则、来源 locator、正反例和冲突决策。
2. `Pronouncer.default().analyze()` 能生成合法 `PronunciationPlan`。
3. 普通拼写重音不确定时产生 `PRONUNCIATION_NEEDS_REVIEW`。
4. 黄金词表至少 320 条，所有记录已批准并引用已登记来源。
5. 规范化、音节、重音、G2P、管线和审计测试全部通过。
6. Ruff、mypy、pytest、95% 覆盖率和黄金审计门禁全部通过。
7. 没有录音、模型、网页或长文本功能混入阶段 1。
