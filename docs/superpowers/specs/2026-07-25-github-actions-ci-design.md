# LatinTTS GitHub Actions 双平台 CI 设计

- 状态：已批准
- 设计批准日期：2026-07-25
- 适用分支：`codex/fix-mms-model-revision`
- 目标仓库：`SkyRainCho/LatinTTS`

## 1. 文档目的

本规格为 LatinTTS 增加首个 GitHub Actions 持续集成工作流，并修复当前
`ruff format --check .` 因历史实施计划中的代码块格式而失败的问题。

CI 必须在 pull request 和 `main` 分支 push 上自动运行，以只读权限完成静态质量检查、
金标审计和 Windows/Linux 双平台全量测试。真实 MMS GPU 冒烟测试继续保留为本地操作，
不进入普通 CI。

## 2. 已确认决策

1. 使用一个 `.github/workflows/ci.yml` 工作流。
2. CI 包含一个单平台质量 job 和一个双平台测试矩阵 job。
3. 测试矩阵使用 `windows-latest` 与 `ubuntu-latest`。
4. Python 固定为项目唯一受支持的 `3.10` 系列。
5. 每个平台只执行一次带覆盖率的全量 pytest，不重复执行不带覆盖率的同一套测试。
6. 每个平台独立满足 `95.00%` 覆盖率门槛。
7. `docs/superpowers/plans/` 只从 Ruff formatter 排除，仍可被 Ruff linter 检查。
8. 同一分支的新工作流运行会取消旧运行。
9. 工作流只授予 `contents: read`。
10. 不在 CI 下载真实 MMS 模型、运行真实 CUDA 对齐或读取 `local-data/`。

## 3. 方案比较

### 3.1 采用方案：质量 job + 测试矩阵 job

`quality` 在 Ubuntu 上执行快速、平台无关的门禁；`tests` 在 Windows 与 Ubuntu 上执行同一条
全量覆盖率命令。这样避免重复运行 Ruff、mypy 和金标审计，同时保留跨平台行为证明。

### 3.2 未采用：所有检查放入矩阵

该方案配置更短，但会在两个平台重复执行所有静态检查，增加无意义的 runner 时间和日志噪音。

### 3.3 未采用：拆分多个 workflow

单独的质量与测试 workflow 隔离更强，但当前仓库只有一套 Python 包和一套发布门禁，拆分会增加
触发、并发取消和 required checks 管理成本。

## 4. Ruff 格式边界

在 `pyproject.toml` 中新增 formatter 专属排除：

```toml
[tool.ruff.format]
exclude = ["docs/superpowers/plans/**"]
```

该设置只影响 `ruff format`。现有 `[tool.ruff.lint]` 规则保持不变，因此历史计划中的 Python
代码块仍受 lint 检查；普通源码、测试和其他 Markdown 文档仍受 formatter 检查。

选择目录级排除而不是格式化当前三个文件，原因如下：

- `docs/superpowers/plans/` 是日期化的历史实施快照；
- 自动格式化会产生与功能无关的大型 diff；
- 目录级策略可避免未来 Ruff 版本再次重写历史计划；
- formatter 专属配置不会降低源码 lint 覆盖。

## 5. Workflow 触发与权限

工作流名称为 `CI`，触发条件为：

```yaml
on:
  pull_request:
  push:
    branches:
      - main
```

顶层权限为：

```yaml
permissions:
  contents: read
```

并发组以 workflow 名称和 pull request head ref 或 branch ref 组成：

```yaml
concurrency:
  group: ci-${{ github.workflow }}-${{ github.head_ref || github.ref }}
  cancel-in-progress: true
```

这确保同一 PR 的旧提交不继续占用 runner，同时不会互相取消不同分支的运行。

## 6. `quality` job

`quality` 使用 `ubuntu-latest`，步骤固定为：

1. 使用 `actions/checkout@v6` 检出代码。
2. 使用 `actions/setup-python@v6` 安装 Python 3.10，并启用 pip 缓存。
3. 运行 `python -m pip install --upgrade pip`。
4. 运行 `python -m pip install -e ".[dev]"`。
5. 依次执行：

```text
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
git diff --check
```

任一步骤非零退出即令 job 失败。

## 7. `tests` 矩阵 job

`tests` 使用以下矩阵：

```yaml
strategy:
  fail-fast: false
  matrix:
    os:
      - windows-latest
      - ubuntu-latest
```

两个平台分别执行：

1. 使用 `actions/checkout@v6` 检出代码。
2. 使用 `actions/setup-python@v6` 安装 Python 3.10，并按 `pyproject.toml` 启用 pip 缓存。
3. 升级 pip。
4. 安装 `-e ".[dev]"`。
5. 执行：

```text
python -m pytest --cov=latintts --cov-report=term-missing --cov-fail-under=95 -q
```

`fail-fast: false` 保证一个平台失败时另一个平台仍完成并给出独立诊断。

## 8. 跨平台覆盖意图

- Windows job 覆盖 Windows 文件句柄、目录共享与原子替换契约。
- Ubuntu job 覆盖 POSIX fork、文件和目录 symlink 契约。
- 平台特定的 `pytest.skip` 属于预期行为；每个平台必须保证其余测试全部通过。
- 覆盖率门槛在每个平台独立执行，不合并 coverage 文件。

## 9. 非目标

- 不运行 `latintts.corpus align --smoke-test` 的真实模型路径。
- 不安装 `requirements/corpus.txt`。
- 不使用 GPU runner。
- 不上传 coverage artifact 或接入第三方覆盖率服务。
- 不发布包、不创建 release、不修改 PR 状态。
- 不把现有 Draft PR 自动标记为 Ready for review。

## 10. 验证与验收

实施后必须在本地验证：

```text
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m latintts.audit tests/fixtures/gold_pronunciations.jsonl
git diff --check
```

还必须验证 workflow YAML 可被解析，提交并推送到当前 PR 分支，然后等待 GitHub Actions 的
`quality`、`tests (windows-latest)` 和 `tests (ubuntu-latest)` 全部成功。

验收标准：

- 历史计划文件保持字节不变；
- 本地 Ruff format 门禁由失败转为成功；
- PR 出现三个预期 CI check；
- 双平台测试均达到至少 `95.00%` 覆盖率；
- 工作区不包含 CI 产生的非忽略文件；
- 真实语料与模型缓存不上传 GitHub。

## 11. 官方参考

- Ruff 支持在 `[tool.ruff.format]` 中设置 formatter 专属 `exclude`：
  <https://docs.astral.sh/ruff/configuration/>
- GitHub 官方 Python Actions 指南：
  <https://docs.github.com/en/actions/tutorials/build-and-test-code/python>
- `actions/setup-python@v6` 的缓存和最小权限说明：
  <https://github.com/actions/setup-python>
- `actions/checkout@v6`：
  <https://github.com/actions/checkout>
