# Coder — 零框架自研编码 Agent

> **English / English 速览**
>
> **Coder is a fully self-built coding agent with no agent framework.**
> The whole ReAct loop (Think → Act → Observe) is hand-rolled: context
> management, tool execution, output parsing, loop termination and error
> recovery are all implemented from scratch. On top of the base loop it ships
> **7 mechanism enhancements** (plan-execute-verify orchestration, a failure
> pattern library with strategy rotation, message grading, automatic memory
> curation, tool fallback & adaptive tool routing, and planner-template
> learning) — each toggleable, each unit-tested.
>
> - **Deterministic & reproducible** — same task + same LLM + same tools ⇒
>   bit-identical results. **420+ offline tests** run with no LLM API call.
> - **Works out of the box** with any OpenAI-compatible gateway
>   (Agnes / DeepSeek / OpenAI / vLLM).
> - **Interfaces:** one-shot CLI, interactive REPL, and a FastAPI + SSE web UI
>   with multi-session tabs, live streaming, and an offline demo record/replay
>   mode.
>
> 📌 中文完整文档（创新点、架构、安装、用法）见下方。

一个**完全自研、不依赖任何 Agent 框架**的编码 Agent，核心是 ReAct 循环
（Think → Act → Observe）。上下文管理、工具执行、输出解析、循环终止、错误恢复
等每一块核心逻辑都手写实现。默认模型为 `agnes-3.0-flash`（OpenAI 兼容接口，
可换 Agnes / DeepSeek / OpenAI / vLLM 等任意网关）。

> **420+ 个测试全部离线通过**（无需调用 LLM API）；Agent 核心循环完全确定、
> 可复现——同任务、同 LLM、同工具环境下可逐位复现结果。

---

## 一、创新点（本项目与现有编码 Agent 的核心差异）

在基础 ReAct 循环之上，`Coder` 内置 **7 项机制增强**。每一项都手写、有单测、
可开关——你可以**全开**（默认 `full` 模式）或**全关**（纯 ReAct 基线）
各跑一遍，直观看到差异。

| # | 创新点 | 实现文件 | 解决了什么 |
|---|---|---|---|
| 1 | **Plan-Execute-Verify 编排器** | `planner.py` + `plan_executor.py` | 复杂任务先规划 → 子目标拓扑分层 → 并行执行 → 验证收敛；简单任务自动降级走 ReAct，**零降智** |
| 2 | **失败模式库 + 策略轮换 + 跨会话知识** | `failure_patterns.py` | 同类失败命令连败时自动**分类**、**轮换修复策略**、沉淀为**跨会话知识库**，避免模型在同一失败上反复烧步数 |
| 3 | **消息分级 + 上下文保护** | `message_grading.py` + `context.py` | 消息按重要性分级（CRITICAL/HIGH/LOW），上下文窗口满时**优先保留关键消息**，长任务不失忆 |
| 4 | **记忆自动整理（重要性驱逐）** | `memory_curator.py` + `memory_tool.py` | 记忆去重 + 重要性打分 + 容量上限，**自动淘汰**低价值条目 |
| 5 | **工具失败降级路由** | `tool_fallback.py` | 某工具连续失败 N 次后，自动给出"换一种工具 / 命令"的**降级建议** |
| 6 | **自适应工具 schema 路由** | `tool_routing.py` | 按任务阶段（探索/修改/验证/文档）**动态裁剪**暴露给模型的工具 schema，省 token |
| 7 | **规划模板跨会话学习** | `planner.py`（`remember_plan` / `recall_template`） | 成功计划沉淀为**模板**，相似任务免 LLM 规划调用，**越用越快** |

### 为什么这些是创新而非花架子：三条可复现证据

**① 跨实现对照**（`tests/cross_agent_benchmark.py` + `tests/cross_agent_report.md`）
把 `Coder`（全增强 vs 纯 ReAct 基线）与 **3 个真实开源 Agent 真身**
（mini-swe-agent / OneCode / smolagents）放在**同一任务、同一确定性 LLM、
同一工具环境**下跑，模型变量被完全控制，差异全部来自框架机制：

| 实现 | 步数 | 工具失败 | 框架干预 |
|---|---|---|---|
| mini-swe-agent(DefaultAgent) | 12 | 4 | 0 |
| OneCode(AgentLoop) | 9 | 3 | 0 |
| smolagents(CodeAgent) | 9 | 3 | 0 |
| coder_agent(纯 ReAct 基线) | 8 | 3 | 0 |
| **coder_agent(全增强)** | **8** | **3** | **1** |

全增强版在**步数/失败数**上与基线及三个开源真身**持平或更优**，且独有
"失败信号结构化 + 策略轮换 + 工具降级路由"的**框架主动干预**——这是这些开源
Agent 核心循环不具备的方法层差异。
（对照范围限定：商业闭源 Agent 如 Claude Code / Cursor / Copilot 核心循环
不可审计，本对照不对其做可达/不可达断言。）

**② 端到端真实任务**（`tests/e2e_full_enhanced_bugfix.py`）
全增强 Agent 用确定性 LLM 驱动，修一个**真实多 bug 任务**（`calculator.py`
同时有"除零不抛异常"+"整数除法"两个 bug），并**独立进程做地面真值验证**
（代码行为正确 + `pytest` 全绿）。证明 7 项增强确实带来"效果很好"。

**③ 全量回归**：`tests/` + `coder_agent/` 共 420+ 个测试离线通过，
每项增强各有独立单测。

---

## 二、架构

```
CLI / REPL / Web UI
  └─ Agent（ReAct 循环：Think → Act → Observe）
       ├─ LLMClient          OpenAI 兼容 API（Agnes / DeepSeek / OpenAI / vLLM…）
       │                     + urllib 传输层回退（网络不稳时降级）
       ├─ Planner/PlanExec   创新点 1：Plan-Execute-Verify 编排器
       ├─ FailureLibrary     创新点 2：失败模式库 + 策略轮换 + 跨会话知识
       ├─ MessageGrading     创新点 3：消息分级 + 上下文保护
       ├─ MemoryCurator      创新点 4：记忆自动整理（重要性驱逐）
       ├─ ToolFallbackRouter 创新点 5：工具失败降级路由
       ├─ ToolRouting        创新点 6：自适应工具 schema 路由
       ├─ PolicyGate         ALLOW / LOG / DENY —— 代码级安全，不靠 prompt
       ├─ ContextManager     3 层压缩 + 逐消息预算上限
       ├─ Verifier           pytest + 语法 + git-diff，变更门控重试
       ├─ HookRegistry       AGENT_STARTED/ENDED、PRE/POST_TOOL_USE、TURN_STOPPED
       ├─ SubagentRunner     4 个只读 / 专家子 Agent
       └─ ToolRegistry       5 核心工具 + 5 MCP 工具 + 5 Skills
```

---

## 三、安装

```bash
pip install -e ".[dev,ui]"
```

在项目根目录的 `.env` 中配置 API 访问（**切勿提交**该文件）：

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.agnes-ai.cn/v1   # 任意 OpenAI 兼容网关
MODEL_NAME=agnes-3.0-flash
```

---

## 四、使用

### 1. 单任务 CLI

```bash
# 一次性：跑任务并打印最终答案
python -m coder_agent.main "Fix the bug in tasks/bug_fix/calculator.py"

# 安装后的控制台脚本（pip install -e . 之后）
coder "Fix the bug in tasks/bug_fix/calculator.py" --mode full

# 常用参数
python -m coder_agent.main --inspect        # 查看上下文预算与模式后退出
python -m coder_agent.main --list-models    # 查看模型配置
python -m coder_agent.main "task" \
    --mode full \                           # goal | plan | dry-run | full
    --max-tokens 8000 \                     # 上下文压缩预算
    --keep-rounds 6 \                       # 最近 N 轮原样保留
    --token-budget 100000 \                 # 预算耗尽时收尾并停止
    --session-output session.jsonl \        # 记录会话供 --resume 使用
    --trace-output trace.jsonl              # JSONL 执行轨迹
```

### 2. 交互 REPL（推荐）

```bash
python -m coder_agent.ui.cli.repl            # 或：coder-repl（pip install -e . 后）
python -m coder_agent.ui.cli.repl --workspace path/to/project --mode goal
```

输入任务回车即跑，进度实时打印（`✓ read_file`、`── turn 2 ──`）。斜杠命令：

| 命令 | 作用 |
|---|---|
| `/run <task>` | 对任务跑 Agent（自动记录到 `~/.coder_sessions/`） |
| `/resume <file> [instruction]` | 恢复已记录的会话并继续 |
| `/sessions` | 列出已记录会话 |
| `/mode <name>` | 切换模式（对下一个 `/run` 生效） |
| `/status` | 上下文用量：token / 预算 / 轮数 |
| `/tools` | 列出模型可调用工具（核心 / MCP / Skills） |
| `/compact` | 压缩对话历史 |
| `/history` | 查看最近消息 |
| `/trace` | 查看上次运行的执行指标 |
| `/clear` | 重置显示状态 |
| `/help`、`/exit` | 略 |

粘贴到 stdin（批处理，无需 TTY）：

```bash
echo "/run List python files and summarize" | python -m coder_agent.ui.cli.repl
```

### 3. 会话恢复 —— 崩溃续跑

每次 `/run`（及带 `--session-output` 的 CLI 运行）都会把每条消息镜像进
**追加式 JSONL 日志**。Agent 被中断（API 掉线、Ctrl+C、终端关闭）后，
让新会话指向该日志即可带着完整的既往对话、状态、文件追踪继续：

```bash
# CLI
python -m coder_agent.main "task" --session-output session.jsonl   # …被中断！
python -m coder_agent.main --resume session.jsonl                  # 续跑（默认提示）
python -m coder_agent.main --resume session.jsonl "focus on the tests next"

# REPL
/run fix the calculator ...          # 自动记录
/resume ~/.coder_sessions/session_xxx.jsonl continue with the tests
```

日志可**链式续跑**：续跑时追加到同一文件，所以一次中断的"恢复"本身也能被恢复。

### 执行模式

| 模式 | 行为 |
|---|---|
| `goal` | 正常执行（默认） |
| `plan` | 只读：写/跑类工具被策略拒绝 |
| `dry-run` | 写操作只模拟并报告，不落盘 |
| `full` | 放宽策略拒绝（路径安全仍强制） |

### 演示回放模式 —— 无需 API 的确定性演示

录制的 SSE 事件序列存入 `~/.coder_replays/`，回放完全离线、不依赖 LLM API。
前端输入框输入命令即可：

```
/record D:/replays/demo.jsonl    # 录制本次运行
<运行任务...>
/replay                          # 回放最近一条 trace（无需 API）
/replay D:/replays/demo.jsonl    # 回放指定 trace
/health                          # 演示前检查 API 连通性与延迟
```

回放的时序与真实运行一致（按时间戳 sleep），前端 UI 零改动，所有 12 个
hook 事件（AgentStarted / ToolStart / Tool / VerifierResult / Turn / Compress 等）
都会正常渲染。答辩口径："回放用的就是 SessionJournal 机制本身。"

---

## 五、设计原则

- **模型负责决策，程序负责约束** —— 模型决定"做什么"；PolicyGate、路径
  安全检查与环境变量隔离决定它"可以做什么"。
- **每一步可观测** —— JSONL 轨迹、实时进度、上下文检查器。
- **错误是恢复路径而非崩溃** —— 格式错误注入纠正 prompt；API 错误指数
  退避；验证失败仅当工作区确有变更才重试。
- **验证优先于自述** —— "完成"由 pytest / 语法 / git-diff 说了算。

---

## 六、测试

```bash
python -m pytest tests/ coder_agent/ -q     # 420+ 个测试，约 3 分钟，无需 API 调用
```

关键入口：
- `tests/cross_agent_benchmark.py` —— 跨实现对照（含 3 个开源 Agent 真身）
- `tests/cross_agent_report.md` —— 对照量化报告（自动生成）
- `tests/e2e_full_enhanced_bugfix.py` —— 全增强端到端真实任务基准
- `tests/test_e2e.py` —— 基础 ReAct 闭环端到端
