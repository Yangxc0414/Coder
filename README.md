# coder-agent (Coder)

A minimal, self-implemented coding agent with a ReAct loop.
No agent frameworks — every piece of core logic (context management, tool
execution, output parsing, loop termination, error recovery) is hand-written.
All 392 tests pass offline (no LLM API calls); the agent's core loop is fully
deterministic and reproducible.

```
CoderRepl / CLI
  └─ Agent (ReAct Loop: Think → Act → Observe)
       ├─ LLMClient          OpenAI-compatible API (Agnes / DeepSeek / OpenAI / vLLM…)
       ├─ PolicyGate         ALLOW / LOG / DENY — code-level safety, not prompts
       ├─ ContextManager     3-layer compression + per-message budget caps
       ├─ AgentState/Memory  progress & long-term notes injected into system prompt
       ├─ RecoveryStrategy   6 error types × differentiated recovery
       ├─ Verifier           pytest + syntax + git-diff, mutation-gated retries
       ├─ HookRegistry       AGENT_STARTED/ENDED, PRE/POST_TOOL_USE, TURN_STOPPED
       ├─ SubagentRunner     4 read-only/specialist sub-agents
       └─ ToolRegistry       5 core tools + 5 MCP tools + 5 Skills
```

## Install

```bash
pip install -e ".[dev,ui]"
```

Configure API access in a `.env` file at the project root (**never commit it**):

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.agnes-ai.cn/v1   # any OpenAI-compatible gateway
MODEL_NAME=agnes-2.5-flash
```

## Usage

### 1. Single-task CLI

```bash
# one-shot: agent runs the task and prints the final answer
python -m coder_agent.main "Fix the bug in tasks/bug_fix/calculator.py"

# installed console script (after pip install -e .)
coder "Fix the bug in tasks/bug_fix/calculator.py" --mode full

# useful flags
python -m coder_agent.main --inspect        # show context budget & modes, exit
python -m coder_agent.main --list-models    # show model configuration
python -m coder_agent.main "task" \
    --mode full \                           # goal | plan | dry-run | full
    --max-tokens 8000 \                     # context compression budget
    --keep-rounds 6 \                       # recent rounds kept verbatim
    --token-budget 100000 \                 # wrap up & stop when exhausted
    --session-output session.jsonl \        # journal messages for later --resume
    --trace-output trace.jsonl              # JSONL execution trace
```

### 2. Interactive REPL (recommended)

```bash
python -m coder_agent.ui.cli.repl            # or: coder-repl (after pip install -e .)
python -m coder_agent.ui.cli.repl --workspace path/to/project --mode goal
```

Type a task and press Enter to run the agent; progress prints live
(`✓ read_file`, `── turn 2 ──`). Slash commands:

| Command | Action |
|---|---|
| `/run <task>` | Run the agent on a task (auto-journaled to `~/.coder_sessions/`) |
| `/resume <file> [instruction]` | Restore a journaled session and continue it |
| `/sessions` | List journaled sessions |
| `/mode <name>` | Switch mode — applies to the next `/run` |
| `/status` | Context usage: tokens / budget / rounds |
| `/tools` | List tools the model can call (core / MCP / Skills) |
| `/compact` | Compress conversation history |
| `/history` | Show recent messages |
| `/trace` | Show execution metrics of the last run |
| `/clear` | Reset display state |
| `/help`, `/exit` | You guessed it |

Paste into stdin (batch mode, no TTY needed):

```bash
echo "/run List python files and summarize" | python -m coder_agent.ui.cli.repl
```

### 3. Session resume — crash recovery

Every `/run` (and every CLI run with `--session-output`) mirrors each message
into an append-only JSONL journal. If the agent is interrupted (API drop,
Ctrl+C, terminal closed), point a new session at the journal and it continues
with the full prior conversation, state and file tracking rebuilt:

```bash
# CLI
python -m coder_agent.main "task" --session-output session.jsonl   # ... interrupted!
python -m coder_agent.main --resume session.jsonl                  # continue (default prompt)
python -m coder_agent.main --resume session.jsonl "focus on the tests next"

# REPL
/run fix the calculator ...          # journaled automatically
/resume ~/.coder_sessions/session_xxx.jsonl continue with the tests
```

The journal is chain-resumable: resuming appends to the same file, so an
interrupted *recovery* can itself be recovered.

### Execution modes

| Mode | Behavior |
|---|---|
| `goal` | Normal execution (default) |
| `plan` | Read-only: write/run tools are policy-denied |
| `dry-run` | Writes are simulated and reported, nothing touches disk |
| `full` | Relaxed policy denials (path safety still enforced) |

### Demo replay mode — deterministic presentations without API

录制的 SSE 事件序列存入 `~/.coder_replays/`，回放时完全离线、不依赖 LLM API。
前端输入框输入命令即可使用：

```
/record D:/replays/demo.jsonl    # 录制本次运行
<运行任务...>
/replay                          # 回放最近一条 trace（无需 API）
/replay D:/replays/demo.jsonl    # 回放指定 trace
/health                          # 演示前检查 API 连通性与延迟
```

回放的时序与真实运行一致（按时间戳 sleep），前端 UI 零改动，所有 12 个
hook 事件（AgentStarted / ToolStart / Tool / VerifierResult / Turn / Compress 等）
都会正常渲染。答辩口径："回放用的就是 SessionJournal 机制本身"。

## Design principles

- **模型负责决策，程序负责约束** — the model decides *what* to do; PolicyGate,
  path-safety checks and environment-variable isolation decide what it *may* do.
- **Every step is observable** — JSONL trace, live progress, context inspector.
- **Errors are recovery paths, not crashes** — format errors inject correction
  prompts; API errors back off exponentially; verification failures only retry
  when the workspace actually changed.
- **Verification over self-report** — "done" means pytest/syntax/git-diff say so.

## Innovations (创新点)

Beyond the base ReAct loop, `Coder` ships **seven mechanism enhancements** that
make it behave better on real tasks. Each is hand-written, unit-tested, and
switchable — you can run the agent *with* all of them (the default `full`
mode) or *without* (baseline) and watch the difference.

| # | 创新点 | 实现 | 作用 |
|---|---|---|---|
| 1 | **Plan-Execute-Verify 编排器** | `planner.py` + `plan_executor.py` | 复杂任务先规划 → 子目标拓扑分层 → 并行执行 → 验证收敛，简单任务自动走 ReAct（零降智） |
| 2 | **失败模式库 + 策略轮换 + 跨会话知识** | `failure_patterns.py` | 同类失败命令连败时结构化分类、轮换修复策略、沉淀为跨会话知识库，避免模型在同一失败上反复烧步数 |
| 3 | **消息分级 + 上下文保护** | `message_grading.py` + `context.py` | 按重要性分级（CRITICAL/HIGH/LOW），窗口满时优先保留关键消息，长任务不"失忆" |
| 4 | **记忆自动整理（重要性驱逐）** | `memory_curator.py` + `memory_tool.py` | 记忆去重 + 重要性打分 + 容量上限，自动淘汰低价值条目 |
| 5 | **工具失败降级路由** | `tool_fallback.py` | 某工具连续失败 N 次后，自动给出"换一种工具/命令"的降级建议 |
| 6 | **自适应工具 schema 路由** | `tool_routing.py` | 按任务阶段（探索/修改/验证/文档）动态裁剪暴露给模型的工具 schema，省 token |
| 7 | **规划模板跨会话学习** | `planner.py`（`remember_plan` / `recall_template`） | 成功的计划沉淀为模板，相似任务免 LLM 规划调用，越用越快 |

**为什么这些是创新而不是花架子**：每一项都有独立单测 + 一条端到端证据。
跨实现对照实验（`tests/cross_agent_benchmark.py` + `tests/cross_agent_report.md`）
把 `Coder`（全增强 vs 纯 ReAct 基线）和 **3 个真实开源 agent 真身**
（mini-swe-agent / OneCode / smolagents）放在**同一任务、同一确定性 LLM、
同一工具环境**下跑，模型变量被完全控制，差异全部来自框架机制：

| 实现 | 步数 | 工具失败 | 框架干预 |
|---|---|---|---|
| mini-swe-agent(DefaultAgent) | 12 | 4 | 0 |
| OneCode(AgentLoop) | 9 | 3 | 0 |
| smolagents(CodeAgent) | 9 | 3 | 0 |
| coder_agent(纯 ReAct 基线) | 8 | 3 | 0 |
| **coder_agent(全增强)** | **8** | **3** | **1** |

结论：全增强版在步数/失败数上与基线及三个开源真身**持平或更优**，且独有
"失败信号结构化 + 策略轮换 + 工具降级路由"的框架主动干预——这是这些开源 agent
核心 loop 不具备的方法层差异。端到端真实任务基准（`tests/e2e_full_enhanced_bugfix.py`）
再以确定性 LLM 驱动、地面真值验证（代码行为正确 + pytest 全绿）确认 7 项增强
确实带来"效果很好"的可复现结果。

## Testing

```bash
python -m pytest tests/ -q     # 392 tests, ~1.5 min, no API calls needed
```
