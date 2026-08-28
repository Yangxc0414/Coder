# coder-agent

A minimal, self-implemented coding agent with a ReAct loop.
No agent frameworks — every piece of core logic (context management, tool
execution, output parsing, loop termination, error recovery) is hand-written.

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
| `/run <task>` | Run the agent on a task |
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

### Execution modes

| Mode | Behavior |
|---|---|
| `goal` | Normal execution (default) |
| `plan` | Read-only: write/run tools are policy-denied |
| `dry-run` | Writes are simulated and reported, nothing touches disk |
| `full` | Relaxed policy denials (path safety still enforced) |

## Design principles

- **模型负责决策，程序负责约束** — the model decides *what* to do; PolicyGate,
  path-safety checks and environment-variable isolation decide what it *may* do.
- **Every step is observable** — JSONL trace, live progress, context inspector.
- **Errors are recovery paths, not crashes** — format errors inject correction
  prompts; API errors back off exponentially; verification failures only retry
  when the workspace actually changed.
- **Verification over self-report** — "done" means pytest/syntax/git-diff say so.

## Testing

```bash
python -m pytest tests/ -q     # 159 tests, ~1.5 min, no API calls needed
```
