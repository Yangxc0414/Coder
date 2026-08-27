# coder-agent

A minimal, self-implemented coding agent with a ReAct loop.
No agent frameworks used — core logic is hand-written.

## Quick Start

```bash
# Set up environment
cp .env.example .env
# Edit .env with your API key and base URL

# Install
pip install -e ".[dev]"

# Run
python -m coder_agent.main "Fix the bug in tests/example.py"
```

## Architecture

```
Agent (ReAct Loop)
  └─ LLMClient      → OpenAI-compatible API
  └─ ToolRegistry   → read_file, write_file, list_files, run_command
  └─ PolicyGate     → Dangerous command blocking, path safety
```

## Design Principles

- **模型负责决策，程序负责约束** — Policy Gate enforces security at code level
- **每步独立执行** — subprocess.run per command, no persistent shell session
- **错误可恢复** — FormatError injection lets the model self-correct
- **轨迹可观测** — Every step logged for debugging and interview demo
