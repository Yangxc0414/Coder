"""CLI entrypoint for coder-agent with mode selection and context inspection."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from coder_agent.agent import Agent
from coder_agent.inspector import ContextInspector
from coder_agent.llm.client import LLMClient
from coder_agent.mode import AgentMode, MODE_DESCRIPTIONS
from coder_agent.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool
from coder_agent.tools.registry import ToolRegistry
from coder_agent.tools.search import SearchTextTool
from coder_agent.tools.shell import RunCommandTool


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="coder-agent: A minimal coding agent with ReAct loop"
    )
    parser.add_argument("task", nargs="?", help="Programming task description")
    parser.add_argument("--task-file", help="Read task from file")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "agnes-2.5-flash"),
                        help="Model name (default: agnes-2.5-flash)")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"),
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--list-models", action="store_true",
                        help="Show available model configurations")
    parser.add_argument("--mode", choices=[m.value for m in AgentMode],
                        default="goal", help="Agent execution mode: goal/plan/dry-run/full")
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="Context token budget (default: 8000)")
    parser.add_argument("--keep-rounds", type=int, default=6,
                        help="Recent rounds to keep full (default: 6)")
    parser.add_argument("--trace-output", type=str, help="Path to save trace.jsonl")
    parser.add_argument("--inspect", action="store_true",
                        help="Show context configuration and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.list_models:
        print("\nConfigured Models:")
        print(f"  MODEL_NAME={os.getenv('MODEL_NAME', 'agnes-2.5-flash')}")
        print(f"  OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL', 'https://api.agnes-ai.cn/v1')}")
        print("\nSupported providers (any OpenAI-compatible):")
        print("  - Agnes AI (default)")
        print("  - DeepSeek")
        print("  - OpenAI")
        print("  - OpenRouter")
        print("  - Local vLLM/Ollama")
        sys.exit(0)

    setup_logging(args.verbose)

    if args.inspect:
        print("\n=== Context Configuration ===")
        print(f"  Max tokens:  {args.max_tokens}")
        print(f"  Keep rounds: {args.keep_rounds}")
        print(f"  Budget:      {max(1000, args.max_tokens - 1500)} tokens")
        print("\nAvailable Modes:")
        for mode, desc in MODE_DESCRIPTIONS.items():
            print(f"  {mode.value:<10} — {desc}")
        sys.exit(0)

    task = args.task
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8").strip()
    if not task:
        parser.error("Please provide a task (argument or --task-file)")

    mode = AgentMode(args.mode)
    registry = ToolRegistry()
    workspace = Path(args.workspace).resolve()

    dry_run = (mode == AgentMode.DRY_RUN)
    registry.register(ReadFileTool(workspace))
    registry.register(WriteFileTool(workspace, dry_run=dry_run))
    registry.register(ListFilesTool(workspace))
    registry.register(SearchTextTool(workspace))
    registry.register(RunCommandTool(workspace))

    llm = LLMClient(model=args.model, base_url=args.base_url)

    print(f"\nTask: {task[:100]}{'...' if len(task) > 100 else ''}")
    print(f"Workspace: {workspace}")
    print(f"Model: {args.model}")
    print(f"Mode: {mode.value} — {MODE_DESCRIPTIONS[mode]}")
    print(f"Context: max={args.max_tokens} tokens, keep={args.keep_rounds} rounds")
    print("=" * 60)

    trace_path = Path(args.trace_output) if args.trace_output else None
    agent = Agent(
        llm_client=llm, registry=registry, workspace=workspace,
        trace_output=trace_path, mode=mode,
    )
    result = agent.run(task)

    print("\n" + "=" * 60)
    agent.inspector.print_status(agent.messages)
    print("\n" + "=" * 60)
    print(f"\nFinal Answer:\n{result}")
    if agent.trace.get_entries():
        print("\n" + agent.trace.summarize())


if __name__ == "__main__":
    main()
