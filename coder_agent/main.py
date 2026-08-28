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
from coder_agent.context import ContextManager
from coder_agent.inspector import ContextInspector
from coder_agent.journal import DEFAULT_RESUME_PROMPT, SessionJournal, load_journal, replay_state
from coder_agent.llm.client import LLMClient
from coder_agent.mode import AgentMode, MODE_DESCRIPTIONS
from coder_agent.tools.registry import create_default_registry
from coder_agent.verifier import Verifier

# Check llm-verifier availability at startup
try:
    from coder_agent.verifier_llm import _import_llm_verifier
    _LLM_VERIFIER_AVAILABLE = True
except ImportError:
    _LLM_VERIFIER_AVAILABLE = False


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _force_utf8_console() -> None:
    """Guard against UnicodeEncodeError on Windows GBK consoles (cmd.exe).

    Emoji and box-drawing chars in our output would crash plain print()
    under cp936; reconfigure streams to UTF-8 with replacement instead.
    """
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass


def main() -> None:
    _force_utf8_console()
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
    parser.add_argument("--session-output", type=str,
                        help="Journal every message to this JSONL file (enables /resume later)")
    parser.add_argument("--resume", type=str,
                        help="Restore a session from a journal file and continue it "
                             "(task argument becomes the continuation instruction)")
    parser.add_argument("--inspect", action="store_true",
                        help="Show context configuration and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--llm-verifier-mode",
        choices=["off", "progress", "select", "full"],
        default=os.getenv("LLM_VERIFIER_MODE", "off"),
        help=(
            "LLM verifier mode (requires pip install 'coder-agent[verifier]'):\n"
            "  off      — disabled (default)\n"
            "  progress — online per-step scoring via ProgressTracker\n"
            "  select   — final best-of-N selection via llm_verifier.select()\n"
            "  full     — both progress tracking + final selection"
        ),
    )
    parser.add_argument(
        "--llm-verifier-model",
        default=os.getenv("LLM_VERIFIER_MODEL", "gemini-2.5-flash"),
        help="Verifier model (default: gemini-2.5-flash, also supports deepseek-v4-flash)",
    )
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
    if not task and not args.resume:
        parser.error("Please provide a task (argument or --task-file)")

    mode = AgentMode(args.mode)
    workspace = Path(args.workspace).resolve()
    registry = create_default_registry(workspace, mode)

    llm = LLMClient(model=args.model, base_url=args.base_url)

    if args.resume:
        restored = load_journal(args.resume)
        if not restored["messages"]:
            parser.error(f"Journal has no messages: {args.resume}")
        task = task or DEFAULT_RESUME_PROMPT
        print(f"\nResuming session: {args.resume}")
        print(f"Restored: {len(restored['messages'])} messages")
    print(f"\nTask: {task[:100]}{'...' if len(task) > 100 else ''}")
    print(f"Workspace: {workspace}")
    print(f"Model: {args.model}")
    print(f"Mode: {mode.value} — {MODE_DESCRIPTIONS[mode]}")
    print(f"Context: max={args.max_tokens} tokens, keep={args.keep_rounds} rounds")

    # LLM verifier status
    if args.llm_verifier_mode != "off":
        if _LLM_VERIFIER_AVAILABLE:
            print(f"LLM Verifier: {args.llm_verifier_mode} (model={args.llm_verifier_model})")
        else:
            print("LLM Verifier: ⚠️  enabled but NOT installed — run: pip install 'coder-agent[verifier]'")
    print("=" * 60)

    trace_path = Path(args.trace_output) if args.trace_output else None
    journal = None
    if args.resume:
        journal = SessionJournal(args.resume, append=True)
    elif args.session_output:
        journal = SessionJournal(args.session_output)

    agent = Agent(
        llm_client=llm, registry=registry, workspace=workspace,
        trace_output=trace_path, mode=mode,
        context_manager=ContextManager(
            max_tokens=args.max_tokens, keep_rounds=args.keep_rounds
        ),
        verifier=Verifier(workspace, task=task),
        llm_verifier_mode=args.llm_verifier_mode,
        llm_verifier_model=args.llm_verifier_model,
        journal=journal,
    )

    if args.resume:
        agent.messages = restored["messages"]
        replay_state(restored["messages"], agent.state)
        for m in restored["messages"]:
            if m.get("role") == "user":
                agent.state.task_goal = (m.get("content") or "")[:200]
                break
        if agent.state.modified_files:
            print(f"Modified so far: {', '.join(agent.state.modified_files)}")

    try:
        result = agent.run(task, resume=bool(args.resume))
    finally:
        if journal:
            journal.close()

    print("\n" + "=" * 60)
    agent.inspector.print_status(agent.messages)
    print("\n" + "=" * 60)
    print(f"\nFinal Answer:\n{result}")
    if agent.trace.get_entries():
        print("\n" + agent.trace.summarize())


if __name__ == "__main__":
    main()
