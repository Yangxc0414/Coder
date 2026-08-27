"""CLI entrypoint for coder-agent."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from coder_agent.agent import Agent
from coder_agent.llm.client import LLMClient
from coder_agent.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool
from coder_agent.tools.registry import ToolRegistry
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
    parser.add_argument(
        "--task-file",
        help="Read task description from a file",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "deepseek-chat"),
        help="Model name (default: from MODEL_NAME env or 'deepseek-chat')",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    task = args.task
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8").strip()
    if not task:
        parser.error("Please provide a task (argument or --task-file)")

    # Initialize components
    registry = ToolRegistry()
    workspace = Path(args.workspace).resolve()

    registry.register(ReadFileTool(workspace))
    registry.register(WriteFileTool(workspace))
    registry.register(ListFilesTool(workspace))
    registry.register(RunCommandTool(workspace))

    llm = LLMClient(model=args.model, base_url=args.base_url)

    print(f"\nTask: {task[:100]}{'...' if len(task) > 100 else ''}")
    print(f"Workspace: {workspace}")
    print(f"Model: {args.model}")
    print("=" * 60)

    agent = Agent(
        llm_client=llm,
        registry=registry,
        workspace=workspace,
    )
    result = agent.run(task)

    print("\n" + "=" * 60)
    print(f"Final Answer:\n{result}")


if __name__ == "__main__":
    main()
