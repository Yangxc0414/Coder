"""CLI UI — interactive REPL with Rich + prompt_toolkit.

Inspired by OneCode's InlineRepl pattern:
- Static output region (Rich Console)
- Dynamic input via prompt_toolkit
- Slash commands (/mode, /status, /compact, etc.)
- Transient permission modal
- Streaming tool output
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from coder_agent.agent import Agent
from coder_agent.inspector import ContextInspector
from coder_agent.llm.client import LLMClient
from coder_agent.mode import AgentMode, MODE_DESCRIPTIONS
from coder_agent.tools.registry import create_default_registry
from coder_agent.trace import TraceRecorder
from coder_agent.verifier import Verifier


@dataclass
class CliState:
    """Runtime state shared across the REPL."""
    workspace: Path
    mode: AgentMode
    max_tokens: int
    keep_rounds: int
    model: str
    base_url: str | None
    trace_recorder: TraceRecorder
    context_inspector: ContextInspector
    messages: list[dict] = None  # type: ignore[assignment]
    steps: int = 0
    last_answer: str = ""

    def __post_init__(self):
        if self.messages is None:
            self.messages = []


class CoderRepl:
    """Interactive REPL for coder-agent."""

    def __init__(
        self,
        workspace: Path,
        model: str = "agnes-2.5-flash",
        base_url: str | None = None,
        mode: AgentMode = AgentMode.GOAL,
        max_tokens: int = 8000,
        keep_rounds: int = 6,
    ) -> None:
        self.workspace = workspace.resolve()
        self.model = model
        self.base_url = base_url
        self.mode = mode
        self.max_tokens = max_tokens
        self.keep_rounds = keep_rounds

        self.state = CliState(
            workspace=self.workspace,
            mode=mode,
            max_tokens=max_tokens,
            keep_rounds=keep_rounds,
            model=model,
            base_url=base_url,
            trace_recorder=TraceRecorder(),
            context_inspector=ContextInspector(max_tokens=max_tokens, keep_rounds=keep_rounds),
        )
        self._console = Console()
        self._agent: Agent | None = None
        self._session_id: int = 0

    def _build_agent(self, task: str) -> Agent:
        """Build a fresh Agent for a task."""
        self._session_id += 1
        registry = create_default_registry(self.workspace, self.mode)

        llm = LLMClient(model=self.model, base_url=self.base_url)
        verifier = Verifier(self.workspace, task=task)

        agent = Agent(
            llm_client=llm,
            registry=registry,
            workspace=self.workspace,
            trace_output=None,
            verifier=verifier,
            mode=self.mode,
        )
        return agent

    def _print_header(self) -> None:
        """Print session header."""
        self._console.print()
        self._console.print(Panel(
            f"[bold cyan]Coder-Agent[/bold cyan] — [yellow]{self.mode.value}[/yellow] mode",
            subtitle=f"Workspace: {self.workspace}  |  Model: {self.model}",
            border_style="cyan",
        ))
        self._console.print()

    def _show_status(self) -> None:
        """Show current context status."""
        info = self.state.context_inspector.inspect(self.state.messages)
        self._console.print()
        self._console.print("[bold yellow]═" + "─" * 50 + "═[/bold yellow]")
        self._console.print("[bold]📊 Context Status[/bold]")
        self._console.print(f"  Messages:    {info['total_messages']}")
        self._console.print(f"  Tokens:      {info['total_tokens']} / {info['max_tokens']}")
        self._console.print(f"  Utilization: {info['utilization_pct']}%")
        self._console.print(f"  Rounds:      {info['rounds_count']} (keeping {info['keep_rounds']} full)")
        status_icon = "⚠️" if info['needs_compression'] else "✅"
        self._console.print(f"  Status:      {status_icon} {'NEEDS COMPRESSION' if info['needs_compression'] else 'OK'}")
        self._console.print(f"  Mode:        {self.state.mode.value}")
        self._console.print(f"  Steps:       {self.state.steps}")
        if self.state.trace_recorder.get_entries():
            metrics = self.state.trace_recorder.get_metrics()
            self._console.print(f"  Tool calls:  {metrics['total_tool_calls']} ({metrics['successful_calls']} ok)")
            self._console.print(f"  Success:     {metrics['success_rate']:.1%}")
        self._console.print("[bold yellow]═" + "─" * 50 + "═[/bold yellow]")
        self._console.print()

    def _show_help(self) -> None:
        """Show available commands."""
        help_text = """
[bold yellow]Available Commands:[/bold yellow]
  [cyan]/run <task>[/cyan]     Run agent on a task
  [cyan]/mode <name>[/cyan]     Switch mode: goal / plan / dry-run / full
  [cyan]/status[/cyan]          Show context status
  [cyan]/compact[/cyan]         Force context compression
  [cyan]/history[/cyan]         Show recent memory
  [cyan]/trace[/cyan]           Show trace summary
  [cyan]/clear[/cyan]           Clear conversation
  [cyan]/exit[/cyan]            Exit

[bold yellow]Modes:[/bold yellow]
  goal      — Normal execution
  plan      — Read-only analysis
  dry-run   — Show changes without applying
  full      — Relaxed policy checks
"""
        self._console.print(Panel(help_text.strip(), title="Help", border_style="yellow"))

    def _handle_command(self, text: str) -> bool:
        """Handle a slash command. Returns True to continue, False to exit."""
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/exit" or cmd == "/quit":
            return False
        elif cmd == "/help" or cmd == "/h":
            self._show_help()
        elif cmd == "/status" or cmd == "/stat":
            self._show_status()
        elif cmd == "/clear" or cmd == "/cls":
            self.state.messages = []
            self.state.steps = 0
            self._console.print("[dim]Context cleared.[/dim]")
        elif cmd == "/mode":
            if not arg:
                self._console.print("[yellow]Current mode:[/yellow] " + self.state.mode.value)
                self._console.print("[dim]Usage: /mode <goal|plan|dry-run|full>[/dim]")
            else:
                try:
                    self.state.mode = AgentMode(arg.lower())
                    self._console.print(f"[green]✓ Mode switched to: {self.state.mode.value}[/green]")
                except ValueError:
                    self._console.print(f"[red]✗ Unknown mode: {arg}. Use: goal, plan, dry-run, full[/red]")
        elif cmd == "/compact":
            n = len(self.state.messages)
            if n <= 2 * self.state.keep_rounds:
                self._console.print("[dim]No compression needed ({n} messages, under threshold).[/dim]")
            else:
                # Keep system + recent rounds, drop older tool results
                # Preserve the first message (system context placeholder) and last 2*keep_rounds
                keep = max(2, self.state.keep_rounds * 2)
                self.state.messages = self.state.messages[-keep:]
                self._console.print(f"[green]✓ Compressed {n} → {len(self.state.messages)} messages[/green]")
        elif cmd == "/history":
            if self.state.messages:
                self._console.print("[bold]Recent Messages:[/bold]")
                for i, msg in enumerate(self.state.messages[-10:]):
                    role = msg.get("role", "?")
                    content = (msg.get("content") or "")[:80]
                    self._console.print(f"  {i}: [{role}] {content}")
            else:
                self._console.print("[dim]No messages yet.[/dim]")
        elif cmd == "/trace":
            entries = self.state.trace_recorder.get_entries()
            if entries:
                self._console.print(self.state.trace_recorder.summarize())
            else:
                self._console.print("[dim]No trace data yet.[/dim]")
        elif cmd == "/run":
            if not arg:
                self._console.print("[red]Usage: /run <task description>[/red]")
            else:
                self._run_task(arg)
        else:
            self._console.print(f"[red]Unknown command: {cmd}. Type /help for usage.[/red]")
        return True

    def _run_task(self, task: str) -> None:
        """Run the agent on a task."""
        self._console.print(f"[bold cyan]→ {task}[/bold cyan]")
        self._console.print()

        agent = self._build_agent(task)

        try:
            answer = agent.run(task)
            self.state.steps = agent.state.step
            self.state.last_answer = answer
            self.state.messages = agent.messages

            # Print answer with Rich formatting
            self._console.print(Panel(
                answer,
                title=f"✅ Answer (step {agent.state.step})",
                border_style="green",
            ))

            # Show context status
            self.state.context_inspector.print_status(agent.messages)

            # Show trace summary
            if agent.trace.get_entries():
                self._console.print(agent.trace.summarize())

        except Exception as e:
            self._console.print(f"[red]❌ Error: {e}[/red]")
            # Still sync what we can on error
            self.state.steps = agent.state.step
            self.state.messages = agent.messages

    def run(self) -> int:
        """Run the interactive REPL."""
        self._print_header()
        self._show_help()

        # Check if running in non-interactive mode
        if not sys.stdin.isatty():
            # Batch mode: read from stdin
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    if not self._handle_command(line):
                        return 0
                else:
                    self._run_task(line)
            return 0

        # Interactive mode with prompt_toolkit
        history_path = self.workspace / ".coder_history"
        session = PromptSession(
            history=FileHistory(str(history_path)),
            complete=False,
        )

        prompt = "➜ [bold cyan]task[/bold cyan] "

        while True:
            try:
                text = session.prompt(prompt).strip()
                if not text:
                    continue
                if text.startswith("/"):
                    if not self._handle_command(text):
                        break
                else:
                    self._run_task(text)
            except KeyboardInterrupt:
                self._console.print()
                break
            except EOFError:
                break

        self._console.print("\n[dim]Goodbye! 👋[/dim]")
        return 0


def main() -> int:
    """CLI entrypoint for interactive mode."""
    import argparse
    parser = argparse.ArgumentParser(description="coder-agent: Interactive CLI")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--model", default="agnes-2.5-flash", help="Model name")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--mode", default="goal", choices=["goal", "plan", "dry-run", "full"])
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--batch", action="store_true", help="Batch mode (read from stdin)")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    repl = CoderRepl(
        workspace=workspace,
        model=args.model,
        base_url=args.base_url,
        mode=AgentMode(args.mode),
        max_tokens=args.max_tokens,
    )
    return repl.run()


if __name__ == "__main__":
    raise SystemExit(main())
