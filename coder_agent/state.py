"""AgentState — tracks the agent's execution progress and context."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Tracks the agent's current execution state.

    This is separate from the LLM context (messages) and memory.
    It provides the agent with a persistent sense of "where am I?"
    across all steps of a task.

    Designed to be injected into the System Prompt so the model
    always knows the current progress even after context compression.
    """

    # Execution progress
    step: int = 0
    max_steps: int = 50

    # File tracking
    current_file: str | None = None
    modified_files: list[str] = field(default_factory=list)
    read_files: set[str] = field(default_factory=set)

    # Task tracking
    task_goal: str = ""
    completed_subgoals: list[str] = field(default_factory=list)

    # Recent action history (for loop detection)
    recent_actions: list[tuple[str, str]] = field(default_factory=list)
    # Each entry: (tool_name, args_summary)

    def mark_file_read(self, path: str) -> None:
        """Record that a file was read."""
        self.read_files.add(path)

    def mark_file_modified(self, path: str) -> None:
        """Record that a file was modified."""
        if path not in self.modified_files:
            self.modified_files.append(path)
        self.current_file = path

    def add_recent_action(self, tool_name: str, args_summary: str) -> None:
        """Add a recent action for loop detection."""
        self.recent_actions.append((tool_name, args_summary))
        # Keep only last 10
        if len(self.recent_actions) > 10:
            self.recent_actions = self.recent_actions[-10:]

    def add_subgoal(self, subgoal: str) -> None:
        self.completed_subgoals.append(subgoal)

    def to_status_prompt(self) -> str:
        """Generate a status summary string for injection into System Prompt."""
        parts = [f"Step {self.step}/{self.max_steps}"]

        if self.current_file:
            parts.append(f"Editing: {self.current_file}")

        if self.modified_files:
            parts.append(f"Modified: {', '.join(self.modified_files)}")

        if self.read_files:
            recent = sorted(self.read_files)[-5:]
            parts.append(f"Read: {', '.join(recent)}")

        if self.completed_subgoals:
            parts.append(f"Done: {len(self.completed_subgoals)} subgoals")

        return " | ".join(parts)

    def get_loop_risk(self) -> bool:
        """Check if the agent might be stuck in a loop.

        Returns True if the same file has been read/written
        3+ times in the recent actions.
        """
        if len(self.recent_actions) < 3:
            return False

        file_counts: dict[str, int] = {}
        for tool_name, args_summary in self.recent_actions[-5:]:
            if tool_name in ("read_file", "write_file"):
                # Extract path from args summary
                import re
                m = re.search(r'"path":\s*"([^"]+)"', args_summary)
                if m:
                    file_counts[m.group(1)] = file_counts.get(m.group(1), 0) + 1

        return any(count >= 3 for count in file_counts.values())
