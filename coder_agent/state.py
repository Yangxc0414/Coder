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

    # Full outcomes (tool, args, result) for repetition-based stuck detection
    recent_outcomes: list[tuple[str, str, str]] = field(default_factory=list)

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

    def add_tool_outcome(
        self, tool_name: str, args_summary: str, result_summary: str
    ) -> None:
        """Record a full (tool, args, result) outcome for stuck detection.

        The result component is what separates a productive iteration
        (same command, different output — the workspace changed) from a
        genuine stuck loop (same command, same failure, over and over).
        """
        self.recent_outcomes.append((tool_name, args_summary, result_summary))
        if len(self.recent_outcomes) > 12:
            del self.recent_outcomes[:-6]
        if len(self.recent_actions) > 10:
            self.recent_actions = self.recent_actions[-10:]

    def add_subgoal(self, subgoal: str) -> None:
        self.completed_subgoals.append(subgoal)

    def to_status_prompt(self) -> str:
        """Generate a status summary string for injection into System Prompt."""
        parts = [f"Step {self.step}/{self.max_steps}"]

        if self.task_goal:
            parts.append(f"Goal: {self.task_goal[:80]}")

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

    def get_repetition_risk(self) -> bool:
        """Stuck detection on (tool, args, RESULT) repetition — openhands
        stuck_detection's insight: a loop is a repeated action AND repeated
        observation. The same command with different results is productive
        iteration (the workspace changed); the same command with the same
        failure is being stuck. File-based churn detection (get_loop_risk)
        misses loops that vary the file.

        Uses recent_outcomes when available (agent records them after each
        execution); falls back to recent_actions (args-only) otherwise.

        Returns True if the same outcome appears 3+ times in the window,
        or the last 3 outcomes are identical.
        """
        if self.recent_outcomes:
            window = self.recent_outcomes[-6:]
        else:
            if len(self.recent_actions) < 3:
                return False
            window = [(t, a, "") for t, a in self.recent_actions[-6:]]

        if len(window) < 3:
            return False

        # exact consecutive repeat: last 3 outcomes identical
        if window[-1] == window[-2] == window[-3]:
            return True

        counts: dict[tuple, int] = {}
        for outcome in window:
            counts[outcome] = counts.get(outcome, 0) + 1
        return any(c >= 3 for c in counts.values())
