"""Policy gate — programmatic security checks before tool execution.

Inspired by:
- Cline's ALLOW/ASK/DENY三级策略 (tool-policies.ts)
- Continue's pattern-matching permission checker

Key principle: security is enforced at code level, NOT in the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tools.shell import RunCommandTool

if TYPE_CHECKING:
    from .mode import AgentMode


@dataclass
class PolicyResult:
    """Result of a policy check."""

    approved: bool
    reason: str = ""
    needs_log: bool = False


class PolicyGate:
    """Checks whether a tool call should be allowed to execute.

    Three-tier policy:
    - ALLOW:   Directly approved (read-only, low-risk tools)
    - LOG:     Approved but logged (write operations)
    - DENY:    Blocked with a reason
    """

    # Directly allowed tools
    ALLOW_LIST = frozenset({"read_file", "list_files", "search_text"})

    # Tools that are allowed but should be logged
    LOG_LIST = frozenset({"write_file"})

    # Dangerous command substrings (checked inside run_command).
    # Single source of truth: RunCommandTool.DANGEROUS_PATTERNS — the two
    # lists once drifted (PolicyGate lacked sudo rules), letting
    # `sudo cat /etc/shadow` pass the gate layer. Never re-duplicate.
    DANGEROUS_COMMAND_PATTERNS = frozenset(RunCommandTool.DANGEROUS_PATTERNS)

    def check(
        self,
        tool_name: str,
        args: dict,
        mode: "AgentMode" = None,
    ) -> PolicyResult:
        """Check if a tool call is approved.

        Args:
            tool_name: Name of the tool being called
            args: Arguments passed to the tool
            mode: AgentMode affecting policy strictness
        """
        # DRY_RUN mode: allow everything but don't execute writes
        if mode and mode.value == "dry-run":
            return PolicyResult(approved=True, needs_log=True)

        # FULL mode: relax policy denials (path safety still enforced by tools)
        if mode and mode.value == "full":
            if tool_name in self.ALLOW_LIST or tool_name in self.LOG_LIST:
                return PolicyResult(approved=True)
            if tool_name == "run_command":
                cmd = args.get("command", "")
                for pattern in self.DANGEROUS_COMMAND_PATTERNS:
                    if pattern.lower() in cmd.lower():
                        return PolicyResult(
                            approved=False,
                            reason=f"Dangerous command blocked: contains '{pattern}'",
                        )
                return PolicyResult(approved=True, needs_log=True)
            return PolicyResult(approved=True)  # Full mode: allow unknown tools too

        # PLAN mode: read-only, block writes and commands
        if mode and mode.value == "plan":
            if tool_name in self.ALLOW_LIST:
                return PolicyResult(approved=True)
            # Block anything that modifies state
            return PolicyResult(
                approved=False,
                reason=f"Plan mode is read-only: '{tool_name}' is not a read-only operation",
            )

        # GOAL mode (default): normal policy
        if tool_name in self.ALLOW_LIST:
            return PolicyResult(approved=True)

        if tool_name in self.LOG_LIST:
            return PolicyResult(approved=True, needs_log=True)

        if tool_name == "run_command":
            cmd = args.get("command", "")
            for pattern in self.DANGEROUS_COMMAND_PATTERNS:
                if pattern.lower() in cmd.lower():
                    return PolicyResult(
                        approved=False,
                        reason=f"Dangerous command blocked: contains '{pattern}'",
                    )
            return PolicyResult(approved=True, needs_log=True)

        # Unknown tool — deny
        return PolicyResult(
            approved=False,
            reason=f"Unknown or unregistered tool: '{tool_name}'",
        )
