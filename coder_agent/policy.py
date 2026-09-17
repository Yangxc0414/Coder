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
    """Result of a policy check.

    SOTA alignment (OneCode guard ``to_tool_error``): a denial is not a bare
    error string — it carries WHY (reason) and a concrete alternative
    (suggestion), so the model can route around the denial instead of
    retrying into the same wall.
    """

    approved: bool
    reason: str = ""
    needs_log: bool = False
    suggestion: str = ""


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
    # memory: writes only to .coder_memory.md (own bookkeeping)
    # task: delegation — severity depends on the child subagent; allowed
    #       with logging, and PLAN mode still blocks it (children may write)
    # install_skill/install_mcp: agent 自主下载扩展（git clone 白名单域名到
    #       ~/.coder_extensions/，不进工作区/git）；带日志审计，PLAN 模式只读会拦截
    LOG_LIST = frozenset({
        "write_file", "memory", "task",
        "install_skill", "install_mcp",
    })

    # Dangerous command substrings (checked inside run_command).
    # Single source of truth: RunCommandTool.DANGEROUS_PATTERNS — the two
    # lists once drifted (PolicyGate lacked sudo rules), letting
    # `sudo cat /etc/shadow` pass the gate layer. Never re-duplicate.
    DANGEROUS_COMMAND_PATTERNS = frozenset(RunCommandTool.DANGEROUS_PATTERNS)

    # 结构化拒绝载荷（SOTA: OneCode guard to_tool_error）：每类拒绝自带
    # 一条可执行的替代路径建议，模型拿到后能绕行而不是反复撞墙重试。
    SUGGESTION_DANGEROUS_CMD = (
        "Pick a non-destructive equivalent (read instead of delete, avoid "
        "sudo/format/force flags). If the destructive action is truly "
        "required, stop and ask the user first — do not retry the blocked "
        "command with different casing or quoting."
    )
    SUGGESTION_PLAN_MODE = (
        "Plan mode is read-only: investigate with read_file / search_text / "
        "list_files and end with the plan as your final answer. The user "
        "can re-run in goal/full mode to apply the change."
    )
    SUGGESTION_UNKNOWN_TOOL = (
        "This tool name is not registered. Use one of the available tools "
        "(see the tool list in the system prompt); do not invent tool names."
    )

    def _deny_dangerous(self, pattern: str) -> "PolicyResult":
        return PolicyResult(
            approved=False,
            reason=f"Dangerous command blocked: contains '{pattern}'",
            suggestion=self.SUGGESTION_DANGEROUS_CMD,
        )

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
                        return self._deny_dangerous(pattern)
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
                suggestion=self.SUGGESTION_PLAN_MODE,
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
                    return self._deny_dangerous(pattern)
            return PolicyResult(approved=True, needs_log=True)

        # Unknown tool — deny
        return PolicyResult(
            approved=False,
            reason=f"Unknown or unregistered tool: '{tool_name}'",
            suggestion=self.SUGGESTION_UNKNOWN_TOOL,
        )
