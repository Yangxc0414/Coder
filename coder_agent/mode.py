"""Agent modes — control behavior during execution.

Modes:
- GOAL:      Default mode. Agent plans and executes to complete the task.
- PLAN:      Read-only analysis mode. Agent explores codebase and outputs a plan.
- DRY_RUN:   Show what would be changed without actually modifying files.
- FULL:      unrestricted mode (bypasses PolicyGate deny list, still checks path safety).
"""

from __future__ import annotations
from enum import Enum


class AgentMode(str, Enum):
    GOAL = "goal"       # 正常模式：规划+执行
    PLAN = "plan"       # 计划模式：只分析不修改
    DRY_RUN = "dry-run" # 干跑模式：显示变更但不写入
    FULL = "full"       # 完全模式：宽松策略检查


# Human-readable descriptions for CLI help
MODE_DESCRIPTIONS = {
    AgentMode.GOAL: "Normal mode - agent plans and executes to complete the task",
    AgentMode.PLAN: "Plan only - analyze codebase and output execution plan (read-only)",
    AgentMode.DRY_RUN: "Dry run - show what would change without modifying files",
    AgentMode.FULL: "Full access - bypass policy denials (path safety still enforced)",
}
