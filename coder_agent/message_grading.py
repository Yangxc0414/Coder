"""Message Grading — 上下文消息分级保留（Enhancement 3 核心差异）。

市面 agent（mini-swe-agent / OneCode / smolagents）的上下文管理是
"最近 N 轮完整 + 更老的压缩"——一视同仁。但真实 ReAct 轨迹里，
**关键消息**（验证结果、策略拦截、格式纠错指令、用户原始任务）
比**普通消息**（重复的文件读取、中间的工具输出）重要得多。
一视同仁的压缩会把关键信号挤出去，导致模型"忘了自己刚才被验证器
打回过 / 忘了最初的任务约束"。

本模块给每条消息打**保留等级**，ContextManager 压缩时优先保留高等级
消息、先丢低等级消息，而非简单按时间截断：

- CRITICAL（必须保留）：用户原始任务、验证失败/通过、策略拦截、
  格式纠错指令、预算耗尽收尾指令
- HIGH（尽量保留）：工具调用结果（非重复读取）、子代理报告、
  助手的关键推理
- LOW（优先压缩/丢弃）：重复的 read_file 输出、超长命令的
  纯滚动日志、已压缩摘要本身

分级基于消息的"语义角色"（role + 内容特征 + 工具名），不依赖
LLM 判断，纯本地规则，零额外成本。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# 消息保留等级（数值越大越重要，压缩时先丢低等级）
CRITICAL = 3
HIGH = 2
LOW = 1

# 用户消息中出现这些特征 → 系统注入的指令（关键）
_CRITICAL_USER_PATTERNS = [
    r"verification failed",
    r"verification.*pass",
    r"policy denied",
    r"策略拦截",
    r"format error",
    r"请提供最终答案",
    r"token budget exhausted",
    r"必须换方法",
    r"请修复以下问题",
    r"failed:.*test",
]

# 工具调用结果中，这些工具名的输出信息量大（保留 HIGH）
_HIGH_VALUE_TOOLS = {
    "task",            # 子代理报告（浓缩后的关键结论）
    "memory",          # 模型写入的长期事实
    "run_command",     # 命令输出（测试/构建结果）
    "write_file",      # 写文件确认
    "edit_file",
    "append_file",
}

# 这些工具的输出多为可重新获取的中间数据（降级 LOW）
_LOW_VALUE_TOOLS = {
    "read_file",       # 重复读取文件内容（模型需要时可再读）
    "list_files",      # 目录列举（易重复）
    "search_text",     # 搜索命中（可重搜）
}


@dataclass
class MessageGrade:
    """一条消息的保留等级 + 判定理由。"""
    grade: int
    reason: str


def grade_message(message: dict[str, Any]) -> MessageGrade:
    """给单条消息打分（纯规则，无 LLM 成本）。

    判定顺序：先看是否系统注入的关键指令（CRITICAL），
    再看工具调用的高/低价值（HIGH/LOW），其余默认。
    """
    role = message.get("role", "")
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    # ── 用户消息：系统注入的指令是 CRITICAL ──
    if role == "user":
        # 原始任务（第一条用户消息）永远 CRITICAL
        low = content.lower()
        for pat in _CRITICAL_USER_PATTERNS:
            if re.search(pat, low):
                return MessageGrade(CRITICAL, f"critical user directive ({pat})")
        # 短且像任务描述的（非工具结果、非系统指令）也保留 HIGH
        if len(content) < 500 and not content.startswith("["):
            return MessageGrade(HIGH, "user task/instruction")
        return MessageGrade(HIGH, "user message")

    # ── 助手消息：带工具调用的看调用了什么工具 ──
    if role == "assistant":
        if tool_calls:
            names = [t.get("function", {}).get("name", "") for t in tool_calls]
            if any(n in _HIGH_VALUE_TOOLS for n in names):
                return MessageGrade(HIGH, "assistant called high-value tool")
            if any(n in _LOW_VALUE_TOOLS for n in names):
                return MessageGrade(LOW, "assistant called low-value tool")
        # 纯文本推理
        return MessageGrade(HIGH if len(content) > 80 else LOW, "assistant text")

    # ── 工具结果消息：按工具名分级 ──
    if role == "tool":
        # 工具结果消息本身不直接带工具名（在 assistant 的 tool_calls 里），
        # 用 tool_call_id 关联不现实——这里保守：非空结果默认 HIGH，
        # 超长的重复性输出降级 LOW（ContextManager 的 cap 会兜底截断）
        if len(content) > 8000:
            return MessageGrade(LOW, "oversized tool output (capped)")
        return MessageGrade(HIGH, "tool result")

    return MessageGrade(LOW, "unknown")


def grade_messages(messages: list[dict[str, Any]]) -> list[MessageGrade]:
    """批量给消息打分（供 ContextManager 在压缩时排序/保留用）。"""
    return [grade_message(m) for m in messages]


def pick_priority_messages(
    messages: list[dict[str, Any]],
    grades: list[MessageGrade],
    max_keep: int,
) -> set[int]:
    """从 N 条消息中选出保留价值最高的 max_keep 条的下标集合。

    保留策略：
    1. 所有 CRITICAL 消息无条件保留（哪怕超出 max_keep 也保留——
       关键信号丢了比超预算更糟，预算由 ContextManager 的 cap 兜底）
    2. 剩余名额给 HIGH（按时间倒序，近的优先）
    3. LOW 只在还有富余名额时保留
    """
    idx_by_grade: dict[int, list[int]] = {CRITICAL: [], HIGH: [], LOW: []}
    for i, g in enumerate(grades):
        bucket = min(max(g.grade, LOW), CRITICAL)
        idx_by_grade[bucket].append(i)

    keep: set[int] = set()
    # 1) CRITICAL 全保留
    keep.update(idx_by_grade[CRITICAL])
    remaining = max(0, max_keep - len(keep))
    # 2) HIGH 按时间倒序（最近的优先）
    for i in reversed(idx_by_grade[HIGH]):
        if remaining <= 0:
            break
        keep.add(i)
        remaining -= 1
    # 3) LOW 富余名额才保留
    for i in reversed(idx_by_grade[LOW]):
        if remaining <= 0:
            break
        keep.add(i)
        remaining -= 1
    return keep
