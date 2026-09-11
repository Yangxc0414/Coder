"""消息分级保留测试 — 验证 ContextManager 压缩时关键消息优先存活。"""

from __future__ import annotations

from coder_agent.context import ContextManager
from coder_agent.message_grading import (
    CRITICAL,
    HIGH,
    LOW,
    grade_message,
    grade_messages,
    pick_priority_messages,
)


def _tool_msg(tool_name: str, call_id: str, content: str) -> dict:
    """构造一对 assistant tool_calls + tool 结果（ReAct 轨迹的真实形态）。"""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }


class TestGradeMessage:
    def test_original_user_task_high(self):
        g = grade_message({"role": "user", "content": "修复 calculator.py 的 bug"})
        assert g.grade == HIGH

    def test_verification_failure_critical(self):
        g = grade_message({"role": "user", "content": "Verification failed:\nFAIL: 2 NEW failing test(s)"})
        assert g.grade == CRITICAL

    def test_policy_denial_critical(self):
        g = grade_message({"role": "user", "content": "Policy denied: command blocked"})
        assert g.grade == CRITICAL

    def test_token_budget_critical(self):
        g = grade_message({"role": "user", "content": "Token budget exhausted. Stop calling tools"})
        assert g.grade == CRITICAL

    def test_high_value_tool_high(self):
        g = grade_message({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "task"}}],
        })
        assert g.grade == HIGH

    def test_low_value_tool_low(self):
        g = grade_message({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read_file"}}],
        })
        assert g.grade == LOW

    def test_oversized_tool_output_low(self):
        g = grade_message({"role": "tool", "content": "x" * 9000})
        assert g.grade == LOW

    def test_normal_tool_result_high(self):
        g = grade_message({"role": "tool", "content": "tests passed, 3/3"})
        assert g.grade == HIGH


class TestPickPriority:
    def test_critical_always_kept(self):
        msgs = [
            {"role": "user", "content": "short low msg"},  # HIGH (user)
            {"role": "user", "content": "Verification failed: x"},  # CRITICAL
            {"role": "tool", "content": "y" * 9000},  # LOW
        ]
        grades = grade_messages(msgs)
        keep = pick_priority_messages(msgs, grades, max_keep=1)
        # 哪怕 max_keep=1，CRITICAL 那条也必须在保留集里
        assert 1 in keep

    def test_high_before_low_by_recency(self):
        msgs = [
            {"role": "tool", "content": "a" * 9000},  # LOW (0)
            {"role": "tool", "content": "ok"},         # HIGH (1)
            {"role": "tool", "content": "b" * 9000},   # LOW (2)
        ]
        grades = grade_messages(msgs)
        keep = pick_priority_messages(msgs, grades, max_keep=1)
        assert keep == {1}  # HIGH 优先于 LOW

    def test_all_low_fills_by_recency(self):
        msgs = [
            {"role": "tool", "content": "old" + "a" * 9000},  # LOW
            {"role": "tool", "content": "new" + "b" * 9000},  # LOW
        ]
        grades = grade_messages(msgs)
        keep = pick_priority_messages(msgs, grades, max_keep=1)
        assert keep == {1}  # 最近的 LOW 优先


class TestContextGradingIntegration:
    def test_critical_survives_compression(self):
        """验证失败指令在压缩后仍保留（市面 agent 会把它挤出去）。"""
        cm = ContextManager(max_tokens=1500, keep_rounds=2, model="gpt-4o")
        # 构造 4 轮：轮1是普通工具读取(LOW)，轮2是验证失败(CRITICAL)，
        # 轮3-4 是最近轮（完整保留）
        msgs = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}], "content": ""},
            {"role": "tool", "tool_call_id": "t1", "content": "file contents" + "x" * 5000},
            # 验证失败指令（CRITICAL）
            {"role": "user", "content": "Verification failed: 1 NEW failing test. Please fix."},
            {"role": "assistant", "tool_calls": [{"function": {"name": "edit_file"}}], "content": ""},
            {"role": "tool", "tool_call_id": "t2", "content": "edited ok"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "done"},
        ]
        result = cm.build_messages("system", msgs)
        # 压缩发生后（compress_count > 0）
        assert cm.last_compression is not None
        assert cm.last_compression["compressed"] > 0
        # 验证失败指令必须存活（在结果里）
        text = "".join(str(m.get("content") or "") for m in result)
        assert "Verification failed" in text, "关键验证指令被压缩挤出了！"

    def test_grading_disabled_keeps_all_recent(self):
        """关闭分级时行为与旧版一致（最近轮次全保留，无"已省略"占位）。"""
        cm = ContextManager(max_tokens=1500, keep_rounds=1, model="gpt-4o",
                           grade_messages=False)
        msgs = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}], "content": ""},
            {"role": "tool", "tool_call_id": "t1", "content": "file" + "x" * 5000},
            {"role": "user", "content": "Verification failed: fix"},
            {"role": "assistant", "content": "done"},
        ]
        result = cm.build_messages("system", msgs)
        # keep_rounds=1 → 旧轮次被压缩；分级关闭时最近轮次原样保留
        text = "".join(str(m.get("content") or "") for m in result)
        assert "已按重要性省略" not in text  # 无分级占位摘要
        assert "Verification failed" in text  # 最近轮次完整保留
