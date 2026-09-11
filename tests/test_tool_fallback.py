"""工具失败降级路由测试 — 失败分类 / 降级建议 / 连败阈值触发。"""

from __future__ import annotations

from coder_agent.tool_fallback import (
    ToolFallbackRouter,
    classify_tool_failure,
    fallback_suggestion,
)


class TestClassify:
    def test_not_found(self):
        assert classify_tool_failure("No such file or directory: a.py") == "not_found"

    def test_permission(self):
        assert classify_tool_failure("Permission denied") == "permission"

    def test_timeout(self):
        assert classify_tool_failure("Command timed out after 60s") == "timeout"

    def test_command_not_found(self):
        assert classify_tool_failure("python3: command not found") == "command_not_found"

    def test_command_not_found_windows(self):
        assert classify_tool_failure("'python3' 不是内部或外部命令") == "command_not_found"

    def test_syntax(self):
        assert classify_tool_failure("SyntaxError: expected ':'") == "syntax"

    def test_is_a_directory(self):
        assert classify_tool_failure("read error: is a directory") == "is_a_directory"

    def test_unknown(self):
        assert classify_tool_failure("weird failure") == "unknown"


class TestFallbackSuggestion:
    def test_read_file_not_found(self):
        s = fallback_suggestion("read_file", "not_found")
        assert "list_files" in s or "search_text" in s

    def test_run_command_not_found(self):
        s = fallback_suggestion("run_command", "command_not_found")
        assert "which" in s or "python" in s

    def test_generic_not_found(self):
        s = fallback_suggestion("search_text", "not_found")
        assert "list_files" in s or "search_text" in s

    def test_unknown_category(self):
        s = fallback_suggestion("read_file", "unknown")
        assert s  # 非空


class TestRouter:
    def test_single_failure_no_advice(self):
        r = ToolFallbackRouter(threshold=2)
        r.note("read_file", "No such file: a.py")
        assert r.due_advice("read_file") is None

    def test_double_failure_advice(self):
        r = ToolFallbackRouter(threshold=2)
        r.note("read_file", "No such file: a.py")
        r.note("read_file", "No such file: b.py")
        advice = r.due_advice("read_file")
        assert advice is not None
        assert "read_file" in advice

    def test_advice_only_once(self):
        r = ToolFallbackRouter(threshold=2)
        r.note("read_file", "No such file: a.py")
        r.note("read_file", "No such file: b.py")
        first = r.due_advice("read_file")
        assert first is not None
        # 同一 (工具,类别) 第二次不再唠叨
        assert r.due_advice("read_file") is None

    def test_success_resets_streak(self):
        r = ToolFallbackRouter(threshold=2)
        r.note("read_file", "No such file: a.py")
        r.note("read_file", None)  # 成功清零
        r.note("read_file", "No such file: c.py")
        assert r.due_advice("read_file") is None  # 重新从 1 起

    def test_reset_clears_all(self):
        r = ToolFallbackRouter(threshold=2)
        r.note("read_file", "No such file: a.py")
        r.note("read_file", "No such file: b.py")
        r.reset()
        assert r.due_advice("read_file") is None

    def test_different_categories_separate(self):
        r = ToolFallbackRouter(threshold=2)
        r.note("run_command", "timed out")        # timeout 类
        r.note("run_command", "command not found")  # 不同类，各自计数
        # 各自只 1 次，不达阈值
        assert r.due_advice("run_command") is None

    def test_different_tools_isolated(self):
        r = ToolFallbackRouter(threshold=2)
        r.note("read_file", "No such file: a")
        r.note("write_file", "No such file: b")
        # 各 1 次，互不影响
        assert r.due_advice("read_file") is None
        assert r.due_advice("write_file") is None
