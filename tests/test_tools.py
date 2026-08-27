"""单元测试：Tool 协议、工具注册、安全检查、路径防护"""

import os
import tempfile
from pathlib import Path

import pytest

from coder_agent.policy import PolicyGate, PolicyResult
from coder_agent.tools.base import ToolResult
from coder_agent.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool
from coder_agent.tools.registry import ToolRegistry
from coder_agent.tools.shell import RunCommandTool


# ── ToolResult ──────────────────────────────────────────────

class TestToolResult:
    def test_success(self):
        r = ToolResult(output="hello")
        assert r.success
        assert r.error is None

    def test_error(self):
        r = ToolResult(error="file not found")
        assert not r.success
        assert r.output == ""

    def test_to_message_content_error(self):
        r = ToolResult(error="permission denied")
        assert r.to_message_content() == "(error) permission denied"

    def test_to_message_content_success(self):
        r = ToolResult(output="done")
        assert r.to_message_content() == "done"


# ── ToolRegistry ────────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        # 用一个 mock tool
        class MockTool:
            name = "test_tool"
            def to_tool_def(self):
                return {"type": "function", "function": {"name": "test_tool"}}
        reg.register(MockTool())  # type: ignore
        assert len(reg) == 1
        assert "test_tool" in reg

    def test_duplicate_register_raises(self):
        reg = ToolRegistry()
        class T1:
            name = "dup"
            def to_tool_def(self):
                return {}
        reg.register(T1())  # type: ignore
        with pytest.raises(ValueError, match="already registered"):
            reg.register(T1())  # type: ignore

    def test_unknown_tool_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="Unknown tool"):
            reg.get("nonexistent")


# ── PolicyGate ──────────────────────────────────────────────

class TestPolicyGate:
    @pytest.fixture
    def gate(self):
        return PolicyGate()

    def test_read_file_allowed(self, gate):
        r = gate.check("read_file", {"path": "a.py"})
        assert r.approved
        assert not r.needs_log

    def test_list_files_allowed(self, gate):
        r = gate.check("list_files", {})
        assert r.approved

    def test_write_file_logged(self, gate):
        r = gate.check("write_file", {"path": "a.py", "content": "x"})
        assert r.approved
        assert r.needs_log

    def test_dangerous_rm_rf_blocked(self, gate):
        r = gate.check("run_command", {"command": "rm -rf /"})
        assert not r.approved
        assert "dangerous" in r.reason.lower() or "blocked" in r.reason.lower()

    def test_safe_echo_allowed(self, gate):
        r = gate.check("run_command", {"command": "echo hello"})
        assert r.approved

    def test_sudo_blocked(self, gate):
        r = gate.check("run_command", {"command": "sudo rm -rf /"})
        assert not r.approved

    def test_unknown_tool_denied(self, gate):
        r = gate.check("teleport", {})
        assert not r.approved


# ── ReadFileTool ────────────────────────────────────────────

class TestReadFileTool:
    @pytest.fixture
    def tmp_workspace(self, tmp_path):
        return tmp_path

    def test_read_existing_file(self, tmp_workspace):
        f = tmp_workspace / "hello.txt"
        f.write_text("world")
        tool = ReadFileTool(tmp_workspace)
        result = tool.execute({"path": "hello.txt"})
        assert result.success
        assert result.output == "world"

    def test_read_missing_file(self, tmp_workspace):
        tool = ReadFileTool(tmp_workspace)
        result = tool.execute({"path": "nope.txt"})
        assert not result.success
        assert "not found" in result.error.lower()

    def test_path_traversal_blocked(self, tmp_workspace):
        tool = ReadFileTool(tmp_workspace)
        result = tool.execute({"path": "../outside.txt"})
        assert not result.success
        assert "escapes" in result.error.lower() or "permission" in result.error.lower()

    def test_absolute_path_still_blocked(self, tmp_workspace):
        tool = ReadFileTool(tmp_workspace)
        # 尝试用绝对路径绕过
        result = tool.execute({"path": str(tmp_workspace.parent / "secret.txt")})
        assert not result.success


# ── WriteFileTool ───────────────────────────────────────────

class TestWriteFileTool:
    def test_write_new_file(self, tmp_path):
        tool = WriteFileTool(tmp_path)
        result = tool.execute({"path": "new.txt", "content": "hi"})
        assert result.success
        assert (tmp_path / "new.txt").read_text() == "hi"

    def test_write_nested_creates_dirs(self, tmp_path):
        tool = WriteFileTool(tmp_path)
        result = tool.execute({"path": "a/b/c.txt", "content": "deep"})
        assert result.success
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"

    def test_write_path_traversal_blocked(self, tmp_path):
        tool = WriteFileTool(tmp_path)
        result = tool.execute({"path": "../escape.txt", "content": "x"})
        assert not result.success


# ── ListFilesTool ───────────────────────────────────────────

class TestListFilesTool:
    def test_list_empty(self, tmp_path):
        tool = ListFilesTool(tmp_path)
        result = tool.execute({})
        assert result.success
        assert "(directory is empty)" in result.output

    def test_list_with_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        tool = ListFilesTool(tmp_path)
        result = tool.execute({})
        assert result.success
        assert "a.py" in result.output
        assert "b.py" in result.output

    def test_list_subdirectory(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("x")
        tool = ListFilesTool(tmp_path)
        result = tool.execute({"path": "src"})
        assert result.success
        assert "main.py" in result.output


# ── RunCommandTool ──────────────────────────────────────────

class TestRunCommandTool:
    def test_dangerous_command_blocked(self, tmp_path):
        tool = RunCommandTool(tmp_path)
        result = tool.execute({"command": "rm -rf /"})
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_safe_echo(self, tmp_path):
        tool = RunCommandTool(tmp_path)
        result = tool.execute({"command": "echo hello"})
        assert result.success
        assert "hello" in result.output

    def test_timeout(self, tmp_path):
        tool = RunCommandTool(tmp_path)
        result = tool.execute({"command": "sleep 5", "timeout": 1})
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_cwd_isolation(self, tmp_path):
        tool = RunCommandTool(tmp_path)
        result = tool.execute({"command": "pwd"})
        assert result.success
        # 输出应包含 workspace 路径
        assert str(tmp_path) in result.output or str(tmp_path).replace("\\", "/") in result.output

    def test_env_isolation(self, tmp_path):
        """验证 shell 执行不暴露敏感环境变量"""
        import os
        os.environ["SECRET_API_KEY"] = "should_not_appear"
        tool = RunCommandTool(tmp_path)
        result = tool.execute({"command": "env"})
        assert result.success
        assert "SECRET_API_KEY" not in result.output
        del os.environ["SECRET_API_KEY"]
