"""Real-run improvement tests — findings from the Todo-App self-test
(50 steps / 103 tool calls / 377K tokens; 57 reads where 6 sufficed,
13 identical command failures without strategy change)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from coder_agent.agent import Agent
from coder_agent.llm.client import LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.tools.filesystem import ReadFileTool, WriteFileTool
from coder_agent.tools.registry import ToolRegistry
from coder_agent.tools.shell import RunCommandTool


class TestReadFileCache:
    """read_file unchanged-file short-circuit (47 of 57 reads were redundant)."""

    def _registry(self, tmp_path: Path) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(ReadFileTool(tmp_path))
        reg.register(WriteFileTool(tmp_path))
        return reg

    def test_second_read_returns_unchanged_notice(self, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("hello world", encoding="utf-8")
        tool = ReadFileTool(tmp_path)
        first = tool.execute({"path": "a.txt"})
        assert first.output == "hello world"
        second = tool.execute({"path": "a.txt"})
        assert second.success
        assert "[unchanged]" in second.output
        assert "hello world"[:12] in second.output  # preview present
        assert len(second.output) < 600  # short, not full content

    def test_full_read_after_modification(self, tmp_path: Path):
        """mtime changes (e.g. the agent rewrote the file) → full content."""
        tool = ReadFileTool(tmp_path)
        f = tmp_path / "a.txt"
        f.write_text("v1", encoding="utf-8")
        tool.execute({"path": "a.txt"})
        # rewrite via the tool (like the agent would) — mtime changes
        time.sleep(0.01)  # Windows mtime granularity
        (tmp_path / "a.txt").write_text("v2 with completely different content", encoding="utf-8")
        second = tool.execute({"path": "a.txt"})
        assert "[unchanged]" not in second.output
        assert second.output == "v2 with completely different content"

    def test_deleted_file_invalidates_cache(self, tmp_path: Path):
        tool = ReadFileTool(tmp_path)
        f = tmp_path / "gone.txt"
        f.write_text("x", encoding="utf-8")
        tool.execute({"path": "gone.txt"})
        f.unlink()
        result = tool.execute({"path": "gone.txt"})
        assert not result.success
        assert "File not found" in result.error

    def test_cache_size_bounded(self, tmp_path: Path):
        tool = ReadFileTool(tmp_path)
        for i in range(80):
            p = tmp_path / f"f_{i}.txt"
            p.write_text(str(i), encoding="utf-8")
            tool.execute({"path": f"f_{i}.txt"})
        assert len(tool._cache) <= ReadFileTool.CACHE_LIMIT


class TestCommandFailureHints:
    """Adaptive strategy hint after 3 consecutive command failures
    (real run: 13 identical failures, model never switched approach)."""

    class ScriptedLLM:
        model = "mock"

        def __init__(self, commands: list[str]):
            self.commands = list(commands)
            self.calls: list[list[dict]] = []

        def chat(self, messages, tools=None, max_tokens=4096):
            self.calls.append([dict(m) for m in messages])
            if self.commands:
                cmd = self.commands.pop(0)
                return LLMResponse(
                    content="",
                    tool_calls=[{"id": "t1", "name": "run_command",
                                 "arguments": '{"command": "%s"}' % cmd}],
                    finish_reason="tool_calls", usage=None,
                )
            return LLMResponse(content="gave up properly", tool_calls=None,
                               finish_reason="stop", usage=None)

    def _agent(self, tmp_path: Path, llm) -> Agent:
        reg = ToolRegistry()
        reg.register(RunCommandTool(tmp_path))
        return Agent(llm_client=llm, registry=reg, workspace=tmp_path,
                     mode=AgentMode.GOAL, max_steps=10)

    def test_hint_injected_after_three_failures(self, tmp_path: Path):
        llm = self.ScriptedLLM([
            "definitely_missing_cmd_xyz",
            "definitely_missing_cmd_xyz",
            "definitely_missing_cmd_xyz",
            "definitely_missing_cmd_xyz",
        ])
        agent = self._agent(tmp_path, llm)
        agent.run("try commands")
        hint_messages = [m for m in agent.messages
                         if m.get("role") == "user"
                         and "命令已连续失败" in (m.get("content") or "")]
        assert len(hint_messages) == 1  # 命令类走既有策略提示，注入恰好一次
        if sys.platform == "win32":
            assert "python3" in hint_messages[0]["content"]

    def test_success_resets_streak(self, tmp_path: Path):
        """2 failures → 1 success → hint must NOT fire (streak reset)."""
        llm = self.ScriptedLLM([
            "definitely_missing_cmd_xyz",
            "definitely_missing_cmd_xyz",
            "echo recovered",
        ])
        agent = self._agent(tmp_path, llm)
        agent.run("mixed")
        assert agent._cmd_fail_streak == 0  # last command succeeded
        assert not any("连续失败" in (m.get("content") or "") for m in agent.messages)

    def test_no_hint_without_failures(self, tmp_path: Path):
        from coder_agent.llm.client import LLMResponse as R

        class CleanLLM:
            model = "mock"
            def __init__(self): self.calls = []
            def chat(self, messages, tools=None, max_tokens=4096):
                self.calls.append(list(messages))
                return R(content="done", tool_calls=None, finish_reason="stop", usage=None)

        llm = CleanLLM()
        agent = self._agent(tmp_path, llm)
        agent.run("simple")
        assert not any("连续失败" in (m.get("content") or "") for m in agent.messages)


class TestBackgroundCommand:
    """background=true — blocking services no longer die at tool timeout
    (real run: the agent could never keep http.server alive)."""

    def test_background_returns_immediately_with_log(self, tmp_path: Path):
        tool = RunCommandTool(tmp_path)
        t0 = time.time()
        result = tool.execute({
            "command": f'"{sys.executable}" -c "import time; time.sleep(1.2); print(\'BG_DONE_OK\')"',
            "background": True,
        })
        elapsed = time.time() - t0
        assert result.success
        assert "PID" in result.output
        assert "日志文件" in result.output
        assert elapsed < 1.0  # did not block for the 1.2s sleep

        # log file eventually carries the output
        log_rel = result.output.split("日志文件: ")[1].split("\n")[0].strip()
        log_path = tmp_path / log_rel
        deadline = time.time() + 5
        while time.time() < deadline:
            if "BG_DONE_OK" in log_path.read_text(encoding="utf-8"):
                break
            time.sleep(0.2)
        assert "BG_DONE_OK" in log_path.read_text(encoding="utf-8")

    def test_dangerous_command_blocked_even_in_background(self, tmp_path: Path):
        tool = RunCommandTool(tmp_path)
        result = tool.execute({"command": "rm -rf /", "background": True})
        assert not result.success
        assert "blocked" in result.error

    def test_foreground_still_default(self, tmp_path: Path):
        tool = RunCommandTool(tmp_path)
        result = tool.execute({"command": "echo hello_fg"})
        assert result.success
        assert "hello_fg" in result.output


class TestSafeEnv:
    """Root cause of the 13 failed commands in the Todo self-test:
    the old whitelist stripped SYSTEMROOT → every spawned Python died at
    startup with _Py_HashRandomization_Init. New strategy: blacklist
    credential-looking vars, keep the OS essentials."""

    def test_secrets_filtered(self):
        import os
        env = RunCommandTool._safe_env()
        assert not any("KEY" in k.upper() or "TOKEN" in k.upper() for k in env)

    def test_system_essentials_kept(self):
        import os
        env = RunCommandTool._safe_env()
        for essential in ("SYSTEMROOT", "PATH"):
            if essential in os.environ:
                assert essential in env, f"{essential} stripped — Python init will fail"

    def test_spawned_python_actually_works(self, tmp_path: Path):
        """The regression that mattered: python subprocess must survive."""
        tool = RunCommandTool(tmp_path)
        result = tool.execute({
            "command": f'"{sys.executable}" -c "print(\'PY_ALIVE\')"',
        })
        assert result.success, result.error
        assert "PY_ALIVE" in result.output
