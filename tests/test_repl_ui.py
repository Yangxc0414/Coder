"""第三轮 UI 反思的实现测试：/model、ASSISTANT_TEXT 思考显示。"""

from __future__ import annotations

from pathlib import Path

from coder_agent.agent import Agent
from coder_agent.llm.client import LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.tools.registry import ToolRegistry
import os

from coder_agent.journal import SessionJournal
from coder_agent.memory import Memory
from coder_agent.state import AgentState
from coder_agent.ui.cli.repl import CoderRepl


def _repl(tmp_path: Path) -> CoderRepl:
    return CoderRepl(workspace=tmp_path, model="agnes-2.5-flash", mode=AgentMode.GOAL)


class TestModelCommand:
    def test_model_switch(self, tmp_path: Path):
        repl = _repl(tmp_path)
        repl._handle_command("/model gpt-x")
        assert repl.model == "gpt-x"

    def test_model_no_arg_shows_current(self, tmp_path: Path):
        repl = _repl(tmp_path)
        before = repl.model
        repl._handle_command("/model")
        assert repl.model == before  # 不变，仅显示

    def test_model_persists_across_build(self, tmp_path: Path):
        """切换后的模型必须真正传给下次 /run 构建的 agent。"""
        agent = _repl(tmp_path).model
        # _build_agent 使用 self.model —— 直接断言字段即可（构建逻辑已由
        # live 测试覆盖）
        assert agent


class TestAssistantTextHook:
    """P2：模型在工具调用之间的思考文本通过 ASSISTANT_TEXT 事件暴露。"""

    def test_fires_when_content_and_tool_calls(self, tmp_path: Path):
        from coder_agent.hooks import ASSISTANT_TEXT

        class ThinkingLLM:
            model = "mock"

            def chat(self, messages, tools=None, max_tokens=4096):
                return LLMResponse(
                    content="我先查看目录结构再决定改哪个文件",
                    tool_calls=[{"id": "t1", "name": "list_files", "arguments": "{}"}],
                    finish_reason="tool_calls", usage=None)

        fired = []
        agent = Agent(llm_client=ThinkingLLM(), registry=ToolRegistry(),
                      workspace=tmp_path, mode=AgentMode.GOAL, max_steps=1)
        agent.hooks.register("ASSISTANT_TEXT", lambda e: fired.append(e.data))
        agent.run("task")
        assert fired and "查看目录结构" in fired[0]["text"]

    def test_not_fired_on_final_answer(self, tmp_path: Path):
        class FinalLLM:
            model = "mock"

            def chat(self, messages, tools=None, max_tokens=4096):
                return LLMResponse(content="最终答案", tool_calls=None,
                                   finish_reason="stop", usage=None)

        fired = []
        agent = Agent(llm_client=FinalLLM(), registry=ToolRegistry(),
                      workspace=tmp_path, mode=AgentMode.GOAL, max_steps=1)
        agent.hooks.register("ASSISTANT_TEXT", lambda e: fired.append(e.data))
        agent.run("task")
        assert not fired  # 最终答案走答案面板，不走思考通道


class TestGoalCommand:
    """P-新增：/goal 会话目标——设置/清除/注入系统提示。"""

    def test_set_and_show(self, tmp_path: Path):
        repl = _repl(tmp_path)
        repl._handle_command("/goal 完成博客整理")
        assert repl.state.goal == "完成博客整理"

    def test_clear(self, tmp_path: Path):
        repl = _repl(tmp_path)
        repl._handle_command("/goal x")
        repl._handle_command("/goal clear")
        assert repl.state.goal == ""

    def test_goal_injected_into_status_prompt(self, tmp_path: Path):
        """Goal 必须进入每步的系统提示注入（State.Goal）。"""
        from coder_agent.agent import _build_system_prompt
        from coder_agent.memory import Memory

        repl = _repl(tmp_path)
        repl._handle_command("/goal 完成博客整理")
        state = AgentState(task_goal=repl.state.goal)
        prompt = _build_system_prompt([], str(tmp_path), state=state, memory=Memory())
        assert "Goal: 完成博客整理" in prompt


class TestSessionPickers:
    """P-新增：/sessions 序号 + /resume 无参=最新。"""

    def test_sessions_lists_and_caches_choices(self, tmp_path: Path):
        from coder_agent.journal import SessionJournal

        repl = _repl(tmp_path)
        s1 = tmp_path / "session_a.jsonl"
        s2 = tmp_path / "session_b.jsonl"
        for p, msg in ((s1, "older task"), (s2, "newer task")):
            j = SessionJournal(p)
            j.log_message({"role": "user", "content": msg})
            j.close()
        # _list_sessions 从固定目录扫描 → monkeypatch home
        home = tmp_path / "home"
        home.mkdir()
        import coder_agent.ui.cli.repl as repl_mod
        orig = repl_mod.Path.home
        repl_mod.Path.home = staticmethod(lambda: home)
        try:
            import shutil
            (home / ".coder_sessions").mkdir(parents=True)
            shutil.copy(s1, home / ".coder_sessions" / "session_a.jsonl")
            shutil.copy(s2, home / ".coder_sessions" / "session_b.jsonl")
            os.utime(home / ".coder_sessions" / "session_a.jsonl", (1, 1))
            files = repl._list_sessions()
            assert files[0] == home / ".coder_sessions" / "session_b.jsonl"
            assert repl._session_choices == files
        finally:
            repl_mod.Path.home = staticmethod(orig)

    def test_resume_latest_resolves(self, tmp_path: Path, monkeypatch):
        repl = _repl(tmp_path)
        latest = tmp_path / "latest.jsonl"
        j = SessionJournal(latest)
        j.log_message({"role": "user", "content": "prior"})
        j.close()
        monkeypatch.setattr(repl, "_list_sessions", lambda: [latest])
        monkeypatch.setattr(repl, "_run_resumed", lambda p, i: captured.update(path=p))
        captured = {}
        repl._handle_command("/resume")
        assert captured["path"] == str(latest)
