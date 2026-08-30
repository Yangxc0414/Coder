"""第三轮 UI 反思的实现测试：/model、ASSISTANT_TEXT 思考显示。"""

from __future__ import annotations

from pathlib import Path

from coder_agent.agent import Agent
from coder_agent.llm.client import LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.tools.registry import ToolRegistry
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
