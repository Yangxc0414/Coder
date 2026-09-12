"""Framework wiring tests — verify the pieces actually connect.

Covers the integration seams that unit tests miss:
- create_default_registry with/without extensions
- ExtensionResult <-> ToolResult interface compatibility
- Agent per-instance max_steps (no global mutation)
- CLI entrypoint wiring (verifier, context budget)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coder_agent.extensions.base import ExtensionResult, SubagentDefinition, SubagentRequest
from coder_agent.llm.client import LLMClient, LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.tools.registry import ToolRegistry, create_default_registry


class TestExtensionResultInterface:
    """ExtensionResult must be drop-in compatible with ToolResult."""

    def test_success_property(self):
        assert ExtensionResult(output="ok").success is True
        assert ExtensionResult(error="bad").success is False

    def test_to_message_content_output(self):
        r = ExtensionResult(output="hello")
        assert r.to_message_content() == "hello"

    def test_to_message_content_error(self):
        r = ExtensionResult(error="boom")
        assert r.to_message_content() == "(error) boom"

    def test_to_message_content_empty(self):
        assert ExtensionResult().to_message_content() == "(no output)"


class TestDefaultRegistry:
    """create_default_registry wires core tools + extensions."""

    def test_core_tools_present(self, tmp_path: Path):
        reg = create_default_registry(tmp_path, AgentMode.GOAL, with_extensions=False)
        names = reg.list_names()
        for expected in ("read_file", "write_file", "list_files", "search_text", "run_command"):
            assert expected in names

    def test_no_extensions_when_disabled(self, tmp_path: Path):
        reg = create_default_registry(tmp_path, AgentMode.GOAL, with_extensions=False)
        assert len(reg.list_names()) == 5

    def test_mcp_tools_registered(self, tmp_path: Path):
        reg = create_default_registry(tmp_path, AgentMode.GOAL, with_extensions=True)
        names = reg.list_names()
        assert "mcp_git_git_log" in names
        assert "mcp_system_system_info" in names

    def test_skills_registered(self, tmp_path: Path):
        reg = create_default_registry(tmp_path, AgentMode.GOAL, with_extensions=True)
        names = reg.list_names()
        assert "skill_code_review" in names
        assert "skill_test_writer" in names

    def test_extension_tool_count(self, tmp_path: Path):
        reg = create_default_registry(tmp_path, AgentMode.GOAL, with_extensions=True)
        # 5 core + 2 install + 5 mcp + 17 skills = 29
        # （install_skill / install_mcp 是 agent 自主下载扩展的元工具；
        #  若 ~/.coder_extensions/ 有已装扩展还会再加，故断言下限）
        assert len(reg.list_names()) >= 29
        assert "install_skill" in reg.list_names()
        assert "install_mcp" in reg.list_names()

    def test_skill_executes_through_registry(self, tmp_path: Path):
        reg = create_default_registry(tmp_path, AgentMode.GOAL)
        skill = reg.get("skill_code_review")
        result = skill.execute({"target": "foo.py"})
        assert result.success
        assert "code_review" in result.to_message_content()

    def test_dry_run_flag_propagates(self, tmp_path: Path):
        reg = create_default_registry(tmp_path, AgentMode.DRY_RUN, with_extensions=False)
        writer = reg.get("write_file")
        assert writer.dry_run is True

    def test_full_mode_flag_propagates(self, tmp_path: Path):
        reg = create_default_registry(tmp_path, AgentMode.GOAL, with_extensions=False)
        writer = reg.get("write_file")
        assert writer.dry_run is False


class TestAgentMaxSteps:
    """Agent honors per-instance max_steps without touching the module global."""

    def _looping_llm(self):
        """Mock LLM that always requests a tool call, never finishes."""

        class LoopingLLM:
            model = "mock"

            def chat(self, messages, tools=None, max_tokens=4096):
                return LLMResponse(
                    content="thinking...",
                    tool_calls=[{"id": "t1", "name": "list_files", "arguments": "{}"}],
                    finish_reason="tool_calls",
                    usage=None,
                )

        return LoopingLLM()

    def test_max_steps_respected(self, tmp_path: Path):
        from coder_agent.agent import Agent

        reg = create_default_registry(tmp_path, AgentMode.GOAL, with_extensions=False)
        agent = Agent(
            llm_client=self._looping_llm(),
            registry=reg,
            workspace=tmp_path,
            mode=AgentMode.GOAL,
            max_steps=3,
        )
        answer = agent.run("loop forever")
        assert "maximum steps" in answer.lower()
        assert agent.state.step == 3

    def test_global_max_steps_not_mutated(self, tmp_path: Path):
        from coder_agent.agent import Agent, MAX_STEPS

        reg = create_default_registry(tmp_path, AgentMode.GOAL, with_extensions=False)
        agent = Agent(
            llm_client=self._looping_llm(),
            registry=reg,
            workspace=tmp_path,
            mode=AgentMode.GOAL,
            max_steps=2,
        )
        agent.run("loop forever")
        import coder_agent.agent as agent_module
        assert agent_module.MAX_STEPS == MAX_STEPS == 50

    def test_default_max_steps(self, tmp_path: Path):
        from coder_agent.agent import Agent, MAX_STEPS

        agent = Agent(
            llm_client=self._looping_llm(),
            registry=ToolRegistry(),
            workspace=tmp_path,
        )
        assert agent._max_steps == MAX_STEPS


class TestSubagentRunnerWiring:
    """SubagentRunner uses per-instance limits and public LLM attributes."""

    def test_unknown_type_is_error(self, tmp_path: Path):
        from coder_agent.agent import Agent
        from coder_agent.extensions.base import SubagentRunner

        agent = Agent(
            llm_client=LLMClient(model="mock"),
            registry=ToolRegistry(),
            workspace=tmp_path,
        )
        runner = SubagentRunner(parent_agent=agent)
        result = runner.run(SubagentRequest(prompt="x", subagent_type="nonexistent"))
        assert result.is_error
        assert "nonexistent" in result.final_output

    def test_builtin_types_registered(self, tmp_path: Path):
        from coder_agent.agent import Agent

        agent = Agent(
            llm_client=LLMClient(model="mock"),
            registry=ToolRegistry(),
            workspace=tmp_path,
        )
        assert set(agent.subagent_runner._definitions.keys()) == {
            "researcher", "test_specialist", "security_scanner", "documenter",
        }

    def test_researcher_has_readonly_toolset(self, tmp_path: Path):
        from coder_agent.agent import Agent

        reg = create_default_registry(tmp_path, AgentMode.GOAL, with_extensions=False)
        agent = Agent(
            llm_client=LLMClient(model="mock"),
            registry=reg,
            workspace=tmp_path,
        )
        defn = agent.subagent_runner._definitions["researcher"]
        filtered = [t for t in reg.list_names() if t in defn.tools]
        assert "write_file" not in filtered
        assert "run_command" not in filtered
        assert set(filtered) == {"read_file", "list_files", "search_text"}


class TestLLMClientAttributes:
    """LLMClient exposes base_url publicly for subagent construction."""

    def test_base_url_from_arg(self):
        llm = LLMClient(model="m", base_url="https://example.com/v1")
        assert llm.base_url == "https://example.com/v1"

    def test_base_url_from_env(self, monkeypatch):
        import coder_agent.llm.client as client_mod
        monkeypatch.setattr(client_mod, "_USER_CFG", {})  # 隔离用户配置
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
        llm = LLMClient(model="m")
        assert llm.base_url == "https://env.example.com/v1"
