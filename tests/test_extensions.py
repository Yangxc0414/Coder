"""Tests for the extensions system (Skills, MCP tools, Subagents)."""

from __future__ import annotations

import pytest

from coder_agent.extensions import (
    ExtensionResult,
    Skill,
    SubagentDefinition,
    SubagentRequest,
    SubagentResult,
)
from coder_agent.extensions.mcp.builtins import create_builtin_mcp_tools
from coder_agent.extensions.skills.builtin import get_builtin_skills
from coder_agent.extensions.subagents.builtin import get_builtin_subagents


class TestSkill:
    """Tests for Skill class."""

    def test_basic_skill(self):
        skill = Skill(
            name="test_skill",
            description="A test skill",
            command="Do something with {target}",
            when_to_use="When testing",
        )
        assert skill.name == "skill_test_skill"
        assert "A test skill" in skill.description
        assert "When testing" in skill.description

    def test_skill_without_when_to_use(self):
        skill = Skill(name="s", description="desc", command="cmd")
        assert skill.name == "skill_s"
        assert "Use this when" not in skill.description

    def test_skill_execute(self):
        skill = Skill(
            name="review",
            description="Review code",
            command="Review {target}",
        )
        result = skill.execute({"target": "foo.py"})
        assert result.output == "[Skill: review]\nReview foo.py"
        assert result.metadata["skill_name"] == "review"
        assert result.metadata["target"] == "foo.py"

    def test_skill_parameters(self):
        skill = Skill(name="s", description="d", command="c")
        params = skill.parameters
        assert params["required"] == ["target"]
        assert "target" in params["properties"]
        assert "context" in params["properties"]

    def test_skill_allowed_tools(self):
        skill = Skill(
            name="s",
            description="d",
            command="c",
            allowed_tools=("read_file", "write_file"),
        )
        assert skill.allowed_tools == ("read_file", "write_file")

    def test_skill_context_mode(self):
        skill = Skill(name="s", description="d", command="c", context="fork")
        assert skill.context_mode == "fork"

    def test_skill_to_tool_def(self):
        skill = Skill(name="s", description="d", command="c")
        defn = skill.to_tool_def()
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "skill_s"


class TestBuiltInSkills:
    """Tests for built-in skills."""

    def test_get_builtin_skills(self):
        skills = get_builtin_skills()
        assert len(skills) == 5

    def test_code_review_skill(self):
        skills = get_builtin_skills()
        review = next(s for s in skills if s.name == "skill_code_review")
        assert "bug" in review.description.lower() or "review" in review.description.lower()
        assert "After writing" in review._when_to_use

    def test_test_writer_skill(self):
        skills = get_builtin_skills()
        writer = next(s for s in skills if s.name == "skill_test_writer")
        assert "pytest" in writer._command.lower() or "test" in writer._command.lower()

    def test_security_audit_skill(self):
        skills = get_builtin_skills()
        audit = next(s for s in skills if s.name == "skill_security_audit")
        assert "SQL injection" in audit._command


class TestMcpTools:
    """Tests for MCP tools."""

    def test_create_builtin_mcp_tools(self):
        tools = create_builtin_mcp_tools(".")
        assert len(tools) == 5

    def test_mcp_git_log(self):
        tools = create_builtin_mcp_tools(".")
        git_log = next(t for t in tools if t.name == "mcp_git_git_log")
        result = git_log.execute({"limit": 5})
        assert result.output or result.error  # succeeds or errors gracefully

    def test_mcp_git_diff(self):
        tools = create_builtin_mcp_tools(".")
        git_diff = next(t for t in tools if t.name == "mcp_git_git_diff")
        result = git_diff.execute({})
        # git diff may return empty output in a fresh repo — just verify no crash
        assert result.output is not None

    def test_mcp_system_info(self):
        tools = create_builtin_mcp_tools(".")
        sys_info = next(t for t in tools if t.name == "mcp_system_system_info")
        result = sys_info.execute({})
        assert "Python" in result.output

    def test_mcp_tool_definition(self):
        from coder_agent.extensions.base import McpTool
        tool = McpTool(
            server_name="test",
            name="hello",
            description="Say hello",
            parameters={"type": "object", "properties": {}},
        )
        defn = tool.to_tool_def()
        assert defn["function"]["name"] == "mcp_test_hello"
        assert "[MCP: test]" in defn["function"]["description"]


class TestSubagentDefinition:
    """Tests for SubagentDefinition."""

    def test_basic_definition(self):
        defn = SubagentDefinition(
            name="test_agent",
            system_prompt="You are a tester.",
            when_to_use="When testing",
            max_steps=10,
            read_only=True,
        )
        assert defn.name == "test_agent"
        assert defn.max_steps == 10
        assert defn.read_only is True
        assert defn.tools == ("*",)
        assert defn.disallowed_tools == ()

    def test_definition_with_tool_restrictions(self):
        defn = SubagentDefinition(
            name="reader",
            system_prompt="Read only.",
            when_to_use="For reading",
            tools=("read_file", "search_text"),
            disallowed_tools=("write_file",),
        )
        assert defn.tools == ("read_file", "search_text")
        assert defn.disallowed_tools == ("write_file",)


class TestBuiltInSubagents:
    """Tests for built-in subagents."""

    def test_get_builtin_subagents(self):
        subagents = get_builtin_subagents()
        assert len(subagents) == 4

    def test_researcher_is_read_only(self):
        subagents = get_builtin_subagents()
        researcher = next(s for s in subagents if s.name == "researcher")
        assert researcher.read_only is True
        assert "write_file" not in researcher.tools

    def test_test_specialist_can_write(self):
        subagents = get_builtin_subagents()
        tester = next(s for s in subagents if s.name == "test_specialist")
        assert "write_file" in tester.tools
        assert "run_command" in tester.tools

    def test_security_scanner_is_read_only(self):
        subagents = get_builtin_subagents()
        scanner = next(s for s in subagents if s.name == "security_scanner")
        assert scanner.read_only is True


class TestExtensionResult:
    """Tests for ExtensionResult dataclass."""

    def test_default_result(self):
        result = ExtensionResult()
        assert result.output == ""
        assert result.error is None
        assert result.metadata == {}

    def test_result_with_values(self):
        result = ExtensionResult(output="hello", error=None, metadata={"key": "val"})
        assert result.output == "hello"
        assert result.metadata["key"] == "val"


class TestSubagentRequest:
    """Tests for SubagentRequest."""

    def test_basic_request(self):
        req = SubagentRequest(prompt="Fix the bug", subagent_type="test_specialist")
        assert req.prompt == "Fix the bug"
        assert req.subagent_type == "test_specialist"
        assert req.metadata == {}


class TestSubagentResult:
    """Tests for SubagentResult."""

    def test_success_result(self):
        result = SubagentResult(
            subagent_type="researcher",
            final_output="The code has 3 modules.",
            steps_used=5,
        )
        assert result.is_error is False
        assert result.steps_used == 5

    def test_error_result(self):
        result = SubagentResult(
            subagent_type="unknown",
            final_output="Unknown type",
            is_error=True,
        )
        assert result.is_error is True
