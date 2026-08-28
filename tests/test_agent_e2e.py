"""Tests for LLM parser, tokenizer, and Agent with Mock LLM."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from coder_agent.agent import Agent
from coder_agent.llm.parser import FormatError, parse_tool_calls
from coder_agent.llm.tokenizer import count_messages_tokens, count_tokens
from coder_agent.tools.filesystem import ReadFileTool, WriteFileTool
from coder_agent.tools.registry import ToolRegistry


# ── Parser Tests ───────────────────────────────────────────

class TestParseToolCalls:
    def test_parse_valid_tool_call(self):
        result = parse_tool_calls(
            [{"id": "tc1", "name": "read_file", "arguments": '{"path": "a.py"}'}],
            ["read_file"],
        )
        assert len(result) == 1
        assert result[0].tool_name == "read_file"
        assert result[0].arguments == {"path": "a.py"}

    def test_parse_empty_list_raises(self):
        with pytest.raises(FormatError):
            parse_tool_calls([], ["read_file"])

    def test_parse_none_raises(self):
        with pytest.raises(FormatError):
            parse_tool_calls(None, ["read_file"])

    def test_parse_unknown_tool_raises(self):
        with pytest.raises(FormatError):
            parse_tool_calls(
                [{"id": "tc1", "name": "unknown", "arguments": "{}"}],
                ["read_file"],
            )

    def test_parse_invalid_json_raises(self):
        with pytest.raises(FormatError):
            parse_tool_calls(
                [{"id": "tc1", "name": "read_file", "arguments": "not-json"}],
                ["read_file"],
            )

    def test_parse_multiple_calls(self):
        calls = [
            {"id": "tc1", "name": "read_file", "arguments": '{"path": "a.py"}'},
            {"id": "tc2", "name": "read_file", "arguments": '{"path": "b.py"}'},
        ]
        result = parse_tool_calls(calls, ["read_file"])
        assert len(result) == 2
        assert result[0].call_id == "tc1"
        assert result[1].call_id == "tc2"


# ── Tokenizer Tests ────────────────────────────────────────

class TestTokenizer:
    def test_count_tokens(self):
        assert count_tokens("hello world") > 0
        assert count_tokens("") == 0

    def test_count_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        tokens = count_messages_tokens(msgs)
        assert tokens > 0

    def test_tool_call_adds_tokens(self):
        msgs_basic = [
            {"role": "assistant", "content": ""},
        ]
        msgs_with_tc = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path":"a.py"}'
                }}
            ]},
        ]
        tokens_basic = count_messages_tokens(msgs_basic)
        tokens_tc = count_messages_tokens(msgs_with_tc)
        assert tokens_tc > tokens_basic  # tool_calls add tokens


# ── Agent E2E Mock Tests ───────────────────────────────────

class MockLLM:
    """Mock LLM for deterministic E2E testing."""

    def __init__(self, responses) -> None:
        self.responses = responses
        self.call_count = 0

    def chat(self, **kwargs):
        self.call_count += 1
        resp = self.responses[self.call_count - 1]

        class R:
            pass
        r = R()
        r.content = resp.get("content")
        r.tool_calls = resp.get("tool_calls")
        r.finish_reason = resp.get("finish_reason", "stop")
        return r


class TestAgentE2E:
    def test_read_file_flow(self, tmp_path: Path) -> None:
        """Agent reads a file and returns content."""
        (tmp_path / "note.txt").write_text("Hello from agent!")

        registry = ToolRegistry()
        registry.register(ReadFileTool(tmp_path))
        registry.register(WriteFileTool(tmp_path))

        llm = MockLLM([
            {"tool_calls": [{"id": "tc1", "name": "read_file", "arguments": '{"path": "note.txt"}'}]},
            {"content": "The file contains: Hello from agent!"},
        ])

        agent = Agent(llm_client=llm, registry=registry, workspace=tmp_path)
        ans = agent.run("Read note.txt")

        assert "Hello" in ans
        assert agent.state.step >= 1
        assert "note.txt" in agent.state.read_files
        assert len(agent.trace.get_entries()) > 0

    def test_write_file_flow(self, tmp_path: Path) -> None:
        """Agent writes a file."""
        registry = ToolRegistry()
        registry.register(ReadFileTool(tmp_path))
        registry.register(WriteFileTool(tmp_path))

        llm = MockLLM([
            {"tool_calls": [{"id": "tc1", "name": "write_file", "arguments": '{"path": "out.txt", "content": "hi"}'}]},
            {"content": "File written."},
        ])

        agent = Agent(llm_client=llm, registry=registry, workspace=tmp_path)
        ans = agent.run("Write hi to out.txt")

        assert (tmp_path / "out.txt").exists()
        assert (tmp_path / "out.txt").read_text() == "hi"

    def test_format_error_recovery(self, tmp_path: Path) -> None:
        """Agent recovers from FormatError."""
        registry = ToolRegistry()
        registry.register(ReadFileTool(tmp_path))

        llm = MockLLM([
            # First call: invalid JSON
            {"tool_calls": [{"id": "tc1", "name": "read_file", "arguments": "invalid-json"}]},
            # Second call: valid
            {"tool_calls": [{"id": "tc2", "name": "read_file", "arguments": '{"path": "a.py"}'}]},
            # Third call: done
            {"content": "Done"},
        ])

        agent = Agent(llm_client=llm, registry=registry, workspace=tmp_path)
        ans = agent.run("Read a.py")

        assert "Done" in ans
        assert agent._n_format_errors == 0  # recovered

    def test_max_steps_limit(self, tmp_path: Path) -> None:
        """Agent stops after MAX_STEPS."""
        registry = ToolRegistry()
        registry.register(ReadFileTool(tmp_path))

        # Always return tool_calls to prevent termination
        llm = MockLLM([
            {"tool_calls": [{"id": "tc1", "name": "read_file", "arguments": '{"path": "x"}'}]},
        ] * 60)  # more than MAX_STEPS

        agent = Agent(llm_client=llm, registry=registry, workspace=tmp_path)
        ans = agent.run("Do something")

        assert "maximum steps" in ans.lower()

    def test_verifier_integration(self, tmp_path: Path) -> None:
        """Verifier runs after agent completes."""
        from coder_agent.verifier import Verifier

        (tmp_path / "calc.py").write_text("def add(a,b): return a+b\n")
        (tmp_path / "test_calc.py").write_text("from calc import add\ndef test(): assert add(1,2)==3\n")

        registry = ToolRegistry()
        registry.register(ReadFileTool(tmp_path))
        registry.register(WriteFileTool(tmp_path))

        llm = MockLLM([
            {"content": "I have completed the task."},
        ])

        verifier = Verifier(tmp_path)
        agent = Agent(
            llm_client=llm, registry=registry, workspace=tmp_path,
            verifier=verifier,
        )
        ans = agent.run("Do something")

        # Verifier should have run
        passed, summary = verifier.check()
        assert passed
        assert "pytest" in summary.lower()

    def test_trace_output_to_file(self, tmp_path: Path) -> None:
        """Trace is written to JSONL file."""
        trace_file = tmp_path / "trace.jsonl"

        registry = ToolRegistry()
        registry.register(ReadFileTool(tmp_path))

        llm = MockLLM([
            {"content": "Done"},
        ])

        agent = Agent(
            llm_client=llm, registry=registry, workspace=tmp_path,
            trace_output=trace_file,
        )
        agent.run("Test trace")

        assert trace_file.exists()
        lines = trace_file.read_text().strip().split("\n")
        assert len(lines) > 0
        # Each line is valid JSON
        for line in lines:
            entry = json.loads(line)
            assert "timestamp" in entry
            assert "event" in entry

    def test_state_tracking(self, tmp_path: Path) -> None:
        """AgentState tracks file operations."""
        registry = ToolRegistry()
        registry.register(ReadFileTool(tmp_path))
        registry.register(WriteFileTool(tmp_path))

        llm = MockLLM([
            {"tool_calls": [{"id": "tc1", "name": "read_file", "arguments": '{"path": "a.py"}'}]},
            {"tool_calls": [{"id": "tc2", "name": "write_file", "arguments": '{"path": "b.py", "content": "x"}'}]},
            {"content": "Done"},
        ])

        agent = Agent(llm_client=llm, registry=registry, workspace=tmp_path)
        agent.run("Test state")

        assert "a.py" in agent.state.read_files
        assert "b.py" in agent.state.modified_files
        assert agent.state.step >= 2

    def test_memory_recording(self, tmp_path: Path) -> None:
        """Memory records tool executions."""
        registry = ToolRegistry()
        registry.register(ReadFileTool(tmp_path))

        llm = MockLLM([
            {"tool_calls": [{"id": "tc1", "name": "read_file", "arguments": '{"path": "a.py"}'}]},
            {"content": "Done"},
        ])

        agent = Agent(llm_client=llm, registry=registry, workspace=tmp_path)
        agent.run("Test memory")

        assert len(agent.memory.short_term) >= 1
        assert agent.memory.recall("test_key") is None
        agent.memory.remember("test_key", "value")
        assert agent.memory.recall("test_key") == "value"
