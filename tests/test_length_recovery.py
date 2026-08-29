"""finish_reason=length recovery tests (improvement F from
doc/deep_comparison.md, sourced from OneCode loop.py:586-615:
a response cut off by max_tokens mid-answer is retried once with a
doubled budget instead of surfacing as a parse error)."""

from __future__ import annotations

from pathlib import Path

from coder_agent.agent import Agent
from coder_agent.llm.client import LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.tools.registry import ToolRegistry


class TruncatingLLM:
    """First call returns a length-truncated answer; the escalated retry
    (with doubled max_tokens) returns the full one."""

    model = "mock"

    def __init__(self) -> None:
        self.max_tokens_seen: list[int] = []

    def chat(self, messages, tools=None, max_tokens=4096):
        self.max_tokens_seen.append(max_tokens)
        if max_tokens <= 4096:
            return LLMResponse(
                content="partial answer that was cut off mid-",
                tool_calls=None,
                finish_reason="length",
                usage=None,
            )
        return LLMResponse(
            content="full complete answer",
            tool_calls=None,
            finish_reason="stop",
            usage=None,
        )


class TestLengthEscalation:
    def test_truncated_response_retried_with_doubled_budget(self, tmp_path: Path):
        llm = TruncatingLLM()
        agent = Agent(
            llm_client=llm, registry=ToolRegistry(), workspace=tmp_path,
            mode=AgentMode.GOAL,
        )
        answer = agent.run("task")
        assert answer == "full complete answer"
        assert llm.max_tokens_seen == [4096, 8192]

    def test_escalation_happens_once_per_run(self, tmp_path: Path):
        class AlwaysTruncated(TruncatingLLM):
            def chat(self, messages, tools=None, max_tokens=4096):
                self.max_tokens_seen.append(max_tokens)
                return LLMResponse(
                    content="still truncated", tool_calls=None,
                    finish_reason="length", usage=None,
                )

        llm = AlwaysTruncated()
        agent = Agent(
            llm_client=llm, registry=ToolRegistry(), workspace=tmp_path,
            mode=AgentMode.GOAL,
        )
        answer = agent.run("task")
        # exactly one escalation: [4096, 8192] — the second truncation is
        # accepted as-is rather than looping forever
        assert llm.max_tokens_seen == [4096, 8192]
        assert answer == "still truncated"
        assert "length_truncated" in [e["event"] for e in agent.trace.get_entries()]

    def test_tool_call_responses_never_escalate(self, tmp_path: Path):
        from coder_agent.tools.filesystem import ListFilesTool

        class ToolCallLLM:
            model = "mock"

            def __init__(self):
                self.max_tokens_seen: list[int] = []

            def chat(self, messages, tools=None, max_tokens=4096):
                self.max_tokens_seen.append(max_tokens)
                return LLMResponse(
                    content="", tool_calls=[
                        {"id": "t1", "name": "list_files", "arguments": "{}"},
                    ],
                    finish_reason="length", usage=None,
                )

        llm = ToolCallLLM()
        registry = ToolRegistry()
        registry.register(ListFilesTool(tmp_path))
        agent = Agent(
            llm_client=llm, registry=registry, workspace=tmp_path,
            mode=AgentMode.GOAL, max_steps=1,
        )
        agent.run("task")
        # tool_calls with finish_reason=length are parsed, not escalated —
        # every call (incl. the forced-final round) stays at the base budget
        assert all(t == 4096 for t in llm.max_tokens_seen)

    def test_counter_resets_between_runs(self, tmp_path: Path):
        llm = TruncatingLLM()
        agent = Agent(
            llm_client=llm, registry=ToolRegistry(), workspace=tmp_path,
            mode=AgentMode.GOAL,
        )
        agent.run("first")
        agent.run("second")
        # second run escalates again from the original budget
        assert llm.max_tokens_seen == [4096, 8192, 4096, 8192]
