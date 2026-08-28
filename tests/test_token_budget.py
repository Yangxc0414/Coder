"""Token budget tests — runaway-cost gate (the second of three runaway gates:
MAX_STEPS / token_budget / verify-gating)."""

from __future__ import annotations

from pathlib import Path

from coder_agent.agent import Agent
from coder_agent.llm.client import LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.tools.registry import ToolRegistry


class ScriptedLLM:
    """Queued responses with configurable usage; records calls."""

    model = "mock"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._i = 0
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, max_tokens=4096):
        self.calls.append([dict(m) for m in messages])
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return resp


def _resp(content="", tool_calls=None, prompt=0, completion=0):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion} if (prompt or completion) else None
    return LLMResponse(content=content, tool_calls=tool_calls, finish_reason="stop", usage=usage)


def _agent(tmp_path: Path, llm, budget):
    return Agent(
        llm_client=llm, registry=ToolRegistry(), workspace=tmp_path,
        mode=AgentMode.GOAL, token_budget=budget, max_steps=10,
    )


class TestTokenBudget:
    def test_over_budget_requests_wrapup_then_accepts(self, tmp_path: Path):
        # step 1 burns the whole budget; step 2 is the wrap-up answer
        llm = ScriptedLLM([
            _resp("thinking", prompt=150, completion=100),   # 250 >= 100
            _resp("wrapped up: half done", prompt=50, completion=20),
        ])
        agent = _agent(tmp_path, llm, budget=100)
        answer = agent.run("big task")

        assert answer == "wrapped up: half done"
        assert agent._n_steps == 2
        # wrap-up instruction was injected before the final call
        last_call_contents = [m.get("content") for m in llm.calls[1]]
        assert any("Token budget exhausted" in (c or "") for c in last_call_contents)

    def test_under_budget_completes_normally(self, tmp_path: Path):
        llm = ScriptedLLM([_resp("all done", prompt=10, completion=5)])
        agent = _agent(tmp_path, llm, budget=100_000)
        assert agent.run("easy task") == "all done"
        assert agent._n_steps == 1
        assert agent._tokens_used == 15

    def test_straggler_tool_calls_after_budget_are_intercepted(self, tmp_path: Path):
        # model ignores the wrap-up request and tries more tool calls
        llm = ScriptedLLM([
            _resp(tool_calls=[{"id": "t1", "name": "list_files", "arguments": "{}"}],
                  prompt=200, completion=10),
            _resp(tool_calls=[{"id": "t2", "name": "list_files", "arguments": "{}"}],
                  content="partial summary", prompt=10, completion=5),
        ])
        agent = _agent(tmp_path, llm, budget=100)
        answer = agent.run("stubborn task")

        # hard stop: the straggler's tool call was never executed
        assert answer == "partial summary"
        assert agent._n_steps == 2
        assert "budget_terminated" in [e["event"] for e in agent.trace.get_entries()]

    def test_no_budget_means_unlimited(self, tmp_path: Path):
        llm = ScriptedLLM([_resp("ok", prompt=10_000_000, completion=1)])
        agent = _agent(tmp_path, llm, budget=None)
        assert agent.run("task") == "ok"

    def test_usage_none_never_triggers(self, tmp_path: Path):
        llm = ScriptedLLM([_resp("fine")])  # usage=None (mock style)
        agent = _agent(tmp_path, llm, budget=1)
        assert agent.run("task") == "fine"

    def test_counter_resets_between_runs(self, tmp_path: Path):
        llm = ScriptedLLM([_resp("a", prompt=500, completion=0), _resp("b", prompt=1, completion=0)])
        agent = _agent(tmp_path, llm, budget=10_000)
        agent.run("first")
        agent.run("second")
        assert agent._tokens_used == 1  # second run starts from zero
