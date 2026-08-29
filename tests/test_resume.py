"""End-to-end resume tests — interrupt one agent, continue with another.

Scenario: agent A journals a partial run and "crashes"; agent B is built
fresh, restores the journal, and continues. B's LLM must see the full
prior conversation, and B's own messages must be journaled back into the
same file (chained recovery).
"""

from __future__ import annotations

import json
from pathlib import Path

from coder_agent.agent import Agent
from coder_agent.journal import (
    DEFAULT_RESUME_PROMPT,
    SessionJournal,
    load_journal,
    replay_state,
)
from coder_agent.llm.client import LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.tools.filesystem import WriteFileTool
from coder_agent.tools.registry import ToolRegistry


class RecordingLLM:
    """Returns queued responses; records every message list it is given."""

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


def _final(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=None, finish_reason="stop", usage=None)


def _tool_call(name: str, arguments: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[{"id": "t1", "name": name, "arguments": arguments}],
        finish_reason="tool_calls",
        usage=None,
    )


def _registry(tmp_path: Path) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WriteFileTool(tmp_path))
    return reg


class TestResumeE2E:
    def test_interrupted_session_restores_full_history(self, tmp_path: Path):
        journal_path = tmp_path / "session.jsonl"

        # --- Agent A: runs one step (a write), then is "interrupted" ---
        llm_a = RecordingLLM([
            _tool_call("write_file", '{"path": "calc.py", "content": "x = 1"}'),
        ])
        journal_a = SessionJournal(journal_path)
        agent_a = Agent(
            llm_client=llm_a, registry=_registry(tmp_path),
            workspace=tmp_path, mode=AgentMode.GOAL,
            max_steps=1, journal=journal_a,  # forced "crash" after step 1
        )
        answer_a = agent_a.run("fix the calculator")
        journal_a.close()
        assert "maximum steps" in answer_a.lower()

        # --- Agent B: fresh, resumes the journal ---
        restored = load_journal(journal_path)
        # user + assistant(tool_call) + tool result + step-limit notice +
        # the model's unanswered wrap-up attempt (improvement D journaled it)
        assert len(restored["messages"]) == 5

        llm_b = RecordingLLM([_final("resumed and finished")])
        journal_b = SessionJournal(journal_path, append=True)
        agent_b = Agent(
            llm_client=llm_b, registry=_registry(tmp_path),
            workspace=tmp_path, mode=AgentMode.GOAL,
            journal=journal_b,
        )
        agent_b.messages = restored["messages"]
        replay_state(restored["messages"], agent_b.state)
        agent_b.state.task_goal = "fix the calculator"

        answer_b = agent_b.run("continue the fix", resume=True)
        journal_b.close()

        assert answer_b == "resumed and finished"

        # B's LLM saw the full prior conversation + continuation instruction
        first_call = llm_b.calls[0]
        roles = [m["role"] for m in first_call]
        assert roles[0] == "system"
        contents = [m.get("content") for m in first_call]
        assert "fix the calculator" in contents  # original task preserved
        assert "x = 1" in json.dumps(first_call)  # journaled tool_call args restored
        assert "continue the fix" in contents  # continuation appended last

        # State was replayed from the journal
        assert "calc.py" in agent_b.state.modified_files

    def test_resumed_journal_is_chain_resumable(self, tmp_path: Path):
        journal_path = tmp_path / "chain.jsonl"

        llm_a = RecordingLLM([_final("part one")])
        journal_a = SessionJournal(journal_path)
        agent_a = Agent(
            llm_client=llm_a, registry=ToolRegistry(),
            workspace=tmp_path, mode=AgentMode.GOAL, journal=journal_a,
        )
        agent_a.run("task one")
        journal_a.close()

        # resume once
        restored = load_journal(journal_path)
        llm_b = RecordingLLM([_final("part two")])
        journal_b = SessionJournal(journal_path, append=True)
        agent_b = Agent(
            llm_client=llm_b, registry=ToolRegistry(),
            workspace=tmp_path, mode=AgentMode.GOAL, journal=journal_b,
        )
        agent_b.messages = restored["messages"]
        agent_b.run("keep going", resume=True)
        journal_b.close()

        # the same journal now holds BOTH rounds and is loadable again
        reloaded = load_journal(journal_path)
        contents = [m.get("content") for m in reloaded["messages"]]
        assert contents.count("task one") == 1
        assert "keep going" in contents
        # meta appears exactly once across the whole chain
        meta_lines = sum(
            1 for l in journal_path.read_text(encoding="utf-8").splitlines()
            if l.strip() and __import__("json").loads(l).get("type") == "meta"
        )
        assert meta_lines == 1

    def test_resume_without_prior_messages_behaves_normally(self, tmp_path: Path):
        """run(resume=True) on a fresh agent must not crash or lose the task."""
        llm = RecordingLLM([_final("done")])
        agent = Agent(
            llm_client=llm, registry=ToolRegistry(),
            workspace=tmp_path, mode=AgentMode.GOAL,
        )
        answer = agent.run("just do it", resume=True)
        assert answer == "done"
        assert llm.calls[0][-1]["content"] == "just do it"

    def test_default_resume_prompt_usable(self):
        assert "interrupted" in DEFAULT_RESUME_PROMPT.lower()
