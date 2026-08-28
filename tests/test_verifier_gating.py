"""Verifier gating tests — regression for the 50-step flail incident.

Real-API run: the agent answered correctly at step 2, but the workspace-wide
verifier kept failing on a PRE-EXISTING broken test, injecting "fix it"
prompts the read-only agent could never satisfy — burning all 50 steps.

Rule under test: a failed verification may only trigger a retry when the
agent actually mutated the workspace since the previous failed check.
"""

from __future__ import annotations

from pathlib import Path

from coder_agent.agent import Agent
from coder_agent.llm.client import LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.tools.registry import ToolRegistry


class ScriptedLLM:
    """Returns queued responses in order."""

    model = "mock"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._i = 0

    def chat(self, messages, tools=None, max_tokens=4096):
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return resp


def _final(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=None, finish_reason="stop", usage=None)


def _tool_call(name: str, arguments: str = "{}") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[{"id": "t1", "name": name, "arguments": arguments}],
        finish_reason="tool_calls",
        usage=None,
    )


class AlwaysFailVerifier:
    """Stands in for a workspace with pre-existing broken tests."""

    def check(self):
        return False, "pytest: Tests FAILED (pre-existing failure)"


class RecordingVerifier:
    """Fails the first N checks, then passes."""

    def __init__(self, fail_times: int = 1) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def check(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            return False, "pytest: Tests FAILED"
        return True, "all checks passed"


def _agent(tmp_path: Path, llm, verifier) -> Agent:
    return Agent(
        llm_client=llm,
        registry=ToolRegistry(),
        workspace=tmp_path,
        mode=AgentMode.GOAL,
        verifier=verifier,
        max_steps=10,
    )


class TestVerifierGating:
    def test_no_mutations_accepts_answer_despite_failure(self, tmp_path: Path):
        """Correct answer + pre-existing failure must NOT loop to max steps."""
        llm = ScriptedLLM([_final("The bug is divide() returns None.")])
        agent = _agent(tmp_path, llm, AlwaysFailVerifier())
        answer = agent.run("summarize the bug")
        assert answer == "The bug is divide() returns None."
        assert agent._n_steps == 1  # answered on the first step, no retry
        assert agent.trace.get_entries()  # the acceptance decision is traced

    def test_mutation_triggers_one_retry_then_accepts(self, tmp_path: Path):
        """After a mutation, one retry is injected; no further change -> accept."""
        llm = ScriptedLLM([
            _tool_call("write_file", '{"path": "a.txt", "content": "x"}'),
            _final("attempt 1"),
            _final("attempt 2 - unchanged"),
        ])
        # real write tool so the mutation actually counts
        from coder_agent.tools.filesystem import WriteFileTool

        agent = _agent(tmp_path, llm, AlwaysFailVerifier())
        agent.registry.register(WriteFileTool(tmp_path))
        answer = agent.run("fix it")
        # retry was injected once (mutation), second final accepted (no new mutation)
        assert answer == "attempt 2 - unchanged"
        assert agent._n_steps == 3

    def test_passing_verifier_returns_immediately(self, tmp_path: Path):
        llm = ScriptedLLM([_final("done")])
        verifier = RecordingVerifier(fail_times=0)
        agent = _agent(tmp_path, llm, verifier)
        answer = agent.run("task")
        assert answer == "done"
        assert verifier.calls == 1

    def test_mutations_counter_tracks_writes(self, tmp_path: Path):
        from coder_agent.tools.filesystem import WriteFileTool

        llm = ScriptedLLM([
            _tool_call("write_file", '{"path": "a.txt", "content": "x"}'),
            _final("done"),
        ])
        agent = _agent(tmp_path, llm, None)
        agent.registry.register(WriteFileTool(tmp_path))
        agent.run("task")
        assert agent._n_mutations == 1

    def test_reads_do_not_count_as_mutations(self, tmp_path: Path):
        from coder_agent.tools.filesystem import ReadFileTool

        (tmp_path / "f.txt").write_text("hi")
        llm = ScriptedLLM([
            _tool_call("read_file", '{"path": "f.txt"}'),
            _final("summary"),
        ])
        agent = _agent(tmp_path, llm, AlwaysFailVerifier())
        agent.registry.register(ReadFileTool(tmp_path))
        answer = agent.run("read and summarize")
        assert answer == "summary"
        assert agent._n_mutations == 0
        assert agent._n_steps == 2  # one step for the read, one for the answer
