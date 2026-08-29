"""Stuck detection tests — (tool, args, RESULT) repetition hashing
(improvement C from doc/deep_comparison.md, sourced from openhands
stuck_detection).

Core insight under test: a loop is a repeated action AND a repeated
observation. The same command with DIFFERENT results is productive
iteration (the workspace changed between runs) — hashing actions alone
would false-positive on every normal edit-test loop.
"""

from __future__ import annotations

from coder_agent.state import AgentState


class TestRepetitionRiskWithOutcomes:
    def test_same_command_same_failure_is_stuck(self):
        """The classic runaway: identical broken test command, identical
        failure, three times in a row."""
        s = AgentState()
        for _ in range(3):
            s.add_tool_outcome("run_command", '{"command": "pytest -q"}', "FAILED: test_x")
        assert s.get_repetition_risk() is True

    def test_same_command_different_result_is_productive(self):
        """Edit-test loop: same pytest command, but the output changes
        because the code changed between runs. NOT a stuck loop."""
        s = AgentState()
        for i, outcome in enumerate(["FAILED: 3 errors", "FAILED: 1 error", "passed"]):
            s.add_tool_outcome("run_command", '{"command": "pytest -q"}', outcome)
        assert s.get_repetition_risk() is False

    def test_frequent_identical_outcome_in_window(self):
        s = AgentState()
        outcomes = [
            ("run_command", "cmdA", "err1"),
            ("read_file", "b.py", "content..."),
            ("run_command", "cmdA", "err1"),
            ("search_text", "c", "found"),
            ("run_command", "cmdA", "err1"),
        ]
        for t, a, r in outcomes:
            s.add_tool_outcome(t, a, r)
        assert s.get_repetition_risk() is True

    def test_varied_outcomes_no_risk(self):
        s = AgentState()
        for i in range(5):
            s.add_tool_outcome("read_file", f'{{"path": "f{i}.py"}}', f"content {i}")
        assert s.get_repetition_risk() is False

    def test_double_repeat_only_is_safe(self):
        s = AgentState()
        s.add_tool_outcome("run_command", "A", "err")
        s.add_tool_outcome("run_command", "A", "err")
        assert s.get_repetition_risk() is False

    def test_outcome_window_trimmed(self):
        s = AgentState()
        for i in range(20):
            s.add_tool_outcome("read_file", f"f{i}", f"out{i}")
        assert len(s.recent_outcomes) <= 12


class TestRepetitionRiskFallback:
    """Without outcomes recorded (e.g. direct state use), falls back to
    args-only hashing on recent_actions."""

    def test_fallback_uses_recent_actions(self):
        s = AgentState()
        for _ in range(3):
            s.add_recent_action("run_command", '{"command": "pytest -q"}')
        assert s.get_repetition_risk() is True

    def test_fallback_varied_is_safe(self):
        s = AgentState()
        for i in range(5):
            s.add_recent_action("read_file", f'{{"path": "f{i}.py"}}')
        assert s.get_repetition_risk() is False

    def test_too_few_actions(self):
        s = AgentState()
        s.add_recent_action("run_command", "A")
        s.add_recent_action("run_command", "A")
        assert s.get_repetition_risk() is False
