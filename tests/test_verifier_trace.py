"""Tests for Verifier and TraceRecorder."""

import json
import tempfile
from pathlib import Path

import pytest

from coder_agent.trace import TraceRecorder
from coder_agent.verifier import Verifier


class TestVerifier:
    def test_no_tests_skipped(self, tmp_path: Path) -> None:
        """No test files → verifier passes (nothing to check)."""
        (tmp_path / "main.py").write_text("def hello(): return 42\n")
        v = Verifier(tmp_path)
        passed, summary = v.check()
        assert passed
        assert "No test files" in summary

    def test_syntax_check_passes(self, tmp_path: Path) -> None:
        """Valid Python → syntax check passes."""
        (tmp_path / "good.py").write_text("x = 1 + 2\n")
        v = Verifier(tmp_path)
        passed, _ = v.check()
        assert passed

    def test_syntax_check_fails(self, tmp_path: Path) -> None:
        """Invalid Python → syntax check fails."""
        (tmp_path / "bad.py").write_text("def broken(\n")
        v = Verifier(tmp_path)
        passed, summary = v.check()
        assert not passed
        assert "syntax" in summary.lower()

    def test_pytest_passes(self, tmp_path: Path) -> None:
        """Valid test → pytest passes."""
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        (tmp_path / "test_calc.py").write_text(
            "from calc import add\ndef test_add(): assert add(1, 2) == 3\n"
        )
        v = Verifier(tmp_path)
        passed, summary = v.check()
        assert passed
        assert "pytest" in summary.lower()

    def test_pytest_fails(self, tmp_path: Path) -> None:
        """Failing test → pytest fails."""
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        (tmp_path / "test_calc.py").write_text(
            "from calc import add\ndef test_add(): assert add(1, 2) == 999\n"
        )
        v = Verifier(tmp_path)
        passed, summary = v.check()
        assert not passed
        assert "pytest" in summary.lower()

    def test_git_diff_skipped_without_git(self, tmp_path: Path) -> None:
        """No .git → git check skipped."""
        (tmp_path / "a.py").write_text("x=1\n")
        v = Verifier(tmp_path)
        passed, summary = v.check()
        assert passed
        assert "git" in summary.lower()

    def test_elapsed_time(self, tmp_path: Path) -> None:
        v = Verifier(tmp_path)
        v.check()
        assert v.elapsed >= 0


class TestTraceRecorder:
    def test_record_and_summarize(self, tmp_path: Path) -> None:
        trace_file = tmp_path / "trace.jsonl"
        tr = TraceRecorder(trace_file)

        tr.record(1, "llm_response", has_tool_calls=True)
        tr.record(1, "tool_execution", tool="read_file", success=True, output_len=100)
        tr.record(2, "tool_execution", tool="write_file", success=False, error="denied")
        tr.record(2, "final_answer", answer_preview="Done")

        assert len(tr.get_entries()) == 4
        summary = tr.summarize()
        assert "Steps: 1" in summary
        assert "Tool calls: 2" in summary
        assert "50.0%" in summary

        # Verify JSONL file
        lines = trace_file.read_text().strip().split("\n")
        assert len(lines) == 4
        for line in lines:
            entry = json.loads(line)
            assert "timestamp" in entry
            assert "elapsed_seconds" in entry
            assert "step" in entry
            assert "event" in entry

    def test_metrics(self, tmp_path: Path) -> None:
        trace_file = tmp_path / "trace.jsonl"
        tr = TraceRecorder(trace_file)
        tr.record(1, "llm_response")
        tr.record(1, "tool_execution", success=True)
        tr.record(2, "tool_execution", success=True)
        tr.record(2, "final_answer")

        metrics = tr.get_metrics()
        assert metrics["total_steps"] == 1
        assert metrics["total_tool_calls"] == 2
        assert metrics["success_rate"] == 1.0

    def test_empty_trace(self) -> None:
        tr = TraceRecorder(None)
        metrics = tr.get_metrics()
        assert metrics["total_steps"] == 0
        assert metrics["total_tool_calls"] == 0
