"""Baseline-diff verification tests.

The verifier must hold the task responsible only for regressions it
introduces — failures that already existed before the task started are
filtered out. Regression guard for the 50-step flail incident.
"""

from __future__ import annotations

import sys
from pathlib import Path

from coder_agent.verifier import Verifier


def _write_test(tmp_path: Path, body: str) -> None:
    (tmp_path / "test_sample.py").write_text(body, encoding="utf-8")


FAILING_ONE = "from sample import add\n\ndef test_one():\n    assert add(1, 1) == 3\n"
FAILING_TWO = FAILING_ONE + "\ndef test_two():\n    assert add(2, 2) == 5\n"


class TestBaselineDiff:
    def test_baseline_captures_preexisting_failures(self, tmp_path: Path):
        (tmp_path / "sample.py").write_text("def add(a, b):\n    return a + b\n")
        _write_test(tmp_path, FAILING_TWO)
        v = Verifier(tmp_path, timeout=60)
        n_fail, n_syn = v.establish_baseline()
        assert n_fail == 2
        assert n_syn == 0

    def test_no_new_failures_passes_despite_preexisting(self, tmp_path: Path):
        """The 50-step flail scenario: pre-existing failure must not fail the task."""
        (tmp_path / "sample.py").write_text("def add(a, b):\n    return a + b\n")
        _write_test(tmp_path, FAILING_TWO)
        v = Verifier(tmp_path)
        v.establish_baseline()

        # agent did nothing about the tests — check must still pass
        passed, summary = v.check()
        assert passed, summary
        assert "pre-existing" in summary

    def test_new_failure_fails_even_with_baseline(self, tmp_path: Path):
        (tmp_path / "sample.py").write_text("def add(a, b):\n    return a + b\n")
        _write_test(tmp_path, FAILING_TWO)
        v = Verifier(tmp_path)
        v.establish_baseline()

        # agent breaks a previously-passing test
        _write_test(tmp_path, FAILING_TWO + "\ndef test_three():\n    assert 1 == 2\n")
        passed, summary = v.check()
        assert not passed
        assert "NEW" in summary

    def test_fixing_preexisting_failure_also_passes(self, tmp_path: Path):
        (tmp_path / "sample.py").write_text("def add(a, b):\n    return a + b\n")
        _write_test(tmp_path, FAILING_TWO)
        v = Verifier(tmp_path)
        v.establish_baseline()

        # agent FIXES one baseline failure — remaining one is still baseline
        _write_test(tmp_path, FAILING_ONE)
        passed, summary = v.check()
        assert passed, summary

    def test_no_baseline_strict_as_before(self, tmp_path: Path):
        """Without establish_baseline(), behavior is unchanged (strict)."""
        (tmp_path / "sample.py").write_text("def add(a, b):\n    return a + b\n")
        _write_test(tmp_path, FAILING_ONE)
        v = Verifier(tmp_path)
        passed, _ = v.check()
        assert not passed

    def test_baseline_covers_syntax_errors(self, tmp_path: Path):
        (tmp_path / "broken.py").write_text("def f(:\n    pass\n", encoding="utf-8")
        v = Verifier(tmp_path)
        n_fail, n_syn = v.establish_baseline()
        assert n_syn >= 1

        passed, summary = v.check()
        assert passed, summary  # pre-existing syntax error filtered

        # agent adds a NEW broken file → fails
        (tmp_path / "worse.py").write_text("x = = 1\n", encoding="utf-8")
        passed, summary = v.check()
        assert not passed
        assert "NEW" in summary

    def test_all_fixed_means_all_pass(self, tmp_path: Path):
        (tmp_path / "sample.py").write_text("def add(a, b):\n    return a + b\n")
        _write_test(tmp_path, FAILING_ONE)
        v = Verifier(tmp_path)
        v.establish_baseline()

        _write_test(tmp_path, "from sample import add\n\ndef test_one():\n    assert add(1, 1) == 2\n")
        passed, summary = v.check()
        assert passed, summary
