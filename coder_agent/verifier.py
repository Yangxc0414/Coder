"""Verifier — objective task completion check.

A task is not done just because the model says so.
The Verifier independently validates the result by running
tests, checking syntax, and inspecting git diffs.

Inspired by: Continue's verification checks, SWE-agent's evaluation.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    """Result of a single verification check."""

    name: str
    passed: bool
    message: str
    detail: str = ""


class Verifier:
    """Objectively verifies whether a task is truly complete.

    Runs multiple checks:
    1. pytest — if test files exist
    2. syntax — compile all .py files
    3. git diff — if workspace is a git repo

    Only when ALL checks pass, the task is considered complete.
    """

    def __init__(
        self,
        workspace: Path,
        task: str = "",
        timeout: int = 60,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.task = task
        self.timeout = timeout
        self._start_time = time.time()
        self._results: list[CheckResult] = []

    def check(self) -> tuple[bool, str]:
        """Run all checks and return (all_passed, summary)."""
        self._results = []
        self._start_time = time.time()

        # Check 1: Run pytest if test files exist
        test_result = self._check_tests()
        self._results.append(test_result)

        # Check 2: Syntax check on all Python files
        syntax_result = self._check_syntax()
        self._results.append(syntax_result)

        # Check 3: Git diff (informational)
        git_result = self._check_git()
        self._results.append(git_result)

        all_passed = all(r.passed for r in self._results)
        summary = self._format_summary()
        return all_passed, summary

    def _check_tests(self) -> CheckResult:
        """Run pytest on any test files found in the workspace."""
        test_files = list(self.workspace.rglob("test_*.py"))
        if not test_files:
            # No tests to run — this is acceptable
            return CheckResult(
                name="pytest",
                passed=True,
                message="No test files found (skipped)",
            )

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_files[0].parent), "-q", "--tb=short"],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
            )
            passed = result.returncode == 0
            output = result.stdout[-500:] if result.stdout else ""
            error = result.stderr[-200:] if result.stderr else ""
            detail = output + error if not passed else ""
            return CheckResult(
                name="pytest",
                passed=passed,
                message=f"Tests {'passed' if passed else 'FAILED'}",
                detail=detail,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                name="pytest",
                passed=False,
                message=f"Tests timed out after {self.timeout}s",
            )
        except Exception as e:
            return CheckResult(
                name="pytest",
                passed=False,
                message=f"Test execution error: {e}",
            )

    def _check_syntax(self) -> CheckResult:
        """Check Python syntax for all .py files in workspace."""
        py_files = list(self.workspace.rglob("*.py"))
        if not py_files:
            return CheckResult(
                name="syntax",
                passed=True,
                message="No Python files to check",
            )

        errors = []
        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                compile(source, str(py_file), "exec")
            except SyntaxError as e:
                errors.append(f"{py_file.relative_to(self.workspace)}:{e}")
            except Exception as e:
                errors.append(f"{py_file.relative_to(self.workspace)}: {e}")

        passed = len(errors) == 0
        return CheckResult(
            name="syntax",
            passed=passed,
            message=f"{len(py_files)} files, {'0 errors' if passed else f'{len(errors)} error(s)'}",
            detail="\n".join(errors[:5]) if errors else "",
        )

    def _check_git(self) -> CheckResult:
        """Check git diff if workspace is a git repo."""
        git_dir = self.workspace / ".git"
        if not git_dir.exists():
            return CheckResult(
                name="git_diff",
                passed=True,
                message="Not a git repo (skipped)",
            )

        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
            )
            diff_output = result.stdout.strip()
            has_changes = bool(diff_output and "no changes" not in diff_output.lower())
            return CheckResult(
                name="git_diff",
                passed=True,  # Informational only
                message=f"Git diff: {'has changes' if has_changes else 'no changes'}",
                detail=diff_output[:300] if diff_output else "",
            )
        except Exception:
            return CheckResult(
                name="git_diff",
                passed=True,
                message="Git check skipped (error)",
            )

    def _format_summary(self) -> str:
        lines = [f"Verifier results ({self.elapsed:.1f}s):"]
        for r in self._results:
            status = "✅" if r.passed else "❌"
            lines.append(f"  {status} {r.name}: {r.message}")
            if r.detail:
                for line in r.detail.split("\n")[:3]:
                    lines.append(f"       {line}")
        all_passed = all(r.passed for r in self._results)
        lines.append(f"\n{'PASS' if all_passed else 'FAIL'}: Task verification {'passed' if all_passed else 'failed'}")
        return "\n".join(lines)

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    def get_metrics(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": round(self.elapsed, 2),
            "checks_passed": sum(1 for r in self._results if r.passed),
            "checks_total": len(self._results),
            "all_passed": all(r.passed for r in self._results),
        }
