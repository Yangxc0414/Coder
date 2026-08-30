"""Built-in skills for the coding agent."""

from __future__ import annotations

from ..base import Skill


# Code Review Skill
CODE_REVIEW_SKILL = Skill(
    name="code_review",
    description="""Review code for bugs, style issues, and best practices.
    
    Analyzes the target file(s) and provides a structured review with:
    - Bug risks
    - Style improvements
    - Performance suggestions
    - Security concerns""",
    command="""Please review the code in {target} thoroughly. Check for:
1. Bugs and edge cases
2. Style consistency
3. Performance issues
4. Security vulnerabilities
5. Best practices violations

Provide a structured review with specific line references.""",
    when_to_use="After writing or modifying code, before committing",
    allowed_tools=("read_file", "search_text"),
    context="inline",
)

# Test Writer Skill
TEST_WRITER_SKILL = Skill(
    name="test_writer",
    description="""Write comprehensive tests for a given module or function.
    
    Generates unit tests covering:
    - Normal cases
    - Edge cases
    - Error cases
    - Integration scenarios""",
    command="""Write tests for {target}. Include:
1. Basic functionality tests
2. Edge cases (empty input, large input, null values)
3. Error handling tests
4. Integration tests if applicable

Use pytest format. Place tests in a test_*.py file next to the target.""",
    when_to_use="After implementing a new function or module",
    allowed_tools=("read_file", "write_file", "run_command"),
    context="inline",
)

# Refactor Skill
REFACTOR_SKILL = Skill(
    name="refactor",
    description="""Refactor code to improve structure without changing behavior.
    
    Focuses on:
    - Extracting functions
    - Reducing duplication
    - Improving naming
    - Simplifying complex logic""",
    command="""Refactor {target} to improve code quality. Focus on:
1. Extract repeated code into functions
2. Improve variable and function names
3. Simplify complex conditional logic
4. Add docstrings where missing

Ensure all tests still pass after refactoring.""",
    when_to_use="When code becomes hard to read or maintain",
    allowed_tools=("read_file", "write_file", "run_command"),
    context="inline",
)

# Docstring Skill
DOCSTRING_SKILL = Skill(
    name="docstring",
    description="""Add or improve docstrings for functions and classes.
    
    Follows Google or NumPy docstring style with:
    - Brief description
    - Args/Parameters section
    - Returns section
    - Raises section (if applicable)
    - Examples (optional)""",
    command="""Add docstrings to {target}. Use Google style:
    
    Args:
        param_name: description
    
    Returns:
        description of return value
    
    Raises:
        ExceptionType: when condition""",
    when_to_use="Before marking code as complete",
    allowed_tools=("read_file", "write_file"),
    context="inline",
)

# Security Audit Skill
SECURITY_AUDIT_SKILL = Skill(
    name="security_audit",
    description="""Audit code for security vulnerabilities.
    
    Checks for:
    - SQL injection
    - Path traversal
    - Hardcoded secrets
    - Insecure dependencies
    - Input validation gaps""",
    command="""Security audit for {target}. Check for:
1. SQL injection vulnerabilities
2. Path traversal issues
3. Hardcoded secrets or API keys
4. Insecure password handling
5. Missing input validation
6. Unsafe deserialization

Report each finding with severity (Critical/High/Medium/Low).""",
    when_to_use="Before deploying code or handling sensitive data",
    allowed_tools=("read_file", "search_text"),
    context="inline",
)

# ── 实用 Skills（网上 agent 生态流行的任务类型）────────────────────────

# Commit Message Skill（规范 Git 提交信息）
COMMIT_MESSAGE_SKILL = Skill(
    name="commit_message",
    description="""Generate a conventional Git commit message from the current diff.

    Produces a concise, Conventional Commits style message
    (feat/fix/docs/refactor/perf/test/chore) with a scope and
    a bullet-point body summarizing the changes.""",
    command="""Generate a conventional commit message for the changes in {target}.
1. Inspect the diff (use mcp_git_git_diff or git diff via run_command)
2. Determine the change type: feat / fix / docs / refactor / perf / test / chore
3. Write a subject line ≤ 50 chars, imperative mood
4. Add a short bullet body with the key changes

Format:
<type>(<scope>): <subject>

- <change 1>
- <change 2>""",
    when_to_use="After making changes and before committing",
    allowed_tools=("run_command", "read_file"),
    context="inline",
)

# Changelog Skill（变更日志生成）
GENERATE_CHANGELOG_SKILL = Skill(
    name="generate_changelog",
    description="""Generate a CHANGELOG entry from recent git history.

    Groups recent commits by type (Added/Changed/Fixed/Removed)
    and writes a Keep-a-Changelog style entry for the current version.""",
    command="""Generate a changelog entry for {target}.
1. Inspect recent git history (mcp_git_git_log or git log via run_command)
2. Group changes: Added / Changed / Fixed / Removed / Security
3. Reference issue/PR numbers when visible
4. Write the entry under a version heading with date

Format:
## [x.y.z] - YYYY-MM-DD
### Added
- ...
### Fixed
- ...""",
    when_to_use="Before tagging a release or publishing",
    allowed_tools=("run_command", "read_file"),
    context="inline",
)

# Explain Code Skill（代码讲解，教学式）
EXPLAIN_CODE_SKILL = Skill(
    name="explain_code",
    description="""Explain how a piece of code works, step by step.

    Walks through the control flow, data structures, and design
    decisions in plain language, with line references.""",
    command="""Explain the code in {target} thoroughly.
1. Read the file and identify its purpose
2. Walk through the control flow step by step
3. Highlight key data structures and algorithms
4. Note design decisions and trade-offs
5. Point out anything unusual or hard to follow

Use line references and plain language.""",
    when_to_use="When asked to explain or teach a codebase",
    allowed_tools=("read_file", "search_text"),
    context="inline",
)

# Fix Bug Skill（系统性 bug 修复流程）
FIX_BUG_SKILL = Skill(
    name="fix_bug",
    description="""Systematically debug and fix a reported bug.

    Follows a reproduce → locate → fix → verify cycle instead of
    guessing at changes, and runs the relevant tests afterward.""",
    command="""Fix the bug in {target} systematically.
1. REPRODUCE: understand the failure and how to trigger it
2. LOCATE: trace the code path and find the root cause
3. FIX: make the minimal change that addresses the root cause
4. VERIFY: run the relevant tests (run_command) to confirm
5. REGRESSION: check no adjacent code depends on old behavior

Report root cause and the fix rationale.""",
    when_to_use="When a bug is reported or tests fail",
    allowed_tools=("read_file", "write_file", "run_command", "search_text"),
    context="inline",
)

# Performance Optimization Skill（性能优化）
OPTIMIZE_PERFORMANCE_SKILL = Skill(
    name="optimize_performance",
    description="""Analyze and optimize code performance.

    Profiles hot paths, identifies algorithmic and I/O bottlenecks,
    and proposes (or applies) optimizations with before/after
    measurements.""",
    command="""Optimize the performance of {target}.
1. Identify hot paths and bottlenecks (algorithmic complexity, I/O, loops)
2. Propose targeted optimizations with reasoning
3. Apply changes where safe
4. Measure before/after when possible (run_command)
5. Report estimated gains

Do not optimize prematurely — focus on real bottlenecks.""",
    when_to_use="When code is slow or consuming excessive resources",
    allowed_tools=("read_file", "write_file", "run_command"),
    context="inline",
)

# README Generator Skill（README 生成）
GENERATE_README_SKILL = Skill(
    name="generate_readme",
    description="""Generate or refresh a project README.

    Scans the project structure to produce a README with overview,
    quick start, usage examples, configuration, and structure.""",
    command="""Generate a README for the project at {target}.
1. Scan the project structure (list_files, read key files)
2. Summarize what the project does
3. Write sections: Overview / Quick Start / Usage / Configuration / Project Structure
4. Keep it concise and practical with real commands

Write to README.md at the project root.""",
    when_to_use="When starting a new project or README is missing/outdated",
    allowed_tools=("read_file", "write_file", "list_files", "run_command"),
    context="inline",
)


def get_builtin_skills() -> list[Skill]:
    """Return all built-in skills."""
    return [
        CODE_REVIEW_SKILL,
        TEST_WRITER_SKILL,
        REFACTOR_SKILL,
        DOCSTRING_SKILL,
        SECURITY_AUDIT_SKILL,
        COMMIT_MESSAGE_SKILL,
        GENERATE_CHANGELOG_SKILL,
        EXPLAIN_CODE_SKILL,
        FIX_BUG_SKILL,
        OPTIMIZE_PERFORMANCE_SKILL,
        GENERATE_README_SKILL,
    ]
