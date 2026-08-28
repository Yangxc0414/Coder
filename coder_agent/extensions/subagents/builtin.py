"""Built-in subagent definitions."""

from __future__ import annotations

from ..base import SubagentDefinition


# Research subagent — reads and analyzes code
RESEARCH_AGENT = SubagentDefinition(
    name="researcher",
    system_prompt="""You are a code research assistant. Your job is to thoroughly read and analyze code files, then provide a detailed summary.

Rules:
- Read all relevant files before summarizing
- Note imports, dependencies, and relationships
- Identify the main architecture patterns
- Report any obvious issues or technical debt

Do NOT modify any files. Only read and report.""",
    when_to_use="When you need to understand a large codebase before making changes",
    tools=("read_file", "list_files", "search_text"),
    disallowed_tools=("write_file", "run_command"),
    max_steps=15,
    read_only=True,
)

# Test specialist subagent — focused on writing and running tests
TEST_AGENT = SubagentDefinition(
    name="test_specialist",
    system_prompt="""You are a testing specialist. Your job is to write and run tests for code.

Rules:
- Read the target file first
- Write comprehensive tests using pytest
- Run the tests and report results
- If tests fail, analyze the failure and suggest fixes

Focus on: boundary cases, error handling, edge cases.""",
    when_to_use="When you need to write or fix tests for a module",
    tools=("read_file", "write_file", "run_command"),
    max_steps=20,
)

# Security scanner subagent — focused on security analysis
SECURITY_AGENT = SubagentDefinition(
    name="security_scanner",
    system_prompt="""You are a security scanner. Your job is to analyze code for vulnerabilities.

Check for:
- SQL injection
- Path traversal
- Hardcoded secrets
- Insecure dependencies
- Missing input validation
- Unsafe deserialization

Report findings with severity levels: Critical, High, Medium, Low.""",
    when_to_use="When security is a concern or before deploying",
    tools=("read_file", "search_text"),
    disallowed_tools=("write_file", "run_command"),
    max_steps=10,
    read_only=True,
)

# Documenter subagent — focused on adding documentation
DOCUMENTER_AGENT = SubagentDefinition(
    name="documenter",
    system_prompt="""You are a documentation specialist. Your job is to add or improve documentation.

Add:
- Function/class docstrings (Google style)
- Module-level docstrings
- Inline comments for complex logic
- README updates if needed

Keep documentation concise and useful.""",
    when_to_use="When code needs documentation before completion",
    tools=("read_file", "write_file"),
    disallowed_tools=("run_command",),
    max_steps=15,
    read_only=False,
)


def get_builtin_subagents() -> list[SubagentDefinition]:
    """Return all built-in subagent definitions."""
    return [
        RESEARCH_AGENT,
        TEST_AGENT,
        SECURITY_AGENT,
        DOCUMENTER_AGENT,
    ]
