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


def get_builtin_skills() -> list[Skill]:
    """Return all built-in skills."""
    return [
        CODE_REVIEW_SKILL,
        TEST_WRITER_SKILL,
        REFACTOR_SKILL,
        DOCSTRING_SKILL,
        SECURITY_AUDIT_SKILL,
    ]
