"""Parse and validate tool calls from a model response.

Design inspired by mini-swe-agent's parse_toolcall_actions:
- Raises FormatError on malformed output (recoverable)
- Validates tool name against available tools
- Validates JSON argument structure
"""

from __future__ import annotations

from .parser_error import FormatError
from .parsed_call import ParsedToolCall


def parse_tool_calls(
    tool_calls: list[dict] | None,
    available_tools: list[str],
) -> list[ParsedToolCall]:
    """Parse raw tool calls into validated ParsedToolCall objects.

    Args:
        tool_calls: Raw tool_calls from model response (list of dicts with
            'id', 'name', 'arguments' keys).
        available_tools: List of registered tool names for validation.

    Returns:
        List of ParsedToolCall objects.

    Raises:
        FormatError: If parsing fails (caller should inject correction prompt).
    """
    if not tool_calls:
        raise FormatError(
            "No tool calls found in the response. "
            "You must call at least one tool to perform actions.",
        )

    parsed: list[ParsedToolCall] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args_raw = tc.get("arguments", "{}")
        call_id = tc.get("id", "")

        # Validate tool name
        if name not in available_tools:
            raise FormatError(
                f"Unknown tool '{name}'. Available tools: {available_tools}",
                raw_output=str(args_raw),
            )

        # Parse arguments JSON
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError as e:
            raise FormatError(
                f"Invalid JSON in tool '{name}' arguments: {e}",
                raw_output=str(args_raw),
            )

        # Validate arguments is a dict
        if not isinstance(args, dict):
            raise FormatError(
                f"Tool '{name}' arguments must be a JSON object, got {type(args).__name__}",
                raw_output=str(args_raw),
            )

        parsed.append(ParsedToolCall(tool_name=name, arguments=args, call_id=call_id))

    return parsed
