"""Parse and validate tool calls from a model response.

Five validation layers, each with its own actionable correction message:
  1. structure   — tool_calls present and well-formed
  2. name        — tool exists in the registry (whitelist)
  3. json        — arguments are valid JSON
  4. object      — arguments are a JSON object
  5. schema      — required args present, types match (with safe coercion),
                   unknown args stripped

Any failure raises FormatError (recoverable — the caller injects the message
as a user turn so the model can self-correct; mini-swe-agent's
format_error_template pattern, generalized per layer).

Design inspired by:
- mini-swe-agent's parse_toolcall_actions
- smolagents' validate_tool_arguments (tools.py:1361)
"""

from __future__ import annotations

import json
import logging

from .parser_error import FormatError
from .parsed_call import ParsedToolCall

logger = logging.getLogger(__name__)

# JSON-schema type system subset used by our tool definitions
_JSON_TYPES = {"string", "number", "integer", "boolean", "array", "object"}


def _coerce(value: object, expected: str) -> object:
    """Try to safely convert *value* to the expected JSON type.

    Returns the converted value, or raises ValueError when no safe
    conversion exists. Safe coercions only — no silent content changes.
    """
    if expected == "integer":
        if isinstance(value, bool):  # bool is an int subclass — reject it
            raise ValueError(f"boolean is not an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            return int(value.strip())  # may raise ValueError — intended
        raise ValueError(f"cannot convert {type(value).__name__} to integer")
    if expected == "number":
        if isinstance(value, bool):
            raise ValueError(f"boolean is not a number")
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return float(value.strip())
        raise ValueError(f"cannot convert {type(value).__name__} to number")
    if expected == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
        raise ValueError(f"cannot convert {value!r} to boolean")
    if expected == "string":
        if isinstance(value, str):
            return value
        raise ValueError(f"expected a quoted string, got {type(value).__name__}")
    if expected == "array":
        if isinstance(value, list):
            return value
        raise ValueError(f"expected a JSON array, got {type(value).__name__}")
    if expected == "object":
        if isinstance(value, dict):
            return value
        raise ValueError(f"expected a JSON object, got {type(value).__name__}")
    return value


def _validate_schema(name: str, args: dict, schema: dict) -> dict:
    """Layer 5 — validate *args* against the tool's parameter schema.

    Policy:
    - missing required argument → FormatError (the tool would fail anyway)
    - wrong type → safe coercion when possible, else FormatError
    - unknown arguments → stripped silently (models sometimes annotate;
      failing the loop over harmless extra keys costs more than it saves)

    Returns the (possibly coerced/stripped) arguments dict.
    """
    if not isinstance(schema, dict):
        return args
    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    # missing required
    for req in required:
        if req not in args:
            raise FormatError(
                f"Missing required argument '{req}' for tool '{name}'. "
                f"Required: {list(required)}. Provide it and call again.",
                raw_output=json.dumps(args, ensure_ascii=False),
            )

    # type check + safe coercion on known properties (top level only)
    validated: dict = {}
    for key, value in args.items():
        prop = properties.get(key)
        if prop is None:
            # unknown argument — strip, don't crash
            logger.debug("Stripping unknown argument '%s' for tool '%s'", key, name)
            continue
        expected = prop.get("type")
        if expected in _JSON_TYPES:
            try:
                validated[key] = _coerce(value, expected)
            except (ValueError, TypeError) as e:
                raise FormatError(
                    f"Argument '{key}' for tool '{name}' must be of type "
                    f"'{expected}' ({e}). Fix the argument and call again.",
                    raw_output=json.dumps(args, ensure_ascii=False),
                )
        else:
            validated[key] = value
    return validated


def parse_tool_calls(
    tool_calls: list[dict] | None,
    available_tools: list[str],
    schemas: dict[str, dict] | None = None,
) -> list[ParsedToolCall]:
    """Parse raw tool calls into validated ParsedToolCall objects.

    Args:
        tool_calls: Raw tool_calls from model response (list of dicts with
            'id', 'name', 'arguments' keys).
        available_tools: List of registered tool names for validation.
        schemas: Optional map of tool name → parameter JSON schema. When
            provided, layer-5 schema validation (required/types/unknown)
            runs; without it, behavior is unchanged (backward compatible).

    Returns:
        List of ParsedToolCall objects.

    Raises:
        FormatError: If any validation layer fails (caller should inject
            the message as a correction prompt).
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

        # Layer 2: tool name whitelist
        if name not in available_tools:
            raise FormatError(
                f"Unknown tool '{name}'. The tool does not exist — check the "
                f"spelling. Available tools: {available_tools}",
                raw_output=str(args_raw),
            )

        # Layer 3: JSON parse
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError as e:
            raise FormatError(
                f"Invalid JSON in tool '{name}' arguments: {e}. The arguments "
                f"string may have been cut off — resend the complete call.",
                raw_output=str(args_raw),
            )

        # Layer 4: object type
        if not isinstance(args, dict):
            raise FormatError(
                f"Tool '{name}' arguments must be a JSON object "
                f"(dict), got {type(args).__name__}. Example: "
                f"{{\"path\": \"file.py\"}}",
                raw_output=str(args_raw),
            )

        # Layer 5: parameter schema (required / types / unknown)
        if schemas and name in schemas:
            args = _validate_schema(name, args, schemas[name])

        parsed.append(ParsedToolCall(tool_name=name, arguments=args, call_id=call_id))

    return parsed
