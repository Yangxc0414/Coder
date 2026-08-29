"""Layer-5 parameter schema validation tests (improvement A from
doc/deep_comparison.md, sourced from smolagents validate_tool_arguments
and mini-swe-agent's per-layer error templates)."""

from __future__ import annotations

import pytest

from coder_agent.llm.parser import FormatError, parse_tool_calls

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "limit": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
    },
    "required": ["path", "content"],
}


def _call(name="write_file", arguments='{"path": "a.py", "content": "x"}'):
    return [{"id": "t1", "name": name, "arguments": arguments}]


class TestLayer5SchemaValidation:
    def test_missing_required_argument(self):
        with pytest.raises(FormatError) as e:
            parse_tool_calls(
                _call(arguments='{"path": "a.py"}'),
                ["write_file"],
                schemas={"write_file": WRITE_SCHEMA},
            )
        assert "Missing required argument 'content'" in str(e.value)

    def test_valid_args_pass_through(self):
        result = parse_tool_calls(
            _call(), ["write_file"], schemas={"write_file": WRITE_SCHEMA}
        )
        assert result[0].arguments["path"] == "a.py"

    def test_unknown_arguments_stripped_not_fatal(self):
        """Design choice: models sometimes annotate with extra keys —
        stripping costs nothing, a FormatError retry loop costs steps."""
        result = parse_tool_calls(
            _call(arguments='{"path": "a.py", "content": "x", "reasoning": "because"}'),
            ["write_file"],
            schemas={"write_file": WRITE_SCHEMA},
        )
        assert "reasoning" not in result[0].arguments

    def test_type_coercion_int_to_number(self):
        result = parse_tool_calls(
            _call(arguments='{"path": "a", "content": "b", "ratio": 3}'),
            ["write_file"],
            schemas={"write_file": WRITE_SCHEMA},
        )
        assert result[0].arguments["ratio"] == 3

    def test_type_coercion_numeric_string_to_integer(self):
        result = parse_tool_calls(
            _call(arguments='{"path": "a", "content": "b", "limit": "42"}'),
            ["write_file"],
            schemas={"write_file": WRITE_SCHEMA},
        )
        assert result[0].arguments["limit"] == 42
        assert isinstance(result[0].arguments["limit"], int)

    def test_bool_rejected_for_integer(self):
        """bool is an int subclass in Python — must not silently pass."""
        with pytest.raises(FormatError) as e:
            parse_tool_calls(
                _call(arguments='{"path": "a", "content": "b", "limit": true}'),
                ["write_file"],
                schemas={"write_file": WRITE_SCHEMA},
            )
        assert "must be of type 'integer'" in str(e.value)

    def test_non_coercible_type_error_message(self):
        with pytest.raises(FormatError) as e:
            parse_tool_calls(
                _call(arguments='{"path": "a", "content": "b", "limit": "abc"}'),
                ["write_file"],
                schemas={"write_file": WRITE_SCHEMA},
            )
        assert "must be of type 'integer'" in str(e.value)

    def test_string_type_rejects_unquoted_number(self):
        with pytest.raises(FormatError) as e:
            parse_tool_calls(
                _call(arguments='{"path": 42, "content": "b"}'),
                ["write_file"],
                schemas={"write_file": WRITE_SCHEMA},
            )
        assert "must be of type 'string'" in str(e.value)

    def test_boolean_string_coercion(self):
        result = parse_tool_calls(
            _call(arguments='{"path": "a", "content": "b", "flag": "true"}'),
            ["write_file"],
            schemas={"write_file": WRITE_SCHEMA},
        )
        assert result[0].arguments["flag"] is True

    def test_no_schemas_backward_compatible(self):
        """Without schemas, behavior is exactly as before layer 5."""
        result = parse_tool_calls(_call(), ["write_file"])
        assert result[0].arguments == {"path": "a.py", "content": "x"}


class TestPerLayerMessages:
    """Each layer's correction message must be actionable."""

    def test_unknown_tool_message_lists_alternatives(self):
        with pytest.raises(FormatError) as e:
            parse_tool_calls(_call(name="read_fil"), ["read_file"])
        assert "Available tools: ['read_file']" in str(e.value)

    def test_json_error_mentions_truncation_possibility(self):
        with pytest.raises(FormatError) as e:
            parse_tool_calls(
                _call(arguments='{"path": "a.py", "cont'), ["write_file"]
            )
        assert "cut off" in str(e.value)

    def test_object_error_shows_example(self):
        with pytest.raises(FormatError) as e:
            parse_tool_calls(
                _call(arguments='["not", "an", "object"]'), ["write_file"]
            )
        assert "Example" in str(e.value)
