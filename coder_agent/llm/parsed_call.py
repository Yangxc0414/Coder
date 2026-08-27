"""Parsed tool call data class."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedToolCall:
    """A validated tool call extracted from a model response."""

    tool_name: str
    arguments: dict
    call_id: str
