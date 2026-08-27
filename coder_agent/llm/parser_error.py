"""Custom exception for model output format errors."""

from __future__ import annotations


class FormatError(Exception):
    """Model output did not conform to the expected format.

    This is a recoverable error — the agent should inject a correction
    prompt and retry, rather than crashing.
    """

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output
