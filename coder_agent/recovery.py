"""Recovery strategies — handle various error types with appropriate recovery.

Design inspired by:
- mini-swe-agent: FormatError retry with consecutive counting
- SWE-agent: autosubmit on unexpected errors, shell syntax check
- smolagents: error classification (GenerationError, ExecutionError, etc.)

Strategy matrix:
  Error Type          →  Recovery Action
  ─────────────────────────────────────────────────────
  FormatError         →  Inject correction prompt, retry
  APIConnectionError  →  Wait and retry (exponential backoff)
  APIRateLimitError   →  Wait and retry (longer delay)
  ToolExecutionError  →  Log + continue (tool already returned error)
  PolicyDenial        →  Inject alternative suggestion
  LoopDetected        →  Inject divergence prompt
  MaxStepsExceeded    →  Request summary from model
  UnexpectedError     →  Graceful exit with trajectory save
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Classification of recoverable errors."""

    FORMAT_ERROR = "format_error"
    API_CONNECTION = "api_connection"
    API_RATE_LIMIT = "api_rate_limit"
    TOOL_EXECUTION = "tool_execution"
    POLICY_DENIAL = "policy_denial"
    LOOP_DETECTED = "loop_detected"
    MAX_STEPS = "max_steps"
    UNEXPECTED = "unexpected"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    recovered: bool       # True = continue loop, False = terminate
    action: str           # What was done (for trace)
    message: str | None = None  # Message to inject (if any)


class RecoveryStrategy:
    """Handles error recovery with differentiated strategies.

    Each error type has a dedicated handler that decides:
    1. Whether to recover (continue) or terminate
    2. What message to inject into the conversation
    3. Whether to wait before retrying
    """

    # Max consecutive retries per error type
    MAX_RETRIES = {
        ErrorType.FORMAT_ERROR: 3,
        ErrorType.API_CONNECTION: 3,
        ErrorType.API_RATE_LIMIT: 5,
        ErrorType.TOOL_EXECUTION: 1,
        ErrorType.POLICY_DENIAL: 1,
        ErrorType.LOOP_DETECTED: 1,
    }

    # Delay before retry (seconds)
    RETRY_DELAY = {
        ErrorType.API_CONNECTION: 2,
        ErrorType.API_RATE_LIMIT: 5,
    }

    def __init__(self) -> None:
        self._error_counts: dict[ErrorType, int] = {}
        self._last_error_time: float = 0

    def handle(self, error: Exception, agent: "Agent") -> RecoveryResult:
        """Route error to appropriate handler and execute recovery."""
        error_type = self._classify(error)
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        count = self._error_counts[error_type]
        max_retries = self.MAX_RETRIES.get(error_type, 1)

        logger.warning(
            "Recovery: %s (attempt %d/%d): %s",
            error_type.value, count, max_retries, error,
        )

        handler = getattr(self, f"_handle_{error_type.value}", None)
        if handler:
            return handler(error, agent, count, max_retries)

        # Default: terminate
        return RecoveryResult(recovered=False, action="unknown_error")

    def _classify(self, error: Exception) -> ErrorType:
        """Classify exception into ErrorType."""
        exc_type = type(error).__name__
        exc_module = type(error).__module__
        exc_msg = str(error).lower()

        if "format" in exc_type.lower() or "json" in exc_msg:
            return ErrorType.FORMAT_ERROR
        # httpx/httpcore connection errors
        if ("connect" in exc_type.lower() or "eof" in exc_msg or
                "connection" in exc_msg or
                "httpcore" in exc_module or "httpx" in exc_module):
            return ErrorType.API_CONNECTION
        if "rate" in exc_msg or "limit" in exc_msg or "429" in exc_msg:
            return ErrorType.API_RATE_LIMIT
        if "policy" in exc_msg or "denied" in exc_msg:
            return ErrorType.POLICY_DENIAL
        if "loop" in exc_msg:
            return ErrorType.LOOP_DETECTED
        if "max steps" in exc_msg or "maximum" in exc_msg:
            return ErrorType.MAX_STEPS
        if "tool" in exc_type.lower() or "execution" in exc_type.lower():
            return ErrorType.TOOL_EXECUTION

        return ErrorType.UNEXPECTED

    # ── Handlers ─────────────────────────────────────────────

    def _handle_format_error(
        self, error: Exception, agent: "Agent", count: int, max_retries: int
    ) -> RecoveryResult:
        """Recover from model output format errors."""
        if count >= max_retries:
            return RecoveryResult(
                recovered=False,
                action="max_format_errors",
                message="Too many format errors. Please provide a final answer.",
            )
        return RecoveryResult(
            recovered=True,
            action="format_error_retry",
            message=(
                f"Your previous response had a format error: {error}. "
                "Please fix it and call the tools again with correct JSON."
            ),
        )

    def _handle_api_connection(
        self, error: Exception, agent: "Agent", count: int, max_retries: int
    ) -> RecoveryResult:
        """Recover from API connection errors with exponential backoff."""
        if count >= max_retries:
            return RecoveryResult(
                recovered=False,
                action="max_api_retries",
                message="API connection failed after multiple retries.",
            )
        delay = self.RETRY_DELAY.get(ErrorType.API_CONNECTION, 2) * (2 ** (count - 1))
        logger.info("API connection error, retrying in %ds...", delay)
        time.sleep(delay)
        return RecoveryResult(
            recovered=True,
            action="api_connection_retry",
            message=None,
        )

    def _handle_api_rate_limit(
        self, error: Exception, agent: "Agent", count: int, max_retries: int
    ) -> RecoveryResult:
        """Recover from API rate limiting."""
        if count >= max_retries:
            return RecoveryResult(
                recovered=False,
                action="max_rate_limit_retries",
                message="API rate limit exceeded after multiple retries.",
            )
        delay = self.RETRY_DELAY.get(ErrorType.API_RATE_LIMIT, 5) * count
        logger.info("API rate limited, waiting %ds...", delay)
        time.sleep(delay)
        return RecoveryResult(
            recovered=True,
            action="api_rate_limit_retry",
            message=None,
        )

    def _handle_policy_denial(
        self, error: Exception, agent: "Agent", count: int, max_retries: int
    ) -> RecoveryResult:
        """Guide model away from denied actions."""
        return RecoveryResult(
            recovered=True,
            action="policy_denial_guidance",
            message=(
                "The requested action was blocked by security policy. "
                "Try a safer alternative (e.g., use 'cat' instead of 'rm', "
                "or read the file first before modifying)."
            ),
        )

    def _handle_loop_detected(
        self, error: Exception, agent: "Agent", count: int, max_retries: int
    ) -> RecoveryResult:
        """Break out of repetitive patterns."""
        return RecoveryResult(
            recovered=True,
            action="loop_detection_guidance",
            message=(
                "You appear to be stuck in a loop, repeatedly operating on "
                "the same file(s). Review your plan and try a different approach: "
                "read other files, make the change directly, or run tests."
            ),
        )

    def _handle_max_steps(
        self, error: Exception, agent: "Agent", count: int, max_retries: int
    ) -> RecoveryResult:
        """Request summary when max steps reached."""
        return RecoveryResult(
            recovered=False,
            action="max_steps_reached",
            message=(
                "You've reached the maximum number of steps. "
                "Please provide a summary of what you've done and what remains."
            ),
        )

    def _handle_tool_execution(
        self, error: Exception, agent: "Agent", count: int, max_retries: int
    ) -> RecoveryResult:
        """Tool execution errors are already handled in _execute_tool_call.
        This handler is a fallback for unhandled tool errors."""
        return RecoveryResult(
            recovered=True,
            action="tool_execution_logged",
            message=None,
        )

    def _handle_unexpected(
        self, error: Exception, agent: "Agent", count: int, max_retries: int
    ) -> RecoveryResult:
        """Unexpected errors → graceful exit."""
        return RecoveryResult(
            recovered=False,
            action="unexpected_error",
            message=f"Unexpected error: {error}",
        )

    def reset(self) -> None:
        """Reset error counts (called at start of each run)."""
        self._error_counts = {}

    def get_stats(self) -> dict[str, int]:
        """Return error counts for tracing."""
        return dict(self._error_counts)
