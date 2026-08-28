"""Hook system — event-based extensibility at key lifecycle points.

Design inspired by OneCode's HookRegistry.

Events:
    PRE_TOOL_USE  — before a tool is executed (can block via raise)
    POST_TOOL_USE — after a tool succeeds/fails
    TURN_STOPPED  — after each turn completes (LLM response processed)
    AGENT_STARTED — when agent.run() begins
    AGENT_ENDED   — when agent.run() finishes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HookEvent:
    """A hook event with metadata."""
    name: str
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"HookEvent({self.name}, {self.data})"

    def with_data(self, **kwargs: Any) -> "HookEvent":
        """Return a new HookEvent with merged data."""
        merged = {**self.data, **kwargs}
        return HookEvent(name=self.name, data=merged)


# Event type aliases
PRE_TOOL_USE = HookEvent("PRE_TOOL_USE")
POST_TOOL_USE = HookEvent("POST_TOOL_USE")
TURN_STOPPED = HookEvent("TURN_STOPPED")
AGENT_STARTED = HookEvent("AGENT_STARTED")
AGENT_ENDED = HookEvent("AGENT_ENDED")


class HookRegistry:
    """Registry for hook callbacks.

    Usage:
        hooks = HookRegistry()
        hooks.register("PRE_TOOL_USE", my_callback)
        hooks.fire(PRE_TOOL_USE, tool_name="read_file", args={"path": "foo.py"})
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[[HookEvent], Any]]] = {}

    def register(self, event_name: str, callback: Callable[[HookEvent], Any]) -> None:
        """Register a callback for an event."""
        self._listeners.setdefault(event_name, []).append(callback)

    def unregister(self, event_name: str, callback: Callable[[HookEvent], Any]) -> bool:
        """Unregister a callback. Returns True if found and removed."""
        listeners = self._listeners.get(event_name, [])
        if callback in listeners:
            listeners.remove(callback)
            return True
        return False

    def fire(self, event: HookEvent) -> list[Any]:
        """Fire an event and collect results from all listeners."""
        results = []
        for cb in self._listeners.get(event.name, []):
            try:
                results.append(cb(event))
            except Exception as exc:
                logger.warning("Hook callback error in %s: %s", event.name, exc, exc_info=True)
        return results

    def clear(self, event_name: str | None = None) -> None:
        """Clear listeners for a specific event or all events."""
        if event_name:
            self._listeners.pop(event_name, None)
        else:
            self._listeners.clear()


# ── Built-in hook implementations ───────────────────────────

def install_logging_hooks(hooks: HookRegistry) -> None:
    """Install default logging hooks."""
    def _log_pre(event: HookEvent) -> None:
        data = event.data
        logger.info(
            "[HOOK] PRE_TOOL_USE tool=%s args=%s",
            data.get("tool_name"),
            str(data.get("args", {}))[:80],
        )

    def _log_post(event: HookEvent) -> None:
        data = event.data
        logger.info(
            "[HOOK] POST_TOOL_USE tool=%s success=%s error=%s",
            data.get("tool_name"),
            data.get("success"),
            (data.get("error") or "")[:60],
        )

    hooks.register("PRE_TOOL_USE", _log_pre)
    hooks.register("POST_TOOL_USE", _log_post)


def install_trace_hooks(hooks: HookRegistry, trace) -> None:
    """Install hooks that record events to the trace."""
    def _trace_event(event: HookEvent) -> None:
        # trace.record signature is (step, event, **data); pull step out of
        # the payload when present so it is not passed twice.
        data = dict(event.data)
        step = data.pop("step", 0)
        trace.record(step, "hook", hook_event=event.name, **data)

    hooks.register("PRE_TOOL_USE", _trace_event)
    hooks.register("POST_TOOL_USE", _trace_event)
    hooks.register("TURN_STOPPED", _trace_event)
