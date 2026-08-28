"""Tests for the hook system."""

from __future__ import annotations

import pytest

from coder_agent.hooks import (
    HookEvent,
    HookRegistry,
    AGENT_ENDED,
    AGENT_STARTED,
    POST_TOOL_USE,
    PRE_TOOL_USE,
    TURN_STOPPED,
    install_logging_hooks,
    install_trace_hooks,
)


class TestHookEvent:
    """Tests for HookEvent dataclass."""

    def test_create_event(self):
        event = HookEvent("PRE_TOOL_USE", data={"tool_name": "read_file"})
        assert event.name == "PRE_TOOL_USE"
        assert event.data["tool_name"] == "read_file"

    def test_with_data_method(self):
        event = PRE_TOOL_USE.with_data(tool_name="write_file", args={"path": "a.py"})
        assert event.name == "PRE_TOOL_USE"
        assert event.data["tool_name"] == "write_file"
        assert event.data["args"] == {"path": "a.py"}

    def test_with_data_merges(self):
        event = HookEvent("TEST", data={"a": 1})
        merged = event.with_data(b=2)
        assert merged.data == {"a": 1, "b": 2}

    def test_str_representation(self):
        event = HookEvent("TEST", data={"x": 1})
        assert "TEST" in str(event)
        assert "x" in str(event)


class TestHookRegistry:
    """Tests for HookRegistry."""

    def test_register_and_fire(self):
        registry = HookRegistry()
        received = []
        registry.register("TEST", lambda e: received.append(e.data))
        registry.fire(HookEvent("TEST", data={"key": "val"}))
        assert received == [{"key": "val"}]

    def test_unregister(self):
        registry = HookRegistry()
        called = [False]

        def cb(e):
            called[0] = True

        registry.register("TEST", cb)
        registry.unregister("TEST", cb)
        registry.fire(HookEvent("TEST"))
        assert not called[0]

    def test_unregister_nonexistent(self):
        registry = HookRegistry()
        assert not registry.unregister("NOPE", lambda e: None)

    def test_multiple_listeners(self):
        registry = HookRegistry()
        results = []
        registry.register("TEST", lambda e: results.append(1))
        registry.register("TEST", lambda e: results.append(2))
        registry.fire(HookEvent("TEST"))
        assert results == [1, 2]

    def test_clear_specific_event(self):
        registry = HookRegistry()
        registry.register("A", lambda e: None)
        registry.register("B", lambda e: None)
        registry.clear("A")
        assert "A" not in registry._listeners
        assert "B" in registry._listeners

    def test_clear_all(self):
        registry = HookRegistry()
        registry.register("A", lambda e: None)
        registry.register("B", lambda e: None)
        registry.clear()
        assert not registry._listeners

    def test_hook_error_isolation(self):
        """One failing hook shouldn't break others."""
        registry = HookRegistry()
        captured: list[str] = []

        def good_cb(e):
            captured.append("good")

        def bad_cb(e):
            raise ValueError("boom")

        registry.register("TEST", good_cb)
        registry.register("TEST", bad_cb)
        registry.fire(HookEvent("TEST"))
        assert "good" in captured  # good callback still ran

    def test_pre_tool_use_constant(self):
        assert PRE_TOOL_USE.name == "PRE_TOOL_USE"

    def test_post_tool_use_constant(self):
        assert POST_TOOL_USE.name == "POST_TOOL_USE"

    def test_turn_stopped_constant(self):
        assert TURN_STOPPED.name == "TURN_STOPPED"

    def test_agent_started_constant(self):
        assert AGENT_STARTED.name == "AGENT_STARTED"

    def test_agent_ended_constant(self):
        assert AGENT_ENDED.name == "AGENT_ENDED"


class TestInstallHooks:
    """Tests for built-in hook installers."""

    def test_install_logging_hooks(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        registry = HookRegistry()
        install_logging_hooks(registry)
        registry.fire(PRE_TOOL_USE.with_data(tool_name="read_file", args={"path": "x.py"}))
        assert "PRE_TOOL_USE" in caplog.text

    def test_install_logging_hooks_post(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        registry = HookRegistry()
        install_logging_hooks(registry)
        registry.fire(POST_TOOL_USE.with_data(tool_name="write_file", success=True))
        assert "POST_TOOL_USE" in caplog.text

    def test_trace_hooks_actually_record(self):
        """Regression: trace hooks once passed wrong args to record() and the
        error was silently swallowed — hooks appeared to work but recorded
        nothing."""
        from coder_agent.trace import TraceRecorder

        trace = TraceRecorder()
        registry = HookRegistry()
        install_trace_hooks(registry, trace)

        registry.fire(PRE_TOOL_USE.with_data(tool_name="read_file", args={"path": "x"}))
        registry.fire(POST_TOOL_USE.with_data(tool_name="read_file", success=True))
        registry.fire(TURN_STOPPED.with_data(step=3))  # 'step' in payload must not collide

        events = [e["event"] for e in trace.get_entries()]
        assert events.count("hook") == 3
        assert trace.get_entries()[-1]["step"] == 3
