"""Trace Recorder — persistent execution trace in JSONL format.

Records every step of the Agent's execution to a file,
enabling post-hoc analysis and debugging.

Inspired by: SWE-agent's trajectory saving, mini-swe-agent's serialization.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class TraceRecorder:
    """Records Agent execution traces to a JSONL file.

    Each line is a JSON object with:
    - timestamp: ISO format
    - elapsed: seconds since trace started
    - step: agent step number
    - event: type of event
    - data: event-specific data
    """

    def __init__(self, output_path: Path | None = None) -> None:
        self.output_path = output_path
        self._entries: list[dict] = []
        self._start_time = time.time()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Clear existing file
            output_path.write_text("")

    def record(self, step: int, event: str, **data: Any) -> None:
        """Record a single trace entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(time.time() - self._start_time, 3),
            "step": step,
            "event": event,
            **data,
        }
        self._entries.append(entry)
        if self.output_path:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_metrics(self) -> dict[str, Any]:
        """Compute execution metrics from trace entries."""
        if not self._entries:
            return {"total_steps": 0, "total_tool_calls": 0, "success_rate": 0.0}

        tool_events = [e for e in self._entries if e["event"] == "tool_execution"]
        successes = [e for e in tool_events if e.get("success")]
        errors = [e for e in tool_events if not e.get("success")]

        return {
            "total_steps": len([e for e in self._entries if e["event"] == "llm_response"]),
            "total_tool_calls": len(tool_events),
            "successful_calls": len(successes),
            "failed_calls": len(errors),
            "success_rate": round(len(successes) / len(tool_events), 3) if tool_events else 0.0,
            "elapsed_seconds": round(time.time() - self._start_time, 2),
        }

    def get_entries(self) -> list[dict]:
        """Return all trace entries (for in-memory access)."""
        return list(self._entries)

    def summarize(self) -> str:
        """Generate a human-readable summary of the trace."""
        metrics = self.get_metrics()
        lines = [
            "=== Trace Summary ===",
            f"  Steps: {metrics['total_steps']}",
            f"  Tool calls: {metrics['total_tool_calls']} "
            f"({metrics['successful_calls']} ok, {metrics['failed_calls']} fail)",
            f"  Success rate: {metrics['success_rate']:.1%}",
            f"  Elapsed: {metrics['elapsed_seconds']}s",
        ]
        return "\n".join(lines)
