"""Session journal — append-only message log that doubles as a recovery source.

Every message the agent appends to its conversation is mirrored here as one
JSONL line. The journal is therefore:

1. Runtime: nothing (zero overhead beyond one write per message)
2. After a crash / interruption: a replayable record — ``load_journal``
   rebuilds the exact message list, and ``replay_state`` re-derives file
   tracking for AgentState, letting a fresh agent continue where the old
   one stopped.

This is the "dual-use" design: the same event stream that serves
observability serves session recovery. No separate persistence layer.

Design reference: OneCode's append-only transcript resume (services/context/recovery.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .state import AgentState


class SessionJournal:
    """Writes one JSONL line per agent message, plus meta/end events.

    Args:
        path: Journal file path. Parent directories are created.
        append: When True, keep the existing file and append (used when
            resuming a session — the journal stays complete and chain-resumable).
            When False (default), truncate and start a fresh session.
    """

    def __init__(self, path: str | Path, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            self._fh = open(self.path, "a", encoding="utf-8")
        else:
            self._fh = open(self.path, "w", encoding="utf-8")
            self._write({"type": "meta", "created": time.time(), "version": 1})

    def log_message(self, message: dict[str, Any]) -> None:
        """Mirror one conversation message."""
        self._write({"type": "message", "message": message})

    def log_event(self, event_type: str, **data: Any) -> None:
        """Record a session-level event (e.g. 'end')."""
        self._write({"type": event_type, **data})

    def _write(self, obj: dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


def load_journal(path: str | Path) -> dict[str, Any]:
    """Replay a journal file.

    Returns:
        {
            "messages": [message dicts in original order],
            "meta": the session meta record (or None),
            "events": non-message records (e.g. 'end'),
        }
    """
    messages: list[dict] = []
    meta: dict | None = None
    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind == "message":
                messages.append(obj["message"])
            elif kind == "meta":
                meta = obj
            else:
                events.append(obj)
    return {"messages": messages, "meta": meta, "events": events}


def replay_state(messages: list[dict], state: AgentState) -> None:
    """Re-derive AgentState file tracking from journaled messages.

    Scans assistant tool_calls for read_file/write_file and restores
    state.mark_file_read / mark_file_modified, so loop detection and the
    system-prompt state injection work correctly after a resume.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            path = args.get("path", "")
            if not path:
                continue
            if name == "read_file":
                state.mark_file_read(path)
            elif name == "write_file":
                state.mark_file_modified(path)
