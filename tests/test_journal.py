"""Session journal tests — the writer side of /resume recovery."""

from __future__ import annotations

import json
from pathlib import Path

from coder_agent.journal import SessionJournal, load_journal, replay_state
from coder_agent.state import AgentState


class TestSessionJournalWriter:
    def test_fresh_journal_writes_meta(self, tmp_path: Path):
        path = tmp_path / "sessions" / "s1.jsonl"
        j = SessionJournal(path)
        j.close()
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        assert lines[0]["type"] == "meta"
        assert lines[0]["version"] == 1

    def test_log_message_roundtrip(self, tmp_path: Path):
        path = tmp_path / "s.jsonl"
        j = SessionJournal(path)
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "t1", "type": "function", "index": 0,
                "function": {"name": "write_file", "arguments": "{\"path\": \"a.py\"}"},
            }],
        }
        j.log_message(msg)
        j.log_message({"role": "tool", "tool_call_id": "t1", "content": "Written 1 chars"})
        j.close()

        loaded = load_journal(path)
        assert loaded["messages"] == [msg, {"role": "tool", "tool_call_id": "t1", "content": "Written 1 chars"}]

    def test_unicode_content_preserved(self, tmp_path: Path):
        path = tmp_path / "s.jsonl"
        j = SessionJournal(path)
        j.log_message({"role": "user", "content": "修复除零 bug ✓ 中文"})
        j.close()
        loaded = load_journal(path)
        assert loaded["messages"][0]["content"] == "修复除零 bug ✓ 中文"

    def test_append_mode_keeps_history(self, tmp_path: Path):
        """Resume must append to the same journal — chained recovery."""
        path = tmp_path / "s.jsonl"
        j1 = SessionJournal(path)
        j1.log_message({"role": "user", "content": "original task"})
        j1.close()

        j2 = SessionJournal(path, append=True)
        j2.log_message({"role": "user", "content": "continue please"})
        j2.close()

        loaded = load_journal(path)
        contents = [m["content"] for m in loaded["messages"]]
        assert contents == ["original task", "continue please"]
        # meta must appear exactly once (fresh journals only)
        assert sum(1 for l in path.read_text(encoding="utf-8").splitlines()
                   if json.loads(l)["type"] == "meta") == 1

    def test_log_event(self, tmp_path: Path):
        path = tmp_path / "s.jsonl"
        j = SessionJournal(path)
        j.log_event("end", reason="final_answer", steps=3)
        j.close()
        loaded = load_journal(path)
        assert loaded["events"][-1]["type"] == "end"
        assert loaded["events"][-1]["reason"] == "final_answer"


class TestReplayState:
    def test_replays_file_tracking(self):
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "t1", "type": "function", "index": 0,
                "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"},
            }]},
            {"role": "tool", "tool_call_id": "t1", "content": "..."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "t2", "type": "function", "index": 0,
                "function": {"name": "write_file", "arguments": "{\"path\": \"b.py\"}"},
            }]},
        ]
        state = AgentState()
        replay_state(messages, state)
        assert "a.py" in state.read_files
        assert "b.py" in state.modified_files

    def test_survives_malformed_arguments(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "t1", "type": "function", "index": 0,
                "function": {"name": "write_file", "arguments": "not-json{"},
            }]},
        ]
        state = AgentState()
        replay_state(messages, state)  # must not raise
        assert state.modified_files == []
