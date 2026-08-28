"""Context hygiene tests — regression guards for the 584K-token incident.

Real-API testing revealed that a single uncapped tool output (list_files over
a workspace containing vendored repos) blew past the model's context window
because it sat inside the "recent rounds kept full" window, beyond the reach
of round compression.
"""

from __future__ import annotations

from coder_agent.context import ContextManager
from coder_agent.llm.tokenizer import count_messages_tokens
from coder_agent.tools.filesystem import ListFilesTool, ReadFileTool
from pathlib import Path


class TestOversizedMessageCap:
    """ContextManager must cap single messages inside the recent window."""

    def _monster_messages(self) -> list[dict]:
        monster = "src/file_" + "x" * 500_000 + ".py"
        return [
            {"role": "user", "content": "list the files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "index": 0,
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": monster},
            {"role": "user", "content": "thanks"},
        ]

    def test_monster_message_stays_in_budget(self):
        cm = ContextManager(max_tokens=8000, keep_rounds=2)
        result = cm.build_messages("You are an agent.", self._monster_messages())
        tokens = count_messages_tokens(result)
        assert tokens < 8000, f"context still over budget: {tokens}"

    def test_truncation_marker_present(self):
        cm = ContextManager(max_tokens=8000, keep_rounds=2)
        result = cm.build_messages("You are an agent.", self._monster_messages())
        joined = "\n".join(str(m.get("content", "")) for m in result)
        assert "truncated by ContextManager" in joined

    def test_assistant_tool_calls_structure_preserved(self):
        """Truncation must not break assistant tool_calls <-> tool pairing."""
        cm = ContextManager(max_tokens=8000, keep_rounds=2)
        result = cm.build_messages("You are an agent.", self._monster_messages())
        assistants = [m for m in result if m.get("role") == "assistant"]
        assert assistants, "assistant message lost"
        assert assistants[0].get("tool_calls"), "tool_calls dropped"

    def test_normal_messages_untouched(self):
        cm = ContextManager(max_tokens=8000, keep_rounds=2)
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = cm.build_messages("sys", msgs)
        contents = [m.get("content") for m in result if m.get("role") != "system"]
        assert "hello" in contents
        assert "hi there" in contents

    def test_original_messages_not_mutated(self):
        cm = ContextManager(max_tokens=8000, keep_rounds=2)
        msgs = self._monster_messages()
        original_len = len(msgs[2]["content"])
        cm.build_messages("sys", msgs)
        assert len(msgs[2]["content"]) == original_len


class TestToolOutputCaps:
    """Tool layer applies coarse absolute caps as first-line defense."""

    def test_list_files_capped(self, tmp_path: Path):
        # create 500 files
        for i in range(500):
            (tmp_path / f"f_{i:03d}.txt").write_text("x")
        tool = ListFilesTool(tmp_path)
        result = tool.execute({"path": "."})
        assert result.success
        assert "shallow overview" in result.output

    def test_list_files_large_tree_shows_top_level_map(self, tmp_path: Path):
        # a deep vendored tree must not bury the top-level directories
        vendored = tmp_path / "vendor" / "lib"
        vendored.mkdir(parents=True)
        for i in range(400):
            (vendored / f"c_{i:03d}.c").write_text("int main() {}")
        (tmp_path / "tasks").mkdir()
        (tmp_path / "tasks" / "todo.txt").write_text("do it")
        tool = ListFilesTool(tmp_path)
        result = tool.execute({"path": "."})
        # top-level entries stay visible even when the tree is huge
        assert "vendor/" in result.output
        assert "tasks/" in result.output
        assert "todo.txt" not in result.output  # depth elided, not dumped

    def test_list_files_small_dir_full(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        tool = ListFilesTool(tmp_path)
        result = tool.execute({"path": "."})
        assert "truncated" not in result.output

    def test_read_file_capped(self, tmp_path: Path):
        big = tmp_path / "big.txt"
        big.write_text("y" * 100_000)
        tool = ReadFileTool(tmp_path)
        result = tool.execute({"path": "big.txt"})
        assert result.success
        assert len(result.output) < ReadFileTool.MAX_CHARS + 500
        assert "truncated" in result.output

    def test_read_file_normal_full(self, tmp_path: Path):
        f = tmp_path / "ok.txt"
        f.write_text("hello world")
        tool = ReadFileTool(tmp_path)
        result = tool.execute({"path": "ok.txt"})
        assert result.output == "hello world"


class TestDisplayFixes:
    """Empty-state display: no phantom rounds or tokens."""

    def test_zero_tokens_for_empty_messages(self):
        assert count_messages_tokens([]) == 0

    def test_zero_rounds_for_empty_messages(self):
        from coder_agent.inspector import ContextInspector

        info = ContextInspector().inspect([])
        assert info["rounds_count"] == 0
        assert info["total_messages"] == 0
