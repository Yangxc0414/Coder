"""Overflow offloading tests — context hygiene without data loss
(improvement B from doc/deep_comparison.md, sourced from opencode
truncate.ts / OneCode ToolResultStorage)."""

from __future__ import annotations

from pathlib import Path

from coder_agent.tools.filesystem import ReadFileTool
from coder_agent.tools.overflow import OFFLOAD_DIR, offload_overflow
from coder_agent.tools.search import SearchTextTool


class TestOffloadHelper:
    def test_writes_full_content(self, tmp_path: Path):
        full = "line\n" * 1000
        result = offload_overflow(tmp_path, full, source="read_file", preview_chars=100)
        assert OFFLOAD_DIR in result
        # extract the offload path from the pointer line
        pointer = [l for l in result.splitlines() if "FULL content saved" in l][0]
        rel = pointer.split(": ", 1)[1].split(" — ")[0]
        saved = tmp_path / rel
        assert saved.exists()
        assert saved.read_text(encoding="utf-8") == full

    def test_preview_and_pointer_shape(self, tmp_path: Path):
        result = offload_overflow(tmp_path, "A" * 5000, source="search_text", preview_chars=200)
        assert result.startswith("A" * 200)
        assert "output truncated for context (5000 chars total)" in result
        assert "use search_text" in result


class TestReadFileOffload:
    def test_big_read_offloads_with_pointer(self, tmp_path: Path):
        big = tmp_path / "big.txt"
        big.write_text("y" * 100_000, encoding="utf-8")
        tool = ReadFileTool(tmp_path)
        result = tool.execute({"path": "big.txt"})
        assert result.success
        assert "FULL content saved to" in result.output
        assert OFFLOAD_DIR in result.output
        # the preview cap still holds (plus the pointer tail)
        assert len(result.output) < ReadFileTool.MAX_CHARS + 600
        # the offloaded file is retrievable via search_text (the recommended
        # path — read_file on a >40K file would just offload again)
        rel = [l for l in result.output.splitlines() if "FULL content saved" in l][0]
        rel_path = rel.split(": ", 1)[1].split(" — ")[0]
        searcher = SearchTextTool(tmp_path)
        found = searcher.execute({"pattern": "yyy", "path": rel_path, "max_results": 3})
        assert found.success
        assert "yyy" in found.output

    def test_small_read_untouched(self, tmp_path: Path):
        (tmp_path / "ok.txt").write_text("hello")
        tool = ReadFileTool(tmp_path)
        result = tool.execute({"path": "ok.txt"})
        assert result.output == "hello"
        assert not (tmp_path / OFFLOAD_DIR).exists()


class TestSearchOffload:
    def test_many_matches_offload_pointer(self, tmp_path: Path):
        for i in range(50):
            (tmp_path / f"f_{i:02d}.txt").write_text("needle here\n" * 200)
        tool = SearchTextTool(tmp_path)
        result = tool.execute({"pattern": "needle", "max_results": 300})
        assert result.success
        if "FULL content saved" in result.output:
            assert OFFLOAD_DIR in result.output
            assert "search_text" in result.output
