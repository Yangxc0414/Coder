"""记忆自动整理测试 — 去重 / 重要性打分 / 淘汰 / 跨会话元数据。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from coder_agent.memory import Memory
from coder_agent.memory_curator import (
    MemoryCurator,
    classify_key,
    importance_score,
)
from coder_agent.tools.memory_tool import MemoryTool


class TestClassify:
    def test_preference_high(self):
        assert classify_key("用户偏好") == 3

    def test_convention_high(self):
        assert classify_key("项目约定") == 3

    def test_temporary_low(self):
        assert classify_key("当前状态") == 0

    def test_unknown_default(self):
        assert classify_key("random_key") == 1


class TestScore:
    def test_permanent_beats_temporary(self):
        now = time.time()
        s_perm = importance_score("用户偏好", 1, now, now)
        s_temp = importance_score("临时状态", 1, now, now)
        assert s_perm > s_temp

    def test_frequency_boosts(self):
        now = time.time()
        s1 = importance_score("k", 1, now, now)
        s10 = importance_score("k", 10, now, now)
        assert s10 > s1

    def test_freshness_boosts(self):
        now = time.time()
        fresh = importance_score("k", 1, now, now)
        stale = importance_score("k", 1, now - 3600, now)
        assert fresh > stale


class TestCurator:
    def test_note_write_accumulates(self, tmp_path: Path):
        mem = Memory()
        c = MemoryCurator(mem, tmp_path)
        mem.remember("k", "v")
        c.note_write("k")
        c.note_write("k")
        assert c._meta["k"]["times"] == 2

    def test_dedupe_merges_same_content(self, tmp_path: Path):
        mem = Memory()
        mem.remember("a", "same content")
        mem.remember("b", "same content")
        c = MemoryCurator(mem, tmp_path)
        # 让 a 的权重更高（偏好类）→ 保留 a 丢 b
        mem2 = Memory()
        mem2.remember("用户偏好a", "same")
        mem2.remember("临时b", "same")
        c2 = MemoryCurator(mem2, tmp_path)
        dup = c2.dedupe()
        assert dup == 1
        assert "用户偏好a" in mem2.long_term
        assert "临时b" not in mem2.long_term

    def test_consolidate_drops_lowest(self, tmp_path: Path):
        mem = Memory()
        # 塞满超过 MAX_KEEP 条：永久类 vs 临时类
        for i in range(35):
            key = f"用户偏好{i}" if i < 30 else f"临时状态{i}"
            mem.remember(key, f"v{i}")
        c = MemoryCurator(mem, tmp_path)
        res = c.consolidate()
        assert len(mem.long_term) <= MemoryCurator.MAX_KEEP
        # 临时类的应被优先淘汰
        kept_temp = [k for k in mem.long_term if k.startswith("临时状态")]
        assert len(kept_temp) < 5  # 35 条里只有 5 条临时类，多数被丢

    def test_meta_persists_across_sessions(self, tmp_path: Path):
        mem = Memory()
        mem.remember("k", "v")
        c = MemoryCurator(mem, tmp_path)
        c.note_write("k")
        # 模拟新会话：新 Memory + 新 Curator 应从盘上恢复 times
        mem2 = Memory()
        mem2.remember("k", "v")
        c2 = MemoryCurator(mem2, tmp_path)
        assert c2._meta["k"]["times"] >= 2

    def test_run_end_consolidation_hook(self, tmp_path: Path):
        mem = Memory()
        tool = MemoryTool(mem, tmp_path)
        for i in range(40):
            tool.execute({"action": "remember", "key": f"tmp_{i}",
                         "content": "x"})
        # 运行结束钩子触发整理
        tool.consolidate_on_run_end()
        assert len(mem.long_term) <= MemoryTool.MAX_ENTRIES


class TestMemoryToolActions:
    def test_curate_action(self, tmp_path: Path):
        mem = Memory()
        tool = MemoryTool(mem, tmp_path)
        for i in range(35):
            tool.execute({"action": "remember", "key": f"key_{i}",
                         "content": f"val_{i}"})
        r = tool.execute({"action": "curate"})
        assert r.success
        assert "自动整理" in r.output

    def test_list_shows_ranking(self, tmp_path: Path):
        mem = Memory()
        tool = MemoryTool(mem, tmp_path)
        tool.execute({"action": "remember", "key": "用户偏好",
                     "content": "喜欢简洁"})
        r = tool.execute({"action": "list"})
        assert r.success
        assert "重要性" in r.output
        assert "用户偏好" in r.output

    def test_file_written(self, tmp_path: Path):
        mem = Memory()
        tool = MemoryTool(mem, tmp_path)
        tool.execute({"action": "remember", "key": "k", "content": "v"})
        assert (tmp_path / ".coder_memory.md").exists()
        assert (tmp_path / ".coder_memory_meta.json").exists()
