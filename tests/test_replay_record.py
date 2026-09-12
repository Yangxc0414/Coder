"""回归测试：演示录制/回放功能（_json 未导入导致 500 的修复）。

覆盖两个曾导致 P0 demo 失败的 bug：
1. record_start / record_event 使用 _json 但未导入 → /api/record/start 500
2. recent_replays 事件计数逻辑反向（排除以 { 开头的行）→ events 恒为 0
"""

import json
import os
import tempfile
from pathlib import Path


def test_record_start_writes_meta(tmp_path, monkeypatch):
    from coder_agent.ui.web.runner import RunManager

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    mgr = RunManager(tmp_path)
    trace = mgr.record_start("demo task", mode="full")
    p = Path(trace)
    assert p.is_file(), f"录制文件未创建: {p}"
    first = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert first.get("type") == "meta"
    assert first.get("task") == "demo task"


def test_record_event_appends(tmp_path, monkeypatch):
    from coder_agent.ui.web.runner import RunManager

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    mgr = RunManager(tmp_path)
    trace = mgr.record_start("t", mode="full")
    mgr.record_event(trace, {"kind": "AGENT_STARTED", "data": {"task": "t"}})
    lines = Path(trace).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, f"应追加 1 条事件: {lines}"
    ev = json.loads(lines[1])
    assert ev.get("kind") == "AGENT_STARTED"
    assert "timestamp" in ev


def test_recent_replays_counts_events(tmp_path, monkeypatch):
    from coder_agent.ui.web.runner import RunManager

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    mgr = RunManager(tmp_path)
    trace = mgr.record_start("t", mode="full")
    mgr.record_event(trace, {"kind": "AGENT_STARTED", "data": {}})
    mgr.record_event(trace, {"kind": "POST_TOOL_USE", "data": {}})
    replays = mgr.recent_replays()
    assert replays, "recent_replays 应返回非空"
    mine = next(r for r in replays if r["path"] == trace)
    # 事件数 = meta 行 + 2 条事件（修复前恒为 0）
    assert mine["events"] >= 3, f"事件计数错误: {mine}"
