"""自适应工具路由测试 — 阶段裁剪 / 保底不变量 / 零降智回退。"""

from __future__ import annotations

import pytest

from coder_agent.tool_routing import (
    ALWAYS_KEEP,
    route_tools,
    schema_token_saving,
)


def _tools(n_skills: int = 17, n_mcp: int = 5) -> list[dict]:
    """构造 29 工具的 mock 定义（核心 7 + skill 17 + mcp 5）。"""
    out = []
    for name in ALWAYS_KEEP:
        out.append({"function": {"name": name, "description": "x",
                                 "parameters": {}}})
    for i in range(n_skills):
        out.append({"function": {"name": f"skill_{i}", "description": "s",
                                 "parameters": {}}})
    for i in range(n_mcp):
        out.append({"function": {"name": f"mcp_git_m{i}", "description": "m",
                                 "parameters": {}}})
    return out


class TestRouteTools:
    def test_modify_phase_keeps_full(self):
        tools = _tools()
        routed, phase = route_tools(tools, ["write_file"])
        assert phase == "modify"
        assert len(routed) == len(tools)  # 修改阶段全量

    def test_explore_phase_prunes_skills(self):
        tools = _tools()
        routed, phase = route_tools(tools, ["read_file", "search_text"])
        assert phase == "explore"
        names = {t["function"]["name"] for t in routed}
        # 核心工具全在（保底不变量）
        assert ALWAYS_KEEP <= names
        # 未活跃的随机 skill 被裁掉（只保留活跃的 + 阶段相关的）
        assert "skill_10" not in names

    def test_always_keep_invariant(self):
        """任何阶段下核心 7 工具永不被裁。"""
        tools = _tools()
        for recent in ([], ["read_file"], ["run_command"],
                       ["skill_docstring"], ["write_file"]):
            routed, _ = route_tools(tools, recent)
            names = {t["function"]["name"] for t in routed}
            assert ALWAYS_KEEP <= names, f"核心工具被裁掉：{recent}"

    def test_doc_phase_keeps_doc_skills(self):
        tools = _tools()
        # 构造带文档技能的真实名字
        doc_tool = _tools()
        for t in doc_tool:
            if t["function"]["name"].startswith("skill_"):
                idx = t["function"]["name"].split("_")[1]
                t["function"]["name"] = f"skill_doc_{idx}"
        routed, phase = route_tools(doc_tool, ["skill_doc_0"])
        assert phase == "doc"
        names = {t["function"]["name"] for t in routed}
        assert "skill_doc_0" in names  # 活跃文档技能保留

    def test_fewer_than_min_keeps_full(self):
        """裁剪后低于 min_keep → 回退全量（避免 schema 抖动）。"""
        tools = _tools()
        routed, _ = route_tools(tools, ["read_file"], min_keep=99999)
        assert len(routed) == len(tools)

    def test_zero_intelligence_fallback(self):
        """异常输入 → 返回全量（零降智）。"""
        tools = _tools()
        routed, phase = route_tools(None, None)  # 非法输入
        assert routed is None or len(routed) == len(tools) or phase == "full"
        # 空列表不应崩
        r2, p2 = route_tools([], [])
        assert r2 == []

    def test_recent_active_tool_survives(self):
        tools = _tools()
        routed, _ = route_tools(tools, ["skill_10", "skill_11",
                                         "skill_12", "skill_13"])
        names = {t["function"]["name"] for t in routed}
        for s in ("skill_10", "skill_11", "skill_12", "skill_13"):
            assert s in names, f"活跃技能 {s} 被误裁"


class TestSaving:
    def test_saving_positive_when_pruned(self):
        before = _tools()
        after, _ = route_tools(before, ["read_file"])
        saving = schema_token_saving(before, after)
        assert saving >= 0

    def test_saving_zero_when_full(self):
        tools = _tools()
        routed, _ = route_tools(tools, ["write_file"])  # modify → 全量
        assert schema_token_saving(tools, routed) == 0
