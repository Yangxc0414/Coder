"""规划模板学习测试 — 跨会话计划复用（Enhancement 7）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coder_agent.planner import Planner, PlanResult, Subgoal, MIN_SUBGOALS


class _FakeLLM:
    model = "mock"

    def __init__(self, plan_items=None):
        self.calls = 0
        self.plan_items = plan_items or []

    def chat(self, messages, max_tokens=4096, **kw):
        from coder_agent.llm.client import LLMResponse
        self.calls += 1
        import json as _json
        return LLMResponse(content=_json.dumps(self.plan_items),
                           tool_calls=None, finish_reason="stop", usage=None)


def _make_plan(goals: list[tuple[str, str]]) -> PlanResult:
    subs = [Subgoal(index=i, goal=g, subagent_type=t, depends_on=[])
            for i, (g, t) in enumerate(goals)]
    tmp = Planner(llm_client=_FakeLLM())
    layers = tmp._topological_layers(subs)
    return PlanResult(subgoals=subs, layers=layers)


class TestTemplateLearning:
    def test_remember_then_recall(self, tmp_path: Path):
        llm = _FakeLLM()
        p = Planner(llm_client=llm, workspace=tmp_path)
        plan = _make_plan([("调研架构", "researcher"), ("写测试", "test_specialist")])
        p.remember_plan("写模块并跑测试并生成文档", plan, succeeded=True,
                        workspace=tmp_path)
        # 同类任务（归一化后相同签名）→ 命中模板，不调 LLM
        before = llm.calls
        recalled = p.recall_template("写模块并跑测试并生成文档")
        assert recalled is not None
        assert llm.calls == before  # 没调 LLM
        assert len(recalled.subgoals) == 2

    def test_plan_uses_cached_template(self, tmp_path: Path):
        llm = _FakeLLM()
        p = Planner(llm_client=llm, workspace=tmp_path)
        plan = _make_plan([("a", "researcher"), ("b", "test_specialist")])
        p.remember_plan("任务X", plan, succeeded=True, workspace=tmp_path)
        llm.calls = 0
        result = p.plan("任务X")
        assert result is not None
        assert llm.calls == 0  # 命中缓存，免 LLM 分解调用

    def test_different_task_still_calls_llm(self, tmp_path: Path):
        llm = _FakeLLM(plan_items=[
            {"index": 0, "goal": "g0", "subagent_type": "researcher", "depends_on": []},
            {"index": 1, "goal": "g1", "subagent_type": "documenter", "depends_on": [0]},
        ])
        p = Planner(llm_client=llm, workspace=tmp_path)
        p.remember_plan("任务A", _make_plan([("x", "researcher"), ("y", "documenter")]),
                        succeeded=True, workspace=tmp_path)
        result = p.plan("完全另一个任务描述ABC")
        assert result is not None
        assert llm.calls == 1  # 未命中模板 → 调 LLM 现分解

    def test_template_persisted_to_disk(self, tmp_path: Path):
        p = Planner(llm_client=_FakeLLM(), workspace=tmp_path)
        p.remember_plan("持久化任务", _make_plan([("a", "researcher"), ("b", "test_specialist")]),
                        succeeded=True, workspace=tmp_path)
        f = tmp_path / "plan_templates.json"
        assert f.exists()
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["templates"]) == 1

    def test_new_planner_loads_from_disk(self, tmp_path: Path):
        p1 = Planner(llm_client=_FakeLLM(), workspace=tmp_path)
        p1.remember_plan("跨会话任务", _make_plan([("a", "researcher"), ("b", "documenter")]),
                         succeeded=True, workspace=tmp_path)
        # 模拟新会话：全新 Planner 实例，从盘上载入模板
        p2 = Planner(llm_client=_FakeLLM(), workspace=tmp_path)
        recalled = p2.recall_template("跨会话任务")
        assert recalled is not None
        assert len(recalled.subgoals) == 2

    def test_failed_plan_not_remembered(self, tmp_path: Path):
        llm = _FakeLLM()
        p = Planner(llm_client=llm, workspace=tmp_path)
        p.remember_plan("失败任务", _make_plan([("a", "researcher"), ("b", "documenter")]),
                        succeeded=False, workspace=tmp_path)
        # 失败的规划不沉淀
        assert p.recall_template("失败任务") is None

    def test_normalized_signature(self, tmp_path: Path):
        """同类任务（大小写/空白差异）应归一到同一签名。"""
        p = Planner(llm_client=_FakeLLM(), workspace=tmp_path)
        plan = _make_plan([("a", "researcher"), ("b", "documenter")])
        p.remember_plan("Fix the Bug", plan, succeeded=True, workspace=tmp_path)
        # 归一化后同签名（大小写+空白不敏感）
        assert p.recall_template("fix  the   bug") is not None
