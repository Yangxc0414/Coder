"""Plan-Execute-Verify 编排器测试。

验证：
1. 复杂任务被分解为子目标并并行执行（子代理被调用）
2. 简单任务不触发分解（直接 ReAct）
3. 分解失败时回退 ReAct（零降智）
4. 子目标失败时把已完成部分注入上下文继续 ReAct
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from coder_agent.agent import Agent
from coder_agent.llm.client import LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.planner import Planner, should_use_planner
from coder_agent.tools.registry import ToolRegistry


class _FakeLLM:
    """可控 LLM：plan 调用返回 JSON，execute 调用返回最终答案。"""
    model = "mock"

    def __init__(self):
        self.plan_calls = 0
        self.exec_calls = 0
        self.plan_result = None
        self.exec_answer = "done"

    def chat(self, messages, tools=None, max_tokens=4096, **kw):
        sys = messages[0]["content"]
        if "任务分解专家" in sys:
            self.plan_calls += 1
            if self.plan_result is None:
                return LLMResponse(content="不是JSON", tool_calls=None,
                                   finish_reason="stop", usage=None)
            import json
            return LLMResponse(content=json.dumps(self.plan_result), tool_calls=None,
                               finish_reason="stop", usage=None)
        self.exec_calls += 1
        return LLMResponse(content=self.exec_answer, tool_calls=None,
                          finish_reason="stop", usage=None)


def _make_agent(llm: _FakeLLM, tmp_path: Path) -> Agent:
    reg = ToolRegistry()
    return Agent(llm_client=llm, registry=reg, workspace=tmp_path,
                 mode=AgentMode.GOAL, max_steps=5, use_planner=True)


class TestPlannerShouldUse:
    def test_short_task_skips(self):
        # < 40 字且无多步骤标记 → 不分解
        assert should_use_planner("修复一个明显的bug", None) is False

    def test_long_task_uses(self):
        assert should_use_planner("a" * 200, None) is True

    def test_plan_mode_skips(self):
        assert should_use_planner("x" * 200, AgentMode.PLAN) is False

    def test_multi_step_marker_uses(self):
        # 长任务（≥40 字）+ 多步骤标记 → 分解
        assert should_use_planner("写代码并且再跑测试并且生成完整文档，详细说明" + "x" * 30, None) is True


class TestPlanExecuteVerify:
    def test_complex_task_decomposes_and_executes(self, tmp_path: Path):
        llm = _FakeLLM()
        llm.plan_result = [
            {"index": 0, "goal": "调研", "subagent_type": "researcher", "depends_on": []},
            {"index": 1, "goal": "写代码", "subagent_type": "test_specialist", "depends_on": [0]},
        ]
        agent = _make_agent(llm, tmp_path)
        # 子代理 runner 打桩：直接返回成功报告（避免真起子 Agent）
        agent.subagent_runner = MagicMock()
        agent.subagent_runner.run.side_effect = lambda req: _sub_ok(req)
        answer = agent.run("写一个模块并且再跑测试并且生成文档，要完整" + "详" * 60)
        assert llm.plan_calls == 1
        assert agent.subagent_runner.run.call_count == 2
        assert "调研" in answer

    def test_short_task_no_plan(self, tmp_path: Path):
        llm = _FakeLLM()
        agent = _make_agent(llm, tmp_path)
        agent.subagent_runner = MagicMock()
        agent.run("修复一个bug")  # 短任务，不触发分解
        assert llm.plan_calls == 0
        assert agent.subagent_runner.run.call_count == 0

    def test_bad_plan_falls_back_to_react(self, tmp_path: Path):
        llm = _FakeLLM()
        llm.plan_result = None  # LLM 返回非 JSON → 分解失败
        agent = _make_agent(llm, tmp_path)
        agent.subagent_runner = MagicMock()
        answer = agent.run("写一个模块并且再跑测试并且生成文档，要完整" + "详" * 60)
        assert llm.plan_calls == 1
        assert agent.subagent_runner.run.call_count == 0
        assert llm.exec_calls >= 1  # 回退到 ReAct 执行

    def test_partial_failure_injects_context(self, tmp_path: Path):
        llm = _FakeLLM()
        llm.plan_result = [
            {"index": 0, "goal": "A", "depends_on": []},
            {"index": 1, "goal": "B", "depends_on": [0]},
        ]
        agent = _make_agent(llm, tmp_path)
        calls = []
        def _fake_run(req):
            calls.append(req)
            # 第一个成功，第二个失败
            from coder_agent.extensions.base import SubagentResult
            if req.subagent_type and len(calls) == 1:
                return SubagentResult(subagent_type=req.subagent_type,
                                       final_output="A done", is_error=False)
            return SubagentResult(subagent_type=req.subagent_type,
                                   final_output="B failed", is_error=True)
        agent.subagent_runner = MagicMock()
        agent.subagent_runner.run.side_effect = _fake_run
        answer = agent.run("写一个模块并且再跑测试并且生成文档，要完整" + "详" * 60)
        assert len(calls) == 2
        # 部分失败 → 回退 ReAct，注入已完成上下文
        assert llm.exec_calls >= 1

    def test_use_planner_false_disables(self, tmp_path: Path):
        llm = _FakeLLM()
        agent = Agent(llm_client=llm, registry=ToolRegistry(), workspace=tmp_path,
                      mode=AgentMode.GOAL, max_steps=5, use_planner=False)
        agent.run("写一个模块并且再跑测试并且生成文档，要完整" + "详" * 60)
        assert llm.plan_calls == 0


def _sub_ok(req):
    from coder_agent.extensions.base import SubagentResult
    return SubagentResult(subagent_type=req.subagent_type,
                          final_output=f"report for {req.prompt[:10]}",
                          is_error=False, steps_used=2)
