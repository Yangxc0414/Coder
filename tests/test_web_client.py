"""Web 客户端测试 — 框架无关的 RunManager（不启动 FastAPI 也能测）。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from coder_agent.journal import SessionJournal
from coder_agent.llm.client import LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.ui.web.runner import RunManager
from coder_agent.tools.registry import ToolRegistry


class ScriptedLLM:
    model = "mock"

    def __init__(self, answers: list[str], n_tool_calls: int = 0):
        self.answers = answers
        self.n_tool = n_tool_calls
        self.calls = 0
        self.seen_messages: list[list[dict]] = []

    def chat(self, messages, tools=None, max_tokens=4096):
        self.calls += 1
        self.seen_messages.append([dict(m) for m in messages])
        if self.n_tool > 0:
            self.n_tool -= 1
            return LLMResponse(content="thinking", tool_calls=[
                {"id": "t1", "name": "list_files", "arguments": "{}"}],
                finish_reason="tool_calls", usage=None)
        content = self.answers.pop(0) if self.answers else "done"
        return LLMResponse(content=content, tool_calls=None,
                           finish_reason="stop", usage=None)


def _factory(tmp_path: Path, llm: ScriptedLLM):
    from coder_agent.agent import Agent
    from coder_agent.tools.filesystem import ListFilesTool

    def factory():
        reg = ToolRegistry()
        reg.register(ListFilesTool(tmp_path))
        return Agent(llm_client=llm, registry=reg,
                     workspace=tmp_path, mode=AgentMode.GOAL, max_steps=5)
    return factory


def _wait_answer(mgr: RunManager, timeout: float = 10.0, run_id: str | None = None) -> dict:
    deadline = time.time() + timeout
    events = []
    while time.time() < deadline:
        evs = mgr.drain_events(run_id=run_id)
        events.extend(evs)
        if any(e["kind"] == "done" for e in evs):
            break
        time.sleep(0.05)
    kinds = [e["kind"] for e in events]
    return {"events": events, "kinds": kinds}


class TestRunManager:
    def test_run_produces_answer_and_event_stream(self, tmp_path: Path):
        llm = ScriptedLLM(answers=["RESULT_OK"])
        mgr = RunManager(tmp_path, agent_factory=_factory(tmp_path, llm))
        mgr.start("do the thing")
        out = _wait_answer(mgr)
        assert "answer" in out["kinds"]
        ans = next(e for e in out["events"] if e["kind"] == "answer")
        assert ans["answer"] == "RESULT_OK"
        assert not mgr.running

    def test_tool_events_streamed(self, tmp_path: Path):
        llm = ScriptedLLM(answers=["finished"], n_tool_calls=1)
        mgr = RunManager(tmp_path, agent_factory=_factory(tmp_path, llm))
        mgr.start("task")
        out = _wait_answer(mgr)
        assert "tool" in out["kinds"]
        tool_ev = next(e for e in out["events"] if e["kind"] == "tool")
        assert tool_ev["tool"] == "list_files"

    def test_abort_stops_running_agent(self, tmp_path: Path):
        """停止按钮语义：运行中的任务被协作式终止，不再继续。"""
        class SlowLoopLLM:
            model = "mock"
            def chat(self, messages, tools=None, max_tokens=4096):
                time.sleep(0.2)  # 让观察窗口追得上
                return LLMResponse(content="", tool_calls=[
                    {"id": "t1", "name": "list_files", "arguments": "{}"}],
                    finish_reason="tool_calls", usage=None)

        mgr = RunManager(tmp_path, agent_factory=_factory(tmp_path, SlowLoopLLM()))
        mgr.start("long task")
        time.sleep(0.6)  # 确认任务在跑
        assert mgr.running
        mgr.abort()
        out = _wait_answer(mgr, timeout=15)
        assert "done" in out["kinds"]
        blob = json.dumps(out["events"], ensure_ascii=False)
        assert "停止" in blob or "stopped" in blob
        time.sleep(0.5)
        assert not mgr.running

    def test_parallel_runs_independent(self, tmp_path: Path):
        """并行会话：两个任务可同时启动，事件流按 run_id 互相独立。"""
        llm = ScriptedLLM(answers=["X"])
        mgr = RunManager(tmp_path, agent_factory=_factory(tmp_path, llm))
        r1 = mgr.start("task A")
        assert r1["ok"]
        # 第二个任务在第一个运行中启动：不拒绝，独立 run_id
        r2 = mgr.start("task B")
        assert r2["ok"]
        assert r1["run_id"] != r2["run_id"]
        out1 = _wait_answer(mgr, run_id=r1["run_id"])
        out2 = _wait_answer(mgr, run_id=r2["run_id"])
        # 两个事件流各自独立到达 done
        assert "done" in out1["kinds"]
        assert "done" in out2["kinds"]
        assert "answer" in out1["kinds"]
        assert "answer" in out2["kinds"]
        assert not mgr.running


class TestWebResume:
    def test_resume_restores_history_to_llm(self, tmp_path: Path):
        jpath = tmp_path / "s.jsonl"
        j = SessionJournal(jpath)
        j.log_message({"role": "user", "content": "原始任务"})
        j.log_message({"role": "assistant", "content": "已完成一半"})
        j.close()

        llm = ScriptedLLM(answers=["继续完成"])
        factory = _factory(tmp_path, llm)
        mgr = RunManager(tmp_path, agent_factory=factory)
        mgr.start("继续完成剩余部分", resume_path=str(jpath))
        out = _wait_answer(mgr)

        flat = json.dumps(llm.seen_messages[0], ensure_ascii=False)
        assert "原始任务" in flat
        assert "已完成一半" in flat
        assert "继续完成剩余部分" in flat
        assert next(e["answer"] for e in out["events"] if e["kind"] == "answer") == "继续完成"
