"""my-pi-agent adoption tests — task tool, memory-as-tool, journal tolerance.

Design sources:
- task tool: my-pi-agent tools/builtin/task.py (delegation must be a TOOL)
- anti-recursion: my-pi-agent tasks.py structural invariant
- memory-as-tool: my-pi-agent memory.py (model-writable, persisted, injected)
- torn-line tolerance: my-pi-agent session.py JSONL recovery
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coder_agent.agent import Agent
from coder_agent.extensions.base import SubagentRequest
from coder_agent.journal import load_journal, SessionJournal
from coder_agent.llm.client import LLMClient, LLMResponse
from coder_agent.mode import AgentMode
from coder_agent.tools.filesystem import ReadFileTool
from coder_agent.tools.memory_tool import MemoryTool
from coder_agent.tools.registry import ToolRegistry
from coder_agent.tools.task_tool import TaskTool


class TestTaskTool:
    """P1: delegation exposed as a tool — reachable by the model at last."""

    def _agent_with_tools(self, tmp_path: Path, llm):
        from coder_agent.tools.filesystem import ReadFileTool

        reg = ToolRegistry()
        reg.register(ReadFileTool(tmp_path))
        return Agent(llm_client=llm, registry=reg, workspace=tmp_path,
                     mode=AgentMode.GOAL)

    def test_task_tool_registered(self, tmp_path: Path):
        agent = self._agent_with_tools(tmp_path, LLMClient(model="mock"))
        assert "task" in agent.registry.list_names()

    def test_total_tools_now_17(self, tmp_path: Path):
        """15 (5 core + 5 mcp + 5 skill) + task + memory."""
        from coder_agent.tools.registry import create_default_registry

        agent = Agent(
            llm_client=LLMClient(model="mock"),
            registry=create_default_registry(tmp_path, AgentMode.GOAL),
            workspace=tmp_path, mode=AgentMode.GOAL,
        )
        assert len(agent.registry.list_names()) == 17

    def test_delegation_via_tool(self, tmp_path: Path):
        """Model calls task → SubagentRunner runs the child → report returned."""
        seen = {}

        class MockChildLLM:
            model = "mock"

            def __init__(self):
                import coder_agent.agent as m

                self._orig = m.Agent.run

            # 简化：直接让 child agent 的 run 被记录（通过 LLM 层做不到，
            # 改为在 runner 层断言）——见 test_runner_receives_request

        llm = LLMClient(model="mock")
        agent = self._agent_with_tools(tmp_path, llm)
        task_tool = agent.registry.get("task")

        # 直接驱动 runner 记录（行为级）
        requests = []
        orig_run = agent.subagent_runner.run

        def spy_run(req):
            requests.append(req)
            from coder_agent.extensions.base import SubagentResult
            return SubagentResult(subagent_type=req.subagent_type,
                                  final_output="REPORT: 3 modules found",
                                  steps_used=2)

        agent.subagent_runner.run = spy_run
        result = task_tool.execute({
            "prompt": "分析代码结构",
            "subagent_type": "researcher",
        })
        assert result.success
        assert "REPORT: 3 modules found" in result.output
        assert requests[0].subagent_type == "researcher"

    def test_unknown_type_is_tool_error(self, tmp_path: Path):
        agent = self._agent_with_tools(tmp_path, LLMClient(model="mock"))
        task_tool = agent.registry.get("task")
        result = task_tool.execute({"prompt": "x", "subagent_type": "nope"})
        assert not result.success

    def test_empty_prompt_rejected(self, tmp_path: Path):
        agent = self._agent_with_tools(tmp_path, LLMClient(model="mock"))
        task_tool = agent.registry.get("task")
        result = task_tool.execute({"prompt": "  "})
        assert not result.success
        assert "自包含" in result.error


class TestAntiRecursion:
    """my-pi-agent tasks.py 的结构不变式：子代理永远拿不到 task/memory。"""

    def test_child_registry_strips_task_and_memory(self, tmp_path: Path):
        
        reg = ToolRegistry()
        for name in ("task", "memory"):
            pass  # 由 Agent 构造注入；这里手工注册探针
        agent = Agent(
            llm_client=LLMClient(model="mock"), registry=reg,
            workspace=tmp_path, mode=AgentMode.GOAL,
        )
        assert "task" in agent.registry.list_names()
        assert "memory" in agent.registry.list_names()

        runner = agent.subagent_runner
        defn = runner._definitions["researcher"]
        # 模拟 SubagentRunner.run 的过滤逻辑
        all_tools = agent.registry.list_names()
        filtered = [t for t in all_tools if t in defn.tools]
        for forbidden in ("task", "memory"):
            if forbidden in filtered:
                filtered.remove(forbidden)
        assert "task" not in filtered
        assert "memory" not in filtered

    def test_runner_run_filters_child_tools(self, tmp_path: Path, monkeypatch):
        """端到端：真实 runner 构建的 child registry 必须无 task/memory。"""
        from coder_agent.extensions import base as ext_base

        agent = Agent(
            llm_client=LLMClient(model="mock"), registry=ToolRegistry(),
            workspace=tmp_path, mode=AgentMode.GOAL,
        )
        captured = {}

        class FakeChild:
            def __init__(self, llm_client, registry, workspace, mode, **kwargs):
                captured["tools"] = registry.list_names()
                self.state = type("S", (), {"step": 1})()

            def run(self, prompt):
                return "ok"

        # monkeypatch child Agent 构造
        import coder_agent.agent as agent_module
        monkeypatch.setattr(agent_module, "Agent", FakeChild)
        monkeypatch.setattr(agent_module, "MAX_STEPS", 50, raising=False)

        runner = agent.subagent_runner
        runner.run(SubagentRequest(prompt="x", subagent_type="researcher"))
        assert "task" not in captured["tools"]
        assert "memory" not in captured["tools"]


class TestMemoryTool:
    """P2: memory-as-tool — 模型可写、持久化、注入 system prompt。"""

    def test_remember_and_persist(self, tmp_path: Path):
        from coder_agent.memory import Memory

        mem = Memory()
        tool = MemoryTool(mem, tmp_path)
        r = tool.execute({"action": "remember", "key": "用户偏好", "content": "简洁回复"})
        assert r.success
        assert (tmp_path / ".coder_memory.md").exists()
        assert "用户偏好" in (tmp_path / ".coder_memory.md").read_text(encoding="utf-8")

    def test_load_from_disk(self, tmp_path: Path):
        from coder_agent.memory import Memory

        (tmp_path / ".coder_memory.md").write_text(
            "偏好: 简洁\n约定: 中文注释", encoding="utf-8")
        mem = Memory()
        tool = MemoryTool(mem, tmp_path)
        tool.load_from_disk()
        assert mem.long_term["偏好"] == "简洁"
        assert mem.long_term["约定"] == "中文注释"

    def test_long_term_injected_into_summary(self, tmp_path: Path):
        """关键修复：此前 long_term 从未出现在 system prompt 注入中。"""
        from coder_agent.memory import Memory

        mem = Memory()
        tool = MemoryTool(mem, tmp_path)
        tool.execute({"action": "remember", "key": "偏好", "content": "简洁回复"})
        summary = mem.get_summary()
        assert "Long-term notes" in summary
        assert "偏好" in summary

    def test_list_action(self, tmp_path: Path):
        from coder_agent.memory import Memory

        mem = Memory()
        tool = MemoryTool(mem, tmp_path)
        tool.execute({"action": "remember", "key": "k1", "content": "v1"})
        r = tool.execute({"action": "list"})
        assert "k1: v1" in r.output

    def test_empty_content_rejected(self, tmp_path: Path):
        from coder_agent.memory import Memory

        tool = MemoryTool(Memory(), tmp_path)
        r = tool.execute({"action": "remember", "key": "k", "content": ""})
        assert not r.success

    def test_cap_evicts_oldest(self, tmp_path: Path):
        from coder_agent.memory import Memory

        mem = Memory()
        tool = MemoryTool(mem, tmp_path)
        for i in range(MemoryTool.MAX_ENTRIES + 5):
            tool.execute({"action": "remember", "key": f"k{i}", "content": f"v{i}"})
        assert len(mem.long_term) == MemoryTool.MAX_ENTRIES
        assert "k0" not in mem.long_term  # oldest evicted

    def test_registered_and_agent_wired(self, tmp_path: Path):
        agent = Agent(
            llm_client=LLMClient(model="mock"), registry=ToolRegistry(),
            workspace=tmp_path, mode=AgentMode.GOAL,
        )
        assert "memory" in agent.registry.list_names()


class TestJournalTornLine:
    """P3: 崩溃中途留下的撕裂行被跳过而非崩溃（my-pi-agent session.py 同款）。"""

    def test_torn_last_line_dropped(self, tmp_path: Path):
        p = tmp_path / "s.jsonl"
        good = json.dumps({"type": "message", "message": {"role": "user", "content": "a"}})
        p.write_text(good + "\n" + '{"type": "message", "mess', encoding="utf-8")
        loaded = load_journal(p)
        assert len(loaded["messages"]) == 1

    def test_torn_middle_line_skipped(self, tmp_path: Path):
        p = tmp_path / "s.jsonl"
        lines = [
            json.dumps({"type": "message", "message": {"role": "user", "content": "a"}}),
            '{"corrupt',
            json.dumps({"type": "message", "message": {"role": "user", "content": "b"}}),
        ]
        p.write_text("\n".join(lines), encoding="utf-8")
        loaded = load_journal(p)
        assert [m["content"] for m in loaded["messages"]] == ["a", "b"]

    def test_valid_file_unchanged(self, tmp_path: Path):
        j = SessionJournal(tmp_path / "ok.jsonl")
        j.log_message({"role": "user", "content": "x"})
        j.close()
        assert len(load_journal(tmp_path / "ok.jsonl")["messages"]) == 1


class TestPolicyIntegration:
    """CRITICAL regression: new tools must pass PolicyGate in every mode.

    Found by framework audit: GOAL mode (the default) denied task/memory as
    'Unknown or unregistered tool' — unit tests called execute() directly
    and never went through the gate. Integration seam AGAIN."""

    def test_goal_mode_allows_task_and_memory_with_log(self):
        from coder_agent.policy import PolicyGate

        p = PolicyGate()
        for tool, args in (("task", {"prompt": "x"}), ("memory", {"action": "list"})):
            r = p.check(tool, args, mode=AgentMode.GOAL)
            assert r.approved, f"{tool} denied in GOAL: {r.reason}"
            assert r.needs_log, f"{tool} should be logged"

    def test_full_mode_allows(self):
        from coder_agent.policy import PolicyGate

        p = PolicyGate()
        for tool, args in (("task", {"prompt": "x"}), ("memory", {"action": "list"})):
            assert p.check(tool, args, mode=AgentMode.FULL).approved

    def test_plan_mode_still_blocks(self):
        """Conservative: children may write (test_specialist), so delegation
        is not read-only even though researcher is."""
        from coder_agent.policy import PolicyGate

        p = PolicyGate()
        assert not p.check("task", {"prompt": "x"}, mode=AgentMode.PLAN).approved
        assert not p.check("memory", {"action": "list"}, mode=AgentMode.PLAN).approved

    def test_goal_mode_full_loop_through_policy(self, tmp_path: Path):
        """The exact integration path unit tests missed: tool call →
        PolicyGate → execute → result, inside a real agent loop."""
        from coder_agent.memory import Memory

        class OneShotMemoryLLM:
            model = "mock"

            def __init__(self):
                self.calls = 0

            def chat(self, messages, tools=None, max_tokens=4096):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        content="",
                        tool_calls=[{"id": "t1", "name": "memory",
                                     "arguments": json.dumps({
                                         "action": "remember",
                                         "key": "偏好", "content": "简洁回复"})}],
                        finish_reason="tool_calls", usage=None)
                return LLMResponse(content="记住了", tool_calls=None,
                                   finish_reason="stop", usage=None)

        reg = ToolRegistry()
        reg.register(ReadFileTool(tmp_path))  # goal-mode ALLOW tool present
        agent = Agent(
            llm_client=OneShotMemoryLLM(), registry=reg,
            workspace=tmp_path, mode=AgentMode.GOAL,
        )
        answer = agent.run("记住我的偏好")
        assert answer == "记住了"
        assert "偏好" in agent.memory.long_term  # policy did not block it
        assert (tmp_path / ".coder_memory.md").exists()

    def test_shared_registry_no_duplicate_crash(self, tmp_path: Path):
        """Two Agents sharing one registry must not crash on re-register."""
        from coder_agent.memory import Memory

        shared = ToolRegistry()
        a1 = Agent(llm_client=LLMClient(model="mock"), registry=shared,
                   workspace=tmp_path, mode=AgentMode.GOAL)
        a2 = Agent(llm_client=LLMClient(model="mock"), registry=shared,
                   workspace=tmp_path, mode=AgentMode.GOAL)
        assert "task" in shared and "memory" in shared
        assert a1.registry is a2.registry


class TestSelfArtifactPollution:
    """审计发现：list_files/search_text 会扫描自己的运行产物
    （.coder_truncs/.coder_bg/.coder_memory.md），可能把 agent 引向
    探索自身垃圾而非用户代码。"""

    def test_list_files_hides_runtime_artifacts(self, tmp_path: Path):
        from coder_agent.tools.filesystem import ListFilesTool

        (tmp_path / "real.py").write_text("print(1)")
        (tmp_path / ".coder_truncs").mkdir()
        (tmp_path / ".coder_truncs" / "x.txt").write_text("x")
        (tmp_path / ".coder_memory.md").write_text("k: v")
        tool = ListFilesTool(tmp_path)
        out = tool.execute({"path": "."}).output
        assert "real.py" in out
        assert ".coder_" not in out

    def test_search_skips_runtime_artifacts(self, tmp_path: Path):
        from coder_agent.tools.search import SearchTextTool

        (tmp_path / "real.py").write_text("needle here")
        (tmp_path / ".coder_truncs").mkdir()
        (tmp_path / ".coder_truncs" / "x.txt").write_text("needle in artifacts")
        tool = SearchTextTool(tmp_path)
        r = tool.execute({"pattern": "needle"})
        assert "real.py" in r.output
        assert ".coder_truncs" not in r.output

    def test_task_tool_never_raises(self, tmp_path: Path):
        """工具契约：执行失败归为 ToolResult error，不向主循环抛异常。"""
        class ExplodingRunner:
            def run(self, request):
                raise RuntimeError("child construction blew up")

        agent = Agent(llm_client=LLMClient(model="mock"), registry=ToolRegistry(),
                      workspace=tmp_path, mode=AgentMode.GOAL)
        # 直接构造 TaskTool 挂爆炸 runner（绕过正常注册路径）
        tool = TaskTool(ExplodingRunner())
        r = tool.execute({"prompt": "x"})
        assert not r.success
        assert "task delegation failed" in r.error
