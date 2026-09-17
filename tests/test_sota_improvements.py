"""回归测试：三项 SOTA 对齐改进（调研后吸收，见 doc 调研清单）。

1. Post-edit verification loop  （aider linter / opencode LSP 诊断思想）
   —— 写文件后立即跑廉价确定性校验，结论拼进工具输出，让模型同一步内自修。
2. Reactive compaction + 压缩限次 （OneCode reactive_compact + goose 限次）
   —— provider 报 context overflow 时反应式压缩一次再重试，封顶 2 次防死循环。
3. Structured policy denial payload （OneCode guard to_tool_error）
   —— 策略拒绝不再裸报错，而是带 reason + 具体替代路径，模型可绕行而非撞墙。

全部离线、确定性（Mock LLM，不调 API）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coder_agent.agent import Agent
from coder_agent.context import ContextManager
from coder_agent.mode import AgentMode
from coder_agent.policy import PolicyGate
from coder_agent.recovery import ErrorType, RecoveryStrategy
from coder_agent.tools.filesystem import WriteFileTool
from coder_agent.tools.postcheck import post_edit_check
from coder_agent.tools.registry import ToolRegistry
from coder_agent.tools.base import ToolResult


# ───────────────────────────────────────────────────────────
# 1. Post-edit verification loop
# ───────────────────────────────────────────────────────────
class TestPostEditCheck:
    def test_python_ok(self):
        v = post_edit_check(Path("a.py"), "def f():\n    return 1\n")
        assert v and "OK" in v

    def test_python_syntax_error(self):
        v = post_edit_check(Path("a.py"), "def f(:\n")
        assert v and "FAILED" in v and "ast.parse" in v

    def test_json_ok(self):
        v = post_edit_check(Path("a.json"), '{"a": 1}')
        assert v and "OK" in v

    def test_json_bad(self):
        v = post_edit_check(Path("a.json"), "{bad json")
        assert v and "FAILED" in v

    def test_unknown_type_no_check(self):
        assert post_edit_check(Path("a.txt"), "anything") is None

    def test_write_file_appends_verdict(self, tmp_path: Path):
        tool = WriteFileTool(tmp_path)
        r = tool.execute({"path": "ok.py", "content": "x = 1\n"})
        assert r.success
        assert "post-edit check" in r.output
        assert "OK" in r.output

    def test_write_file_reports_syntax_error(self, tmp_path: Path):
        tool = WriteFileTool(tmp_path)
        r = tool.execute({"path": "bad.py", "content": "def f(:\n"})
        # 文件仍被写入（校验不阻止写），但结果里带语法错误提示
        assert r.success
        assert "FAILED" in r.output
        assert (tmp_path / "bad.py").exists()

    def test_write_file_dry_run_skips_check(self, tmp_path: Path):
        tool = WriteFileTool(tmp_path, dry_run=True)
        r = tool.execute({"path": "x.py", "content": "def f(:\n"})
        assert "DRY RUN" in r.output
        assert "post-edit check" not in r.output


# ───────────────────────────────────────────────────────────
# 2. Reactive compaction + compaction cap
# ───────────────────────────────────────────────────────────
class TestReactiveCompaction:
    def test_scale_halves_and_caps(self):
        cm = ContextManager(max_tokens=10_000, keep_rounds=6, model="gpt-4o")
        assert cm._reactive_scale == 1.0
        assert cm.reactive_compact() is True
        assert cm._reactive_scale == 0.5
        assert cm.reactive_compact() is True
        assert cm._reactive_scale == 0.25
        # 第三次：已达上限（MAX_CONTEXT_ERROR_COMPACTIONS=2），拒绝再压
        assert cm.reactive_compact() is False
        assert cm.reactive_compactions == 2

    def test_reset(self):
        cm = ContextManager(max_tokens=10_000, keep_rounds=6, model="gpt-4o")
        cm.reactive_compact()
        cm.reactive_compact()
        cm.reset_reactive()
        assert cm._reactive_scale == 1.0
        assert cm.reactive_compactions == 0

    def test_budget_scales_with_reactive(self):
        cm = ContextManager(max_tokens=10_000, keep_rounds=2, model="gpt-4o")
        msgs = [
            {"role": "user", "content": "q"} ,
            {"role": "assistant", "content": "a"},
        ] * 6
        cm.reactive_compact()
        out = cm.build_messages("system", msgs)
        # 压缩比例生效后仍在预算内（预算被下调）
        from coder_agent.llm.tokenizer import count_messages_tokens
        assert count_messages_tokens(out, "gpt-4o") <= 10_000 * 0.5 + 1

    def test_recovery_classifies_overflow(self):
        rs = RecoveryStrategy()
        e1 = RuntimeError("This model's maximum context length is 32768")
        assert rs._classify(e1) is ErrorType.CONTEXT_OVERFLOW
        e2 = RuntimeError("prompt is too long: 200000 > 196500 tokens maximum")
        assert rs._classify(e2) is ErrorType.CONTEXT_OVERFLOW
        # 普通连接错误不能被误判成溢出
        e3 = RuntimeError("Connection reset by peer")
        assert rs._classify(e3) is not ErrorType.CONTEXT_OVERFLOW

    def test_recover_then_cap_terminates(self):
        cm = ContextManager(max_tokens=10_000, keep_rounds=6, model="gpt-4o")

        class A:  # 鸭子类型 Agent（仅需 .context）
            context = cm

        rs = RecoveryStrategy()
        agent = A()
        err = RuntimeError("maximum context length exceeded")
        r1 = rs.handle(err, agent)
        assert r1.recovered is True and r1.action == "context_reactive_compact"
        r2 = rs.handle(err, agent)
        assert r2.recovered is True
        # 第三次超上限 → 终止
        r3 = rs.handle(err, agent)
        assert r3.recovered is False
        assert r3.action == "max_context_compactions"


# ───────────────────────────────────────────────────────────
# 3. Structured policy denial payload
# ───────────────────────────────────────────────────────────
class TestStructuredPolicyDenial:
    def test_dangerous_command_has_suggestion(self):
        gate = PolicyGate()
        r = gate.check("run_command", {"command": "sudo rm -rf /"}, mode=AgentMode.FULL)
        assert not r.approved
        assert r.suggestion and "non-destructive" in r.suggestion

    def test_plan_mode_has_suggestion(self):
        gate = PolicyGate()
        r = gate.check("write_file", {"path": "x", "content": "y"}, mode=AgentMode.PLAN)
        assert not r.approved
        assert r.suggestion and "read-only" in r.suggestion

    def test_unknown_tool_has_suggestion(self):
        gate = PolicyGate()
        r = gate.check("totally_bogus_tool", {}, mode=AgentMode.GOAL)
        assert not r.approved
        assert r.suggestion

    def test_approved_has_no_suggestion(self):
        gate = PolicyGate()
        r = gate.check("read_file", {"path": "x"}, mode=AgentMode.GOAL)
        assert r.approved and r.suggestion == ""

    def test_agent_denied_message_carries_suggestion(self, tmp_path: Path):
        """agent 层：被策略拒绝的工具结果消息里带 'Suggested alternative'。"""
        class MockLLM:
            def __init__(self):
                self.calls = 0
            def chat(self, **kw):
                self.calls += 1
                class R: pass
                r = R()
                if self.calls == 1:
                    r.tool_calls = [{"id": "t1", "name": "write_file",
                                     "arguments": '{"path":"a","content":"1"}'}]
                    r.content = None
                else:
                    r.content = "Done (plan mode blocked the write)."
                    r.tool_calls = None
                r.finish_reason = "stop"
                return r

        reg = ToolRegistry()
        reg.register(WriteFileTool(tmp_path))
        llm = MockLLM()
        agent = Agent(llm_client=llm, registry=reg, workspace=tmp_path,
                      mode=AgentMode.PLAN)
        agent.run("Write a file")
        # 找到那条被拒的 tool 消息，验证它带替代建议
        denied = [m for m in agent.messages
                  if m.get("role") == "tool" and "Policy denied" in str(m.get("content"))]
        assert denied, "应存在被策略拒绝的工具消息"
        assert "Suggested alternative" in denied[0]["content"]
        assert "read-only" in denied[0]["content"]


# 防止未使用导入报错
_ = ToolResult
