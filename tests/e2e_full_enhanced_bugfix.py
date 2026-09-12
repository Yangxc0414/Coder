"""真实端到端 bug-fix 基准：全增强 Agent 在确定性 LLM 下修多 bug 任务。

补"效果很好"缺口——跨实现对照只测**机制**（同任务 5 配置对照），
本测试补**端到端真实任务**：一个多 bug 的 calculator.py（整数除法、
除零返回 None 而非抛异常），全增强 Agent 必须：
1. 定位并修复两个 bug；
2. 修复后跑 `pytest` 确认全绿；
3. 代码行为正确（divide(1,0) 抛 ZeroDivisionError，add(1,2)=3）。

用确定性 LLM（ScriptedBugFixLLM）驱动，模型变量被控制，
与跨实现对照口径一致，结果可复现、可审计。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coder_agent.agent import Agent
from coder_agent.mode import AgentMode
from coder_agent.tools.registry import create_default_registry
from tests.cross_agent_benchmark import ScriptedRetryLLM


class ScriptedBugFixLLM(ScriptedRetryLLM):
    """全增强 Agent 在同一多 bug 任务上的固定轨迹。

    确定性脚本（不依赖外部 LLM）：
      1-2  读 calculator.py / test_calculator.py（定位）
      3    写 calculator.py（修两个 bug：除零改 raise，// 改 /）
      4    跑 pytest 验证
      5    若 pytest 通过 → 最终答案；否则重读/重写 → 最多 2 轮
    分支读最近 tool 观察（pytest 结果）——像真 agent 一样"看反馈再决定"，
    证明全增强配置的循环在真实任务上收敛，并覆盖检测/失败路径。
    """

    FIXED = (
        "def add(a, b): return a + b\n"
        "def subtract(a, b): return a - b\n"
        "def multiply(a, b): return a * b\n"
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        raise ZeroDivisionError('division by zero')\n"
        "    return a / b\n"
    )

    def _last_tool_obs(self, messages):
        """取最近一条 role=tool 的观察文本。"""
        last = ""
        for m in reversed(messages or []):
            role = m.get("role")
            if role in ("tool", "tool_result"):
                last = str(m.get("content", ""))
                break
        return last

    def _pytest_passed(self, obs: str) -> bool:
        import re
        m = re.search(r"(\d+) passed", obs)
        fails = len(re.findall(r"(failed|error)", obs, re.I))
        passed = m is not None and int(m.group(1)) >= 5
        return passed and fails == 0

    def chat(self, messages, tools=None, max_tokens=4096, **kw):
        from coder_agent.llm.client import LLMResponse
        self.chat_calls += 1
        self.i += 1
        step = self.i
        obs = self._last_tool_obs(messages)

        if step == 1:
            return LLMResponse(content="reading calculator.py",
                tool_calls=[{"id": "r1", "name": "read_file",
                    "arguments": json.dumps({"path": "calculator.py"})}],
                finish_reason="tool_calls", usage=None)
        if step == 2:
            return LLMResponse(content="reading test_calculator.py",
                tool_calls=[{"id": "r2", "name": "read_file",
                    "arguments": json.dumps({"path": "test_calculator.py"})}],
                finish_reason="tool_calls", usage=None)
        # 写修复（第 3 步，及第 6 步若上轮 pytest 失败则重写）
        if step in (3, 6, 9):
            return LLMResponse(content="fixing both bugs",
                tool_calls=[{"id": f"w{step}", "name": "write_file",
                    "arguments": json.dumps(
                        {"path": "calculator.py", "content": self.FIXED})}],
                finish_reason="tool_calls", usage=None)
        # 跑 pytest（第 4 步，及第 7 步若上轮失败）
        if step in (4, 7, 10):
            return LLMResponse(content="running pytest",
                tool_calls=[{"id": f"c{step}", "name": "run_command",
                    "arguments": json.dumps(
                        {"command": "python -m pytest -q"})}],
                finish_reason="tool_calls", usage=None)
        # 观察步：看 pytest 结果再决定（第 5/8/11 步）
        if self._pytest_passed(obs):
            return LLMResponse(
                content="done: fixed calculator.py, all tests pass",
                tool_calls=None, finish_reason="stop", usage=None)
        # pytest 未通过 → 继续重读/重写（回到 6）
        return LLMResponse(
            content="tests not green, re-checking",
            tool_calls=[{"id": f"r{step}", "name": "read_file",
                "arguments": json.dumps({"path": "calculator.py"})}],
            finish_reason="tool_calls", usage=None)


def _seed_task(ws: Path) -> None:
    (ws / "calculator.py").write_text(
        "def add(a, b): return a + b\n"
        "def subtract(a, b): return a - b\n"
        "def multiply(a, b): return a * b\n"
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        return None  # BUG: should raise ZeroDivisionError\n"
        "    return a // b  # BUG: int division, should be true division\n",
        encoding="utf-8")
    (ws / "test_calculator.py").write_text(
        "import pytest\n"
        "from calculator import add, subtract, multiply, divide\n"
        "\n"
        "def test_add(): assert add(1, 2) == 3\n"
        "def test_subtract(): assert subtract(5, 3) == 2\n"
        "def test_multiply(): assert multiply(2, 3) == 6\n"
        "def test_divide_true(): assert divide(7, 2) == 3.5\n"
        "def test_divide_by_zero():\n"
        "    with pytest.raises(ZeroDivisionError):\n"
        "        divide(1, 0)\n",
        encoding="utf-8")


def run_full_enhanced_bug_fix() -> dict:
    """全增强 Agent 跑真实多 bug 任务，返回可审计指标。"""
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _seed_task(ws)
        agent = Agent(
            llm_client=ScriptedBugFixLLM(),
            registry=create_default_registry(ws, AgentMode.FULL, with_extensions=False),
            workspace=ws, mode=AgentMode.FULL, max_steps=12, use_planner=True,
        )
        answer = agent.run(
            "修复 calculator.py 的两个 bug：1) divide 在除零时应抛 "
            "ZeroDivisionError 而非返回 None；2) divide 应做真除法而非 //。"
            "修复后运行 pytest 确认全部测试通过。"
        )

        # 代码行为验证（ground truth，不信任模型自述）
        code_ok = True
        import importlib.util
        spec = importlib.util.spec_from_file_location("calculator", ws / "calculator.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        try:
            assert mod.divide(7, 2) == 3.5, "true division failed"
        except Exception as e:
            code_ok = False
        try:
            mod.divide(1, 0)
            code_ok = False  # 没抛异常
        except ZeroDivisionError:
            pass
        except Exception:
            code_ok = False

        # pytest 是否通过（独立进程，ground truth）
        p = subprocess.run(
            [sys.executable, "-m", "pytest", str(ws / "test_calculator.py"), "-q"],
            capture_output=True, text=True, timeout=30, cwd=str(ws))
        pytest_ok = p.returncode == 0

        entries = agent.trace.get_entries()
        interventions = sum(
            1 for e in entries if e.get("event") in
            ("tool_fallback_hint", "command_strategy_hint", "plan_created"))
        return {
            "code_behavior_correct": code_ok,
            "pytest_passed": pytest_ok,
            "final_answer": answer,
            "interventions": interventions,
            "success": code_ok and pytest_ok,
        }


if __name__ == "__main__":
    res = run_full_enhanced_bug_fix()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    sys.exit(0 if res["success"] else 1)
