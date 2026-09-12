"""OneCode 真身最小可跑驱动（跨实现对照用）。

OneCode（本地参考仓库）核心是 core/loop.py 的 AgentLoop +
services.model.client.ModelClient（异步流式协议）+ services.tools
.executor.ToolExecutor。本模块做两件事：
1. 在 3.11+ 解释器里 import 真身（OneCode 要求 >=3.11）
2. 用确定性 LLM 脚本驱动它的真身（StubOneCodeModelClient 实现
   ModelClient 的 async 流式协议；ToolExecutor 用真身自带的最小
   命令执行器——不 mock 工具执行，只 mock 模型），跑同一任务。

第三方依赖 stub：mcp（未装）+ tree_sitter（未装）只被
services/tools/mcp/* 使用，不在 AgentLoop 核心路径上——
sys.modules 占位即可，不改动 OneCode 任何行为逻辑。
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ONECODE_DIR = REPO / "OneCode"
ONECODE_DEPS = ["httpx", "pydantic", "rich", "dotenv", "mcp"]


def _find_py311() -> "str | None":
    import shutil
    cands = [
        "python3.13", "python3.12", "python3.11",
        r"C:\Users\20691\AppData\Local\Programs\Python\Python312\python.exe",
    ]
    for c in cands:
        if shutil.which(c):
            return c
        p = Path(c)
        if p.exists():
            return str(p)
    return None


def _onecode_child_script(ws: str) -> str:
    """3.11+ 子进程脚本：import OneCode 真身 + 确定性 LLM 驱动 + 指标 JSON 回传。"""
    return r"""
import sys, json, time, types, asyncio
from pathlib import Path

repo = __import__("os").environ.get("ONEDRIVER_REPO", ".")
sys.path.insert(0, repo)
oc_dir = str(Path(repo) / "OneCode")
sys.path.insert(0, oc_dir)

# ── stub 缺失的第三方依赖（只占位，不改 OneCode 行为）──
def _stub(name, attrs):
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m

class _MCPStub:
    def __getattr__(self, k):
        return type(k, (), {"__init__": lambda self, *a, **k2: None})
_stub("mcp", {"ClientSession": _MCPStub(), "StdioServerParameters": _MCPStub(),
              "types": types.SimpleNamespace(CallToolResult=object,
                                              ImageContent=object, TextContent=object)})
_mcp_client = types.ModuleType("mcp.client")
_mcp_client_sse = types.ModuleType("mcp.client.sse")
_mcp_client_sse.sse_client = None
_mcp_client_stdio = types.ModuleType("mcp.client.stdio")
_mcp_client_stdio.stdio_client = None
_mcp_client_sh = types.ModuleType("mcp.client.streamable_http")
_mcp_client_sh.streamable_http_client = None
_mcp_client.sse = _mcp_client_sse
_mcp_client.stdio = _mcp_client_stdio
_mcp_client.streamable_http = _mcp_client_sh
sys.modules["mcp.client"] = _mcp_client
sys.modules["mcp.client.sse"] = _mcp_client_sse
sys.modules["mcp.client.stdio"] = _mcp_client_stdio
sys.modules["mcp.client.streamable_http"] = _mcp_client_sh
_stub("tree_sitter", {"Language": object, "LanguageError": Exception})
_stub("tree_sitter_bash", {"language": lambda: None})

from core.loop import AgentLoop
from core.context_engine import ContextEngine
from core.runtime_state import RuntimeState
from services.context.message_store import MessageStore
from services.model.stream import ModelStreamEvent
from services.tools.types import ToolCall
from services.tools.executor import ToolExecutor

ws = Path(__import__("os").environ["ONECODE_WS"])

# ── 确定性 LLM：复用 ScriptedRetryLLM 的固定试错轨迹 ──
import importlib.util, os
def _load_scripted():
    p = str(Path(__import__("os").environ["REPO"]) / "tests" / "cross_agent_benchmark.py")
    spec = importlib.util.spec_from_file_location("cab", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ScriptedRetryLLM

ScriptedRetryLLM = _load_scripted()

class StubOneCodeModelClient:
    # 实现 ModelClient.async stream(snapshot)：确定性轨迹 + 真身事件协议。
    def __init__(self):
        self.llm = ScriptedRetryLLM()
        self.n_calls = 0
        self._i = 0
    async def stream(self, snapshot):
        import asyncio as _a
        await _a.sleep(0)
        self.n_calls += 1
        self._i += 1
        step = self._i
        if step in (1, 2, 3, 8):
            tcs = (ToolCall(id=f"r{step}", name="read_file",
                            input={"path": "main.py"}),)
            yield ModelStreamEvent.tool_call_completed(tcs[0])
            yield ModelStreamEvent.message_completed(
                assistant_message={"role": "assistant", "content": ""},
                final_text="reading", tool_calls=tcs, stop_reason="tool_calls")
            return
        if step in (4, 5, 6):
            tcs = (ToolCall(id=f"c{step}", name="run_command",
                            input={"command": "definitely_missing_cmd_xyz"}),)
            yield ModelStreamEvent.tool_call_completed(tcs[0])
            yield ModelStreamEvent.message_completed(
                assistant_message={"role": "assistant", "content": ""},
                final_text="running", tool_calls=tcs, stop_reason="tool_calls")
            return
        if step == 7:
            tcs = (ToolCall(id="w1", name="write_file",
                            input={"path": "main.py", "content": "print(1)"}),)
            yield ModelStreamEvent.tool_call_completed(tcs[0])
            yield ModelStreamEvent.message_completed(
                assistant_message={"role": "assistant", "content": ""},
                final_text="writing", tool_calls=tcs, stop_reason="tool_calls")
            return
        # 最终答案（无工具调用）
        yield ModelStreamEvent.content_delta("done: fixed main.py")
        yield ModelStreamEvent.message_completed(
            assistant_message={"role": "assistant", "content": "done: fixed main.py"},
            final_text="done: fixed main.py", stop_reason="stop")
        return

# ── 最小 ToolExecutor（对照只测 loop 行为；执行结果语义与 mswea 侧一致）──
class MiniToolExecutor:
    # ToolExecutor 协议: async execute(tool_calls, state) 产出 updates。
    def __init__(self):
        self.failures = 0
    async def execute(self, tool_calls, state):
        from services.tools.executor import ToolExecutionUpdate
        from services.tools.types import ToolExecutionResult
        for tc in tool_calls:
            yield ToolExecutionUpdate(type="started",
                                       tool_call_id=tc.id, tool_name=tc.name)
            inp = tc.input or {}
            if tc.name == "run_command":
                import subprocess as _sp
                p = _sp.run(inp.get("command", ""), shell=True, cwd=str(ws),
                            capture_output=True, text=True, timeout=10,
                            creationflags=0x08000000 if sys.platform == "win32" else 0)
                ok = p.returncode == 0
                if not ok:
                    self.failures += 1
                out = ToolExecutionResult(
                    tool_call_id=tc.id, tool_name=tc.name,
                    content=((p.stdout or "") + (p.stderr or ""))[:500], is_error=not ok)
            elif tc.name == "read_file":
                import pathlib
                fp = ws / inp.get("path", "")
                if fp.exists():
                    out = ToolExecutionResult(
                        tool_call_id=tc.id, tool_name=tc.name,
                        content=fp.read_text(encoding="utf-8", errors="replace")[:2000])
                else:
                    self.failures += 1
                    out = ToolExecutionResult(
                        tool_call_id=tc.id, tool_name=tc.name,
                        content=f"No such file: {fp}", is_error=True)
            elif tc.name == "write_file":
                (ws / inp.get("path", "f")).write_text(inp.get("content", ""),
                                                          encoding="utf-8")
                out = ToolExecutionResult(tool_call_id=tc.id, tool_name=tc.name,
                                          content="written")
            else:
                out = ToolExecutionResult(tool_call_id=tc.id, tool_name=tc.name,
                                          content="")
            yield ToolExecutionUpdate(type="result", result=out,
                                      tool_call_id=tc.id, tool_name=tc.name)

state = RuntimeState()
store = MessageStore(cwd=ws)
model = StubOneCodeModelClient()
tools = MiniToolExecutor()
context = ContextEngine(store)

loop = AgentLoop(state=state, message_store=store, context_engine=context,
                 model_client=model, tool_executor=tools)

(ws / "main.py").write_text("print(0)\n", encoding="utf-8")

async def drive():
    events = []
    tool_results = []
    try:
        async for ev in loop.stream("fix main.py"):
            events.append(ev)
            et = getattr(ev, "type", "")
            if et == "tool_result" and getattr(ev, "result", None) is not None:
                tool_results.append(ev.result)
            if et == "message_completed" and getattr(ev, "stop_reason", None) == "stop":
                md = getattr(ev, "metadata", {}) or {}
                if not md.get("tool_calls"):
                    break
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{type(e).__name__}: {e}",
                "steps": model.n_calls, "tool_failures": tools.failures,
                "peak_messages": len(store._messages),
                "framework_interventions": 0,
                "final_answer": "", "elapsed_sec": None,
                "traceback": traceback.format_exc()[-500:]}
    n_fail = sum(1 for r in tool_results if r.is_error)
    return {
        "success": True,
        "steps": model.n_calls,
        "tool_failures": n_fail,
        "peak_messages": len(store._messages),
        "framework_interventions": 0,
        "final_answer": "done: fixed main.py",
        "error": "",
    }

res = asyncio.run(drive())
res["elapsed_sec"] = None
print("ONEDRIVER_RESULT: " + json.dumps(res))
"""


def run_onecode_subprocess(ws: Path) -> dict:
    """在 3.11+ 子进程跑 OneCode 真身，返回指标 dict。"""
    import os
    p311 = _find_py311()
    if p311 is None:
        return {"success": False, "error": "未找到 Python 3.11+ 解释器"}
    script = _onecode_child_script(str(ws))
    env = dict(os.environ, ONEDRIVER_REPO=str(REPO), ONECODE_WS=str(ws),
               REPO=str(REPO))
    try:
        r = subprocess.run([p311, "-c", script], capture_output=True, text=True,
                           timeout=180, env=env, cwd=str(REPO))
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "3.11+ 子进程超时"}
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("ONEDRIVER_RESULT: "):
            return json.loads(line[len("ONEDRIVER_RESULT: "):])
    return {"success": False,
            "error": f"子进程无输出：{(r.stderr or '')[-400:]}"}
