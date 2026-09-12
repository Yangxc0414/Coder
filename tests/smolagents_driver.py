"""smolagents 真身最小可跑驱动（跨实现对照用）。

smolagents（本地参考仓库 smolagents-src/）核心是 CodeAgent +
Model（generate(messages, stop_sequences=...) → ChatMessage）+
Tool（forward + 类属性 description/name/inputs/output_type）。
CodeAgent 让模型直接写 Python 代码调用工具（python_interpreter
范式），本地 LocalPythonExecutor 执行。

本模块做两件事：
1. 在 Python 3.10+ 子进程里 import 真身（smolagents 要求 >=3.10，
   本地 3.12 已装齐依赖：huggingface_hub/requests/rich/jinja2/
   pillow/dotenv）
2. 用确定性 LLM 脚本驱动它的真身（StubSmolagentsModel 实现 Model
   的 generate()，输出 <code> 块范式；工具用真身 Tool 子类注册，
   行为与 ScriptedRetryLLM 轨迹一一对应），跑同一任务，
   结果指标经 JSON 回传（SMOLDriver_RESULT 标记行）。
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SMOLAGENTS_SRC = REPO / "smolagents-src" / "src"


def _find_py310() -> "str | None":
    import shutil
    cands = [
        "python3.13", "python3.12", "python3.11", "python3.10",
        r"C:\Users\20691\AppData\Local\Programs\Python\Python312\python.exe",
    ]
    for c in cands:
        if shutil.which(c):
            return c
        p = Path(c)
        if p.exists():
            return str(p)
    return None


def _smol_child_script(ws: str) -> str:
    """3.10+ 子进程脚本：import smolagents 真身 + 确定性 LLM 驱动 + 指标 JSON 回传。"""
    return r"""
import sys, json, time, re

repo = __import__("os").environ.get("SMOLDRIVER_REPO", ".")
smol_src = str(__import__("os").environ["SMOLDRIVER_SRC"])
sys.path.insert(0, smol_src)

from smolagents import CodeAgent
from smolagents.models import Model, ChatMessage, MessageRole
from smolagents.tools import Tool
from smolagents.monitoring import TokenUsage

ws = __import__("os").environ["SMOLDRIVER_WS"]
(ws_path := __import__("pathlib").Path(ws)).joinpath("main.py").write_text(
    "print(0)\n", encoding="utf-8")

# ── 真身 Tool 子类（只 mock 模型，工具执行走真身 LocalPythonExecutor）──
def _mktool(name, desc, inputs, fn, output_type="any"):
    return type(name, (Tool,), {
        "name": name,
        "description": desc,
        "inputs": {k: dict(v) for k, v in inputs.items()},
        "output_type": output_type,
        "forward": staticmethod(fn),
    })()

read_file = _mktool(
    "read_file", "Read the content of a file at the given path.",
    {"path": {"type": "string", "description": "File path to read"}},
    lambda path: __import__("pathlib").Path(ws_path, path).read_text(
        encoding="utf-8", errors="replace"),
    output_type="string",
)
write_file = _mktool(
    "write_file", "Write content to a file at the given path.",
    {"path": {"type": "string", "description": "File path to write"},
     "content": {"type": "string", "description": "Content to write"}},
    lambda path, content: (
        __import__("pathlib").Path(ws_path, path).write_text(
            content, encoding="utf-8"), "written"),
)
run_command = _mktool(
    "run_command", "Run a shell command and return exit code + output.",
    {"command": {"type": "string", "description": "Shell command to run"}},
    lambda command: _run_cmd(command),
)
def _run_cmd(command):
    import subprocess as _sp
    p = _sp.run(command, shell=True, cwd=str(ws_path), capture_output=True,
                text=True, timeout=10, encoding="utf-8", errors="replace",
                creationflags=0x08000000 if sys.platform == "win32" else 0)
    return f"returncode: {p.returncode}\n" + (
        ((p.stdout or "") + (p.stderr or ""))[:500] or "(no output)")

# ── 确定性 LLM：与 ScriptedRetryLLM 相同的 10 步试错轨迹，
#    但按 smolagents CodeAgent 的 <code> 块范式输出 ──
SMOL_TRAJ = {
    1:  "read_file('main.py')",
    2:  "read_file('main.py')",
    3:  "read_file('main.py')",
    4:  "run_command('definitely_missing_cmd_xyz')",
    5:  "run_command('definitely_missing_cmd_xyz')",
    6:  "run_command('definitely_missing_cmd_xyz')",
    7:  "write_file('main.py', 'print(1)\\n')",
    8:  "read_file('main.py')",
}

class StubSmolagentsModel(Model):
    model_id = "smol-mock"
    def __init__(self):
        super().__init__(model_id="smol-mock")
        self.n_calls = 0
    def generate(self, messages, **kw):
        # 与 mswea/onecode 侧同一计数语义：一次模型调用 = 一次 generate
        self.n_calls += 1
        step = self.n_calls
        if step in SMOL_TRAJ:
            code = SMOL_TRAJ[step]
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"<code>\n{code}\n</code>",
                token_usage=TokenUsage(input_tokens=0, output_tokens=0),
            )
        # 最终答案（final_answer 是真身 CodeAgent 内置终止协议）
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"<code>\nfinal_answer('done: fixed main.py')\n</code>",
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )

model = StubSmolagentsModel()
from smolagents.monitoring import AgentLogger, LogLevel
agent = CodeAgent(
    tools=[read_file, write_file, run_command],
    model=model,
    max_steps=20,
    additional_authorized_imports=[],
    logger=AgentLogger(level=LogLevel.ERROR),
)

t0 = time.time()
err = ""
answer = ""
ok = True
try:
    answer = agent.run("fix main.py")
except Exception as e:
    ok, err = False, f"{type(e).__name__}: {e}"
dt = time.time() - t0

# ── 指标：步数=模型调用次数，工具失败=动作步观测含失败信号 ──
from smolagents import ActionStep
fails = 0
for s in agent.memory.steps:
    if not isinstance(s, ActionStep):
        continue
    obs = str(getattr(s, "observations", "") or "")
    if re.search(r"returncode:\s*[1-9]|not recognized|No such file", obs):
        fails += 1
res = {
    "success": ok,
    "steps": model.n_calls,
    "tool_failures": fails,
    "peak_messages": len(agent.memory.steps) * 3,  # 每步 user+assistant+observation 3 条消息
    "framework_interventions": 0,
    "final_answer": str(answer)[:200],
    "elapsed_sec": round(dt, 2),
    "error": err,
}
print("SMOLDRIVER_RESULT: " + json.dumps(res))
"""


def run_smolagents_subprocess(ws: Path) -> dict:
    """在 3.10+ 子进程里驱动 smolagents 真身，返回指标 dict。"""
    import os
    p310 = _find_py310()
    if p310 is None:
        return {"success": False, "error": "未找到 Python 3.10+ 解释器"}
    script = _smol_child_script(str(ws))
    env = dict(os.environ,
               SMOLDRIVER_REPO=str(REPO),
               SMOLDRIVER_SRC=str(SMOLAGENTS_SRC),
               SMOLDRIVER_WS=str(ws))
    try:
        r = subprocess.run([p310, "-c", script], capture_output=True, text=True,
                           timeout=180, env=env, cwd=str(REPO))
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "3.10+ 子进程超时"}
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("SMOLDRIVER_RESULT: "):
            return json.loads(line[len("SMOLDRIVER_RESULT: "):])
    return {"success": False,
            "error": f"子进程无输出：{((r.stdout or '') + (r.stderr or ''))[-400:]}"}


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        print(json.dumps(run_smolagents_subprocess(Path(td)), ensure_ascii=False, indent=2))
