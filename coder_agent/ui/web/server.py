"""Web 客户端服务 — FastAPI + SSE。

启动：python -m coder_agent.ui.web.server
浏览器：http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .runner import RunManager, load_config

app = FastAPI(title="Coder-Agent Web")

# 工作区状态：支持页面中"打开文件夹"切换（运行中禁止切换）
_state = {"manager": RunManager(Path(".").resolve())}


def get_manager() -> RunManager:
    return _state["manager"]


def set_workspace(path: Path) -> None:
    if get_manager().running:
        raise HTTPException(status_code=409, detail="任务运行中，无法切换工作区")
    _state["manager"] = RunManager(path.resolve())

STATIC_DIR = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    task: str
    mode: str | None = None
    model: str | None = None
    resume: str | None = None
    goal: str | None = None


class ModelRequest(BaseModel):
    model: str


class ConfigRequest(BaseModel):
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/run")
def api_run(req: RunRequest):
    result = get_manager().start(req.task, mode=req.mode, model=req.model, goal=req.goal)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("error"))
    return result


@app.post("/api/abort")
def api_abort():
    get_manager().abort()
    return {"ok": True}


@app.get("/api/events")
async def api_events():
    async def gen():
        loop = asyncio.get_event_loop()
        # 在线程池里消费阻塞队列，逐条转 SSE
        for event in get_manager().stream_events():
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/sessions")
def api_sessions():
    session_dir = Path.home() / ".coder_sessions"
    files = (sorted(session_dir.glob("session_*.jsonl"),
                    key=lambda f: f.stat().st_mtime, reverse=True)
             if session_dir.exists() else [])
    out = []
    from coder_agent.journal import load_journal
    for f in files[:20]:  # 限制返回 20 个，避免过多
        try:
            journal = load_journal(f)
            # 提取更多内容作为预览（前 100 字符）
            user_messages = [m.get("content", "") for m in journal["messages"]
                             if m.get("role") == "user"]
            first_user = (user_messages[0][:100] if user_messages else "?")
            # 如果有多个用户消息，显示总数
            task_preview = first_user
            if len(user_messages) > 1:
                task_preview += f" ... (+{len(user_messages)-1} more)"
        except Exception:
            task_preview = "?"
        out.append({"file": str(f), "name": f.name, "task": task_preview})
    return {"sessions": out}


@app.post("/api/resume")
def api_resume(req: RunRequest):
    """resume 复用 run 管线：task 作为续跑指令；resume 缺省时恢复最新会话。"""
    if not req.task:
        raise HTTPException(status_code=400, detail="续跑指令不能为空")
    resume_path = req.resume
    if not resume_path:
        session_dir = Path.home() / ".coder_sessions"
        files = (sorted(session_dir.glob("session_*.jsonl"),
                        key=lambda f: f.stat().st_mtime, reverse=True)
                 if session_dir.exists() else [])
        if not files:
            raise HTTPException(status_code=404, detail="没有可恢复的会话——先运行一次任务")
        resume_path = str(files[0])
    return get_manager().start(req.task, mode=req.mode, model=req.model,
                         resume_path=resume_path)


@app.get("/api/fs/browse")
def api_fs_browse(path: str = ""):
    """列出目录下的子目录（供"打开文件夹"导航）；path 为空时列出盘符。"""
    import string

    if not path:
        drives = [d + ":\\" for d in string.ascii_uppercase
                  if os.path.exists(d + ":\\")]
        return {"current": "", "parent": None, "dirs": drives, "drives": True}
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(status_code=404, detail="目录不存在")
    dirs = []
    try:
        for child in sorted(p.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                dirs.append(str(child))
    except (OSError, PermissionError) as e:
        raise HTTPException(status_code=403, detail=f"无法读取目录: {e}")
    parent = str(p.parent) if p.parent != p else None
    return {"current": str(p), "parent": parent, "dirs": dirs, "drives": False}


@app.post("/api/workspace")
def api_workspace(req: dict):
    target = Path(req.get("path", "")).expanduser()
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="目录不存在")
    set_workspace(target)
    m = get_manager()
    return {"ok": True, "workspace": str(m.workspace)}


@app.get("/api/workspace")
def api_get_workspace():
    return {"workspace": str(get_manager().workspace)}


@app.get("/api/tools")
def api_tools():
    return {"tools": get_manager().tool_specs()}


@app.get("/api/models")
def api_models():
    import urllib.request

    base = (os.getenv("OPENAI_BASE_URL") or "https://api.agnes-ai.cn/v1").rstrip("/")
    key = os.getenv("OPENAI_API_KEY") or ""
    try:
        req = urllib.request.Request(
            f"{base}/models", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = sorted(str(m.get("id")) for m in data.get("data", []) if m.get("id"))
        return {"models": models}
    except Exception:
        return {"models": []}


@app.get("/api/file")
def api_file(path: str):
    """只读查看工作区内文件（commonpath 防逃逸；大文件截断）。"""
    from coder_agent.tools.filesystem import _resolve_within

    m = get_manager()
    try:
        resolved = _resolve_within(m.workspace, path)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="文件不存在或为目录")
    content = resolved.read_text(encoding="utf-8", errors="replace")
    truncated = len(content) > 60_000
    return {"path": str(resolved), "content": content[:60_000],
            "truncated": truncated, "size": len(content)}


@app.get("/api/files")
def api_files(path: str = ""):
    """列出工作区内某目录的一层文件（供文件树；防逃逸）。"""
    from coder_agent.tools.filesystem import _resolve_within

    m = get_manager()
    try:
        target = _resolve_within(m.workspace, path or ".")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="不是目录")
    files = [f.name for f in sorted(target.iterdir())
             if f.is_file() and not f.name.startswith(".")]
    return {"files": files, "dir": str(target)}


@app.get("/api/status")
def api_status():
    m = get_manager()
    return {"running": m.running, "model": m.model, "mode": m.mode,
            "workspace": str(m.workspace)}


# ── API 配置（模型 / Base URL / Key）──────────────────────────────────
# 持久化到 ~/.coder_config.json，立即影响后续运行。


@app.get("/api/config")
def api_get_config():
    m = get_manager()
    cfg = load_config()
    env_url = os.getenv("OPENAI_BASE_URL")
    return {
        "model": m.model,
        "base_url": m.base_url or env_url or "https://api.agnes-ai.cn/v1",
        "api_key_set": bool(m.api_key or os.getenv("OPENAI_API_KEY")),
        "workspace": str(m.workspace),
    }


@app.post("/api/config")
def api_set_config(req: ConfigRequest):
    m = get_manager()
    m.set_config(model=req.model, base_url=req.base_url, api_key=req.api_key)
    return {"ok": True, "model": m.model}


# ── 会话详情（历史回放，按 turn 分组）─────────────────────────────────


@app.get("/api/session")
def api_session(file: str = ""):
    """加载一个历史会话，按 turn 分组返回（前端可展开查看每个 turn）。"""
    session_dir = Path.home() / ".coder_sessions"
    if not file:
        raise HTTPException(status_code=400, detail="file 参数不能为空")
    p = Path(file)
    try:
        p.resolve().relative_to(session_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="路径越界")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="会话不存在")

    from coder_agent.journal import load_journal
    data = load_journal(p)
    messages = data.get("messages") or []

    turns: list[dict] = []
    cur: dict | None = None
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            if cur and (cur.get("steps") or cur.get("answer")):
                turns.append(cur)
            cur = {"user": msg.get("content", ""), "steps": [], "answer": None}
        elif role == "assistant":
            tcs = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            if cur is None:
                cur = {"user": "", "steps": [], "answer": None}
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    cur["steps"].append({
                        "tool": fn.get("name", "?"),
                        "args": (fn.get("arguments") or "{}"),
                        "result": None, "success": True, "pending": True,
                    })
                if content:
                    cur["steps"].append({
                        "tool": "_think", "args": content,
                        "result": None, "success": True, "pending": False,
                    })
            elif content:
                cur["answer"] = content
        elif role == "tool":
            if cur and cur["steps"]:
                for st in reversed(cur["steps"]):
                    if st.get("pending"):
                        st["result"] = msg.get("content", "")
                        st["success"] = not str(msg.get("content", "")).startswith("(error)")
                        st["pending"] = False
                        break
    if cur and (cur.get("steps") or cur.get("answer")):
        turns.append(cur)

    task = turns[0]["user"] if turns else "?"
    return {"file": str(p), "task": task[:200], "turns": turns,
            "meta": data.get("meta")}


@app.get("/api/commands")
def api_commands():
    """返回可用命令列表（供前端动态加载）。"""
    return {
        "commands": [
            {"name": "/help", "desc": "显示命令帮助", "hasArgs": False},
            {"name": "/status", "desc": "显示运行状态与当前工作区", "hasArgs": False},
            {"name": "/tools", "desc": "列出模型可用的工具", "hasArgs": False},
            {"name": "/model", "desc": "切换模型（无参数显示列表，可输入名称）", "hasArgs": True},
            {"name": "/mode", "desc": "切换执行模式 full/goal/plan/dry-run", "hasArgs": True},
            {"name": "/goal", "desc": "设置会话目标（注入后续每次运行）", "hasArgs": True},
            {"name": "/sessions", "desc": "列出会话（含任务预览）", "hasArgs": False},
            {"name": "/resume", "desc": "恢复会话（无参=最新；或输入序号）", "hasArgs": True},
            {"name": "/clear", "desc": "清空对话显示", "hasArgs": False},
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
