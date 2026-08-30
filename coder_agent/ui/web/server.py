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

from .runner import RunManager

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
    for f in files[:10]:
        try:
            first_user = next((m.get("content", "")[:50] for m in
                               load_journal(f)["messages"] if m.get("role") == "user"), "?")
        except Exception:
            first_user = "?"
        out.append({"file": str(f), "name": f.name, "task": first_user})
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
    return manager.start(req.task, mode=req.mode, model=req.model,
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

    base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
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


@app.get("/api/status")
def api_status():
    m = get_manager()
    return {"running": m.running, "model": m.model, "mode": m.mode,
            "workspace": str(m.workspace)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
