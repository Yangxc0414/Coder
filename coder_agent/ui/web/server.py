"""Web 客户端服务 — FastAPI + SSE。

启动：python -m coder_agent.ui.web.server
浏览器：http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .runner import RunManager

app = FastAPI(title="Coder-Agent Web")

WORKSPACE = Path(".").resolve()
manager = RunManager(WORKSPACE)

STATIC_DIR = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    task: str
    mode: str | None = None
    model: str | None = None
    resume: str | None = None


class ModelRequest(BaseModel):
    model: str


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/run")
def api_run(req: RunRequest):
    result = manager.start(req.task, mode=req.mode, model=req.model)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("error"))
    return result


@app.post("/api/abort")
def api_abort():
    manager.abort()
    return {"ok": True}


@app.get("/api/events")
async def api_events():
    async def gen():
        loop = asyncio.get_event_loop()
        # 在线程池里消费阻塞队列，逐条转 SSE
        for event in manager.stream_events():
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


@app.get("/api/status")
def api_status():
    return {"running": manager.running, "model": manager.model, "mode": manager.mode}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
