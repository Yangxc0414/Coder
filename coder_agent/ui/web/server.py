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


class ReplayRequest(BaseModel):
    trace: str  # trace 文件路径


class ModelRequest(BaseModel):
    model: str


class ConfigRequest(BaseModel):
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    context_window: int | None = None
    context_ratio: float | None = None
    keep_rounds: int | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/run")
def api_run(req: RunRequest):
    result = get_manager().start(req.task, mode=req.mode, model=req.model, goal=req.goal)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("error"))
    # 演示录制：start() 返回后，hook 会自动写入 trace（runner.py 中注入）
    m = get_manager()
    if getattr(m, "_recording_path", None):
        # 记录任务元信息到 trace 头部（如果还没有 meta）
        import json as _json, datetime as _dt, pathlib as _P
        p = _P.Path(m._recording_path)
        first_line = p.read_text(encoding="utf-8").splitlines()[0]
        try:
            meta = _json.loads(first_line)
            if meta.get("type") != "meta":
                meta = {}
        except Exception:
            meta = {}
        meta.update({
            "type": "meta",
            "task": req.task[:200],
            "mode": req.mode or m.mode,
            "model": req.model or m.model,
            "workspace": str(m.workspace),
            "started": _dt.datetime.now().isoformat(),
        })
        rest = p.read_text(encoding="utf-8").splitlines()[1:]
        p.write_text(
            _json.dumps(meta, ensure_ascii=False) + "\n" + "\n".join(rest),
            encoding="utf-8")
    return result


@app.post("/api/abort")
def api_abort(req: dict | None = None):
    """停止指定 run_id 的运行；缺省停止全部。"""
    run_id = (req or {}).get("run_id") if req else None
    if not run_id:
        run_id = None
    get_manager().abort(run_id=run_id)
    return {"ok": True}


@app.get("/api/events")
async def api_events(run_id: str | None = None):
    async def gen():
        loop = asyncio.get_event_loop()
        # 在线程池里消费指定会话的阻塞队列，逐条转 SSE
        for event in get_manager().stream_events(run_id=run_id):
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
            workspace = (journal.get("meta") or {}).get("workspace")
        except Exception:
            task_preview = "?"
            workspace = None
        out.append({"file": str(f), "name": f.name, "task": task_preview,
                    "workspace": workspace})
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
    """列出目录下的子目录与文件（供网页内文件夹选择器导航）。

    path 为空时列出盘符。文件列表用于展示（不可作为工作区选择）。
    """
    import string

    if not path:
        drives = [d + ":\\" for d in string.ascii_uppercase
                  if os.path.exists(d + ":\\")]
        return {"current": "", "parent": None, "dirs": drives,
                "drives": True, "files": []}
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(status_code=404, detail="目录不存在")
    dirs = []
    files = []
    try:
        for child in sorted(p.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                dirs.append(str(child))
            else:
                files.append(child.name)
    except (OSError, PermissionError) as e:
        raise HTTPException(status_code=403, detail=f"无法读取目录: {e}")
    parent = str(p.parent) if p.parent != p else None
    return {"current": str(p), "parent": parent, "dirs": dirs,
            "drives": False, "files": files}


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
    return get_manager().tool_specs()  # {"tools": [...], "counts": {...}}


@app.get("/api/extensions")
def api_extensions():
    """扩展系统详情：Skills 与 MCP 工具定义（/skills /mcp 命令数据源）。"""
    return get_manager().extensions_info()


class ExtToggleRequest(BaseModel):
    kind: str      # "skill" | "mcp"
    name: str      # 工具名（如 skill_code_review / mcp_git_git_log）
    enabled: bool


@app.post("/api/extensions/toggle")
def api_extensions_toggle(req: ExtToggleRequest):
    """启用/禁用扩展（持久化到 ~/.coder_config.json，下次运行生效）。"""
    return get_manager().toggle_extension(req.kind, req.name, req.enabled)


@app.get("/api/models")
def api_models():
    # 复用 LLMClient 的配置解析（用户配置 > 环境变量）与 HTTP 客户端
    from coder_agent.llm.client import LLMClient
    m = get_manager()
    client = LLMClient(model=m.model, api_key=m.api_key, base_url=m.base_url)
    return {"models": client.list_models()}


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


@app.post("/api/fs/pick")
def api_fs_pick():
    """弹出系统原生文件夹选择对话框（PowerShell FolderBrowserDialog）。

    浏览器出于安全无法直接打开系统目录选择器，这里通过后端进程弹出。
    用户取消时返回 ok=False，前端可回退到自绘浏览器。
    """
    import subprocess
    ps_script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "Add-Type -AssemblyName System.Drawing;"
        # 置顶 owner 窗体（可见 1x1、屏幕外、无边框）→ 对话框显示在最前
        "$w = New-Object System.Windows.Forms.Form;"
        "$w.TopMost = $true;"
        "$w.ShowInTaskbar = $false;"
        "$w.FormBorderStyle = 'None';"
        "$w.Opacity = 0.01;"
        "$w.Size = New-Object System.Drawing.Size(1,1);"
        "$w.StartPosition = 'Manual';"
        "$w.Location = New-Object System.Drawing.Point(-32000,-32000);"
        "$w.Show();"
        "$w.Activate();"
        "$f = New-Object System.Windows.Forms.FolderBrowserDialog;"
        "$f.Description = '选择工作区文件夹';"
        "$f.ShowNewFolderButton = $true;"
        "if ($f.ShowDialog($w) -eq [System.Windows.Forms.DialogResult]::OK) "
        "{ Write-Output $f.SelectedPath } else { Write-Output '' };"
        "$w.Close();"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
            capture_output=True, text=True, timeout=180)
        picked = (r.stdout or "").strip()
        if picked and os.path.isdir(picked):
            return {"ok": True, "path": picked}
        return {"ok": False, "detail": "未选择文件夹"}
    except Exception as e:
        return {"ok": False, "detail": f"系统对话框不可用: {e}"}


# ── 运行上下文 / 压缩 / 追踪（对应 CLI 的 /history /compact /trace）──────


@app.get("/api/context")
def api_context(run_id: str | None = None):
    m = get_manager()
    return {"running": m.running,
            "messages": m.agent_summary(run_id=run_id),
            "run_id": run_id or m._last_run_id}


@app.post("/api/compact")
def api_compact(req: dict | None = None):
    m = get_manager()
    run_id = (req or {}).get("run_id") if req else None
    result = m.compact_last(run_id=run_id)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("error"))
    return result


@app.get("/api/trace")
def api_trace(run_id: str | None = None):
    m = get_manager()
    entries = m.trace_summary(run_id=run_id)
    return {"entries": entries or [], "run_id": run_id or m._last_run_id}


@app.get("/api/status")
def api_status():
    m = get_manager()
    return {"running": m.running, "model": m.model, "mode": m.mode,
            "workspace": str(m.workspace), "runs": m.runs_status()}


# ── API 配置（模型 / Base URL / Key）──────────────────────────────────
# 持久化到 ~/.coder_config.json，立即影响后续运行。


@app.get("/api/config")
def api_get_config():
    m = get_manager()
    cfg = load_config()
    env_url = os.getenv("OPENAI_BASE_URL")
    info = m.context_info()
    return {
        "model": m.model,
        "base_url": m.base_url or env_url or "https://api.agnes-ai.cn/v1",
        "api_key_set": bool(m.api_key or os.getenv("OPENAI_API_KEY")),
        "workspace": str(m.workspace),
        # 上下文压缩配置（模型感知）
        "context_window": info["context_window"],
        "context_ratio": info["ratio"],
        "keep_rounds": info["keep_rounds"],
        "context_budget": info["budget"],
    }


@app.post("/api/config")
def api_set_config(req: ConfigRequest):
    m = get_manager()
    m.set_config(model=req.model, base_url=req.base_url, api_key=req.api_key,
                 context_window=req.context_window,
                 context_ratio=req.context_ratio,
                 keep_rounds=req.keep_rounds)
    info = m.context_info()
    return {"ok": True, "model": m.model, **info}


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
            "meta": data.get("meta"),
            "stats": _session_stats(messages)}


def _session_stats(messages: list[dict]) -> dict:
    """历史会话统计：token 估算 / 轮次 / 工具调用分布 / 文件读写。"""
    from coder_agent.journal import extract_tool_actions
    from coder_agent.llm.tokenizer import count_messages_tokens

    tools: dict[str, int] = {}
    writes: list[str] = []
    reads: list[str] = []
    turns = 0
    for m in messages:
        if m.get("role") == "user":
            turns += 1
    for action in extract_tool_actions(messages):
        name = action["name"]
        tools[name] = tools.get(name, 0) + 1
        path = action["args"].get("path", "")
        if not path:
            continue
        if name in ("write_file", "edit_file", "append_file"):
            if path not in writes:
                writes.append(path)
        elif name == "read_file":
            if path not in reads:
                reads.append(path)
    tokens_est = count_messages_tokens(messages) if messages else 0
    return {
        "messages": len(messages),
        "turns": turns,
        "tools": tools,
        "writes": writes[-10:],
        "reads": reads[-10:],
        "tokens_est": tokens_est,
    }


@app.delete("/api/session")
def api_session_delete(file: str = ""):
    """删除一个历史会话文件（仅供 UI 会话管理）。"""
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
    p.unlink(missing_ok=True)
    return {"ok": True, "file": str(p)}


@app.get("/api/commands")
def api_commands():
    """返回可用命令列表（供前端动态加载，与 CLI 共用同一份定义）。"""
    from coder_agent.ui.commands import COMMANDS
    return {"commands": COMMANDS}


# ── 演示回放模式 ───────────────────────────────────────────────────────


@app.get("/api/replays")
def api_replays(limit: int = 5):
    """列出最近的回放 trace 文件（供 /replay 无参时选最近一条）。"""
    return {"traces": get_manager().recent_replays(limit=limit)}


@app.get("/api/health")
def api_health():
    """演示前健康检查：后端侧测 API 连通性与延迟（避免浏览器 CORS 限制）。"""
    import time as _time

    m = get_manager()
    env_url = os.getenv("OPENAI_BASE_URL")
    base_url = m.base_url or env_url or "https://api.agnes-ai.cn/v1"
    info = m.context_info()
    result = {
        "model": m.model,
        "base_url": base_url,
        "api_key_set": bool(m.api_key or os.getenv("OPENAI_API_KEY")),
        "workspace": str(m.workspace),
        "context_window": info["context_window"],
        "running": m.running,
    }
    from coder_agent.llm.client import LLMClient

    client = LLMClient(model=m.model, api_key=m.api_key, base_url=m.base_url)
    t0 = _time.time()
    try:
        models = client.list_models()
        result["ok"] = bool(models)
        result["models_count"] = len(models) if models else 0
    except Exception as e:
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
    result["latency_ms"] = int((_time.time() - t0) * 1000)
    return result


@app.get("/api/replay")
async def api_replay(trace: str):
    """回放一条已录制的 trace 文件——完全离线，不依赖 API。"""
    import json as _json
    from pathlib import Path as _P

    path = _P(trace).expanduser()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="trace 文件不存在")
    # 安全：只允许 ~/.coder_replays/ 下的文件
    replays_dir = _P.home() / ".coder_replays"
    try:
        path.resolve().relative_to(replays_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="路径越界，只能在 ~/.coder_replays/ 内回放")

    manager = get_manager()
    iter_ = manager.replay_events(path)

    async def gen():
        try:
            for event in iter_:
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            # 把异常转为前端可消费的 error 事件，避免 StreamingResponse 崩溃
            yield f"data: {_json.dumps({'kind': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/record/start")
def api_record_start(req: RunRequest):
    """开始录制：返回 trace 路径，后续 /api/run 会自动写入。"""
    m = get_manager()
    m._recording_path = m.record_start(req.task, mode=req.mode)
    return {"ok": True, "trace": m._recording_path}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
