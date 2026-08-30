"""RunManager — UI 适配层（框架无关，可独立测试）。

架构（借鉴 my-pi-agent 的 hooks-as-event-source 思想）：
- Agent 在 worker 线程运行，核心代码零改动
- 已有的 6 个生命周期钩子作为事件源，转发到线程安全队列
- 支持多个并行会话：每个 run_id 独立线程/事件队列/agent/journal
- UI 线程从各自队列消费事件渲染界面；abort 通过 request_abort 标志
"""

from __future__ import annotations

import datetime
import inspect
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator

from coder_agent.llm.client import LLMClient
from coder_agent.tools.registry import create_default_registry

CONFIG_FILE = Path.home() / ".coder_config.json"


def load_config() -> dict:
    """读取用户 API 配置（~/.coder_config.json），不存在时返回空 dict。"""
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


class _RunSession:
    """单次运行的独立状态：事件队列 / 线程 / agent / journal / 快照。"""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.thread: threading.Thread | None = None
        self.agent: Any = None
        self.journal: Any = None
        self.final_answer: str | None = None
        self.error: str | None = None
        self.messages: list[dict] = []
        self.trace: Any = None


class RunManager:
    """多会话运行管理器：每次 start() 创建独立运行，可并行。"""

    def __init__(
        self,
        workspace: Path,
        agent_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self._agent_factory = agent_factory or self._default_agent_factory
        self._sessions: dict[str, _RunSession] = {}
        self._seq = 0
        self._lock = threading.Lock()
        self._last_run_id: str | None = None
        self.model = "agnes-2.5-flash"
        self.mode = "full"
        # API 配置：优先读取 ~/.coder_config.json，其次环境变量
        cfg = load_config()
        self.model = cfg.get("model") or self.model
        self.base_url: str | None = cfg.get("base_url") or None
        self.api_key: str | None = cfg.get("api_key") or None

    # ── 配置 ────────────────────────────────────────────────────────────

    def set_config(self, model: str | None = None,
                   base_url: str | None = None,
                   api_key: str | None = None) -> dict:
        """更新模型/API 配置并持久化（立即影响后续运行）。"""
        cfg = load_config()
        if model:
            self.model = model
            cfg["model"] = model
        if base_url is not None:
            self.base_url = base_url or None
            if base_url:
                cfg["base_url"] = base_url
            else:
                cfg.pop("base_url", None)
        if api_key is not None:
            self.api_key = api_key or None
            if api_key:
                cfg["api_key"] = api_key
            else:
                cfg.pop("api_key", None)
        save_config(cfg)
        return {"ok": True, "model": self.model}

    # ── 运行管理 ────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        """任一会话在运行即视为忙碌（供 UI 状态显示）。"""
        return any(s.thread is not None and s.thread.is_alive()
                   for s in self._sessions.values())

    def runs_status(self) -> list[dict]:
        """所有会话的运行状态（供 /api/status 展示并行列表）。"""
        out = []
        for rid, s in self._sessions.items():
            alive = s.thread is not None and s.thread.is_alive()
            out.append({"run_id": rid, "running": alive,
                        "error": s.error, "answer": s.final_answer})
        return out

    def start(self, task: str, mode: str | None = None, model: str | None = None,
              resume_path: str | None = None, goal: str | None = None) -> dict:
        """启动一次新运行（允许与已有运行并行）。"""
        with self._lock:
            self._seq += 1
            run_id = f"run_{self._seq}"
            session = _RunSession(run_id)
            self._sessions[run_id] = session
            self._last_run_id = run_id
        if mode:
            self.mode = mode
        if model:
            self.model = model
        self._pending_goal = goal
        session.thread = threading.Thread(
            target=self._run, args=(run_id, task, resume_path), daemon=True)
        session.thread.start()
        return {"ok": True, "run_id": run_id}

    def _session(self, run_id: str | None) -> _RunSession | None:
        if run_id:
            return self._sessions.get(run_id)
        # 无参：最新会话（兼容旧前端/测试）
        return self._sessions.get(self._last_run_id or "")

    def _emit(self, run_id: str, kind: str, **data: Any) -> None:
        session = self._sessions.get(run_id)
        if session is not None:
            session.events.put({"kind": kind, **data})

    # ── Agent 构建（journal 每会话独立）────────────────────────────────

    def _make_journal(self, resume_path: str | None):
        from coder_agent.journal import SessionJournal
        if resume_path:
            return SessionJournal(resume_path, append=True,
                                  workspace=self.workspace)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return SessionJournal(
            Path.home() / ".coder_sessions" / f"session_{stamp}_{0}.jsonl",
            workspace=self.workspace)

    def _default_agent_factory(self, journal=None, resume_path=None):
        from coder_agent.agent import Agent
        from coder_agent.verifier import Verifier
        from coder_agent.mode import AgentMode
        registry = create_default_registry(self.workspace, AgentMode(self.mode))
        return Agent(
            llm_client=LLMClient(model=self.model, api_key=self.api_key,
                                 base_url=self.base_url),
            registry=registry, workspace=self.workspace,
            mode=AgentMode(self.mode),
            verifier=Verifier(self.workspace, task=""),
            journal=journal,
        )

    def _build_agent(self, resume_path: str | None, journal):
        """创建 agent：默认工厂接收 journal/resume_path；注入的无参工厂兼容。"""
        fn = self._agent_factory
        try:
            names = set(inspect.signature(fn).parameters)
        except (ValueError, TypeError):
            names = set()
        if "journal" in names or "resume_path" in names:
            kwargs = {}
            if "journal" in names:
                kwargs["journal"] = journal
            if "resume_path" in names:
                kwargs["resume_path"] = resume_path
            return fn(**kwargs)
        return fn()

    def _run(self, run_id: str, task: str, resume_path: str | None = None) -> None:
        session = self._sessions[run_id]
        journal = self._make_journal(resume_path)
        session.journal = journal
        try:
            agent = self._build_agent(resume_path, journal)
            session.agent = agent
            if self._pending_goal:
                agent.state.task_goal = self._pending_goal  # 注入系统提示

            def on_stream(delta: str):
                # 逐 token 流式输出（前端打字机效果）
                self._emit(run_id, "stream", delta=delta)

            def on_toolstart(e):
                # 工具开始执行（前端显示"正在调用…"）
                args = e.data.get("args") or {}
                arg_str = ""
                try:
                    if isinstance(args, dict):
                        arg_str = str(args.get("path")
                                      or args.get("command")
                                      or args.get("pattern") or "")
                except Exception:
                    arg_str = ""
                self._emit(run_id, "toolstart", tool=e.data.get("tool_name", "?"),
                           target=arg_str[:120])

            agent.stream_callback = on_stream
            agent.hooks.register("PRE_TOOL_USE", on_toolstart)

            if resume_path:
                from coder_agent.journal import load_journal, replay_state
                restored = load_journal(resume_path)
                agent.messages = restored["messages"]
                replay_state(restored["messages"], agent.state)
                self._emit(run_id, "status",
                           message=f"已恢复 {len(restored['messages'])} 条消息")

            def on_text(e):
                self._emit(run_id, "text",
                           text=(e.data.get("text") or "")[:800],
                           step=agent._n_steps)

            def on_tool(e):
                self._emit(run_id, "tool",
                           tool=e.data.get("tool_name", "?"),
                           success=bool(e.data.get("success")),
                           step=agent._n_steps,
                           target=str(e.data.get("target") or "")[:160],
                           detail=str(e.data.get("error") or "")[:120],
                           preview=str(e.data.get("preview") or "")[:600])

            def on_turn(e):
                self._emit(run_id, "turn", step=e.data.get("step"),
                           tokens=agent._tokens_used)

            agent.hooks.register("ASSISTANT_TEXT", on_text)
            agent.hooks.register("POST_TOOL_USE", on_tool)
            agent.hooks.register("TURN_STOPPED", on_turn)

            answer = agent.run(task)
            session.final_answer = answer
            self._emit(run_id, "answer", answer=answer)
        except Exception as e:
            session.error = f"{type(e).__name__}: {e}"
            self._emit(run_id, "error", error=session.error)
        finally:
            # 保存消息/追踪快照（供 /history /compact /trace）
            if session.agent is not None:
                session.messages = list(
                    getattr(session.agent, "messages", []) or [])
                session.trace = getattr(session.agent, "trace", None)
            try:
                if session.journal is not None:
                    session.journal.close()
            except OSError:
                pass
            self._emit(run_id, "done")
            session.agent = None

    # ── 停止 ────────────────────────────────────────────────────────────

    def abort(self, run_id: str | None = None) -> None:
        """协作式停止：指定 run_id 或全部运行。"""
        if run_id:
            targets = [self._sessions[run_id]] if run_id in self._sessions else []
        else:
            targets = list(self._sessions.values())
        for s in targets:
            if s.agent is not None:
                s.agent.request_abort()
                self._emit(s.run_id, "status", message="停止请求已发送")

    # ── 事件消费 ────────────────────────────────────────────────────────

    def drain_events(self, run_id: str | None = None) -> list[dict]:
        """非阻塞取走指定会话（缺省=最新）的全部事件。"""
        session = self._session(run_id)
        if session is None:
            return []
        out = []
        while True:
            try:
                out.append(session.events.get_nowait())
            except queue.Empty:
                break
        return out

    def stream_events(self, run_id: str | None = None) -> Generator[dict, None, None]:
        """阻塞式逐条产出指定会话（缺省=最新）的事件，直到该运行结束。"""
        session = self._session(run_id)
        if session is None:
            return
        while True:
            events = self.drain_events(run_id or session.run_id)
            if events:
                yield from events
                if events[-1].get("kind") == "done":
                    return
            elif not (session.thread is not None and session.thread.is_alive()):
                residual = self.drain_events(run_id or session.run_id)
                if residual:
                    yield from residual
                    if residual[-1].get("kind") == "done":
                        return
                return
            else:
                time.sleep(0.15)

    # ── 上下文 / 压缩 / 追踪（作用于最新会话，对应 CLI /history 等）──────

    def _latest(self) -> _RunSession | None:
        return self._sessions.get(self._last_run_id or "")

    def agent_summary(self, n: int = 10) -> list[dict]:
        """最近一次运行的最近 n 条消息摘要（/history 数据源）。"""
        s = self._latest()
        if s is None:
            return []
        msgs = s.messages or []
        out = []
        for i, m in enumerate(msgs[-n:]):
            role = m.get("role", "?")
            content = (m.get("content") or "")[:120]
            if m.get("tool_calls"):
                names = [tc.get("function", {}).get("name", "?")
                         for tc in m.get("tool_calls", [])]
                content = content or f"[工具调用: {', '.join(names)}]"
            out.append({"index": i, "role": role, "content": content})
        return out

    def compact_last(self, keep: int = 10) -> dict | None:
        """手动压缩最近一次运行的消息历史（保留最近 keep 条）。"""
        s = self._latest()
        if s is None:
            return {"ok": False, "error": "尚无历史消息可压缩"}
        if s.thread is not None and s.thread.is_alive():
            return {"ok": False, "error": "任务运行中，无法压缩"}
        msgs = s.messages or []
        n = len(msgs)
        if n <= keep:
            return {"ok": False, "error": f"无需压缩（{n} 条 ≤ {keep} 条阈值）"}
        s.messages = msgs[-keep:]
        return {"ok": True, "before": n, "after": len(s.messages)}

    def trace_summary(self) -> list[dict] | None:
        """最近一次运行的 trace 摘要（/trace 数据源）。"""
        s = self._latest()
        if s is None or s.trace is None:
            return None
        try:
            return s.trace.get_entries()
        except Exception:
            return None

    # ── 工具清单 ────────────────────────────────────────────────────────

    def tool_specs(self) -> list[dict]:
        """当前工作区下模型可用的工具清单（/tools 命令数据源）。"""
        from coder_agent.mode import AgentMode
        registry = create_default_registry(self.workspace, AgentMode(self.mode))
        specs = []
        for schema in registry.list_tools():
            fn = schema.get("function", {})
            specs.append({"name": fn.get("name", "?"),
                          "description": (fn.get("description") or "")[:100]})
        return specs
