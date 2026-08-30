"""RunManager — UI 适配层（框架无关，可独立测试）。

架构（借鉴 my-pi-agent 的 hooks-as-event-source 思想）：
- Agent 在 worker 线程运行，核心代码零改动
- 已有的 6 个生命周期钩子作为事件源，转发到线程安全队列
- UI 线程从队列消费事件渲染界面；abort 通过 request_abort 标志
"""

from __future__ import annotations

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


class RunManager:
    """单运行实例管理器：一次只跑一个任务（个人客户端足够）。"""

    def __init__(
        self,
        workspace: Path,
        agent_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self._agent_factory = agent_factory or self._default_agent_factory
        self._agent = None
        self._events: "queue.Queue[dict]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._final_answer: str | None = None
        self._error: str | None = None
        self._last_messages: list[dict] = []
        self._last_trace = None
        self._journal = None
        self.model = "agnes-2.5-flash"
        self.mode = "full"
        # API 配置：优先读取 ~/.coder_config.json，其次环境变量
        cfg = load_config()
        self.model = cfg.get("model") or self.model
        self.base_url: str | None = cfg.get("base_url") or None
        self.api_key: str | None = cfg.get("api_key") or None

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

    def _default_agent_factory(self, resume_path: str | None = None):
        from coder_agent.agent import Agent
        from coder_agent.verifier import Verifier
        from coder_agent.mode import AgentMode
        from coder_agent.journal import SessionJournal
        import datetime

        # Web 运行同样落盘 journal（与 CLI 同目录），resume 时追加
        if resume_path:
            self._journal = SessionJournal(resume_path, append=True,
                                           workspace=self.workspace)
        else:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._journal = SessionJournal(
                Path.home() / ".coder_sessions" / f"session_{stamp}_{0}.jsonl",
                workspace=self.workspace)

        registry = create_default_registry(self.workspace, AgentMode(self.mode))
        return Agent(
            llm_client=LLMClient(model=self.model, api_key=self.api_key,
                                 base_url=self.base_url),
            registry=registry, workspace=self.workspace,
            mode=AgentMode(self.mode),
            verifier=Verifier(self.workspace, task=""),
            journal=self._journal,
        )

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, task: str, mode: str | None = None, model: str | None = None,
              resume_path: str | None = None, goal: str | None = None) -> dict:
        with self._lock:
            if self.running:
                return {"ok": False, "error": "已有任务在运行——请先停止或等待完成"}
        if mode:
            self.mode = mode
        if model:
            self.model = model
        self._final_answer = None
        self._error = None
        self._events = queue.Queue()  # 新运行清空事件
        self._pending_goal = goal
        self._thread = threading.Thread(
            target=self._run, args=(task, resume_path), daemon=True)
        self._thread.start()
        return {"ok": True, "run_id": id(self._thread)}

    def _emit(self, kind: str, **data: Any) -> None:
        self._events.put({"kind": kind, **data})

    def _build_agent(self, resume_path: str | None = None):
        """创建 agent：优先传 resume_path（默认工厂用于 journal 追加），
        外部注入的无参工厂保持兼容。"""
        import inspect
        try:
            sig = inspect.signature(self._agent_factory)
            accepts = any(
                p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                or p.kind == inspect.Parameter.VAR_POSITIONAL
                for p in sig.parameters.values()
            )
        except (ValueError, TypeError):
            accepts = False
        if accepts:
            return self._agent_factory(resume_path)
        return self._agent_factory()

    def _run(self, task: str, resume_path: str | None = None) -> None:
        try:
            agent = self._build_agent(resume_path)
            self._agent = agent
            self._journal = getattr(self, "_journal", None)
            if self._pending_goal:
                agent.state.task_goal = self._pending_goal  # 注入系统提示
            if resume_path:
                from coder_agent.journal import load_journal, replay_state
                restored = load_journal(resume_path)
                agent.messages = restored["messages"]
                replay_state(restored["messages"], agent.state)
                self._emit("status", message=f"已恢复 {len(restored['messages'])} 条消息")

            def on_text(e):
                self._emit("text", text=(e.data.get("text") or "")[:800], step=agent._n_steps)

            def on_tool(e):
                self._emit("tool", tool=e.data.get("tool_name", "?"),
                           success=bool(e.data.get("success")),
                           step=agent._n_steps,
                           target=str(e.data.get("target") or "")[:160],
                           detail=str(e.data.get("error") or "")[:120],
                           preview=str(e.data.get("preview") or "")[:600])

            def on_turn(e):
                self._emit("turn", step=e.data.get("step"),
                           tokens=agent._tokens_used)

            agent.hooks.register("ASSISTANT_TEXT", on_text)
            agent.hooks.register("POST_TOOL_USE", on_tool)
            agent.hooks.register("TURN_STOPPED", on_turn)

            answer = agent.run(task)
            self._final_answer = answer
            self._emit("answer", answer=answer)
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            self._emit("error", error=self._error)
        finally:
            # 保存最近一次的消息/追踪快照（供 /history /compact /trace）
            if self._agent is not None:
                self._last_messages = list(getattr(self._agent, "messages", []) or [])
                self._last_trace = getattr(self._agent, "trace", None)
            try:
                if getattr(self, "_journal", None) is not None:
                    self._journal.close()
            except OSError:
                pass
            self._emit("done")
            self._agent = None

    def agent_summary(self, n: int = 10) -> list[dict]:
        """最近一次运行的最近 n 条消息摘要（/history 数据源）。"""
        msgs = getattr(self, "_last_messages", []) or []
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
        """手动压缩最近一次运行的消息历史（保留最近 keep 条）。

        对应 CLI 的 /compact 语义；运行中不可压缩。
        """
        if self.running:
            return {"ok": False, "error": "任务运行中，无法压缩"}
        msgs = getattr(self, "_last_messages", None)
        if not msgs:
            return {"ok": False, "error": "尚无历史消息可压缩"}
        n = len(msgs)
        if n <= keep:
            return {"ok": False, "error": f"无需压缩（{n} 条 ≤ {keep} 条阈值）"}
        self._last_messages = msgs[-keep:]
        return {"ok": True, "before": n, "after": len(self._last_messages)}

    def trace_summary(self) -> list[dict] | None:
        """最近一次运行的 trace 摘要（/trace 数据源）。"""
        trace = getattr(self, "_last_trace", None)
        if trace is None:
            return None
        try:
            return trace.get_entries()
        except Exception:
            return None

    def abort(self) -> None:
        if self._agent is not None:
            self._agent.request_abort()
            self._emit("status", message="停止请求已发送")

    def drain_events(self) -> list[dict]:
        """非阻塞取走当前全部事件（SSE 轮询/SSE 流均可消费）。"""
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    def stream_events(self) -> Generator[dict, None, None]:
        """阻塞式逐条产出事件，直到运行结束且队列排空。"""
        while True:
            events = self.drain_events()
            if events:
                yield from events
                if events[-1].get("kind") == "done":
                    return
            elif not self.running:
                residual = self.drain_events()
                if residual:
                    yield from residual
                    if residual[-1].get("kind") == "done":
                        return
                return
            else:
                time.sleep(0.15)
