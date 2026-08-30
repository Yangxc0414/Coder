"""RunManager — UI 适配层（框架无关，可独立测试）。

架构（借鉴 my-pi-agent 的 hooks-as-event-source 思想）：
- Agent 在 worker 线程运行，核心代码零改动
- 已有的 6 个生命周期钩子作为事件源，转发到线程安全队列
- UI 线程从队列消费事件渲染界面；abort 通过 request_abort 标志
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable, Generator

from coder_agent.llm.client import LLMClient
from coder_agent.tools.registry import create_default_registry


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
        self.model = "agnes-2.5-flash"
        self.mode = "full"

    def _default_agent_factory(self):
        from coder_agent.agent import Agent
        from coder_agent.verifier import Verifier
        from coder_agent.mode import AgentMode
        from coder_agent.ui.cli.repl import CoderRepl  # 复用注册逻辑
        registry = create_default_registry(self.workspace, AgentMode(self.mode))
        return Agent(
            llm_client=LLMClient(model=self.model),
            registry=registry, workspace=self.workspace,
            mode=AgentMode(self.mode),
            verifier=Verifier(self.workspace, task=""),
        )

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, task: str, mode: str | None = None, model: str | None = None,
              resume_path: str | None = None) -> dict:
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
        self._thread = threading.Thread(
            target=self._run, args=(task, resume_path), daemon=True)
        self._thread.start()
        return {"ok": True, "run_id": id(self._thread)}

    def _emit(self, kind: str, **data: Any) -> None:
        self._events.put({"kind": kind, **data})

    def _run(self, task: str, resume_path: str | None = None) -> None:
        try:
            agent = self._agent_factory()
            self._agent = agent
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
                           detail=str(e.data.get("error") or "")[:120])

            def on_turn(e):
                self._emit("turn", step=e.data.get("step"))

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
            self._emit("done")
            self._agent = None

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


import time  # noqa: E402  (stream_events 用)
