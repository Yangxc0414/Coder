"""RunManager — UI 适配层（框架无关，可独立测试）。

架构（借鉴 my-pi-agent 的 hooks-as-event-source 思想）：
- Agent 在 worker 线程运行，核心代码零改动
- 已有的 6 个生命周期钩子作为事件源，转发到线程安全队列
- 支持多个并行会话：每个 run_id 独立线程/事件队列/agent/journal
- UI 线程从各自队列消费事件渲染界面；abort 通过 request_abort 标志
- 演示回放模式：提供 replay_sse() 同步流，按时间戳播放 trace 文件中的事件
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
        self.model = "agnes-3.0-flash"
        self.mode = "full"
        # API 配置：优先读取 ~/.coder_config.json，其次环境变量
        cfg = load_config()
        self.model = cfg.get("model") or self.model
        self.base_url: str | None = cfg.get("base_url") or None
        self.api_key: str | None = cfg.get("api_key") or None

    # ── 配置 ────────────────────────────────────────────────────────────

    def set_config(self, model: str | None = None,
                   base_url: str | None = None,
                   api_key: str | None = None,
                   context_window: int | None = None,
                   context_ratio: float | None = None,
                   keep_rounds: int | None = None) -> dict:
        """更新模型/API/上下文配置并持久化（立即影响后续运行）。"""
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
        if context_window is not None:
            if context_window > 0:
                cfg["context_window"] = int(context_window)
            else:
                cfg.pop("context_window", None)
        if context_ratio is not None:
            if 0 < context_ratio <= 1:
                cfg["context_ratio"] = float(context_ratio)
            else:
                cfg.pop("context_ratio", None)
        if keep_rounds is not None:
            if keep_rounds > 0:
                cfg["keep_rounds"] = int(keep_rounds)
            else:
                cfg.pop("keep_rounds", None)
        save_config(cfg)
        return {"ok": True, "model": self.model}

    def context_info(self) -> dict:
        """当前上下文压缩配置（窗口/预算/保留轮数）。"""
        from coder_agent.context import resolve_context_window
        cfg = load_config()
        window = int(cfg.get("context_window") or resolve_context_window(self.model))
        ratio = float(cfg.get("context_ratio") or 0.8)
        budget = int(window * ratio)
        keep = (int(cfg["keep_rounds"]) if cfg.get("keep_rounds")
                else max(4, min(32, budget // 2500)))  # 自适应
        return {
            "model": self.model,
            "context_window": window,
            "budget": budget,
            "ratio": ratio,
            "keep_rounds": keep,
        }

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
        session.thread = threading.Thread(
            target=self._run, args=(run_id, task, resume_path, goal), daemon=True)
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
        from coder_agent.context import ContextManager, resolve_context_window
        cfg = load_config()
        window = int(cfg.get("context_window") or resolve_context_window(self.model))
        ratio = float(cfg.get("context_ratio") or 0.8)
        keep = int(cfg["keep_rounds"]) if cfg.get("keep_rounds") else None
        context_manager = ContextManager(
            max_tokens=int(window * ratio), keep_rounds=keep,
            model=self.model, context_window=window)
        registry = create_default_registry(self.workspace, AgentMode(self.mode))
        return Agent(
            llm_client=LLMClient(model=self.model, api_key=self.api_key,
                                 base_url=self.base_url),
            registry=registry, workspace=self.workspace,
            mode=AgentMode(self.mode),
            verifier=Verifier(self.workspace, task=""),
            journal=journal,
            context_manager=context_manager,
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

    def _run(self, run_id: str, task: str, resume_path: str | None = None,
             goal: str | None = None) -> None:
        session = self._sessions[run_id]
        journal = self._make_journal(resume_path)
        session.journal = journal
        try:
            agent = self._build_agent(resume_path, journal)
            session.agent = agent
            if goal:
                # 会话级目标：按 run 传递（多会话并行互不覆盖）
                agent.state.task_goal = goal

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

            # 演示录制：捕获所有 hook 事件写入 trace 文件
            _record_path = getattr(self, "_recording_path", None)
            if _record_path:
                def _on_event(e):
                    self.record_event(_record_path, {"kind": e.name, "data": dict(e.data)})
                for _evt in ["AGENT_STARTED", "AGENT_ENDED", "ASSISTANT_TEXT",
                             "PRE_TOOL_USE", "POST_TOOL_USE", "TURN_STOPPED",
                             "VERIFIER_RESULT", "FORMAT_ERROR", "LOOP_DETECTED",
                             "RECOVERY_EVENT", "LENGTH_RETRY", "BUDGET_EXHAUSTED"]:
                    agent.hooks.register(_evt, _on_event)

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

            def on_turn(e):
                last_turn_step[0] = e.data.get("step")
                self._emit(run_id, "turn", step=e.data.get("step"),
                           tokens=agent._tokens_used)
                # 每步附带 AgentState + Memory 摘要（严格对应框架状态）
                st = agent.state
                mem = agent.memory
                self._emit(run_id, "state",
                           step=int(getattr(st, "step", 0) or 0),
                           max_steps=int(getattr(agent, "_max_steps", 50)),
                           goal=str(getattr(st, "task_goal", "") or ""),
                           editing=str(getattr(st, "current_file", "") or ""),
                           modified=list(getattr(st, "modified_files", []) or [])[-5:],
                           reads=sorted(getattr(st, "read_files", set()) or set())[-5:],
                           subgoals=len(getattr(st, "completed_subgoals", []) or []),
                           memory_keys=list(getattr(mem, "long_term", {}) or {}),
                           memory_entries=[
                               {"k": str(k)[:40],
                                "v": str(getattr(mem, "long_term", {}).get(k, ""))[:80]}
                               for k in list(getattr(mem, "long_term", {}) or {})[:5]
                           ],
                           # 短期记忆：最近工具动作（自动记录，滚动窗口）
                           memory_recent=[
                               {"step": getattr(en, "step", 0),
                                "tool": getattr(en, "tool", "?"),
                                "args": str(getattr(en, "args_summary", ""))[:60],
                                "ok": bool(getattr(en, "success", False))}
                               for en in list(getattr(mem, "short_term", []) or [])[-6:]
                           ],
                           )
                # 检测上下文压缩是否发生（三层压缩展示）
                comp = getattr(getattr(agent, "context", None),
                               "last_compression", None)
                if comp and comp.get("compressed", 0) > 0 and comp != last_comp[0]:
                    last_comp[0] = comp
                    self._emit(run_id, "compress", **comp)

            def on_verifier(e):
                self._emit(run_id, "verifier", passed=bool(e.data.get("passed")),
                           summary=str(e.data.get("summary") or "")[:400],
                           step=e.data.get("step"),
                           accept_reason=str(e.data.get("accept_reason") or ""),
                           results=e.data.get("results") or [])

            def on_fw_event(kind):
                def cb(e):
                    data = dict(e.data)
                    data.pop("step", None)
                    detail = data.pop("error", None) or data.pop("message", None) \
                        or data.pop("reason", None) or data.pop("action", None) or ""
                    self._emit(run_id, "event", type=kind,
                               detail=str(detail)[:160], **{k: v for k, v in data.items()
                                                            if k in ("attempt", "max_attempts",
                                                                     "budget", "used", "new_max_tokens")})
                return cb

            def on_started(e):
                self._emit(run_id, "started",
                           task=str(e.data.get("task") or "")[:120])

            def on_ended(e):
                self._emit(run_id, "ended",
                           steps=e.data.get("steps"),
                           reason=str(e.data.get("reason") or "")[:60])

            last_turn_step: list = [None]
            last_comp: list = [None]
            agent.hooks.register("ASSISTANT_TEXT", on_text)
            agent.hooks.register("POST_TOOL_USE", on_tool)
            agent.hooks.register("TURN_STOPPED", on_turn)
            agent.hooks.register("VERIFIER_RESULT", on_verifier)
            agent.hooks.register("FORMAT_ERROR", on_fw_event("format_error"))
            agent.hooks.register("LOOP_DETECTED", on_fw_event("loop_detected"))
            agent.hooks.register("RECOVERY_EVENT", on_fw_event("recovery"))
            agent.hooks.register("LENGTH_RETRY", on_fw_event("length_retry"))
            agent.hooks.register("BUDGET_EXHAUSTED", on_fw_event("budget_exhausted"))
            agent.hooks.register("AGENT_STARTED", on_started)
            agent.hooks.register("AGENT_ENDED", on_ended)

            answer = agent.run(task)
            session.final_answer = answer
            # 纯回答任务（无工具调用）不会触发 TURN_STOPPED：
            # 补发 turn 事件，保证前端 steps/tokens 统计有数据
            if last_turn_step[0] != agent._n_steps:
                self._emit(run_id, "turn", step=agent._n_steps,
                           tokens=agent._tokens_used)
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
            # 录制一次性：本次运行结束即停止录制，避免后续无关任务
            # 继续追加写进同一个 trace 文件
            self._recording_path = None
            try:
                if session.journal is not None:
                    session.journal.close()
            except OSError:
                pass
            self._emit(run_id, "done")
            session.agent = None

    # ── 停止 ────────────────────────────────────────────────────────────

    def abort(self, run_id: str | None = None) -> None:
        """协作式停止：指定 run_id 或全部运行。

        run_id 缺省时只停"最新"会话（与 stream_events/_session 的
        缺省语义一致）——避免一次 UI 停止误杀用户并行的其它会话。
        """
        if run_id:
            targets = [self._sessions[run_id]] if run_id in self._sessions else []
        else:
            latest = self._sessions.get(self._last_run_id or "")
            targets = [latest] if latest else []
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

    def _target(self, run_id: str | None) -> _RunSession | None:
        """指定会话优先；未指定时回退到最新会话（兼容单会话场景）。"""
        if run_id and run_id in self._sessions:
            return self._sessions[run_id]
        return self._latest()

    def agent_summary(self, n: int = 10, run_id: str | None = None) -> list[dict]:
        """指定会话（缺省=最新）的最近 n 条消息摘要（/history 数据源）。"""
        s = self._target(run_id)
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

    def compact_last(self, keep: int = 10, run_id: str | None = None) -> dict | None:
        """手动压缩指定会话（缺省=最新）的消息历史（保留最近 keep 条）。"""
        s = self._target(run_id)
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

    def trace_summary(self, run_id: str | None = None) -> list[dict] | None:
        """指定会话（缺省=最新）的 trace 摘要（/trace 数据源）。"""
        s = self._target(run_id)
        if s is None or s.trace is None:
            return None
        try:
            return s.trace.get_entries()
        except Exception:
            return None

    # ── 工具清单 ────────────────────────────────────────────────────────

    def tool_specs(self) -> dict:
        """当前工作区下模型可用的工具清单（按 core/skill/mcp 分类）。

        /tools 命令数据源；分类与扩展系统（extensions/）一致：
        skill_* 前缀 = 内置技能，mcp_* 前缀 = MCP 工具。
        """
        from coder_agent.mode import AgentMode
        registry = create_default_registry(self.workspace, AgentMode(self.mode))
        disabled_skills, disabled_mcp = self._disabled_extensions()
        specs = []
        counts = {"core": 0, "skill": 0, "mcp": 0}
        for schema in registry.list_tools():
            fn = schema.get("function", {})
            name = fn.get("name", "?")
            if name.startswith("skill_"):
                category = "skill"
                if name in disabled_skills:
                    continue
            elif name.startswith("mcp_"):
                category = "mcp"
                if name in disabled_mcp:
                    continue
            elif name in ("install_skill", "install_mcp"):
                category = "core"  # 下载/安装扩展的元工具，归核心
            else:
                category = "core"
            counts[category] += 1
            specs.append({"name": name,
                          "description": (fn.get("description") or "")[:100],
                          "category": category,
                          "parameters": fn.get("parameters", {})})
        return {"tools": specs, "counts": counts}

    def _disabled_extensions(self) -> tuple[set, set]:
        """从配置读取被禁用的扩展（skills / mcp 工具名集合）。"""
        cfg = load_config()
        return (set(cfg.get("disabled_skills") or []),
                set(cfg.get("disabled_mcp") or []))

    def toggle_extension(self, kind: str, name: str, enabled: bool) -> dict:
        """启用/禁用扩展（skill 或 mcp 工具），持久化到 ~/.coder_config.json。"""
        if kind not in ("skill", "mcp"):
            return {"ok": False, "error": "kind 必须为 skill 或 mcp"}
        cfg = load_config()
        key = "disabled_skills" if kind == "skill" else "disabled_mcp"
        disabled = set(cfg.get(key) or [])
        if enabled:
            disabled.discard(name)
        else:
            disabled.add(name)
        cfg[key] = sorted(disabled)
        save_config(cfg)
        return {"ok": True, "kind": kind, "name": name, "enabled": enabled}

    def extensions_info(self) -> dict:
        """扩展系统详情：Skills 与 MCP 工具的定义（/skills /mcp 数据源）。

        直接从 extensions 模块读取真实定义（描述/何时使用/参数/服务器），
        与模型看到的工具 schema 同源；被禁用的扩展标记 enabled=False。
        """
        from coder_agent.extensions.skills.builtin import get_builtin_skills
        from coder_agent.extensions.mcp.builtins import create_builtin_mcp_tools

        disabled_skills, disabled_mcp = self._disabled_extensions()

        skills = []
        for s in get_builtin_skills():
            desc = s.description
            when = ""
            if "\n\nUse this when: " in desc:
                desc, when = desc.split("\n\nUse this when: ", 1)
            skills.append({
                "name": s.name,
                "display": s.name.replace("skill_", ""),
                "description": desc.strip()[:150],
                "when_to_use": when.strip()[:100],
                "parameters": s.parameters,
                "allowed_tools": list(getattr(s, "allowed_tools", ()) or ()),
                "enabled": s.name not in disabled_skills,
                "source": "builtin",
                "repo_url": None,
            })
        # 追加已安装 skill（~/.coder_extensions/，跨会话保留）
        from coder_agent.extensions.installer import ExtensionInstaller
        installed = ExtensionInstaller().list_installed()
        for s in installed.get("skills", []):
            skills.append({
                "name": s.name,
                "display": s.name.replace("skill_", ""),
                "description": (s.description or "").strip()[:150],
                "when_to_use": (getattr(s, "_when_to_use", "") or "").strip()[:100],
                "parameters": s.parameters,
                "allowed_tools": list(getattr(s, "allowed_tools", ()) or ()),
                "enabled": s.name not in disabled_skills,
                "source": "installed",
                "repo_url": None,  # 真实 URL 见 manifest；列表层不重读
            })
        mcp: dict[str, list[dict]] = {}
        for t in create_builtin_mcp_tools(self.workspace):
            server = getattr(t, "_server_name", "?")
            mcp.setdefault(server, []).append({
                "name": t.name,
                "display": t.name.replace(f"mcp_{server}_", ""),
                "description": (t.description or "").replace(
                    f"[MCP: {server}] ", "")[:150],
                "parameters": t.parameters,
                "enabled": t.name not in disabled_mcp,
                "source": "builtin",
                "repo_url": None,
            })
        for t in installed.get("mcp", []):
            server = getattr(t, "_server_name", "installed")
            mcp.setdefault(server, []).append({
                "name": t.name,
                "display": t.name.replace(f"mcp_{server}_", ""),
                "description": (t.description or "").replace(
                    f"[MCP: {server}] ", "")[:150],
                "parameters": t.parameters,
                "enabled": t.name not in disabled_mcp,
                "source": "installed",
                "repo_url": None,
            })
        # 子代理（任务委托编排，展示型）
        from coder_agent.extensions.subagents import get_builtin_subagents
        subagents = []
        for d in get_builtin_subagents():
            subagents.append({
                "name": getattr(d, "name", "?"),
                "tools": list(getattr(d, "tools", ()) or ()),
                "read_only": bool(getattr(d, "read_only", False)),
                "max_steps": getattr(d, "max_steps", None),
                "when_to_use": str(getattr(d, "when_to_use", "") or "")[:100],
            })
        return {"skills": skills, "mcp": mcp, "subagents": subagents}

    # ── 演示回放模式 ───────────────────────────────────────────────────

    def _replay_session(self, trace_path: Path) -> tuple[str, int]:
        """从 trace 文件创建临时会话，返回 (run_id, event_count)。"""
        import json as _json
        try:
            raw = trace_path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"无法读取 trace 文件: {e}")

        events: list[dict] = []
        meta: dict = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if obj.get("type") == "meta":
                meta = obj
                continue
            if obj.get("type") in ("event", "tool_result"):
                events.append(obj.get("data") or obj)
            elif isinstance(obj, dict) and "kind" in obj:
                events.append(obj)

        if not events:
            raise ValueError("trace 文件中没有可回放的事件")

        with self._lock:
            self._seq += 1
            run_id = f"replay_{self._seq}"
            session = _RunSession(run_id)
            self._sessions[run_id] = session

        # 把 replay 事件按时间戳顺序放入队列（模拟真实时序）
        for ev in events:
            ts = ev.get("timestamp") or 0
            ev.setdefault("kind", ev.get("type", "event"))
            # 过滤掉已废弃的 type 字段，前端按 kind 判断
            ev.pop("type", None)
            # 加延迟字段，回放时用 sleep 模拟
            session.events.put_nowait(ev)

        return run_id, len(events)

    def replay_events(self, trace_path: Path) -> Generator[dict, None, None]:
        """同步回放 trace 文件中的事件流（用于 /api/replay SSE 端点）。

        每两条事件之间按 timestamp 差值 sleep，模拟真实运行节奏。
        用于演示：不需要 API，完全离线回放。
        """
        import time as _time
        run_id, n = self._replay_session(trace_path)
        session = self._sessions[run_id]

        # 先 emit started
        yield {"kind": "started", "task": "📜 回放模式 — 演示用，不调用 LLM"}

        last_ts = 0.0
        for i, ev in enumerate(session.events.queue):
            ts = ev.get("timestamp") or 0
            if ts > last_ts:
                _time.sleep(min((ts - last_ts) / 1000.0, 0.5))  # 最多等 0.5s
            last_ts = ts
            yield ev
            # 缓冲输出，避免过快
            if i % 5 == 4:
                _time.sleep(0.02)

        # 补齐 done + answer
        yield {"kind": "done"}
        yield {"kind": "answer", "answer": "[回放完成] 共 " + str(n) + " 条事件"}

        # 清理
        with self._lock:
            self._sessions.pop(run_id, None)

    def record_start(self, task: str, mode: str | None = None) -> str:
        """开始录制：记录当前配置作为 trace 元数据。
        返回 trace 路径（创建空文件，待后续 append）。
        """
        import datetime as _dt
        import json as _json
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_dir = Path.home() / ".coder_replays"
        rec_dir.mkdir(parents=True, exist_ok=True)
        path = rec_dir / f"record_{stamp}.jsonl"
        meta = {
            "type": "meta",
            "created": _dt.datetime.now().isoformat(),
            "task": task[:200],
            "mode": mode or self.mode,
            "model": self.model,
            "workspace": str(self.workspace),
        }
        path.write_text(_json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
        return str(path)

    def record_event(self, trace_path: str, event: dict) -> None:
        """追加一条事件到 trace 文件。"""
        import datetime as _dt
        p = Path(trace_path)
        if not p.exists():
            return
        ev = dict(event)
        ev["timestamp"] = int(_dt.datetime.now().timestamp() * 1000)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def recent_replays(self, limit: int = 5) -> list[dict]:
        """列出最近的回放 trace 文件（供 /replay 无参时选最近一条）。"""
        rec_dir = Path.home() / ".coder_replays"
        files = sorted(rec_dir.glob("record_*.jsonl"),
                       key=lambda f: f.stat().st_mtime, reverse=True)[:limit]
        out = []
        for f in files:
            try:
                meta = json.loads(f.read_text(encoding="utf-8").splitlines()[0])
                lines = f.read_text(encoding="utf-8").splitlines()
                # 事件数 = 可回放的事件行（含 meta 行本身与 event/tool_result 行）
                ev_count = sum(1 for l in lines
                               if l.strip() and not l.strip().startswith("#"))
                out.append({"path": str(f), "task": meta.get("task", "?"),
                            "mode": meta.get("mode", "?"),
                            "events": ev_count})
            except Exception:
                pass
        return out
