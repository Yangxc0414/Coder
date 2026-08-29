"""ReAct Loop — core agent that orchestrates LLM calls and tool execution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .context import ContextManager
from .llm.client import LLMClient, LLMResponse
from .llm.parser import FormatError, parse_tool_calls
from .memory import Memory
from .policy import PolicyGate, PolicyResult
from .state import AgentState
from .tools.base import ToolResult
from .tools.registry import ToolRegistry
from .trace import TraceRecorder
from .verifier import Verifier
from .recovery import RecoveryStrategy
from .mode import AgentMode
from .inspector import ContextInspector
from .hooks import HookRegistry, install_logging_hooks, install_trace_hooks, PRE_TOOL_USE, POST_TOOL_USE, TURN_STOPPED, AGENT_STARTED, AGENT_ENDED
from .extensions.base import SubagentRunner
from .journal import SessionJournal

try:
    from .verifier_llm import ProgressTracker, select as llm_select
    _LLM_VERIFIER_AVAILABLE = True
except ImportError:
    _LLM_VERIFIER_AVAILABLE = False
    ProgressTracker = None  # type: ignore[assignment, misc]
    llm_select = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

MAX_STEPS = 50
MAX_CONSECUTIVE_FORMAT_ERRORS = 3

SYSTEM_PROMPT = """You are a programming assistant agent. Your job is to complete programming tasks by reading files, writing code, and running commands.

Available tools:
{tool_descriptions}

## How to work:
1. [PLAN] Read relevant files first. Understand the codebase structure. Identify what needs to change.
2. [ACT] Execute the plan step by step using tools. Read before you write, test after you modify.
3. [VERIFY] Run tests or checks to confirm the task is complete. Only provide final_answer when confident.

Workspace: {workspace}
"""


def _build_system_prompt(tools: list[dict], workspace: str, state: AgentState | None = None, memory: Memory | None = None) -> str:
    descs = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}"
        for t in tools
    )
    prompt = SYSTEM_PROMPT.format(tool_descriptions=descs, workspace=workspace)

    # Inject state and memory summaries
    extras = []
    if state:
        status = state.to_status_prompt()
        if status and status != "Step 0/50":
            extras.append(f"[Status] {status}")
    if memory:
        mem_summary = memory.get_summary()
        if mem_summary:
            extras.append(mem_summary)
    if extras:
        prompt += "\n\n" + "\n".join(extras)
    return prompt


class Agent:
    """Minimal ReAct Loop agent.

    Control flow:
        while steps < MAX_STEPS:
            response = llm.chat(messages, tools)   # Think
            if no tool_calls: return content        # Done
            for tc in parse(tool_calls):            # Act
                policy.check(tc)
                result = tool.execute(args)
                messages.append(result)             # Observe

    Design inspired by:
    - mini-swe-agent DefaultAgent.run() (linear history + FormatError recovery)
    - smolagents process_tool_calls() (parallel execution, error分层)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: ToolRegistry,
        workspace: Path,
        policy_gate: PolicyGate | None = None,
        context_manager: ContextManager | None = None,
        trace_output: Path | None = None,
        verifier: Verifier | None = None,
        mode: AgentMode = AgentMode.GOAL,
        llm_verifier_mode: str = "off",
        llm_verifier_model: str = "gemini-2.5-flash",
        max_steps: int = MAX_STEPS,
        journal: "SessionJournal | None" = None,
        token_budget: int | None = None,
    ) -> None:
        self.llm = llm_client
        self.registry = registry
        self.workspace = Path(workspace).resolve()
        self.policy = policy_gate or PolicyGate()
        self.context = context_manager or ContextManager()
        self.state = AgentState()
        self.memory = Memory()
        self._loop_warning_injected = False
        self.trace = TraceRecorder(trace_output)
        self._verifier = verifier
        self.recovery = RecoveryStrategy()
        self.mode = mode
        self._max_steps = max_steps
        self.journal = journal
        self._token_budget = token_budget
        self._tokens_used = 0
        self._budget_notice_given = False
        self._llm_max_tokens = 4096
        self._length_escalated = False
        self.inspector = ContextInspector()
        self.messages: list[dict] = []
        self._n_steps = 0
        self._n_format_errors = 0
        self._n_mutations = 0
        # Command-failure adaptation (real-run finding: 13 consecutive
        # failures with the same wrong approach — e.g. `python3` on Windows)
        self._cmd_fail_streak = 0
        self._cmd_hint_injected = False
        # 0, not None: a failed check with zero mutations means the failure
        # predates this task (pre-existing broken tests) — never retry it.
        self._verify_snapshot: int = 0

        # LLM verifier setup (optional, enabled via --llm-verifier-mode)
        self._llm_verifier_mode = llm_verifier_mode if _LLM_VERIFIER_AVAILABLE else "off"
        self._progress_tracker: ProgressTracker | None = None
        self._candidate_answers: list[str] = []
        self._llm_verifier_model = llm_verifier_model
        self._llm_warned = False
        if self._llm_verifier_mode != "off":
            self._progress_tracker = ProgressTracker(
                problem="",  # set at run() time
                model=llm_verifier_model,
            )

        # Hook registry
        self.hooks = HookRegistry()
        install_logging_hooks(self.hooks)
        install_trace_hooks(self.hooks, self.trace)

        # Subagent runner
        self.subagent_runner = SubagentRunner(parent_agent=self)
        from coder_agent.extensions.subagents import get_builtin_subagents
        for defn in get_builtin_subagents():
            self.subagent_runner.register(defn)

        # task tool — expose delegation to the model (my-pi-agent pattern:
        # subagents hidden in Python plumbing are unreachable to the model).
        # Anti-recursion: SubagentRunner.run strips "task"/"memory" from the
        # child registry, so children can never re-delegate.
        # Guard: tolerate a shared registry across Agent instances.
        from coder_agent.tools.task_tool import TaskTool
        if "task" not in self.registry:
            self.registry.register(TaskTool(self.subagent_runner))

        # memory tool — model-writable long-term facts, persisted to
        # .coder_memory.md and injected into the system prompt (my-pi-agent
        # memory-as-tool pattern, simplified for our sync design).
        from coder_agent.tools.memory_tool import MemoryTool
        if "memory" not in self.registry:
            memory_tool = MemoryTool(self.memory, self.workspace)
            memory_tool.load_from_disk()
            self.registry.register(memory_tool)

    def _append_message(self, message: dict[str, Any]) -> None:
        """Single funnel for conversation mutation — keeps the session
        journal complete without instrumenting every call site."""
        self.messages.append(message)
        if self.journal is not None:
            self.journal.log_message(message)

    def run(self, task: str, resume: bool = False) -> str:
        """Run the agent on a programming task. Returns the final answer.

        Args:
            task: Task description, or the continuation instruction when
                ``resume=True``.
            resume: When True, keep existing ``self.messages`` (pre-loaded
                from a journal by the caller) and append ``task`` as the
                continuation instruction instead of resetting the session.
        """
        self.hooks.fire(AGENT_STARTED.with_data(task=task))
        if resume:
            self._append_message({"role": "user", "content": task})
        else:
            self._append_message({"role": "user", "content": task})
            self.state.task_goal = task
        self._n_steps = 0
        self._n_format_errors = 0
        self.recovery.reset()
        self._candidate_answers = []
        self._llm_warned = False
        self._n_mutations = 0
        self._tokens_used = 0
        self._budget_notice_given = False
        self._length_escalated = False
        self._llm_max_tokens = 4096
        self._cmd_fail_streak = 0
        self._cmd_hint_injected = False
        # 0, not None: a failed check with zero mutations means the failure
        # predates this task (pre-existing broken tests) — never retry it.
        self._verify_snapshot: int = 0
        if self._progress_tracker:
            self._progress_tracker.problem = task

        # Baseline-diff verification: capture pre-existing failures BEFORE the
        # task starts, so check() only holds the task responsible for
        # regressions it introduces. (Mutation gating below then decides when
        # a failed check is worth retrying.)
        if self._verifier is not None:
            establish = getattr(self._verifier, "establish_baseline", None)
            if callable(establish):
                try:
                    n_fail, n_syn = establish()
                    self.trace.record(
                        0, "verifier_baseline",
                        test_failures=n_fail, syntax_error_files=n_syn,
                    )
                    if n_fail or n_syn:
                        logger.info(
                            "Verifier baseline: %d pre-existing test failure(s), "
                            "%d syntax-error file(s)", n_fail, n_syn,
                        )
                except Exception as e:
                    logger.warning("Verifier baseline capture failed: %s", e)

        while self._n_steps < self._max_steps:
            self._n_steps += 1
            logger.info("=== Step %d ===", self._n_steps)

            try:
                response = self._query_llm()

                # Cumulative token accounting (getattr: tolerate duck-typed
                # LLM responses, e.g. test doubles without usage)
                usage = getattr(response, "usage", None) or {}
                self._tokens_used += int(usage.get("prompt_tokens") or 0) + int(
                    usage.get("completion_tokens") or 0
                )

                self.trace.record(
                    self._n_steps, "llm_response",
                    has_tool_calls=response.tool_calls is not None,
                    finish_reason=response.finish_reason,
                    tokens_used=self._tokens_used,
                )

                # Token budget: one wrap-up round, then a hard stop.
                # The three ways an agent runs away — too many steps, too
                # many tokens, refusing to accept completion — each get
                # their own gate (MAX_STEPS / token_budget / verify-gating).
                over_budget = (
                    self._token_budget is not None
                    and self._tokens_used >= self._token_budget
                )
                if over_budget and not self._budget_notice_given:
                    self._budget_notice_given = True
                    logger.warning(
                        "Token budget %d exhausted (used %d) — requesting wrap-up",
                        self._token_budget, self._tokens_used,
                    )
                    self.trace.record(
                        self._n_steps, "budget_exhausted",
                        tokens_used=self._tokens_used, budget=self._token_budget,
                    )
                    self._append_message({
                        "role": "user",
                        "content": (
                            "Token budget exhausted. Stop calling tools and give a "
                            "final answer summarizing what was accomplished and what "
                            "remains."
                        ),
                    })
                    continue
                if over_budget:
                    # Wrap-up round already granted — hard stop even if the
                    # model tried to call more tools.
                    answer = response.content or "(no content)"
                    self.trace.record(
                        self._n_steps, "budget_terminated",
                        tokens_used=self._tokens_used,
                    )
                    return answer

                if not response.tool_calls:
                    answer = response.content or "(no content)"
                    self.trace.record(
                        self._n_steps, "final_answer",
                        answer_preview=answer[:300],
                    )
                    logger.info("Agent completed with answer (step %d)", self._n_steps)

                    # Save candidate for LLM verifier selection (select/full modes)
                    if self._llm_verifier_mode in ("select", "full"):
                        self._candidate_answers.append(answer)

                    # Run verifier if configured.
                    # Gate rule: verification can only be retried when the
                    # agent actually changed something since the last failed
                    # check. Otherwise the failure predates or is unrelated to
                    # this task (e.g. a pre-existing broken test in the
                    # workspace) and retrying would just burn steps — the
                    # real-API incident that turned a 2-step answer into a
                    # 50-step flail.
                    if self._verifier:
                        passed, summary = self._verifier.check()
                        self.trace.record(
                            self._n_steps, "verification",
                            passed=passed,
                            summary=summary[:200],
                        )
                        if not passed:
                            snapshot = self._n_mutations
                            if snapshot == self._verify_snapshot:
                                logger.warning(
                                    "Verification failed but nothing changed "
                                    "since the last check — accepting answer"
                                )
                                self.trace.record(
                                    self._n_steps, "verification_accept",
                                    reason="no mutations since last failed check",
                                )
                            else:
                                self._verify_snapshot = snapshot
                                logger.warning("Verification failed, injecting prompt")
                                self._append_message({
                                    "role": "user",
                                    "content": f"Verification failed:\n{summary}\nPlease fix the issues and try again."
                                })
                                continue

                    # LLM verifier: select best candidate (select/full modes)
                    if self._llm_verifier_mode in ("select", "full") and len(self._candidate_answers) >= 2:
                        sel_result = llm_select(
                            problem=task,
                            candidates=self._candidate_answers,
                            model=self._llm_verifier_model,
                        )
                        self.trace.record(
                            self._n_steps, "llm_verifier_select",
                            best_index=sel_result.index,
                            scores=sel_result.scores,
                            ranking=sel_result.ranking,
                            n_candidates=len(self._candidate_answers),
                        )
                        answer = self._candidate_answers[sel_result.index]
                        logger.info(
                            "LLMVerifier selected candidate %d/%d  scores=%s",
                            sel_result.index + 1, len(self._candidate_answers), sel_result.scores,
                        )

                    return answer

                try:
                    parsed = parse_tool_calls(
                        response.tool_calls,
                        self.registry.list_names(),
                        schemas=self.registry.get_parameter_schemas(),
                    )
                except FormatError as e:
                    self._handle_format_error(e)
                    continue

                for pc in parsed:
                    self._execute_tool_call(pc)

                # Loop detection: same-file churn (get_loop_risk) OR exact
                # action repetition (get_repetition_risk — catches loops that
                # vary the file but repeat the identical failing action)
                if (
                    (self.state.get_loop_risk() or self.state.get_repetition_risk())
                    and not self._loop_warning_injected
                ):
                    logger.warning("Loop detected, injecting guidance")
                    self._append_message({
                        "role": "user",
                        "content": (
                            "You seem to be repeating the same actions without "
                            "progress. Consider reviewing your plan and trying a "
                            "different approach."
                        )
                    })
                    self._loop_warning_injected = True

                # Fire TURN_STOPPED hook
                self.hooks.fire(TURN_STOPPED.with_data(step=self._n_steps))

            except Exception as e:
                logger.error("Unexpected agent error: %s", e, exc_info=True)
                # Attempt recovery
                result = self.recovery.handle(e, self)
                if result.recovered:
                    if result.message:
                        self._append_message({
                            "role": "user",
                            "content": result.message,
                        })
                    self.trace.record(
                        self._n_steps, "recovery",
                        action=result.action, recovered=True,
                    )
                    continue
                else:
                    self.trace.record(
                        self._n_steps, "recovery",
                        action=result.action, recovered=False,
                        message=result.message,
                    )
                    return f"Agent terminated: {result.action}. {result.message or ''}"

        # Step budget exhausted — give the model one forced final-answer
        # round (smolagents' provide_final_answer pattern) instead of a bare
        # termination string: 40 steps of work deserve a summary, not a stub.
        self._append_message({
            "role": "user",
            "content": (
                "Step limit reached. Stop calling tools and provide your final "
                "answer now, summarizing what was accomplished and what remains."
            ),
        })
        self._n_steps += 1
        try:
            response = self._query_llm()
            if response.content and not response.tool_calls:
                self.trace.record(
                    self._n_steps, "final_answer",
                    answer_preview=response.content[:300],
                )
                self.hooks.fire(AGENT_ENDED.with_data(
                    steps=self._n_steps,
                    final_state=self.state.to_status_prompt(),
                ))
                return response.content
        except Exception as e:
            logger.warning("Forced final answer failed: %s", e)

        self.hooks.fire(AGENT_ENDED.with_data(
            steps=self._n_steps,
            final_state=self.state.to_status_prompt(),
        ))
        return "Agent reached maximum steps without completing the task."

    def _query_llm(self) -> LLMResponse:
        # Build context-bounded message list with dynamic state/memory injection
        messages_for_api = self.context.build_messages(
            _build_system_prompt(
                self.registry.list_tools(), str(self.workspace),
                state=self.state, memory=self.memory,
            ),
            self.messages,
        )

        response = self.llm.chat(
            messages=messages_for_api,
            tools=self.registry.list_tools(),
            max_tokens=self._llm_max_tokens,
        )

        # Length-truncation recovery (OneCode loop.py:586-615 pattern):
        # a response cut off by max_tokens mid-answer used to surface as a
        # JSON parse FormatError; retry once with doubled budget instead.
        if (
            response.finish_reason == "length"
            and not response.tool_calls
            and not self._length_escalated
        ):
            self._length_escalated = True
            self._llm_max_tokens *= 2
            logger.warning(
                "Response truncated by max_tokens — retrying with %d",
                self._llm_max_tokens,
            )
            self.trace.record(self._n_steps, "length_truncated")
            response = self.llm.chat(
                messages=messages_for_api,
                tools=self.registry.list_tools(),
                max_tokens=self._llm_max_tokens,
            )

        # Build assistant message in the format the API expects.
        # Some providers (e.g. Agnes) require tool_calls to include
        # "type": "function" and an "index" field.
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or "",
        }
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "index": i,
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for i, tc in enumerate(response.tool_calls)
            ]
        self._append_message(msg)
        return response

    def _execute_tool_call(self, parsed: Any) -> None:
        tool = self.registry.get(parsed.tool_name)

        # Fire PRE_TOOL_USE hook (can block by raising)
        self.hooks.fire(PRE_TOOL_USE.with_data(
            tool_name=parsed.tool_name,
            args=parsed.arguments,
            call_id=parsed.call_id,
        ))

        # Update state
        self.state.step = self._n_steps
        self.state.add_recent_action(parsed.tool_name, str(parsed.arguments))

        # Update state file tracking
        if parsed.tool_name == "read_file":
            self.state.mark_file_read(parsed.arguments.get("path", ""))
        elif parsed.tool_name == "write_file":
            self.state.mark_file_modified(parsed.arguments.get("path", ""))

        # Track workspace mutations — verification retries are gated on this.
        # task 也计入：委派对父代理不透明，子代理可能写文件；漏计会让
        # 子代理引入的变更绕过验证重试门控（采纳 task 工具后的新交互）。
        if parsed.tool_name in ("write_file", "run_command", "task"):
            self._n_mutations += 1

        # Policy check
        policy_result: PolicyResult = self.policy.check(
            parsed.tool_name, parsed.arguments, mode=self.mode
        )
        if not policy_result.approved:
            result = ToolResult(error=f"Policy denied: {policy_result.reason}")
            logger.warning("[POLICY DENIED] %s: %s", parsed.tool_name, policy_result.reason)
        else:
            result = tool.execute(parsed.arguments)
            if policy_result.needs_log:
                logger.info(
                    "[TOOL] %s -> %s",
                    parsed.tool_name,
                    (result.output or "(no output)")[:100],
                )

        # Record in memory (after execution)
        self.memory.record(
            step=self._n_steps,
            tool_name=parsed.tool_name,
            args=parsed.arguments,
            success=result.success,
            output=result.output or "",
        )

        # Command-failure streak tracking → adaptive strategy hint.
        # Real-run finding: 13 consecutive failed commands with the SAME
        # wrong approach (`python3` on Windows) — the model never switched
        # strategy because nothing told it to.
        if parsed.tool_name == "run_command":
            if result.success:
                self._cmd_fail_streak = 0
            else:
                self._cmd_fail_streak += 1
                if self._cmd_fail_streak >= 3 and not self._cmd_hint_injected:
                    self._cmd_hint_injected = True
                    import sys as _sys
                    platform_hint = (
                        "Windows 环境没有 python3 命令，请使用 python；"
                        if _sys.platform == "win32" else ""
                    )
                    self._append_message({
                        "role": "user",
                        "content": (
                            f"命令已连续失败 {self._cmd_fail_streak} 次。"
                            "请立即换一种策略，而不是重试同样的命令。常见对策:\n"
                            f"1. {platform_hint}"
                            "阻塞型服务（如 http.server）会一直占用直到超时——"
                            "改用 run_command 的 background 参数启动，或改用非阻塞方式验证\n"
                            "2. 把验证逻辑写成脚本文件再运行，输出更可控\n"
                            "3. 输出重定向到文件（> out.txt）后用 read_file 查看\n"
                            "4. 检查命令本身在该操作系统上是否存在"
                        ),
                    })
                    self.trace.record(
                        self._n_steps, "command_strategy_hint",
                        streak=self._cmd_fail_streak,
                    )

        # Full outcome for repetition-based stuck detection (action+observation)
        self.state.add_tool_outcome(
            parsed.tool_name,
            str(parsed.arguments),
            (result.output or result.error or "")[:200],
        )

        tool_msg: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": parsed.call_id,
            "content": result.to_message_content(),
        }
        self._append_message(tool_msg)
        self.trace.record(
            self._n_steps, "tool_execution",
            tool=parsed.tool_name,
            success=result.success,
            output_len=len(result.output),
        )
        self._n_format_errors = 0  # successful step resets error counter

        # Fire POST_TOOL_USE hook
        self.hooks.fire(POST_TOOL_USE.with_data(
            tool_name=parsed.tool_name,
            success=result.success,
            error=result.error,
            output_len=len(result.output),
        ))

        # LLM progress tracking (progress/full modes)
        if self._progress_tracker is not None:
            step_desc = (
                f"{parsed.tool_name}({parsed.arguments.get('path', parsed.arguments.get('command', '?')[:40])})"
            )
            score = self._progress_tracker.update(step_desc)
            self.trace.record(
                self._n_steps, "llm_progress",
                score=score, step_desc=step_desc[:80],
            )
            if score < 0.15 and not self._llm_warned:
                logger.warning("LLM progress score low (%.3f), injecting guidance", score)
                self._append_message({
                    "role": "user",
                    "content": (
                        f"The progress verifier indicates your current approach may be off-track "
                        f"(score: {score:.2f}). Re-evaluate your plan and consider a different strategy."
                    ),
                })
                self._llm_warned = True

    def _handle_format_error(self, error: FormatError) -> None:
        self._n_format_errors += 1
        logger.warning(
            "Format error %d/%d: %s",
            self._n_format_errors,
            MAX_CONSECUTIVE_FORMAT_ERRORS,
            error,
        )
        self.trace.record(
            self._n_steps, "format_error",
            error=str(error)[:200], attempt=self._n_format_errors,
        )
        correction = (
            f"Your previous response had a format error: {error}. "
            "Please fix it and call the tools again."
        )
        self._append_message({"role": "user", "content": correction})

        if self._n_format_errors >= MAX_CONSECUTIVE_FORMAT_ERRORS:
            logger.error("Max format errors reached, requesting final answer.")
            self._append_message({
                "role": "user",
                "content": (
                    "You have made too many format errors. "
                    "Please provide a clear final answer describing what you would have done."
                ),
            })
