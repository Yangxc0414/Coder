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
        self.inspector = ContextInspector()
        self.messages: list[dict] = []
        self._n_steps = 0
        self._n_format_errors = 0
        self._n_mutations = 0
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
        # 0, not None: a failed check with zero mutations means the failure
        # predates this task (pre-existing broken tests) — never retry it.
        self._verify_snapshot: int = 0
        if self._progress_tracker:
            self._progress_tracker.problem = task

        while self._n_steps < self._max_steps:
            self._n_steps += 1
            logger.info("=== Step %d ===", self._n_steps)

            try:
                response = self._query_llm()
                self.trace.record(
                    self._n_steps, "llm_response",
                    has_tool_calls=response.tool_calls is not None,
                    finish_reason=response.finish_reason,
                )

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
                        response.tool_calls, self.registry.list_names()
                    )
                except FormatError as e:
                    self._handle_format_error(e)
                    continue

                for pc in parsed:
                    self._execute_tool_call(pc)

                # Loop detection
                if self.state.get_loop_risk() and not self._loop_warning_injected:
                    logger.warning("Loop detected, injecting guidance")
                    self._append_message({
                        "role": "user",
                        "content": (
                            "You seem to be repeatedly operating on the same file(s). "
                            "Consider reviewing your plan and trying a different approach."
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

        # Track workspace mutations — verification retries are gated on this
        if parsed.tool_name in ("write_file", "run_command"):
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
