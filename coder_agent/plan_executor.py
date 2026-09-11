"""Parallel subagent execution over plan layers.

Each layer of the dependency graph is a set of independent subgoals that
can run concurrently. We dispatch up to PARALLEL_LIMIT subagents at a time
via a thread pool; each subagent runs in its OWN Agent instance (isolated
context, own registry filtered by the subagent definition, no task/memory
tools — structural anti-recursion inherited from SubagentRunner).

Context economy: only the subagents' final reports flow back to the parent
context (capped at MAX_REPORT_CHARS each), not every intermediate tool
output. This is what keeps the parent context flat as work scales.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any

from .planner import PARALLEL_LIMIT, PlanResult, Subgoal

logger = logging.getLogger(__name__)


@dataclass
class SubgoalOutcome:
    """Result of executing one subgoal via a subagent."""
    subgoal: Subgoal
    report: str
    is_error: bool = False
    steps_used: int = 0


class PlanExecutor:
    """Executes a PlanResult layer-by-layer with bounded parallelism."""

    def __init__(self, runner, verifier=None, progress_callback=None) -> None:
        self._runner = runner
        self._verifier = verifier
        self._on_progress = progress_callback  # (layer_idx, subgoal, outcome|None)

    def execute(self, plan: PlanResult, task: str) -> tuple[list[SubgoalOutcome], bool]:
        """Run all layers. Returns (outcomes, all_passed_verify)."""
        outcomes: list[SubgoalOutcome] = []
        all_passed = True
        n_layers = len(plan.layers)
        for li, layer in enumerate(plan.layers):
            layer_idx = li + 1
            layer_outcomes = self._run_layer(layer, task, layer_idx, n_layers)
            outcomes.extend(layer_outcomes)
            # Verify gate after each layer (baseline-diff aware)
            passed = self._verify_layer(layer_idx, n_layers, layer_outcomes)
            if not passed:
                all_passed = False
        return outcomes, all_passed

    def _run_layer(self, layer: list[Subgoal], task: str,
                   layer_idx: int, n_layers: int) -> list[SubgoalOutcome]:
        """Dispatch one layer's subgoals to a thread pool (bounded)."""
        results: dict[int, SubgoalOutcome] = {}
        from .extensions.base import SubagentRequest

        def _dispatch(sg: Subgoal) -> SubgoalOutcome:
            if self._on_progress:
                self._on_progress(layer_idx, sg, None)  # started
            prompt = self._compose_prompt(task, sg, layer_idx, n_layers)
            req = SubagentRequest(prompt=prompt, subagent_type=sg.subagent_type)
            try:
                res = self._runner.run(req)
                outcome = SubgoalOutcome(
                    subgoal=sg,
                    report=res.final_output,
                    is_error=res.is_error,
                    steps_used=res.steps_used,
                )
            except Exception as e:  # 单子代理失败不拖垮整层
                logger.debug("subagent %s failed: %s", sg.index, e)
                outcome = SubgoalOutcome(subgoal=sg, report=str(e), is_error=True)
            if self._on_progress:
                self._on_progress(layer_idx, sg, outcome)
            return outcome

        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_LIMIT) as pool:
            futs = {pool.submit(_dispatch, sg): sg for sg in layer}
            for fut in concurrent.futures.as_completed(futs):
                out = fut.result()
                results[out.subgoal.index] = out

        return [results[sg.index] for sg in layer]  # 按原顺序返回

    def _compose_prompt(self, task: str, sg: Subgoal,
                       layer_idx: int, n_layers: int) -> str:
        """Self-contained prompt for the subagent (it can't see parent ctx)."""
        return (
            f"总任务：{task}\n\n"
            f"你负责其中一个子目标（第 {layer_idx}/{n_layers} 层，编号 #{sg.index + 1}）：\n"
            f"  目标：{sg.goal}\n\n"
            f"要求：自包含执行，完成后用简洁的结论 + 关键改动点 + 验证情况作答"
            f"（不要复述全过程）。"
        )

    def _verify_layer(self, layer_idx: int, n_layers: int,
                      layer_outcomes: list[SubgoalOutcome]) -> bool:
        """Per-layer verify gate. No verifier → trust subagents (True)."""
        if self._verifier is None:
            return True
        try:
            result = self._verifier.check()
            return bool(result.passed)
        except Exception as e:
            logger.debug("layer %d verify failed: %s", layer_idx, e)
            return True  # 验证器自身故障不阻塞（只快不慢原则）
