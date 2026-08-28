"""LLM-based verifier — thin wrapper around the llm-verifier package.

Wraps the official llm-verifier API exactly as documented:
  https://llm-as-a-verifier.com/docs/

Public API (mirrors official functions):
  - select(problem, candidates, criteria, model, n_evaluations, pivots)
      → SelectResult(index, scores, ranking)   Best-of-N selection
  - compare(problem, candidate_a, candidate_b, criteria, model)
      → (reward_a, reward_b)                  Pairwise comparison
  - track(problem, steps, criteria, model, n_evaluations)
      → TrackResult(scores)                   Offline trajectory scoring
  - ProgressTracker(problem, criteria, model, n_evaluations)
      → update(step_desc) -> float            Online per-step scoring

Usage:
  1. pip install "coder-agent[verifier]"   (pulls in llm-verifier)
  2. Set LLM_VERIFIER_MODEL in .env or pass --llm-verifier-model
  3. Pass --use-llm-verifier to enable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Public dataclasses (match official return types) ─────────────────


@dataclass
class SelectResult:
    """Result of llm_verifier.select() — Best-of-N selection."""

    index: int            # best candidate index
    scores: list[float]   # score per candidate, same order as input
    ranking: list[int]    # indices sorted by score descending


@dataclass
class TrackResult:
    """Result of llm_verifier.track() — offline trajectory scoring."""

    scores: list[float]   # score after each checkpoint step


# ── Internal helper ───────────────────────────────────────────────────


def _import_llm_verifier() -> Any:
    """Lazy-import with a clear error message."""
    try:
        import llm_verifier  # noqa: F401
        return llm_verifier
    except ImportError as e:
        raise ImportError(
            "llm-verifier is not installed.\n"
            "  pip install 'coder-agent[verifier]'\n"
            "  or: pip install llm-verifier"
        ) from e


# ── Best-of-N Selection ──────────────────────────────────────────────
# Doc: https://llm-as-a-verifier.com/docs/basic_usage/best_of_n_selection.html


def select(
    problem: str,
    candidates: list[str],
    criteria: dict[str, str] | None = None,
    model: str = "gemini-2.5-flash",
    n_evaluations: int = 4,
    pivots: int = 2,
) -> SelectResult:
    """Call llm_verifier.select() and return structured result.

    Args:
        problem:       Task description.
        candidates:    List of candidate solutions (strings).
        criteria:      Dict of criterion name → description.
                       Defaults to coding-oriented criteria.
        model:         Verifier model (e.g. 'gemini-2.5-flash', 'deepseek-v4-flash').
        n_evaluations: Number of independent evaluations per criterion.
        pivots:        Number of pivot candidates for tournament (pivots < len(candidates)).

    Returns:
        SelectResult with best index, per-candidate scores, and full ranking.
    """
    if len(candidates) <= 1:
        return SelectResult(
            index=0 if candidates else -1,
            scores=[1.0] if candidates else [],
            ranking=list(range(len(candidates))),
        )

    criteria = criteria or {
        "Completeness": "Does the solution fully address the original task?",
        "Correctness":  "Is the solution correct? Any bugs or logical errors?",
        "Style":        "Is the code clean, readable, and following good practices?",
    }

    try:
        llm = _import_llm_verifier()
        result = llm.select(
            problem=problem,
            candidates=candidates,
            criteria=criteria,
            model=model,
            n_evaluations=n_evaluations,
            pivots=min(pivots, len(candidates) - 1),
        )
        scores = [float(s) for s in result.scores]
        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best_idx = int(result.index)
        logger.info("select() → best=%d  scores=%s  ranking=%s", best_idx, scores, ranking)
        return SelectResult(index=best_idx, scores=scores, ranking=ranking)
    except Exception as e:
        logger.warning("select() failed, falling back: %s", e)
        n = len(candidates)
        return SelectResult(index=n - 1, scores=[0.5] * n, ranking=list(range(n - 1, -1, -1)))


# ── Pairwise Comparison ──────────────────────────────────────────────
# Doc: https://llm-as-a-verifier.com/docs/basic_usage/best_of_n_selection.html


def compare(
    problem: str,
    candidate_a: str,
    candidate_b: str,
    criteria: dict[str, str] | None = None,
    model: str = "gemini-2.5-flash",
) -> tuple[float, float]:
    """Call llm_verifier.compare() for direct pairwise scoring.

    Returns:
        (reward_a, reward_b) — both in [0, 1].
    """
    criteria = criteria or {
        "Correctness": "Does the code correctly solve the problem?",
    }
    try:
        llm = _import_llm_verifier()
        ra, rb = llm.compare(problem, candidate_a, candidate_b, criteria=criteria, model=model)
        return float(ra), float(rb)
    except Exception as e:
        logger.warning("compare() failed: %s", e)
        return 0.5, 0.5


# ── Offline Trajectory Scoring ───────────────────────────────────────
# Doc: https://llm-as-a-verifier.com/docs/basic_usage/progress_tracking.html


def track(
    problem: str,
    steps: list[str],
    criteria: dict[str, str] | None = None,
    model: str = "gemini-2.5-flash",
    n_evaluations: int = 4,
) -> TrackResult:
    """Call llm_verifier.track() to score a finished trajectory.

    Args:
        problem:       Task description.
        steps:         List of step descriptions (in execution order).
        criteria:      Scoring criteria.
        model:         Verifier model.
        n_evaluations: Independent evaluations per criterion (averaged).

    Returns:
        TrackResult with a score after each step.
    """
    if not steps:
        return TrackResult(scores=[])

    criteria = criteria or {
        "Progress":   "Is the agent making meaningful progress toward the goal?",
        "Correctness":"Are the changes so far correct and on-track?",
    }
    checkpoint_steps = list(range(1, len(steps) + 1))

    try:
        llm = _import_llm_verifier()
        result = llm.track(
            problem=problem,
            steps=steps,
            checkpoint_steps=checkpoint_steps,
            n_evaluations=n_evaluations,
            criteria=criteria,
            model=model,
        )
        scores = [float(s) for s in result.scores]
        logger.debug("track() → %d steps, scores=%s", len(scores), scores)
        return TrackResult(scores=scores)
    except Exception as e:
        logger.warning("track() failed: %s", e)
        return TrackResult(scores=[0.0] * len(steps))


# ── Online Progress Tracker ──────────────────────────────────────────
# Doc: https://llm-as-a-verifier.com/docs/basic_usage/progress_tracking.html


class ProgressTracker:
    """Live progress tracker — scores each agent step as it happens.

    Mirrors the official llm_verifier.ProgressTracker API.

    Usage:
        tracker = ProgressTracker(problem, model="gemini-2.5-flash")
        score = tracker.update("Read the problem statement")   # 0.001
        score = tracker.update("Wrote initial implementation")  # 0.42
        score = tracker.update("Tested and fixed bug")          # 0.97
    """

    def __init__(
        self,
        problem: str,
        criteria: dict[str, str] | None = None,
        model: str = "gemini-2.5-flash",
        n_evaluations: int = 3,
    ) -> None:
        self.problem = problem
        self.criteria = criteria or {
            "Progress":   "Is the agent making meaningful progress toward the goal?",
            "Correctness":"Are the changes so far correct and on-track?",
        }
        self.model = model
        self.n_evaluations = n_evaluations
        self._steps: list[str] = []
        self._scores: list[float] = []

    def update(self, step_description: str) -> float:
        """Score the latest step and return the current score (0.0–1.0).

        Internally re-calls track() over the full history so far,
        matching the official ProgressTracker behavior.
        """
        self._steps.append(step_description)
        try:
            result = track(
                problem=self.problem,
                steps=self._steps,
                criteria=self.criteria,
                model=self.model,
                n_evaluations=self.n_evaluations,
            )
            score = result.scores[-1] if result.scores else 0.0
            self._scores.append(score)
            return score
        except Exception as e:
            logger.warning("ProgressTracker.update() failed: %s", e)
            self._scores.append(0.0)
            return 0.0

    @property
    def scores(self) -> list[float]:
        """All accumulated scores."""
        return list(self._scores)

    @property
    def steps(self) -> list[str]:
        """All accumulated step descriptions."""
        return list(self._steps)
