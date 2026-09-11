"""Plan-Execute-Verify orchestrator — the core differentiator.

市面多数 agent（mini-swe-agent / OneCode / smolagents）都是纯 ReAct 线性循环：
模型逐步试错，上下文随步骤线性膨胀。Claude Code / OpenHands 有 plan 模式但
plan 与 execute 是同一份上下文里的顺序阶段，无结构化分解、无验证门控。

本模块把 ReAct 升级为 **Plan → Execute → Verify** 三段式：

1. **Plan**（planner）：LLM 把任务分解为 2-8 个原子子目标（JSON 数组），
   每个子目标标注依赖关系。失败回退到纯 ReAct（不降智）。
2. **Execute**（executor）：依赖图拓扑分层。同一层内无依赖的子目标
   **并行派发**给子代理（ThreadPool，上限 3），各子代理在独立上下文中
   运行，结果按层汇总回主上下文——上下文占用 = 子代理报告数 × 报告长度，
   而非 全部步骤 × 全部工具输出。
3. **Verify**（verifier gate）：每层完成后跑 Verifier 基线对比，
   失败且无文件变更 → 该层标记完成（存量问题不背锅）；失败且有变更
   → 注入修复指令进入下一层继续。

零 LLM 依赖保证：所有 LLM 调用点失败时自动回退 ReAct，系统只快不慢。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 子目标数量边界：太少没分解价值，太多并行收益被调度开销吃掉
MIN_SUBGOALS = 2
MAX_SUBGOALS = 8
# 并行度：API 限流敏感（DeepSeek 5s/窗口），3 路是经验最优
PARALLEL_LIMIT = 3
# 子代理报告回传父上下文的字符上限（与 TaskTool.MAX_REPORT_CHARS 一致）
MAX_REPORT_CHARS = 12_000

PLAN_SYSTEM_PROMPT = """\
你是任务分解专家。把编程任务分解为 2-{max} 个原子子目标。

规则：
- 每个子目标必须独立可执行、可验证（有明确的完成判据）
- 标注 depends_on：该子目标开始前必须先完成哪些子目标（按 index）
- 无依赖的子目标会并行执行，越多并行越快
- 涉及同一文件的修改放在同一层或标注依赖（避免写冲突）
- 纯文本调研类子目标交给 researcher，写测试交给 test_specialist，
  写文档交给 documenter，安全审计交给 security_scanner

严格输出 JSON 数组（不要 markdown 代码块、不要解释）：
[
  {{"index": 0, "goal": "子目标描述", "subagent_type": "researcher", "depends_on": []}},
  ...
]
"""


@dataclass
class Subgoal:
    """一个原子子目标。"""
    index: int
    goal: str
    subagent_type: str = "researcher"
    depends_on: list[int] = field(default_factory=list)


@dataclass
class PlanResult:
    """分解结果 + 拓扑分层。"""
    subgoals: list[Subgoal]
    layers: list[list[Subgoal]]

    def prompt(self) -> str:
        lines = []
        for i, sg in enumerate(self.subgoals, 1):
            dep = f"（依赖 #{sg.depends_on}）" if sg.depends_on else "（无依赖）"
            lines.append(f"{i}. [{sg.subagent_type}] {sg.goal} {dep}")
        return "\n".join(lines)


class Planner:
    """LLM 任务分解器。"""

    def __init__(self, llm_client, timeout: int = 30) -> None:
        self._llm = llm_client
        self._timeout = timeout

    def plan(self, task: str, max_subgoals: int = MAX_SUBGOALS) -> PlanResult | None:
        """分解任务。任何异常/格式错误返回 None（调用方回退 ReAct）。"""
        prompt = PLAN_SYSTEM_PROMPT.format(max=max_subgoals)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"任务：{task}\n请输出 JSON 数组。"},
        ]
        try:
            # timeout 是 LLMClient 的 OpenAI API 参数；测试用的 duck-typed
            # double 不接受它——缺参时报错即视为"分解不可用"，回退 ReAct
            try:
                resp = self._llm.chat(messages, max_tokens=4096, timeout=self._timeout)
            except TypeError:
                resp = self._llm.chat(messages, max_tokens=4096)
            raw = resp.content or ""
        except Exception as e:
            logger.debug("planner LLM call failed: %s", e)
            return None
        goals = self._parse(raw, task)
        if goals is None or len(goals) < MIN_SUBGOALS:
            return None
        layers = self._topological_layers(goals)
        if layers is None:
            return None  # 环依赖 → 整体回退
        return PlanResult(subgoals=goals, layers=layers)

    def _parse(self, raw: str, task: str) -> list[Subgoal] | None:
        """容忍 markdown 代码块包裹 + 首尾杂质的 JSON 解析。"""
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.S)
        if m:
            text = m.group(1)
        elif not text.startswith("["):
            # 截出第一个 [ 到最后一个 ]
            i, j = text.find("["), text.rfind("]")
            if i < 0 or j <= i:
                return None
            text = text[i:j + 1]
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(items, list):
            return None
        goals: list[Subgoal] = []
        for it in items[:MAX_SUBGOALS]:
            if not isinstance(it, dict) or not str(it.get("goal", "")).strip():
                continue
            goals.append(Subgoal(
                index=it.get("index", len(goals)),
                goal=str(it["goal"]).strip(),
                subagent_type=str(it.get("subagent_type") or "researcher").strip(),
                depends_on=[int(d) for d in (it.get("depends_on") or [])
                            if isinstance(d, (int, str)) and str(d).lstrip("-").isdigit()],
            ))
        # 依赖越界裁剪（LLM 幻觉保护）
        valid = {g.index for g in goals}
        for g in goals:
            g.depends_on = [d for d in g.depends_on if d in valid and d != g.index]
        # 去重（按 index 首次出现）
        seen: set[int] = set()
        deduped = []
        for g in goals:
            if g.index in seen:
                continue
            seen.add(g.index)
            deduped.append(g)
        # 无依赖且 goal 与任务原文雷同（LLM 没分解，复读任务）→ 视为无效
        if len(deduped) >= MIN_SUBGOALS and all(
            not g.depends_on and g.goal in task for g in deduped
        ):
            return None
        return deduped or None

    def _topological_layers(self, goals: list[Subgoal]) -> list[list[Subgoal]] | None:
        """Kahn 分层。检测到环 → None（整体回退 ReAct，宁慢勿错）。"""
        remaining = {g.index: g for g in goals}
        layers: list[list[Subgoal]] = []
        done: set[int] = set()
        while remaining:
            current = [
                g for g in remaining.values()
                if all(d in done for d in g.depends_on)
            ]
            if not current:
                return None  # 环
            layers.append(sorted(current, key=lambda g: g.index))
            for g in current:
                remaining.pop(g.index)
                done.add(g.index)
        return layers


def should_use_planner(task: str, mode: Any) -> bool:
    """任务足够复杂（多步骤/大改）且非只读模式才启用分解。"""
    # 只读模式（plan / dry-run）不做实际修改，分解并行无意义
    try:
        from coder_agent.mode import AgentMode
        if mode in (AgentMode.PLAN, AgentMode.DRY_RUN):
            return False
    except Exception:
        pass
    # 短任务（< 40 字且无多步骤标记）不分解——分解本身的 LLM 调用不划算
    if len(task) < 40:
        return False
    multi_step_markers = ("并且", "然后", "再", "同时", "分别", "并", "then",
                          "and then", "as well as", "also")
    return any(m in task.lower() for m in multi_step_markers) or len(task) > 120
