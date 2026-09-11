"""Adaptive Tool Context Routing — 按任务阶段动态裁剪工具 schema（Enhancement 6）。

市面 agent 的做法：全部 29 个工具的 schema 从第一步到最后一步
**一次性全量注入每次 LLM 请求**（OpenAI function-calling 的 schema
随每个请求携带）。估算每个工具定义约 150-250 tokens，29 个工具
≈ 4.5-7K tokens **每轮都白付**——而绝大多数任务根本用不到一半工具
（修个 bug 不需要 PPT/Excel 技能；写文档不需要 run_command）。

本模块按"任务阶段"动态裁剪每轮注入的 schema：

1. **活跃集追踪**：最近 N 轮实际调用过的工具进入活跃集；
   从未调用 + 与当前阶段无关的工具被裁掉
2. **阶段推断**（纯本地规则）：
   - 探索阶段（只读多、无写入）→ 保留读/搜/列 + 子代理
   - 修改阶段（有写入）→ 全量（改代码可能要跑命令验证）
   - 验证阶段（最近跑过 run_command/pytest）→ 保留命令+读+搜
   - 文档/收尾阶段（最近 skill_doc* / 写 README）→ 保留文档类
3. **保底不变量**：核心 5 工具（read/write/list/search/run_command）
   + task + memory **永远保留**——裁剪只影响扩展工具（17 Skill +
   5 MCP），主循环能力永不缺失
4. **零降智回退**：任何异常 → 返回全量工具列表（裁剪只省 token，
   绝不制造新失败模式）

省 token 的效果可量化：29 工具全量注入 ≈ 5-7K tokens/轮；
任务中期裁剪到核心+活跃（约 10 个）≈ 2K tokens/轮——
长任务（30+ 轮）累计省 90-120K tokens。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 永远保留的核心工具（主循环能力，裁剪不碰）
ALWAYS_KEEP: frozenset[str] = frozenset({
    "read_file", "write_file", "list_files", "search_text",
    "run_command", "task", "memory",
})
# 最近 N 轮内调用过的工具视为"活跃"，保留在 schema 里
RECENT_WINDOW = 4
# 裁剪下限：低于该数量不再裁（避免 schema 抖动过大导致模型行为突变）
MIN_KEEP = 8


def _phase_from_recent(recent_tools: list[str]) -> str:
    """从最近调用的工具序列推断任务阶段（纯规则）。"""
    if not recent_tools:
        return "explore"
    # 只看最近 4 个（窗口内）
    tail = recent_tools[-4:]
    has_write = any(t in ("write_file", "edit_file", "append_file") for t in tail)
    has_cmd = any(t == "run_command" for t in tail)
    has_doc = any(t.startswith("skill_doc") or "readme" in t.lower()
                  for t in tail)
    has_read = any(t in ("read_file", "search_text", "list_files") for t in tail)
    if has_doc:
        return "doc"
    if has_cmd:
        return "verify"
    if has_write:
        return "modify"
    if has_read:
        return "explore"
    return "explore"


def route_tools(
    all_tools: list[dict[str, Any]],
    recent_tool_calls: list[str],
    min_keep: int = MIN_KEEP,
) -> tuple[list[dict[str, Any]], str]:
    """按任务阶段裁剪工具 schema。

    Args:
        all_tools: 全量工具定义（registry.list_tools()）
        recent_tool_calls: 最近 N 次实际调用的工具名（时间正序）

    Returns:
        (裁剪后的工具定义列表, 阶段名)。保底不变量：ALWAYS_KEEP 永远在。
        任何异常 → (all_tools, "full")，零降智。
    """
    try:
        names_in = {
            t.get("function", {}).get("name", "") for t in all_tools}
        # 保底不变量：核心工具永不被裁
        keep = set(ALWAYS_KEEP & names_in)
        phase = _phase_from_recent(recent_tool_calls)

        if phase in ("explore", "doc", "verify"):
            # 探索/文档/验证阶段：核心 + 近期活跃 + 与阶段相关的扩展
            active = set(recent_tool_calls[-RECENT_WINDOW:])
            phase_extra: set[str] = set()
            if phase == "explore":
                # 探索期保留只读类 MCP 与调研技能
                phase_extra = {n for n in names_in
                               if n.startswith("mcp_git_") or n.startswith("mcp_system_")
                               or n in ("skill_research", "skill_explain_code")}
            elif phase == "doc":
                phase_extra = {n for n in names_in
                               if n.startswith("skill_doc")
                               or n in ("skill_generate_readme", "skill_commit_message",
                                        "skill_generate_changelog")}
            else:  # verify
                phase_extra = {n for n in names_in
                               if n.startswith("skill_test")
                               or n.startswith("mcp_debug_")}
            keep |= active & names_in
            keep |= phase_extra
        else:
            # modify 阶段：全量（改代码可能要跑命令/任意技能）
            keep = set(names_in)

        routed = [t for t in all_tools
                  if t.get("function", {}).get("name", "") in keep]
        # 保底数量：低于 min_keep 就退回全量（避免 schema 剧烈抖动）
        if len(routed) < min_keep:
            return all_tools, phase
        return routed, phase
    except Exception as e:
        logger.debug("tool routing failed, keeping full set: %s", e)
        return all_tools, "full"


def schema_token_saving(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> int:
    """估算裁剪后每轮省下的 schema token 数（每工具约 200 tokens）。"""
    return max(0, (len(before) - len(after)) * 200)
