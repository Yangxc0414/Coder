"""memory tool — the MODEL can read/write long-term facts.

Design adopted from my-pi-agent (memory.py): passive memory that only the
host code touches is invisible to the model; making it a tool turns memory
into a capability. Simplified for our sync design:
- store: workspace `.coder_memory.md`, one `key: content` line per entry
- injected into the system prompt via Memory.get_summary (rebuilt each step,
  so new memories are visible immediately — no frozen-snapshot machinery)
- cap 30 entries: over-limit drops the oldest, and the tool says so
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


class MemoryTool(Tool):
    name = "memory"
    description = (
        "记住或查看跨步骤的长期事实（用户偏好、项目约定、已确认的配置等）。"
        "记住的内容会注入到后续每一步的系统提示中。适合记录：用户纠正过的偏好、"
        "反复出现的约定、关键决策。不适合记录临时状态（State 已覆盖）。"
        "支持 action=curate 自动整理记忆（去重 + 按重要性淘汰低价值条目）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "list", "curate"],
                "description": "remember=写入一条长期记忆；list=查看全部；"
                               "curate=自动整理（去重+淘汰低价值条目）",
            },
            "key": {
                "type": "string",
                "description": "记忆条目的短标签（action=remember 时必填，如 '用户偏好'）",
            },
            "content": {
                "type": "string",
                "description": "要记住的内容（action=remember 时必填）",
            },
        },
        "required": ["action"],
    }

    MAX_ENTRIES = 30
    # 单条内容上限：summary 注入截断 100 字符，但存储若不设限，
    # 模型可写入超大条目让 .coder_memory.md 无界膨胀
    MAX_CONTENT_CHARS = 2_000

    def __init__(self, memory, workspace: Path) -> None:
        # memory: coder_agent.memory.Memory 实例（鸭子类型，避免循环导入）
        self._memory = memory
        self._path = Path(workspace).resolve() / ".coder_memory.md"
        # 自动整理器：去重 + 重要性打分淘汰（跨会话学习）
        from ..memory_curator import MemoryCurator
        self._curator = MemoryCurator(memory, self._path.parent)

    def execute(self, args: dict[str, Any]) -> ToolResult:
        action = args.get("action", "list")
        if action == "remember":
            key = (args.get("key") or "").strip()[:100]
            content = (args.get("content") or "").strip()
            if not key or not content:
                return ToolResult(error="remember 需要 key 和 content 两个参数")
            # 单行化：换行会让记忆条目在 system prompt 中伪装成独立指令行
            # （持久化注入通道——条目会进入之后每次会话的系统提示）
            key = " ".join(key.split())
            content = " ".join(content.split())
            if len(content) > self.MAX_CONTENT_CHARS:
                content = content[: self.MAX_CONTENT_CHARS] + "...[内容过长已截断，记住要点即可]"
            self._memory.remember(key, content)
            self._curator.note_write(key)  # 记入整理元数据（次数/时间/分类）
            self._persist()
            n = len(self._memory.long_term)
            return ToolResult(output=f"已记住 [{key}]（长期记忆共 {n} 条，已注入后续系统提示）")
        if action == "list":
            if not self._memory.long_term:
                return ToolResult(output="（长期记忆为空）")
            # 按重要性排序展示（增强 4：整理后的记忆有轻重之分）
            ranked = self._curator.ranked(top_n=len(self._memory.long_term))
            lines = [
                f"- {k}（重要性 {score:.1f}，出现 {meta.get('times', 1)} 次）: {str(v)[:120]}"
                for k, score, meta in ranked
                for v in [self._memory.long_term.get(k)]
            ]
            return ToolResult(output="长期记忆（按重要性排序）:\n" + "\n".join(lines))
        if action == "curate":
            # 自动整理：去重 + 按重要性淘汰低分条目到 MAX_ENTRIES
            dup = self._curator.dedupe()
            res = self._curator.consolidate()
            self._persist()
            return ToolResult(output=(
                f"记忆已自动整理：去重 {dup} 条，淘汰低价值 {res['dropped']} 条，"
                f"保留 {res['kept']} 条（按 语义权重+复现频率+新鲜度 打分）"
            ))
        return ToolResult(error=f"未知 action: {action}")

    def load_from_disk(self) -> None:
        """启动时把持久化的记忆装回 long_term（供系统提示注入）。"""
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or ": " not in line:
                continue
            key, _, content = line.partition(": ")
            self._memory.remember(key, content)

    def _persist(self) -> None:
        # 落盘前按重要性淘汰（替代旧版"淘汰最老"——保留最没用才淘汰）
        self._curator.consolidate()
        entries = self._memory.long_term
        lines = [f"{k}: {v}" for k, v in entries.items()]
        self._path.write_text("\n".join(lines), encoding="utf-8")

    def consolidate_on_run_end(self) -> None:
        """运行结束钩子：整理由 agent 在 AGENT_ENDED 时调用（知识沉淀）。"""
        try:
            self._curator.consolidate()
            self._persist()
        except Exception:
            # 整理失败不影响主流程（记忆可用但暂不整理）
            import logging
            logging.getLogger(__name__).debug("memory curation failed", exc_info=True)
