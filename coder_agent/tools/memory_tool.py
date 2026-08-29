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
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "list"],
                "description": "remember=写入一条长期记忆；list=查看全部",
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
            self._persist()
            n = len(self._memory.long_term)
            return ToolResult(output=f"已记住 [{key}]（长期记忆共 {n} 条，已注入后续系统提示）")
        if action == "list":
            if not self._memory.long_term:
                return ToolResult(output="（长期记忆为空）")
            lines = [f"- {k}: {str(v)[:120]}" for k, v in self._memory.long_term.items()]
            return ToolResult(output="长期记忆:\n" + "\n".join(lines))
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
        entries = self._memory.long_term
        if len(entries) > self.MAX_ENTRIES:
            # 淘汰最早的条目（dict 保序）
            for k in list(entries)[: len(entries) - self.MAX_ENTRIES]:
                entries.pop(k)
        lines = [f"{k}: {v}" for k, v in entries.items()]
        self._path.write_text("\n".join(lines), encoding="utf-8")
