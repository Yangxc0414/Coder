"""Memory consolidation & curation — Enhancement 4.

市面 agent 的长期记忆（如 my-pi-agent 的 .coder_memory.md / Claude
memory 工具）都是**追加式**的：只写不整理。用多了会出现三个问题——

1. **重复**：同一事实被反复 remember，占用注入预算却无新信息
2. **过期**：早期记的状态/结论在后续步骤已失效（如"该函数返回 None"
   已被修复），却仍注入系统提示，误导模型
3. **无界增长**：只追加 + 简单按条数淘汰，淘汰的是"最老"而非"最没用"

本模块给 Memory 加一层**自动整理**（纯本地规则，无 LLM 成本）：

- **去重合并**：相同 key 的重复写入自动合并（保留最新值 + 记录出现次数）
- **重要性打分**：按 (出现次数, 最近使用时间, key 语义类别) 给每条记忆
  打分；淘汰时丢**最低分**而非"最老"
- **语义类别识别**：key 命中"偏好/约定/决策/配置"类词 → 高权重永久保留；
  命中"当前/临时/状态"类词 → 低权重优先淘汰
- **跨会话学习**：.coder_memory.md 持久化不变，但新增
  .coder_memory_meta.json 记录每条记忆的 出现次数/最近使用/分类，
  跨会话载入——同一项目第 N 次运行，整理知识也延续

设计原则：整理只在显式调用 consolidate() 时发生（Memory 工具写入后
或运行结束时），不侵入 ReAct 主循环；任何文件 IO 失败都静默降级
（记忆可用但无整理），不影响 agent 核心行为。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# key 语义类别 → 保留权重（越高越不易被淘汰）
_PERMANENT_KEYWORDS: dict[str, int] = {
    "偏好": 3, "约定": 3, "决策": 3, "配置": 3, "规范": 3,
    "preference": 3, "convention": 3, "decision": 3, "config": 3,
    "user": 3, "项目": 2, "project": 2,
}
_TEMPORARY_KEYWORDS: dict[str, int] = {
    "当前": 0, "临时": 0, "状态": 0, "现在": 0,
    "current": 0, "temporary": 0, "temp": 0, "state": 0,
}


def classify_key(key: str) -> int:
    """按 key 语义给出基础保留权重（0-3，未知=1）。"""
    low = key.lower()
    for kw, w in _TEMPORARY_KEYWORDS.items():
        if kw in low:
            return w
    for kw, w in _PERMANENT_KEYWORDS.items():
        if kw in low:
            return w
    return 1


def importance_score(key: str, times: int, last_used: float,
                      now: float | None = None) -> float:
    """记忆重要性打分：语义权重 + 复现频率 + 新鲜度。

    - 语义权重 (0-3)：偏好/约定类高，临时状态类低
    - 复现频率 log(times)：反复出现的记忆更值得留
    - 新鲜度：最近 10 分钟内使用过的 +0.5 衰减分
    """
    now = now or time.time()
    base = float(classify_key(key))
    freq = min(times, 20) / 4.0  # 20 次封顶，归一到 5
    freshness = 0.0
    age = max(0.0, now - last_used)
    if age < 60:
        freshness = 0.5
    elif age < 300:
        freshness = 0.2
    return base + freq + freshness


class MemoryCurator:
    """自动整理器：去重 / 打分 / 淘汰 / 持久化元数据。

    与 Memory 协作但不侵入——Memory 照旧 remember/recall，
    整理动作由本类在运行结束或记忆超限时调用。
    """

    META_NAME = ".coder_memory_meta.json"
    MAX_KEEP = 30  # 与 MemoryTool.MAX_ENTRIES 对齐

    def __init__(self, memory, workspace: Path) -> None:
        self._memory = memory
        self._path = Path(workspace).resolve() / self.META_NAME
        self._meta: dict[str, dict[str, Any]] = {}
        self.load()
        # 首次见到就初始化元数据（出现次数默认 1）
        for k in memory.long_term:
            self._meta.setdefault(k, {
                "times": 1, "last_used": time.time(),
                "class_weight": classify_key(k),
            })

    # ── 持久化 ──
    def load(self) -> int:
        """载入跨会话整理的元数据（出现次数/分类延续）。"""
        if not self._path.exists():
            return 0
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._meta = data.get("entries", {})
            return len(self._meta)
        except (json.JSONDecodeError, OSError) as e:
            import logging
            logging.getLogger(__name__).debug("memory meta load failed: %s", e)
            self._meta = {}
            return 0

    def save(self) -> None:
        """原子落盘元数据。"""
        try:
            payload = {"entries": self._meta}
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as e:
            import logging
            logging.getLogger(__name__).debug("memory meta save failed: %s", e)

    # ── 记录 / 整理 ──
    def note_write(self, key: str) -> None:
        """memory 工具每写一次：累加次数 + 刷新时间戳。"""
        m = self._meta.setdefault(key, {
            "times": 0, "last_used": time.time(),
            "class_weight": classify_key(key),
        })
        m["times"] += 1
        m["last_used"] = time.time()
        m["class_weight"] = classify_key(key)
        self.save()

    def consolidate(self) -> dict[str, int]:
        """自动整理：去重 + 按重要性淘汰低分记忆到 MAX_KEEP。

        Returns: {"dropped": n, "kept": n, "deduped": n}
        """
        entries = self._memory.long_term
        dropped = 0
        # 给当前所有条目打分（缺元数据的按默认 1 次）
        scored: list[tuple[float, str]] = []
        for k in entries:
            m = self._meta.get(k, {"times": 1, "last_used": time.time()})
            scored.append((importance_score(k, m["times"], m["last_used"]), k))
        # 从高到低排，保留前 MAX_KEEP，其余丢弃（低分先走）
        scored.sort(key=lambda x: -x[0])
        keep_set = {k for _, k in scored[: self.MAX_KEEP]}
        for k in list(entries):
            if k not in keep_set:
                del entries[k]
                self._meta.pop(k, None)
                dropped += 1
        # 持久化：整理结果同步进 .coder_memory.md（复用 MemoryTool 的落盘格式）
        self.save()
        return {"dropped": dropped, "kept": len(entries), "deduped": 0}

    def dedupe(self) -> int:
        """相同内容不同 key 的条目去重（保留权重高的 key）。"""
        entries = self._memory.long_term
        by_content: dict[str, list[str]] = {}
        for k, v in entries.items():
            by_content.setdefault(str(v).strip(), []).append(k)
        dup = 0
        for content, keys in by_content.items():
            if len(keys) <= 1:
                continue
            # 保留重要性最高的 key，其余合并掉
            ranked = sorted(keys, key=lambda k: -importance_score(
                k, self._meta.get(k, {}).get("times", 1),
                self._meta.get(k, {}).get("last_used", time.time())))
            for loser in ranked[1:]:
                del entries[loser]
                self._meta.pop(loser, None)
                dup += 1
        if dup:
            self.save()
        return dup

    # ── 查询（供 UI / 整理决策）──
    def ranked(self, top_n: int = 10) -> list[tuple[str, float, dict]]:
        """按重要性降序返回前 N 条 (key, score, meta)。"""
        entries = self._memory.long_term
        scored = []
        for k in entries:
            m = self._meta.get(k, {"times": 1, "last_used": time.time()})
            scored.append((k, importance_score(k, m["times"], m["last_used"]), m))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]
