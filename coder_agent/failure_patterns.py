"""失败模式库 + 修复策略建议器 — Verification 2.0 的核心差异。

市面 agent（SWE-agent / OpenHands）验证失败后只会注入一句
"测试失败了请修复"——模型在同样的失败上反复用同样的方法，
烧满步数。本模块把"失败"变成结构化信号：

1. **指纹化失败**：每次验证失败提取 (类型, 文件, 签名) 三元组指纹，
   相同指纹第 2 次出现即判定"同一失败模式"。
2. **策略轮换**：同一指纹失败 2 次后，注入"换方法"指令
   （附按失败类型给出的修复策略库：测试断言类/语法类/导入类/类型类/
   运行时类各自给不同的突破口提示）。
3. **跨会话持久化**：失败指纹 + 是否被后续解决 落到
   .coder_failure_patterns.json，下次运行启动时载入——同类失败
   直接告诉模型"历史 N 次失败后用什么方法成功了"，把试错知识
   沉淀为可复用资产（市面 agent 全部会话无记忆，每次都从零试错）。

设计原则：纯本地文件 IO，无 LLM 依赖；建议以 user message 注入
主循环（与现有验证门控消息同通道），不改变任何终止语义。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STORE_NAME = ".coder_failure_patterns.json"

# 按失败类别的修复策略库（注入给模型的"换方法"提示，来自真实 debug 经验）
REPAIR_STRATEGIES: dict[str, list[str]] = {
    "assert": [
        "先读失败断言处的实际值与期望值差异，不要猜",
        "检查是 off-by-one / 边界值 / 浮点精度哪一类，针对性改",
        "用最小复现：构造让该断言单独失败的输入，逐个变量定位",
    ],
    "syntax": [
        "定位报错行号前后 5 行，检查括号/冒号/缩进",
        "常见：f-string 内嵌套引号、walrus 运算符误用、装饰器缺 @",
        "python -c 'import ast; ast.parse(open(文件).read())' 可单独验证语法",
    ],
    "import": [
        "确认模块路径相对于 PYTHONPATH 是否正确（src 布局 vs 扁平布局）",
        "循环导入 → 把 import 移到函数内或提取公共模块",
        "检查是否把测试目录当包导入导致重名遮蔽",
    ],
    "type": [
        "看实际传入类型与期望类型的差异，不要只看报错信息",
        "None 传播：调用链上游某处返回了 None 被下游当对象用",
        "str vs int：JSON 解析后的数字常是 str，先转换再比较",
    ],
    "runtime": [
        "读 traceback 最底层一帧（不是最顶层）定位真正出错点",
        "区分'修代码'与'修调用方'：错在调用方就先改调用",
        "加一行 print 复现路径比盲改更快收敛",
    ],
}


@dataclass
class FailureRecord:
    """一次验证失败的指纹化记录。"""
    fingerprint: str
    category: str
    files: list[str] = field(default_factory=list)
    message: str = ""
    first_seen: float = 0.0
    occurrences: int = 1
    solved: bool = False
    strategy: str = ""


def _classify_failure(summary: str, detail: str) -> str:
    """从验证输出粗分类失败模式（决定用哪组修复策略）。"""
    text = (summary + " " + detail).lower()
    if "syntaxerror" in text or "indentation" in text or "expected token" in text:
        return "syntax"
    if "modulenotfound" in text or "importerror" in text or "cannot import" in text:
        return "import"
    if "typeerror" in text or "unsupported operand" in text:
        return "type"
    if "assert" in text:
        return "assert"
    return "runtime"


def _fingerprint(category: str, files: list[str], summary: str) -> str:
    """稳定指纹：同类别 + 同文件集合 + 同摘要哈希。"""
    key = category + "|" + ",".join(sorted(files)) + "|" + (summary or "")[:120]
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


class FailurePatternLibrary:
    """失败模式库：指纹化记录 + 策略轮换 + 跨会话持久化。"""

    MAX_STORE = 200  # 防止无界增长

    def __init__(self, workspace: Path) -> None:
        self._path = Path(workspace) / STORE_NAME
        self._records: dict[str, FailureRecord] = {}
        self.load()

    # ── 持久化 ──
    def load(self) -> int:
        """载入历史失败模式（跨会话知识沉淀）。"""
        if not self._path.exists():
            return 0
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("failure pattern store unreadable: %s", e)
            return 0
        for item in data.get("records", [])[-self.MAX_STORE:]:
            rec = FailureRecord(
                fingerprint=item.get("fingerprint", ""),
                category=item.get("category", "runtime"),
                files=item.get("files", []),
                message=item.get("message", ""),
                first_seen=item.get("first_seen", 0.0),
                occurrences=item.get("occurrences", 1),
                solved=item.get("solved", False),
                strategy=item.get("strategy", ""),
            )
            if rec.fingerprint:
                self._records[rec.fingerprint] = rec
        return len(self._records)

    def save(self) -> None:
        """落盘（原子写：先写临时文件再 rename，避免半截 JSON）。"""
        try:
            payload = {"records": [vars(r) for r in self._records.values()][-self.MAX_STORE:]}
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as e:
            logger.debug("failure pattern save failed: %s", e)

    # ── 记录 / 查询 ──
    def record(self, category: str, files: list[str], summary: str,
               detail: str = "") -> FailureRecord:
        fp = _fingerprint(category, files, summary)
        rec = self._records.get(fp)
        if rec is None:
            rec = FailureRecord(fingerprint=fp, category=category,
                                files=files, message=summary[:200],
                                first_seen=time.time())
            self._records[fp] = rec
        else:
            rec.occurrences += 1
        rec.strategy = REPAIR_STRATEGIES.get(category, REPAIR_STRATEGIES["runtime"])[0]
        self.save()
        return rec

    def occurrences(self, fingerprint: str) -> int:
        rec = self._records.get(fingerprint)
        return rec.occurrences if rec else 0

    def strategy_for(self, fingerprint: str) -> str:
        """同指纹第 N 次失败 → 轮换第 N 条策略（逼模型换方法）。"""
        rec = self._records.get(fingerprint)
        if rec is None:
            return ""
        pool = REPAIR_STRATEGIES.get(rec.category, REPAIR_STRATEGIES["runtime"])
        idx = (rec.occurrences - 1) % len(pool)
        return pool[idx]

    def solved_note(self, fingerprint: str) -> str:
        """该指纹历史是否已被解决（跨会话知识）。"""
        rec = self._records.get(fingerprint)
        if rec and rec.solved and rec.strategy:
            return (f"历史经验：同类失败曾出现 {rec.occurrences} 次并成功修复，"
                    f"当时有效策略：{rec.strategy}")
        return ""

    def mark_solved(self, fingerprints: list[str]) -> int:
        """验证通过时，把当前已知失败标记为已解决（知识闭环）。"""
        n = 0
        for fp in fingerprints:
            rec = self._records.get(fp)
            if rec and not rec.solved:
                rec.solved = True
                n += 1
        if n:
            self.save()
        return n

    # ── 注入主循环的修复建议消息 ──
    def repair_advice(self, category: str, summary: str, detail: str,
                      same_failure_count: int) -> str | None:
        """生成注入主循环的修复指令。首轮失败给常规提示；
        第 2 轮起（同指纹）强制"换方法" + 策略轮换 + 历史经验。"""
        if same_failure_count < 2:
            return None
        lines = [
            f"同一失败已出现 {same_failure_count} 次——之前的方法无效，必须换方法：",
        ]
        pool = REPAIR_STRATEGIES.get(category, REPAIR_STRATEGIES["runtime"])
        for i in range(min(same_failure_count, len(pool))):
            lines.append(f"  尝试方向 {i + 1}: {pool[i]}")
        return "\n".join(lines)
