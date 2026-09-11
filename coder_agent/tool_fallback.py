"""Tool Failure Fallback — 工具失败自动降级建议（Enhancement 5）。

市面 agent（mini-swe-agent / OneCode / smolagents）工具失败后只把
error 字符串回给模型，模型常常**换个参数重试同一工具**（如 read_file
报"路径不存在"后反复换路径名读），烧步数。本模块把工具失败变成
**结构化降级建议**：

1. **失败分类**：工具报错按语义分为 不存在 / 无权限 / 超时 / 命令
   不存在 / 语法错 等类别（纯规则，无 LLM 成本）
2. **降级路由表**：每个失败类别给"下一步该干什么"的具体建议
   （如 read_file 路径不存在 → 先 list_files 确认目录 / search_text
   按文件名搜；run_command 命令不存在 → 检查 PATH / 用 which 确认）
3. **与 ReAct 同通道**：建议以 user message 注入主循环（与现有命令
   失败策略提示同机制），不改变任何终止语义；建议只在"同类工具连续
   失败 ≥ 2 次"时触发（避免首犯就唠叨）

设计原则：全部本地规则；建议文案是"方向"而非"替模型做决定"，
保留模型自主权（只快不慢、只指路不代驾）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolFailure:
    """一次工具失败的分类结果。"""
    tool: str
    error: str
    category: str  # not_found / permission / timeout / command_not_found /
                   # syntax / timeout / unknown
    suggestion: str = ""


# 工具失败语义分类（正则按顺序匹配，先中的先赢）
_CATEGORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    # command not found 必须先于 not_found/timeout（"python3: command not found"
    # 同时含 "no such file"/"not found"，要优先归到 command_not_found）
    ("command_not_found", re.compile(r"command not found|not recognized|"
                                    r"不是内部或外部命令|no such command|"
                                    r"\bno python3\b|'python3'\b", re.I)),
    ("timeout", re.compile(r"timed out|timeout|超时", re.I)),
    ("not_found", re.compile(r"no such file|does not exist|not found|找不到|不存在", re.I)),
    ("permission", re.compile(r"permission denied|access denied|拒绝|permissionerror", re.I)),
    ("syntax", re.compile(r"syntaxerror|syntax error|indentationerror", re.I)),
    ("is_a_directory", re.compile(r"is a directory|不是文件", re.I)),
]


def classify_tool_failure(error_text: str) -> str:
    """把工具错误文本归入一个失败类别。"""
    text = error_text or ""
    for cat, pat in _CATEGORY_PATTERNS:
        if pat.search(text):
            return cat
    return "unknown"


# 降级路由表：(工具, 失败类别) → 下一步建议
_FALLBACK_ROUTES: dict[tuple[str, str], str] = {
    # 读文件失败
    ("read_file", "not_found"): (
        "文件不存在——先 list_files 查看目录结构确认实际路径，"
        "或用 search_text 按文件名/内容搜出真实位置再读"
    ),
    ("read_file", "is_a_directory"): (
        "目标是目录不是文件——改用 list_files 列内容，或 list_files 后挑出具体文件再读"
    ),
    ("read_file", "permission"): (
        "无权限读该文件——确认路径是否正确；若是系统目录考虑换工作区内等价文件"
    ),
    # 写文件失败
    ("write_file", "not_found"): (
        "写入的父目录不存在——先 list_files 确认目录层级，必要时用 run_command 建目录"
    ),
    ("write_file", "permission"): (
        "写入被拒——检查是否在工作区外（路径逃逸会被拦截）或文件只读"
    ),
    # 搜索失败
    ("search_text", "not_found"): (
        "没搜到——换个更宽松的关键词（如函数名而非完整语句），"
        "或先 list_files 确认文件扩展名范围"
    ),
    # 命令失败
    ("run_command", "command_not_found"): (
        "命令不存在——用 which/where 确认命令名（Windows 没有 python3，用 python；"
        "Linux 没有 where，用 which）；或检查是否未安装该工具"
    ),
    ("run_command", "timeout"): (
        "命令超时——把长任务改为 background 模式，或缩短命令加 --timeout 参数"
    ),
    ("run_command", "syntax"): (
        "命令语法错——检查引号/转义（Windows 下双引号嵌套用单引号，"
        "反斜杠在 Git Bash 里需转义）"
    ),
}


def fallback_suggestion(tool: str, category: str) -> str:
    """按 (工具, 类别) 查降级建议；无精确匹配给通用建议。"""
    if (tool, category) in _FALLBACK_ROUTES:
        return _FALLBACK_ROUTES[(tool, category)]
    if category == "not_found":
        return "目标不存在——先确认路径/名称，用 list_files 或 search_text 定位"
    if category == "permission":
        return "权限不足——检查路径是否越界或文件属性，换工作区内等价操作"
    if category == "timeout":
        return "超时——改后台执行或加超时参数"
    return "工具失败——分析报错找根因，不要原样重试，换个方法"


class ToolFallbackRouter:
    """工具失败自动降级路由器：跟踪同类连续失败，达阈值给建议。

    用法（agent._execute_tool_call 内）：
        router.note(tool_name, result.error or "")   # 每次工具执行后
        advice = router.due_advice()                  # 该不该注入建议
    建议只在"同一工具同类别连续失败 >= 2"时产出，避免首犯唠叨；
    工具一旦成功，其计数清零。
    """

    def __init__(self, threshold: int = 2) -> None:
        self._streak: dict[tuple[str, str], int] = {}
        self._threshold = threshold
        self._injected: set[tuple[str, str]] = set()

    def note(self, tool_name: str, error: str | None) -> None:
        """记录一次工具执行（error 非 None 即失败）。"""
        if error is None:
            # 成功 → 清零该工具所有类别的连败计数
            for key in [k for k in self._streak if k[0] == tool_name]:
                del self._streak[key]
            return
        category = classify_tool_failure(error)
        key = (tool_name, category)
        self._streak[key] = self._streak.get(key, 0) + 1

    def due_advice(self, tool_name: str) -> str | None:
        """该工具是否达到连败阈值且尚未注入过建议。"""
        for key, streak in self._streak.items():
            if key[0] != tool_name:
                continue
            if streak >= self._threshold and key not in self._injected:
                self._injected.add(key)  # 一个 (工具,类别) 只唠叨一次
                return (
                    f"{tool_name} 已连续失败 {streak} 次"
                    f"（{key[1]} 类）——{fallback_suggestion(tool_name, key[1])}"
                )
        return None

    def reset(self) -> None:
        """新 run 开始时重置。"""
        self._streak = {}
        self._injected = set()
