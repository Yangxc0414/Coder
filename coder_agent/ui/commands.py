"""共享斜杠命令定义 — CLI 与 Web 客户端共用。

CLI（coder_agent/ui/cli/repl.py）与 Web（coder_agent/ui/web/server.py
的 /api/commands）各自维护一份命令列表容易失同步；这里集中定义，
两端都从这里取，新增命令只改一处。
"""

from __future__ import annotations

COMMANDS: list[dict] = [
    {"name": "/help", "desc": "显示命令帮助", "hasArgs": False},
    {"name": "/status", "desc": "显示运行状态与当前工作区", "hasArgs": False},
    {"name": "/tools", "desc": "列出模型可用的工具（按核心/Skill/MCP 分类）", "hasArgs": False},
    {"name": "/skills", "desc": "显示内置 Skills 详情（描述/何时使用/参数）", "hasArgs": False},
    {"name": "/mcp", "desc": "显示 MCP 工具详情（按服务器分组）", "hasArgs": False},
    {"name": "/model", "desc": "切换模型（无参数显示列表，可输入名称）", "hasArgs": True,
     "argHint": "<模型名>，如 agnes-2.5-pro"},
    {"name": "/mode", "desc": "切换执行模式 full/plan/dry-run（目标用 /goal）", "hasArgs": True,
     "argHint": "full | plan | dry-run"},
    {"name": "/goal", "desc": "设置会话目标并立即开始执行（持续直到达成）", "hasArgs": True,
     "argHint": "<目标描述>，或 clear 清除"},
    {"name": "/sessions", "desc": "列出会话（含任务预览）", "hasArgs": False},
    {"name": "/resume", "desc": "恢复会话（无参=最新；或输入序号）", "hasArgs": True,
     "argHint": "<会话序号>，空=最新"},
    {"name": "/compact", "desc": "手动压缩最近一次运行的上下文", "hasArgs": False},
    {"name": "/history", "desc": "显示最近一次运行的最近消息", "hasArgs": False},
    {"name": "/trace", "desc": "显示最近一次运行的工具追踪摘要", "hasArgs": False},
    {"name": "/clear", "desc": "清空对话显示", "hasArgs": False},
    # ── 演示回放模式 ─────────────────────────────────────────────────────
    {"name": "/record", "desc": "录制本次运行到 trace 文件（用于离线回放）",
     "hasArgs": True,
     "argHint": "<输出路径，如 D:/replays/demo.jsonl>"},
    {"name": "/replay", "desc": "从 trace 文件回放事件流（不依赖 API）",
     "hasArgs": True,
     "argHint": "<trace 文件路径，或留空选最近一条>"},
    {"name": "/health", "desc": "检查 API 连通性与延迟（演示前必跑）", "hasArgs": False},
]
