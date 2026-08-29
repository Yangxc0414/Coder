"""task tool — expose subagent delegation to the model.

Design adopted from my-pi-agent (tools/builtin/task.py + tasks.py):
delegation must be a TOOL, not hidden Python plumbing — otherwise the
model can never use it (our Todo runs proved this: SubagentRunner existed
but was unreachable).

Anti-recursion invariant (structural, not prompt-level): child agents built
by SubagentRunner must never receive the `task` tool — enforced in
SubagentRunner.run by filtering the child registry.
"""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolResult


class TaskTool(Tool):
    name = "task"
    description = (
        "委派子任务给专业子代理（在独立上下文中运行，返回最终报告）。"
        "可用类型: researcher(只读代码分析/架构梳理, 适合动手前摸底)、"
        "test_specialist(编写并运行测试)、security_scanner(安全审计)、"
        "documenter(补充文档)。适合把独立的大块调查/验证工作并行外包，"
        "保持主上下文精简。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "交给子代理的完整任务描述（要自包含——子代理看不到主对话）",
            },
            "subagent_type": {
                "type": "string",
                "description": "子代理类型（可选，默认 researcher）",
                "enum": ["researcher", "test_specialist", "security_scanner", "documenter"],
                "default": "researcher",
            },
        },
        "required": ["prompt"],
    }

    def __init__(self, runner) -> None:
        # runner: coder_agent.extensions.base.SubagentRunner（避免循环导入，鸭子类型）
        self._runner = runner

    def execute(self, args: dict[str, Any]) -> ToolResult:
        from ..extensions.base import SubagentRequest

        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(error="prompt 不能为空——子代理看不到主对话，请给出自包含的任务描述")
        subagent_type = args.get("subagent_type", "researcher")

        try:
            result = self._runner.run(SubagentRequest(prompt=prompt, subagent_type=subagent_type))
        except Exception as e:
            # 工具契约：不向主循环抛异常（子代理构造/连接失败归为观察）
            return ToolResult(error=f"task delegation failed: {e}")
        if result.is_error:
            return ToolResult(error=result.final_output)
        output = (
            f"[{result.subagent_type} 报告] (耗时 {result.steps_used} 步)\n{result.final_output}"
        )
        return ToolResult(output=output)
