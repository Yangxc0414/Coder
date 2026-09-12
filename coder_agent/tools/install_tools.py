"""install_skill / install_mcp — 模型在对话中自主下载并注册扩展。

对齐 task_tool.py / memory_tool.py 的范式：
- 构造器注入 registry（可即时注册）+ workspace + mode（dry-run 只模拟）
- execute() 返回 ToolResult（success / error），绝不向主循环抛异常
- 装成功立刻 registry.register(...)，本会话后续步骤即可调用
- 持久化走 installer.list_installed()，新会话 create_default_registry 复用
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool, ToolResult
from ..extensions.installer import ExtensionInstaller


class _InstallToolBase(Tool):
    """共享：持有 registry + installer，execute 子类实现。"""

    def __init__(self, registry, workspace: Path, mode) -> None:
        self._registry = registry
        self._workspace = Path(workspace)
        self._mode = mode
        self._dry_run = bool(getattr(mode, "value", None)) == "dry-run"
        self._installer = ExtensionInstaller(dry_run=self._dry_run)

    def _register_safe(self, tool) -> None:
        name = tool.name
        if name in self._registry:
            return
        try:
            self._registry.register(tool)
        except Exception as e:  # 重名/结构错误不阻塞，仅提示
            pass


class InstallSkillTool(_InstallToolBase):
    name = "install_skill"
    description = (
        "从 GitHub 仓库下载并安装一个 Skill（prompt 型扩展），装完立刻可在后续步骤调用。"
        "当用户要求'下载/安装某个 skill'或任务需要某个仓库里的 skill 时使用。"
        "参数: repo_url(GitHub/GitLab 仓库地址), subpath(仓库内 skill 目录, 可省), "
        "skill_name(可选, 缺省用 subpath 或仓库名)。安装会 git clone 到 ~/.coder_extensions/ "
        "(不进工作区、不进 git)，并解析 SKILL.md / skill.json。"
        "注意: 仅支持 github.com / gitlab.com 的 git 仓库 URL。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "repo_url": {"type": "string", "description": "GitHub/GitLab 仓库地址"},
            "subpath": {"type": "string", "description": "仓库内 skill 子目录（可选）"},
            "skill_name": {"type": "string", "description": "skill 短名（可选）"},
        },
        "required": ["repo_url"],
    }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        repo_url = (args.get("repo_url") or "").strip()
        subpath = (args.get("subpath") or "").strip() or None
        skill_name = (args.get("skill_name") or "").strip() or None
        if not repo_url:
            return ToolResult(error="install_skill 需要 repo_url 参数")
        try:
            inst = self._installer.install_skill(repo_url, subpath, skill_name)
        except Exception as e:
            return ToolResult(error=f"install_skill 失败: {e}")
        if self._dry_run:
            return ToolResult(output=(
                f"[dry-run] 模拟安装 skill {inst.skill.name} "
                f"(来源 {repo_url})，未实际落盘"
            ))
        self._register_safe(inst.skill)
        return ToolResult(output=(
            f"已安装 skill：{inst.skill.name}（来源 {repo_url}）。\n"
            f"该 skill 现已注册，可在后续步骤以工具名 {inst.skill.name} 调用。"
        ))


class InstallMcpTool(_InstallToolBase):
    name = "install_mcp"
    description = (
        "从 GitHub 仓库下载并安装一个 MCP server（命令型工具集），装完立刻可用。"
        "当用户要求'下载/安装某个 MCP 工具'或任务需要某个 MCP server 时使用。"
        "参数: repo_url(GitHub/GitLab 仓库), subpath(server 子目录, 可省), "
        "server_name(可选), launch_command(启动命令, 可选; 缺省按 package.json/main.py 推断, "
        "如 'npx my-server' 或 'python main.py')。"
        "MCP server 通过 stdio JSON-RPC 启动（带 30s 超时 + 密钥隔离环境变量）。"
        "注意: 仅支持 github.com / gitlab.com 的 git 仓库 URL。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "repo_url": {"type": "string", "description": "GitHub/GitLab 仓库地址"},
            "subpath": {"type": "string", "description": "仓库内 MCP server 子目录（可选）"},
            "server_name": {"type": "string", "description": "MCP server 短名（可选）"},
            "launch_command": {"type": "string", "description": "MCP server 启动命令（可选）"},
        },
        "required": ["repo_url"],
    }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        repo_url = (args.get("repo_url") or "").strip()
        subpath = (args.get("subpath") or "").strip() or None
        server_name = (args.get("server_name") or "").strip() or None
        launch = (args.get("launch_command") or "").strip() or None
        if not repo_url:
            return ToolResult(error="install_mcp 需要 repo_url 参数")
        try:
            inst = self._installer.install_mcp(repo_url, subpath, server_name, launch)
        except Exception as e:
            return ToolResult(error=f"install_mcp 失败: {e}")
        if self._dry_run:
            return ToolResult(output=(
                f"[dry-run] 模拟安装 MCP server {inst.server_name} "
                f"(来源 {repo_url}, 启动 {inst.launch_command})，未实际落盘"
            ))
        self._register_safe(inst.mcp_tool)
        return ToolResult(output=(
            f"已安装 MCP server：{inst.server_name}（来源 {repo_url}，启动 {inst.launch_command}）。\n"
            f"该 MCP 工具现已注册，可在后续步骤以工具名 {inst.mcp_tool.name} 调用。"
        ))
