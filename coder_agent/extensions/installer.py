"""On-demand extension installer — agent 自己从 GitHub 下载 Skill / MCP。

设计（详见 计划）：
- 来源 = GitHub 仓库（git clone --depth 1），白名单域名防 file:// / javascript://。
- Skill：一个目录一个 skill（Anthropic 约定：SKILL.md / skill.json / skill.yaml），
  解析后构造 base.Skill，写 manifest.json 持久化。
- MCP：clone 后定位 server 启动脚本，构造 McpTool + stdio 子进程 executor
  （subprocess.Popen + timeout + _safe_env，绝不 shell=True）。
- 全部落 ~/.coder_extensions/（不进工作区、不进 git），失败回滚删目录。
- dry-run 模式只"模拟安装"不落盘。
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Skill, McpTool


# ── 安装根目录 ─────────────────────────────────────────────
EXTENSIONS_DIR = Path.home() / ".coder_extensions"


# 白名单域名（防 file:// / javascript:// / 任意 http）
_ALLOWED_HOSTS = ("github.com", "gitlab.com")
# 允许的 URL 形式（仅 HTTPS / git 协议，禁明文 http）：
#   https://github.com/owner/repo[.git][?ref=]
#   git@github.com:owner/repo
_URL_RE = re.compile(
    r"^(?:"
    r"https://(github\.com|gitlab\.com)/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(\.git)?(\?[A-Za-z0-9_=&.\-]+)?"
    r"|git@(github\.com|gitlab\.com):[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+"
    r")$",
    re.IGNORECASE,
)


def _validate_repo_url(url: str) -> str:
    """校验 repo_url 在白名单域名的合法 git 形式，返回归一化 URL。"""
    u = (url or "").strip()
    if not u:
        raise ValueError("repo_url 不能为空")
    if not _URL_RE.match(u):
        raise ValueError(
            f"不安全的仓库 URL: {u!r}。仅支持 https://github.com|gitlab.com/owner/repo"
        )
    return u


def _safe_env() -> dict[str, str]:
    """复用 shell.py 的密钥隔离策略：传全部环境变量，但剔除名字像凭据的。"""
    import re as _re

    secret = _re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", _re.IGNORECASE)
    return {k: v for k, v in os.environ.items() if not secret.search(k)}


@dataclass
class InstalledSkill:
    """一次成功的 skill 安装结果。"""

    skill: Skill
    manifest_path: Path
    repo_url: str
    skill_name: str
    source: str = "installed"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "skill",
            "name": self.skill.name,
            "display": self.skill_name,
            "repo_url": self.repo_url,
            "description": self.skill.description,
            "when_to_use": getattr(self.skill, "_when_to_use", None) or "",
            "command": getattr(self.skill, "_command", ""),
            "allowed_tools": list(getattr(self.skill, "_allowed_tools", ()) or ()),
            "installed_at": str(self.manifest_path and datetime.datetime.now().isoformat()),
        }


@dataclass
class InstalledMcp:
    """一次成功的 MCP server 安装结果。"""

    mcp_tool: McpTool
    manifest_path: Path
    repo_url: str
    server_name: str
    launch_command: str
    source: str = "installed"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "mcp",
            "name": self.mcp_tool.name,
            "display": self.server_name,
            "repo_url": self.repo_url,
            "launch_command": self.launch_command,
            "description": self.mcp_tool.description,
            "installed_at": datetime.datetime.now().isoformat(),
        }


class ExtensionInstaller:
    """按需下载 + 注册 + 持久化 Skill / MCP。

    用法（agent 工具内部）：
        inst = ExtensionInstaller(dry_run=False)
        sk = inst.install_skill("https://github.com/owner/skill-repo",
                                 subpath="skills/xlsx", skill_name="xlsx")
        registry.register(sk.skill)          # 立即可用
        inst.list_installed()                # 跨会话恢复
    """

    def __init__(self, dry_run: bool = False, extensions_dir: Path | None = None) -> None:
        self._dry_run = dry_run
        self._root = Path(extensions_dir) if extensions_dir else EXTENSIONS_DIR

    # ── 公共：git clone（白名单 + 受限目录 + 回滚）─────────────
    def _clone(self, repo_url: str, target: Path) -> None:
        """把 repo_url clone 到 target（depth 1）。失败抛异常，由调用方回滚。"""
        _validate_repo_url(repo_url)
        if self._dry_run:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        cmd = ["git", "clone", "--depth", "1", repo_url, str(target)]
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            cwd=str(self._root), env=_safe_env(),
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        if r.returncode != 0:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError(f"git clone 失败 (rc={r.returncode}): {(r.stderr or r.stdout)[-400:]}")

    def _rollback(self, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    # ── Skill ─────────────────────────────────────────────────
    def install_skill(
        self,
        repo_url: str,
        subpath: str | None = None,
        skill_name: str | None = None,
    ) -> InstalledSkill:
        """从 GitHub 仓库下载一个 Skill 并构造 Skill 实例。

        约定：skill 目录内含 SKILL.md（YAML front-matter + 正文）或 skill.json。
        subpath 为仓库内的相对子目录；缺省时用 skill_name 定位。
        """
        name = (skill_name or subpath or repo_url.rstrip("/").split("/")[-1]).strip().lstrip(".")
        name = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)[:40] or "skill"
        dest = self._root / name
        self._clone(repo_url, dest)

        skill_dir = dest / (subpath or name)
        skill_dir = skill_dir if skill_dir.is_dir() else dest
        try:
            parsed = self._parse_skill_dir(skill_dir, name)
        except Exception as e:
            self._rollback(dest)
            raise RuntimeError(f"解析 skill 定义失败: {e}")

        manifest = dest / "manifest.json"
        if not self._dry_run:
            manifest.write_text(
                json.dumps({
                    "kind": "skill", "name": parsed["name"], "display": name,
                    "repo_url": repo_url, "subpath": subpath or "",
                    "installed_at": datetime.datetime.now().isoformat(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        skill = Skill(
            name=parsed["name"],
            description=parsed["description"],
            command=parsed["command"],
            when_to_use=parsed.get("when_to_use") or None,
            allowed_tools=tuple(parsed.get("allowed_tools") or ()),
        )
        return InstalledSkill(
            skill=skill, manifest_path=manifest,
            repo_url=repo_url, skill_name=name,
        )

    @staticmethod
    def _parse_skill_dir(skill_dir: Path, fallback_name: str) -> dict[str, Any]:
        """读取 SKILL.md / skill.json，解析出 Skill 字段。"""
        jfile = skill_dir / "skill.json"
        if jfile.is_file():
            d = json.loads(jfile.read_text(encoding="utf-8"))
            return {
                "name": d.get("name", fallback_name),
                "description": d.get("description", ""),
                "command": d.get("command") or d.get("prompt") or d.get("description", ""),
                "when_to_use": d.get("when_to_use", ""),
                "allowed_tools": d.get("allowed_tools", []),
            }
        mfile = skill_dir / "SKILL.md"
        if mfile.is_file():
            text = mfile.read_text(encoding="utf-8", errors="replace")
            front, body = _split_frontmatter(text)
            return {
                "name": front.get("name", fallback_name),
                "description": front.get("description", "") or (body.strip()[:200] or ""),
                "command": front.get("command") or body.strip(),
                "when_to_use": front.get("when_to_use", ""),
                "allowed_tools": front.get("allowed_tools", []),
            }
        # 兜底：用目录名 + README
        rd = skill_dir / "README.md"
        desc = rd.read_text(encoding="utf-8", errors="replace")[:200] if rd.is_file() else ""
        return {
            "name": fallback_name, "description": desc or f"Installed skill {fallback_name}",
            "command": f"Use the {fallback_name} skill to {desc}",
            "when_to_use": "", "allowed_tools": [],
        }

    # ── MCP ───────────────────────────────────────────────────
    def install_mcp(
        self,
        repo_url: str,
        subpath: str | None = None,
        server_name: str | None = None,
        launch_command: str | None = None,
    ) -> InstalledMcp:
        """从 GitHub 仓库下载一个 MCP server 并构造 McpTool（stdio 子进程 executor）。

        launch_command 缺省时按 subpath 里找 package.json / pyproject / main.py 推断。
        工具名为 mcp_<server_name>_<tool>，execute 通过 stdio JSON-RPC 转发。
        """
        sname = (server_name or subpath or repo_url.rstrip("/").split("/")[-1]).strip()
        sname = re.sub(r"[^A-Za-z0-9_.\-]", "_", sname)[:40] or "mcp"
        dest = self._root / f"mcp_{sname}"
        self._clone(repo_url, dest)

        base = dest / (subpath or sname)
        base = base if base.is_dir() else dest
        cmd = launch_command or self._infer_mcp_command(base)
        if not cmd:
            self._rollback(dest)
            raise RuntimeError(
                f"无法推断 MCP server 启动命令，请显式传 launch_command（如 'npx {sname}' / 'python main.py'）"
            )

        executor = self._make_stdio_executor(cmd, base)
        tool = McpTool(
            server_name=sname, name=sname,
            description=f"[MCP: {sname}] Installed MCP server (stdio) — {sname}",
            parameters={"type": "object", "properties": {}, "additionalProperties": True},
            executor=executor,
        )
        manifest = dest / "manifest.json"
        if not self._dry_run:
            manifest.write_text(
                json.dumps({
                    "kind": "mcp", "name": tool.name, "display": sname,
                    "repo_url": repo_url, "launch_command": cmd,
                    "installed_at": datetime.datetime.now().isoformat(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        return InstalledMcp(
            mcp_tool=tool, manifest_path=manifest,
            repo_url=repo_url, server_name=sname, launch_command=cmd,
        )

    @staticmethod
    def _infer_mcp_command(base: Path) -> str | None:
        """按常见约定推断启动命令：npx (package.json bin) / uvx / python main.py。"""
        pj = base / "package.json"
        if pj.is_file():
            try:
                pkg = json.loads(pj.read_text(encoding="utf-8"))
                name = pkg.get("name") or "server"
                return f"npx {name}"
            except Exception:
                return None
        main = base / "main.py"
        if main.is_file():
            return f"{_python_exe()} {main.as_posix()}"
        pyproject = base / "pyproject.toml"
        if pyproject.is_file():
            name = base.name
            return f"uvx {name}"
        return None

    def _make_stdio_executor(self, cmd: str, cwd: Path):
        """构造一个 stdio MCP executor：转发 tools/call，带 timeout + 密钥隔离。"""

        def executor(args: dict[str, Any]) -> dict[str, str]:
            import shlex
            import urllib.request as _ur  # noqa: F401  (占位，避免误用网络)
            parts = shlex.split(cmd) if os.name != "nt" else cmd.split()
            try:
                proc = subprocess.Popen(
                    parts, cwd=str(cwd),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, env=_safe_env(),
                    creationflags=0x08000000 if os.name == "nt" else 0,
                )
            except Exception as e:
                return {"output": "", "error": f"MCP server 启动失败: {e}"}
            # 最小 JSON-RPC：tools/call 透传 args；拿不到标准 MCP 库时降级为原样回显
            try:
                req = json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "call", "arguments": args},
                }).encode() + b"\n"
                out, err = proc.communicate(req, timeout=30)
                text = (out.decode(errors="replace") or "").strip()
                if not text:
                    return {"output": f"(MCP server 无输出) args={json.dumps(args)[:300]}",
                            "error": None}
                return {"output": text[:4000], "error": (err.decode(errors="replace")[:500] or None)}
            except subprocess.TimeoutExpired:
                proc.kill()
                return {"output": "", "error": "MCP server 30s 超时无响应"}
            finally:
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
        return executor

    # ── 持久化 / 恢复 ─────────────────────────────────────────
    def list_installed(self) -> dict[str, list[dict[str, Any]]]:
        """扫描 ~/.coder_extensions/*/manifest.json，重建可注册实例。

        返回 {"skills": [Skill...], "mcp": [McpTool...]}（跨会话恢复用）。
        """
        skills: list[Any] = []
        mcp: list[Any] = []
        if not self._root.exists():
            return {"skills": skills, "mcp": mcp}
        for manifest in self._root.glob("*/manifest.json"):
            try:
                d = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("kind") == "skill":
                sk = Skill(
                    name=d["display"] or d.get("name"),
                    description=d.get("description", ""),
                    command=d.get("command", ""),
                    when_to_use=d.get("when_to_use") or None,
                    allowed_tools=tuple(d.get("allowed_tools", ())),
                )
                skills.append(sk)
            elif d.get("kind") == "mcp":
                tool = McpTool(
                    server_name=d.get("display", "mcp"),
                    name=d.get("display", "mcp"),
                    description=d.get("description", ""),
                    parameters={"type": "object", "properties": {}},
                    executor=None,  # 跨会话恢复时 executor 需重新构造；此处占位
                )
                mcp.append(tool)
        return {"skills": skills, "mcp": mcp}


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """极简 YAML front-matter 解析（key: value / key: [a, b]），不依赖 pyyaml。"""
    front: dict[str, Any] = {}
    body = text
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() in ("---", "..."):
                body = "\n".join(lines[i + 1:])
                for ln in lines[1:i]:
                    if ":" in ln:
                        k, _, v = ln.partition(":")
                        v = v.strip()
                        if v.startswith("[") and v.endswith("]"):
                            front[k.strip()] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
                        else:
                            front[k.strip()] = v.strip("\"'")
                break
    return front, body


def _python_exe() -> str:
    import sys
    return sys.executable or "python"
