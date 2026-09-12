"""tests for the on-demand Skill/MCP installer (agent 自主下载扩展)。

覆盖：URL 白名单、Skill 解析、MCP 子进程 executor、dry-run 不落盘、
跨会话恢复（list_installed）、agent 注册 install 工具、policy 放行。
网络/git 用 mock，全部离线可跑。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coder_agent.extensions.installer import (
    ExtensionInstaller,
    _validate_repo_url,
    _split_frontmatter,
)


# ── URL 白名单 ─────────────────────────────────────────────
@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo",
    "https://github.com/owner/repo.git",
    "https://gitlab.com/a/b",
    "git@github.com:owner/repo",
])
def test_url_whitelist_allows(url):
    assert _validate_repo_url(url) == url


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "https://evil.com/x",
    "http://github.com/a/b",
    "javascript:alert(1)",
    "",
    "https://github.com/owner",  # 缺 repo 段
])
def test_url_whitelist_rejects(url):
    with pytest.raises(ValueError):
        _validate_repo_url(url)


# ── SKILL.md front-matter 解析 ─────────────────────────────
def test_split_frontmatter_basic():
    text = "---\nname: xlsx\ndescription: analyze xlsx\nwhen_to_use: data work\n---\nBODY PROMPT"
    front, body = _split_frontmatter(text)
    assert front["name"] == "xlsx"
    assert front["description"] == "analyze xlsx"
    assert front["when_to_use"] == "data work"
    assert "BODY PROMPT" in body


def test_split_frontmatter_list():
    text = "---\nallowed_tools: [read_file, run_command]\n---\nx"
    front, _ = _split_frontmatter(text)
    assert front["allowed_tools"] == ["read_file", "run_command"]


# ── install_skill（mock 掉 git clone，写临时目录）──────────
def test_install_skill_parses_skill_md(tmp_path, monkeypatch):
    """构造一个含 SKILL.md 的临时目录，mock _clone 把内容塞进去。"""
    root = tmp_path / "ext"
    installer = ExtensionInstaller(extensions_dir=root)

    def fake_clone(url, target):
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(
            "---\nname: my_skill\ndescription: a test skill\nwhen_to_use: testing\n"
            "allowed_tools: [read_file]\n---\nDo the thing on {target}.\n",
            encoding="utf-8")

    monkeypatch.setattr(installer, "_clone", fake_clone)

    inst = installer.install_skill("https://github.com/o/r", skill_name="my_skill")
    assert inst.skill.name == "skill_my_skill"
    assert "a test skill" in inst.skill.description
    assert "Do the thing" in inst.skill._command
    # 可注册进 registry
    from coder_agent.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register(inst.skill)
    assert "skill_my_skill" in reg.list_names()


def test_install_skill_dry_run_no_manifest(tmp_path, monkeypatch):
    root = tmp_path / "ext"
    installer = ExtensionInstaller(dry_run=True, extensions_dir=root)

    def fake_clone(u, t):
        t.mkdir(parents=True, exist_ok=True)
        (t / "SKILL.md").write_text(
            "---\nname: s\ndescription: d\n---\nc", encoding="utf-8")

    monkeypatch.setattr(installer, "_clone", fake_clone)
    inst = installer.install_skill("https://github.com/o/r", skill_name="s")
    assert not inst.manifest_path.exists(), "dry-run 不应写 manifest"


# ── install_mcp（mock 推断 + 子进程 executor）──────────────
def test_install_mcp_infers_npx_command(tmp_path, monkeypatch):
    root = tmp_path / "ext"
    installer = ExtensionInstaller(extensions_dir=root)

    def fake_clone(u, t):
        t.mkdir(parents=True, exist_ok=True)
        (t / "package.json").write_text('{"name":"my-mcp"}', encoding="utf-8")

    monkeypatch.setattr(installer, "_clone", fake_clone)
    inst = installer.install_mcp("https://github.com/o/r", server_name="my_mcp")
    assert "npx" in inst.launch_command
    # executor 存在且可调用（server 不存在 → 返回 error 而非抛）
    res = inst.mcp_tool.executor({"foo": "bar"})
    assert res.get("error") is not None or res.get("output") is not None


def test_mcp_executor_rejects_missing_server(tmp_path):
    installer = ExtensionInstaller(extensions_dir=tmp_path / "ext")
    executor = installer._make_stdio_executor("npx does-not-exist-mcp", tmp_path)
    res = executor({"x": 1})
    assert res.get("error"), "缺失的 MCP server 应返回 error 而非崩溃"


# ── 跨会话恢复 ─────────────────────────────────────────────
def test_list_installed_recovers(tmp_path):
    root = tmp_path / "ext"
    sk_dir = root / "s1"
    sk_dir.mkdir(parents=True)
    (sk_dir / "manifest.json").write_text(json.dumps({
        "kind": "skill", "name": "skill_s1", "display": "s1",
        "description": "d", "command": "c", "allowed_tools": ["read_file"],
    }), encoding="utf-8")
    mc_dir = root / "mcp_m1"
    mc_dir.mkdir(parents=True)
    (mc_dir / "manifest.json").write_text(json.dumps({
        "kind": "mcp", "name": "mcp_m1_m1", "display": "m1", "description": "d",
    }), encoding="utf-8")
    got = ExtensionInstaller(extensions_dir=root).list_installed()
    assert [s.name for s in got["skills"]] == ["skill_s1"]
    assert [m.name for m in got["mcp"]] == ["mcp_m1_m1"]


# ── agent 注册 + policy 放行 ───────────────────────────────
def test_agent_registers_install_tools(tmp_path):
    from coder_agent.agent import Agent
    from coder_agent.mode import AgentMode
    from coder_agent.llm.client import LLMResponse
    from coder_agent.tools.registry import ToolRegistry
    from coder_agent.tools.filesystem import ReadFileTool, WriteFileTool
    from coder_agent.tools.search import SearchTextTool
    from coder_agent.tools.shell import RunCommandTool

    class DummyLLM:
        model = "mock"

        def chat(self, *a, **k):
            return LLMResponse(content="done", tool_calls=None,
                               finish_reason="stop", usage=None)

    reg = ToolRegistry()
    reg.register(ReadFileTool(tmp_path))
    reg.register(WriteFileTool(tmp_path))
    reg.register(SearchTextTool(tmp_path))
    reg.register(RunCommandTool(tmp_path))

    agent = Agent(llm_client=DummyLLM(), registry=reg, workspace=tmp_path,
                  mode=AgentMode.FULL)
    assert "install_skill" in agent.registry.list_names()
    assert "install_mcp" in agent.registry.list_names()


def test_policy_allows_install_tools():
    from coder_agent.policy import PolicyGate
    from coder_agent.mode import AgentMode
    g = PolicyGate()
    for t in ("install_skill", "install_mcp"):
        assert g.check(t, {}, mode=AgentMode.GOAL).approved
        assert g.check(t, {}, mode=AgentMode.FULL).approved
        # PLAN 只读 → 拦截
        assert not g.check(t, {}, mode=AgentMode.PLAN).approved


def test_runner_extensions_info_includes_source():
    from coder_agent.ui.web.runner import RunManager
    import tempfile
    m = RunManager(Path(tempfile.gettempdir()))
    info = m.extensions_info()
    assert all("source" in s for s in info["skills"])
    for tools in info["mcp"].values():
        assert all("source" in t for t in tools)
