"""Web 服务端点测试 — 工作区切换与目录浏览（TestClient，不起真服务）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from coder_agent.ui.web import server as web_server


@pytest.fixture()
def client():
    return TestClient(web_server.app)


class TestWorkspaceSwitch:
    def test_get_workspace(self, client, tmp_path: Path):
        web_server._state["manager"] = web_server.RunManager(tmp_path)
        r = client.get("/api/workspace")
        assert r.json()["workspace"] == str(tmp_path.resolve())

    def test_switch_workspace(self, client, tmp_path: Path):
        target = tmp_path / "project"
        target.mkdir()
        r = client.post("/api/workspace", json={"path": str(target)})
        assert r.json() == {"ok": True, "workspace": str(target.resolve())}
        assert web_server.get_manager().workspace == target.resolve()

    def test_switch_requires_existing_dir(self, client):
        r = client.post("/api/workspace", json={"path": "D:/no/such/dir/xyz"})
        assert r.status_code == 404

    def test_switch_blocked_while_running(self, client, tmp_path: Path):
        web_server._state["manager"] = web_server.RunManager(tmp_path)
        m = web_server.get_manager()
        # 模拟运行中：往会话表里放一个活跃线程
        import threading
        from coder_agent.ui.web.runner import _RunSession
        block = threading.Event()
        s = _RunSession("run_test")
        s.thread = threading.Thread(target=block.wait, daemon=True)
        s.thread.start()
        m._sessions["run_test"] = s
        try:
            r = client.post("/api/workspace", json={"path": str(tmp_path)})
            assert r.status_code == 409
        finally:
            block.set()


class TestBrowseEndpoint:
    def test_browse_root_lists_drives(self, client):
        r = client.get("/api/fs/browse")
        data = r.json()
        assert data["drives"] is True
        assert any("C:\\" in d for d in data["dirs"])

    def test_browse_dir_lists_subdirs(self, client, tmp_path: Path):
        (tmp_path / "sub1").mkdir()
        (tmp_path / "sub2").mkdir()
        (tmp_path / "file.txt").write_text("x")
        r = client.get("/api/fs/browse", params={"path": str(tmp_path)})
        data = r.json()
        assert str(tmp_path / "sub1") in data["dirs"]
        assert str(tmp_path / "sub2") in data["dirs"]
        # 文件不混入目录导航
        assert all("file.txt" not in d for d in data["dirs"])

    def test_browse_missing_dir_404(self, client):
        assert client.get("/api/fs/browse", params={"path": "D:/no/such/xyz"}).status_code == 404


class TestFilesEndpoints:
    def test_files_lists_one_level(self, client, tmp_path: Path):
        web_server._state["manager"] = web_server.RunManager(tmp_path)
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / ".hidden").write_text("h")
        r = client.get("/api/files", params={"path": str(tmp_path)})
        files = r.json()["files"]
        assert "a.py" in files and "sub" not in files and ".hidden" not in files

    def test_file_read_with_containment(self, client, tmp_path: Path):
        web_server._state["manager"] = web_server.RunManager(tmp_path)
        (tmp_path / "a.py").write_text("print('hi')")
        ok = client.get("/api/file", params={"path": str(tmp_path / "a.py")})
        assert "print" in ok.json()["content"]
        bad = client.get("/api/file", params={"path": str(tmp_path.parent / "outside.txt")})
        assert bad.status_code == 403

    def test_file_missing_404(self, client, tmp_path: Path):
        web_server._state["manager"] = web_server.RunManager(tmp_path)
        assert client.get("/api/file", params={"path": str(tmp_path / "no.txt")}).status_code == 404


class TestSessionDetail:
    def test_session_detail_grouped_by_turn(self, client, tmp_path: Path):
        """/api/session 按 turn 分组回放，含工具调用参数与结果。"""
        import json
        from pathlib import Path as P

        session_dir = P.home() / ".coder_sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        f = session_dir / "session_test_turns.jsonl"
        lines = [
            {"type": "meta", "created": 1.0, "version": 1},
            {"type": "message", "message": {"role": "user", "content": "写一个 hello.py"}},
            {"type": "message", "message": {"role": "assistant", "content": "我先看看目录",
                                            "tool_calls": [{"id": "c1", "type": "function",
                                                            "function": {"name": "read_file",
                                                                         "arguments": '{"path": "hello.py"}'}}]}},
            {"type": "message", "message": {"role": "tool", "tool_call_id": "c1",
                                            "content": "(error) 文件不存在"}},
            {"type": "message", "message": {"role": "assistant", "content": "文件不存在，我来创建它",
                                            "tool_calls": [{"id": "c2", "type": "function",
                                                            "function": {"name": "write_file",
                                                                         "arguments": '{"path": "hello.py", "content": "print(1)"}'}}]}},
            {"type": "message", "message": {"role": "tool", "tool_call_id": "c2", "content": "已写入"}},
            {"type": "message", "message": {"role": "assistant", "content": "✅ 完成：hello.py 已创建"}},
        ]
        f.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n",
                     encoding="utf-8")
        try:
            r = client.get("/api/session", params={"file": str(f)})
            assert r.status_code == 200
            data = r.json()
            assert len(data["turns"]) == 1
            turn = data["turns"][0]
            assert turn["user"] == "写一个 hello.py"
            tools = [s["tool"] for s in turn["steps"] if s["tool"] != "_think"]
            assert tools == ["read_file", "write_file"]
            read_step = next(s for s in turn["steps"] if s["tool"] == "read_file")
            write_step = next(s for s in turn["steps"] if s["tool"] == "write_file")
            # read_file 失败标记、write_file 成功标记
            assert read_step["success"] is False
            assert read_step["result"].startswith("(error)")
            assert write_step["success"] is True
            assert write_step["result"] == "已写入"
            assert turn["answer"] == "✅ 完成：hello.py 已创建"
        finally:
            f.unlink(missing_ok=True)

    def test_session_detail_rejects_escape(self, client):
        r = client.get("/api/session", params={"file": "C:/Windows/system32/drivers/etc/hosts"})
        assert r.status_code == 403

    def test_session_detail_missing_file_404(self, client):
        r = client.get("/api/session", params={"file": str(Path.home() / ".coder_sessions" / "nope.jsonl")})
        assert r.status_code == 404


class TestConfigEndpoint:
    @pytest.fixture(autouse=True)
    def _isolate_config_file(self, tmp_path, monkeypatch):
        """把配置读写隔离到临时文件，避免污染用户真实 ~/.coder_config.json。"""
        from coder_agent.ui.web import runner
        fake = tmp_path / "coder_config.json"
        monkeypatch.setattr(runner, "CONFIG_FILE", fake)

    def test_get_config_defaults(self, client, tmp_path: Path):
        web_server._state["manager"] = web_server.RunManager(tmp_path)
        r = client.get("/api/config")
        data = r.json()
        assert "model" in data and "base_url" in data
        assert "api_key_set" in data

    def test_set_config_persists(self, client, tmp_path: Path):
        web_server._state["manager"] = web_server.RunManager(tmp_path)
        r = client.post("/api/config", json={"model": "test-model", "base_url": "http://localhost:9/v1"})
        assert r.status_code == 200
        m = web_server.get_manager()
        assert m.model == "test-model"
        assert m.base_url == "http://localhost:9/v1"
        # 从配置文件可恢复
        from coder_agent.ui.web.runner import load_config
        cfg = load_config()
        assert cfg.get("model") == "test-model"
        # 隔离 fixture 已把配置指向临时文件，无需清理真实配置
