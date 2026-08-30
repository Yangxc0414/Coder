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
        # 模拟运行中
        import threading
        block = threading.Event()
        m._thread = threading.Thread(target=block.wait, daemon=True)
        m._thread.start()
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
