"""回归测试：Windows 路径会话删除/加载的越界判定（_within_dir）。

修复背景：旧实现用 ``Path.resolve().relative_to`` 判越界，Windows 上
resolve() 可能返回 UNC（``//?/C:/...``）或大小写/盘符不一致导致
误判 403，会话删除/加载按钮"点了没反应"。
"""

import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coder_agent.ui.web.server import _within_dir, _delete_session


def test_within_dir_normal_windows_path():
    sd = Path.home() / ".coder_sessions"
    good = str(sd / "session_x.jsonl")
    assert _within_dir(Path(good), sd) is True


def test_within_dir_posix_form_on_windows():
    # /c/Users/... POSIX 形式应被判为 C:\... 盘符路径，在目录内
    if os.name != "nt":
        import pytest
        pytest.skip("Windows-only normalization")
    sd = Path.home() / ".coder_sessions"
    posix = "/c/Users/20691/.coder_sessions/session_x.jsonl"
    # 用真实的 home 构造
    home_norm = Path.home()
    posix_form = "/" + str(home_norm)[0] + str(home_norm)[2:].replace("\\", "/") + "/.coder_sessions/session_x.jsonl"
    assert _within_dir(Path(posix_form), sd) is True


def test_within_dir_unc_rejected():
    sd = Path.home() / ".coder_sessions"
    good = str(sd / "session_x.jsonl")
    assert _within_dir(Path("//?/" + good), sd) is False


def test_within_dir_outside_rejected():
    sd = Path.home() / ".coder_sessions"
    outside = str(Path.home() / "别的.jsonl")
    assert _within_dir(Path(outside), sd) is False


def test_delete_session_requires_existing_file(tmp_path, monkeypatch):
    """_delete_session 对不存在的文件返回 404，对越界路径返回 403。"""
    import coder_agent.ui.web.server as srv

    session_dir = tmp_path / ".coder_sessions"
    session_dir.mkdir()
    target = session_dir / "session_y.jsonl"
    target.write_text("{}", encoding="utf-8")

    # 让 Path.home() 指向 tmp_path
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    # 越界路径 → 403
    from fastapi import HTTPException
    outside = str(tmp_path / "evil.jsonl")
    try:
        srv._delete_session(outside)
        assert False, "应抛 403"
    except HTTPException as e:
        assert e.status_code == 403

    # 不存在 → 404
    try:
        srv._delete_session(str(session_dir / "nope.jsonl"))
        assert False, "应抛 404"
    except HTTPException as e:
        assert e.status_code == 404

    # 存在 → 删除成功
    res = srv._delete_session(str(target))
    assert res["ok"] is True
    assert not target.exists()
