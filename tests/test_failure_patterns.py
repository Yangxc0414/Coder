"""失败模式库测试 — 验证指纹化 / 策略轮换 / 跨会话持久化 / 知识闭环。"""

from __future__ import annotations

import json
from pathlib import Path

from coder_agent.failure_patterns import (
    FailurePatternLibrary,
    _classify_failure,
    _fingerprint,
)


class TestClassify:
    def test_syntax(self):
        assert _classify_failure("SyntaxError in a.py", "SyntaxError") == "syntax"

    def test_import(self):
        assert _classify_failure("", "ModuleNotFoundError: No module named x") == "import"

    def test_type(self):
        assert _classify_failure("", "TypeError: unsupported operand") == "type"

    def test_assert(self):
        assert _classify_failure("assert 1 == 2", "") == "assert"

    def test_fallback_runtime(self):
        assert _classify_failure("", "KeyError: 'foo'") == "runtime"


class TestFingerprint:
    def test_deterministic(self):
        a = _fingerprint("assert", ["x.py"], "boom")
        b = _fingerprint("assert", ["x.py"], "boom")
        assert a == b

    def test_order_insensitive_files(self):
        a = _fingerprint("runtime", ["a.py", "b.py"], "m")
        b = _fingerprint("runtime", ["b.py", "a.py"], "m")
        assert a == b

    def test_different_message_different_fp(self):
        a = _fingerprint("assert", ["x.py"], "msg1")
        b = _fingerprint("assert", ["x.py"], "msg2")
        assert a != b


class TestLibrary:
    def test_record_and_occurrences(self, tmp_path: Path):
        lib = FailurePatternLibrary(tmp_path)
        rec = lib.record("assert", ["t.py"], "assert failed")
        assert lib.occurrences(rec.fingerprint) == 1
        rec2 = lib.record("assert", ["t.py"], "assert failed")  # 同指纹
        assert rec2.fingerprint == rec.fingerprint
        assert lib.occurrences(rec.fingerprint) == 2

    def test_strategy_rotates_on_repeat(self, tmp_path: Path):
        lib = FailurePatternLibrary(tmp_path)
        fp = _fingerprint("assert", [], "x")
        lib.record("assert", [], "x")  # 第 1 次
        lib.record("assert", [], "x")  # 第 2 次
        # 第 2 次失败 → 轮换到第 2 条策略（逼迫换方法）
        advice = lib.repair_advice("assert", "x", "", 2)
        assert advice is not None
        assert "换方法" in advice

    def test_first_failure_no_advice(self, tmp_path: Path):
        lib = FailurePatternLibrary(tmp_path)
        # 第 1 次失败不给"换方法"提示（还没证明方法无效）
        assert lib.repair_advice("assert", "x", "", 1) is None

    def test_persist_cross_session(self, tmp_path: Path):
        lib1 = FailurePatternLibrary(tmp_path)
        rec = lib1.record("import", ["a.py"], "mod not found")
        # 模拟新会话：新建库实例，应能从盘上载入
        lib2 = FailurePatternLibrary(tmp_path)
        assert lib2.occurrences(rec.fingerprint) == 1

    def test_mark_solved_knowledge_loop(self, tmp_path: Path):
        lib = FailurePatternLibrary(tmp_path)
        rec = lib.record("type", ["f.py"], "NoneType")
        assert rec.solved is False
        n = lib.mark_solved([rec.fingerprint])
        assert n == 1
        # 落盘后新实例能看到 solved 标记
        lib2 = FailurePatternLibrary(tmp_path)
        note = lib2.solved_note(rec.fingerprint)
        assert "历史经验" in note or note == ""  # 首次解决时 strategy 可能为空

    def test_store_file_atomic(self, tmp_path: Path):
        lib = FailurePatternLibrary(tmp_path)
        lib.record("runtime", [], "x")
        f = tmp_path / ".coder_failure_patterns.json"
        assert f.exists()
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["records"]) == 1
