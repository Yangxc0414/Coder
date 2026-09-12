"""回归测试：全增强 Agent 在真实多 bug 任务上的端到端效果。

确定性 LLM（ScriptedBugFixLLM）驱动，模型变量被控制，
断言代码行为正确（ground truth）+ pytest 全绿——
为"7 项增强效果很好"提供可复现的端到端证据。
"""

import json
import pytest


def test_full_enhanced_e2e_bug_fix(tmp_path):
    from tests.e2e_full_enhanced_bugfix import run_full_enhanced_bug_fix
    res = run_full_enhanced_bug_fix()
    assert res["code_behavior_correct"], f"代码行为错误: {res}"
    assert res["pytest_passed"], f"pytest 未全绿: {res['final_answer']}"
    assert res["success"], res


def test_full_enhanced_e2e_bug_fix_json_serializable(tmp_path):
    """结果可被 JSON 序列化（供报告/审计落盘）。"""
    from tests.e2e_full_enhanced_bugfix import run_full_enhanced_bug_fix
    res = run_full_enhanced_bug_fix()
    assert json.dumps(res, ensure_ascii=False)  # 不抛异常即可
