"""端到端 ReAct 闭环测试脚本。

验证 Agent 能完成以下三类任务：
1. 读取文件并返回内容
2. 创建新文件
3. 修复有 bug 的计算器（完整 bug fix 流程）

使用方法：
    python tests/test_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from coder_agent.agent import Agent
from coder_agent.llm.client import LLMClient
from coder_agent.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool
from coder_agent.tools.registry import ToolRegistry
from coder_agent.tools.shell import RunCommandTool


def make_agent(workspace: Path) -> Agent:
    """构建 Agent 实例"""
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace))
    registry.register(WriteFileTool(workspace))
    registry.register(ListFilesTool(workspace))
    registry.register(RunCommandTool(workspace))

    llm = LLMClient(
        model=os.getenv("MODEL_NAME", "agnes-2.5-flash"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    return Agent(llm_client=llm, registry=registry, workspace=workspace)


# ── 测试 1：读取文件 ────────────────────────────────────────

def test_read_file_task(tmp_path: Path) -> bool:
    """Agent 读取一个文件并返回内容"""
    # 准备测试文件
    (tmp_path / "note.txt").write_text("Hello from coder-agent!")

    agent = make_agent(tmp_path)
    answer = agent.run("请读取 note.txt 文件，告诉我里面写了什么")

    # 验证：answer 中应包含 "Hello"
    ok = "Hello" in answer or "hello" in answer.lower()
    print(f"  answer: {answer[:200]}...")
    print(f"  trace steps: {len(agent._trace)}")
    return ok


# ── 测试 2：创建文件 ────────────────────────────────────────

def test_write_file_task(tmp_path: Path) -> bool:
    """Agent 创建一个新 Python 文件"""
    agent = make_agent(tmp_path)
    answer = agent.run(
        "在 workspace 中创建一个名叫 greeting.py 的文件，"
        "里面定义一个 greet(name) 函数，打印 Hello, {name}!"
    )

    # 验证文件是否创建成功
    created = (tmp_path / "greeting.py").exists()
    if created:
        content = (tmp_path / "greeting.py").read_text()
        has_greet = "greet" in content and "Hello" in content
    else:
        has_greet = False

    print(f"  file created: {created}, has greet: {has_greet}")
    print(f"  answer: {answer[:200]}...")
    print(f"  trace steps: {len(agent._trace)}")
    return created and has_greet


# ── 测试 3：修复 bug（完整流程）────────────────────────────

def test_bug_fix_task(tmp_path: Path) -> bool:
    """Agent 修复 calculator.py 中的除零 bug，并运行测试"""
    # 复制测试任务到 workspace
    src_task = PROJECT_ROOT / "tasks" / "bug_fix"
    if src_task.exists():
        import shutil
        dest = tmp_path / "tasks"
        shutil.copytree(src_task, dest)
    else:
        # 手动创建测试文件
        calc_dir = tmp_path / "tasks" / "bug_fix"
        calc_dir.mkdir(parents=True)
        (calc_dir / "calculator.py").write_text("""
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b):
    if b == 0:
        return None  # BUG: should raise ZeroDivisionError
    return a / b
""")
        (calc_dir / "test_calculator.py").write_text("""
import pytest
from calculator import divide
def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
""")

    agent = make_agent(tmp_path)
    answer = agent.run(
        "修复 tasks/bug_fix/calculator.py 中的 bug："
        "divide 函数在除以零时应该抛出 ZeroDivisionError，而不是返回 None。"
        "修复后运行测试确认通过。"
    )

    # 验证：测试文件是否修复，测试是否通过
    test_file = tmp_path / "tasks" / "bug_fix" / "test_calculator.py"
    calc_file = tmp_path / "tasks" / "bug_fix" / "calculator.py"

    test_passed = False
    if test_file.exists() and calc_file.exists():
        calc_content = calc_file.read_text()
        # 检查是否修复了 bug（不再返回 None）
        has_fix = "raise" in calc_content or "ZeroDivisionError" in calc_content
        if has_fix:
            # 运行测试
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-q"],
                cwd=str(tmp_path / "tasks" / "bug_fix"),
                capture_output=True,
                text=True,
                timeout=30,
            )
            test_passed = result.returncode == 0
            print(f"  pytest output: {result.stdout[-200:]}")

    print(f"  bug fixed: {has_fix if 'has_fix' in dir() else False}")
    print(f"  tests passed: {test_passed}")
    print(f"  answer: {answer[:300]}...")
    print(f"  trace steps: {len(agent._trace)}")
    return test_passed


# ── 主测试入口 ─────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("ReAct 闭环端到端测试")
    print("=" * 60)

    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        print("\n[测试 1] 读取文件")
        results["read_file"] = test_read_file_task(tmp_path)
        print(f"  → {'✅ 通过' if results['read_file'] else '❌ 失败'}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("\n[测试 2] 创建文件")
        results["write_file"] = test_write_file_task(tmp_path)
        print(f"  → {'✅ 通过' if results['write_file'] else '❌ 失败'}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("\n[测试 3] 修复 Bug（完整 ReAct 闭环）")
        results["bug_fix"] = test_bug_fix_task(tmp_path)
        print(f"  → {'✅ 通过' if results['bug_fix'] else '❌ 失败'}")

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"  {name}: {status}")
    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 全部通过！ReAct 闭环验证成功。")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，需要调整 Prompt 或工具设计。")
        sys.exit(1)


if __name__ == "__main__":
    main()
