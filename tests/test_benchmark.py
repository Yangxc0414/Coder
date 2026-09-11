"""基准对照测试 — 把"全增强版优于纯 ReAct 基线"变成可测量的证据。

这是 coder_agent 相对市面开源 agent（mini-swe-agent / OneCode /
smolagents 的纯 ReAct 范式）方法优势的量化对照实验：同一任务、
同一确定性 LLM 替身、同一工具环境下，跑两种框架配置，差异全部来自
框架机制（控制了模型变量）。

测试项：
1. 基准跑通且判定通过（full 不劣于 baseline 且至少一项严格更优）
2. 两个 agent 的步数/失败数一致（证明 LLM 行为序列被控制为相同，
   差异确实来自框架而非模型）
3. 判定逻辑自洽（no_worse 与 strictly_better 蕴含 passed）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.benchmark_agent import run_benchmark


class TestBenchmark:
    def test_benchmark_passes(self, tmp_path: Path):
        res = run_benchmark(tmp_path)
        assert res["passed"], (
            f"基准未通过：no_worse={res['no_worse']} "
            f"strictly_better={res['strictly_better']} summary={res['summary']}"
        )

    def test_controlled_llm_behavior(self, tmp_path: Path):
        """两个 agent 的步数与失败数必须一致（LLM 行为被控制为相同）。"""
        res = run_benchmark(tmp_path)
        b, f = res["baseline"], res["full"]
        assert b.steps == f.steps, "步数不一致——LLM 行为未被控制"
        assert b.tool_failures == f.tool_failures, "失败数不一致"

    def test_verdict_logic_consistent(self, tmp_path: Path):
        """判定逻辑自洽：passed 必须由 no_worse 且 strictly_better 推出。"""
        res = run_benchmark(tmp_path)
        expected = res["no_worse"] and res["strictly_better"]
        assert res["passed"] is expected

    def test_full_has_more_intervention(self, tmp_path: Path):
        """全增强版应体现框架干预（失败提示注入 / 上下文分级保留）。

        这是与纯 ReAct 基线（市面 agent 的范式）的本质差异：
        基线对试错放任自流，全增强版主动注入"换方法"建议或
        省略低价值重复消息。
        """
        res = run_benchmark(tmp_path)
        b, f = res["baseline"], res["full"]
        # 至少一项干预差异（hints 或 上下文消息数）
        assert (f.fallback_hints > b.fallback_hints
                or f.peak_messages < b.peak_messages), (
            "全增强版未体现出框架干预——对比失效")
