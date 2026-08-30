# Tests

测试目录涵盖了 `coder-agent` 项目的核心功能验证，包括端到端 ReAct 闭环、Verifier 门控机制、REPL UI 交互、上下文长度管理、工具层安全性与沙箱、以及 my-pi-agent 采纳（task/memoiry 工具）等关键改进。

## 快速开始

```bash
# 运行全部测试
pytest tests/

# 运行单个文件
pytest tests/test_tools.py -v

# 运行单个用例
pytest tests/test_verifier_trace.py::TestVerifier::test_pytest_passes -v
```

## 测试分类

### 端到端与 Agent 流程

| 测试文件 | 描述 |
|---------|------|
| `test_e2e.py` | 真实环境端到端测试：Agent 读取文件、创建文件、修复除零 bug 计算器 |
| `test_agent_e2e.py` | Mock LLM 端到端测试：parser、tokenizer、Agent 循环（读/写/格式化恢复/步骤上限/Verifier 集成/Trace 输出/State 追踪/Memory 记录） |
| `test_resume.py` | 会话中断后恢复的 E2E：Agent A 中断 → Journal 持久化 → Agent B 恢复并继续（链式恢复） |
| `test_journal.py` | SessionJournal 写入与 replay：meta、消息往返、Unicode、append 模式、malformed arguments 容错 |
| `test_hooks.py` | Hook 系统：HookEvent/注册/注销/错误隔离/built-in 安装器（logging、trace） |

### Verifier 与上下文管理

| 测试文件 | 描述 |
|---------|------|
| `test_verifier_baseline.py` | Verifier 基础能力：无测试跳过、语法检查、pytest 通过/失败、git diff 跳过、计时 |
| `test_verifier_gating.py` | 回归守卫：50-step 失控事件——Verifier 不应因预存在失败而阻止正确完成的任务 |
| `test_verifier_trace.py` | TraceRecorder：步骤记录、成功/失败统计、JSONL 输出、指标计算 |
| `test_context_guard.py` | 上下文卫生：584K-token 事件回归——巨型消息截断、tool output 绝对上限、空状态显示 |
| `test_overflow.py` | 溢出卸载：超大文件 read/search 时自动 offload 到 `.coder_overflow/` 并保留指针 |
| `test_token_budget.py` | Token 预算：上下文窗口配额控制与预算告警 |

### 工具层与安全性

| 测试文件 | 描述 |
|---------|------|
| `test_tools.py` | Tool 协议/注册/PolicyGate 安全检查/路径防护（白名单工具、未注册工具拒绝） |
| `test_parser_schema.py` | Layer-5 参数 Schema 校验：必填字段、类型强制转换（str→int/bool→bool）、未知字段静默剥离、错误信息质量 |

### UI 与 Shell

| 测试文件 | 描述 |
|---------|------|
| `test_repl_ui.py` | REPL UI 第三轮：`/model` 切换、ASSISTANT_TEXT 思考内容显示 |
| `test_web_client.py` / `test_web_server.py` | Web 客户端与服务器通信测试 |
| `test_shell.py` | Shell 工具单元测试 |

### my-pi-agent 采纳（深度对比改进）

| 测试文件 | 描述 |
|---------|------|
| `test_my_pi_adoption.py` | Task 工具（委托即工具）、反递归（子代理剥离 task/memory）、Memory 工具（持久化+注入 system prompt）、撕裂行容忍、PolicyGate 全模式集成、工作区隔离（cross-prefix/cross-drive/CWD escape） |
| `test_extensions.py` | 扩展系统：Skill、MCP 工具、SubagentDefinition、Built-in Subagents |
| `test_framework_wiring.py` | 框架接线：Agent ↔ Verifier ↔ ContextManager ↔ Memory 的依赖注入 |
| `test_recovery.py` | 崩溃恢复：异常恢复路径与状态重建 |
| `test_length_recovery.py` | 长度超限恢复：超出 context 窗口后的降级策略 |
| `test_stuck_detection.py` | 卡住检测：重复 tool call 检测与自动退避 |
| `test_realrun_improvements.py` | 真实运行改进回归：综合验证各项修复的实际效果 |

## 设计原则

- **隔离性**：所有测试使用 `tmp_path`（pytest fixture）保证工作区互不干扰。
- **Mock 优先**：`test_agent_e2e.py` 使用 `MockLLM` 确保确定性，不依赖外部 LLM API。
- **回归守卫**：重大事件（584K-token、50-step 失控、工作区前缀绕过）均有专门测试文件记录。
- **分层验证**：从工具层（`test_tools.py`）→ 解析层（`test_parser_schema.py`）→ Agent 层（`test_agent_e2e.py`）→ 端到端（`test_e2e.py`），逐层覆盖。
