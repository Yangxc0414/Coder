# 项目反思与改进计划

> 日期：2026-08-28
> 对比对象：OneCode（参考架构）+ 开源 Coding Agent 项目
> 目的：找出差距，明确改进方向，完善面试表述

---

## 一、当前实现 vs OneCode 架构对比

### 1.1 核心循环

| 维度 | 我们的实现 | OneCode | 差距 |
|---|---|---|---|
| **Loop 结构** | 同步 while 循环 | async/await + 事件驱动 | 我们的更简单，适合面试展示 |
| **流式输出** | ❌ 无 | ✅ content_delta 实时渲染 | OneCode 可实时显示模型输出 |
| **状态管理** | AgentState dataclass | RuntimeState + MessageStore | OneCode 有持久化 session |
| **事件钩子** | ✅ 已实现（5种事件） | HookRegistry (9种事件) | 我们实现了核心事件 |

**OneCode HookEvent 列表：**
```python
PRE_TOOL_USE         # 工具执行前
POST_TOOL_USE        # 工具执行后
TOOL_ERROR           # 工具执行错误
USER_PROMPT_SUBMIT   # 用户提交提示
ASSISTANT_MESSAGE_COMPLETED  # 助手消息完成
TURN_STOPPED         # 回合结束
TASK_CREATED         # 任务创建
TASK_COMPLETED       # 任务完成
PRE_COMPACT          # 压缩前
POST_COMPACT         # 压缩后
COMPACT_FAILED       # 压缩失败
```

**改进建议**：当前不需要完整 Hooks 系统，但可以在关键节点加简单回调：
```python
# agent.py 中可加的钩子
def _on_tool_start(self, tool_name: str, args: dict): ...
def _on_tool_end(self, tool_name: str, result: ToolResult): ...
def _on_step_end(self, step: int, state: AgentState): ...
```

### 1.2 上下文管理

| 维度 | 我们的实现 | OneCode | 差距 |
|---|---|---|---|
| **压缩策略** | 三层裁剪（固定） | 多级压缩（micro/full/reactive） | OneCode 更智能 |
| **Token 估算** | tiktoken | 自研 estimate_messages_tokens | 我们更准确 |
| **Session Memory** | ❌ 无 | SessionMemoryStore + 自动提取 | OneCode 有长期记忆 |
| **Reactive Compact** | ❌ 无 | API error 时自动压缩 | OneCode 更健壮 |

**OneCode Compaction 策略：**
- `MICRO_COMPACT`：替换旧 tool result 为占位符
- `FULL_COMPACT`：整段替换为摘要
- `REACTIVE_COMPACT`：API 报错时触发

**改进建议**：当前三层裁剪已够用，但可以加一个简单的"tool result 截断"策略。

### 1.3 工具系统

| 维度 | 我们的实现 | OneCode | 差距 |
|---|---|---|---|
| **工具协议** | Tool ABC | ToolDescriptor + 分类 | OneCode 有 read_only 标记 |
| **工具数量** | 5个 | 15+（含subagent/skill/MCP） | OneCode 更丰富 |
| **权限系统** | PolicyGate 静态 | PermissionPolicy + SessionStore | OneCode 可持久化授权 |
| **冲突检测** | ❌ 无 | FileStateCache + conflicts | OneCode 防止并发修改 |

**OneCode PLAN_MODE 白名单：**
```python
PLAN_MODE_ALLOWED_TOOLS = {
    "read_file", "grep", "glob", "bash",
    "ask_user_question", "enter_plan_mode",
    "exit_plan_mode", "agent",
}
```

**改进建议**：我们的 Plan 模式已经通过 PolicyGate 实现，可以借鉴 OneCode 的白名单方式让代码更清晰。

### 1.4 错误恢复

| 维度 | 我们的实现 | OneCode | 差距 |
|---|---|---|---|
| **恢复策略** | 6种差异化 | ModelRetryRunner + 多种决策 | OneCode 更细粒度 |
| **流式重试** | ❌ 无 | 流式中断后可恢复 | OneCode 体验更好 |
| **错误日志** | TraceRecorder | ErrorLogRecorder + JsonlErrorLogSink | OneCode 更结构化 |

**OneCode RetryDecision：**
```python
@dataclass
class RetryDecision:
    should_retry: bool
    delay_seconds: float
    max_retries: int
    error_type: str
```

**改进建议**：当前 Recovery 已覆盖主要场景，可以加一个简单的"指数退避"配置。

### 1.5 Plan / Task 系统

| 维度 | 我们的实现 | OneCode | 差距 |
|---|---|---|---|
| **Plan 持久化** | ❌ 无 | PlanStore (Markdown文件) | OneCode 可保存/恢复计划 |
| **Task 管理** | ❌ 无 | TaskStore (JSON文件) | OneCode 可拆解子任务 |
| **Subagent** | ❌ 无 | SubagentRunner | OneCode 可并行处理 |
| **Skill** | ❌ 无 | LoaderSkillCatalogProvider | OneCode 可扩展技能 |

**OneCode Plan 文件结构：**
```
<workspace>/.onecode/plans/<slug>.md
```

**改进建议**：这是 P1/P2 可以加入的功能，让 Agent 可以"规划-执行-验证"分离。

### 1.6 Session / Resume

| 维度 | 我们的实现 | OneCode | 差距 |
|---|---|---|---|
| **Session 持久化** | ❌ 无 | MessageStore (transcript) | OneCode 可恢复中断 |
| **Resume 命令** | ❌ 无 | `/resume <session_id>` | OneCode 用户体验好 |

**改进建议**：当前不需要，但可以作为加分项。

---

## 二、与 mini-swe-agent / SWE-agent 对比

### 2.1 mini-swe-agent（最值得学习）

| 维度 | 我们的实现 | mini-swe-agent | 差距 |
|---|---|---|---|
| **Agent 行数** | 273行 | ~190行 | 我们更多功能 |
| **执行方式** | subprocess.run 每步独立 | subprocess.run 每步独立 | ✅ 一致 |
| **历史管理** | ContextManager 裁剪 | 线性追加（无裁剪） | 我们更先进 |
| **工具** | 5个专用工具 | 仅 bash | 我们更丰富 |
| **评估** | Verifier 类 | SWE-bench 外部 | 各有侧重 |

**mini-swe-agent 核心设计（值得学习）：**
```python
# 极简的 run() 方法
def run(self, task=""):
    self.messages = [system_msg, user_msg]
    while True:
        try:
            self.step()
            self.n_consecutive_format_errors = 0
        except FormatError as e:
            self.n_consecutive_format_errors += 1
            if self.n_consecutive_format_errors >= 3:
                self.messages.append(exit_msg)
            else:
                self.messages.extend(e.messages)
        finally:
            self.save(output_path)
        if self.messages[-1].get("role") == "exit":
            break
```

**改进建议**：我们可以借鉴 mini-swe-agent 的 `save()` 每步保存轨迹，增强鲁棒性。

### 2.2 SWE-agent

| 维度 | 我们的实现 | SWE-agent | 差距 |
|---|---|---|---|
| **Agent 架构** | 单 Agent | RetryAgent + DefaultAgent | SWE 有重试循环 |
| **环境隔离** | 无 | Docker/Singularity | SWE 更安全 |
| **轨迹格式** | JSONL | .traj (包含完整 history) | SWE 更详细 |
| **展示机制** | ❌ 无 | Demo JSON (few-shot) | SWE 可给模型示范 |

**SWE-agent 展示机制（值得学习）：**
```python
# 可以给模型看历史成功的执行示例
demonstration_template = """
Here is an example of how to solve a similar problem:
{demonstration}
"""
```

**改进建议**：可以加一个简单的"示范"功能，在 System Prompt 中注入历史成功案例。

---

## 三、与 Aider / Cline 对比

### 3.1 Aider

| 维度 | 我们的实现 | Aider | 差距 |
|---|---|---|---|
| **编辑策略** | write_file 覆盖 | search/replace + diff | Aider 更精确 |
| **Repo Map** | ❌ 无 | tree_sitter + 缓存 | Aider 更智能 |
| **Git 集成** | ❌ 无 | 自动 commit + diff | Aider 更安全 |
| **多文件编辑** | ❌ 无 | 同时编辑多个文件 | Aider 更强大 |

**Aider Repo Map（值得学习）：**
```
repo
├── modules
│   ├── auth.py (lines 1-50)
│   └── db.py (lines 1-80)
├── classes
│   └── User (lines 10-30)
└── functions
    ├── authenticate (auth.py:15)
    └── get_user (db.py:25)
```

**改进建议**：可以加一个简单的"代码地图"功能，扫描文件生成函数/类列表。

### 3.2 Cline

| 维度 | 我们的实现 | Cline | 差距 |
|---|---|---|---|
| **权限确认** | 静默拦截 | 弹出确认框 | Cline UX 更好 |
| **Human-in-the-loop** | ❌ 无 | ask_user_question 工具 | Cline 可交互 |
| **MCP 支持** | ❌ 无 | 完整 MCP 协议 | Cline 可扩展 |
| **Subagent** | ❌ 无 | general subagent | Cline 可并行 |

**改进建议**：可以加一个 `ask_user_question` 工具，让 Agent 在不确定时询问用户。

---

## 四、当前项目的独特优势

### 4.1 相比所有参考项目的优势

| 优势 | 说明 | 面试价值 |
|---|---|---|
| **零框架依赖** | 纯手写，无 LangChain/AutoGen 等 | 证明底层理解 |
| **Policy Gate 三级策略** | 代码层安全，非 Prompt 约束 | 工程安全意识 |
| **Context/State/Memory 三分离** | 清晰的概念边界 | 系统设计能力 |
| **Verifier 驱动完成** | 不信任模型自说自话 | 严谨的工程思维 |
| **Recovery 策略矩阵** | 6种错误类型差异化处理 | 鲁棒性设计 |
| **4种执行模式** | goal/plan/dry-run/full | 灵活性与安全性 |
| **交互式 CLI** | Rich + prompt_toolkit | 用户体验意识 |

### 4.2 相比 OneCode 的独特之处

OneCode 是一个**完整的产品级框架**，有：
- MCP 协议支持
- Skill 系统
- Subagent 并行
- 流式输出
- Session 恢复
- 复杂的权限确认 UI

我们的项目是一个**精简的教学级实现**，有：
- 更清晰的代码结构（适合面试展示）
- 更少但更核心的功能（避免过度工程）
- 明确的"模型负责决策，程序负责约束"理念
- 所有核心逻辑手写，无黑盒依赖

---

## 五、后续改进建议（按优先级）

### P0：必须完成（提交物要求）

| # | 改进项 | 工作量 | 优先级 |
|---|---|---|---|
| 1 | Git 公开仓库 + push | 10分钟 | 🔴 高 |
| 2 | README.txt（1000字） | 30分钟 | 🔴 高 |
| 3 | 演示视频（≤2分钟） | 1小时 | 🔴 高 |
| 4 | 打包提交物 | 10分钟 | 🔴 高 |

### P1：增强功能（锦上添花）

| # | 改进项 | 借鉴来源 | 工作量 |
|---|---|---|---|
| 5 | Hook 系统（PRE/POST_TOOL） | OneCode hooks | 半天 |
| 6 | Plan 文件持久化 | OneCode PlanStore | 半天 |
| 7 | ask_user_question 工具 | Cline | 2小时 |
| 8 | Session 恢复（/resume） | OneCode resume | 半天 |
| 9 | 代码地图（简单版） | Aider Repo Map | 2小时 |

### P2：高级功能（有时间再做）

| # | 改进项 | 借鉴来源 | 工作量 |
|---|---|---|---|
| 10 | ✅ Subagent 并行 | 已完成 | — |
| 11 | 流式输出 | OneCode streaming | 后续 |
| 12 | 长期记忆自动提取 | OneCode LongTermMemory | 后续 |
| 13 | ✅ MCP 支持 | 已完成（简化版） | — |

---

## 六、面试技术故事（最终版）

> "我的 Agent 从一个最小 ReAct Loop 出发——参考 mini-swe-agent 的 190 行极简设计，但我加入了工程化所需的增强。
>
> **核心循环**就是 20 行代码的 while 循环：LLM 调用 → 解析 tool_calls → Policy Gate 检查 → 工具执行 → 结果回传。所有逻辑都是手写的，没有引入任何 Agent 框架。
>
> **第一个工程问题**是 API 兼容性——Agnes 网关要求 tool_calls 带 type/index 字段，我在 _query_llm() 中做了消息格式规范化。
>
> **第二个问题是安全**。我实现了 Policy Gate——三级策略（ALLOW/LOG/DENY）+ 危险命令黑名单 + 路径穿越检查（os.path.commonpath）。我没有把安全寄托在 Prompt 上，而是做了代码层的硬拦截。
>
> **第三个问题是长对话**。我实现了三层裁剪的 ContextManager：保留 System Prompt + 最近 6 轮完整对话 + 旧消息摘要。用 tiktoken 做 token 估算，实测 45 条消息压缩到 3 条，token 减半。
>
> **第四个问题是'模型失忆'**。我区分了 Context/State/Memory 三个概念，State 和 Memory 会动态注入到 System Prompt 中，让模型即使在被压缩上下文后也能感知当前进度。我还实现了 Loop Detection，检测连续 3 次操作同一文件时注入引导。
>
> **第五个问题是'模型说完成≠真完成'**。我实现了独立的 Verifier——跑 pytest、检查语法、看 git diff。只有全部通过才算任务完成。
>
> **第六个问题是错误恢复**。我实现了 6 种错误类型的差异化 Recovery 策略：格式错误注入修正提示，API 连接错误指数退避，策略拒绝注入安全建议。
>
> **最后我加了 4 种执行模式**（goal/plan/dry-run/full）和一个交互式 CLI，用 Rich + prompt_toolkit 实现，支持实时状态查看。
>
> **扩展系统**方面，我设计了 Hook 系统（5 个生命周期事件）、MCP 工具层（git/system/debug 内置工具）、Skill 系统（5 个领域专用 skill）和 Subagent 系统（4 种专业子 Agent），全部通过 ExtensionTool ABC 协议统一接入 Registry。整个系统共 121 个测试覆盖核心功能和扩展模块。"
>
> 核心思想始终是：**模型负责决策，程序负责约束**。

---

## 七、对标 OneCode 的具体差距总结

| 类别 | OneCode 有但我们没有的 | 对我们的影响 |
|---|---|---|
| **架构** | Hook 系统、异步 Loop、流式输出 | 我们的同步设计更简单，适合面试展示 |
| **功能** | ✅ Subagent/MCP/Skill 已实现 | Subagent、MCP、Skill、Task 管理 | 扩展系统已补齐 |
| **持久化** | Session Resume、Long-term Memory | 可以增加，但不是必须的 |
| **UI** | 复杂的 TTY 界面、权限确认弹窗 | 我们的简单 REPL 更清晰 |
| **压缩** | Reactive Compact、Micro-Compact | 我们的三层裁剪已够用 |

**结论**：我们的项目在**核心 Agent 能力**上与 OneCode 持平，在**简洁性和可解释性**上更优，非常适合面试展示。OneCode 的优势在于产品化功能（MCP、Subagent、流式 UI），这些不是推免项目的必须的。
