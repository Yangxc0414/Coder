# P3 扩展系统设计 — Hooks / MCP / Skills / Subagents

> 设计文档 — 2026-08-27

## 1. 概述

在 P0-P2 完成 ReAct 循环、上下文管理、状态追踪、恢复策略的基础上，本阶段引入**扩展系统**，使 Agent 具备可插拔能力：

| 组件 | 文件位置 | 功能 |
|------|---------|------|
| Hook 系统 | `coder_agent/hooks.py` | 生命周期事件回调（PRE/POST_TOOL_USE, TURN_STOPPED 等） |
| MCP 工具 | `coder_agent/extensions/mcp/` | 外部工具协议，支持 git/system/debug 内置工具 |
| Skill 系统 | `coder_agent/extensions/skills/` | 领域专用 prompt+工具组合，5 个内置 skill |
| Subagent | `coder_agent/extensions/subagents/` | 子 Agent 并行执行，4 种内置子 Agent 类型 |

设计参考：OneCode 的 `services/mcp/`, `services/skills/`, `services/subagents/`, `services/hooks/`

---

## 2. Hook 系统

### 2.1 设计目标

在 Agent 生命周期的关键节点提供事件回调机制，支持：
- **日志记录**：自动记录每次工具调用
- **审计追踪**：记录所有操作到 trace
- **安全拦截**：在工具执行前检查敏感操作
- **自定义逻辑**：用户可注册自己的回调

### 2.2 事件类型

| 事件常量 | 触发时机 | 数据字段 |
|---------|---------|---------|
| `AGENT_STARTED` | `agent.run()` 开始 | `task`: 任务描述 |
| `PRE_TOOL_USE` | 工具执行前 | `tool_name`, `args`, `call_id` |
| `POST_TOOL_USE` | 工具执行后 | `tool_name`, `success`, `error`, `output_len` |
| `TURN_STOPPED` | 每轮 LLM 响应处理完 | `step`: 步数 |
| `AGENT_ENDED` | `agent.run()` 结束 | `steps`, `final_state` |

### 2.3 API

```python
from coder_agent.hooks import HookRegistry, PRE_TOOL_USE, install_logging_hooks, install_trace_hooks

# 创建注册表
hooks = HookRegistry()

# 安装默认钩子
install_logging_hooks(hooks)   # 自动记录日志
install_trace_hooks(hooks, trace)  # 自动记录到 trace

# 注册自定义钩子
def my_hook(event):
    if event.data.get("tool_name") == "write_file":
        print(f"Writing to {event.data['args'].get('path')}")

hooks.register("PRE_TOOL_USE", my_hook)

# 触发事件
hooks.fire(PRE_TOOL_USE.with_data(tool_name="read_file", args={"path": "foo.py"}))
```

### 2.4 在 Agent 中的集成

```python
# agent.py
from .hooks import HookRegistry, install_logging_hooks, install_trace_hooks, ...

def __init__(self, ...):
    ...
    self.hooks = HookRegistry()
    install_logging_hooks(self.hooks)
    install_trace_hooks(self.hooks, self.trace)

def run(self, task: str) -> str:
    self.hooks.fire(AGENT_STARTED.with_data(task=task))
    # ... 主循环 ...
    self.hooks.fire(AGENT_ENDED.with_data(steps=self._n_steps, ...))

def _execute_tool_call(self, parsed):
    # 执行前
    self.hooks.fire(PRE_TOOL_USE.with_data(...))
    # ... 执行工具 ...
    # 执行后
    self.hooks.fire(POST_TOOL_USE.with_data(...))
```

---

## 3. MCP (Model Context Protocol) 工具

### 3.1 设计目标

MCP 允许外部服务器暴露工具给 Agent。本实现是一个**简化版**，支持：
- 内置 Mock 工具（git、system info、debug）
- 可扩展为完整 MCP 客户端（JSON-RPC over stdio）

### 3.2 内置 MCP 工具

| 工具名 | 功能 |
|-------|------|
| `mcp_git_git_log` | 查看最近 git 提交历史 |
| `mcp_git_git_status` | 查看 git 状态（修改/暂存/未跟踪文件） |
| `mcp_git_git_diff` | 查看未提交的变更 diff |
| `mcp_system_system_info` | 获取系统/Python 环境信息 |
| `mcp_debug_read_trace` | 读取 Agent 执行 trace 日志 |

### 3.3 API

```python
from coder_agent.extensions.mcp import create_builtin_mcp_tools
from coder_agent.extensions.base import McpTool

# 创建内置 MCP 工具
tools = create_builtin_mcp_tools(workspace="/path/to/project")

# 添加自定义 MCP 工具
def my_executor(args):
    return {"output": "result", "error": ""}

custom_tool = McpTool(
    server_name="my_server",
    name="my_tool",
    description="My custom tool",
    parameters={"type": "object", "properties": {}},
    executor=my_executor,
)
tools.append(custom_tool)
```

### 3.4 在 Agent 中的集成

```python
# agent.py
from coder_agent.extensions.mcp import create_builtin_mcp_tools

def __init__(self, ...):
    ...
    # 将 MCP 工具注册到 registry
    mcp_tools = create_builtin_mcp_tools(self.workspace)
    for tool in mcp_tools:
        self.registry.register(tool)
```

---

## 4. Skill 系统

### 4.1 设计目标

Skill 是**可复用的 prompt + 工具组合**。当模型选择调用某个 Skill 时，系统会：
1. 执行 Skill 的 command 模板（填充 target/context 参数）
2. 将结果作为 LLM 指令注入对话
3. 可选：限制该 Skill 可用的工具子集

### 4.2 内置 Skills

| Skill 名称 | 功能 | 可用工具 | 触发条件 |
|-----------|------|---------|---------|
| `code_review` | 代码审查（bug/风格/性能/安全） | read_file, search_text | 写完代码后 |
| `test_writer` | 编写 pytest 测试 | read_file, write_file, run_command | 实现新函数后 |
| `refactor` | 重构代码 | read_file, write_file, run_command | 代码难以维护时 |
| `docstring` | 添加 Google 风格文档 | read_file, write_file | 标记完成前 |
| `security_audit` | 安全漏洞扫描 | read_file, search_text | 部署前/处理敏感数据 |

### 4.3 API

```python
from coder_agent.extensions.skills import get_builtin_skills
from coder_agent.extensions.base import Skill

# 获取所有内置 Skills
skills = get_builtin_skills()

# 创建自定义 Skill
my_skill = Skill(
    name="my_review",
    description="Review code for style issues",
    command="Check {target} for PEP8 compliance and suggest fixes.",
    when_to_use="When checking code quality",
    allowed_tools=("read_file",),
    context="inline",  # 或 "fork"（隔离上下文）
)
```

### 4.4 在 Agent 中的集成

Skills 作为特殊工具注册到 registry。当模型调用 `skill_code_review(target="foo.py")` 时：
1. Skill.execute() 生成指令文本
2. 结果作为 tool message 返回给 LLM
3. LLM 根据指令执行后续操作

---

## 5. Subagent 系统

### 5.1 设计目标

Subagent 是**拥有独立上下文和工具集的微型 Agent**。主 Agent 可以：
- 创建 subagent 处理特定子任务
- 限制 subagent 的工具权限（如只读）
- 同步等待 subagent 完成并获取结果

### 5.2 内置 Subagent 类型

| 类型 | 工具集 | 最大步数 | 特点 |
|-----|-------|---------|------|
| `researcher` | read_file, list_files, search_text | 15 | 只读，用于代码分析 |
| `test_specialist` | read_file, write_file, run_command | 20 | 专注测试编写 |
| `security_scanner` | read_file, search_text | 10 | 只读，安全审计 |
| `documenter` | read_file, write_file | 15 | 专注文档编写 |

### 5.3 API

```python
from coder_agent.extensions.subagents import get_builtin_subagents
from coder_agent.extensions.base import SubagentDefinition, SubagentRequest, SubagentRunner

# 获取内置定义
definitions = get_builtin_subagents()

# 创建自定义 Subagent
researcher = SubagentDefinition(
    name="code_analyzer",
    system_prompt="Analyze code and report issues.",
    when_to_use="Before making changes to understand codebase",
    tools=("read_file", "search_text"),
    disallowed_tools=("write_file", "run_command"),
    max_steps=10,
    read_only=True,
)

# 运行 subagent
runner = SubagentRunner(parent_agent=agent)
runner.register(researcher)
result = runner.run(SubagentRequest(
    prompt="Analyze the main.py file and report architecture",
    subagent_type="code_analyzer",
))
print(result.final_output)
print(f"Steps used: {result.steps_used}")
```

### 5.4 在 Agent 中的集成

```python
# agent.py
from coder_agent.extensions.subagents import get_builtin_subagents, SubagentRunner

def __init__(self, ...):
    ...
    self.subagent_runner = SubagentRunner(parent_agent=self)
    for defn in get_builtin_subagents():
        self.subagent_runner.register(defn)
```

---

## 6. 文件结构

```
coder_agent/
├── hooks.py                     # 新增：Hook 系统
├── agent.py                     # 修改：集成 hooks
└── extensions/
    ├── __init__.py              # 新增：扩展包导出
    ├── base.py                  # 新增：基础类（Skill, McpTool, Subagent*）
    ├── mcp/
    │   ├── __init__.py          # 新增
    │   ├── client.py            # 保留（简化版 MCP 客户端）
    │   └── builtins.py          # 新增：内置 MCP 工具（git/system/debug）
    ├── skills/
    │   ├── __init__.py          # 新增
    │   └── builtin.py           # 修改：修正 import 路径
    └── subagents/
        ├── __init__.py          # 新增
        └── builtin.py           # 修改：修正 import 路径

tests/
├── test_hooks.py                # 新增：15 个 hook 测试
└── test_extensions.py           # 新增：30 个扩展测试
```

---

## 7. 测试覆盖

| 模块 | 测试数 | 覆盖范围 |
|-----|-------|---------|
| `test_hooks.py` | 15 | HookEvent, HookRegistry, 安装钩子 |
| `test_extensions.py` | 30 | Skill, MCP 工具, Subagent 定义, 内置组件 |
| **合计** | **45** | 新增测试 |
| 原有测试 | 76 | 全部通过（无回归） |
| **总计** | **121** | |

---

## 8. 后续改进方向

1. **完整 MCP 协议**：支持 JSON-RPC over stdio/SSE，连接真实 MCP 服务器
2. **流式输出**：使用 SSE 流式返回 LLM 响应
3. **Plan 持久化**：将计划保存为 Markdown 文件
4. **Session 恢复**：从 trace 恢复中断的会话
5. **并发 Subagent**：使用 asyncio 并行运行多个 subagent
6. **Skill 文件加载**：从 YAML/JSON 文件动态加载 skill 定义
