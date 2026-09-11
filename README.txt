Coder-Agent

一、仓库地址
https://gitee.com/doraemon0414/coder-agent

二、如何运行
1. 环境：Python 3.8+，安装依赖：pip install -e ".[dev,ui]"
2. 配置：在工作区复制 .env.example 为 .env，填入
   OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME（任意 OpenAI 兼容网关）
3. 使用方式：
   a. 单任务：python -m coder_agent.main "你的编程任务" --mode full
   b. 交互界面：python -m coder_agent.ui.cli.repl
      支持斜杠命令：/run /goal /mode /model /status /tools等
   c. Web 客户端（可选）：pip install -e ".[web]" 后
      python -m coder_agent.ui.web.server，浏览器打开 http://127.0.0.1:8000
      页面可"打开文件夹"切换任意工作区，聊天式界面实时显示

三、特色功能
1. 零框架：ReAct 循环、上下文管理、输出解析、错误恢复全部自研，
   核心依赖仅 openai/tiktoken/python-dotenv，无任何 agent 框架
2. 四大支柱：
   · 上下文与记忆：模型感知三层压缩（单消息 60% 占比保护）、工具输出
     溢出转存、消息分级保留（关键指令如验证失败/策略拦截优先存活，
     低价值重复读取自动省略）、分层记忆（短期自动记录+模型可写长期
     记忆+按重要性自动整理淘汰）
   · 工具与安全：29 个工具（5 核心+5 MCP+17 Skill+task 委派+memory）、
     PolicyGate 三级安全（危险命令黑名单/路径逃逸校验/环境变量过滤）、
     parser 五层验证、工具失败自动降级路由（同类失败连败 2 次注入换方法建议）、
     自适应工具 schema 路由（按任务阶段裁剪 29→~12 工具定义省 token）、
     task 工具委派 4 类子代理（结构化反递归）
   · 规划与编排：Plan-Execute-Verify 三段式——LLM 把复杂任务分解为
     原子子目标依赖图，无依赖子目标线程池并行派发子代理执行，每层
     验证门控；成功计划沉淀为模板库（plan_templates.json，跨会话
     复用免 LLM 分解调用）；全部失败自动回退 ReAct（零降智）
   · 评估与恢复：独立 Verifier（pytest/语法/git 三重检查+基线对比
     +变更门控）、失败模式库（失败指纹化+按类别轮换修复策略+跨会话
     知识沉淀，.coder_failure_patterns.json）、三道终止闸（步数收尾轮/
     token 预算/验证门控）、8 型差异化错误恢复、JSONL 全链路 trace
3. 会话恢复：每次运行镜像 journal，/resume 断点续跑，链式可恢复
4. 演示回放：/record 录制运行 trace，/replay 离线回放（不依赖 API，
   用于演示环境兜底）；/health 后端侧 API 连通性检查
5. 测试：python -m pytest tests/（390 例，无需 API key）
6. 基准对照：python -m tests.benchmark_agent
   coder_agent 全增强版 vs 纯 ReAct 基线（市面开源 agent 范式）的量化对照——
   同一任务/同一确定性 LLM/同一工具环境，差异全部来自框架机制
   （控制了模型变量）：步数/失败数持平，框架主动干预（失败换方法提示
   注入）仅在增强版生效。把"方法优于基线"从定性表述变成可测量的证据

四、设计参考
实现了 ReAct / 上下文预算 / 子代理委派 / 失败恢复等通用范式；参考了
mini-swe-agent（FormatError 恢复）、OneCode（输出截断加倍重试）、
smolagents（工具验证）、my-pi-agent（记忆工具化/结构化反递归）的
实现思路，全部代码为本仓库独立实现与测试。
API key 仅通过 .env 提供，已列入 .gitignore，绝不出现在仓库中。
