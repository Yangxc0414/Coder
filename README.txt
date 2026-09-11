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
2. 三大支柱：
   · 上下文与记忆：三层压缩+单消息占比保护、工具输出溢出转存、
     Context/State/Memory 三分离动态注入、memory 工具（模型可写长期记忆）
   · 工具与编排：17 个工具（5 核心+5 MCP+5 Skill+task 委派+memory）、
     PolicyGate 三级安全（危险命令黑名单/路径逃逸校验/环境变量过滤）、
     parser 五层验证、task 工具委派 4 类子代理（结构化反递归）
   · 评估与恢复：独立 Verifier（pytest/语法/git 三重检查+变更门控）、三道终止闸（步数收尾轮/token 预算/验证门控）、
     8 型差异化错误恢复、JSONL 全链路 trace
3. 会话恢复：每次运行镜像 journal，/resume 断点续跑，链式可恢复。