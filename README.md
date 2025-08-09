# Claude Voice Assistant

基于 Claude Code Agents 模式的智能语音助手开发项目

## 🎯 项目目标

构建一个配合 Claude Code 工作的语音助手，未来扩展为通用电脑语音助手

## 🏗️ 系统架构

```
用户交互层 (语音/文本)
    ↓
智能中枢层 (7个核心 Agents)
├── coordinator (协调中心)
├── task-planner (任务规划) 
├── presence-monitor (状态监控)
├── auto-worker (自主执行)
├── security-guardian (安全监护)
├── handover-manager (交接管理)
└── session-manager (会话管理)
    ↓
适配器层 (插件化扩展)
├── Claude Code 适配器
├── 系统操作适配器  
├── IDE 适配器
└── 办公软件适配器
    ↓
执行层 (工具API调用)
```

## 🛠️ 技术栈

- **语音处理**: OpenAI Whisper + Azure Speech
- **自然语言处理**: spaCy + 自定义规则引擎
- **异步框架**: asyncio + aiofiles
- **系统集成**: win32api + pyautogui + psutil
- **配置管理**: YAML + Pydantic
- **CLI界面**: Typer + Rich

## 📋 开发阶段

| 阶段 | 时间 | 内容 | 状态 |
|------|------|------|------|
| Phase 1 | 2-3周 | 核心架构 + Agents 系统 | 🔄 进行中 |
| Phase 2 | 1-2周 | Claude Code 集成 | ⏳ 计划中 |
| Phase 3 | 2-3周 | 通用系统适配器 | ⏳ 计划中 |
| Phase 4 | 2周 | 智能学习系统 | ⏳ 计划中 |
| Phase 5 | 1-2周 | 插件系统 + 多用户 | ⏳ 计划中 |
| Phase 6 | 1周 | 测试 + 部署 | ⏳ 计划中 |

## 📁 项目结构

```
claude-voice-assistant/
├── docs/                  # 开发文档
├── src/                   # 源代码
│   ├── core/             # 核心系统
│   ├── agents/           # 智能代理
│   ├── adapters/         # 适配器
│   ├── speech/           # 语音处理
│   └── utils/            # 工具库
├── tests/                # 测试代码
├── config/              # 配置文件
└── backup/              # 文档备份
```

## 🚀 快速开始

1. **环境准备**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置设置**
   ```bash
   cp config/default.yaml config/user.yaml
   # 编辑 config/user.yaml
   ```

3. **运行系统**
   ```bash
   python src/main.py
   ```

## 📚 文档索引

- [系统架构设计](docs/architecture.md)
- [Agents 系统详解](docs/agents.md)
- [适配器开发指南](docs/adapters.md)
- [Claude Code 集成](docs/claude-code.md)
- [API 参考文档](docs/api.md)

## 🔐 安全特性

- 操作权限控制
- 危险操作拦截
- 用户确认机制
- 审计日志记录

## 🤝 贡献指南

请参考 [开发指南](docs/development.md)

## 📄 许可证

MIT License