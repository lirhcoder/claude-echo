# 使用 Claude Code Development Agents 指南

## 🚀 快速开始

### 1. 环境准备

确保您在 Claude Code 中打开了项目目录：
```bash
cd C:\development\claude-echo
```

### 2. 激活开发 Agents 模式

在 Claude Code 中，您可以通过以下方式激活开发 Agents：

```
@coordinator 开始开发语音助手项目的第一阶段
```

## 🎯 使用场景和命令

### 场景1: 开始架构设计

**用户命令**:
```
我需要开始设计语音助手的系统架构，重点是分层架构和插件化系统
```

**预期执行流程**:
1. **Development Coordinator** 分析需求
2. 分配给 **Architecture Agent** 主要开发
3. **Testing Agent** 准备架构测试
4. **Documentation Agent** 更新架构文档

**命令示例**:
```
@coordinator 请 Architecture Agent 设计分层架构，包括：
- 用户交互层
- 智能中枢层  
- 适配器层
- 执行层
```

### 场景2: 开发语音处理功能

**用户命令**:
```
实现高质量的语音识别和合成系统，支持中英文
```

**预期执行流程**:
1. **Coordinator** 分析语音需求
2. **Speech Agent** 开发语音功能
3. **Architecture Agent** 提供架构支持
4. **Testing Agent** 进行语音质量测试

**命令示例**:
```
@coordinator 请 Speech Agent 开发：
- Whisper 语音识别集成
- pyttsx3 语音合成
- 中英文意图解析
- 语音质量优化
```

### 场景3: 构建核心 Agents 系统

**用户命令**:
```
开发7个核心智能代理，实现AI协作架构
```

**预期执行流程**:
1. **Coordinator** 制定 Agent 开发计划
2. **Core-Agents Agent** 逐一开发7个代理
3. **Architecture Agent** 提供框架支持
4. **Testing Agent** 测试 Agent 协作

**命令示例**:
```
@coordinator 请 Core-Agents Agent 开发：
- Coordinator (协调中心)
- Task-Planner (任务规划)
- Presence-Monitor (状态监控)
- Auto-Worker (自主执行)
- Security-Guardian (安全监护)
- Handover-Manager (交接管理)  
- Session-Manager (会话管理)
```

### 场景4: 集成 Claude Code

**用户命令**:
```
实现与 Claude Code 的深度集成，支持语音控制编程
```

**预期执行流程**:
1. **Coordinator** 分析集成需求
2. **Adapters Agent** 开发 Claude Code 适配器
3. **Speech Agent** 优化编程语音识别
4. **Integration Agent** 测试端到端集成

**命令示例**:
```
@coordinator 请开发 Claude Code 集成：
- Adapters Agent 创建 Claude Code 适配器
- Speech Agent 优化编程词汇识别
- Integration Agent 验证语音编程工作流
```

## 🔧 高级使用技巧

### 1. 指定特定 Agent

如果您知道需要哪个专业 Agent，可以直接指定：

```
@architecture 请设计适配器管理系统的架构
```

```
@speech 优化语音识别的准确率，重点处理中文识别
```

```
@testing 为语音处理模块创建完整的测试套件
```

### 2. 多 Agent 协作任务

对于复杂任务，可以明确指定多个 Agent 协作：

```
@coordinator 请协调以下工作：
- Architecture Agent 设计插件架构
- Adapters Agent 实现适配器接口
- Testing Agent 创建适配器测试
- Documentation Agent 编写使用文档
```

### 3. 阶段性开发

按照开发阶段组织工作：

```
@coordinator 开始第一阶段开发：
目标：完成基础架构和语音处理
时间：2-3周
主要 Agents：Architecture, Speech
```

### 4. 质量检查和优化

```
@coordinator 进行全面质量检查：
- Testing Agent 运行所有测试
- Architecture Agent 检查代码结构
- Documentation Agent 更新文档
- Integration Agent 验证集成
```

## 📋 Agent 专业领域对照表

| Agent | 专业领域 | 适用场景 |
|-------|----------|----------|
| **Architecture Agent** | 系统架构、框架设计 | 设计架构、创建基础类、技术选型 |
| **Speech Agent** | 语音处理、NLP | 语音识别、合成、意图解析 |
| **Core-Agents Agent** | AI代理、异步协作 | 开发智能代理、Agent通信 |
| **Adapters Agent** | 系统集成、API封装 | 外部系统集成、适配器开发 |
| **Testing Agent** | 测试、质量保证 | 编写测试、性能基准、质量检查 |
| **Documentation Agent** | 技术文档、API文档 | 文档编写、用户手册、规范制定 |
| **Integration Agent** | 系统集成、端到端测试 | 集成验证、部署测试 |
| **DevOps Agent** | CI/CD、运维、监控 | 自动化部署、监控设置 |

## 🔄 典型开发工作流

### 完整功能开发流程

```
1. 需求分析
   @coordinator 分析需求: [具体需求描述]

2. 架构设计
   @architecture 设计 [功能名称] 的架构

3. 核心开发
   @[专业agent] 实现 [具体功能]

4. 测试验证  
   @testing 为 [功能] 创建测试套件

5. 集成验证
   @integration 验证 [功能] 的集成

6. 文档更新
   @documentation 更新 [功能] 的文档

7. 最终检查
   @coordinator 进行质量检查和验收
```

### 快速原型开发流程

```
1. 快速原型
   @coordinator 创建 [功能] 的MVP原型

2. 核心验证
   @[专业agent] 实现核心功能

3. 基础测试
   @testing 创建基础测试

4. 迭代改进
   @coordinator 基于测试结果优化
```

## 🛠️ 实用命令模板

### 新功能开发
```
@coordinator 开发新功能: [功能名称]
需求描述: [详细需求]
优先级: [高/中/低]
预期时间: [时间估计]
相关组件: [相关模块]
```

### 问题修复
```
@coordinator 修复问题: [问题描述]
错误现象: [具体表现]
影响范围: [影响的功能]  
紧急程度: [紧急/一般]
```

### 性能优化
```
@coordinator 优化性能: [优化目标]
当前性能: [具体指标]
目标性能: [期望指标]
优化范围: [具体模块]
```

### 重构改进
```
@coordinator 重构代码: [重构范围]
重构原因: [原因说明]
预期改进: [期望效果]
风险评估: [潜在风险]
```

## 📊 进度跟踪和质量监控

### 查看项目进度
```
@coordinator 汇报项目整体进度
```

### 检查特定模块状态
```
@coordinator 检查 [模块名称] 的开发状态
```

### 质量评估
```
@coordinator 进行全面质量评估，包括：
- 代码质量检查
- 测试覆盖率分析  
- 性能基准测试
- 文档完整性检查
```

## ⚠️ 注意事项

### 1. 任务描述要清晰
- 明确说明具体需求和期望结果
- 提供必要的上下文信息
- 指明优先级和时间要求

### 2. 合理分配任务复杂度
- 复杂任务让 Coordinator 分解
- 专业任务直接指派给专业 Agent
- 跨领域任务需要多 Agent 协作

### 3. 及时跟踪进度
- 定期检查任务完成状态
- 主动解决依赖阻塞问题
- 及时调整计划和资源分配

### 4. 保持质量标准
- 每个阶段都要进行质量检查
- 重要功能需要充分测试
- 保持代码和文档同步更新

通过这种 Sub Agents 开发模式，您可以高效地完成复杂的语音助手项目开发，每个 Agent 在其专业领域发挥最大价值，协同工作确保项目成功！