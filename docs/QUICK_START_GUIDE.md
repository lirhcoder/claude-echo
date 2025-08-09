# Claude Voice Assistant 快速上手指南

## 🚀 5分钟快速开始

### 第一步：安装和启动

#### Windows 用户
```cmd
# 1. 下载项目（假设已有源码）
cd claude-echo

# 2. 一键安装
install.bat

# 3. 启动应用
start_claude_voice.bat
```

#### macOS/Linux 用户
```bash
# 1. 下载项目
cd claude-echo

# 2. 一键安装
./install.sh

# 3. 启动应用
./start_claude_voice.sh
```

### 第二步：验证安装

启动成功后，你应该看到：

```
🚀 启动 Claude Voice Assistant...
⚠️  Alpha测试版本 - Mock模式运行
✅ 配置加载成功
✅ 适配器管理器初始化完成
✅ 事件系统启动成功
📋 输入 'help' 查看可用命令
```

---

## 💡 基础操作教程

### 文本命令模式

在 Alpha 测试阶段，主要通过文本命令进行交互：

#### 1. 文件操作命令

```bash
# 创建文件
> create file hello.py

# 读取文件
> read file hello.py

# 写入内容到文件
> write to hello.py content "print('Hello World')"

# 列出目录文件
> list files in current directory
```

#### 2. 代码生成命令

```bash
# 生成Python函数
> generate function add_numbers in python

# 生成类
> generate class Calculator in python

# 生成React组件
> generate component Button in javascript
```

#### 3. 项目分析命令

```bash
# 分析项目结构
> analyze project structure

# 获取项目信息
> get project info

# 查找文件
> search files with pattern "*.py"
```

#### 4. 系统命令

```bash
# 运行测试
> run tests

# Git状态查询
> check git status

# 格式化代码
> format code in current file
```

---

## 🎯 核心概念理解

### 1. 适配器系统

Claude Voice Assistant 使用适配器模式集成不同的工具：

- **Claude Code适配器**: 处理编程相关任务
- **文件系统适配器**: 管理文件操作
- **语音接口适配器**: 处理语音交互（Alpha版本中禁用）

### 2. 事件驱动架构

所有操作都通过事件系统协调：

```
用户输入 → 意图解析 → 事件发送 → 适配器处理 → 结果返回
```

### 3. Mock模式

Alpha版本运行在Mock模式下：
- 模拟语音识别和合成
- 模拟Claude Code集成
- 使用安全的文件操作

---

## 🧪 测试功能演示

### 演示1：创建简单Python项目

```bash
# 1. 创建项目目录
> create directory my_project

# 2. 创建主文件
> create file my_project/main.py

# 3. 生成函数代码
> generate function calculate_area for rectangle in python

# 4. 写入生成的代码到文件
> write generated code to my_project/main.py

# 5. 读取确认
> read file my_project/main.py
```

### 演示2：项目分析

```bash
# 1. 分析当前项目
> analyze project structure

# 2. 搜索Python文件
> search files with pattern "*.py"

# 3. 获取项目统计信息
> get project statistics
```

### 演示3：Mock语音交互测试

```bash
# 1. 启动语音交互模拟
> enable mock voice mode

# 2. 模拟语音输入
> simulate voice command "创建一个新的Python函数"

# 3. 查看解析结果
> show last intent parsing result
```

---

## 🔧 配置和自定义

### 基础配置文件

编辑 `config/test_config.yaml` 进行自定义：

```yaml
# 启用更详细的日志
logging:
  level: DEBUG
  
# 调整操作超时
performance:
  operation_timeout: 60  # 增加到60秒

# 修改安全设置
security:
  policies:
    alpha_testing: ["low", "medium", "high"]  # 允许更多操作
```

### 测试数据目录

在 `test_projects/` 中创建测试项目：

```
test_projects/
├── python_demo/
├── javascript_demo/
└── mixed_project/
```

---

## ⚠️ 已知限制和注意事项

### Alpha版本限制

1. **语音功能禁用**: 避免音频依赖问题
2. **模拟模式运行**: Claude Code集成使用模拟响应
3. **安全限制**: 只允许安全的文件操作
4. **功能子集**: 仅包含核心功能

### 安全注意事项

- 不要在重要项目目录中进行测试
- 定期备份测试数据
- 注意日志文件可能包含敏感信息

### 性能考虑

- 首次启动较慢（需要初始化）
- 大文件操作可能需要更长时间
- 建议在专用测试环境中运行

---

## 🐛 问题排查

### 常见错误及解决方案

#### 1. 启动失败

```
❌ 配置文件加载失败
```

**解决方案**:
- 检查 `config/test_config.yaml` 文件是否存在
- 验证YAML语法是否正确
- 尝试使用默认配置

#### 2. 命令无响应

```
⏳ 命令执行中...（长时间无响应）
```

**解决方案**:
- 检查 `logs/alpha_test.log` 查看详细错误
- 尝试重新启动应用
- 减少并发操作数量

#### 3. 文件权限错误

```
❌ Permission denied: /path/to/file
```

**解决方案**:
- 确保对测试目录有读写权限
- 在Windows上以管理员身份运行
- 检查文件是否被其他程序占用

### 日志分析

检查关键日志信息：

```bash
# 查看最新的错误日志
tail -n 50 logs/alpha_test.log

# 搜索特定错误
grep "ERROR" logs/alpha_test.log

# 查看性能统计
grep "Performance" logs/alpha_test.log
```

---

## 📊 测试任务清单

### 基础功能验证

- [ ] 成功启动应用
- [ ] 执行基本文件操作
- [ ] 测试配置加载
- [ ] 验证日志输出

### 适配器测试

- [ ] Claude Code适配器响应
- [ ] 文件系统操作正常
- [ ] 错误处理机制

### 集成测试

- [ ] 端到端命令执行
- [ ] 多步骤操作流程
- [ ] 异常恢复测试

---

## 🎓 进阶使用

### 自定义适配器

为了测试适配器扩展性，可以尝试：

```python
# 在 src/adapters/ 中创建自定义适配器
class MyCustomAdapter(BaseAdapter):
    @property
    def adapter_id(self):
        return "my_custom"
    
    async def execute_command(self, command, parameters):
        # 自定义逻辑
        pass
```

### 事件监听

监听系统事件：

```python
# 添加事件监听器
event_system.subscribe(
    ["adapter.command_executed"],
    my_event_handler
)
```

### 配置热重载

修改配置并热重载：

```bash
# 修改配置文件后
> reload config

# 验证新配置
> show current config
```

---

## 📞 获取帮助

### 内置帮助

```bash
# 查看所有可用命令
> help

# 查看特定命令帮助
> help create file

# 查看适配器状态
> status
```

### 资源链接

- 📖 [完整文档](../docs/README.md)
- 🏗️ [架构设计](../docs/architecture.md)
- 🧪 [测试指南](../testing/alpha_test_checklist.md)
- 🔧 [开发指南](../docs/development.md)

### 技术支持

- **GitHub Issues**: 报告bugs和功能请求
- **讨论区**: 技术交流和经验分享
- **文档Wiki**: 详细的技术文档

---

**恭喜！您已经掌握了 Claude Voice Assistant 的基础使用。开始探索和测试吧！** 🎉