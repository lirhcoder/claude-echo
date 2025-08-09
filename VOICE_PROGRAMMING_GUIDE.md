# Claude 语音编程使用指南

## 🚀 快速开始

### 1. 基础设置
```python
from src.adapters import ClaudeCodeAdapter

# 创建适配器
adapter = ClaudeCodeAdapter({
    'working_directory': './my_project',
    'code_generation': {
        'coding_style': 'standard',
        'include_comments': True
    }
})

# 初始化
await adapter.initialize()
```

### 2. 运行测试验证
```bash
python test_adapter_basic.py
```

## 🎤 语音命令示例

### 文件操作
```
"创建文件 hello.py"
"读取文件 main.py"
"列出当前目录的所有文件"
```

### 代码生成
```
"创建一个函数叫 add_numbers，接收两个参数"
"生成一个 Python 类叫 Calculator"
"写一个 React 组件叫 Button，有 text 和 onClick 属性"
```

### 代码修改
```
"重构这个函数，让它更简洁"
"给这个类添加注释"
"格式化当前文件的代码"
```

### 项目操作
```
"分析这个项目的结构"
"运行所有测试"
"提交代码，消息是'添加新功能'"
"构建项目"
```

## 💻 程序化使用

### 直接命令执行
```python
# 生成函数
result = await adapter.execute_command("generate_function", {
    "name": "calculate_area",
    "description": "Calculate rectangle area",
    "language": "python",
    "parameters": ["width", "height"]
})

if result.success:
    print("Generated code:")
    print(result.data['generated_code'])
```

### 语音意图处理
```python
from src.adapters.enhanced_intent_parser import EnhancedIntentParser
from src.speech.types import RecognitionResult

# 创建意图解析器
parser = EnhancedIntentParser()

# 模拟语音识别结果
recognition = RecognitionResult(
    text="Create a function called hello_world",
    confidence=0.95,
    processing_time=0.1
)

# 解析意图
intent = await parser.parse_code_intent(recognition)

# 执行意图
result = await adapter.process_voice_intent(intent)
```

## 🔧 高级配置

### 安全设置
```python
config = {
    'security': {
        'allowed_operations': ['read', 'write', 'create', 'generate'],
        'restricted_operations': ['delete', 'system_admin'],
        'require_confirmation': ['delete_file', 'refactor_large_file']
    }
}
```

### 代码生成设置
```python
config = {
    'code_generation': {
        'default_language': 'python',
        'coding_style': 'pep8',
        'include_comments': True,
        'include_type_hints': True,
        'include_docstrings': True
    }
}
```

## 📊 监控和统计

### 获取运行状态
```python
state = await adapter.get_current_state()
print(f"状态: {state['status']}")
print(f"可用工具: {state['available_tools']}")
print(f"项目类型: {state['project_info']['type']}")
```

### 查看统计信息
```python
stats = state['statistics']
print(f"执行命令数: {stats['commands_executed']}")
print(f"生成代码数: {stats['code_generated']}")
print(f"创建文件数: {stats['files_created']}")
print(f"错误次数: {stats['errors_count']}")
```

## 🎯 最佳实践

### 1. 语音命令技巧
- 使用清晰、具体的描述
- 包含必要的参数信息
- 避免复杂的嵌套结构
- 使用标准编程术语

### 2. 错误处理
```python
try:
    result = await adapter.execute_command("generate_function", params)
    if not result.success:
        print(f"命令执行失败: {result.error}")
except Exception as e:
    print(f"发生错误: {e}")
```

### 3. 资源管理
```python
try:
    adapter = ClaudeCodeAdapter(config)
    await adapter.initialize()
    
    # 执行操作...
    
finally:
    await adapter.cleanup()  # 确保资源清理
```

## 🐛 常见问题

### Q: 为什么代码生成失败？
A: 检查参数是否完整，确保描述清晰，验证目标语言支持。

### Q: 如何提高语音识别准确度？
A: 使用标准编程术语，避免口语化表达，在安静环境中使用。

### Q: 文件操作权限被拒绝？
A: 检查安全配置中的 allowed_operations 和 protected_files 设置。

### Q: 项目分析结果不准确？
A: 确保项目目录结构完整，包含必要的配置文件（package.json, requirements.txt等）。

## 📚 支持的编程语言

- **Python**: 完整支持，包括类型提示和文档字符串
- **JavaScript**: 支持 ES6+ 语法和 JSDoc
- **TypeScript**: 支持类型定义和接口
- **React**: 支持函数组件和类组件
- **Java**: 基础支持
- **C++**: 基础支持

## 🔗 相关资源

- [完整配置参考](./config/claude_code_config.yaml)
- [架构文档](./docs/claude-code.md)
- [API文档](./docs/development.md)
- [集成测试](./test_adapter_basic.py)

---

**需要帮助？** 查看 [Claude Code Integration Report](./CLAUDE_CODE_INTEGRATION_REPORT.md) 获取详细技术信息。