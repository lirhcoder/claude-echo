# 开发指南

## 🚀 开发环境设置

### 环境要求

```bash
# Python 版本
Python >= 3.9

# 系统要求
Windows 10/11 (主要支持)
macOS 10.15+ (部分支持)
Linux (实验性支持)
```

### 依赖安装

```bash
# 1. 克隆项目
git clone <repository-url>
cd claude-voice-assistant

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 开发依赖
pip install -r requirements-dev.txt
```

### 开发工具配置

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```

## 📁 项目结构详解

```
claude-voice-assistant/
├── src/                          # 源代码
│   ├── core/                     # 核心系统
│   │   ├── __init__.py
│   │   ├── base.py              # 基础类和接口
│   │   ├── exceptions.py        # 自定义异常
│   │   └── config.py            # 配置管理
│   │
│   ├── agents/                   # 智能代理
│   │   ├── __init__.py
│   │   ├── coordinator.py       # 协调中心
│   │   ├── task_planner.py      # 任务规划
│   │   ├── presence_monitor.py  # 状态监控
│   │   ├── auto_worker.py       # 自主执行
│   │   ├── security_guardian.py # 安全监护
│   │   ├── handover_manager.py  # 交接管理
│   │   └── session_manager.py   # 会话管理
│   │
│   ├── adapters/                 # 适配器
│   │   ├── __init__.py
│   │   ├── base_adapter.py      # 适配器基类
│   │   ├── claude_code.py       # Claude Code适配器
│   │   ├── system_adapter.py    # 系统操作适配器
│   │   ├── ide_adapter.py       # IDE适配器
│   │   └── office_adapter.py    # 办公软件适配器
│   │
│   ├── speech/                   # 语音处理
│   │   ├── __init__.py
│   │   ├── recognizer.py        # 语音识别
│   │   ├── synthesizer.py       # 语音合成
│   │   ├── intent_parser.py     # 意图解析
│   │   └── voice_interface.py   # 语音界面
│   │
│   ├── utils/                    # 工具库
│   │   ├── __init__.py
│   │   ├── logger.py            # 日志系统
│   │   ├── decorators.py        # 装饰器
│   │   ├── async_utils.py       # 异步工具
│   │   └── file_utils.py        # 文件工具
│   │
│   └── main.py                   # 程序入口
│
├── tests/                        # 测试代码
│   ├── unit/                     # 单元测试
│   ├── integration/              # 集成测试
│   ├── fixtures/                 # 测试数据
│   └── conftest.py              # pytest配置
│
├── config/                       # 配置文件
│   ├── default.yaml             # 默认配置
│   ├── development.yaml         # 开发配置
│   └── production.yaml          # 生产配置
│
├── docs/                         # 文档
├── requirements.txt              # 依赖列表
├── requirements-dev.txt          # 开发依赖
├── setup.py                     # 安装脚本
└── README.md                    # 项目说明
```

## 🔧 核心组件开发

### 1. 适配器开发

#### 创建新适配器

```python
# src/adapters/example_adapter.py
from typing import Dict, Any, List
from .base_adapter import BaseAdapter, CommandResult

class ExampleAdapter(BaseAdapter):
    """示例适配器"""
    
    def __init__(self):
        super().__init__()
        self._initialize_adapter()
    
    @property
    def adapter_id(self) -> str:
        return "example"
    
    @property
    def supported_commands(self) -> List[str]:
        return ["example_command", "another_command"]
    
    async def execute_command(self, command: str, parameters: Dict[str, Any]) -> CommandResult:
        """执行命令实现"""
        
        try:
            if command == "example_command":
                return await self._handle_example_command(parameters)
            elif command == "another_command":
                return await self._handle_another_command(parameters)
            else:
                return CommandResult(
                    success=False,
                    error=f"不支持的命令: {command}"
                )
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                details={"exception_type": type(e).__name__}
            )
    
    async def _handle_example_command(self, parameters: Dict[str, Any]) -> CommandResult:
        """处理示例命令"""
        
        # 实现具体逻辑
        result_data = {"message": "命令执行成功"}
        
        return CommandResult(
            success=True,
            data=result_data,
            command="example_command",
            parameters=parameters
        )
    
    def is_available(self) -> bool:
        """检查适配器可用性"""
        # 实现可用性检查逻辑
        return True
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "status": "ready",
            "last_command": self.last_executed_command,
            "error_count": self.error_count
        }
    
    def get_command_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """基于上下文提供命令建议"""
        suggestions = []
        
        # 基于上下文生成建议
        if context.get("current_file"):
            suggestions.append("example_command")
        
        return suggestions
```

#### 注册适配器

```python
# src/adapters/__init__.py
from .example_adapter import ExampleAdapter

# 在适配器管理器中注册
def register_all_adapters(adapter_manager):
    """注册所有适配器"""
    
    adapters = [
        ExampleAdapter(),
        # 其他适配器...
    ]
    
    for adapter in adapters:
        adapter_manager.register_adapter(adapter)
```

### 2. Agent 开发

#### 创建新 Agent

```python
# src/agents/example_agent.py
import asyncio
from typing import Dict, Any, Optional
from ..core.base import BaseAgent

class ExampleAgent(BaseAgent):
    """示例智能代理"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.state = AgentState.IDLE
        
    async def initialize(self):
        """初始化代理"""
        await self._setup_resources()
        self.state = AgentState.READY
        
    async def process(self, input_data: Any) -> Any:
        """处理输入数据"""
        
        try:
            self.state = AgentState.PROCESSING
            
            # 实现处理逻辑
            result = await self._process_internal(input_data)
            
            self.state = AgentState.READY
            return result
            
        except Exception as e:
            self.state = AgentState.ERROR
            raise AgentProcessingError(f"代理处理失败: {e}")
    
    async def _process_internal(self, data: Any) -> Any:
        """内部处理逻辑"""
        
        # 具体实现
        await asyncio.sleep(0.1)  # 模拟处理时间
        return {"processed": True, "data": data}
    
    async def cleanup(self):
        """清理资源"""
        await self._cleanup_resources()
        self.state = AgentState.STOPPED
```

## 🧪 测试开发

### 单元测试

```python
# tests/unit/test_example_adapter.py
import pytest
from unittest.mock import Mock, patch
from src.adapters.example_adapter import ExampleAdapter

class TestExampleAdapter:
    
    @pytest.fixture
    def adapter(self):
        """测试适配器实例"""
        return ExampleAdapter()
    
    def test_adapter_id(self, adapter):
        """测试适配器ID"""
        assert adapter.adapter_id == "example"
    
    def test_supported_commands(self, adapter):
        """测试支持的命令列表"""
        commands = adapter.supported_commands
        assert "example_command" in commands
        assert "another_command" in commands
    
    @pytest.mark.asyncio
    async def test_execute_command_success(self, adapter):
        """测试命令执行成功"""
        
        result = await adapter.execute_command(
            "example_command", 
            {"param1": "value1"}
        )
        
        assert result.success is True
        assert result.data["message"] == "命令执行成功"
    
    @pytest.mark.asyncio
    async def test_execute_command_unsupported(self, adapter):
        """测试不支持的命令"""
        
        result = await adapter.execute_command("unsupported", {})
        
        assert result.success is False
        assert "不支持的命令" in result.error
    
    def test_is_available(self, adapter):
        """测试可用性检查"""
        assert adapter.is_available() is True
```

### 集成测试

```python
# tests/integration/test_agent_cooperation.py
import pytest
from src.agents.coordinator import Coordinator
from src.agents.task_planner import TaskPlanner
from src.agents.auto_worker import AutoWorker

class TestAgentCooperation:
    
    @pytest.fixture
    async def coordinator(self):
        """协调器实例"""
        coordinator = Coordinator()
        await coordinator.initialize()
        return coordinator
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, coordinator):
        """测试完整工作流程"""
        
        # 准备测试数据
        user_input = "创建一个测试文件"
        context = create_test_context()
        
        # 执行工作流程
        response = await coordinator.process_request(user_input, context)
        
        # 验证结果
        assert response.success is True
        assert "文件已创建" in response.message
```

## 📊 性能监控

### 性能测试

```python
# tests/performance/test_response_time.py
import time
import pytest
from src.main import VoiceAssistant

class TestPerformance:
    
    @pytest.mark.asyncio
    async def test_response_time(self):
        """测试响应时间"""
        
        assistant = VoiceAssistant()
        await assistant.initialize()
        
        start_time = time.time()
        
        response = await assistant.process_voice_input("测试命令")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # 响应时间应该小于2秒
        assert response_time < 2.0, f"响应时间过长: {response_time}秒"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """测试内存使用"""
        
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss
        
        # 执行大量操作
        assistant = VoiceAssistant()
        await assistant.initialize()
        
        for _ in range(100):
            await assistant.process_voice_input("测试命令")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该小于100MB
        assert memory_increase < 100 * 1024 * 1024, f"内存增长过多: {memory_increase} bytes"
```

## 🐛 调试技巧

### 日志配置

```python
# src/utils/logger.py
import logging
import sys
from pathlib import Path

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """设置日志器"""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # 文件处理器
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
```

### 调试装饰器

```python
# src/utils/decorators.py
import functools
import time
import logging
from typing import Callable, Any

def debug_timing(func: Callable) -> Callable:
    """调试执行时间装饰器"""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            
            logger = logging.getLogger(func.__module__)
            logger.debug(
                f"{func.__name__} 执行完成，用时 {end_time - start_time:.3f} 秒"
            )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            
            logger = logging.getLogger(func.__module__)
            logger.error(
                f"{func.__name__} 执行失败，用时 {end_time - start_time:.3f} 秒，错误: {e}"
            )
            
            raise
    
    return wrapper

def debug_params(func: Callable) -> Callable:
    """调试参数装饰器"""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} 调用参数: args={args}, kwargs={kwargs}")
        
        return await func(*args, **kwargs)
    
    return wrapper
```

## 🔄 持续集成

### GitHub Actions 配置

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src tests --count --max-complexity=10 --statistics
    
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

## 📦 代码规范

### 代码风格

```python
# .flake8
[flake8]
max-line-length = 100
exclude = 
    .git,
    __pycache__,
    build,
    dist,
    venv
ignore = 
    E203,  # whitespace before ':'
    W503   # line break before binary operator
```

```python
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.mypy_cache
  | \.venv
  | build
  | dist
)/
'''
```

### 提交规范

```bash
# 提交消息格式
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型:**
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建或工具链相关

**示例:**
```bash
feat(agents): 添加任务规划代理

- 实现智能任务分解算法
- 添加依赖关系分析
- 支持执行时间估算

Closes #123
```

## 🚀 部署指南

### 本地开发部署

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 安装开发依赖
pip install -e .

# 3. 运行开发服务器
python src/main.py --dev

# 4. 运行测试
pytest tests/
```

### 生产部署

```bash
# 1. 构建分发包
python setup.py sdist bdist_wheel

# 2. 安装
pip install dist/claude-voice-assistant-*.whl

# 3. 运行
claude-voice-assistant --config config/production.yaml
```

这个开发指南提供了完整的开发环境设置、代码规范、测试策略和部署方式，确保团队成员能够高效协作开发。