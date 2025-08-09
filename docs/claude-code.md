# Claude Code 集成指南

## 🎯 集成目标

实现语音助手与 Claude Code 的深度集成，支持通过语音控制进行代码开发、文件操作和工具调用。

## 🏗️ Claude Code 适配器架构

### 核心适配器设计

```python
from typing import Dict, Any, List, Optional
import asyncio
import subprocess
import json
from pathlib import Path

class ClaudeCodeAdapter(BaseAdapter):
    """Claude Code 专门适配器"""
    
    def __init__(self):
        super().__init__()
        self.claude_code_path = self._find_claude_code_installation()
        self.current_session = None
        self.mcp_client = MCPClient()
        
    @property
    def adapter_id(self) -> str:
        return "claude_code"
    
    @property
    def supported_commands(self) -> List[str]:
        return [
            # 文件操作
            "create_file", "edit_file", "read_file", "delete_file",
            "list_files", "search_files",
            
            # 代码生成
            "generate_function", "generate_class", "generate_component",
            "refactor_code", "add_comments", "fix_code",
            
            # 工具调用
            "run_command", "execute_script", "run_tests",
            "format_code", "lint_code", "build_project",
            
            # 项目管理
            "create_project", "analyze_project", "get_project_info",
            
            # Git 操作
            "git_status", "git_commit", "git_push", "git_pull",
            
            # 搜索和导航
            "find_definition", "find_references", "search_code",
            
            # 调试支持
            "set_breakpoint", "start_debug", "debug_step"
        ]
    
    async def execute_command(self, command: str, parameters: Dict[str, Any]) -> CommandResult:
        """执行 Claude Code 命令"""
        
        try:
            # 根据命令类型选择执行方式
            if command in self._direct_commands:
                return await self._execute_direct_command(command, parameters)
            elif command in self._mcp_commands:
                return await self._execute_mcp_command(command, parameters)
            elif command in self._subprocess_commands:
                return await self._execute_subprocess_command(command, parameters)
            else:
                raise UnsupportedCommandError(f"不支持的命令: {command}")
                
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                command=command,
                parameters=parameters
            )
```

### MCP (Model Context Protocol) 集成

```python
class MCPClient:
    """MCP 协议客户端"""
    
    def __init__(self):
        self.connection = None
        self.tools_registry = {}
        self.context_manager = MCPContextManager()
    
    async def connect(self):
        """连接到 Claude Code MCP 服务"""
        try:
            # 启动 Claude Code MCP 服务
            process = await asyncio.create_subprocess_exec(
                self.claude_code_path, "--mcp-server",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 建立连接
            self.connection = await self._establish_mcp_connection(process)
            
            # 获取可用工具列表
            await self._discover_available_tools()
            
            logger.info("MCP 连接已建立")
            
        except Exception as e:
            logger.error(f"MCP 连接失败: {e}")
            raise MCPConnectionError(f"无法连接到 Claude Code: {e}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """调用 MCP 工具"""
        
        if not self.connection:
            await self.connect()
        
        # 构建 MCP 请求
        request = {
            "jsonrpc": "2.0",
            "id": generate_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }
        
        # 发送请求
        response = await self.connection.send_request(request)
        
        if "error" in response:
            raise MCPError(response["error"]["message"])
        
        return response["result"]
    
    async def get_context(self) -> Dict[str, Any]:
        """获取当前上下文"""
        return await self.call_tool("get_context", {})
    
    async def update_context(self, context_updates: Dict[str, Any]):
        """更新上下文"""
        await self.call_tool("update_context", context_updates)
```

## 🎤 语音到代码的智能转换

### 语音意图识别

```python
class CodeIntentParser:
    """代码意图解析器"""
    
    def __init__(self):
        self.patterns = self._load_intent_patterns()
        self.nlp_processor = NLPProcessor()
        
    async def parse_code_intent(self, speech_text: str, context: Context) -> CodeIntent:
        """解析代码相关的语音意图"""
        
        # 1. 预处理语音文本
        cleaned_text = self._preprocess_speech_text(speech_text)
        
        # 2. 实体识别
        entities = await self._extract_code_entities(cleaned_text, context)
        
        # 3. 意图分类
        intent_type = await self._classify_intent(cleaned_text, entities)
        
        # 4. 参数提取
        parameters = await self._extract_parameters(cleaned_text, entities, intent_type)
        
        return CodeIntent(
            type=intent_type,
            entities=entities,
            parameters=parameters,
            confidence=self._calculate_confidence(cleaned_text, intent_type),
            original_text=speech_text
        )
    
    async def _extract_code_entities(self, text: str, context: Context) -> Dict[str, Any]:
        """提取代码相关实体"""
        
        entities = {}
        
        # 编程语言检测
        language = self._detect_programming_language(text, context)
        if language:
            entities["language"] = language
        
        # 文件名检测
        filenames = self._extract_filenames(text)
        if filenames:
            entities["files"] = filenames
        
        # 函数/类名检测
        code_identifiers = self._extract_code_identifiers(text)
        if code_identifiers:
            entities["identifiers"] = code_identifiers
        
        # 路径检测
        paths = self._extract_paths(text)
        if paths:
            entities["paths"] = paths
        
        return entities

# 语音命令示例映射
VOICE_TO_CODE_PATTERNS = {
    # 文件操作
    "创建文件 {filename}": {
        "command": "create_file",
        "parameters": {"filename": "{filename}", "content": ""}
    },
    "打开文件 {filename}": {
        "command": "read_file", 
        "parameters": {"filename": "{filename}"}
    },
    
    # 代码生成
    "创建函数 {function_name}": {
        "command": "generate_function",
        "parameters": {"name": "{function_name}", "language": "auto_detect"}
    },
    "生成 React 组件 {component_name}": {
        "command": "generate_component",
        "parameters": {"type": "react", "name": "{component_name}"}
    },
    
    # 代码编辑
    "添加注释到 {function_name}": {
        "command": "add_comments",
        "parameters": {"target": "{function_name}", "type": "function"}
    },
    "重构这个函数": {
        "command": "refactor_code",
        "parameters": {"target": "current_selection", "type": "function"}
    },
    
    # 项目操作
    "运行测试": {
        "command": "run_tests",
        "parameters": {"scope": "all"}
    },
    "构建项目": {
        "command": "build_project", 
        "parameters": {}
    },
    
    # Git 操作
    "提交代码": {
        "command": "git_commit",
        "parameters": {"message": "auto_generate"}
    },
    "查看状态": {
        "command": "git_status",
        "parameters": {}
    }
}
```

### 智能代码生成

```python
class IntelligentCodeGenerator:
    """智能代码生成器"""
    
    def __init__(self, claude_adapter: ClaudeCodeAdapter):
        self.claude_adapter = claude_adapter
        self.template_manager = CodeTemplateManager()
        self.context_analyzer = CodeContextAnalyzer()
    
    async def generate_code(self, intent: CodeIntent, context: Context) -> GeneratedCode:
        """根据意图生成代码"""
        
        # 1. 分析当前代码上下文
        code_context = await self.context_analyzer.analyze_current_context(context)
        
        # 2. 选择生成策略
        generation_strategy = self._select_generation_strategy(intent, code_context)
        
        # 3. 准备生成参数
        generation_params = await self._prepare_generation_parameters(intent, code_context)
        
        # 4. 调用 Claude Code 生成
        if generation_strategy == GenerationStrategy.TEMPLATE_BASED:
            code = await self._generate_from_template(intent, generation_params)
        elif generation_strategy == GenerationStrategy.AI_ASSISTED:
            code = await self._generate_with_ai_assistance(intent, generation_params)
        elif generation_strategy == GenerationStrategy.PATTERN_MATCHING:
            code = await self._generate_from_patterns(intent, generation_params)
        
        # 5. 后处理和优化
        optimized_code = await self._optimize_generated_code(code, code_context)
        
        return GeneratedCode(
            content=optimized_code,
            language=code_context.language,
            insertion_point=self._determine_insertion_point(intent, code_context),
            confidence=self._calculate_generation_confidence(optimized_code)
        )
    
    async def _generate_with_ai_assistance(self, intent: CodeIntent, params: Dict[str, Any]) -> str:
        """使用 AI 辅助生成代码"""
        
        # 构建生成提示
        prompt = self._build_generation_prompt(intent, params)
        
        # 调用 Claude Code 的代码生成功能
        result = await self.claude_adapter.execute_command(
            "generate_code",
            {
                "prompt": prompt,
                "language": params.get("language"),
                "context": params.get("context"),
                "style": params.get("coding_style", "standard")
            }
        )
        
        if result.success:
            return result.data["generated_code"]
        else:
            raise CodeGenerationError(f"代码生成失败: {result.error}")
```

## 🔧 上下文感知功能

### 智能上下文分析

```python
class CodeContextAnalyzer:
    """代码上下文分析器"""
    
    async def analyze_current_context(self, context: Context) -> CodeContext:
        """分析当前代码上下文"""
        
        code_context = CodeContext()
        
        # 1. 当前文件分析
        if context.current_file:
            file_analysis = await self._analyze_current_file(context.current_file)
            code_context.update(file_analysis)
        
        # 2. 项目结构分析
        project_info = await self._analyze_project_structure(context.working_directory)
        code_context.project_info = project_info
        
        # 3. 光标位置上下文
        cursor_context = await self._analyze_cursor_context(context)
        code_context.cursor_context = cursor_context
        
        # 4. 导入依赖分析
        dependencies = await self._analyze_dependencies(context)
        code_context.dependencies = dependencies
        
        # 5. 代码风格检测
        coding_style = await self._detect_coding_style(context)
        code_context.coding_style = coding_style
        
        return code_context
    
    async def _analyze_current_file(self, file_path: str) -> FileAnalysis:
        """分析当前文件"""
        
        if not Path(file_path).exists():
            return FileAnalysis(exists=False)
        
        # 读取文件内容
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # 语言检测
        language = self._detect_file_language(file_path, content)
        
        # 代码结构分析
        if language in ["python", "javascript", "typescript", "java", "cpp"]:
            structure = await self._analyze_code_structure(content, language)
        else:
            structure = None
        
        return FileAnalysis(
            exists=True,
            language=language,
            size=len(content),
            lines=len(content.splitlines()),
            structure=structure,
            encoding="utf-8"
        )
```

### 智能命令建议

```python
class IntelligentSuggestionEngine:
    """智能建议引擎"""
    
    def __init__(self):
        self.suggestion_models = {
            "file_operations": FileOperationSuggestions(),
            "code_generation": CodeGenerationSuggestions(),
            "refactoring": RefactoringSuggestions(),
            "debugging": DebuggingSuggestions()
        }
    
    async def get_contextual_suggestions(self, context: Context) -> List[Suggestion]:
        """根据上下文获取建议"""
        
        suggestions = []
        
        # 1. 基于当前文件的建议
        if context.current_file:
            file_suggestions = await self._get_file_based_suggestions(context)
            suggestions.extend(file_suggestions)
        
        # 2. 基于项目状态的建议
        project_suggestions = await self._get_project_based_suggestions(context)
        suggestions.extend(project_suggestions)
        
        # 3. 基于错误状态的建议
        if self._detect_errors_in_context(context):
            error_suggestions = await self._get_error_fixing_suggestions(context)
            suggestions.extend(error_suggestions)
        
        # 4. 基于用户历史的建议
        history_suggestions = await self._get_history_based_suggestions(context)
        suggestions.extend(history_suggestions)
        
        # 5. 排序和过滤
        ranked_suggestions = self._rank_suggestions(suggestions, context)
        
        return ranked_suggestions[:10]  # 返回前10个建议
    
    async def _get_file_based_suggestions(self, context: Context) -> List[Suggestion]:
        """基于当前文件的建议"""
        
        suggestions = []
        file_ext = Path(context.current_file).suffix.lower()
        
        # Python 文件特定建议
        if file_ext == ".py":
            suggestions.extend([
                Suggestion("运行当前 Python 文件", "run_python_file", {"file": context.current_file}),
                Suggestion("格式化 Python 代码", "format_python", {"file": context.current_file}),
                Suggestion("检查 Python 语法", "lint_python", {"file": context.current_file})
            ])
        
        # JavaScript/TypeScript 文件建议
        elif file_ext in [".js", ".ts", ".jsx", ".tsx"]:
            suggestions.extend([
                Suggestion("运行 npm 脚本", "run_npm_script", {}),
                Suggestion("格式化 JS/TS 代码", "format_javascript", {"file": context.current_file}),
                Suggestion("检查 TypeScript 类型", "check_typescript", {"file": context.current_file})
            ])
        
        # 通用建议
        suggestions.extend([
            Suggestion("保存文件", "save_file", {"file": context.current_file}),
            Suggestion("查找文件中的函数", "find_functions", {"file": context.current_file}),
            Suggestion("搜索相似文件", "find_similar_files", {"reference": context.current_file})
        ])
        
        return suggestions
```

## 🚀 实际使用场景

### 场景1: 快速创建 React 组件

```python
# 语音输入: "创建一个用户卡片 React 组件"
async def create_react_component_scenario(voice_input: str):
    """创建 React 组件场景"""
    
    # 1. 意图解析
    intent = await code_intent_parser.parse_code_intent(voice_input, current_context)
    # Intent: type=CREATE_COMPONENT, entities={"component_name": "用户卡片", "type": "react"}
    
    # 2. 任务规划
    plan = TaskPlan([
        Task("analyze_project", "分析项目结构"),
        Task("generate_component", "生成组件代码"),
        Task("create_component_file", "创建组件文件"),
        Task("update_index_exports", "更新导出文件"),
        Task("create_storybook_story", "创建 Storybook 故事"),  # 可选
        Task("run_type_check", "检查 TypeScript 类型")  # 如果是 TS 项目
    ])
    
    # 3. 执行
    for task in plan.tasks:
        result = await claude_adapter.execute_command(task.command, task.parameters)
        if not result.success:
            await handle_task_error(task, result.error)
    
    # 4. 反馈
    await voice_interface.speak("用户卡片组件已创建完成，包含基础属性和样式")
```

### 场景2: 智能代码重构

```python
# 语音输入: "重构这个函数，让它更简洁"
async def refactor_code_scenario(voice_input: str):
    """代码重构场景"""
    
    # 1. 获取当前选中的代码
    selected_code = await claude_adapter.get_selected_code()
    
    # 2. 分析代码结构
    code_analysis = await code_analyzer.analyze_code(selected_code)
    
    # 3. 生成重构建议
    refactor_suggestions = await refactoring_engine.generate_suggestions(code_analysis)
    
    # 4. 应用最佳重构
    refactored_code = await refactoring_engine.apply_refactoring(
        selected_code, refactor_suggestions[0]
    )
    
    # 5. 替换原代码
    await claude_adapter.replace_selected_code(refactored_code)
    
    # 6. 运行测试确保功能正确
    test_result = await claude_adapter.run_relevant_tests()
    
    if test_result.success:
        await voice_interface.speak("代码重构完成，所有测试通过")
    else:
        await voice_interface.speak(f"重构完成，但有 {test_result.failed_count} 个测试失败")
```

### 场景3: 静音模式开发

```python
# 静音模式：用户离开时自动完成开发任务
async def silent_development_scenario():
    """静音模式开发场景"""
    
    # 用户离开前的指令："我去开会了，帮我完成 TODO 列表中的任务"
    
    pending_tasks = await todo_manager.get_pending_tasks()
    
    for task in pending_tasks:
        # 安全检查
        if await security_guardian.is_safe_for_autonomous_execution(task):
            
            try:
                # 执行任务
                result = await claude_adapter.execute_autonomous_task(task)
                
                # 记录执行结果
                await execution_log.log_autonomous_execution(task, result)
                
                # 如果是代码修改，运行测试
                if task.type == TaskType.CODE_MODIFICATION:
                    test_result = await claude_adapter.run_tests()
                    if not test_result.success:
                        await rollback_manager.rollback_changes(task)
                
            except Exception as e:
                await error_handler.handle_autonomous_error(task, e)
    
    # 准备汇报
    await handover_manager.prepare_development_report(pending_tasks)
```

## 🛠️ 配置和设置

### 配置文件示例

```yaml
# claude_code_config.yaml
claude_code:
  installation_path: "C:\\Users\\{username}\\AppData\\Local\\Claude\\claude.exe"
  
  # MCP 设置
  mcp:
    enabled: true
    server_port: 8080
    timeout: 30
    max_retries: 3
  
  # 代码生成设置
  code_generation:
    default_language: "auto_detect"
    coding_style: "standard"
    include_comments: true
    include_type_hints: true  # Python/TypeScript
    
  # 文件操作设置
  file_operations:
    auto_backup: true
    backup_directory: "./.voice-assistant-backups"
    max_file_size: "10MB"
    
  # 项目集成
  project_integration:
    auto_detect_framework: true
    supported_frameworks: ["react", "vue", "angular", "django", "flask", "express"]
    
  # 安全设置
  security:
    allowed_operations: ["read", "write", "create", "format", "lint"]
    restricted_operations: ["delete", "system_command"]
    require_confirmation: ["refactor_large_file", "mass_file_operations"]
    
  # 语音识别优化
  speech:
    code_vocabulary: true  # 启用编程词汇优化
    abbreviation_expansion: true  # 自动展开缩写
    punctuation_inference: true  # 推断标点符号
```

### 初始化设置

```python
async def initialize_claude_code_integration():
    """初始化 Claude Code 集成"""
    
    # 1. 检查 Claude Code 安装
    if not await check_claude_code_installation():
        raise IntegrationError("未找到 Claude Code 安装")
    
    # 2. 建立 MCP 连接
    mcp_client = MCPClient()
    await mcp_client.connect()
    
    # 3. 初始化适配器
    claude_adapter = ClaudeCodeAdapter(mcp_client)
    
    # 4. 配置语音识别优化
    speech_engine.load_code_vocabulary()
    speech_engine.enable_programming_context()
    
    # 5. 注册适配器
    adapter_manager.register_adapter(claude_adapter)
    
    logger.info("Claude Code 集成初始化完成")
    
    return claude_adapter
```

这个集成方案确保了语音助手与 Claude Code 的无缝协作，为开发者提供了强大的语音驱动开发体验。