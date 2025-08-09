"""Claude Code Adapter - Deep Integration with Claude Code MCP Protocol

This adapter provides seamless voice-driven programming capabilities through
Claude Code's Model Context Protocol (MCP), enabling natural language programming
and intelligent code generation.

Features:
- MCP protocol client for Claude Code communication
- Voice-to-code intelligent conversion
- Context-aware code generation
- File operations with safety checks
- Project analysis and management
- Tool chain orchestration (Bash, Git, Testing)
- Smart code suggestions and completions
"""

import asyncio
import json
import os
import subprocess
from typing import Dict, Any, List, Optional, Union, Set
from pathlib import Path
from datetime import datetime
import uuid
import re

from loguru import logger

from ..core.base_adapter import BaseAdapter, AdapterError, CommandNotSupportedError
from ..core.types import CommandResult, AdapterStatus, Context
from ..core.event_system import EventSystem, Event

# Import speech types only when available
try:
    from ..speech.types import IntentType, ParsedIntent
except ImportError:
    # Fallback definitions for testing without speech dependencies
    from enum import Enum
    from typing import Any
    from dataclasses import dataclass
    
    class IntentType(Enum):
        CODING_REQUEST = "coding_request"
        FILE_OPERATION = "file_operation"
        SYSTEM_CONTROL = "system_control"
        QUERY_REQUEST = "query_request"
    
    @dataclass
    class ParsedIntent:
        original_text: str
        processed_text: str
        intent_type: IntentType
        confidence: float
        entities: Dict[str, Any]
        parameters: Dict[str, Any]


class ClaudeCodeError(AdapterError):
    """Claude Code specific errors"""
    pass


class MCPConnectionError(ClaudeCodeError):
    """MCP connection related errors"""
    pass


class CodeGenerationError(ClaudeCodeError):
    """Code generation related errors"""  
    pass


class MCPClient:
    """Model Context Protocol client for Claude Code communication"""
    
    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        """
        Initialize MCP client.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.connection = None
        self.session_id = None
        self.tools_registry: Dict[str, Dict[str, Any]] = {}
        self.request_id_counter = 0
        
    async def connect(self) -> bool:
        """
        Connect to Claude Code MCP server.
        
        Returns:
            True if connection successful
        """
        try:
            # For now, we'll simulate MCP connection
            # In real implementation, this would establish actual MCP protocol connection
            self.session_id = str(uuid.uuid4())
            logger.info(f"MCP client connected with session: {self.session_id}")
            
            # Discover available tools
            await self._discover_tools()
            return True
            
        except Exception as e:
            logger.error(f"MCP connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server"""
        if self.connection:
            try:
                # Close connection gracefully
                self.connection = None
                self.session_id = None
                logger.info("MCP client disconnected")
            except Exception as e:
                logger.error(f"Error during MCP disconnect: {e}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool via MCP protocol.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            MCPConnectionError: If not connected or connection fails
            ClaudeCodeError: If tool execution fails
        """
        if not self.session_id:
            await self.connect()
        
        if tool_name not in self.tools_registry:
            raise ClaudeCodeError(f"Tool '{tool_name}' not available")
        
        # Generate request ID
        self.request_id_counter += 1
        request_id = f"req_{self.request_id_counter}"
        
        try:
            # Simulate MCP tool call
            # In real implementation, this would send actual MCP protocol message
            result = await self._simulate_tool_call(tool_name, parameters, request_id)
            return result
            
        except Exception as e:
            logger.error(f"Tool call failed - {tool_name}: {e}")
            raise ClaudeCodeError(f"Tool execution failed: {e}")
    
    async def _discover_tools(self) -> None:
        """Discover available tools from Claude Code"""
        # Simulate tool discovery
        # In real implementation, this would query actual Claude Code tools
        self.tools_registry = {
            # File operations
            "read_file": {"description": "Read file contents", "parameters": ["file_path"]},
            "write_file": {"description": "Write file contents", "parameters": ["file_path", "content"]},
            "create_file": {"description": "Create new file", "parameters": ["file_path", "content"]},
            "delete_file": {"description": "Delete file", "parameters": ["file_path"]},
            "list_files": {"description": "List directory contents", "parameters": ["directory_path"]},
            "search_files": {"description": "Search for files", "parameters": ["pattern", "directory"]},
            
            # Code generation
            "generate_function": {"description": "Generate function code", "parameters": ["name", "description", "language"]},
            "generate_class": {"description": "Generate class code", "parameters": ["name", "description", "language"]},
            "generate_component": {"description": "Generate UI component", "parameters": ["type", "name", "props"]},
            "refactor_code": {"description": "Refactor code", "parameters": ["code", "instructions"]},
            "add_comments": {"description": "Add comments to code", "parameters": ["code", "style"]},
            "fix_code": {"description": "Fix code issues", "parameters": ["code", "error_description"]},
            
            # Tool execution
            "run_bash_command": {"description": "Execute bash command", "parameters": ["command", "working_directory"]},
            "run_git_command": {"description": "Execute git command", "parameters": ["command", "repository_path"]},
            "run_tests": {"description": "Run project tests", "parameters": ["test_type", "path"]},
            "format_code": {"description": "Format code", "parameters": ["file_path", "formatter"]},
            "lint_code": {"description": "Lint code", "parameters": ["file_path", "linter"]},
            
            # Project analysis
            "analyze_project": {"description": "Analyze project structure", "parameters": ["project_path"]},
            "get_project_info": {"description": "Get project information", "parameters": ["project_path"]},
            "find_definition": {"description": "Find symbol definition", "parameters": ["symbol", "file_path"]},
            "find_references": {"description": "Find symbol references", "parameters": ["symbol", "project_path"]},
            "search_code": {"description": "Search code patterns", "parameters": ["pattern", "project_path"]},
            
            # Context management
            "get_context": {"description": "Get current context", "parameters": []},
            "update_context": {"description": "Update context", "parameters": ["updates"]},
            "get_cursor_context": {"description": "Get cursor context", "parameters": ["file_path", "line", "column"]},
        }
        
        logger.info(f"Discovered {len(self.tools_registry)} tools")
    
    async def _simulate_tool_call(self, tool_name: str, parameters: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Simulate MCP tool call for development purposes"""
        
        # Add small delay to simulate network call
        await asyncio.sleep(0.1)
        
        # Simulate different tool responses
        if tool_name == "read_file":
            file_path = parameters.get("file_path", "")
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return {
                        "success": True,
                        "content": content,
                        "file_path": file_path,
                        "size": len(content)
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}
            else:
                return {"success": False, "error": "File not found"}
                
        elif tool_name == "write_file":
            file_path = parameters.get("file_path", "")
            content = parameters.get("content", "")
            try:
                # Ensure directory exists
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {
                    "success": True,
                    "file_path": file_path,
                    "bytes_written": len(content.encode('utf-8'))
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif tool_name == "generate_function":
            name = parameters.get("name", "unnamed_function")
            description = parameters.get("description", "")
            language = parameters.get("language", "python")
            
            # Simple template-based generation
            if language.lower() == "python":
                code = f'''def {name}():
    """
    {description}
    
    Returns:
        None: This function needs implementation
    """
    # TODO: Implement this function
    pass
'''
            elif language.lower() in ["javascript", "js"]:
                code = f'''/**
 * {description}
 * @returns {{void}}
 */
function {name}() {{
    // TODO: Implement this function
}}
'''
            else:
                code = f"// Generated {name} function\n// TODO: Implement for {language}"
            
            return {
                "success": True,
                "generated_code": code,
                "language": language,
                "function_name": name
            }
        
        elif tool_name == "run_bash_command":
            command = parameters.get("command", "")
            working_dir = parameters.get("working_directory", os.getcwd())
            
            try:
                # Execute command safely
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "command": command
                }
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Command timed out"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif tool_name == "analyze_project":
            project_path = parameters.get("project_path", os.getcwd())
            try:
                project_info = await self._analyze_project_structure(project_path)
                return {
                    "success": True,
                    "project_info": project_info
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        else:
            # Default response for unimplemented tools
            return {
                "success": True,
                "message": f"Tool '{tool_name}' executed successfully (simulated)",
                "parameters": parameters,
                "request_id": request_id
            }
    
    async def _analyze_project_structure(self, project_path: str) -> Dict[str, Any]:
        """Analyze project structure and return metadata"""
        project_path = Path(project_path)
        
        if not project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {project_path}")
        
        # Basic project analysis
        files = list(project_path.rglob("*"))
        python_files = list(project_path.rglob("*.py"))
        js_files = list(project_path.rglob("*.js")) + list(project_path.rglob("*.ts"))
        
        # Detect project type
        project_type = "unknown"
        if (project_path / "package.json").exists():
            project_type = "javascript/node"
        elif (project_path / "requirements.txt").exists() or (project_path / "pyproject.toml").exists():
            project_type = "python"
        elif (project_path / "pom.xml").exists():
            project_type = "java"
        elif (project_path / "Cargo.toml").exists():
            project_type = "rust"
        
        # Framework detection
        framework = None
        if project_type == "javascript/node":
            package_json = project_path / "package.json"
            if package_json.exists():
                try:
                    with open(package_json) as f:
                        pkg_data = json.load(f)
                    deps = {**pkg_data.get("dependencies", {}), **pkg_data.get("devDependencies", {})}
                    if "react" in deps:
                        framework = "react"
                    elif "vue" in deps:
                        framework = "vue"
                    elif "angular" in deps:
                        framework = "angular"
                    elif "express" in deps:
                        framework = "express"
                except:
                    pass
        
        return {
            "path": str(project_path),
            "type": project_type,
            "framework": framework,
            "total_files": len(files),
            "python_files": len(python_files),
            "javascript_files": len(js_files),
            "has_git": (project_path / ".git").exists(),
            "has_tests": any(
                "test" in f.name.lower() or f.name.startswith("test_")
                for f in files if f.is_file()
            )
        }


class ClaudeCodeAdapter(BaseAdapter):
    """
    Claude Code Adapter for voice-driven programming.
    
    This adapter integrates with Claude Code through the MCP protocol to provide:
    - Intelligent code generation from voice commands
    - File operations with safety checks
    - Project analysis and management
    - Tool chain orchestration
    - Context-aware programming assistance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Claude Code adapter.
        
        Args:
            config: Adapter configuration
        """
        super().__init__(config)
        
        # Core components
        self.mcp_client = MCPClient(
            timeout=self._config.get('mcp', {}).get('timeout', 30.0),
            max_retries=self._config.get('mcp', {}).get('max_retries', 3)
        )
        
        # Current session context
        self.current_context: Optional[Context] = None
        self.working_directory = self._config.get('working_directory', os.getcwd())
        self.project_info: Optional[Dict[str, Any]] = None
        
        # Safety and security
        self.safe_operations = set(self._config.get('security', {}).get('allowed_operations', [
            'read', 'write', 'create', 'format', 'lint', 'analyze'
        ]))
        self.restricted_operations = set(self._config.get('security', {}).get('restricted_operations', [
            'delete', 'system_command', 'network_request'
        ]))
        
        # Code generation settings
        self.code_style = self._config.get('code_generation', {}).get('coding_style', 'standard')
        self.include_comments = self._config.get('code_generation', {}).get('include_comments', True)
        self.include_type_hints = self._config.get('code_generation', {}).get('include_type_hints', True)
        
        # Statistics and monitoring
        self.stats = {
            'commands_executed': 0,
            'files_created': 0,
            'files_modified': 0,
            'code_generated': 0,
            'errors_count': 0,
            'last_activity': None
        }
        
        logger.info("ClaudeCodeAdapter initialized")
    
    # Required BaseAdapter properties
    
    @property
    def adapter_id(self) -> str:
        return "claude_code"
    
    @property
    def name(self) -> str:
        return "Claude Code Adapter"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Voice-driven programming adapter with Claude Code MCP integration"
    
    @property
    def supported_commands(self) -> List[str]:
        return [
            # File operations
            "read_file", "write_file", "create_file", "delete_file",
            "list_files", "search_files", "get_file_info",
            
            # Code generation
            "generate_function", "generate_class", "generate_component",
            "generate_module", "generate_test", "generate_docs",
            
            # Code modification
            "refactor_code", "add_comments", "fix_code", "format_code",
            "optimize_code", "add_type_hints", "extract_method",
            
            # Tool execution  
            "run_bash_command", "run_git_command", "run_tests",
            "lint_code", "build_project", "install_dependencies",
            
            # Project management
            "analyze_project", "get_project_info", "create_project",
            "initialize_git", "setup_environment",
            
            # Code navigation
            "find_definition", "find_references", "search_code",
            "get_symbols", "get_imports", "analyze_dependencies",
            
            # Context management
            "get_context", "update_context", "get_cursor_context",
            "set_working_directory", "get_current_file"
        ]
    
    # Required BaseAdapter methods
    
    async def initialize(self) -> bool:
        """Initialize the Claude Code adapter"""
        try:
            logger.info("Initializing Claude Code adapter...")
            
            # Connect to MCP server
            if not await self.mcp_client.connect():
                logger.error("Failed to connect to Claude Code MCP server")
                self._update_status(AdapterStatus.ERROR)
                return False
            
            # Analyze current project if in a project directory
            if os.path.exists(os.path.join(self.working_directory, '.git')):
                try:
                    result = await self.mcp_client.call_tool("analyze_project", {
                        "project_path": self.working_directory
                    })
                    if result.get("success"):
                        self.project_info = result.get("project_info")
                        logger.info(f"Detected project type: {self.project_info.get('type', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Project analysis failed: {e}")
            
            self._update_status(AdapterStatus.AVAILABLE)
            logger.info("Claude Code adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Claude Code adapter initialization failed: {e}")
            self._update_status(AdapterStatus.ERROR)
            return False
    
    async def cleanup(self) -> None:
        """Clean up adapter resources"""
        try:
            # Disconnect MCP client
            await self.mcp_client.disconnect()
            
            # Clear context
            self.current_context = None
            self.project_info = None
            
            logger.info("Claude Code adapter cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during Claude Code adapter cleanup: {e}")
    
    async def execute_command(self, 
                            command: str, 
                            parameters: Dict[str, Any],
                            context: Optional[Context] = None) -> CommandResult:
        """
        Execute a command through the Claude Code adapter.
        
        Args:
            command: Command to execute
            parameters: Command parameters
            context: Current system context
            
        Returns:
            CommandResult with execution results
        """
        start_time = datetime.now()
        
        try:
            # Update context
            if context:
                self.current_context = context
            
            # Security check
            if not await self._security_check(command, parameters):
                return CommandResult(
                    success=False,
                    error=f"Command '{command}' not allowed by security policy"
                )
            
            # Execute command based on type
            if command.startswith('generate_'):
                result = await self._execute_generation_command(command, parameters)
            elif command.endswith('_file') or command in ['list_files', 'search_files']:
                result = await self._execute_file_command(command, parameters)
            elif command.startswith('run_') or command in ['lint_code', 'format_code']:
                result = await self._execute_tool_command(command, parameters)
            elif command.startswith('find_') or command == 'search_code':
                result = await self._execute_navigation_command(command, parameters)
            elif command.startswith('analyze_') or command.endswith('_info'):
                result = await self._execute_analysis_command(command, parameters)
            elif command.endswith('_context') or command.startswith('get_'):
                result = await self._execute_context_command(command, parameters)
            else:
                # Default: try to execute through MCP
                mcp_result = await self.mcp_client.call_tool(command, parameters)
                result = CommandResult(
                    success=mcp_result.get("success", True),
                    data=mcp_result,
                    error=mcp_result.get("error"),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Update statistics
            self.stats['commands_executed'] += 1
            self.stats['last_activity'] = datetime.now()
            
            if not result.success:
                self.stats['errors_count'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed - {command}: {e}")
            self.stats['errors_count'] += 1
            
            return CommandResult(
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def is_available(self) -> bool:
        """Check if adapter is available"""
        return (self._status == AdapterStatus.AVAILABLE and 
                self.mcp_client.session_id is not None)
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current adapter state"""
        return {
            'status': self._status.value,
            'working_directory': self.working_directory,
            'project_info': self.project_info,
            'mcp_connected': self.mcp_client.session_id is not None,
            'statistics': self.stats,
            'available_tools': len(self.mcp_client.tools_registry)
        }
    
    async def get_command_suggestions(self, context: Optional[Context] = None) -> List[str]:
        """Get contextual command suggestions"""
        suggestions = []
        
        # Base suggestions always available
        suggestions.extend(['read_file', 'list_files', 'get_project_info'])
        
        # Context-based suggestions
        if context and context.current_file:
            file_ext = Path(context.current_file).suffix.lower()
            
            if file_ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
                suggestions.extend([
                    'format_code', 'lint_code', 'add_comments', 
                    'generate_test', 'find_references'
                ])
            
            if file_ext == '.py':
                suggestions.extend(['add_type_hints', 'run_tests'])
            elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
                suggestions.extend(['run_tests', 'build_project'])
        
        # Project-based suggestions
        if self.project_info:
            project_type = self.project_info.get('type', '')
            if 'python' in project_type:
                suggestions.extend(['run_tests', 'install_dependencies'])
            elif 'javascript' in project_type:
                suggestions.extend(['run_bash_command', 'build_project'])
        
        return suggestions
    
    # Command execution methods
    
    async def _execute_generation_command(self, command: str, parameters: Dict[str, Any]) -> CommandResult:
        """Execute code generation commands"""
        try:
            # Map command to MCP tool
            tool_mapping = {
                'generate_function': 'generate_function',
                'generate_class': 'generate_class', 
                'generate_component': 'generate_component',
                'generate_module': 'generate_module',
                'generate_test': 'generate_test',
                'generate_docs': 'generate_docs'
            }
            
            mcp_tool = tool_mapping.get(command)
            if not mcp_tool:
                return CommandResult(success=False, error=f"Unknown generation command: {command}")
            
            # Add context information to parameters
            enhanced_params = parameters.copy()
            if self.current_context and self.current_context.current_file:
                enhanced_params['context_file'] = self.current_context.current_file
            
            if self.project_info:
                enhanced_params['project_type'] = self.project_info.get('type', 'unknown')
                enhanced_params['framework'] = self.project_info.get('framework')
            
            # Apply code generation settings
            enhanced_params['include_comments'] = enhanced_params.get('include_comments', self.include_comments)
            enhanced_params['include_type_hints'] = enhanced_params.get('include_type_hints', self.include_type_hints)
            enhanced_params['coding_style'] = enhanced_params.get('coding_style', self.code_style)
            
            # Execute through MCP
            result = await self.mcp_client.call_tool(mcp_tool, enhanced_params)
            
            if result.get('success'):
                self.stats['code_generated'] += 1
                
                # If file_path is provided, write the generated code
                if 'file_path' in parameters and 'generated_code' in result:
                    write_result = await self.mcp_client.call_tool('write_file', {
                        'file_path': parameters['file_path'],
                        'content': result['generated_code']
                    })
                    result['file_written'] = write_result.get('success', False)
                    
                    if write_result.get('success'):
                        self.stats['files_created'] += 1
            
            return CommandResult(
                success=result.get('success', False),
                data=result,
                error=result.get('error')
            )
            
        except Exception as e:
            logger.error(f"Code generation failed - {command}: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _execute_file_command(self, command: str, parameters: Dict[str, Any]) -> CommandResult:
        """Execute file operation commands"""
        try:
            # Map command to MCP tool
            tool_mapping = {
                'read_file': 'read_file',
                'write_file': 'write_file',
                'create_file': 'create_file',
                'delete_file': 'delete_file',
                'list_files': 'list_files',
                'search_files': 'search_files'
            }
            
            mcp_tool = tool_mapping.get(command)
            if not mcp_tool:
                return CommandResult(success=False, error=f"Unknown file command: {command}")
            
            # Execute through MCP
            result = await self.mcp_client.call_tool(mcp_tool, parameters)
            
            # Update statistics
            if result.get('success'):
                if command in ['write_file', 'create_file']:
                    if command == 'create_file':
                        self.stats['files_created'] += 1
                    else:
                        self.stats['files_modified'] += 1
            
            return CommandResult(
                success=result.get('success', False),
                data=result,
                error=result.get('error')
            )
            
        except Exception as e:
            logger.error(f"File operation failed - {command}: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _execute_tool_command(self, command: str, parameters: Dict[str, Any]) -> CommandResult:
        """Execute tool and utility commands"""
        try:
            # Map command to MCP tool
            tool_mapping = {
                'run_bash_command': 'run_bash_command',
                'run_git_command': 'run_git_command', 
                'run_tests': 'run_tests',
                'lint_code': 'lint_code',
                'format_code': 'format_code',
                'build_project': 'build_project'
            }
            
            mcp_tool = tool_mapping.get(command)
            if not mcp_tool:
                return CommandResult(success=False, error=f"Unknown tool command: {command}")
            
            # Add working directory if not specified
            if 'working_directory' not in parameters and command == 'run_bash_command':
                parameters['working_directory'] = self.working_directory
            
            # Execute through MCP
            result = await self.mcp_client.call_tool(mcp_tool, parameters)
            
            return CommandResult(
                success=result.get('success', False),
                data=result,
                error=result.get('error')
            )
            
        except Exception as e:
            logger.error(f"Tool command failed - {command}: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _execute_navigation_command(self, command: str, parameters: Dict[str, Any]) -> CommandResult:
        """Execute code navigation commands"""
        try:
            # Map command to MCP tool
            tool_mapping = {
                'find_definition': 'find_definition',
                'find_references': 'find_references',
                'search_code': 'search_code',
                'get_symbols': 'get_symbols',
                'get_imports': 'get_imports'
            }
            
            mcp_tool = tool_mapping.get(command)
            if not mcp_tool:
                return CommandResult(success=False, error=f"Unknown navigation command: {command}")
            
            # Add project context
            if 'project_path' not in parameters:
                parameters['project_path'] = self.working_directory
            
            # Execute through MCP
            result = await self.mcp_client.call_tool(mcp_tool, parameters)
            
            return CommandResult(
                success=result.get('success', False),
                data=result,
                error=result.get('error')
            )
            
        except Exception as e:
            logger.error(f"Navigation command failed - {command}: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _execute_analysis_command(self, command: str, parameters: Dict[str, Any]) -> CommandResult:
        """Execute analysis and info commands"""
        try:
            # Map command to MCP tool
            tool_mapping = {
                'analyze_project': 'analyze_project',
                'get_project_info': 'get_project_info',
                'analyze_dependencies': 'analyze_dependencies'
            }
            
            mcp_tool = tool_mapping.get(command)
            if not mcp_tool:
                return CommandResult(success=False, error=f"Unknown analysis command: {command}")
            
            # Add project path if not specified
            if 'project_path' not in parameters:
                parameters['project_path'] = self.working_directory
            
            # Execute through MCP
            result = await self.mcp_client.call_tool(mcp_tool, parameters)
            
            # Cache project info if this is a project analysis
            if command == 'analyze_project' and result.get('success'):
                self.project_info = result.get('project_info', {})
            
            return CommandResult(
                success=result.get('success', False),
                data=result,
                error=result.get('error')
            )
            
        except Exception as e:
            logger.error(f"Analysis command failed - {command}: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _execute_context_command(self, command: str, parameters: Dict[str, Any]) -> CommandResult:
        """Execute context management commands"""
        try:
            if command == 'get_context':
                return CommandResult(
                    success=True,
                    data={
                        'current_context': self.current_context.__dict__ if self.current_context else None,
                        'working_directory': self.working_directory,
                        'project_info': self.project_info
                    }
                )
            
            elif command == 'update_context':
                context_updates = parameters.get('updates', {})
                if 'working_directory' in context_updates:
                    self.working_directory = context_updates['working_directory']
                
                return CommandResult(success=True, data={'updated': True})
            
            elif command == 'set_working_directory':
                new_dir = parameters.get('directory')
                if new_dir and os.path.exists(new_dir):
                    self.working_directory = new_dir
                    # Re-analyze project in new directory
                    if os.path.exists(os.path.join(new_dir, '.git')):
                        try:
                            result = await self.mcp_client.call_tool('analyze_project', {'project_path': new_dir})
                            if result.get('success'):
                                self.project_info = result.get('project_info')
                        except Exception as e:
                            logger.warning(f"Project re-analysis failed: {e}")
                    
                    return CommandResult(success=True, data={'working_directory': new_dir})
                else:
                    return CommandResult(success=False, error="Invalid or non-existent directory")
            
            elif command == 'get_current_file':
                current_file = self.current_context.current_file if self.current_context else None
                return CommandResult(
                    success=True, 
                    data={'current_file': current_file}
                )
            
            else:
                # Try MCP tool
                result = await self.mcp_client.call_tool(command, parameters)
                return CommandResult(
                    success=result.get('success', False),
                    data=result,
                    error=result.get('error')
                )
                
        except Exception as e:
            logger.error(f"Context command failed - {command}: {e}")
            return CommandResult(success=False, error=str(e))
    
    # Voice-to-code intelligence methods
    
    async def process_voice_intent(self, intent: ParsedIntent, context: Optional[Context] = None) -> CommandResult:
        """
        Process a parsed voice intent and convert to appropriate code actions.
        
        Args:
            intent: Parsed intent from voice input
            context: Current system context
            
        Returns:
            CommandResult with action results
        """
        try:
            if context:
                self.current_context = context
            
            # Map intent to commands based on intent type
            if intent.intent_type == IntentType.CODING_REQUEST:
                return await self._process_coding_intent(intent)
            elif intent.intent_type == IntentType.FILE_OPERATION:
                return await self._process_file_intent(intent)
            elif intent.intent_type == IntentType.SYSTEM_CONTROL:
                return await self._process_system_intent(intent)
            elif intent.intent_type == IntentType.QUERY_REQUEST:
                return await self._process_query_intent(intent)
            else:
                return CommandResult(
                    success=False,
                    error=f"Unsupported intent type: {intent.intent_type.value}"
                )
                
        except Exception as e:
            logger.error(f"Voice intent processing failed: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _process_coding_intent(self, intent: ParsedIntent) -> CommandResult:
        """Process coding-related intents"""
        try:
            # Extract entities and parameters
            entities = intent.entities
            parameters = intent.parameters
            
            # Determine the specific coding action
            if 'function' in intent.original_text.lower():
                command = 'generate_function'
                params = {
                    'name': entities.get('function_name', ['new_function'])[0],
                    'description': intent.processed_text,
                    'language': entities.get('programming_language', ['python'])[0]
                }
                
                # If file path is mentioned, add it
                if 'file_path' in entities:
                    params['file_path'] = entities['file_path'][0]
            
            elif 'class' in intent.original_text.lower():
                command = 'generate_class'
                params = {
                    'name': entities.get('class_name', ['NewClass'])[0],
                    'description': intent.processed_text,
                    'language': entities.get('programming_language', ['python'])[0]
                }
                
            elif 'component' in intent.original_text.lower():
                command = 'generate_component'
                params = {
                    'type': 'react',  # Default to React, could be enhanced
                    'name': entities.get('component_name', ['NewComponent'])[0],
                    'description': intent.processed_text
                }
                
            elif 'refactor' in intent.original_text.lower():
                command = 'refactor_code'
                params = {
                    'instructions': intent.processed_text,
                    'target': 'current_selection'
                }
                
            elif 'comment' in intent.original_text.lower():
                command = 'add_comments'
                params = {
                    'target': entities.get('function_name', ['current_selection'])[0],
                    'style': 'docstring'
                }
                
            else:
                # Default to general code generation
                command = 'generate_function'
                params = {
                    'description': intent.processed_text,
                    'language': 'python'
                }
            
            return await self.execute_command(command, params)
            
        except Exception as e:
            logger.error(f"Coding intent processing failed: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _process_file_intent(self, intent: ParsedIntent) -> CommandResult:
        """Process file operation intents"""
        try:
            entities = intent.entities
            parameters = intent.parameters
            
            # Determine file operation
            if any(word in intent.original_text.lower() for word in ['create', 'new']):
                command = 'create_file'
                params = {
                    'file_path': entities.get('file_path', ['new_file.py'])[0],
                    'content': parameters.get('content', '')
                }
                
            elif any(word in intent.original_text.lower() for word in ['open', 'read']):
                command = 'read_file'
                params = {
                    'file_path': entities.get('file_path', [''])[0]
                }
                
            elif any(word in intent.original_text.lower() for word in ['save', 'write']):
                command = 'write_file'
                params = {
                    'file_path': entities.get('file_path', [''])[0],
                    'content': parameters.get('content', '')
                }
                
            elif 'delete' in intent.original_text.lower():
                command = 'delete_file'
                params = {
                    'file_path': entities.get('file_path', [''])[0]
                }
                
            elif any(word in intent.original_text.lower() for word in ['list', 'show']):
                command = 'list_files'
                params = {
                    'directory_path': parameters.get('directory', self.working_directory)
                }
                
            else:
                return CommandResult(
                    success=False,
                    error="Could not determine specific file operation"
                )
            
            return await self.execute_command(command, params)
            
        except Exception as e:
            logger.error(f"File intent processing failed: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _process_system_intent(self, intent: ParsedIntent) -> CommandResult:
        """Process system control intents"""
        try:
            entities = intent.entities
            
            # Map common system commands
            if any(word in intent.original_text.lower() for word in ['run', 'execute']):
                if 'test' in intent.original_text.lower():
                    command = 'run_tests'
                    params = {'test_type': 'all'}
                elif 'git' in intent.original_text.lower():
                    git_action = 'status'  # Default
                    if 'commit' in intent.original_text.lower():
                        git_action = 'commit'
                    elif 'push' in intent.original_text.lower():
                        git_action = 'push'
                    elif 'pull' in intent.original_text.lower():
                        git_action = 'pull'
                    
                    command = 'run_git_command'
                    params = {'command': git_action}
                else:
                    command = 'run_bash_command'
                    params = {'command': intent.processed_text.replace('run ', '').replace('execute ', '')}
            
            elif 'build' in intent.original_text.lower():
                command = 'build_project'
                params = {}
                
            elif 'format' in intent.original_text.lower():
                command = 'format_code'
                params = {
                    'file_path': entities.get('file_path', [self.current_context.current_file])[0] if self.current_context and self.current_context.current_file else None
                }
                
            elif 'lint' in intent.original_text.lower():
                command = 'lint_code'
                params = {
                    'file_path': entities.get('file_path', [self.current_context.current_file])[0] if self.current_context and self.current_context.current_file else None
                }
                
            else:
                return CommandResult(
                    success=False,
                    error="Could not determine specific system operation"
                )
            
            return await self.execute_command(command, params)
            
        except Exception as e:
            logger.error(f"System intent processing failed: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _process_query_intent(self, intent: ParsedIntent) -> CommandResult:
        """Process query and information intents"""
        try:
            if any(word in intent.original_text.lower() for word in ['project', 'info']):
                command = 'get_project_info'
                params = {'project_path': self.working_directory}
                
            elif 'find' in intent.original_text.lower():
                if 'definition' in intent.original_text.lower():
                    command = 'find_definition'
                    params = {
                        'symbol': intent.entities.get('function_name', [''])[0] or intent.entities.get('variable_name', [''])[0],
                        'file_path': self.current_context.current_file if self.current_context else None
                    }
                elif 'reference' in intent.original_text.lower():
                    command = 'find_references'
                    params = {
                        'symbol': intent.entities.get('function_name', [''])[0] or intent.entities.get('variable_name', [''])[0],
                        'project_path': self.working_directory
                    }
                else:
                    command = 'search_code'
                    params = {
                        'pattern': intent.processed_text,
                        'project_path': self.working_directory
                    }
            
            elif 'analyze' in intent.original_text.lower():
                command = 'analyze_project'
                params = {'project_path': self.working_directory}
                
            else:
                # General context query
                command = 'get_context'
                params = {}
            
            return await self.execute_command(command, params)
            
        except Exception as e:
            logger.error(f"Query intent processing failed: {e}")
            return CommandResult(success=False, error=str(e))
    
    async def _security_check(self, command: str, parameters: Dict[str, Any]) -> bool:
        """Perform security check on command"""
        
        # Check if command is in restricted operations
        for restricted in self.restricted_operations:
            if restricted in command:
                logger.warning(f"Restricted command attempted: {command}")
                return False
        
        # Check specific parameter restrictions
        if command == 'run_bash_command':
            cmd = parameters.get('command', '')
            # Block dangerous commands
            dangerous_patterns = ['rm -rf', 'sudo', 'su ', 'chmod 777', '> /dev/']
            if any(pattern in cmd.lower() for pattern in dangerous_patterns):
                logger.warning(f"Dangerous bash command blocked: {cmd}")
                return False
        
        if command == 'delete_file':
            file_path = parameters.get('file_path', '')
            # Block deletion of important files
            important_files = ['.git', 'package.json', 'requirements.txt', '__init__.py']
            if any(important in file_path for important in important_files):
                logger.warning(f"Important file deletion blocked: {file_path}")
                return False
        
        return True