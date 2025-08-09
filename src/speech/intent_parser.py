"""Intent Parsing Module

This module provides intent recognition and parsing functionality for speech inputs,
with specific optimizations for programming and development contexts.
"""

import asyncio
import re
import time
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime
import logging

from loguru import logger

from ..core.base_adapter import BaseAdapter, AdapterError
from ..core.types import CommandResult
from ..core.event_system import EventSystem, Event
from .types import (
    ParsedIntent, IntentType, SpeechEventType, 
    RecognitionResult, PROGRAMMING_KEYWORDS
)


class IntentParsingError(AdapterError):
    """Intent parsing specific error"""
    pass


class IntentParser:
    """
    Advanced intent recognition system for programming-focused voice commands.
    
    Features:
    - Rule-based intent classification
    - Programming context awareness
    - Entity extraction and parameter parsing
    - Context-sensitive command suggestions
    - Multi-language support (Chinese-English)
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        """
        Initialize the intent parser.
        
        Args:
            event_system: Event system for notifications
        """
        self.event_system = event_system
        
        # Intent classification patterns
        self._intent_patterns = self._initialize_intent_patterns()
        
        # Entity extraction patterns
        self._entity_patterns = self._initialize_entity_patterns()
        
        # Programming context keywords
        self._programming_contexts = self._initialize_programming_contexts()
        
        # Command mapping
        self._command_mappings = self._initialize_command_mappings()
        
        # Statistics
        self._stats = {
            'total_parsed': 0,
            'successful_parsed': 0,
            'intent_distribution': {},
            'average_confidence': 0.0,
            'errors': 0
        }
        
        logger.info("IntentParser initialized with rule-based classification")
    
    def _initialize_intent_patterns(self) -> Dict[IntentType, List[Dict[str, Any]]]:
        """Initialize intent classification patterns."""
        return {
            IntentType.CODING_REQUEST: [
                {
                    'patterns': [
                        r'\b(create|write|generate|code|implement|build)\b.*\b(function|method|class|module|script|program)\b',
                        r'\b(add|insert|create)\b.*\b(code|line|block|comment)\b',
                        r'\b(refactor|optimize|improve|fix|debug)\b.*\b(code|function|method)\b',
                        r'\b(写|创建|生成|编写|实现)\b.*(函数|方法|类|模块|脚本|程序|代码)',
                        r'\b(添加|插入|创建)\b.*(代码|行|块|注释)',
                        r'\b(重构|优化|改进|修复|调试)\b.*(代码|函数|方法)'
                    ],
                    'weight': 1.0
                },
                {
                    'patterns': [
                        r'\bdef\b|\bclass\b|\bfunction\b|\bmethod\b',
                        r'\bpython\b|\bjavascript\b|\bjava\b|\bc\+\+\b',
                        r'\b(import|from|include|require)\b',
                        r'\b(for|while|if|else|elif|switch|case)\b.*\bloop\b'
                    ],
                    'weight': 0.8
                }
            ],
            
            IntentType.FILE_OPERATION: [
                {
                    'patterns': [
                        r'\b(open|create|save|delete|rename|move|copy)\b.*\b(file|folder|directory)\b',
                        r'\b(read|write|edit|modify)\b.*\b(file|document|text)\b',
                        r'\b(打开|创建|保存|删除|重命名|移动|复制)\b.*(文件|文件夹|目录)',
                        r'\b(读取|写入|编辑|修改)\b.*(文件|文档|文本)'
                    ],
                    'weight': 1.0
                },
                {
                    'patterns': [
                        r'\.(py|js|html|css|txt|md|json|xml|csv)\b',
                        r'\b(download|upload|backup|sync)\b',
                        r'\bpath\b|\bdirectory\b|\bfolder\b'
                    ],
                    'weight': 0.7
                }
            ],
            
            IntentType.SYSTEM_CONTROL: [
                {
                    'patterns': [
                        r'\b(start|stop|restart|run|execute|launch|kill)\b.*\b(program|application|service|process)\b',
                        r'\b(shutdown|reboot|sleep|wake|lock|unlock)\b.*\b(computer|system|machine)\b',
                        r'\b(启动|停止|重启|运行|执行|启动|杀死)\b.*(程序|应用|服务|进程)',
                        r'\b(关机|重启|睡眠|唤醒|锁定|解锁)\b.*(电脑|系统|机器)'
                    ],
                    'weight': 1.0
                }
            ],
            
            IntentType.APPLICATION_CONTROL: [
                {
                    'patterns': [
                        r'\b(open|close|minimize|maximize|switch to)\b.*\b(chrome|firefox|code|vscode|pycharm|notepad)\b',
                        r'\b(click|press|type|scroll|drag)\b',
                        r'\b(copy|paste|cut|undo|redo|select)\b',
                        r'\b(打开|关闭|最小化|最大化|切换到)\b.*(浏览器|编辑器|代码|记事本)'
                    ],
                    'weight': 1.0
                }
            ],
            
            IntentType.QUERY_REQUEST: [
                {
                    'patterns': [
                        r'\b(what|how|why|when|where|which|who)\b',
                        r'\b(explain|describe|tell me|show me|find|search)\b',
                        r'\b(help|assist|guide|tutorial|documentation)\b',
                        r'\b(什么|如何|为什么|何时|哪里|哪个|谁)',
                        r'\b(解释|描述|告诉我|显示|找到|搜索)',
                        r'\b(帮助|协助|指导|教程|文档)'
                    ],
                    'weight': 1.0
                }
            ],
            
            IntentType.NAVIGATION_REQUEST: [
                {
                    'patterns': [
                        r'\b(go to|navigate to|move to|switch to)\b',
                        r'\b(up|down|left|right|next|previous|back|forward)\b',
                        r'\b(line|column|page|tab|window|panel)\b.*\b(number|position)\b',
                        r'\b(转到|导航到|移动到|切换到)',
                        r'\b(上|下|左|右|下一个|上一个|返回|前进)'
                    ],
                    'weight': 1.0
                }
            ]
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns."""
        return {
            'file_path': [
                r'["\']([^"\']+\.(py|js|html|css|txt|md|json|xml|csv))["\']',
                r'\b([\w\-\.]+\.(py|js|html|css|txt|md|json|xml|csv))\b',
                r'["\']([A-Za-z]:\\[^"\']*)["\']',
                r'["\']([/][^"\']*)["\']'
            ],
            'function_name': [
                r'\b(function|def|method)\s+(\w+)',
                r'\b(\w+)\s*\(',
                r'\b(函数|方法)\s*(\w+)'
            ],
            'variable_name': [
                r'\b(variable|var|let|const)\s+(\w+)',
                r'\b(变量)\s*(\w+)'
            ],
            'class_name': [
                r'\b(class)\s+(\w+)',
                r'\b(类)\s*(\w+)'
            ],
            'line_number': [
                r'\b(line|行)\s*(\d+)',
                r'\b(\d+)\s*(line|行)'
            ],
            'application_name': [
                r'\b(chrome|firefox|safari|edge|vscode|code|pycharm|intellij|notepad|word|excel|powerpoint)\b'
            ],
            'programming_language': [
                r'\b(python|javascript|java|c\+\+|c#|go|rust|php|ruby|swift|kotlin|typescript)\b'
            ],
            'action_verb': [
                r'\b(create|write|generate|delete|modify|update|refactor|optimize|debug|fix|run|execute|test)\b',
                r'\b(创建|写|生成|删除|修改|更新|重构|优化|调试|修复|运行|执行|测试)\b'
            ]
        }
    
    def _initialize_programming_contexts(self) -> Dict[str, List[str]]:
        """Initialize programming context indicators."""
        return {
            'python': [
                'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while',
                'try', 'except', 'finally', 'with', 'as', 'lambda', 'return', 'yield'
            ],
            'javascript': [
                'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'do',
                'switch', 'case', 'default', 'try', 'catch', 'finally', 'async', 'await'
            ],
            'web': [
                'html', 'css', 'javascript', 'react', 'vue', 'angular', 'node', 'express',
                'div', 'span', 'class', 'id', 'style', 'script', 'link'
            ],
            'data': [
                'database', 'sql', 'query', 'select', 'insert', 'update', 'delete',
                'table', 'column', 'row', 'index', 'json', 'csv', 'xml', 'api'
            ],
            'system': [
                'command', 'terminal', 'shell', 'bash', 'powershell', 'cmd',
                'process', 'service', 'registry', 'path', 'environment'
            ]
        }
    
    def _initialize_command_mappings(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize command mappings for different intent types."""
        return {
            'coding_request': [
                {
                    'pattern': r'create.*function',
                    'commands': ['create_function', 'add_function'],
                    'adapters': ['claude_code']
                },
                {
                    'pattern': r'write.*class',
                    'commands': ['create_class', 'generate_class'],
                    'adapters': ['claude_code']
                },
                {
                    'pattern': r'refactor.*code',
                    'commands': ['refactor_code', 'optimize_code'],
                    'adapters': ['claude_code']
                }
            ],
            'file_operation': [
                {
                    'pattern': r'open.*file',
                    'commands': ['open_file', 'edit_file'],
                    'adapters': ['system', 'ide']
                },
                {
                    'pattern': r'save.*file',
                    'commands': ['save_file'],
                    'adapters': ['system', 'ide']
                },
                {
                    'pattern': r'create.*file',
                    'commands': ['create_file', 'new_file'],
                    'adapters': ['system', 'claude_code']
                }
            ],
            'system_control': [
                {
                    'pattern': r'start.*program',
                    'commands': ['start_application', 'launch_program'],
                    'adapters': ['system']
                },
                {
                    'pattern': r'run.*command',
                    'commands': ['execute_command', 'run_shell_command'],
                    'adapters': ['system']
                }
            ],
            'application_control': [
                {
                    'pattern': r'open.*browser',
                    'commands': ['open_browser', 'launch_browser'],
                    'adapters': ['system', 'office']
                },
                {
                    'pattern': r'switch.*tab',
                    'commands': ['switch_tab', 'change_tab'],
                    'adapters': ['system', 'office']
                }
            ]
        }
    
    async def parse_intent(self, 
                          recognition_result: RecognitionResult) -> Optional[ParsedIntent]:
        """
        Parse intent from speech recognition result.
        
        Args:
            recognition_result: Speech recognition result to parse
            
        Returns:
            Parsed intent or None if parsing failed
        """
        try:
            start_time = time.time()
            
            # Emit parsing started event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.INTENT_PARSED.value,
                    data={'text': recognition_result.text},
                    source='intent_parser'
                ))
            
            # Preprocess text
            processed_text = self._preprocess_text(recognition_result.text)
            
            # Classify intent
            intent_type, intent_confidence = self._classify_intent(processed_text)
            
            # Extract entities
            entities = self._extract_entities(processed_text)
            
            # Determine programming context
            programming_context = self._detect_programming_context(processed_text)
            
            # Generate command suggestions
            suggested_commands = self._suggest_commands(intent_type, processed_text, entities)
            
            # Determine target adapters
            target_adapters = self._determine_target_adapters(intent_type, entities, programming_context)
            
            # Create parsed intent
            parsed_intent = ParsedIntent(
                original_text=recognition_result.text,
                processed_text=processed_text,
                intent_type=intent_type,
                confidence=intent_confidence,
                entities=entities,
                parameters=self._extract_parameters(processed_text, entities),
                context_keywords=self._extract_context_keywords(processed_text),
                programming_context=programming_context,
                suggested_commands=suggested_commands,
                target_adapters=target_adapters
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(parsed_intent, processing_time)
            
            # Emit parsing completed event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.INTENT_PARSED.value,
                    data={
                        'intent_type': intent_type.value,
                        'confidence': intent_confidence,
                        'entities': entities,
                        'processing_time': processing_time
                    },
                    source='intent_parser'
                ))
            
            return parsed_intent
            
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            
            # Emit parsing failed event
            if self.event_system:
                await self.event_system.emit(Event(
                    event_type=SpeechEventType.INTENT_PARSE_FAILED.value,
                    data={'error': str(e), 'text': recognition_result.text},
                    source='intent_parser'
                ))
            
            self._stats['errors'] += 1
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for intent classification."""
        # Convert to lowercase for processing
        processed = text.lower()
        
        # Remove extra whitespace
        processed = ' '.join(processed.split())
        
        # Handle common speech-to-text errors
        processed = self._fix_common_errors(processed)
        
        return processed
    
    def _fix_common_errors(self, text: str) -> str:
        """Fix common speech-to-text errors."""
        # Common corrections for programming terms
        corrections = {
            'death': 'def',
            'deaf': 'def',
            'depth': 'def',
            'class room': 'class',
            'funk tion': 'function',
            'funk shun': 'function',
            'very able': 'variable',
            'vair able': 'variable',
            'import ant': 'important',
            'import': 'import',
            'if else': 'if else',
            'for loop': 'for loop',
            'while loop': 'while loop'
        }
        
        processed = text
        for error, correction in corrections.items():
            processed = processed.replace(error, correction)
        
        return processed
    
    def _classify_intent(self, text: str) -> Tuple[IntentType, float]:
        """
        Classify the intent type based on text patterns.
        
        Args:
            text: Processed text to classify
            
        Returns:
            Tuple of (intent_type, confidence)
        """
        best_intent = IntentType.UNKNOWN
        best_score = 0.0
        
        for intent_type, pattern_groups in self._intent_patterns.items():
            intent_score = 0.0
            
            for group in pattern_groups:
                group_score = 0.0
                weight = group.get('weight', 1.0)
                
                for pattern in group['patterns']:
                    if re.search(pattern, text, re.IGNORECASE):
                        group_score = weight
                        break
                
                intent_score = max(intent_score, group_score)
            
            if intent_score > best_score:
                best_score = intent_score
                best_intent = intent_type
        
        # Apply confidence scaling
        confidence = min(0.95, best_score) if best_score > 0.3 else 0.1
        
        return best_intent, confidence
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text."""
        entities = {}
        
        for entity_type, patterns in self._entity_patterns.items():
            matches = []
            
            for pattern in patterns:
                found_matches = re.findall(pattern, text, re.IGNORECASE)
                if found_matches:
                    if isinstance(found_matches[0], tuple):
                        # Extract from tuple matches
                        matches.extend([match[0] if match[0] else match[1] 
                                      for match in found_matches])
                    else:
                        matches.extend(found_matches)
            
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def _detect_programming_context(self, text: str) -> bool:
        """Detect if the text is in a programming context."""
        programming_indicators = 0
        total_words = len(text.split())
        
        if total_words == 0:
            return False
        
        for context_type, keywords in self._programming_contexts.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    programming_indicators += 1
        
        # Consider programming context if more than 20% of words are programming-related
        return (programming_indicators / total_words) > 0.2
    
    def _extract_parameters(self, text: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from text and entities."""
        parameters = {}
        
        # Extract file paths
        if 'file_path' in entities:
            parameters['file_path'] = entities['file_path'][0]
        
        # Extract function/method names
        if 'function_name' in entities:
            parameters['function_name'] = entities['function_name'][0]
        
        # Extract variable names
        if 'variable_name' in entities:
            parameters['variable_name'] = entities['variable_name'][0]
        
        # Extract class names
        if 'class_name' in entities:
            parameters['class_name'] = entities['class_name'][0]
        
        # Extract line numbers
        if 'line_number' in entities:
            try:
                parameters['line_number'] = int(entities['line_number'][0])
            except (ValueError, IndexError):
                pass
        
        # Extract programming language
        if 'programming_language' in entities:
            parameters['language'] = entities['programming_language'][0]
        
        # Extract application names
        if 'application_name' in entities:
            parameters['application'] = entities['application_name'][0]
        
        # Extract action verbs
        if 'action_verb' in entities:
            parameters['action'] = entities['action_verb'][0]
        
        return parameters
    
    def _extract_context_keywords(self, text: str) -> List[str]:
        """Extract context keywords from text."""
        keywords = []
        words = text.split()
        
        # Add programming keywords
        for word in words:
            if any(word in lang_keywords for lang_keywords in PROGRAMMING_KEYWORDS.values()):
                keywords.append(word)
        
        # Add technical terms
        technical_terms = [
            'api', 'database', 'server', 'client', 'framework', 'library',
            'module', 'package', 'dependency', 'version', 'config', 'settings',
            'debug', 'test', 'deploy', 'build', 'compile', 'execute', 'run'
        ]
        
        for word in words:
            if word in technical_terms:
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def _suggest_commands(self, 
                         intent_type: IntentType, 
                         text: str, 
                         entities: Dict[str, Any]) -> List[str]:
        """Suggest commands based on intent and entities."""
        suggestions = []
        
        # Get command mappings for intent type
        mappings_key = intent_type.value
        if mappings_key in self._command_mappings:
            for mapping in self._command_mappings[mappings_key]:
                pattern = mapping['pattern']
                if re.search(pattern, text, re.IGNORECASE):
                    suggestions.extend(mapping['commands'])
        
        # Add entity-based suggestions
        if 'file_path' in entities:
            suggestions.extend(['open_file', 'edit_file', 'read_file'])
        
        if 'function_name' in entities:
            suggestions.extend(['create_function', 'edit_function', 'call_function'])
        
        if 'class_name' in entities:
            suggestions.extend(['create_class', 'edit_class'])
        
        if 'application_name' in entities:
            suggestions.extend(['open_application', 'switch_to_application'])
        
        return list(set(suggestions))  # Remove duplicates
    
    def _determine_target_adapters(self, 
                                  intent_type: IntentType, 
                                  entities: Dict[str, Any], 
                                  programming_context: bool) -> List[str]:
        """Determine which adapters should handle this intent."""
        adapters = []
        
        # Base adapter selection by intent type
        intent_adapter_map = {
            IntentType.CODING_REQUEST: ['claude_code'],
            IntentType.FILE_OPERATION: ['system', 'claude_code'],
            IntentType.SYSTEM_CONTROL: ['system'],
            IntentType.APPLICATION_CONTROL: ['system', 'office'],
            IntentType.QUERY_REQUEST: ['claude_code'],
            IntentType.NAVIGATION_REQUEST: ['system', 'ide']
        }
        
        adapters.extend(intent_adapter_map.get(intent_type, []))
        
        # Add context-specific adapters
        if programming_context:
            adapters.extend(['claude_code', 'ide'])
        
        if 'application_name' in entities:
            app = entities['application_name'][0].lower()
            if app in ['chrome', 'firefox', 'safari', 'edge']:
                adapters.append('office')
            elif app in ['vscode', 'code', 'pycharm', 'intellij']:
                adapters.append('ide')
        
        if 'programming_language' in entities:
            adapters.append('claude_code')
        
        return list(set(adapters))  # Remove duplicates
    
    def _update_stats(self, parsed_intent: ParsedIntent, processing_time: float) -> None:
        """Update parsing statistics."""
        self._stats['total_parsed'] += 1
        
        if parsed_intent.intent_type != IntentType.UNKNOWN:
            self._stats['successful_parsed'] += 1
            
            # Update intent distribution
            intent_name = parsed_intent.intent_type.value
            self._stats['intent_distribution'][intent_name] = \
                self._stats['intent_distribution'].get(intent_name, 0) + 1
            
            # Update average confidence
            old_avg = self._stats['average_confidence']
            count = self._stats['successful_parsed']
            self._stats['average_confidence'] = \
                (old_avg * (count - 1) + parsed_intent.confidence) / count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        return {
            **self._stats,
            'success_rate': (self._stats['successful_parsed'] / max(1, self._stats['total_parsed'])) * 100
        }
    
    async def get_command_suggestions(self, 
                                    text: str,
                                    context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get command suggestions for given text without full parsing.
        
        Args:
            text: Input text
            context: Optional context information
            
        Returns:
            List of suggested commands
        """
        processed_text = self._preprocess_text(text)
        intent_type, _ = self._classify_intent(processed_text)
        entities = self._extract_entities(processed_text)
        
        return self._suggest_commands(intent_type, processed_text, entities)


class IntentParserAdapter(BaseAdapter):
    """Intent Parser Adapter for integration with the core system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intent parser adapter."""
        super().__init__(config)
        self._parser = IntentParser()
    
    @property
    def adapter_id(self) -> str:
        return "intent_parser"
    
    @property
    def name(self) -> str:
        return "Intent Parser Adapter"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Parses user intents from speech with programming context awareness"
    
    @property
    def supported_commands(self) -> List[str]:
        return [
            "parse_intent",
            "suggest_commands",
            "get_statistics",
            "classify_text"
        ]
    
    async def initialize(self) -> bool:
        """Initialize the intent parser."""
        try:
            self._update_status("available")
            logger.info("Intent parser adapter initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize intent parser adapter: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up the intent parser."""
        pass
    
    async def execute_command(self, 
                            command: str, 
                            parameters: Dict[str, Any],
                            context: Optional[Any] = None) -> CommandResult:
        """Execute a parsing command."""
        try:
            if command == "parse_intent":
                recognition_result = parameters.get('recognition_result')
                if not recognition_result:
                    return CommandResult(
                        success=False,
                        error="recognition_result parameter required"
                    )
                
                # Convert dict to RecognitionResult if needed
                if isinstance(recognition_result, dict):
                    from .types import RecognitionResult
                    recognition_result = RecognitionResult(**recognition_result)
                
                parsed_intent = await self._parser.parse_intent(recognition_result)
                return CommandResult(
                    success=parsed_intent is not None,
                    data={'parsed_intent': parsed_intent.__dict__ if parsed_intent else None}
                )
            
            elif command == "suggest_commands":
                text = parameters.get('text', '')
                if not text:
                    return CommandResult(
                        success=False,
                        error="text parameter required"
                    )
                
                suggestions = await self._parser.get_command_suggestions(text, context)
                return CommandResult(
                    success=True,
                    data={'suggestions': suggestions}
                )
            
            elif command == "get_statistics":
                return CommandResult(
                    success=True,
                    data=self._parser.get_statistics()
                )
            
            elif command == "classify_text":
                text = parameters.get('text', '')
                if not text:
                    return CommandResult(
                        success=False,
                        error="text parameter required"
                    )
                
                processed = self._parser._preprocess_text(text)
                intent_type, confidence = self._parser._classify_intent(processed)
                entities = self._parser._extract_entities(processed)
                programming_context = self._parser._detect_programming_context(processed)
                
                return CommandResult(
                    success=True,
                    data={
                        'intent_type': intent_type.value,
                        'confidence': confidence,
                        'entities': entities,
                        'programming_context': programming_context
                    }
                )
            
            else:
                return CommandResult(
                    success=False,
                    error=f"Unknown command: {command}"
                )
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return CommandResult(
                success=False,
                error=str(e)
            )
    
    async def is_available(self) -> bool:
        """Check if the adapter is available."""
        return self._status == "available"
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current adapter state."""
        return {
            'status': self._status,
            'statistics': self._parser.get_statistics()
        }
    
    async def get_command_suggestions(self, context: Optional[Any] = None) -> List[str]:
        """Get command suggestions based on context."""
        return ["parse_intent", "suggest_commands", "get_statistics", "classify_text"]