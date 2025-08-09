"""Enhanced Intent Parser for Claude Code Integration

This module extends the base intent parser with advanced programming-specific
intent recognition, entity extraction, and command suggestion capabilities.

Features:
- Programming-focused intent classification
- Advanced code entity extraction (functions, classes, variables, etc.)
- Context-aware command suggestions
- Natural language to code parameter mapping
- Multi-language programming syntax understanding
"""

import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from loguru import logger

# Import speech modules only when available
try:
    from ..speech.intent_parser import IntentParser
    from ..speech.types import ParsedIntent, IntentType, RecognitionResult
except ImportError:
    # Fallback definitions for testing
    from enum import Enum
    from dataclasses import dataclass
    from typing import Dict, Any
    
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
    
    @dataclass
    class RecognitionResult:
        text: str
        confidence: float
        processing_time: float
    
    class IntentParser:
        def __init__(self, event_system=None):
            pass
        async def parse_intent(self, recognition_result):
            return None

from ..core.event_system import EventSystem, Event
from ..core.types import Context


class CodeIntentType(Enum):
    """Extended intent types for programming tasks"""
    # Code generation
    GENERATE_FUNCTION = "generate_function"
    GENERATE_CLASS = "generate_class"
    GENERATE_COMPONENT = "generate_component"
    GENERATE_MODULE = "generate_module"
    GENERATE_TEST = "generate_test"
    
    # Code modification
    REFACTOR_CODE = "refactor_code"
    ADD_COMMENTS = "add_comments"
    FIX_CODE = "fix_code"
    OPTIMIZE_CODE = "optimize_code"
    ADD_TYPE_HINTS = "add_type_hints"
    EXTRACT_METHOD = "extract_method"
    
    # Project operations
    CREATE_PROJECT = "create_project"
    SETUP_ENVIRONMENT = "setup_environment"
    INSTALL_DEPENDENCIES = "install_dependencies"
    BUILD_PROJECT = "build_project"
    
    # Version control
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"
    GIT_PULL = "git_pull"
    GIT_STATUS = "git_status"
    GIT_BRANCH = "git_branch"
    
    # Testing and quality
    RUN_TESTS = "run_tests"
    DEBUG_CODE = "debug_code"
    LINT_CODE = "lint_code"
    FORMAT_CODE = "format_code"
    
    # Code navigation
    FIND_DEFINITION = "find_definition"
    FIND_REFERENCES = "find_references"
    SEARCH_CODE = "search_code"
    NAVIGATE_TO = "navigate_to"


@dataclass
class CodeEntity:
    """Represents a code entity extracted from speech"""
    type: str  # function, class, variable, file, etc.
    name: str
    language: Optional[str] = None
    parameters: Optional[List[str]] = None
    context: Optional[str] = None
    confidence: float = 1.0


@dataclass
class EnhancedCodeIntent(ParsedIntent):
    """Enhanced intent with programming-specific information"""
    code_intent_type: Optional[CodeIntentType] = None
    code_entities: List[CodeEntity] = None
    target_language: Optional[str] = None
    framework: Optional[str] = None
    code_complexity: str = "simple"  # simple, medium, complex
    requires_dependencies: bool = False
    estimated_lines: Optional[int] = None
    
    def __post_init__(self):
        if self.code_entities is None:
            self.code_entities = []


class EnhancedIntentParser(IntentParser):
    """Enhanced intent parser with Claude Code integration"""
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        """
        Initialize enhanced intent parser.
        
        Args:
            event_system: Event system for notifications
        """
        super().__init__(event_system)
        
        # Enhanced programming patterns
        self._code_intent_patterns = self._initialize_code_intent_patterns()
        self._programming_frameworks = self._initialize_frameworks()
        self._code_templates = self._initialize_code_templates()
        self._complexity_indicators = self._initialize_complexity_indicators()
        
        # Code entity extractors
        self._code_entity_extractors = self._initialize_entity_extractors()
        
        logger.info("Enhanced intent parser initialized for Claude Code integration")
    
    def _initialize_code_intent_patterns(self) -> Dict[CodeIntentType, List[Dict[str, Any]]]:
        """Initialize programming-specific intent patterns"""
        return {
            # Code generation patterns
            CodeIntentType.GENERATE_FUNCTION: [
                {
                    'patterns': [
                        r'\b(create|write|generate|make)\b.*\b(function|method|def)\b',
                        r'\b(add|implement)\b.*\b(function|method)\b.*\b(that|to|which)\b',
                        r'\b(创建|写|生成|制作)\b.*(函数|方法)',
                        r'写一个.*函数',
                        r'创建.*方法'
                    ],
                    'weight': 1.0,
                    'complexity': 'simple'
                }
            ],
            
            CodeIntentType.GENERATE_CLASS: [
                {
                    'patterns': [
                        r'\b(create|write|generate|make)\b.*\b(class|object)\b',
                        r'\b(define|implement)\b.*\b(class|type)\b',
                        r'\b(创建|写|生成|定义)\b.*(类|对象)',
                        r'写一个.*类',
                        r'创建.*对象'
                    ],
                    'weight': 1.0,
                    'complexity': 'medium'
                }
            ],
            
            CodeIntentType.GENERATE_COMPONENT: [
                {
                    'patterns': [
                        r'\b(create|generate|make)\b.*(react|vue|angular).*\b(component|widget)\b',
                        r'\b(build|develop)\b.*\b(ui|interface)\b.*\b(component|element)\b',
                        r'创建.*(react|vue|angular).*组件',
                        r'生成.*界面.*组件'
                    ],
                    'weight': 1.0,
                    'complexity': 'medium'
                }
            ],
            
            CodeIntentType.REFACTOR_CODE: [
                {
                    'patterns': [
                        r'\b(refactor|restructure|reorganize)\b',
                        r'\b(improve|optimize|clean up)\b.*\b(code|function|method)\b',
                        r'\b(make.*better|simplify)\b.*\b(code|function)\b',
                        r'\b(重构|重组|优化)\b.*(代码|函数|方法)',
                        r'改进.*代码'
                    ],
                    'weight': 1.0,
                    'complexity': 'complex'
                }
            ],
            
            CodeIntentType.ADD_COMMENTS: [
                {
                    'patterns': [
                        r'\b(add|write|generate)\b.*\b(comment|documentation|docstring)\b',
                        r'\b(document)\b.*\b(function|method|class|code)\b',
                        r'\b(添加|写|生成)\b.*(注释|文档)',
                        r'给.*添加注释'
                    ],
                    'weight': 1.0,
                    'complexity': 'simple'
                }
            ],
            
            CodeIntentType.RUN_TESTS: [
                {
                    'patterns': [
                        r'\b(run|execute|start)\b.*\b(test|tests|testing)\b',
                        r'\b(test)\b.*\b(code|function|application)\b',
                        r'\b(check)\b.*\b(functionality|correctness)\b',
                        r'\b(运行|执行|开始)\b.*(测试)',
                        r'测试.*代码'
                    ],
                    'weight': 1.0,
                    'complexity': 'simple'
                }
            ],
            
            CodeIntentType.GIT_COMMIT: [
                {
                    'patterns': [
                        r'\b(commit|save|store)\b.*\b(changes|code|work)\b',
                        r'\b(git commit|check in|save to git)\b',
                        r'\b(提交|保存|存储)\b.*(更改|代码|工作)',
                        r'git.*提交'
                    ],
                    'weight': 1.0,
                    'complexity': 'simple'
                }
            ],
            
            CodeIntentType.FIX_CODE: [
                {
                    'patterns': [
                        r'\b(fix|repair|correct|debug)\b.*\b(error|bug|issue|problem)\b',
                        r'\b(solve|resolve)\b.*\b(problem|issue)\b',
                        r'\b(修复|修理|纠正|调试)\b.*(错误|bug|问题)',
                        r'解决.*问题'
                    ],
                    'weight': 1.0,
                    'complexity': 'complex'
                }
            ]
        }
    
    def _initialize_frameworks(self) -> Dict[str, List[str]]:
        """Initialize framework detection patterns"""
        return {
            'react': ['react', 'jsx', 'tsx', 'next.js', 'gatsby'],
            'vue': ['vue', 'vuejs', 'nuxt'],
            'angular': ['angular', 'typescript', 'ng'],
            'django': ['django', 'python', 'web framework'],
            'flask': ['flask', 'python', 'micro framework'],
            'express': ['express', 'node.js', 'javascript'],
            'spring': ['spring', 'java', 'spring boot'],
            'laravel': ['laravel', 'php'],
            'rails': ['rails', 'ruby', 'ruby on rails']
        }
    
    def _initialize_code_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize code generation templates"""
        return {
            'function': {
                'python': '''def {name}({parameters}):
    """
    {description}
    
    Args:
        {args_description}
    
    Returns:
        {return_description}
    """
    # TODO: Implement this function
    pass
''',
                'javascript': '''/**
 * {description}
 * {param_docs}
 * @returns {{{return_type}}} {return_description}
 */
function {name}({parameters}) {{
    // TODO: Implement this function
}}
''',
                'typescript': '''/**
 * {description}
 * {param_docs}
 * @returns {return_type} {return_description}
 */
function {name}({parameters}): {return_type} {{
    // TODO: Implement this function
}}
'''
            },
            'class': {
                'python': '''class {name}:
    """
    {description}
    
    Attributes:
        {attributes}
    """
    
    def __init__(self{init_params}):
        """
        Initialize {name}.
        
        Args:
            {init_args_description}
        """
        # TODO: Initialize attributes
        pass
''',
                'javascript': '''/**
 * {description}
 */
class {name} {{
    /**
     * Create a {name}.
     * {param_docs}
     */
    constructor({parameters}) {{
        // TODO: Initialize properties
    }}
}}
''',
                'typescript': '''/**
 * {description}
 */
class {name} {{
    {properties}
    
    /**
     * Create a {name}.
     * {param_docs}
     */
    constructor({parameters}) {{
        // TODO: Initialize properties
    }}
}}
'''
            },
            'component': {
                'react': '''import React from 'react';

/**
 * {description}
 * {param_docs}
 */
const {name} = ({{props}}) => {{
    return (
        <div className="{css_class}">
            {{/* TODO: Implement component */}}
            <h1>{name} Component</h1>
        </div>
    );
}};

export default {name};
''',
                'vue': '''<template>
  <div class="{css_class}">
    <!-- TODO: Implement template -->
    <h1>{name} Component</h1>
  </div>
</template>

<script>
export default {{
  name: '{name}',
  props: {{
    {props}
  }},
  data() {{
    return {{
      // TODO: Add component data
    }};
  }},
  methods: {{
    // TODO: Add component methods
  }}
}};
</script>

<style scoped>
.{css_class} {{
  /* TODO: Add component styles */
}}
</style>
'''
            }
        }
    
    def _initialize_complexity_indicators(self) -> Dict[str, List[str]]:
        """Initialize complexity detection indicators"""
        return {
            'simple': [
                'simple', 'basic', 'easy', 'quick', 'small', 
                'minimal', '简单', '基础', '简易', '快速'
            ],
            'medium': [
                'moderate', 'standard', 'normal', 'typical', 'regular',
                '中等', '标准', '常规', '普通'
            ],
            'complex': [
                'complex', 'advanced', 'sophisticated', 'detailed', 'comprehensive',
                'elaborate', '复杂', '高级', '详细', '完整', '精细'
            ]
        }
    
    def _initialize_entity_extractors(self) -> Dict[str, List[str]]:
        """Initialize code entity extraction patterns"""
        return {
            'function_name': [
                r'\b(function|method|def)\s+(\w+)',
                r'函数\s*(\w+)',
                r'方法\s*(\w+)',
                r'called\s+(\w+)',
                r'named\s+(\w+)',
                r'叫做\s*(\w+)',
                r'名为\s*(\w+)'
            ],
            'class_name': [
                r'\b(class)\s+(\w+)',
                r'类\s*(\w+)',
                r'对象\s*(\w+)',
                r'called\s+(\w+)',
                r'named\s+(\w+)'
            ],
            'component_name': [
                r'(component|组件)\s+(\w+)',
                r'(\w+)\s+(component|组件)',
                r'called\s+(\w+)',
                r'named\s+(\w+)'
            ],
            'variable_name': [
                r'\b(variable|var|let|const)\s+(\w+)',
                r'变量\s*(\w+)',
                r'called\s+(\w+)',
                r'named\s+(\w+)'
            ],
            'parameters': [
                r'with\s+(parameters?|arguments?|params?)\s+(.+?)(?:\.|,|$)',
                r'takes?\s+(.+?)(?:\s+as\s+)?(parameters?|arguments?|params?)',
                r'参数\s*(.+?)(?:\.|，|$)',
                r'传入\s*(.+?)(?:\.|，|$)'
            ],
            'return_type': [
                r'returns?\s+(\w+)',
                r'return\s+type\s+(\w+)', 
                r'返回\s*(\w+)',
                r'-> \s*(\w+)'
            ],
            'language': [
                r'\b(python|javascript|typescript|java|c\+\+|c#|go|rust|php|ruby|swift|kotlin)\b',
                r'用\s*(python|javascript|typescript|java|c\+\+|c#|go|rust|php|ruby|swift|kotlin)',
                r'in\s+(python|javascript|typescript|java|c\+\+|c#|go|rust|php|ruby|swift|kotlin)'
            ]
        }
    
    async def parse_code_intent(self, recognition_result: RecognitionResult, 
                               context: Optional[Context] = None) -> Optional[EnhancedCodeIntent]:
        """
        Parse code-specific intent from recognition result.
        
        Args:
            recognition_result: Speech recognition result
            context: Current system context
            
        Returns:
            Enhanced code intent or None if parsing failed
        """
        try:
            # First do base intent parsing
            base_intent = await super().parse_intent(recognition_result)
            if not base_intent:
                return None
            
            # Enhance with code-specific analysis
            enhanced_intent = await self._enhance_intent_with_code_analysis(base_intent, context)
            
            return enhanced_intent
            
        except Exception as e:
            logger.error(f"Code intent parsing failed: {e}")
            return None
    
    async def _enhance_intent_with_code_analysis(self, base_intent: ParsedIntent, 
                                               context: Optional[Context] = None) -> EnhancedCodeIntent:
        """Enhance base intent with code-specific analysis"""
        
        # Detect code intent type
        code_intent_type = self._detect_code_intent_type(base_intent.processed_text)
        
        # Extract code entities
        code_entities = self._extract_code_entities(base_intent.processed_text)
        
        # Detect target language
        target_language = self._detect_target_language(base_intent.processed_text, context)
        
        # Detect framework
        framework = self._detect_framework(base_intent.processed_text)
        
        # Assess complexity
        complexity = self._assess_code_complexity(base_intent.processed_text)
        
        # Check if dependencies are needed
        requires_deps = self._check_dependencies_needed(base_intent.processed_text, framework)
        
        # Estimate code size
        estimated_lines = self._estimate_code_lines(code_intent_type, complexity)
        
        # Create enhanced intent
        enhanced_intent = EnhancedCodeIntent(
            **base_intent.__dict__,
            code_intent_type=code_intent_type,
            code_entities=code_entities,
            target_language=target_language,
            framework=framework,
            code_complexity=complexity,
            requires_dependencies=requires_deps,
            estimated_lines=estimated_lines
        )
        
        return enhanced_intent
    
    def _detect_code_intent_type(self, text: str) -> Optional[CodeIntentType]:
        """Detect specific code intent type"""
        best_intent = None
        best_score = 0.0
        
        for intent_type, pattern_groups in self._code_intent_patterns.items():
            for group in pattern_groups:
                for pattern in group['patterns']:
                    if re.search(pattern, text, re.IGNORECASE):
                        score = group.get('weight', 1.0)
                        if score > best_score:
                            best_score = score
                            best_intent = intent_type
        
        return best_intent
    
    def _extract_code_entities(self, text: str) -> List[CodeEntity]:
        """Extract code-specific entities"""
        entities = []
        
        for entity_type, patterns in self._code_entity_extractors.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 1:
                        entity_name = match.group(-1)  # Last group is usually the name
                        entities.append(CodeEntity(
                            type=entity_type,
                            name=entity_name,
                            confidence=0.8
                        ))
        
        return entities
    
    def _detect_target_language(self, text: str, context: Optional[Context] = None) -> Optional[str]:
        """Detect target programming language"""
        
        # Check explicit language mentions
        for entity_type, patterns in self._code_entity_extractors.items():
            if entity_type == 'language':
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        return match.group(1).lower()
        
        # Infer from context
        if context and context.current_file:
            file_ext = context.current_file.split('.')[-1].lower()
            ext_to_lang = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'java': 'java',
                'cpp': 'c++',
                'c': 'c',
                'cs': 'c#',
                'go': 'go',
                'rs': 'rust',
                'php': 'php',
                'rb': 'ruby',
                'swift': 'swift',
                'kt': 'kotlin'
            }
            return ext_to_lang.get(file_ext)
        
        # Default to python
        return 'python'
    
    def _detect_framework(self, text: str) -> Optional[str]:
        """Detect web/application framework"""
        for framework, indicators in self._programming_frameworks.items():
            for indicator in indicators:
                if indicator.lower() in text.lower():
                    return framework
        return None
    
    def _assess_code_complexity(self, text: str) -> str:
        """Assess complexity level of requested code"""
        for complexity, indicators in self._complexity_indicators.items():
            for indicator in indicators:
                if indicator.lower() in text.lower():
                    return complexity
        
        # Default complexity based on keywords
        complex_keywords = ['advanced', 'complex', 'comprehensive', 'detailed', 'sophisticated']
        if any(keyword in text.lower() for keyword in complex_keywords):
            return 'complex'
        
        simple_keywords = ['simple', 'basic', 'quick', 'minimal', 'easy']
        if any(keyword in text.lower() for keyword in simple_keywords):
            return 'simple'
        
        return 'medium'
    
    def _check_dependencies_needed(self, text: str, framework: Optional[str]) -> bool:
        """Check if additional dependencies might be needed"""
        dependency_indicators = [
            'with dependencies', 'using library', 'import', 'install', 'package',
            'framework', 'library', 'module', 'plugin'
        ]
        
        if any(indicator in text.lower() for indicator in dependency_indicators):
            return True
        
        # Framework-specific dependencies
        if framework in ['react', 'vue', 'angular', 'django', 'flask', 'express']:
            return True
        
        return False
    
    def _estimate_code_lines(self, intent_type: Optional[CodeIntentType], complexity: str) -> Optional[int]:
        """Estimate number of lines of code to generate"""
        if not intent_type:
            return None
        
        base_estimates = {
            CodeIntentType.GENERATE_FUNCTION: {'simple': 10, 'medium': 25, 'complex': 50},
            CodeIntentType.GENERATE_CLASS: {'simple': 20, 'medium': 50, 'complex': 100},
            CodeIntentType.GENERATE_COMPONENT: {'simple': 30, 'medium': 75, 'complex': 150},
            CodeIntentType.GENERATE_MODULE: {'simple': 50, 'medium': 150, 'complex': 300},
            CodeIntentType.GENERATE_TEST: {'simple': 15, 'medium': 40, 'complex': 80}
        }
        
        estimates = base_estimates.get(intent_type, {'simple': 10, 'medium': 25, 'complex': 50})
        return estimates.get(complexity, 25)
    
    async def generate_code_suggestions(self, enhanced_intent: EnhancedCodeIntent) -> List[str]:
        """Generate smart code suggestions based on enhanced intent"""
        suggestions = []
        
        if enhanced_intent.code_intent_type == CodeIntentType.GENERATE_FUNCTION:
            suggestions.extend([
                f"generate_function with name '{enhanced_intent.code_entities[0].name if enhanced_intent.code_entities else 'new_function'}'",
                f"add_type_hints to the function",
                f"generate_test for the function",
                f"add_comments with docstring style"
            ])
        
        elif enhanced_intent.code_intent_type == CodeIntentType.GENERATE_CLASS:
            suggestions.extend([
                f"generate_class with name '{enhanced_intent.code_entities[0].name if enhanced_intent.code_entities else 'NewClass'}'",
                f"add_init_method to the class", 
                f"generate_test for the class",
                f"add_type_hints to class methods"
            ])
        
        elif enhanced_intent.code_intent_type == CodeIntentType.GENERATE_COMPONENT:
            suggestions.extend([
                f"generate_component for {enhanced_intent.framework or 'react'}",
                f"add_component_props",
                f"add_component_styles",
                f"generate_story for storybook"
            ])
        
        elif enhanced_intent.code_intent_type == CodeIntentType.REFACTOR_CODE:
            suggestions.extend([
                "refactor_current_selection",
                "extract_method from selection",
                "add_error_handling",
                "optimize_performance"
            ])
        
        # Add language-specific suggestions
        if enhanced_intent.target_language == 'python':
            suggestions.extend([
                "add_type_hints",
                "format_with_black", 
                "lint_with_pylint",
                "generate_docstring"
            ])
        elif enhanced_intent.target_language in ['javascript', 'typescript']:
            suggestions.extend([
                "format_with_prettier",
                "lint_with_eslint",
                "add_jsdoc_comments"
            ])
        
        return suggestions[:10]  # Return top 10 suggestions