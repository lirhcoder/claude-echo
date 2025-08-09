# Agents 系统详解

## 🤖 核心 Agents 系统

本项目采用 **AI指导AI** 的创新协作模式，通过 7 个专业化 Agent 实现复杂任务的智能协调执行。

## 🎯 Agents 协作原理

### 双AI架构模式

```
┌─────────────────────────────────────┐
│            AI指挥官层                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ 任务规划 │ │ 决策引擎 │ │ 质量控制 │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└─────────────┬───────────────────────────┘
              │ 文本指令/JSON
┌─────────────▼───────────────────────────┐
│         AI执行员层 (Claude Voice)        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ 命令执行 │ │ 适配器调用│ │ 状态监控 │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────┘
```

## 🏗️ 核心 Agents 详解

### 1. Coordinator (协调中心) 🎯

**职责**: 系统的总指挥，负责整体协调和任务分发

**核心功能**:
- 用户请求的统一入口
- 各 Agent 间的协调调度
- 执行流程的整体控制
- 异常情况的统一处理

**实现架构**:
```python
class Coordinator:
    def __init__(self):
        self.task_planner = TaskPlanner()
        self.security_guardian = SecurityGuardian()
        self.auto_worker = AutoWorker()
        self.presence_monitor = PresenceMonitor()
        self.handover_manager = HandoverManager()
        self.session_manager = SessionManager()
        
        # 内部状态
        self.current_session = None
        self.active_plans = {}
        self.system_status = SystemStatus.IDLE
    
    async def process_request(self, user_input: str, context: Context) -> Response:
        """处理用户请求的主入口"""
        try:
            # 1. 会话管理
            session = await self.session_manager.get_or_create_session(context.user_id)
            
            # 2. 意图解析
            intent = await self.parse_user_intent(user_input, context)
            
            # 3. 安全预检
            if not await self.security_guardian.pre_validate(intent):
                return Response.security_blocked(intent)
            
            # 4. 任务规划
            plan = await self.task_planner.create_execution_plan(intent, context)
            
            # 5. 最终安全检查
            if not await self.security_guardian.validate_plan(plan):
                return Response.security_blocked(plan)
            
            # 6. 执行协调
            if context.user_present:
                # 交互模式
                return await self.execute_interactive_mode(plan)
            else:
                # 静音模式
                return await self.execute_silent_mode(plan)
                
        except Exception as e:
            return await self.handle_error(e, context)
    
    async def execute_interactive_mode(self, plan: ExecutionPlan) -> Response:
        """交互模式执行"""
        # 实时执行，用户可干预
        return await self.auto_worker.execute_with_feedback(plan)
    
    async def execute_silent_mode(self, plan: ExecutionPlan) -> Response:
        """静音模式执行"""
        # 自主执行，准备交接报告
        result = await self.auto_worker.execute_autonomous(plan)
        await self.handover_manager.prepare_handover_report(result)
        return result
```

**关键决策逻辑**:
```python
async def make_execution_decision(self, plan: ExecutionPlan, context: Context) -> ExecutionMode:
    """执行模式决策"""
    
    # 风险评估
    risk_level = await self.assess_risk_level(plan)
    
    # 用户状态评估
    user_presence = await self.presence_monitor.get_user_status()
    
    # 决策矩阵
    if risk_level == RiskLevel.CRITICAL:
        return ExecutionMode.REQUIRE_CONFIRMATION
    elif risk_level == RiskLevel.HIGH and not user_presence.available:
        return ExecutionMode.DEFER_EXECUTION
    elif not user_presence.available and risk_level <= RiskLevel.MEDIUM:
        return ExecutionMode.SILENT_EXECUTION
    else:
        return ExecutionMode.INTERACTIVE_EXECUTION
```

### 2. Task-Planner (任务规划) 📋

**职责**: 智能任务分解和执行计划制定

**核心功能**:
- 复杂任务的智能分解
- 执行步骤的优化排序
- 依赖关系分析
- 资源需求评估
- 时间估算

**智能规划算法**:
```python
class TaskPlanner:
    def __init__(self):
        self.knowledge_base = TaskKnowledgeBase()
        self.pattern_analyzer = TaskPatternAnalyzer()
        self.optimizer = ExecutionOptimizer()
    
    async def create_execution_plan(self, intent: Intent, context: Context) -> ExecutionPlan:
        """创建执行计划"""
        
        # 1. 任务分解 (Decomposition)
        subtasks = await self.decompose_task(intent)
        
        # 2. 依赖分析 (Dependency Analysis)
        dependencies = await self.analyze_dependencies(subtasks, context)
        
        # 3. 资源评估 (Resource Assessment)
        resources = await self.assess_required_resources(subtasks)
        
        # 4. 执行优化 (Optimization)
        optimized_sequence = await self.optimize_execution_sequence(
            subtasks, dependencies, resources
        )
        
        # 5. 时间估算 (Time Estimation)
        estimated_duration = await self.estimate_execution_time(optimized_sequence)
        
        # 6. 风险评估 (Risk Assessment)
        risk_analysis = await self.analyze_execution_risks(optimized_sequence)
        
        return ExecutionPlan(
            id=generate_plan_id(),
            intent=intent,
            tasks=optimized_sequence,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            resource_requirements=resources,
            risk_analysis=risk_analysis,
            context=context
        )
    
    async def decompose_task(self, intent: Intent) -> List[SubTask]:
        """智能任务分解"""
        
        # 基于意图类型分解
        if intent.type == IntentType.FILE_OPERATION:
            return await self.decompose_file_operation(intent)
        elif intent.type == IntentType.CODE_GENERATION:
            return await self.decompose_code_generation(intent)
        elif intent.type == IntentType.SYSTEM_CONTROL:
            return await self.decompose_system_control(intent)
        
        # 通用分解逻辑
        return await self.generic_task_decomposition(intent)
    
    async def optimize_execution_sequence(self, tasks: List[SubTask], 
                                        dependencies: Dependencies,
                                        resources: Resources) -> List[SubTask]:
        """执行序列优化"""
        
        # 1. 拓扑排序处理依赖
        sorted_tasks = self.topological_sort(tasks, dependencies)
        
        # 2. 并行化优化
        parallel_groups = self.identify_parallel_opportunities(sorted_tasks, dependencies)
        
        # 3. 资源调度优化
        optimized_sequence = self.optimize_resource_usage(parallel_groups, resources)
        
        return optimized_sequence
```

**任务分解示例**:
```python
# 示例: "创建一个 React 组件并添加到项目中"
async def decompose_react_component_creation(self, intent: Intent) -> List[SubTask]:
    return [
        SubTask(
            id="analyze_project",
            type=TaskType.ANALYSIS,
            description="分析当前项目结构和技术栈",
            adapter="file_system",
            command="analyze_project_structure",
            parameters={"path": intent.parameters.get("project_path")},
            estimated_duration=timedelta(seconds=10)
        ),
        SubTask(
            id="create_component_file",
            type=TaskType.FILE_CREATION,
            description="创建 React 组件文件",
            adapter="claude_code",
            command="generate_react_component",
            parameters={
                "component_name": intent.parameters.get("component_name"),
                "props": intent.parameters.get("props", [])
            },
            dependencies=["analyze_project"],
            estimated_duration=timedelta(seconds=30)
        ),
        SubTask(
            id="update_exports",
            type=TaskType.FILE_MODIFICATION,
            description="更新 index.js 导出",
            adapter="file_system",
            command="update_exports",
            parameters={
                "export_file": "src/components/index.js",
                "new_export": intent.parameters.get("component_name")
            },
            dependencies=["create_component_file"],
            estimated_duration=timedelta(seconds=5)
        )
    ]
```

### 3. Presence-Monitor (状态监控) 👁️

**职责**: 用户状态和环境上下文的实时监控

**核心功能**:
- 用户在线状态检测
- 活动应用程序监控
- 屏幕内容分析
- 用户行为模式识别
- 环境变化检测

**监控架构**:
```python
class PresenceMonitor:
    def __init__(self):
        self.user_detector = UserPresenceDetector()
        self.app_monitor = ApplicationMonitor()
        self.screen_analyzer = ScreenAnalyzer()
        self.behavior_tracker = BehaviorTracker()
        self.context_builder = ContextBuilder()
        
        # 监控状态
        self.monitoring_active = False
        self.last_activity_time = None
        self.current_context = None
    
    async def start_monitoring(self):
        """开始监控"""
        self.monitoring_active = True
        
        # 启动各种监控任务
        await asyncio.gather(
            self.monitor_user_presence(),
            self.monitor_application_changes(),
            self.monitor_screen_content(),
            self.track_user_behavior()
        )
    
    async def monitor_user_presence(self):
        """用户在线状态监控"""
        while self.monitoring_active:
            try:
                # 1. 鼠标活动检测
                mouse_activity = self.detect_mouse_activity()
                
                # 2. 键盘活动检测
                keyboard_activity = self.detect_keyboard_activity()
                
                # 3. 摄像头检测 (可选)
                camera_detection = await self.detect_face_presence()
                
                # 4. 麦克风检测
                audio_activity = self.detect_audio_activity()
                
                # 5. 综合判断
                presence_status = self.calculate_presence_status(
                    mouse_activity, keyboard_activity, camera_detection, audio_activity
                )
                
                # 6. 更新状态
                await self.update_presence_status(presence_status)
                
                # 7. 检测状态变化
                if self.detect_status_change(presence_status):
                    await self.notify_status_change(presence_status)
                
                await asyncio.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"用户状态监控错误: {e}")
                await asyncio.sleep(5)  # 错误时延长检查间隔
    
    async def get_current_context(self) -> Context:
        """获取当前环境上下文"""
        return Context(
            # 用户状态
            user_present=await self.is_user_present(),
            last_activity=self.last_activity_time,
            attention_level=await self.assess_attention_level(),
            
            # 应用环境
            active_application=await self.app_monitor.get_active_app(),
            open_windows=await self.app_monitor.get_open_windows(),
            current_file=await self.get_current_file_path(),
            
            # 屏幕内容
            screen_content=await self.screen_analyzer.get_screen_summary(),
            cursor_position=await self.get_cursor_position(),
            selected_text=await self.get_selected_text(),
            
            # 历史行为
            recent_commands=self.behavior_tracker.get_recent_commands(),
            working_patterns=self.behavior_tracker.get_working_patterns(),
            
            # 时间信息
            timestamp=datetime.now(),
            timezone=self.get_current_timezone()
        )
```

**智能上下文分析**:
```python
async def analyze_user_intent_from_context(self, context: Context) -> IntentHints:
    """基于上下文分析用户可能的意图"""
    
    hints = IntentHints()
    
    # 基于当前应用推断
    if context.active_application == "code.exe":  # VS Code
        if ".py" in context.current_file:
            hints.add_suggestion("可能需要运行 Python 代码")
            hints.add_suggestion("可能需要调试 Python 程序")
        elif ".js" in context.current_file:
            hints.add_suggestion("可能需要启动开发服务器")
            hints.add_suggestion("可能需要运行测试")
    
    # 基于屏幕内容推断
    if "error" in context.screen_content.lower():
        hints.add_suggestion("可能需要修复代码错误")
        hints.add_suggestion("可能需要查看错误日志")
    
    # 基于行为模式推断
    recent_pattern = self.behavior_tracker.analyze_recent_pattern()
    if recent_pattern == BehaviorPattern.DEBUGGING:
        hints.add_suggestion("继续调试会话")
        hints.add_suggestion("查看调试变量")
    
    return hints
```

### 4. Auto-Worker (自主执行) ⚡

**职责**: 具体任务的智能执行

**核心功能**:
- 执行计划的具体实施
- 适配器的动态选择和调用
- 执行状态的实时监控
- 异常情况的智能处理
- 执行结果的质量评估

**执行引擎**:
```python
class AutoWorker:
    def __init__(self):
        self.adapter_manager = AdapterManager()
        self.execution_monitor = ExecutionMonitor()
        self.error_handler = ErrorHandler()
        self.result_validator = ResultValidator()
        
        # 执行状态
        self.active_executions = {}
        self.execution_history = []
    
    async def execute_plan(self, plan: ExecutionPlan, mode: ExecutionMode) -> ExecutionResult:
        """执行计划"""
        
        execution_id = generate_execution_id()
        execution_context = ExecutionContext(
            id=execution_id,
            plan=plan,
            mode=mode,
            start_time=datetime.now()
        )
        
        try:
            # 1. 执行前准备
            await self.prepare_execution(execution_context)
            
            # 2. 根据模式执行
            if mode == ExecutionMode.INTERACTIVE:
                result = await self.execute_interactive(execution_context)
            elif mode == ExecutionMode.SILENT:
                result = await self.execute_silent(execution_context)
            elif mode == ExecutionMode.STEP_BY_STEP:
                result = await self.execute_step_by_step(execution_context)
            
            # 3. 结果验证
            validated_result = await self.result_validator.validate(result)
            
            # 4. 清理工作
            await self.cleanup_execution(execution_context)
            
            return validated_result
            
        except Exception as e:
            return await self.handle_execution_error(execution_context, e)
    
    async def execute_silent(self, context: ExecutionContext) -> ExecutionResult:
        """静音模式执行"""
        
        results = []
        
        for task in context.plan.tasks:
            try:
                # 1. 选择适配器
                adapter = await self.adapter_manager.get_best_adapter(task)
                
                if not adapter:
                    raise AdapterNotFoundError(f"找不到适合的适配器: {task.adapter}")
                
                # 2. 执行前检查
                if not await adapter.can_execute(task.command, task.parameters):
                    raise ExecutionError(f"适配器无法执行命令: {task.command}")
                
                # 3. 执行任务
                logger.info(f"执行任务: {task.description}")
                
                task_result = await adapter.execute_command(
                    task.command, 
                    task.parameters,
                    context=context
                )
                
                # 4. 结果检查
                if not task_result.success:
                    if task.optional:
                        logger.warning(f"可选任务执行失败: {task.description}")
                        continue
                    else:
                        raise ExecutionError(f"必需任务执行失败: {task_result.error}")
                
                results.append(task_result)
                
                # 5. 进度更新
                await self.update_execution_progress(context, task, task_result)
                
                # 6. 依赖任务检查
                await self.check_dependent_tasks(context, task)
                
            except Exception as e:
                error_result = await self.handle_task_error(context, task, e)
                results.append(error_result)
                
                # 根据错误处理策略决定是否继续
                if not await self.should_continue_after_error(context, task, e):
                    break
        
        return ExecutionResult(
            id=context.id,
            success=all(r.success for r in results),
            results=results,
            duration=datetime.now() - context.start_time,
            mode=context.mode
        )
```

**智能错误处理**:
```python
async def handle_task_error(self, context: ExecutionContext, task: SubTask, error: Exception) -> TaskResult:
    """智能错误处理"""
    
    # 1. 错误分类
    error_type = self.classify_error(error)
    
    # 2. 重试策略
    if error_type in [ErrorType.NETWORK_TIMEOUT, ErrorType.TEMPORARY_RESOURCE_UNAVAILABLE]:
        for attempt in range(3):
            try:
                await asyncio.sleep(2 ** attempt)  # 指数退避
                return await self.retry_task(context, task)
            except:
                continue
    
    # 3. 替代方案
    if error_type == ErrorType.ADAPTER_UNAVAILABLE:
        alternative_adapter = await self.find_alternative_adapter(task)
        if alternative_adapter:
            return await alternative_adapter.execute_command(task.command, task.parameters)
    
    # 4. 降级处理
    if error_type == ErrorType.FEATURE_NOT_SUPPORTED:
        degraded_task = await self.create_degraded_task(task)
        if degraded_task:
            return await self.execute_single_task(context, degraded_task)
    
    # 5. 用户通知
    await self.notify_user_of_error(context, task, error)
    
    return TaskResult(
        task_id=task.id,
        success=False,
        error=str(error),
        error_type=error_type,
        timestamp=datetime.now()
    )
```

### 5. Security-Guardian (安全监护) 🛡️

**职责**: 系统安全控制和风险管理

**核心功能**:
- 操作安全性评估
- 权限验证
- 危险操作拦截
- 用户确认机制
- 审计日志记录

**安全架构**:
```python
class SecurityGuardian:
    def __init__(self):
        self.risk_assessor = RiskAssessor()
        self.permission_manager = PermissionManager()
        self.audit_logger = AuditLogger()
        self.user_confirmer = UserConfirmer()
        
        # 安全配置
        self.security_config = self.load_security_config()
        self.threat_database = ThreatDatabase()
    
    async def validate_request(self, intent: Intent, context: Context) -> SecurityValidation:
        """验证请求安全性"""
        
        validation = SecurityValidation(intent_id=intent.id)
        
        try:
            # 1. 基础权限检查
            permission_check = await self.check_permissions(intent, context)
            validation.add_check("permission", permission_check)
            
            # 2. 风险评估
            risk_assessment = await self.assess_risk(intent, context)
            validation.add_check("risk", risk_assessment)
            
            # 3. 威胁检测
            threat_detection = await self.detect_threats(intent, context)
            validation.add_check("threat", threat_detection)
            
            # 4. 资源限制检查
            resource_check = await self.check_resource_limits(intent, context)
            validation.add_check("resource", resource_check)
            
            # 5. 时间限制检查
            time_check = await self.check_time_restrictions(intent, context)
            validation.add_check("time", time_check)
            
            # 6. 综合决策
            validation.final_decision = self.make_security_decision(validation)
            
            # 7. 审计记录
            await self.audit_logger.log_security_check(validation)
            
            return validation
            
        except Exception as e:
            # 安全检查出错时，默认拒绝
            validation.final_decision = SecurityDecision.DENY
            validation.error = str(e)
            return validation
    
    async def assess_risk(self, intent: Intent, context: Context) -> RiskAssessment:
        """风险评估"""
        
        risk_factors = []
        
        # 1. 操作类型风险
        operation_risk = self.assess_operation_risk(intent.operation_type)
        risk_factors.append(operation_risk)
        
        # 2. 目标资源风险
        target_risk = self.assess_target_risk(intent.target_resources)
        risk_factors.append(target_risk)
        
        # 3. 参数风险
        parameter_risk = self.assess_parameter_risk(intent.parameters)
        risk_factors.append(parameter_risk)
        
        # 4. 上下文风险
        context_risk = self.assess_context_risk(context)
        risk_factors.append(context_risk)
        
        # 5. 历史风险
        history_risk = await self.assess_historical_risk(context.user_id, intent)
        risk_factors.append(history_risk)
        
        # 6. 组合风险评估
        combined_risk = self.calculate_combined_risk(risk_factors)
        
        return RiskAssessment(
            individual_factors=risk_factors,
            combined_risk=combined_risk,
            risk_level=self.categorize_risk_level(combined_risk),
            mitigation_suggestions=self.suggest_mitigations(risk_factors)
        )
```

**安全策略配置**:
```yaml
# security_config.yaml
security:
  risk_thresholds:
    low: 0.2
    medium: 0.5
    high: 0.8
    critical: 0.95
  
  operation_risks:
    file_read: 0.1
    file_write: 0.3
    file_delete: 0.7
    system_command: 0.8
    network_access: 0.6
    registry_modify: 0.9
  
  auto_approval_limits:
    silent_mode: 0.3  # 静音模式最高允许风险级别
    interactive_mode: 0.6  # 交互模式最高允许风险级别
  
  required_confirmations:
    - operation: "delete_file"
      conditions: ["file_size > 1MB", "file_extension in ['.exe', '.dll']"]
    - operation: "system_command"
      conditions: ["contains_keywords(['format', 'delete', 'remove'])"]
  
  blocked_operations:
    - "format_drive"
    - "delete_system_files"
    - "modify_boot_sector"
  
  audit_settings:
    log_all_operations: true
    log_level: "INFO"
    retention_days: 90
```

### 6. Handover-Manager (交接管理) 🤝

**职责**: 用户返回时的智能交接处理

**核心功能**:
- 执行结果摘要生成
- 问题和异常汇总
- 后续建议提供
- 用户打断处理
- 上下文恢复

**交接架构**:
```python
class HandoverManager:
    def __init__(self):
        self.summary_generator = SummaryGenerator()
        self.report_formatter = ReportFormatter()
        self.context_restorer = ContextRestorer()
        self.interruption_handler = InterruptionHandler()
    
    async def prepare_handover_report(self, execution_result: ExecutionResult) -> HandoverReport:
        """准备交接报告"""
        
        report = HandoverReport(
            execution_id=execution_result.id,
            timestamp=datetime.now()
        )
        
        # 1. 执行摘要
        report.summary = await self.generate_execution_summary(execution_result)
        
        # 2. 成功任务汇总
        report.successful_tasks = self.extract_successful_tasks(execution_result)
        
        # 3. 失败任务分析
        report.failed_tasks = await self.analyze_failed_tasks(execution_result)
        
        # 4. 系统状态变化
        report.state_changes = await self.detect_state_changes(execution_result)
        
        # 5. 后续建议
        report.recommendations = await self.generate_recommendations(execution_result)
        
        # 6. 需要关注的事项
        report.attention_items = await self.identify_attention_items(execution_result)
        
        return report
    
    async def generate_execution_summary(self, result: ExecutionResult) -> str:
        """生成执行摘要"""
        
        successful_count = sum(1 for r in result.results if r.success)
        failed_count = len(result.results) - successful_count
        
        summary_parts = []
        
        # 基础统计
        summary_parts.append(f"执行完成: {successful_count}/{len(result.results)} 个任务成功")
        
        if result.duration:
            summary_parts.append(f"用时: {result.duration.total_seconds():.1f} 秒")
        
        # 主要成就
        major_achievements = self.identify_major_achievements(result)
        if major_achievements:
            summary_parts.append(f"主要完成: {', '.join(major_achievements)}")
        
        # 主要问题
        if failed_count > 0:
            major_issues = self.identify_major_issues(result)
            summary_parts.append(f"遇到问题: {', '.join(major_issues)}")
        
        return " | ".join(summary_parts)
```

**智能摘要生成**:
```python
async def generate_intelligent_summary(self, execution_result: ExecutionResult) -> IntelligentSummary:
    """生成智能摘要"""
    
    # 1. 关键成果识别
    key_achievements = []
    for task_result in execution_result.results:
        if task_result.success and task_result.impact_level == ImpactLevel.HIGH:
            key_achievements.append({
                "description": task_result.description,
                "outcome": task_result.outcome,
                "files_affected": task_result.files_affected
            })
    
    # 2. 问题分析
    issues_analysis = []
    for task_result in execution_result.results:
        if not task_result.success:
            issue_analysis = await self.analyze_failure_cause(task_result)
            issues_analysis.append(issue_analysis)
    
    # 3. 系统状态变化
    state_changes = await self.detect_meaningful_changes(execution_result)
    
    # 4. 用户行动建议
    action_recommendations = await self.generate_action_recommendations(
        key_achievements, issues_analysis, state_changes
    )
    
    return IntelligentSummary(
        key_achievements=key_achievements,
        issues_analysis=issues_analysis,
        state_changes=state_changes,
        action_recommendations=action_recommendations,
        confidence_score=self.calculate_summary_confidence(execution_result)
    )
```

### 7. Session-Manager (会话管理) 📚

**职责**: 会话生命周期和状态管理

**核心功能**:
- 会话创建和销毁
- 状态持久化
- 上下文保存和恢复
- 多会话并发管理
- 会话历史记录

**会话管理架构**:
```python
class SessionManager:
    def __init__(self):
        self.active_sessions = {}
        self.session_storage = SessionStorage()
        self.context_manager = ContextManager()
        self.state_serializer = StateSerializer()
    
    async def create_session(self, user_id: str, session_type: SessionType = SessionType.INTERACTIVE) -> Session:
        """创建新会话"""
        
        session_id = generate_session_id()
        session = Session(
            id=session_id,
            user_id=user_id,
            type=session_type,
            created_at=datetime.now(),
            status=SessionStatus.ACTIVE
        )
        
        # 1. 初始化会话上下文
        session.context = await self.context_manager.create_initial_context(user_id)
        
        # 2. 加载用户偏好
        session.preferences = await self.load_user_preferences(user_id)
        
        # 3. 初始化会话状态
        session.state = SessionState()
        
        # 4. 注册活跃会话
        self.active_sessions[session_id] = session
        
        # 5. 持久化会话信息
        await self.session_storage.save_session(session)
        
        return session
    
    async def save_session_state(self, session_id: str):
        """保存会话状态"""
        
        if session_id not in self.active_sessions:
            raise SessionNotFoundError(f"会话不存在: {session_id}")
        
        session = self.active_sessions[session_id]
        
        # 1. 序列化当前状态
        serialized_state = await self.state_serializer.serialize(session.state)
        
        # 2. 保存到持久存储
        await self.session_storage.save_state(session_id, serialized_state)
        
        # 3. 更新最后保存时间
        session.last_saved_at = datetime.now()
        
        logger.info(f"会话状态已保存: {session_id}")
    
    async def restore_session(self, session_id: str) -> Session:
        """恢复会话"""
        
        # 1. 从存储加载会话
        stored_session = await self.session_storage.load_session(session_id)
        
        if not stored_session:
            raise SessionNotFoundError(f"找不到会话: {session_id}")
        
        # 2. 反序列化状态
        session_state = await self.state_serializer.deserialize(stored_session.state_data)
        
        # 3. 重建会话对象
        session = Session(
            id=stored_session.id,
            user_id=stored_session.user_id,
            type=stored_session.type,
            created_at=stored_session.created_at,
            status=SessionStatus.RESTORED
        )
        
        session.state = session_state
        session.context = await self.context_manager.restore_context(stored_session.context_data)
        
        # 4. 重新激活会话
        self.active_sessions[session_id] = session
        session.status = SessionStatus.ACTIVE
        
        return session
```

**会话持久化**:
```python
class SessionStorage:
    """会话持久化存储"""
    
    def __init__(self, storage_path: str = "sessions/"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    async def save_session(self, session: Session):
        """保存会话到文件"""
        
        session_file = self.storage_path / f"{session.id}.json"
        
        session_data = {
            "id": session.id,
            "user_id": session.user_id,
            "type": session.type.value,
            "created_at": session.created_at.isoformat(),
            "last_saved_at": datetime.now().isoformat(),
            "state_data": await self.state_serializer.serialize(session.state),
            "context_data": await self.context_manager.serialize_context(session.context),
            "preferences": session.preferences,
            "execution_history": [
                {
                    "id": exec_record.id,
                    "intent": exec_record.intent.to_dict(),
                    "result": exec_record.result.to_dict(),
                    "timestamp": exec_record.timestamp.isoformat()
                }
                for exec_record in session.execution_history
            ]
        }
        
        async with aiofiles.open(session_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(session_data, indent=2, ensure_ascii=False))
```

## 🔄 Agents 协作流程

### 典型协作场景

```python
async def typical_collaboration_flow(user_input: str, context: Context):
    """典型的 Agents 协作流程"""
    
    # 1. Coordinator 接收请求
    coordinator = Coordinator()
    
    # 2. Session-Manager 管理会话
    session = await coordinator.session_manager.get_or_create_session(context.user_id)
    
    # 3. Presence-Monitor 提供上下文
    current_context = await coordinator.presence_monitor.get_current_context()
    
    # 4. Security-Guardian 安全检查
    intent = await coordinator.parse_user_intent(user_input, current_context)
    security_check = await coordinator.security_guardian.validate_request(intent, current_context)
    
    if not security_check.approved:
        return Response.security_denied(security_check.reason)
    
    # 5. Task-Planner 制定计划
    execution_plan = await coordinator.task_planner.create_execution_plan(intent, current_context)
    
    # 6. Auto-Worker 执行计划
    execution_result = await coordinator.auto_worker.execute_plan(execution_plan)
    
    # 7. Handover-Manager 准备报告
    if not current_context.user_present:
        handover_report = await coordinator.handover_manager.prepare_handover_report(execution_result)
        session.pending_reports.append(handover_report)
    
    # 8. Session-Manager 保存状态
    await coordinator.session_manager.save_session_state(session.id)
    
    return Response.success(execution_result)
```

这个 Agents 系统设计确保了智能、安全、高效的任务执行，同时提供了良好的用户体验和系统可维护性。每个 Agent 都有明确的职责分工，通过协调工作实现复杂任务的智能化处理。