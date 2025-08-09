#!/usr/bin/env python3
"""
Claude Echo 系统健康监控和性能基准测试
Integration Agent - 系统健康监控和性能基准测试

功能包括：
1. 实时系统健康监控
2. 性能基准测试和比较
3. 资源使用监控
4. 错误和异常追踪
5. 可视化监控面板
6. 自动告警机制
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import tempfile
import threading
from collections import deque, defaultdict
import uuid


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthAlert:
    """健康告警"""
    alert_id: str
    component: str
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class SystemMetrics:
    """系统指标快照"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    active_connections: int
    uptime_seconds: float


class SystemHealthMonitor:
    """系统健康监控器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # 监控数据存储
        self.metrics_history: deque = deque(maxlen=1000)  # 最近1000个数据点
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.resolved_alerts: List[HealthAlert] = []
        
        # 基准性能数据
        self.performance_baselines = self.config.get('performance_baselines', {})
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        
        # 监控状态
        self.monitoring_active = False
        self.start_time = datetime.now()
        
        # 组件状态
        self.component_health: Dict[str, Dict[str, Any]] = {}
        
        print("📊 系统健康监控器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """默认监控配置"""
        return {
            'monitoring_interval': 5,  # 秒
            'metrics_retention_minutes': 60,
            'performance_baselines': {
                'event_throughput': 100,  # 事件/秒
                'response_time': 5.0,     # 秒
                'memory_usage_mb': 1024,  # MB
                'cpu_usage_percent': 80,  # %
                'error_rate': 0.05,       # 5%
                'disk_usage_percent': 85   # %
            },
            'alert_thresholds': {
                'cpu_warning': 70,
                'cpu_critical': 90,
                'memory_warning': 80,
                'memory_critical': 95,
                'disk_warning': 80,
                'disk_critical': 95,
                'response_time_warning': 3.0,
                'response_time_critical': 10.0,
                'error_rate_warning': 0.1,
                'error_rate_critical': 0.2
            },
            'components_to_monitor': [
                'core_architecture',
                'learning_system', 
                'speech_system',
                'agent_system',
                'event_system',
                'data_manager'
            ]
        }
    
    async def start_monitoring(self):
        """启动系统监控"""
        if self.monitoring_active:
            print("⚠️ 监控已在运行中")
            return
        
        self.monitoring_active = True
        self.start_time = datetime.now()
        
        print("🔍 启动系统健康监控...")
        print(f"   监控间隔: {self.config['monitoring_interval']}秒")
        print(f"   监控组件: {len(self.config['components_to_monitor'])}个")
        
        # 启动各种监控任务
        tasks = [
            asyncio.create_task(self._system_metrics_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._component_health_loop()),
            asyncio.create_task(self._alert_processing_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n⏹️ 监控被用户停止")
        except Exception as e:
            print(f"\n❌ 监控异常: {e}")
        finally:
            self.monitoring_active = False
    
    async def _system_metrics_loop(self):
        """系统指标收集循环"""
        interval = self.config['monitoring_interval']
        
        while self.monitoring_active:
            try:
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # 检查系统资源告警
                await self._check_system_alerts(metrics)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"⚠️ 系统指标收集错误: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # 模拟系统指标收集（实际应该使用psutil等库）
            import random
            
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=random.uniform(10, 85),
                memory_percent=random.uniform(30, 90),
                memory_mb=random.uniform(200, 1200),
                disk_usage_percent=random.uniform(40, 85),
                network_io={'bytes_sent': random.randint(1000, 10000), 'bytes_recv': random.randint(1000, 10000)},
                active_connections=random.randint(5, 50),
                uptime_seconds=uptime
            )
            
            return metrics
            
        except Exception as e:
            # 返回默认值，避免监控中断
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                memory_mb=0,
                disk_usage_percent=0,
                network_io={'bytes_sent': 0, 'bytes_recv': 0},
                active_connections=0,
                uptime_seconds=0
            )
    
    async def _performance_monitoring_loop(self):
        """性能监控循环"""
        interval = self.config['monitoring_interval'] * 2  # 较慢的监控频率
        
        while self.monitoring_active:
            try:
                # 收集性能指标
                performance_data = await self._collect_performance_metrics()
                
                for metric in performance_data:
                    self.performance_metrics[metric.name].append(metric)
                
                # 检查性能基准
                await self._check_performance_baselines(performance_data)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"⚠️ 性能监控错误: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_performance_metrics(self) -> List[PerformanceMetric]:
        """收集性能指标"""
        import random
        
        # 模拟性能数据收集
        metrics = [
            PerformanceMetric(
                name="event_throughput",
                value=random.uniform(80, 200),
                unit="events/sec",
                metric_type=MetricType.RATE
            ),
            PerformanceMetric(
                name="response_time",
                value=random.uniform(0.5, 8.0),
                unit="seconds",
                metric_type=MetricType.HISTOGRAM
            ),
            PerformanceMetric(
                name="error_rate",
                value=random.uniform(0, 0.15),
                unit="ratio",
                metric_type=MetricType.RATE
            ),
            PerformanceMetric(
                name="active_users",
                value=random.randint(1, 20),
                unit="count",
                metric_type=MetricType.GAUGE
            ),
            PerformanceMetric(
                name="learning_operations_per_minute",
                value=random.uniform(10, 50),
                unit="ops/min",
                metric_type=MetricType.RATE
            )
        ]
        
        return metrics
    
    async def _component_health_loop(self):
        """组件健康检查循环"""
        interval = self.config['monitoring_interval'] * 3  # 更慢的检查频率
        
        while self.monitoring_active:
            try:
                for component in self.config['components_to_monitor']:
                    health_data = await self._check_component_health(component)
                    self.component_health[component] = health_data
                    
                    # 根据组件健康状态生成告警
                    await self._generate_component_alerts(component, health_data)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"⚠️ 组件健康检查错误: {e}")
                await asyncio.sleep(interval)
    
    async def _check_component_health(self, component: str) -> Dict[str, Any]:
        """检查组件健康状态"""
        import random
        
        # 模拟组件健康检查
        await asyncio.sleep(0.1)
        
        # 随机生成健康状态
        health_states = ['healthy', 'warning', 'error']
        weights = [0.7, 0.2, 0.1]  # 70%健康，20%警告，10%错误
        
        status = random.choices(health_states, weights=weights)[0]
        
        health_data = {
            'status': status,
            'last_check': datetime.now().isoformat(),
            'response_time': random.uniform(0.1, 3.0),
            'availability': random.uniform(0.95, 1.0),
            'throughput': random.uniform(50, 200)
        }
        
        if status == 'warning':
            health_data['issues'] = ['响应时间偏高', '连接数较多']
        elif status == 'error':
            health_data['issues'] = ['服务不可用', '连接超时']
        else:
            health_data['issues'] = []
        
        return health_data
    
    async def _alert_processing_loop(self):
        """告警处理循环"""
        while self.monitoring_active:
            try:
                # 检查告警是否需要解决
                await self._process_alert_resolution()
                
                # 显示当前活跃告警摘要
                if len(self.active_alerts) > 0:
                    await self._display_alert_summary()
                
                await asyncio.sleep(30)  # 30秒检查一次
                
            except Exception as e:
                print(f"⚠️ 告警处理错误: {e}")
                await asyncio.sleep(30)
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """检查系统资源告警"""
        thresholds = self.alert_thresholds
        
        # CPU使用率告警
        if metrics.cpu_percent >= thresholds.get('cpu_critical', 90):
            await self._create_alert("system", AlertLevel.CRITICAL, 
                                   f"CPU使用率严重: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent >= thresholds.get('cpu_warning', 70):
            await self._create_alert("system", AlertLevel.WARNING,
                                   f"CPU使用率偏高: {metrics.cpu_percent:.1f}%")
        
        # 内存使用告警
        if metrics.memory_percent >= thresholds.get('memory_critical', 95):
            await self._create_alert("system", AlertLevel.CRITICAL,
                                   f"内存使用率严重: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent >= thresholds.get('memory_warning', 80):
            await self._create_alert("system", AlertLevel.WARNING,
                                   f"内存使用率偏高: {metrics.memory_percent:.1f}%")
        
        # 磁盘使用告警
        if metrics.disk_usage_percent >= thresholds.get('disk_critical', 95):
            await self._create_alert("system", AlertLevel.CRITICAL,
                                   f"磁盘空间不足: {metrics.disk_usage_percent:.1f}%")
        elif metrics.disk_usage_percent >= thresholds.get('disk_warning', 80):
            await self._create_alert("system", AlertLevel.WARNING,
                                   f"磁盘使用率偏高: {metrics.disk_usage_percent:.1f}%")
    
    async def _check_performance_baselines(self, metrics: List[PerformanceMetric]):
        """检查性能基准"""
        baselines = self.performance_baselines
        thresholds = self.alert_thresholds
        
        for metric in metrics:
            baseline = baselines.get(metric.name)
            if not baseline:
                continue
            
            if metric.name == "response_time":
                if metric.value >= thresholds.get('response_time_critical', 10.0):
                    await self._create_alert("performance", AlertLevel.CRITICAL,
                                           f"响应时间严重: {metric.value:.2f}s")
                elif metric.value >= thresholds.get('response_time_warning', 3.0):
                    await self._create_alert("performance", AlertLevel.WARNING,
                                           f"响应时间偏高: {metric.value:.2f}s")
            
            elif metric.name == "error_rate":
                if metric.value >= thresholds.get('error_rate_critical', 0.2):
                    await self._create_alert("performance", AlertLevel.CRITICAL,
                                           f"错误率严重: {metric.value:.1%}")
                elif metric.value >= thresholds.get('error_rate_warning', 0.1):
                    await self._create_alert("performance", AlertLevel.WARNING,
                                           f"错误率偏高: {metric.value:.1%}")
    
    async def _generate_component_alerts(self, component: str, health_data: Dict[str, Any]):
        """生成组件告警"""
        status = health_data.get('status', 'healthy')
        
        if status == 'error':
            issues = health_data.get('issues', ['未知错误'])
            message = f"{component} 组件错误: {', '.join(issues)}"
            await self._create_alert(component, AlertLevel.ERROR, message)
        
        elif status == 'warning':
            issues = health_data.get('issues', ['性能警告'])
            message = f"{component} 组件警告: {', '.join(issues)}"
            await self._create_alert(component, AlertLevel.WARNING, message)
    
    async def _create_alert(self, component: str, level: AlertLevel, message: str):
        """创建告警"""
        # 检查是否已存在相同告警（避免重复）
        alert_key = f"{component}_{level.value}_{hash(message) % 10000}"
        
        if alert_key in self.active_alerts:
            return  # 避免重复告警
        
        alert = HealthAlert(
            alert_id=alert_key,
            component=component,
            level=level,
            message=message,
            timestamp=datetime.now()
        )
        
        self.active_alerts[alert_key] = alert
        
        # 显示新告警
        level_icons = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🔴"
        }
        
        icon = level_icons.get(level, "❓")
        print(f"{icon} [{level.value.upper()}] {component}: {message}")
    
    async def _process_alert_resolution(self):
        """处理告警解决"""
        current_time = datetime.now()
        resolved_alerts = []
        
        for alert_key, alert in list(self.active_alerts.items()):
            # 简单的告警自动解决逻辑（实际应该基于真实状态检查）
            alert_age = (current_time - alert.timestamp).total_seconds()
            
            # 模拟告警自动解决（60秒后随机解决）
            if alert_age > 60 and len(self.active_alerts) > 3:
                import random
                if random.random() < 0.3:  # 30%概率解决
                    alert.resolved = True
                    alert.resolved_at = current_time
                    
                    self.resolved_alerts.append(alert)
                    resolved_alerts.append(alert_key)
                    
                    print(f"✅ 告警已解决: [{alert.component}] {alert.message}")
        
        # 移除已解决的告警
        for alert_key in resolved_alerts:
            del self.active_alerts[alert_key]
    
    async def _display_alert_summary(self):
        """显示告警摘要"""
        if not self.active_alerts:
            return
        
        # 每5分钟显示一次摘要
        if not hasattr(self, '_last_summary_time'):
            self._last_summary_time = datetime.now()
        
        now = datetime.now()
        if (now - self._last_summary_time).total_seconds() < 300:  # 5分钟
            return
        
        self._last_summary_time = now
        
        print(f"\n📢 活跃告警摘要 ({len(self.active_alerts)} 个):")
        
        # 按级别分组
        by_level = defaultdict(list)
        for alert in self.active_alerts.values():
            by_level[alert.level].append(alert)
        
        level_order = [AlertLevel.CRITICAL, AlertLevel.ERROR, AlertLevel.WARNING, AlertLevel.INFO]
        
        for level in level_order:
            alerts = by_level.get(level, [])
            if alerts:
                print(f"   {level.value.upper()}: {len(alerts)} 个")
                for alert in alerts[:3]:  # 显示前3个
                    age = (now - alert.timestamp).total_seconds() / 60
                    print(f"     - [{alert.component}] {alert.message[:50]}... ({age:.0f}分钟前)")
                
                if len(alerts) > 3:
                    print(f"     ... 还有 {len(alerts) - 3} 个告警")
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """生成健康报告"""
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds()
        
        # 最近的系统指标
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # 性能指标摘要
        performance_summary = {}
        for metric_name, metric_history in self.performance_metrics.items():
            if metric_history:
                values = [m.value for m in metric_history]
                performance_summary[metric_name] = {
                    'current': values[-1],
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        # 告警统计
        alert_stats = {
            'active_alerts': len(self.active_alerts),
            'resolved_alerts': len(self.resolved_alerts),
            'alerts_by_level': {}
        }
        
        for alert in self.active_alerts.values():
            level = alert.level.value
            alert_stats['alerts_by_level'][level] = alert_stats['alerts_by_level'].get(level, 0) + 1
        
        # 组件健康摘要
        component_summary = {}
        for component, health in self.component_health.items():
            component_summary[component] = {
                'status': health.get('status', 'unknown'),
                'issues_count': len(health.get('issues', [])),
                'availability': health.get('availability', 0),
                'response_time': health.get('response_time', 0)
            }
        
        return {
            'report_time': current_time.isoformat(),
            'uptime_seconds': uptime,
            'monitoring_active': self.monitoring_active,
            'system_metrics': asdict(latest_metrics) if latest_metrics else None,
            'performance_summary': performance_summary,
            'alert_statistics': alert_stats,
            'component_health': component_summary,
            'data_points_collected': len(self.metrics_history)
        }
    
    def display_dashboard(self):
        """显示监控面板"""
        if not self.metrics_history:
            print("📊 暂无监控数据")
            return
        
        latest_metrics = self.metrics_history[-1]
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("📊 Claude Echo 系统健康监控面板")
        print("="*80)
        
        # 系统状态概览
        print(f"\n🖥️ 系统状态 (运行时间: {uptime/3600:.1f}小时):")
        print(f"   CPU: {latest_metrics.cpu_percent:.1f}%")
        print(f"   内存: {latest_metrics.memory_percent:.1f}% ({latest_metrics.memory_mb:.0f}MB)")
        print(f"   磁盘: {latest_metrics.disk_usage_percent:.1f}%")
        print(f"   连接数: {latest_metrics.active_connections}")
        
        # 性能指标
        if self.performance_metrics:
            print(f"\n⚡ 性能指标:")
            for metric_name, metric_history in self.performance_metrics.items():
                if metric_history:
                    latest = metric_history[-1]
                    print(f"   {metric_name}: {latest.value:.2f} {latest.unit}")
        
        # 组件健康
        print(f"\n🔧 组件健康 ({len(self.component_health)}个组件):")
        for component, health in self.component_health.items():
            status = health.get('status', 'unknown')
            status_icons = {'healthy': '✅', 'warning': '⚠️', 'error': '❌', 'unknown': '❓'}
            icon = status_icons.get(status, '❓')
            
            issues = health.get('issues', [])
            issues_text = f" ({len(issues)} 问题)" if issues else ""
            
            print(f"   {icon} {component}: {status}{issues_text}")
        
        # 告警状态
        if self.active_alerts:
            print(f"\n🚨 活跃告警 ({len(self.active_alerts)}个):")
            
            # 按级别分组显示
            by_level = defaultdict(list)
            for alert in self.active_alerts.values():
                by_level[alert.level].append(alert)
            
            level_icons = {
                AlertLevel.CRITICAL: "🔴",
                AlertLevel.ERROR: "❌",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.INFO: "ℹ️"
            }
            
            for level, alerts in by_level.items():
                icon = level_icons.get(level, "❓")
                print(f"   {icon} {level.value.upper()}: {len(alerts)} 个")
        else:
            print(f"\n✅ 无活跃告警")
        
        print("\n" + "="*80)
    
    def get_metrics_json(self) -> str:
        """获取监控数据的JSON格式"""
        try:
            report = asyncio.run(self.generate_health_report())
            return json.dumps(report, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)


async def run_monitoring_demo():
    """运行监控演示"""
    print("📊 Claude Echo 系统健康监控演示")
    print("Integration Agent - 系统健康监控和性能基准测试")
    print("=" * 60)
    
    # 创建监控器
    monitor = SystemHealthMonitor()
    
    try:
        print("🚀 启动监控演示...")
        
        # 启动监控（在后台运行）
        monitoring_task = asyncio.create_task(monitor.start_monitoring())
        
        # 运行2分钟的演示
        demo_duration = 120  # 秒
        dashboard_interval = 15  # 秒
        
        for i in range(0, demo_duration, dashboard_interval):
            await asyncio.sleep(dashboard_interval)
            
            print(f"\n⏰ 演示进度: {i+dashboard_interval}/{demo_duration}秒")
            monitor.display_dashboard()
            
            # 显示JSON报告（每60秒一次）
            if (i + dashboard_interval) % 60 == 0:
                print(f"\n📄 健康报告JSON:")
                report_json = monitor.get_metrics_json()
                # 只显示前500字符避免输出过长
                print(report_json[:500] + "..." if len(report_json) > 500 else report_json)
        
        # 停止监控
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        print("\n✅ 监控演示完成")
        
        # 生成最终报告
        final_report = await monitor.generate_health_report()
        print(f"\n📋 最终监控报告:")
        print(f"   数据点: {final_report['data_points_collected']}")
        print(f"   运行时间: {final_report['uptime_seconds']/60:.1f}分钟")
        print(f"   活跃告警: {final_report['alert_statistics']['active_alerts']}")
        print(f"   已解决告警: {final_report['alert_statistics']['resolved_alerts']}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程出错: {e}")
        import traceback
        traceback.print_exc()


async def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n⚡ 性能基准测试")
    print("-" * 40)
    
    benchmarks = [
        ("事件处理吞吐量", "events/sec", lambda: asyncio.run(_benchmark_event_throughput())),
        ("响应时间测试", "ms", lambda: asyncio.run(_benchmark_response_time())),
        ("内存使用效率", "MB", lambda: asyncio.run(_benchmark_memory_efficiency())),
        ("并发处理能力", "concurrent_ops", lambda: asyncio.run(_benchmark_concurrency()))
    ]
    
    results = {}
    
    for benchmark_name, unit, benchmark_func in benchmarks:
        print(f"🔍 执行 {benchmark_name}...")
        start_time = time.time()
        
        try:
            result = benchmark_func()
            execution_time = time.time() - start_time
            
            results[benchmark_name] = {
                'value': result,
                'unit': unit,
                'execution_time': execution_time
            }
            
            print(f"   结果: {result:.2f} {unit} ({execution_time:.2f}s)")
            
        except Exception as e:
            print(f"   ❌ 基准测试失败: {e}")
            results[benchmark_name] = {'error': str(e)}
    
    print(f"\n📊 基准测试结果摘要:")
    for name, result in results.items():
        if 'error' not in result:
            print(f"   {name}: {result['value']:.2f} {result['unit']}")
        else:
            print(f"   {name}: 测试失败")
    
    return results


async def _benchmark_event_throughput() -> float:
    """基准测试：事件处理吞吐量"""
    import random
    
    event_count = 1000
    start_time = time.time()
    
    # 模拟事件处理
    processed = 0
    for i in range(event_count):
        # 模拟处理延迟
        await asyncio.sleep(0.001)
        processed += 1
    
    end_time = time.time()
    
    throughput = processed / (end_time - start_time)
    return throughput


async def _benchmark_response_time() -> float:
    """基准测试：响应时间"""
    import random
    
    response_times = []
    
    for i in range(50):  # 50次测试
        start_time = time.time()
        
        # 模拟请求处理
        await asyncio.sleep(random.uniform(0.01, 0.1))
        
        response_time = (time.time() - start_time) * 1000  # 毫秒
        response_times.append(response_time)
    
    # 返回平均响应时间
    return sum(response_times) / len(response_times)


async def _benchmark_memory_efficiency() -> float:
    """基准测试：内存使用效率"""
    import gc
    
    # 获取初始内存状态
    gc.collect()
    initial_objects = len(gc.get_objects())
    
    # 创建对象进行测试
    test_objects = []
    for i in range(1000):
        obj = {
            'id': i,
            'data': f"test_data_{i}" * 10,
            'timestamp': time.time()
        }
        test_objects.append(obj)
    
    peak_objects = len(gc.get_objects())
    
    # 清理对象
    test_objects.clear()
    gc.collect()
    
    final_objects = len(gc.get_objects())
    
    # 计算内存使用效率（对象创建和清理的比率）
    objects_created = peak_objects - initial_objects
    objects_cleaned = peak_objects - final_objects
    
    efficiency = (objects_cleaned / objects_created * 100) if objects_created > 0 else 100
    
    return efficiency


async def _benchmark_concurrency() -> float:
    """基准测试：并发处理能力"""
    import random
    
    async def concurrent_task(task_id: int):
        # 模拟异步任务
        await asyncio.sleep(random.uniform(0.01, 0.05))
        return task_id
    
    # 测试不同并发级别
    concurrency_levels = [10, 20, 50, 100]
    best_throughput = 0
    
    for concurrency in concurrency_levels:
        start_time = time.time()
        
        tasks = [concurrent_task(i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        successful_tasks = sum(1 for r in results if not isinstance(r, Exception))
        throughput = successful_tasks / (end_time - start_time)
        
        best_throughput = max(best_throughput, throughput)
    
    return best_throughput


async def main():
    """主函数"""
    print("🚀 Claude Echo 系统健康监控和性能测试")
    print("Integration Agent - 综合系统监控解决方案")
    print("=" * 60)
    
    # 选择演示模式
    print("\n请选择演示模式:")
    print("1. 系统健康监控演示 (2分钟)")
    print("2. 性能基准测试")
    print("3. 完整演示 (监控 + 基准测试)")
    
    try:
        # 默认运行完整演示
        choice = "3"  # 可以改为input("请输入选择 (1-3): ") 以支持交互
        
        if choice in ["1", "3"]:
            await run_monitoring_demo()
        
        if choice in ["2", "3"]:
            await run_performance_benchmark()
        
        print("\n🎉 系统健康监控和性能测试完成")
        
    except KeyboardInterrupt:
        print("\n⏹️ 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())