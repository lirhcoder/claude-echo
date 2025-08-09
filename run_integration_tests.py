#!/usr/bin/env python3
"""
Claude Echo 集成测试执行器
Integration Agent - 运行所有集成测试并生成综合报告

这个脚本会执行以下测试：
1. 综合集成测试 (comprehensive_integration_test.py)
2. 自动化部署测试 (automated_deployment.py)
3. 系统健康监控测试 (system_health_monitor.py)
4. 端到端功能验证 (end_to_end_validation.py)
"""

import asyncio
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class TestStatus(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    score: Optional[float] = None
    execution_time: Optional[float] = None
    output: Optional[str] = None
    error: Optional[str] = None


class IntegrationTestRunner:
    """集成测试执行器"""
    
    def __init__(self):
        self.test_results: Dict[str, TestResult] = {}
        self.start_time = datetime.now()
        
        # 定义测试套件
        self.test_suite = {
            "comprehensive_integration": {
                "file": "comprehensive_integration_test.py",
                "name": "综合集成测试",
                "description": "验证所有组件间的集成和协作",
                "timeout": 300,  # 5分钟
                "required": True
            },
            "automated_deployment": {
                "file": "automated_deployment.py", 
                "name": "自动化部署测试",
                "description": "测试系统的自动化部署和升级能力",
                "timeout": 180,  # 3分钟
                "required": False
            },
            "system_health_monitor": {
                "file": "system_health_monitor.py",
                "name": "系统健康监控测试", 
                "description": "验证系统健康监控和性能基准",
                "timeout": 180,  # 3分钟
                "required": False
            },
            "end_to_end_validation": {
                "file": "end_to_end_validation.py",
                "name": "端到端功能验证",
                "description": "完整的用户学习生命周期测试",
                "timeout": 240,  # 4分钟
                "required": True
            }
        }
        
        print("🚀 Claude Echo 集成测试执行器初始化完成")
    
    async def run_all_tests(self, selected_tests: Optional[List[str]] = None):
        """运行所有集成测试"""
        print("🔍 开始执行 Claude Echo 集成测试套件")
        print("=" * 70)
        
        # 确定要运行的测试
        tests_to_run = selected_tests or list(self.test_suite.keys())
        
        print(f"📋 测试计划: {len(tests_to_run)} 个测试")
        for test_key in tests_to_run:
            test_info = self.test_suite[test_key]
            print(f"   - {test_info['name']}: {test_info['description']}")
        
        print("\n" + "-" * 70)
        
        # 初始化测试结果
        for test_key in tests_to_run:
            test_info = self.test_suite[test_key]
            self.test_results[test_key] = TestResult(
                name=test_info['name'],
                status=TestStatus.NOT_STARTED
            )
        
        # 串行执行测试（避免资源冲突）
        for test_key in tests_to_run:
            await self._run_single_test(test_key)
        
        # 生成综合报告
        await self._generate_comprehensive_report()
    
    async def _run_single_test(self, test_key: str):
        """运行单个测试"""
        test_info = self.test_suite[test_key]
        test_result = self.test_results[test_key]
        
        print(f"\n🔍 开始测试: {test_info['name']}")
        print(f"   文件: {test_info['file']}")
        print(f"   超时: {test_info['timeout']}秒")
        
        test_file = Path(test_info['file'])
        
        if not test_file.exists():
            test_result.status = TestStatus.SKIPPED
            test_result.error = f"测试文件不存在: {test_file}"
            print(f"   ⏭️ 跳过测试: 文件不存在")
            return
        
        test_result.status = TestStatus.RUNNING
        start_time = time.time()
        
        try:
            # 运行测试
            print(f"   ⚡ 执行中...")
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(test_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=test_info['timeout']
                )
                
                execution_time = time.time() - start_time
                test_result.execution_time = execution_time
                
                if process.returncode == 0:
                    test_result.status = TestStatus.COMPLETED
                    test_result.output = stdout.decode('utf-8', errors='ignore')
                    
                    # 尝试从输出中提取分数
                    score = self._extract_score_from_output(test_result.output)
                    test_result.score = score
                    
                    print(f"   ✅ 测试完成 - 用时: {execution_time:.1f}秒" + 
                          (f", 评分: {score:.1%}" if score else ""))
                else:
                    test_result.status = TestStatus.FAILED
                    test_result.error = stderr.decode('utf-8', errors='ignore')
                    print(f"   ❌ 测试失败 - 返回码: {process.returncode}")
                    
            except asyncio.TimeoutError:
                test_result.status = TestStatus.FAILED
                test_result.error = f"测试超时 ({test_info['timeout']}秒)"
                print(f"   ⏰ 测试超时")
                
                # 终止进程
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except:
                    process.kill()
                    
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error = f"测试执行异常: {str(e)}"
            test_result.execution_time = time.time() - start_time
            print(f"   ❌ 测试异常: {str(e)}")
    
    def _extract_score_from_output(self, output: str) -> Optional[float]:
        """从测试输出中提取评分"""
        try:
            # 查找常见的评分格式
            import re
            
            patterns = [
                r'成功率[:：]\s*(\d+(?:\.\d+)?)[%％]',
                r'综合评分[:：]\s*(\d+(?:\.\d+)?)[%％]', 
                r'总体评分[:：]\s*(\d+(?:\.\d+)?)[%％]',
                r'整体评分[:：]\s*(\d+(?:\.\d+)?)[%％]',
                r'评分[:：]\s*(\d+(?:\.\d+)?)/1\.0+',
                r'score[:：]\s*(\d+(?:\.\d+)?)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    score_str = match.group(1)
                    score = float(score_str)
                    
                    # 如果是百分比格式，转换为小数
                    if '%' in match.group(0) or '％' in match.group(0):
                        score = score / 100.0
                    elif '/1.0' in match.group(0):
                        pass  # 已经是0-1范围
                    elif score > 1.0:
                        score = score / 100.0  # 假设是百分制
                    
                    return min(1.0, max(0.0, score))
            
            return None
            
        except Exception:
            return None
    
    async def _generate_comprehensive_report(self):
        """生成综合测试报告"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*90)
        print("📊 Claude Echo 智能学习系统 - 综合集成测试报告")
        print("Integration Agent - 最终系统集成验证报告")
        print("="*90)
        
        # 测试执行摘要
        total_tests = len(self.test_results)
        completed_tests = sum(1 for r in self.test_results.values() if r.status == TestStatus.COMPLETED)
        failed_tests = sum(1 for r in self.test_results.values() if r.status == TestStatus.FAILED)
        skipped_tests = sum(1 for r in self.test_results.values() if r.status == TestStatus.SKIPPED)
        
        print(f"\n📋 测试执行摘要:")
        print(f"   总测试数: {total_tests}")
        print(f"   ✅ 完成: {completed_tests}")
        print(f"   ❌ 失败: {failed_tests}")
        print(f"   ⏭️ 跳过: {skipped_tests}")
        print(f"   📈 完成率: {completed_tests/total_tests:.1%}")
        print(f"   ⏱️ 总执行时间: {total_time:.1f}秒")
        
        # 详细测试结果
        print(f"\n📊 详细测试结果:")
        
        for test_key, result in self.test_results.items():
            status_icons = {
                TestStatus.COMPLETED: "✅",
                TestStatus.FAILED: "❌",
                TestStatus.SKIPPED: "⏭️",
                TestStatus.NOT_STARTED: "⏸️",
                TestStatus.RUNNING: "⚡"
            }
            
            icon = status_icons.get(result.status, "❓")
            
            print(f"\n   {icon} {result.name}")
            print(f"      状态: {result.status.value}")
            
            if result.execution_time:
                print(f"      执行时间: {result.execution_time:.1f}秒")
            
            if result.score is not None:
                print(f"      评分: {result.score:.1%}")
            
            if result.error:
                print(f"      错误: {result.error[:100]}...")
            
            # 显示关键输出摘要
            if result.output and result.status == TestStatus.COMPLETED:
                key_metrics = self._extract_key_metrics(result.output)
                if key_metrics:
                    print(f"      关键指标:")
                    for metric, value in key_metrics.items():
                        print(f"        - {metric}: {value}")
        
        # 综合性能评估
        print(f"\n⚡ 系统性能评估:")
        
        # 计算综合评分
        scored_tests = [r for r in self.test_results.values() if r.score is not None]
        if scored_tests:
            avg_score = sum(r.score for r in scored_tests) / len(scored_tests)
            print(f"   📊 平均评分: {avg_score:.1%}")
        else:
            avg_score = 0.0
            print(f"   📊 平均评分: 无可用分数")
        
        # 性能指标
        total_exec_time = sum(r.execution_time for r in self.test_results.values() if r.execution_time)
        if total_exec_time > 0:
            print(f"   ⏱️ 测试执行效率: {total_tests/total_exec_time:.2f} 测试/秒")
        
        # 系统稳定性评估
        critical_failures = sum(1 for test_key, result in self.test_results.items() 
                               if result.status == TestStatus.FAILED and self.test_suite[test_key]['required'])
        
        stability_score = max(0.0, 1.0 - (critical_failures / max(1, total_tests)))
        print(f"   🛡️ 系统稳定性: {stability_score:.1%}")
        
        # 集成就绪度评估
        print(f"\n🎯 系统集成就绪度评估:")
        
        readiness_criteria = {
            "核心功能测试通过": completed_tests >= total_tests * 0.8,
            "关键测试无失败": critical_failures == 0,
            "平均性能达标": avg_score >= 0.75,
            "系统稳定性良好": stability_score >= 0.8,
            "测试执行顺利": failed_tests <= total_tests * 0.2
        }
        
        passed_criteria = sum(readiness_criteria.values())
        total_criteria = len(readiness_criteria)
        readiness_score = passed_criteria / total_criteria
        
        for criteria, passed in readiness_criteria.items():
            icon = "✅" if passed else "❌"
            print(f"   {icon} {criteria}")
        
        print(f"\n   📊 就绪度评分: {readiness_score:.1%} ({passed_criteria}/{total_criteria})")
        
        # 最终评估和建议
        print(f"\n🏆 最终系统评估:")
        
        # 计算综合评估分数
        final_score = (
            (completed_tests / total_tests) * 0.3 +  # 测试完成度
            avg_score * 0.4 +                        # 平均性能分数
            stability_score * 0.2 +                  # 系统稳定性
            readiness_score * 0.1                    # 就绪度
        )
        
        if final_score >= 0.9:
            verdict = "🎉 OUTSTANDING - 系统集成质量卓越，可立即部署到生产环境"
            recommendation = "系统已完全就绪，建议立即开始用户验收测试和生产部署"
        elif final_score >= 0.8:
            verdict = "✅ EXCELLENT - 系统集成质量优秀，可以部署到生产环境"
            recommendation = "系统表现优秀，建议在修复少量问题后部署到生产环境"
        elif final_score >= 0.7:
            verdict = "🟢 GOOD - 系统集成质量良好，经过小幅优化后可部署"
            recommendation = "系统基本就绪，建议修复已识别的问题后进行部署"
        elif final_score >= 0.6:
            verdict = "🟡 ACCEPTABLE - 系统基本可用，需要继续优化"
            recommendation = "系统可用性可接受，建议解决关键问题并重新测试"
        else:
            verdict = "🔴 NEEDS IMPROVEMENT - 系统需要重大改进"
            recommendation = "系统需要大幅改进，建议修复所有关键问题后重新进行集成测试"
        
        print(f"   结果: {verdict}")
        print(f"   建议: {recommendation}")
        print(f"   📊 综合评分: {final_score:.2f}/1.00 ({final_score*100:.1f}%)")
        
        # 下一步行动建议
        print(f"\n💡 下一步行动建议:")
        
        if failed_tests > 0:
            print("   🔧 优先修复失败的测试:")
            for test_key, result in self.test_results.items():
                if result.status == TestStatus.FAILED:
                    print(f"     - 修复 {result.name}")
                    if result.error:
                        print(f"       错误: {result.error[:80]}...")
        
        if avg_score < 0.8:
            print("   📈 性能优化建议:")
            print("     - 提升系统响应速度")
            print("     - 优化学习算法准确率")
            print("     - 改善用户体验指标")
        
        if readiness_score < 0.9:
            print("   🎯 就绪度改进:")
            print("     - 完善系统监控机制")
            print("     - 强化错误处理和恢复")
            print("     - 优化资源使用效率")
        
        print("   📋 持续改进:")
        print("     - 建立持续集成/部署流程")
        print("     - 实施用户反馈收集机制")  
        print("     - 定期进行性能基准测试")
        
        # 生成JSON报告
        await self._save_json_report(final_score, readiness_score)
        
        print("\n" + "="*90)
        print(f"🎯 Claude Echo 智能学习系统集成测试完成!")
        print(f"📊 系统综合评分: {final_score:.1%}")
        print(f"🏆 {verdict.split(' - ')[0]}")
        print("="*90)
    
    def _extract_key_metrics(self, output: str) -> Dict[str, str]:
        """从输出中提取关键指标"""
        metrics = {}
        
        try:
            import re
            
            # 提取常见指标
            patterns = {
                '准确率': r'准确率[:：]\s*(\d+(?:\.\d+)?[%％]?)',
                '响应时间': r'响应时间[:：]\s*(\d+(?:\.\d+)?\s*[秒s]?)',
                '内存使用': r'内存[:：]\s*(\d+(?:\.\d+)?\s*MB)',
                'CPU使用': r'CPU[:：]\s*(\d+(?:\.\d+)?[%％]?)',
                '吞吐量': r'吞吐量[:：]\s*(\d+(?:\.\d+)?)',
                '错误率': r'错误率[:：]\s*(\d+(?:\.\d+)?[%％]?)'
            }
            
            for metric_name, pattern in patterns.items():
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    metrics[metric_name] = matches[-1]  # 取最后一个匹配
            
        except Exception:
            pass
        
        return metrics
    
    async def _save_json_report(self, final_score: float, readiness_score: float):
        """保存JSON格式的详细报告"""
        report_data = {
            'timestamp': self.start_time.isoformat(),
            'execution_time_seconds': (datetime.now() - self.start_time).total_seconds(),
            'summary': {
                'total_tests': len(self.test_results),
                'completed_tests': sum(1 for r in self.test_results.values() if r.status == TestStatus.COMPLETED),
                'failed_tests': sum(1 for r in self.test_results.values() if r.status == TestStatus.FAILED),
                'skipped_tests': sum(1 for r in self.test_results.values() if r.status == TestStatus.SKIPPED),
                'average_score': sum(r.score for r in self.test_results.values() if r.score is not None) / max(1, len([r for r in self.test_results.values() if r.score is not None])),
                'final_score': final_score,
                'readiness_score': readiness_score
            },
            'test_results': {}
        }
        
        # 添加详细的测试结果
        for test_key, result in self.test_results.items():
            report_data['test_results'][test_key] = {
                'name': result.name,
                'status': result.status.value,
                'score': result.score,
                'execution_time': result.execution_time,
                'has_error': bool(result.error),
                'error_preview': result.error[:200] if result.error else None
            }
        
        # 保存报告
        report_file = Path(f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n✅ 详细测试报告已保存: {report_file}")
            
        except Exception as e:
            print(f"\n⚠️ 报告保存失败: {e}")
    
    def get_quick_status(self) -> str:
        """获取快速状态摘要"""
        completed = sum(1 for r in self.test_results.values() if r.status == TestStatus.COMPLETED)
        total = len(self.test_results)
        
        if completed == 0:
            return "🔄 测试未开始"
        elif completed == total:
            return "✅ 所有测试完成"
        else:
            return f"⚡ 进行中 ({completed}/{total})"


async def main():
    """主函数"""
    print("🚀 Claude Echo 智能学习系统 - 集成测试套件")
    print("Integration Agent - 综合系统集成验证")
    print("=" * 60)
    
    runner = IntegrationTestRunner()
    
    try:
        # 可以在这里选择特定的测试
        # selected_tests = ["comprehensive_integration", "end_to_end_validation"]
        selected_tests = None  # None 表示运行所有测试
        
        await runner.run_all_tests(selected_tests)
        
        print(f"\n🎉 集成测试套件执行完成!")
        print(f"📊 快速状态: {runner.get_quick_status()}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 集成测试被用户中断")
        print(f"📊 当前状态: {runner.get_quick_status()}")
    except Exception as e:
        print(f"\n❌ 集成测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())