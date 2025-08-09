#!/usr/bin/env python3
"""
Phase 2 AI Agents System Quality Analysis

Simple version that analyzes the completed AI Agents system architecture.
"""

import os
from pathlib import Path
from datetime import datetime

def count_lines(file_path):
    """Count lines in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except:
        return 0

def analyze_agent_file(file_path):
    """Analyze a single agent file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = len(content.split('\n'))
        
        # Count methods
        method_count = content.count('def ')
        async_methods = content.count('async def')
        
        # Count capabilities
        capabilities = content.count('AgentCapability(')
        
        # Check for key patterns
        has_docstring = '"""' in content
        has_error_handling = 'try:' in content and 'except' in content
        inherits_base_agent = 'BaseAgent' in content
        
        return {
            'lines': lines,
            'methods': method_count,
            'async_methods': async_methods,
            'capabilities': capabilities,
            'has_docstring': has_docstring,
            'has_error_handling': has_error_handling,
            'inherits_base_agent': inherits_base_agent
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    """Main analysis function"""
    print("Claude Echo - Phase 2 AI Agents System Analysis")
    print("=" * 50)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    project_root = Path(".")
    agents_path = project_root / "src" / "agents"
    
    if not agents_path.exists():
        print("ERROR: src/agents directory not found")
        return
    
    print("=== PROJECT STRUCTURE ANALYSIS ===")
    
    # Define expected files
    expected_files = {
        "base_agent.py": "BaseAgent foundation class",
        "agent_manager.py": "Agent management center",
        "agent_types.py": "Agent type definitions", 
        "coordinator.py": "Coordinator Agent",
        "task_planner.py": "Task Planner Agent",
        "presence_monitor.py": "Presence Monitor Agent", 
        "auto_worker.py": "Auto Worker Agent",
        "security_guardian.py": "Security Guardian Agent",
        "handover_manager.py": "Handover Manager Agent",
        "session_manager.py": "Session Manager Agent",
        "integration_test.py": "Integration test suite"
    }
    
    found_files = 0
    total_lines = 0
    
    for filename, description in expected_files.items():
        file_path = agents_path / filename
        if file_path.exists():
            lines = count_lines(file_path)
            total_lines += lines
            found_files += 1
            print(f"[OK] {filename}: {lines} lines - {description}")
        else:
            print(f"[MISSING] {filename} - {description}")
    
    completion_rate = (found_files / len(expected_files)) * 100
    print(f"\nProject Structure: {completion_rate:.1f}% complete")
    print(f"Total Code Lines: {total_lines:,}")
    
    print("\n=== CORE AGENTS ANALYSIS ===")
    
    # Analyze the 7 core agents
    core_agents = [
        ("coordinator.py", "Coordinator"),
        ("task_planner.py", "Task Planner"),
        ("presence_monitor.py", "Presence Monitor"),
        ("auto_worker.py", "Auto Worker"), 
        ("security_guardian.py", "Security Guardian"),
        ("handover_manager.py", "Handover Manager"),
        ("session_manager.py", "Session Manager")
    ]
    
    total_capabilities = 0
    analyzed_agents = 0
    total_methods = 0
    
    for filename, agent_name in core_agents:
        file_path = agents_path / filename
        if file_path.exists():
            analysis = analyze_agent_file(file_path)
            if 'error' not in analysis:
                analyzed_agents += 1
                total_capabilities += analysis['capabilities']
                total_methods += analysis['methods']
                
                print(f"[OK] {agent_name}:")
                print(f"     Lines: {analysis['lines']}")
                print(f"     Methods: {analysis['methods']} (async: {analysis['async_methods']})")
                print(f"     Capabilities: {analysis['capabilities']}")
                print(f"     Docstring: {'Yes' if analysis['has_docstring'] else 'No'}")
                print(f"     Error Handling: {'Yes' if analysis['has_error_handling'] else 'No'}")
                print(f"     Inherits BaseAgent: {'Yes' if analysis['inherits_base_agent'] else 'No'}")
            else:
                print(f"[ERROR] {agent_name}: {analysis['error']}")
        else:
            print(f"[MISSING] {agent_name}")
    
    agent_success_rate = (analyzed_agents / len(core_agents)) * 100
    print(f"\nCore Agents Analysis: {agent_success_rate:.1f}% success")
    print(f"Total Capabilities: {total_capabilities}")
    print(f"Total Methods: {total_methods}")
    
    print("\n=== ARCHITECTURE COMPLIANCE ===")
    
    # Check BaseAgent compliance
    base_agent_path = agents_path / "base_agent.py"
    if base_agent_path.exists():
        base_analysis = analyze_agent_file(base_agent_path)
        if 'error' not in base_analysis:
            print(f"[OK] BaseAgent: {base_analysis['lines']} lines, {base_analysis['methods']} methods")
        else:
            print(f"[ERROR] BaseAgent: {base_analysis['error']}")
    else:
        print("[MISSING] BaseAgent foundation class")
    
    # Check integration support
    print("\n=== INTEGRATION READINESS ===")
    
    integration_checks = [
        ("src/core/event_system.py", "Event System"),
        ("src/agents/agent_manager.py", "Agent Manager"),
        ("src/core/config_manager.py", "Config Manager"),
        ("src/agents/integration_test.py", "Integration Test")
    ]
    
    integration_ready = 0
    for check_path, component in integration_checks:
        if Path(check_path).exists():
            integration_ready += 1
            print(f"[OK] {component}")
        else:
            print(f"[MISSING] {component}")
    
    integration_score = (integration_ready / len(integration_checks)) * 100
    
    print(f"\nIntegration Readiness: {integration_score:.1f}%")
    
    print("\n=== OVERALL ASSESSMENT ===")
    
    # Calculate overall score
    overall_score = (completion_rate * 0.3 + agent_success_rate * 0.4 + integration_score * 0.3)
    
    print(f"Project Structure: {completion_rate:.1f}%")
    print(f"Core Agents: {agent_success_rate:.1f}%") 
    print(f"Integration Ready: {integration_score:.1f}%")
    print(f"OVERALL SCORE: {overall_score:.1f}%")
    
    # Assessment conclusion
    if overall_score >= 90:
        conclusion = "EXCELLENT - Production ready"
    elif overall_score >= 80:
        conclusion = "GOOD - Minor optimizations needed"
    elif overall_score >= 70:
        conclusion = "ACCEPTABLE - Some components need work"
    else:
        conclusion = "NEEDS IMPROVEMENT - Major issues detected"
    
    print(f"\nCONCLUSION: {conclusion}")
    print(f"System Scale: {total_lines:,} lines, {total_capabilities} capabilities")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()