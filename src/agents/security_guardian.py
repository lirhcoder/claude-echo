"""Security Guardian Agent - Security Validation and Risk Management

The Security Guardian is responsible for:
- Multi-layered security risk assessment
- Dynamic permission validation and control
- Dangerous operation intelligent interception
- Complete security audit logging
"""

import asyncio
import hashlib
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from ..core.event_system import EventSystem
from ..core.types import RiskLevel, Task
from .base_agent import BaseAgent
from .agent_types import (
    AgentType, AgentRequest, AgentResponse, AgentEvent, AgentCapability
)


class SecurityAction(Enum):
    """Security actions that can be taken"""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"
    QUARANTINE = "quarantine"
    MONITOR = "monitor"


class SecurityLevel(Enum):
    """Security levels for the system"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    risk_threshold: RiskLevel
    allowed_operations: Set[str] = field(default_factory=set)
    blocked_operations: Set[str] = field(default_factory=set)
    requires_confirmation: Set[str] = field(default_factory=set)
    active: bool = True


@dataclass
class SecurityAssessment:
    """Security risk assessment result"""
    assessment_id: str
    target: str
    risk_level: RiskLevel
    risk_score: float
    action: SecurityAction
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityAuditEntry:
    """Security audit log entry"""
    entry_id: str
    timestamp: datetime
    event_type: str
    source: str
    target: str
    action: SecurityAction
    risk_level: RiskLevel
    details: Dict[str, Any] = field(default_factory=dict)


class SecurityGuardian(BaseAgent):
    """
    Security validation and risk management agent.
    
    Provides comprehensive security assessment and enforcement
    for all system operations.
    """
    
    def __init__(self, event_system: EventSystem, config: Optional[Dict[str, Any]] = None):
        super().__init__("security_guardian", event_system, config)
        
        # Security configuration
        self._security_level = SecurityLevel.MEDIUM
        self._policies: Dict[str, SecurityPolicy] = {}
        self._audit_log: List[SecurityAuditEntry] = []
        self._max_audit_entries = self.config.get('max_audit_entries', 10000)
        
        # Risk assessment
        self._risk_cache: Dict[str, SecurityAssessment] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # Threat detection
        self._suspicious_patterns: Set[str] = set()
        self._blocked_operations: Set[str] = set()
        self._quarantine_list: Set[str] = set()
        
        # Statistics
        self._security_stats = {
            'assessments_performed': 0,
            'operations_blocked': 0,
            'confirmations_required': 0,
            'quarantined_items': 0
        }
        
        # Initialize default policies
        self._initialize_default_policies()
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.SECURITY_GUARDIAN
    
    @property
    def name(self) -> str:
        return "Security Guardian"
    
    @property
    def description(self) -> str:
        return "Security validation and risk management agent"
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="assess_risk",
                description="Perform comprehensive security risk assessment",
                input_types=["task", "operation", "request"],
                output_types=["security_assessment"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=200
            ),
            AgentCapability(
                name="validate_permission",
                description="Validate permissions for operations",
                input_types=["permission_request"],
                output_types=["permission_result"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=100
            ),
            AgentCapability(
                name="audit_operation",
                description="Audit and log security-relevant operations",
                input_types=["operation_details"],
                output_types=["audit_result"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=50
            ),
            AgentCapability(
                name="update_policy",
                description="Update security policies",
                input_types=["policy_definition"],
                output_types=["policy_status"],
                risk_level=RiskLevel.HIGH,
                execution_time_ms=300
            ),
            AgentCapability(
                name="get_security_status",
                description="Get current security status and metrics",
                input_types=["status_request"],
                output_types=["security_status"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=100
            ),
            AgentCapability(
                name="quarantine_item",
                description="Quarantine suspicious items or operations",
                input_types=["quarantine_request"],
                output_types=["quarantine_result"],
                risk_level=RiskLevel.HIGH,
                execution_time_ms=150
            )
        ]
    
    async def _initialize_agent(self) -> None:
        """Initialize security guardian specific functionality."""
        self.logger.info("Initializing Security Guardian agent")
        
        # Load security configuration
        await self._load_security_config()
        
        # Start audit log cleanup task
        cleanup_task = asyncio.create_task(self._audit_cleanup_loop())
        self._background_tasks.add(cleanup_task)
        
        # Start threat monitoring task
        monitor_task = asyncio.create_task(self._threat_monitoring_loop())
        self._background_tasks.add(monitor_task)
        
        self.logger.info("Security Guardian initialization complete")
    
    async def _process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming requests."""
        capability = request.target_capability
        start_time = datetime.now()
        
        try:
            if capability == "assess_risk":
                result = await self._assess_risk(request.parameters)
            elif capability == "validate_permission":
                result = await self._validate_permission(request.parameters)
            elif capability == "audit_operation":
                result = await self._audit_operation(request.parameters)
            elif capability == "update_policy":
                result = await self._update_policy(request.parameters)
            elif capability == "get_security_status":
                result = await self._get_security_status(request.parameters)
            elif capability == "quarantine_item":
                result = await self._quarantine_item(request.parameters)
            else:
                raise ValueError(f"Unknown capability: {capability}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {capability}: {e}")
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle agent events."""
        try:
            if event.event_type == "security.threat.detected":
                await self._handle_threat_detection(event)
            elif event.event_type == "security.policy.violation":
                await self._handle_policy_violation(event)
            elif event.event_type == "system.security.alert":
                await self._handle_security_alert(event)
                
        except Exception as e:
            await self._handle_error(e, f"event_handling_{event.event_type}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup security guardian resources."""
        # Save audit log if needed
        await self._save_audit_log()
        self.logger.info("Security Guardian cleanup complete")
    
    # Private implementation methods
    
    async def _assess_risk(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment."""
        target = parameters.get("target", "unknown")
        operation = parameters.get("operation", "unknown")
        context = parameters.get("context", {})
        
        # Check cache first
        cache_key = self._generate_cache_key(target, operation, context)
        if cache_key in self._risk_cache:
            cached_assessment = self._risk_cache[cache_key]
            if datetime.now() - cached_assessment.assessed_at < self._cache_ttl:
                self._security_stats['assessments_performed'] += 1
                return {"security_assessment": self._assessment_to_dict(cached_assessment)}
        
        # Perform new assessment
        assessment = await self._perform_risk_assessment(target, operation, context)
        
        # Cache the result
        self._risk_cache[cache_key] = assessment
        
        # Update statistics
        self._security_stats['assessments_performed'] += 1
        
        # Log the assessment
        await self._log_security_event(
            "risk_assessment",
            target,
            assessment.action,
            assessment.risk_level,
            {"operation": operation, "risk_score": assessment.risk_score}
        )
        
        return {"security_assessment": self._assessment_to_dict(assessment)}
    
    async def _validate_permission(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate permissions for operations."""
        operation = parameters.get("operation", "")
        user = parameters.get("user", "unknown")
        context = parameters.get("context", {})
        
        # Check against policies
        allowed = await self._check_policies(operation, user, context)
        
        # Check quarantine list
        if operation in self._quarantine_list:
            allowed = False
            reason = "Operation is quarantined"
        elif operation in self._blocked_operations:
            allowed = False
            reason = "Operation is blocked by security policy"
        else:
            reason = "Operation permitted" if allowed else "Operation denied by security policy"
        
        result = {
            "permission_result": {
                "allowed": allowed,
                "operation": operation,
                "user": user,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Log the permission check
        await self._log_security_event(
            "permission_check",
            operation,
            SecurityAction.ALLOW if allowed else SecurityAction.DENY,
            RiskLevel.LOW,
            {"user": user, "reason": reason}
        )
        
        return result
    
    async def _audit_operation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Audit and log security-relevant operations."""
        operation = parameters.get("operation", "unknown")
        user = parameters.get("user", "unknown")
        details = parameters.get("details", {})
        risk_level = RiskLevel(parameters.get("risk_level", "low"))
        
        # Create audit entry
        entry = SecurityAuditEntry(
            entry_id=self._generate_audit_id(),
            timestamp=datetime.now(),
            event_type="operation_audit",
            source=user,
            target=operation,
            action=SecurityAction.MONITOR,
            risk_level=risk_level,
            details=details
        )
        
        # Add to audit log
        self._audit_log.append(entry)
        
        # Maintain audit log size
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]
        
        return {
            "audit_result": {
                "audit_id": entry.entry_id,
                "status": "logged",
                "timestamp": entry.timestamp.isoformat()
            }
        }
    
    async def _update_policy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update security policies."""
        policy_data = parameters.get("policy")
        if not policy_data:
            raise ValueError("Policy data required")
        
        policy = SecurityPolicy(**policy_data)
        self._policies[policy.policy_id] = policy
        
        # Log policy update
        await self._log_security_event(
            "policy_update",
            policy.policy_id,
            SecurityAction.MONITOR,
            RiskLevel.HIGH,
            {"policy_name": policy.name, "active": policy.active}
        )
        
        return {
            "policy_status": {
                "policy_id": policy.policy_id,
                "status": "updated",
                "active": policy.active,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def _get_security_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get current security status."""
        status = {
            "security_level": self._security_level.value,
            "active_policies": len([p for p in self._policies.values() if p.active]),
            "total_policies": len(self._policies),
            "audit_entries": len(self._audit_log),
            "quarantined_items": len(self._quarantine_list),
            "blocked_operations": len(self._blocked_operations),
            "statistics": self._security_stats,
            "recent_assessments": len([
                a for a in self._risk_cache.values()
                if datetime.now() - a.assessed_at < timedelta(hours=1)
            ])
        }
        
        return {"security_status": status}
    
    async def _quarantine_item(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Quarantine suspicious items or operations."""
        item = parameters.get("item", "")
        reason = parameters.get("reason", "Security risk detected")
        duration = parameters.get("duration_hours", 24)
        
        if not item:
            raise ValueError("Item to quarantine required")
        
        # Add to quarantine list
        self._quarantine_list.add(item)
        self._security_stats['quarantined_items'] += 1
        
        # Schedule removal from quarantine
        asyncio.create_task(self._schedule_quarantine_removal(item, duration))
        
        # Log quarantine action
        await self._log_security_event(
            "quarantine",
            item,
            SecurityAction.QUARANTINE,
            RiskLevel.HIGH,
            {"reason": reason, "duration_hours": duration}
        )
        
        return {
            "quarantine_result": {
                "item": item,
                "status": "quarantined",
                "reason": reason,
                "duration_hours": duration,
                "quarantined_at": datetime.now().isoformat()
            }
        }
    
    async def _perform_risk_assessment(self, target: str, operation: str, context: Dict[str, Any]) -> SecurityAssessment:
        """Perform detailed risk assessment."""
        risk_score = 0.0
        risk_factors = []
        recommendations = []
        
        # Assess operation type risk
        high_risk_operations = ["delete", "format", "install", "modify_system", "network_access"]
        medium_risk_operations = ["copy", "move", "create", "read_sensitive"]
        
        if any(op in operation.lower() for op in high_risk_operations):
            risk_score += 0.6
            risk_factors.append(f"High-risk operation type: {operation}")
            recommendations.append("Consider requiring user confirmation")
        elif any(op in operation.lower() for op in medium_risk_operations):
            risk_score += 0.3
            risk_factors.append(f"Medium-risk operation type: {operation}")
        
        # Assess target risk
        sensitive_targets = ["system", "registry", "config", "password", "credential"]
        if any(target_type in target.lower() for target_type in sensitive_targets):
            risk_score += 0.4
            risk_factors.append(f"Sensitive target: {target}")
            recommendations.append("Implement additional access controls")
        
        # Assess context risk
        if context.get("elevated_privileges"):
            risk_score += 0.3
            risk_factors.append("Elevated privileges requested")
        
        if context.get("external_network_access"):
            risk_score += 0.2
            risk_factors.append("External network access required")
        
        # Check against patterns
        if operation in self._suspicious_patterns:
            risk_score += 0.5
            risk_factors.append("Matches suspicious pattern")
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
            action = SecurityAction.DENY
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
            action = SecurityAction.REQUIRE_CONFIRMATION
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
            action = SecurityAction.MONITOR
        else:
            risk_level = RiskLevel.LOW
            action = SecurityAction.ALLOW
        
        # Override based on policies
        for policy in self._policies.values():
            if not policy.active:
                continue
                
            if operation in policy.blocked_operations:
                action = SecurityAction.DENY
                risk_level = RiskLevel.HIGH
                risk_factors.append(f"Blocked by policy: {policy.name}")
            elif operation in policy.requires_confirmation:
                if action == SecurityAction.ALLOW:
                    action = SecurityAction.REQUIRE_CONFIRMATION
                risk_factors.append(f"Requires confirmation per policy: {policy.name}")
        
        return SecurityAssessment(
            assessment_id=self._generate_assessment_id(),
            target=target,
            risk_level=risk_level,
            risk_score=risk_score,
            action=action,
            reasons=risk_factors,
            recommendations=recommendations
        )
    
    async def _check_policies(self, operation: str, user: str, context: Dict[str, Any]) -> bool:
        """Check operation against security policies."""
        for policy in self._policies.values():
            if not policy.active:
                continue
            
            # Check blocked operations
            if operation in policy.blocked_operations:
                return False
            
            # Check allowed operations (if specified)
            if policy.allowed_operations and operation not in policy.allowed_operations:
                return False
        
        return True
    
    def _initialize_default_policies(self) -> None:
        """Initialize default security policies."""
        # High-risk operations policy
        high_risk_policy = SecurityPolicy(
            policy_id="high_risk_operations",
            name="High Risk Operations",
            description="Policy for high-risk system operations",
            risk_threshold=RiskLevel.HIGH,
            blocked_operations={"format_disk", "delete_system_files", "modify_boot_sector"},
            requires_confirmation={"install_software", "modify_registry", "network_config"}
        )
        self._policies[high_risk_policy.policy_id] = high_risk_policy
        
        # Data protection policy
        data_policy = SecurityPolicy(
            policy_id="data_protection",
            name="Data Protection",
            description="Policy for protecting sensitive data",
            risk_threshold=RiskLevel.MEDIUM,
            requires_confirmation={"access_credentials", "read_private_files", "export_data"}
        )
        self._policies[data_policy.policy_id] = data_policy
    
    async def _load_security_config(self) -> None:
        """Load security configuration."""
        # Load configuration from config manager or file
        security_config = self.config.get("security", {})
        
        # Set security level
        level = security_config.get("level", "medium")
        try:
            self._security_level = SecurityLevel(level)
        except ValueError:
            self.logger.warning(f"Invalid security level: {level}")
        
        # Load blocked operations
        blocked_ops = security_config.get("blocked_operations", [])
        self._blocked_operations.update(blocked_ops)
        
        # Load suspicious patterns
        patterns = security_config.get("suspicious_patterns", [])
        self._suspicious_patterns.update(patterns)
    
    async def _save_audit_log(self) -> None:
        """Save audit log to persistent storage."""
        try:
            # This would save to a secure persistent store
            self.logger.info(f"Saving {len(self._audit_log)} audit log entries")
        except Exception as e:
            self.logger.error(f"Failed to save audit log: {e}")
    
    def _generate_cache_key(self, target: str, operation: str, context: Dict[str, Any]) -> str:
        """Generate cache key for risk assessment."""
        data = f"{target}:{operation}:{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _generate_assessment_id(self) -> str:
        """Generate unique assessment ID."""
        return f"assess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000:04d}"
    
    def _generate_audit_id(self) -> str:
        """Generate unique audit entry ID."""
        return f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000:04d}"
    
    def _assessment_to_dict(self, assessment: SecurityAssessment) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "assessment_id": assessment.assessment_id,
            "target": assessment.target,
            "risk_level": assessment.risk_level.value,
            "risk_score": assessment.risk_score,
            "action": assessment.action.value,
            "reasons": assessment.reasons,
            "recommendations": assessment.recommendations,
            "assessed_at": assessment.assessed_at.isoformat()
        }
    
    async def _log_security_event(self, event_type: str, target: str, action: SecurityAction, 
                                 risk_level: RiskLevel, details: Dict[str, Any]) -> None:
        """Log a security event."""
        entry = SecurityAuditEntry(
            entry_id=self._generate_audit_id(),
            timestamp=datetime.now(),
            event_type=event_type,
            source=self.agent_id,
            target=target,
            action=action,
            risk_level=risk_level,
            details=details
        )
        
        self._audit_log.append(entry)
        
        # Emit security event
        await self._emit_event("security.event", {
            "event_type": event_type,
            "target": target,
            "action": action.value,
            "risk_level": risk_level.value,
            "details": details
        })
    
    async def _schedule_quarantine_removal(self, item: str, duration_hours: int) -> None:
        """Schedule removal of item from quarantine."""
        await asyncio.sleep(duration_hours * 3600)
        
        if item in self._quarantine_list:
            self._quarantine_list.remove(item)
            self._security_stats['quarantined_items'] -= 1
            
            await self._log_security_event(
                "quarantine_released",
                item,
                SecurityAction.MONITOR,
                RiskLevel.LOW,
                {"reason": "Quarantine period expired"}
            )
    
    async def _audit_cleanup_loop(self) -> None:
        """Background task to clean up old audit entries."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Remove old entries (keep last 30 days)
                cutoff = datetime.now() - timedelta(days=30)
                original_count = len(self._audit_log)
                
                self._audit_log = [
                    entry for entry in self._audit_log
                    if entry.timestamp > cutoff
                ]
                
                removed_count = original_count - len(self._audit_log)
                if removed_count > 0:
                    self.logger.info(f"Cleaned up {removed_count} old audit entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "audit_cleanup")
    
    async def _threat_monitoring_loop(self) -> None:
        """Background threat monitoring loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Analyze recent audit entries for threats
                recent_entries = [
                    entry for entry in self._audit_log
                    if datetime.now() - entry.timestamp < timedelta(hours=1)
                ]
                
                # Simple threat detection (could be more sophisticated)
                high_risk_events = [
                    entry for entry in recent_entries
                    if entry.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                ]
                
                if len(high_risk_events) > 5:  # More than 5 high-risk events in an hour
                    await self._emit_event("security.threat.detected", {
                        "threat_type": "high_risk_activity_spike",
                        "event_count": len(high_risk_events),
                        "time_window": "1_hour"
                    })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "threat_monitoring")
    
    async def _handle_threat_detection(self, event: AgentEvent) -> None:
        """Handle threat detection events."""
        threat_type = event.data.get("threat_type")
        severity = event.data.get("severity", "medium")
        
        self.logger.warning(f"Security threat detected: {threat_type} (severity: {severity})")
        
        # Take appropriate action based on threat
        if severity == "high":
            # Could escalate security level or block certain operations
            pass
    
    async def _handle_policy_violation(self, event: AgentEvent) -> None:
        """Handle policy violation events."""
        policy_id = event.data.get("policy_id")
        violation_type = event.data.get("violation_type")
        
        self.logger.warning(f"Policy violation detected: {policy_id} - {violation_type}")
        
        # Log the violation
        await self._log_security_event(
            "policy_violation",
            policy_id,
            SecurityAction.DENY,
            RiskLevel.HIGH,
            {"violation_type": violation_type}
        )
    
    async def _handle_security_alert(self, event: AgentEvent) -> None:
        """Handle security alert events."""
        alert_type = event.data.get("alert_type")
        details = event.data.get("details", {})
        
        self.logger.info(f"Security alert received: {alert_type}")
        
        # Process the alert based on type
        if alert_type == "suspicious_activity":
            # Could add to monitoring or quarantine
            pass