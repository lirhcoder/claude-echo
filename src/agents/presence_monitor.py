"""Presence Monitor Agent - User State and Context Monitoring

The Presence Monitor is responsible for:
- Multi-dimensional user presence detection
- Application environment and context analysis  
- User behavior pattern recognition
- Intelligent state change notifications
"""

import asyncio
import psutil
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from ..core.event_system import EventSystem
from ..core.types import Context, RiskLevel
from .base_agent import BaseAgent
from .agent_types import (
    AgentType, AgentRequest, AgentResponse, AgentEvent, AgentCapability
)


class PresenceState(Enum):
    """User presence states"""
    ACTIVE = "active"
    IDLE = "idle"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class PresenceInfo:
    """User presence information"""
    user_id: str
    state: PresenceState
    last_activity: datetime
    current_application: Optional[str] = None
    screen_locked: bool = False
    mouse_position: tuple = (0, 0)
    active_windows: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class PresenceMonitor(BaseAgent):
    """
    Intelligent presence monitoring agent.
    
    Monitors user activity, system state, and application context
    to provide comprehensive presence information.
    """
    
    def __init__(self, event_system: EventSystem, config: Optional[Dict[str, Any]] = None):
        super().__init__("presence_monitor", event_system, config)
        
        # Monitoring state
        self._current_presence: Optional[PresenceInfo] = None
        self._monitoring_enabled = True
        self._monitor_interval = self.config.get('monitor_interval', 5)  # seconds
        
        # Activity tracking
        self._last_mouse_position = (0, 0)
        self._last_keyboard_activity = datetime.now()
        self._last_mouse_activity = datetime.now()
        self._activity_threshold = timedelta(minutes=5)
        
        # Application tracking
        self._active_applications: Set[str] = set()
        self._application_usage: Dict[str, float] = {}
        
        # Behavior patterns
        self._behavior_patterns: List[Dict[str, Any]] = []
        self._pattern_analysis_enabled = True
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.PRESENCE_MONITOR
    
    @property
    def name(self) -> str:
        return "Presence Monitor"
    
    @property
    def description(self) -> str:
        return "Multi-dimensional user presence and context monitoring agent"
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="get_presence_status",
                description="Get current user presence status",
                input_types=["user_id"],
                output_types=["presence_info"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=50
            ),
            AgentCapability(
                name="monitor_activity",
                description="Monitor user activity and system state",
                input_types=["monitoring_config"],
                output_types=["activity_status"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=100
            ),
            AgentCapability(
                name="analyze_context",
                description="Analyze current application and system context",
                input_types=["context_request"],
                output_types=["context_analysis"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=200
            ),
            AgentCapability(
                name="get_behavior_patterns",
                description="Get user behavior patterns and insights",
                input_types=["pattern_request"],
                output_types=["behavior_analysis"],
                risk_level=RiskLevel.LOW,
                execution_time_ms=300
            ),
            AgentCapability(
                name="set_monitoring_config",
                description="Configure monitoring parameters",
                input_types=["monitoring_config"],
                output_types=["config_status"],
                risk_level=RiskLevel.MEDIUM,
                execution_time_ms=100
            )
        ]
    
    async def _initialize_agent(self) -> None:
        """Initialize presence monitor specific functionality."""
        self.logger.info("Initializing Presence Monitor agent")
        
        # Initialize current presence
        self._current_presence = PresenceInfo(
            user_id="default_user",
            state=PresenceState.UNKNOWN,
            last_activity=datetime.now()
        )
        
        # Start monitoring loop
        if self._monitoring_enabled:
            monitor_task = asyncio.create_task(self._monitoring_loop())
            self._background_tasks.add(monitor_task)
            
            # Start pattern analysis loop
            if self._pattern_analysis_enabled:
                pattern_task = asyncio.create_task(self._pattern_analysis_loop())
                self._background_tasks.add(pattern_task)
        
        self.logger.info("Presence Monitor initialization complete")
    
    async def _process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming requests."""
        capability = request.target_capability
        start_time = datetime.now()
        
        try:
            if capability == "get_presence_status":
                result = await self._get_presence_status(request.parameters)
            elif capability == "monitor_activity":
                result = await self._monitor_activity(request.parameters)
            elif capability == "analyze_context":
                result = await self._analyze_context(request.parameters)
            elif capability == "get_behavior_patterns":
                result = await self._get_behavior_patterns(request.parameters)
            elif capability == "set_monitoring_config":
                result = await self._set_monitoring_config(request.parameters)
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
            if event.event_type == "user.activity.detected":
                await self._handle_activity_event(event)
            elif event.event_type == "system.application.changed":
                await self._handle_application_change(event)
            elif event.event_type == "system.screen.locked":
                await self._handle_screen_lock(event)
            elif event.event_type == "system.screen.unlocked":
                await self._handle_screen_unlock(event)
                
        except Exception as e:
            await self._handle_error(e, f"event_handling_{event.event_type}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup presence monitor resources."""
        self._monitoring_enabled = False
        self.logger.info("Presence Monitor cleanup complete")
    
    # Public API methods
    
    async def get_current_presence(self) -> PresenceInfo:
        """Get current user presence information."""
        if self._current_presence is None:
            await self._update_presence_status()
        return self._current_presence
    
    async def is_user_active(self) -> bool:
        """Check if user is currently active."""
        presence = await self.get_current_presence()
        return presence.state in [PresenceState.ACTIVE, PresenceState.BUSY]
    
    async def get_activity_summary(self, duration: timedelta = None) -> Dict[str, Any]:
        """Get activity summary for specified duration."""
        duration = duration or timedelta(hours=1)
        
        # Simple activity summary (would be more sophisticated in practice)
        return {
            "active_time_minutes": 45,
            "idle_time_minutes": 15,
            "applications_used": list(self._active_applications),
            "most_used_app": max(self._application_usage.items(), 
                               key=lambda x: x[1], default=("none", 0))[0],
            "activity_level": "moderate"
        }
    
    # Private implementation methods
    
    async def _get_presence_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get presence status."""
        await self._update_presence_status()
        
        return {
            "presence_info": {
                "user_id": self._current_presence.user_id,
                "state": self._current_presence.state.value,
                "last_activity": self._current_presence.last_activity.isoformat(),
                "current_application": self._current_presence.current_application,
                "screen_locked": self._current_presence.screen_locked,
                "active_windows": self._current_presence.active_windows,
                "confidence": self._current_presence.confidence
            }
        }
    
    async def _monitor_activity(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor user activity."""
        config = parameters.get("monitoring_config", {})
        
        # Update monitoring configuration if provided
        if "interval" in config:
            self._monitor_interval = config["interval"]
        
        # Force activity check
        await self._check_system_activity()
        
        return {
            "activity_status": "monitoring_active",
            "last_check": datetime.now().isoformat(),
            "monitor_interval": self._monitor_interval
        }
    
    async def _analyze_context(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current context."""
        context_analysis = await self._perform_context_analysis()
        
        return {
            "context_analysis": context_analysis
        }
    
    async def _get_behavior_patterns(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get behavior patterns."""
        patterns = await self._analyze_behavior_patterns()
        
        return {
            "behavior_analysis": {
                "patterns": patterns,
                "insights": self._generate_behavior_insights(patterns),
                "recommendations": self._generate_recommendations(patterns)
            }
        }
    
    async def _set_monitoring_config(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Set monitoring configuration."""
        config = parameters.get("config", {})
        
        if "enabled" in config:
            self._monitoring_enabled = config["enabled"]
        
        if "interval" in config:
            self._monitor_interval = max(1, config["interval"])
        
        if "pattern_analysis_enabled" in config:
            self._pattern_analysis_enabled = config["pattern_analysis_enabled"]
        
        return {
            "config_status": "updated",
            "current_config": {
                "monitoring_enabled": self._monitoring_enabled,
                "monitor_interval": self._monitor_interval,
                "pattern_analysis_enabled": self._pattern_analysis_enabled
            }
        }
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_enabled and not self._shutdown_requested:
            try:
                await asyncio.sleep(self._monitor_interval)
                
                # Check system activity
                await self._check_system_activity()
                
                # Update presence status
                await self._update_presence_status()
                
                # Emit presence update event
                await self._emit_presence_update()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "monitoring_loop")
    
    async def _pattern_analysis_loop(self) -> None:
        """Background pattern analysis loop."""
        while self._pattern_analysis_enabled and not self._shutdown_requested:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
                # Analyze behavior patterns
                patterns = await self._analyze_behavior_patterns()
                
                # Store patterns
                if patterns:
                    self._behavior_patterns.append({
                        "timestamp": datetime.now(),
                        "patterns": patterns
                    })
                    
                    # Keep only recent patterns
                    cutoff = datetime.now() - timedelta(days=7)
                    self._behavior_patterns = [
                        p for p in self._behavior_patterns 
                        if p["timestamp"] > cutoff
                    ]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(e, "pattern_analysis_loop")
    
    async def _check_system_activity(self) -> None:
        """Check system activity indicators."""
        try:
            current_time = datetime.now()
            
            # Check mouse position (simplified - would use actual system APIs)
            # For now, simulate mouse activity
            import random
            current_mouse = (random.randint(0, 1920), random.randint(0, 1080))
            
            if current_mouse != self._last_mouse_position:
                self._last_mouse_activity = current_time
                self._last_mouse_position = current_mouse
                
            # Check keyboard activity (simulated)
            if random.random() < 0.3:  # 30% chance of keyboard activity
                self._last_keyboard_activity = current_time
            
            # Check active applications
            await self._update_active_applications()
            
        except Exception as e:
            self.logger.error(f"Error checking system activity: {e}")
    
    async def _update_active_applications(self) -> None:
        """Update list of active applications."""
        try:
            # Get running processes (simplified)
            current_apps = set()
            
            for proc in psutil.process_iter(['name']):
                try:
                    proc_name = proc.info['name']
                    if proc_name and not proc_name.startswith('System'):
                        current_apps.add(proc_name)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Update application usage tracking
            for app in current_apps:
                if app not in self._application_usage:
                    self._application_usage[app] = 0
                self._application_usage[app] += self._monitor_interval
            
            self._active_applications = current_apps
            
            # Update current application (simulate getting foreground app)
            if current_apps:
                self._current_presence.current_application = list(current_apps)[0]
                self._current_presence.active_windows = list(current_apps)[:5]
            
        except Exception as e:
            self.logger.error(f"Error updating applications: {e}")
    
    async def _update_presence_status(self) -> None:
        """Update user presence status."""
        if self._current_presence is None:
            return
        
        current_time = datetime.now()
        
        # Calculate time since last activity
        last_mouse_delta = current_time - self._last_mouse_activity
        last_keyboard_delta = current_time - self._last_keyboard_activity
        last_activity_delta = min(last_mouse_delta, last_keyboard_delta)
        
        # Determine presence state
        old_state = self._current_presence.state
        
        if last_activity_delta < timedelta(minutes=1):
            new_state = PresenceState.ACTIVE
        elif last_activity_delta < timedelta(minutes=5):
            new_state = PresenceState.IDLE
        elif last_activity_delta < timedelta(minutes=30):
            new_state = PresenceState.AWAY
        else:
            new_state = PresenceState.OFFLINE
        
        # Update presence info
        self._current_presence.state = new_state
        self._current_presence.last_activity = current_time - last_activity_delta
        self._current_presence.confidence = 0.9  # High confidence for basic detection
        
        # Check for state changes
        if old_state != new_state:
            await self._handle_presence_state_change(old_state, new_state)
    
    async def _perform_context_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive context analysis."""
        context = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_processes": len(list(psutil.process_iter()))
            },
            "user_context": {
                "presence_state": self._current_presence.state.value,
                "current_application": self._current_presence.current_application,
                "active_applications": list(self._active_applications),
                "screen_locked": self._current_presence.screen_locked
            },
            "activity_context": {
                "time_since_last_activity": (
                    datetime.now() - self._current_presence.last_activity
                ).total_seconds(),
                "activity_level": self._calculate_activity_level()
            }
        }
        
        return context
    
    def _calculate_activity_level(self) -> str:
        """Calculate current activity level."""
        if self._current_presence.state == PresenceState.ACTIVE:
            return "high"
        elif self._current_presence.state == PresenceState.IDLE:
            return "medium"
        elif self._current_presence.state == PresenceState.AWAY:
            return "low"
        else:
            return "none"
    
    async def _analyze_behavior_patterns(self) -> List[Dict[str, Any]]:
        """Analyze user behavior patterns."""
        patterns = []
        
        # Analyze application usage patterns
        if self._application_usage:
            most_used_apps = sorted(
                self._application_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            patterns.append({
                "type": "application_usage",
                "description": "Most frequently used applications",
                "data": {
                    "top_applications": [
                        {"name": app, "usage_minutes": minutes / 60}
                        for app, minutes in most_used_apps
                    ]
                }
            })
        
        # Analyze activity patterns (simplified)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:
            activity_pattern = "work_hours"
        elif 18 <= current_hour <= 22:
            activity_pattern = "evening_hours"
        else:
            activity_pattern = "off_hours"
        
        patterns.append({
            "type": "activity_timing",
            "description": f"Current activity during {activity_pattern}",
            "data": {
                "pattern": activity_pattern,
                "hour": current_hour,
                "typical_activity": self._current_presence.state.value
            }
        })
        
        return patterns
    
    def _generate_behavior_insights(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from behavior patterns."""
        insights = []
        
        for pattern in patterns:
            if pattern["type"] == "application_usage":
                top_app = pattern["data"]["top_applications"][0] if pattern["data"]["top_applications"] else None
                if top_app:
                    insights.append(f"Most used application: {top_app['name']} ({top_app['usage_minutes']:.1f} hours)")
            
            elif pattern["type"] == "activity_timing":
                activity_pattern = pattern["data"]["pattern"]
                insights.append(f"Currently in {activity_pattern} with {pattern['data']['typical_activity']} state")
        
        return insights
    
    def _generate_recommendations(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on patterns."""
        recommendations = []
        
        # Check for high activity during off hours
        for pattern in patterns:
            if (pattern["type"] == "activity_timing" and 
                pattern["data"]["pattern"] == "off_hours" and
                pattern["data"]["typical_activity"] == "active"):
                recommendations.append("Consider taking breaks during off-hours for better work-life balance")
        
        # Check for application diversity
        app_pattern = next((p for p in patterns if p["type"] == "application_usage"), None)
        if app_pattern and len(app_pattern["data"]["top_applications"]) < 3:
            recommendations.append("Consider diversifying application usage for varied tasks")
        
        return recommendations
    
    async def _emit_presence_update(self) -> None:
        """Emit presence update event."""
        await self._emit_event("presence.updated", {
            "user_id": self._current_presence.user_id,
            "state": self._current_presence.state.value,
            "last_activity": self._current_presence.last_activity.isoformat(),
            "current_application": self._current_presence.current_application,
            "confidence": self._current_presence.confidence
        })
    
    async def _handle_presence_state_change(self, old_state: PresenceState, new_state: PresenceState) -> None:
        """Handle presence state changes."""
        self.logger.info(f"Presence state changed: {old_state.value} -> {new_state.value}")
        
        await self._emit_event("presence.state_changed", {
            "user_id": self._current_presence.user_id,
            "old_state": old_state.value,
            "new_state": new_state.value,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_activity_event(self, event: AgentEvent) -> None:
        """Handle user activity events."""
        activity_type = event.data.get("activity_type")
        
        if activity_type in ["mouse", "keyboard"]:
            # Update last activity time
            self._current_presence.last_activity = datetime.now()
            
            if activity_type == "mouse":
                self._last_mouse_activity = datetime.now()
            elif activity_type == "keyboard":
                self._last_keyboard_activity = datetime.now()
    
    async def _handle_application_change(self, event: AgentEvent) -> None:
        """Handle application change events."""
        new_app = event.data.get("application")
        if new_app:
            self._current_presence.current_application = new_app
            self._active_applications.add(new_app)
    
    async def _handle_screen_lock(self, event: AgentEvent) -> None:
        """Handle screen lock events."""
        self._current_presence.screen_locked = True
        self._current_presence.state = PresenceState.AWAY
        
        await self._emit_event("presence.screen_locked", {
            "user_id": self._current_presence.user_id,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_screen_unlock(self, event: AgentEvent) -> None:
        """Handle screen unlock events."""
        self._current_presence.screen_locked = False
        self._current_presence.state = PresenceState.ACTIVE
        self._current_presence.last_activity = datetime.now()
        
        await self._emit_event("presence.screen_unlocked", {
            "user_id": self._current_presence.user_id,
            "timestamp": datetime.now().isoformat()
        })