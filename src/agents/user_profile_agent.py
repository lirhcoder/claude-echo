"""User Profile Agent - Multi-User Configuration Management

The User Profile Agent manages individual user profiles, preferences, settings,
and identity management including voice-based user identification and
personalized experience delivery.
"""

import asyncio
import hashlib
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from pathlib import Path

from loguru import logger

from ..core.event_system import EventSystem
from ..core.types import RiskLevel, Context
from ..learning.learning_data_manager import (
    LearningDataManager, LearningData, UserLearningProfile, DataPrivacyLevel
)
from ..learning.learning_events import LearningEventFactory
from .base_agent import BaseAgent
from .agent_types import (
    AgentType, AgentStatus, AgentRequest, AgentResponse, AgentEvent,
    AgentCapability, MessageType, AgentPriority
)


class IdentityConfidence(Enum):
    """Confidence levels for user identity recognition"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


class ProfileSyncStatus(Enum):
    """Profile synchronization status"""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICTED = "conflicted"
    ERROR = "error"


@dataclass
class VoiceProfile:
    """Voice profile for user identification"""
    user_id: str
    voice_signature: str  # Encrypted voice characteristics
    confidence_threshold: float = 0.7
    enrollment_samples: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    accuracy_score: float = 0.0
    is_active: bool = True


@dataclass
class UserPreferences:
    """Comprehensive user preferences"""
    # Communication preferences
    communication_style: str = "balanced"  # formal, casual, balanced
    response_length: str = "medium"  # brief, medium, detailed
    language: str = "en"
    voice_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Interaction preferences
    interaction_mode: str = "voice"  # voice, text, mixed
    confirmation_level: str = "standard"  # minimal, standard, verbose
    error_handling: str = "helpful"  # minimal, helpful, detailed
    
    # Privacy preferences
    data_sharing_consent: bool = True
    analytics_consent: bool = True
    personalization_enabled: bool = True
    data_retention_days: int = 365
    
    # Accessibility preferences
    accessibility_features: List[str] = field(default_factory=list)
    text_size: str = "normal"
    high_contrast: bool = False
    
    # Workflow preferences
    default_workspace: Optional[str] = None
    favorite_commands: List[str] = field(default_factory=list)
    quick_actions: Dict[str, str] = field(default_factory=dict)
    
    # Security preferences
    require_voice_confirmation: bool = False
    session_timeout_minutes: int = 30
    privacy_mode: bool = False


@dataclass
class UserSession:
    """User session information"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    device_info: Dict[str, Any] = field(default_factory=dict)
    location_context: Optional[str] = None
    interaction_count: int = 0
    voice_confidence_history: List[float] = field(default_factory=list)


@dataclass
class UserProfile:
    """Complete user profile"""
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: Optional[str] = None
    display_name: str = ""
    email: Optional[str] = None
    
    # Profile metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    profile_version: str = "1.0"
    
    # Identity and authentication
    voice_profile: Optional[VoiceProfile] = None
    identity_confidence: float = 0.0
    
    # User preferences and settings
    preferences: UserPreferences = field(default_factory=UserPreferences)
    
    # Usage statistics
    total_sessions: int = 0
    total_interactions: int = 0
    average_session_duration: float = 0.0
    
    # Learning and adaptation
    learning_profile: Optional[UserLearningProfile] = None
    
    # Status and synchronization
    is_active: bool = True
    sync_status: ProfileSyncStatus = ProfileSyncStatus.SYNCED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "last_updated": self.last_updated.isoformat(),
            "voice_profile": asdict(self.voice_profile) if self.voice_profile else None,
            "sync_status": self.sync_status.value
        }


class UserProfileAgent(BaseAgent):
    """
    User Profile Agent managing multi-user configurations and identity.
    
    This agent handles:
    - User identity recognition and authentication
    - Personal preferences and settings management
    - Voice profile management for identification
    - Session tracking and management
    - Multi-user environment support
    - Privacy and security settings
    """
    
    def __init__(self, event_system: EventSystem, config: Optional[Dict[str, Any]] = None):
        super().__init__("user_profile_agent", event_system, config)
        
        # Core components
        self.data_manager: Optional[LearningDataManager] = None
        
        # User management
        self._user_profiles: Dict[str, UserProfile] = {}
        self._active_sessions: Dict[str, UserSession] = {}
        self._voice_profiles: Dict[str, VoiceProfile] = {}
        
        # Identity recognition
        self._identity_cache: Dict[str, Tuple[str, float, datetime]] = {}  # signature -> (user_id, confidence, timestamp)
        self._recognition_history: List[Dict[str, Any]] = []
        
        # Profile synchronization
        self._sync_queue: asyncio.Queue = asyncio.Queue()
        self._pending_syncs: Dict[str, datetime] = {}
        
        # Configuration
        self._profile_config = {
            "voice_recognition_enabled": True,
            "auto_create_profiles": True,
            "session_timeout_minutes": 30,
            "max_concurrent_sessions": 10,
            "profile_cache_size": 100,
            "voice_confidence_threshold": 0.7,
            "identity_cache_duration": timedelta(minutes=15)
        }
        self._profile_config.update(self.config.get("user_profiles", {}))
        
        # Statistics
        self._profile_statistics = {
            "total_users": 0,
            "active_sessions": 0,
            "successful_identifications": 0,
            "failed_identifications": 0,
            "profile_updates": 0,
            "voice_enrollments": 0,
            "average_session_duration": 0.0
        }
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.USER_PROFILE_AGENT
    
    @property
    def name(self) -> str:
        return "User Profile Agent"
    
    @property
    def description(self) -> str:
        return "Multi-user configuration and identity management with voice-based recognition"
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="identify_user",
                description="Identify user based on voice characteristics or other identifiers",
                input_types=["voice_sample", "user_context"],
                output_types=["user_identity", "confidence_score"],
                risk_level=RiskLevel.MEDIUM
            ),
            AgentCapability(
                name="manage_user_profile",
                description="Create, update, and manage user profiles",
                input_types=["profile_data", "user_preferences"],
                output_types=["profile_status", "updated_profile"],
                risk_level=RiskLevel.HIGH
            ),
            AgentCapability(
                name="manage_user_preferences",
                description="Manage user preferences and settings",
                input_types=["preference_updates", "user_id"],
                output_types=["preferences", "update_status"],
                risk_level=RiskLevel.MEDIUM
            ),
            AgentCapability(
                name="manage_voice_profile",
                description="Manage voice profiles for user identification",
                input_types=["voice_samples", "enrollment_data"],
                output_types=["voice_profile", "enrollment_status"],
                risk_level=RiskLevel.HIGH
            ),
            AgentCapability(
                name="manage_user_sessions",
                description="Manage user session lifecycle and tracking",
                input_types=["session_data", "user_context"],
                output_types=["session_info", "session_status"],
                risk_level=RiskLevel.MEDIUM
            ),
            AgentCapability(
                name="get_user_context",
                description="Get comprehensive user context for personalization",
                input_types=["user_id", "context_requirements"],
                output_types=["user_context", "personalization_data"],
                risk_level=RiskLevel.LOW
            )
        ]
    
    async def _initialize_agent(self) -> None:
        """Initialize user profile agent components."""
        self.logger.info("Initializing User Profile Agent")
        
        # Initialize learning data manager
        learning_config = self.config.get("learning", {})
        self.data_manager = LearningDataManager(
            event_system=self.event_system,
            config=learning_config
        )
        await self.data_manager.initialize()
        
        # Load existing user profiles
        await self._load_user_profiles()
        
        # Start session management loop
        session_manager = asyncio.create_task(self._session_management_loop())
        self._background_tasks.add(session_manager)
        
        # Start profile synchronization loop
        sync_processor = asyncio.create_task(self._profile_sync_loop())
        self._background_tasks.add(sync_processor)
        
        # Start identity cleanup loop
        cleanup_processor = asyncio.create_task(self._identity_cleanup_loop())
        self._background_tasks.add(cleanup_processor)
        
        # Subscribe to relevant events
        await self._setup_profile_event_subscriptions()
        
        self.logger.info("User Profile Agent initialization complete")
    
    async def _process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming agent requests."""
        capability = request.target_capability
        
        try:
            if capability == "identify_user":
                return await self._handle_user_identification(request)
            elif capability == "manage_user_profile":
                return await self._handle_profile_management(request)
            elif capability == "manage_user_preferences":
                return await self._handle_preferences_management(request)
            elif capability == "manage_voice_profile":
                return await self._handle_voice_profile_management(request)
            elif capability == "manage_user_sessions":
                return await self._handle_session_management(request)
            elif capability == "get_user_context":
                return await self._handle_user_context_request(request)
            else:
                raise ValueError(f"Unknown capability: {capability}")
                
        except Exception as e:
            self.logger.error(f"Error processing request {request.request_id}: {e}")
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle profile-related events."""
        try:
            if event.event_type.startswith("user."):
                await self._handle_user_event(event)
            elif event.event_type.startswith("session."):
                await self._handle_session_event(event)
            elif event.event_type.startswith("voice."):
                await self._handle_voice_event(event)
            elif event.event_type.startswith("learning."):
                await self._handle_learning_event(event)
                
        except Exception as e:
            await self._handle_error(e, f"event_handling_{event.event_type}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup user profile agent resources."""
        try:
            # Save active sessions
            await self._save_active_sessions()
            
            # Save updated profiles
            await self._save_all_profiles()
            
            # Shutdown data manager
            if self.data_manager:
                await self.data_manager.shutdown()
            
            self.logger.info("User Profile Agent cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    # Public API methods
    
    async def identify_user_by_voice(self, 
                                   voice_characteristics: Dict[str, Any],
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Identify user based on voice characteristics.
        
        Args:
            voice_characteristics: Voice feature data
            context: Additional context for identification
            
        Returns:
            Identification results with confidence score
        """
        try:
            # Generate voice signature
            voice_signature = self._generate_voice_signature(voice_characteristics)
            
            # Check identity cache first
            if voice_signature in self._identity_cache:
                cached_user_id, cached_confidence, cached_time = self._identity_cache[voice_signature]
                if datetime.now() - cached_time < self._profile_config["identity_cache_duration"]:
                    return {
                        "success": True,
                        "user_id": cached_user_id,
                        "confidence": cached_confidence,
                        "method": "cached",
                        "cached": True
                    }
            
            # Perform voice matching
            best_match = await self._match_voice_profile(voice_signature, context)
            
            if best_match:
                user_id, confidence = best_match
                
                # Update cache
                self._identity_cache[voice_signature] = (user_id, confidence, datetime.now())
                
                # Update statistics
                if confidence >= self._profile_config["voice_confidence_threshold"]:
                    self._profile_statistics["successful_identifications"] += 1
                else:
                    self._profile_statistics["failed_identifications"] += 1
                
                # Log recognition event
                recognition_event = {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "confidence": confidence,
                    "method": "voice_recognition",
                    "context": context or {}
                }
                self._recognition_history.append(recognition_event)
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "confidence": confidence,
                    "method": "voice_recognition",
                    "cached": False
                }
            else:
                self._profile_statistics["failed_identifications"] += 1
                return {
                    "success": False,
                    "error": "No matching voice profile found",
                    "confidence": 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Error identifying user by voice: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_user_profile(self, 
                                user_data: Dict[str, Any],
                                voice_sample: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new user profile.
        
        Args:
            user_data: User profile information
            voice_sample: Optional voice sample for enrollment
            
        Returns:
            Profile creation results
        """
        try:
            # Create user profile
            profile = UserProfile(
                username=user_data.get("username"),
                display_name=user_data.get("display_name", ""),
                email=user_data.get("email")
            )
            
            # Set preferences if provided
            if "preferences" in user_data:
                pref_data = user_data["preferences"]
                profile.preferences = UserPreferences(**pref_data)
            
            # Create voice profile if sample provided
            if voice_sample and self._profile_config["voice_recognition_enabled"]:
                voice_profile = await self._create_voice_profile(profile.user_id, voice_sample)
                profile.voice_profile = voice_profile
                if voice_profile:
                    self._voice_profiles[profile.user_id] = voice_profile
            
            # Create learning profile
            learning_profile = await self.data_manager.get_user_profile(profile.user_id)
            profile.learning_profile = learning_profile
            
            # Store profile
            self._user_profiles[profile.user_id] = profile
            await self._save_profile(profile)
            
            # Update statistics
            self._profile_statistics["total_users"] += 1
            if voice_sample:
                self._profile_statistics["voice_enrollments"] += 1
            
            self.logger.info(f"Created user profile: {profile.user_id}")
            
            return {
                "success": True,
                "user_id": profile.user_id,
                "profile": profile.to_dict(),
                "voice_enrolled": profile.voice_profile is not None
            }
            
        except Exception as e:
            self.logger.error(f"Error creating user profile: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_user_preferences(self, 
                                    user_id: str,
                                    preference_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            preference_updates: Preference updates
            
        Returns:
            Update results
        """
        try:
            profile = await self._get_user_profile(user_id)
            if not profile:
                return {"success": False, "error": "User profile not found"}
            
            # Update preferences
            for key, value in preference_updates.items():
                if hasattr(profile.preferences, key):
                    setattr(profile.preferences, key, value)
            
            # Update profile metadata
            profile.last_updated = datetime.now()
            profile.sync_status = ProfileSyncStatus.PENDING
            
            # Save profile
            await self._save_profile(profile)
            
            # Update learning data manager preferences
            if self.data_manager:
                await self.data_manager.update_user_preferences(user_id, preference_updates)
            
            # Queue for synchronization
            await self._queue_profile_sync(user_id)
            
            # Update statistics
            self._profile_statistics["profile_updates"] += 1
            
            self.logger.info(f"Updated preferences for user: {user_id}")
            
            return {
                "success": True,
                "updated_preferences": preference_updates,
                "profile_version": profile.profile_version
            }
            
        except Exception as e:
            self.logger.error(f"Error updating user preferences: {e}")
            return {"success": False, "error": str(e)}
    
    async def start_user_session(self, 
                                user_id: str,
                                device_info: Optional[Dict[str, Any]] = None,
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start a new user session.
        
        Args:
            user_id: User identifier
            device_info: Device information
            context: Session context
            
        Returns:
            Session information
        """
        try:
            # Check if user profile exists
            profile = await self._get_user_profile(user_id)
            if not profile:
                return {"success": False, "error": "User profile not found"}
            
            # Check session limit
            active_user_sessions = [s for s in self._active_sessions.values() if s.user_id == user_id and s.is_active]
            if len(active_user_sessions) >= self._profile_config["max_concurrent_sessions"]:
                return {"success": False, "error": "Maximum concurrent sessions reached"}
            
            # Create new session
            session = UserSession(
                user_id=user_id,
                device_info=device_info or {},
                location_context=context.get("location") if context else None
            )
            
            # Store session
            self._active_sessions[session.session_id] = session
            
            # Update profile
            profile.last_login = datetime.now()
            profile.total_sessions += 1
            
            # Update statistics
            self._profile_statistics["active_sessions"] += 1
            
            self.logger.info(f"Started session {session.session_id} for user {user_id}")
            
            return {
                "success": True,
                "session_id": session.session_id,
                "user_id": user_id,
                "session_timeout": profile.preferences.session_timeout_minutes
            }
            
        except Exception as e:
            self.logger.error(f"Error starting user session: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_user_context(self, 
                             user_id: str,
                             context_type: str = "full") -> Dict[str, Any]:
        """
        Get comprehensive user context for personalization.
        
        Args:
            user_id: User identifier
            context_type: Type of context to return
            
        Returns:
            User context data
        """
        try:
            profile = await self._get_user_profile(user_id)
            if not profile:
                return {"success": False, "error": "User profile not found"}
            
            # Build context based on type
            context = {
                "user_id": user_id,
                "display_name": profile.display_name,
                "preferences": asdict(profile.preferences)
            }
            
            if context_type in ["full", "extended"]:
                # Add session information
                active_sessions = [s for s in self._active_sessions.values() 
                                 if s.user_id == user_id and s.is_active]
                context["active_sessions"] = len(active_sessions)
                
                if active_sessions:
                    current_session = active_sessions[0]  # Most recent
                    context["current_session"] = {
                        "session_id": current_session.session_id,
                        "duration_minutes": (datetime.now() - current_session.start_time).total_seconds() / 60,
                        "interaction_count": current_session.interaction_count,
                        "device_info": current_session.device_info
                    }
                
                # Add usage statistics
                context["usage_statistics"] = {
                    "total_sessions": profile.total_sessions,
                    "total_interactions": profile.total_interactions,
                    "average_session_duration": profile.average_session_duration
                }
                
                # Add learning insights if available
                if profile.learning_profile:
                    context["learning_insights"] = {
                        "interaction_patterns": profile.learning_profile.interaction_patterns,
                        "learning_goals": profile.learning_profile.learning_goals
                    }
            
            return {"success": True, "context": context}
            
        except Exception as e:
            self.logger.error(f"Error getting user context: {e}")
            return {"success": False, "error": str(e)}
    
    async def enroll_voice_profile(self, 
                                 user_id: str,
                                 voice_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enroll or update user voice profile.
        
        Args:
            user_id: User identifier
            voice_samples: List of voice samples for training
            
        Returns:
            Enrollment results
        """
        try:
            if not self._profile_config["voice_recognition_enabled"]:
                return {"success": False, "error": "Voice recognition disabled"}
            
            profile = await self._get_user_profile(user_id)
            if not profile:
                return {"success": False, "error": "User profile not found"}
            
            # Process voice samples and create/update voice profile
            voice_profile = await self._train_voice_profile(user_id, voice_samples)
            
            if voice_profile:
                # Update profile
                profile.voice_profile = voice_profile
                self._voice_profiles[user_id] = voice_profile
                
                # Save profile
                await self._save_profile(profile)
                
                # Update statistics
                self._profile_statistics["voice_enrollments"] += 1
                
                self.logger.info(f"Voice profile enrolled for user: {user_id}")
                
                return {
                    "success": True,
                    "voice_profile_id": voice_profile.voice_signature[:8] + "...",
                    "confidence_threshold": voice_profile.confidence_threshold,
                    "enrollment_samples": voice_profile.enrollment_samples
                }
            else:
                return {"success": False, "error": "Voice profile creation failed"}
                
        except Exception as e:
            self.logger.error(f"Error enrolling voice profile: {e}")
            return {"success": False, "error": str(e)}
    
    # Private implementation methods
    
    async def _load_user_profiles(self) -> None:
        """Load existing user profiles from storage."""
        try:
            # Load profiles from learning data manager
            profile_data = await self.data_manager.retrieve_learning_data(
                data_type="user_profile",
                limit=1000
            )
            
            for data in profile_data:
                try:
                    profile_dict = data.data_content
                    profile = self._dict_to_profile(profile_dict)
                    self._user_profiles[profile.user_id] = profile
                    
                    # Load voice profile separately
                    if profile.voice_profile:
                        self._voice_profiles[profile.user_id] = profile.voice_profile
                        
                except Exception as e:
                    self.logger.error(f"Error loading profile {data.data_id}: {e}")
            
            self._profile_statistics["total_users"] = len(self._user_profiles)
            self.logger.info(f"Loaded {len(self._user_profiles)} user profiles")
            
        except Exception as e:
            self.logger.error(f"Error loading user profiles: {e}")
    
    async def _save_profile(self, profile: UserProfile) -> None:
        """Save user profile to storage."""
        try:
            profile_data = LearningData(
                data_id=f"profile_{profile.user_id}",
                user_id=profile.user_id,
                agent_id=self.agent_id,
                data_type="user_profile",
                data_content=profile.to_dict(),
                privacy_level=DataPrivacyLevel.PRIVATE
            )
            
            await self.data_manager.store_learning_data(profile_data)
            profile.sync_status = ProfileSyncStatus.SYNCED
            
        except Exception as e:
            self.logger.error(f"Error saving profile {profile.user_id}: {e}")
            profile.sync_status = ProfileSyncStatus.ERROR
    
    async def _save_all_profiles(self) -> None:
        """Save all modified profiles."""
        for profile in self._user_profiles.values():
            if profile.sync_status != ProfileSyncStatus.SYNCED:
                await self._save_profile(profile)
    
    async def _save_active_sessions(self) -> None:
        """Save active session data."""
        for session in self._active_sessions.values():
            if session.is_active:
                session_data = LearningData(
                    data_id=f"session_{session.session_id}",
                    user_id=session.user_id,
                    agent_id=self.agent_id,
                    data_type="user_session",
                    data_content=asdict(session),
                    privacy_level=DataPrivacyLevel.PRIVATE
                )
                await self.data_manager.store_learning_data(session_data)
    
    async def _get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]
        
        # Try to load from storage
        profile_data = await self.data_manager.retrieve_learning_data(
            user_id=user_id,
            data_type="user_profile",
            limit=1
        )
        
        if profile_data:
            try:
                profile_dict = profile_data[0].data_content
                profile = self._dict_to_profile(profile_dict)
                self._user_profiles[user_id] = profile
                return profile
            except Exception as e:
                self.logger.error(f"Error loading profile for {user_id}: {e}")
        
        return None
    
    def _dict_to_profile(self, profile_dict: Dict[str, Any]) -> UserProfile:
        """Convert dictionary to UserProfile object."""
        # Convert datetime fields
        for field in ["created_at", "last_login", "last_updated"]:
            if field in profile_dict and profile_dict[field]:
                profile_dict[field] = datetime.fromisoformat(profile_dict[field])
        
        # Convert preferences
        if "preferences" in profile_dict:
            profile_dict["preferences"] = UserPreferences(**profile_dict["preferences"])
        
        # Convert voice profile
        if "voice_profile" in profile_dict and profile_dict["voice_profile"]:
            vp_dict = profile_dict["voice_profile"]
            if "last_updated" in vp_dict:
                vp_dict["last_updated"] = datetime.fromisoformat(vp_dict["last_updated"])
            profile_dict["voice_profile"] = VoiceProfile(**vp_dict)
        
        # Convert sync status
        if "sync_status" in profile_dict:
            profile_dict["sync_status"] = ProfileSyncStatus(profile_dict["sync_status"])
        
        return UserProfile(**profile_dict)
    
    def _generate_voice_signature(self, voice_characteristics: Dict[str, Any]) -> str:
        """Generate a voice signature from characteristics."""
        # Simplified voice signature generation
        # In a real implementation, this would use advanced voice processing
        
        # Extract key features
        features = [
            str(voice_characteristics.get("pitch_mean", 0)),
            str(voice_characteristics.get("pitch_variance", 0)),
            str(voice_characteristics.get("formant_f1", 0)),
            str(voice_characteristics.get("formant_f2", 0)),
            str(voice_characteristics.get("speech_rate", 0)),
            str(voice_characteristics.get("intensity_mean", 0))
        ]
        
        # Create signature hash
        signature_data = "|".join(features)
        signature = hashlib.sha256(signature_data.encode()).hexdigest()
        
        return signature[:32]  # Use first 32 characters
    
    async def _match_voice_profile(self, 
                                 voice_signature: str,
                                 context: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, float]]:
        """Match voice signature against known profiles."""
        best_match = None
        best_confidence = 0.0
        
        for user_id, voice_profile in self._voice_profiles.items():
            if not voice_profile.is_active:
                continue
            
            # Calculate similarity (simplified implementation)
            confidence = self._calculate_voice_similarity(voice_signature, voice_profile.voice_signature)
            
            # Apply context boost if available
            if context and "expected_user" in context:
                if context["expected_user"] == user_id:
                    confidence *= 1.2  # Boost expected user
            
            if confidence > best_confidence and confidence >= voice_profile.confidence_threshold:
                best_match = (user_id, confidence)
                best_confidence = confidence
        
        return best_match
    
    def _calculate_voice_similarity(self, signature1: str, signature2: str) -> float:
        """Calculate similarity between voice signatures."""
        # Simplified similarity calculation
        # In a real implementation, this would use sophisticated voice matching algorithms
        
        if signature1 == signature2:
            return 1.0
        
        # Calculate character-level similarity
        common_chars = sum(1 for a, b in zip(signature1, signature2) if a == b)
        similarity = common_chars / len(signature1)
        
        # Apply threshold
        return max(0.0, min(1.0, similarity))
    
    async def _create_voice_profile(self, 
                                  user_id: str,
                                  voice_sample: Dict[str, Any]) -> Optional[VoiceProfile]:
        """Create a voice profile from a sample."""
        try:
            voice_signature = self._generate_voice_signature(voice_sample)
            
            voice_profile = VoiceProfile(
                user_id=user_id,
                voice_signature=voice_signature,
                confidence_threshold=self._profile_config["voice_confidence_threshold"],
                enrollment_samples=1,
                accuracy_score=0.85  # Initial estimate
            )
            
            return voice_profile
            
        except Exception as e:
            self.logger.error(f"Error creating voice profile: {e}")
            return None
    
    async def _train_voice_profile(self, 
                                 user_id: str,
                                 voice_samples: List[Dict[str, Any]]) -> Optional[VoiceProfile]:
        """Train voice profile from multiple samples."""
        try:
            if not voice_samples:
                return None
            
            # Process all samples
            signatures = [self._generate_voice_signature(sample) for sample in voice_samples]
            
            # Use the most representative signature (simplified)
            primary_signature = signatures[0] if signatures else ""
            
            # Calculate confidence based on sample consistency
            consistency_score = 0.8  # Simplified calculation
            
            voice_profile = VoiceProfile(
                user_id=user_id,
                voice_signature=primary_signature,
                confidence_threshold=max(0.6, consistency_score * 0.9),
                enrollment_samples=len(voice_samples),
                accuracy_score=consistency_score
            )
            
            return voice_profile
            
        except Exception as e:
            self.logger.error(f"Error training voice profile: {e}")
            return None
    
    async def _session_management_loop(self) -> None:
        """Background session management loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for expired sessions
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self._active_sessions.items():
                    if not session.is_active:
                        continue
                    
                    # Get user profile to check timeout settings
                    profile = await self._get_user_profile(session.user_id)
                    if not profile:
                        continue
                    
                    timeout_minutes = profile.preferences.session_timeout_minutes
                    time_since_activity = current_time - session.last_activity
                    
                    if time_since_activity.total_seconds() > (timeout_minutes * 60):
                        expired_sessions.append(session_id)
                
                # End expired sessions
                for session_id in expired_sessions:
                    await self._end_session(session_id)
                
                # Update session statistics
                self._profile_statistics["active_sessions"] = len([
                    s for s in self._active_sessions.values() if s.is_active
                ])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session management loop: {e}")
    
    async def _profile_sync_loop(self) -> None:
        """Background profile synchronization loop."""
        while not self._shutdown_requested:
            try:
                # Get next sync request
                try:
                    user_id = await asyncio.wait_for(
                        self._sync_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Perform profile synchronization
                await self._synchronize_profile(user_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in profile sync loop: {e}")
    
    async def _identity_cleanup_loop(self) -> None:
        """Background identity cache cleanup loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes
                
                # Clean expired identity cache entries
                current_time = datetime.now()
                expired_keys = []
                
                for signature, (user_id, confidence, timestamp) in self._identity_cache.items():
                    if current_time - timestamp > self._profile_config["identity_cache_duration"]:
                        expired_keys.append(signature)
                
                for key in expired_keys:
                    del self._identity_cache[key]
                
                # Limit recognition history size
                if len(self._recognition_history) > 1000:
                    self._recognition_history = self._recognition_history[-500:]  # Keep last 500
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in identity cleanup loop: {e}")
    
    async def _end_session(self, session_id: str) -> None:
        """End a user session."""
        if session_id not in self._active_sessions:
            return
        
        session = self._active_sessions[session_id]
        session.is_active = False
        
        # Update user profile statistics
        profile = await self._get_user_profile(session.user_id)
        if profile:
            session_duration = (datetime.now() - session.start_time).total_seconds() / 60
            profile.total_interactions += session.interaction_count
            
            # Update average session duration
            if profile.total_sessions > 0:
                profile.average_session_duration = (
                    (profile.average_session_duration * (profile.total_sessions - 1) + session_duration)
                    / profile.total_sessions
                )
            
            profile.last_updated = datetime.now()
            await self._save_profile(profile)
        
        # Save session data
        await self._save_active_sessions()
        
        self.logger.info(f"Ended session {session_id} for user {session.user_id}")
    
    async def _queue_profile_sync(self, user_id: str) -> None:
        """Queue a profile for synchronization."""
        if user_id not in self._pending_syncs:
            self._pending_syncs[user_id] = datetime.now()
            await self._sync_queue.put(user_id)
    
    async def _synchronize_profile(self, user_id: str) -> None:
        """Synchronize user profile across systems."""
        try:
            profile = await self._get_user_profile(user_id)
            if not profile:
                return
            
            # Perform synchronization (simplified)
            # In a real implementation, this would sync with external systems
            
            profile.sync_status = ProfileSyncStatus.SYNCED
            profile.last_updated = datetime.now()
            
            # Remove from pending syncs
            if user_id in self._pending_syncs:
                del self._pending_syncs[user_id]
            
            self.logger.debug(f"Synchronized profile for user: {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error synchronizing profile {user_id}: {e}")
            profile = await self._get_user_profile(user_id)
            if profile:
                profile.sync_status = ProfileSyncStatus.ERROR
    
    # Request handlers
    
    async def _handle_user_identification(self, request: AgentRequest) -> AgentResponse:
        """Handle user identification requests."""
        try:
            voice_data = request.parameters.get("voice_characteristics")
            context = request.parameters.get("context", {})
            
            if voice_data:
                result = await self.identify_user_by_voice(voice_data, context)
            else:
                # Try other identification methods
                user_id = request.parameters.get("user_id")
                if user_id:
                    profile = await self._get_user_profile(user_id)
                    result = {
                        "success": profile is not None,
                        "user_id": user_id,
                        "confidence": 1.0 if profile else 0.0,
                        "method": "direct_id"
                    }
                else:
                    result = {"success": False, "error": "No identification method provided"}
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=result["success"],
                data=result,
                error=result.get("error")
            )
            
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_profile_management(self, request: AgentRequest) -> AgentResponse:
        """Handle profile management requests."""
        try:
            action = request.parameters.get("action", "get")
            user_id = request.parameters.get("user_id")
            
            if action == "create":
                user_data = request.parameters.get("user_data", {})
                voice_sample = request.parameters.get("voice_sample")
                result = await self.create_user_profile(user_data, voice_sample)
            
            elif action == "get":
                if not user_id:
                    raise ValueError("user_id required for get action")
                profile = await self._get_user_profile(user_id)
                result = {
                    "success": profile is not None,
                    "profile": profile.to_dict() if profile else None
                }
            
            elif action == "update":
                if not user_id:
                    raise ValueError("user_id required for update action")
                updates = request.parameters.get("updates", {})
                # Implement profile updates
                result = {"success": True, "message": "Profile update not implemented"}
            
            elif action == "delete":
                if not user_id:
                    raise ValueError("user_id required for delete action")
                # Implement profile deletion
                result = {"success": True, "message": "Profile deletion not implemented"}
            
            else:
                raise ValueError(f"Unknown action: {action}")
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=result["success"],
                data=result,
                error=result.get("error")
            )
            
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_preferences_management(self, request: AgentRequest) -> AgentResponse:
        """Handle user preferences management."""
        try:
            user_id = request.parameters.get("user_id")
            if not user_id:
                raise ValueError("user_id is required")
            
            action = request.parameters.get("action", "get")
            
            if action == "get":
                profile = await self._get_user_profile(user_id)
                if profile:
                    result = {
                        "success": True,
                        "preferences": asdict(profile.preferences)
                    }
                else:
                    result = {"success": False, "error": "User profile not found"}
            
            elif action == "update":
                updates = request.parameters.get("updates", {})
                result = await self.update_user_preferences(user_id, updates)
            
            else:
                raise ValueError(f"Unknown action: {action}")
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=result["success"],
                data=result,
                error=result.get("error")
            )
            
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_voice_profile_management(self, request: AgentRequest) -> AgentResponse:
        """Handle voice profile management."""
        try:
            user_id = request.parameters.get("user_id")
            action = request.parameters.get("action", "enroll")
            
            if not user_id:
                raise ValueError("user_id is required")
            
            if action == "enroll":
                voice_samples = request.parameters.get("voice_samples", [])
                result = await self.enroll_voice_profile(user_id, voice_samples)
            
            elif action == "get":
                profile = await self._get_user_profile(user_id)
                if profile and profile.voice_profile:
                    result = {
                        "success": True,
                        "voice_profile": {
                            "confidence_threshold": profile.voice_profile.confidence_threshold,
                            "enrollment_samples": profile.voice_profile.enrollment_samples,
                            "accuracy_score": profile.voice_profile.accuracy_score,
                            "is_active": profile.voice_profile.is_active,
                            "last_updated": profile.voice_profile.last_updated.isoformat()
                        }
                    }
                else:
                    result = {"success": False, "error": "No voice profile found"}
            
            else:
                raise ValueError(f"Unknown action: {action}")
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=result["success"],
                data=result,
                error=result.get("error")
            )
            
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_session_management(self, request: AgentRequest) -> AgentResponse:
        """Handle session management requests."""
        try:
            action = request.parameters.get("action", "start")
            
            if action == "start":
                user_id = request.parameters.get("user_id")
                device_info = request.parameters.get("device_info")
                context = request.parameters.get("context")
                
                if not user_id:
                    raise ValueError("user_id is required")
                
                result = await self.start_user_session(user_id, device_info, context)
            
            elif action == "end":
                session_id = request.parameters.get("session_id")
                if not session_id:
                    raise ValueError("session_id is required")
                
                await self._end_session(session_id)
                result = {"success": True, "message": "Session ended"}
            
            elif action == "get":
                session_id = request.parameters.get("session_id")
                if session_id and session_id in self._active_sessions:
                    session = self._active_sessions[session_id]
                    result = {
                        "success": True,
                        "session": asdict(session)
                    }
                else:
                    result = {"success": False, "error": "Session not found"}
            
            else:
                raise ValueError(f"Unknown action: {action}")
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=result["success"],
                data=result,
                error=result.get("error")
            )
            
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_user_context_request(self, request: AgentRequest) -> AgentResponse:
        """Handle user context requests."""
        try:
            user_id = request.parameters.get("user_id")
            context_type = request.parameters.get("context_type", "full")
            
            if not user_id:
                raise ValueError("user_id is required")
            
            result = await self.get_user_context(user_id, context_type)
            
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=result["success"],
                data=result,
                error=result.get("error")
            )
            
        except Exception as e:
            return AgentResponse(
                request_id=request.request_id,
                responding_agent=self.agent_id,
                success=False,
                error=str(e)
            )
    
    # Event handlers
    
    async def _handle_user_event(self, event: AgentEvent) -> None:
        """Handle user-related events."""
        self.logger.debug(f"Handling user event: {event.event_type}")
    
    async def _handle_session_event(self, event: AgentEvent) -> None:
        """Handle session-related events."""
        self.logger.debug(f"Handling session event: {event.event_type}")
    
    async def _handle_voice_event(self, event: AgentEvent) -> None:
        """Handle voice-related events."""
        self.logger.debug(f"Handling voice event: {event.event_type}")
    
    async def _handle_learning_event(self, event: AgentEvent) -> None:
        """Handle learning-related events."""
        self.logger.debug(f"Handling learning event: {event.event_type}")
    
    async def _setup_profile_event_subscriptions(self) -> None:
        """Setup event subscriptions for profile-related events."""
        # Subscribe to user events
        self.event_system.subscribe("user.login", self._handle_user_login)
        self.event_system.subscribe("user.logout", self._handle_user_logout)
        self.event_system.subscribe("user.interaction", self._handle_user_interaction)
        
        # Subscribe to session events
        self.event_system.subscribe("session.*", self._handle_session_event)
        
        # Subscribe to voice events  
        self.event_system.subscribe("voice.*", self._handle_voice_event)
    
    async def _handle_user_login(self, event) -> None:
        """Handle user login events."""
        try:
            user_data = event.data
            user_id = user_data.get("user_id")
            
            if user_id:
                # Update last login
                profile = await self._get_user_profile(user_id)
                if profile:
                    profile.last_login = datetime.now()
                    await self._save_profile(profile)
            
        except Exception as e:
            self.logger.error(f"Error handling user login: {e}")
    
    async def _handle_user_logout(self, event) -> None:
        """Handle user logout events."""
        try:
            user_data = event.data
            session_id = user_data.get("session_id")
            
            if session_id and session_id in self._active_sessions:
                await self._end_session(session_id)
            
        except Exception as e:
            self.logger.error(f"Error handling user logout: {e}")
    
    async def _handle_user_interaction(self, event) -> None:
        """Handle user interaction events."""
        try:
            user_data = event.data
            session_id = user_data.get("session_id")
            
            if session_id and session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                session.last_activity = datetime.now()
                session.interaction_count += 1
            
        except Exception as e:
            self.logger.error(f"Error handling user interaction: {e}")