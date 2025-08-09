"""Learning Data Manager - Centralized Learning Data Management

Provides secure, scalable, and efficient management of learning data
with multi-user isolation, privacy protection, and performance optimization.
"""

import asyncio
import json
import sqlite3
import aiosqlite
from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid
import hashlib
import pickle
from contextlib import asynccontextmanager

from loguru import logger
from cryptography.fernet import Fernet

from ..core.event_system import EventSystem
from ..core.types import RiskLevel
from .learning_events import LearningEvent, LearningEventType, LearningEventData


class DataPrivacyLevel(Enum):
    """Privacy levels for learning data"""
    PUBLIC = "public"          # No privacy restrictions
    INTERNAL = "internal"      # Internal system use only
    PRIVATE = "private"        # User-specific private data
    CONFIDENTIAL = "confidential"  # Highly sensitive data
    

class DataQualityStatus(Enum):
    """Data quality assessment status"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class LearningData:
    """Core learning data structure"""
    data_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    data_type: str = "interaction"
    data_content: Dict[str, Any] = field(default_factory=dict)
    privacy_level: DataPrivacyLevel = DataPrivacyLevel.PRIVATE
    quality_score: float = 1.0
    quality_status: DataQualityStatus = DataQualityStatus.GOOD
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            **asdict(self),
            "privacy_level": self.privacy_level.value,
            "quality_status": self.quality_status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningData':
        """Create from dictionary representation"""
        data_copy = data.copy()
        
        # Convert enum fields
        if "privacy_level" in data_copy:
            data_copy["privacy_level"] = DataPrivacyLevel(data_copy["privacy_level"])
        if "quality_status" in data_copy:
            data_copy["quality_status"] = DataQualityStatus(data_copy["quality_status"])
        
        # Convert datetime fields
        for field_name in ["created_at", "updated_at", "expires_at"]:
            if field_name in data_copy and data_copy[field_name]:
                data_copy[field_name] = datetime.fromisoformat(data_copy[field_name])
        
        return cls(**data_copy)


@dataclass
class UserLearningProfile:
    """Learning profile for a specific user"""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    learning_goals: List[str] = field(default_factory=list)
    privacy_settings: Dict[str, bool] = field(default_factory=dict)
    data_retention_days: int = 365
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    total_interactions: int = 0
    last_interaction: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None
        }


class LearningDataManager:
    """
    Centralized management system for learning data.
    
    Provides secure storage, efficient retrieval, privacy protection,
    and data quality management for the learning system.
    """
    
    def __init__(self, 
                 event_system: EventSystem,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Learning Data Manager.
        
        Args:
            event_system: Event system for communication
            config: Optional configuration dictionary
        """
        self.event_system = event_system
        self.config = config or {}
        
        # Database configuration
        self._db_path = Path(self.config.get('db_path', './data/learning.db'))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Encryption configuration
        self._encryption_enabled = self.config.get('encryption_enabled', True)
        self._encryption_key = self._load_or_create_encryption_key()
        
        # Data retention configuration
        self._default_retention_days = self.config.get('default_retention_days', 365)
        self._cleanup_interval = self.config.get('cleanup_interval_hours', 24)
        
        # Performance configuration
        self._cache_size = self.config.get('cache_size', 1000)
        self._batch_size = self.config.get('batch_size', 100)
        
        # In-memory caches for performance
        self._user_profiles_cache: Dict[str, UserLearningProfile] = {}
        self._data_cache: Dict[str, LearningData] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Statistics and monitoring
        self._stats = {
            'total_data_points': 0,
            'unique_users': 0,
            'data_quality_average': 0.0,
            'storage_size_mb': 0.0,
            'cache_hit_rate': 0.0,
            'last_cleanup': datetime.now()
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        
        # Setup logging
        self.logger = logger.bind(component="learning_data_manager")
    
    async def initialize(self) -> None:
        """Initialize the data manager and database."""
        try:
            self.logger.info("Initializing Learning Data Manager")
            
            # Create database tables
            await self._create_database_schema()
            
            # Load initial statistics
            await self._load_statistics()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Learning Data Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Learning Data Manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the data manager gracefully."""
        try:
            self.logger.info("Shutting down Learning Data Manager")
            
            # Stop background tasks
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self._stats_task and not self._stats_task.done():
                self._stats_task.cancel()
                try:
                    await self._stats_task
                except asyncio.CancelledError:
                    pass
            
            # Final cleanup and statistics update
            await self._cleanup_expired_data()
            await self._update_statistics()
            
            self.logger.info("Learning Data Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def store_learning_data(self, data: LearningData) -> bool:
        """
        Store learning data with privacy protection.
        
        Args:
            data: Learning data to store
            
        Returns:
            True if storage successful
        """
        try:
            # Validate data quality
            await self._validate_data_quality(data)
            
            # Apply privacy protection
            protected_data = await self._apply_privacy_protection(data)
            
            # Store in database
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO learning_data 
                    (data_id, user_id, agent_id, session_id, data_type, data_content,
                     privacy_level, quality_score, quality_status, created_at, updated_at,
                     expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    protected_data.data_id,
                    protected_data.user_id,
                    protected_data.agent_id,
                    protected_data.session_id,
                    protected_data.data_type,
                    self._serialize_data(protected_data.data_content),
                    protected_data.privacy_level.value,
                    protected_data.quality_score,
                    protected_data.quality_status.value,
                    protected_data.created_at.isoformat(),
                    protected_data.updated_at.isoformat(),
                    protected_data.expires_at.isoformat() if protected_data.expires_at else None,
                    self._serialize_data(protected_data.metadata)
                ))
                await db.commit()
            
            # Update cache
            self._data_cache[data.data_id] = protected_data
            self._cache_timestamps[data.data_id] = datetime.now()
            
            # Update user profile if applicable
            if data.user_id:
                await self._update_user_profile(data.user_id, data)
            
            # Update statistics
            self._stats['total_data_points'] += 1
            
            self.logger.debug(f"Stored learning data: {data.data_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store learning data: {e}")
            return False
    
    async def retrieve_learning_data(self, 
                                   user_id: Optional[str] = None,
                                   agent_id: Optional[str] = None,
                                   session_id: Optional[str] = None,
                                   data_type: Optional[str] = None,
                                   privacy_level: Optional[DataPrivacyLevel] = None,
                                   limit: int = 100,
                                   include_expired: bool = False) -> List[LearningData]:
        """
        Retrieve learning data based on filters.
        
        Args:
            user_id: Filter by user ID
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            data_type: Filter by data type
            privacy_level: Maximum privacy level to include
            limit: Maximum number of records to return
            include_expired: Whether to include expired data
            
        Returns:
            List of matching learning data
        """
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            
            if agent_id:
                conditions.append("agent_id = ?")
                params.append(agent_id)
            
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            
            if data_type:
                conditions.append("data_type = ?")
                params.append(data_type)
            
            if privacy_level:
                conditions.append("privacy_level IN (?, ?, ?, ?)")
                # Include all privacy levels up to the specified level
                privacy_levels = ["public", "internal", "private", "confidential"]
                max_level_idx = privacy_levels.index(privacy_level.value)
                params.extend(privacy_levels[:max_level_idx + 1])
            
            if not include_expired:
                conditions.append("(expires_at IS NULL OR expires_at > ?)")
                params.append(datetime.now().isoformat())
            
            # Build and execute query
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
                SELECT * FROM learning_data 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """
            params.append(limit)
            
            results = []
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(query, params) as cursor:
                    async for row in cursor:
                        data = await self._row_to_learning_data(row)
                        if data:
                            results.append(data)
            
            # Decrypt and deserialize data
            decrypted_results = []
            for data in results:
                decrypted_data = await self._remove_privacy_protection(data)
                decrypted_results.append(decrypted_data)
            
            self.logger.debug(f"Retrieved {len(decrypted_results)} learning data records")
            return decrypted_results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve learning data: {e}")
            return []
    
    async def get_user_profile(self, user_id: str) -> Optional[UserLearningProfile]:
        """
        Get or create user learning profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User learning profile
        """
        try:
            # Check cache first
            if user_id in self._user_profiles_cache:
                cache_time = self._cache_timestamps.get(f"profile_{user_id}")
                if cache_time and (datetime.now() - cache_time).seconds < 300:  # 5-minute cache
                    return self._user_profiles_cache[user_id]
            
            # Load from database
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute("""
                    SELECT * FROM user_profiles WHERE user_id = ?
                """, (user_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        profile = await self._row_to_user_profile(row)
                    else:
                        # Create new profile
                        profile = UserLearningProfile(user_id=user_id)
                        await self._save_user_profile(profile)
            
            # Update cache
            self._user_profiles_cache[user_id] = profile
            self._cache_timestamps[f"profile_{user_id}"] = datetime.now()
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to get user profile for {user_id}: {e}")
            return None
    
    async def update_user_preferences(self, user_id: str, 
                                    preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            preferences: Updated preferences
            
        Returns:
            True if update successful
        """
        try:
            profile = await self.get_user_profile(user_id)
            if not profile:
                return False
            
            profile.preferences.update(preferences)
            profile.updated_at = datetime.now()
            
            success = await self._save_user_profile(profile)
            
            if success:
                # Update cache
                self._user_profiles_cache[user_id] = profile
                self._cache_timestamps[f"profile_{user_id}"] = datetime.now()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update user preferences: {e}")
            return False
    
    async def analyze_data_quality(self, data_sample_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze overall data quality across the system.
        
        Args:
            data_sample_size: Size of sample to analyze
            
        Returns:
            Data quality analysis results
        """
        try:
            analysis_results = {
                'total_data_points': 0,
                'quality_distribution': {},
                'average_quality_score': 0.0,
                'data_type_breakdown': {},
                'privacy_level_distribution': {},
                'recommendations': []
            }
            
            # Get data quality statistics
            async with aiosqlite.connect(self._db_path) as db:
                # Total count
                async with db.execute("SELECT COUNT(*) FROM learning_data") as cursor:
                    row = await cursor.fetchone()
                    analysis_results['total_data_points'] = row[0] if row else 0
                
                # Quality distribution
                async with db.execute("""
                    SELECT quality_status, COUNT(*), AVG(quality_score)
                    FROM learning_data 
                    GROUP BY quality_status
                """) as cursor:
                    async for row in cursor:
                        status, count, avg_score = row
                        analysis_results['quality_distribution'][status] = {
                            'count': count,
                            'average_score': avg_score or 0.0
                        }
                
                # Overall average quality score
                async with db.execute("SELECT AVG(quality_score) FROM learning_data") as cursor:
                    row = await cursor.fetchone()
                    analysis_results['average_quality_score'] = row[0] if row else 0.0
                
                # Data type breakdown
                async with db.execute("""
                    SELECT data_type, COUNT(*) FROM learning_data GROUP BY data_type
                """) as cursor:
                    async for row in cursor:
                        data_type, count = row
                        analysis_results['data_type_breakdown'][data_type] = count
                
                # Privacy level distribution
                async with db.execute("""
                    SELECT privacy_level, COUNT(*) FROM learning_data GROUP BY privacy_level
                """) as cursor:
                    async for row in cursor:
                        privacy_level, count = row
                        analysis_results['privacy_level_distribution'][privacy_level] = count
            
            # Generate recommendations
            recommendations = []
            
            if analysis_results['average_quality_score'] < 0.7:
                recommendations.append("Overall data quality is below optimal - consider data validation improvements")
            
            poor_quality_count = analysis_results['quality_distribution'].get('poor', {}).get('count', 0)
            if poor_quality_count > analysis_results['total_data_points'] * 0.1:
                recommendations.append("High proportion of poor quality data detected - review data collection processes")
            
            if len(analysis_results['data_type_breakdown']) < 3:
                recommendations.append("Limited data diversity - consider expanding data collection scope")
            
            analysis_results['recommendations'] = recommendations
            
            self.logger.info(f"Data quality analysis completed: {analysis_results['average_quality_score']:.2f} average score")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze data quality: {e}")
            return {}
    
    async def cleanup_expired_data(self) -> int:
        """
        Clean up expired learning data.
        
        Returns:
            Number of records cleaned up
        """
        return await self._cleanup_expired_data()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        await self._update_statistics()
        return self._stats.copy()
    
    # Private implementation methods
    
    async def _create_database_schema(self) -> None:
        """Create database tables if they don't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            # Learning data table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS learning_data (
                    data_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    agent_id TEXT,
                    session_id TEXT,
                    data_type TEXT NOT NULL,
                    data_content TEXT NOT NULL,
                    privacy_level TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    quality_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    metadata TEXT
                )
            """)
            
            # User profiles table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    interaction_patterns TEXT NOT NULL,
                    learning_goals TEXT NOT NULL,
                    privacy_settings TEXT NOT NULL,
                    data_retention_days INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_interactions INTEGER NOT NULL,
                    last_interaction TEXT
                )
            """)
            
            # Create indexes for performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON learning_data(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON learning_data(agent_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON learning_data(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_data_type ON learning_data(data_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON learning_data(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON learning_data(expires_at)")
            
            await db.commit()
    
    async def _load_statistics(self) -> None:
        """Load initial statistics from database."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                # Total data points
                async with db.execute("SELECT COUNT(*) FROM learning_data") as cursor:
                    row = await cursor.fetchone()
                    self._stats['total_data_points'] = row[0] if row else 0
                
                # Unique users
                async with db.execute("SELECT COUNT(DISTINCT user_id) FROM learning_data WHERE user_id IS NOT NULL") as cursor:
                    row = await cursor.fetchone()
                    self._stats['unique_users'] = row[0] if row else 0
                
                # Average data quality
                async with db.execute("SELECT AVG(quality_score) FROM learning_data") as cursor:
                    row = await cursor.fetchone()
                    self._stats['data_quality_average'] = row[0] if row else 0.0
        
        except Exception as e:
            self.logger.error(f"Failed to load statistics: {e}")
    
    async def _update_statistics(self) -> None:
        """Update performance statistics."""
        await self._load_statistics()
        
        # Calculate storage size
        try:
            self._stats['storage_size_mb'] = self._db_path.stat().st_size / (1024 * 1024)
        except:
            self._stats['storage_size_mb'] = 0.0
        
        # Calculate cache hit rate (simplified)
        total_cache_entries = len(self._data_cache) + len(self._user_profiles_cache)
        self._stats['cache_hit_rate'] = min(total_cache_entries / max(self._cache_size, 1), 1.0)
    
    async def _validate_data_quality(self, data: LearningData) -> None:
        """Validate and assess data quality."""
        quality_score = 1.0
        issues = []
        
        # Check data completeness
        if not data.data_content:
            quality_score -= 0.5
            issues.append("Empty data content")
        
        # Check data freshness
        if data.created_at < datetime.now() - timedelta(days=30):
            quality_score -= 0.2
            issues.append("Data is older than 30 days")
        
        # Check for required fields based on data type
        if data.data_type == "interaction":
            required_fields = ["user_input", "agent_response"]
            missing_fields = [f for f in required_fields if f not in data.data_content]
            if missing_fields:
                quality_score -= 0.3 * len(missing_fields)
                issues.extend([f"Missing field: {f}" for f in missing_fields])
        
        # Determine quality status
        if quality_score >= 0.9:
            data.quality_status = DataQualityStatus.EXCELLENT
        elif quality_score >= 0.7:
            data.quality_status = DataQualityStatus.GOOD
        elif quality_score >= 0.5:
            data.quality_status = DataQualityStatus.FAIR
        elif quality_score >= 0.3:
            data.quality_status = DataQualityStatus.POOR
        else:
            data.quality_status = DataQualityStatus.INVALID
        
        data.quality_score = max(0.0, quality_score)
        
        if issues:
            self.logger.debug(f"Data quality issues for {data.data_id}: {issues}")
    
    async def _apply_privacy_protection(self, data: LearningData) -> LearningData:
        """Apply privacy protection based on privacy level."""
        protected_data = LearningData(**asdict(data))
        
        if data.privacy_level in [DataPrivacyLevel.CONFIDENTIAL, DataPrivacyLevel.PRIVATE]:
            if self._encryption_enabled and self._encryption_key:
                # Encrypt sensitive data
                protected_data.data_content = self._encrypt_data(data.data_content)
                protected_data.metadata = self._encrypt_data(data.metadata)
        
        # Set expiration based on privacy level and user preferences
        if not data.expires_at:
            if data.user_id:
                profile = await self.get_user_profile(data.user_id)
                retention_days = profile.data_retention_days if profile else self._default_retention_days
            else:
                retention_days = self._default_retention_days
            
            # Shorter retention for more sensitive data
            if data.privacy_level == DataPrivacyLevel.CONFIDENTIAL:
                retention_days = min(retention_days, 90)
            elif data.privacy_level == DataPrivacyLevel.PRIVATE:
                retention_days = min(retention_days, 180)
            
            protected_data.expires_at = datetime.now() + timedelta(days=retention_days)
        
        return protected_data
    
    async def _remove_privacy_protection(self, data: LearningData) -> LearningData:
        """Remove privacy protection for data access."""
        if data.privacy_level in [DataPrivacyLevel.CONFIDENTIAL, DataPrivacyLevel.PRIVATE]:
            if self._encryption_enabled and self._encryption_key:
                try:
                    # Decrypt data
                    decrypted_content = self._decrypt_data(data.data_content)
                    decrypted_metadata = self._decrypt_data(data.metadata)
                    
                    # Create new instance with decrypted data
                    decrypted_data = LearningData(**asdict(data))
                    decrypted_data.data_content = decrypted_content
                    decrypted_data.metadata = decrypted_metadata
                    
                    return decrypted_data
                except Exception as e:
                    self.logger.error(f"Failed to decrypt data {data.data_id}: {e}")
        
        return data
    
    def _encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt data dictionary."""
        if not self._encryption_key:
            return data
        
        try:
            fernet = Fernet(self._encryption_key)
            serialized = json.dumps(data).encode()
            encrypted = fernet.encrypt(serialized)
            return {"__encrypted__": encrypted.decode()}
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return data
    
    def _decrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt data dictionary."""
        if not self._encryption_key or "__encrypted__" not in data:
            return data
        
        try:
            fernet = Fernet(self._encryption_key)
            encrypted_data = data["__encrypted__"].encode()
            decrypted = fernet.decrypt(encrypted_data)
            return json.loads(decrypted.decode())
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return data
    
    def _serialize_data(self, data: Dict[str, Any]) -> str:
        """Serialize data for database storage."""
        try:
            return json.dumps(data)
        except Exception as e:
            self.logger.error(f"Serialization failed: {e}")
            return "{}"
    
    def _deserialize_data(self, data_str: str) -> Dict[str, Any]:
        """Deserialize data from database storage."""
        try:
            return json.loads(data_str) if data_str else {}
        except Exception as e:
            self.logger.error(f"Deserialization failed: {e}")
            return {}
    
    def _load_or_create_encryption_key(self) -> Optional[bytes]:
        """Load or create encryption key."""
        if not self._encryption_enabled:
            return None
        
        key_file = Path(self.config.get('encryption_key_file', './data/encryption.key'))
        
        try:
            if key_file.exists():
                return key_file.read_bytes()
            else:
                # Generate new key
                key = Fernet.generate_key()
                key_file.parent.mkdir(parents=True, exist_ok=True)
                key_file.write_bytes(key)
                self.logger.info("Generated new encryption key")
                return key
        except Exception as e:
            self.logger.error(f"Failed to load encryption key: {e}")
            return None
    
    async def _update_user_profile(self, user_id: str, data: LearningData) -> None:
        """Update user profile with new interaction data."""
        try:
            profile = await self.get_user_profile(user_id)
            if not profile:
                return
            
            profile.total_interactions += 1
            profile.last_interaction = data.created_at
            profile.updated_at = datetime.now()
            
            # Update interaction patterns
            data_type = data.data_type
            if data_type not in profile.interaction_patterns:
                profile.interaction_patterns[data_type] = {'count': 0, 'last_seen': None}
            
            profile.interaction_patterns[data_type]['count'] += 1
            profile.interaction_patterns[data_type]['last_seen'] = data.created_at.isoformat()
            
            await self._save_user_profile(profile)
            
        except Exception as e:
            self.logger.error(f"Failed to update user profile: {e}")
    
    async def _save_user_profile(self, profile: UserLearningProfile) -> bool:
        """Save user profile to database."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO user_profiles 
                    (user_id, preferences, interaction_patterns, learning_goals, privacy_settings,
                     data_retention_days, created_at, updated_at, total_interactions, last_interaction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    self._serialize_data(profile.preferences),
                    self._serialize_data(profile.interaction_patterns),
                    json.dumps(profile.learning_goals),
                    self._serialize_data(profile.privacy_settings),
                    profile.data_retention_days,
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                    profile.total_interactions,
                    profile.last_interaction.isoformat() if profile.last_interaction else None
                ))
                await db.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save user profile: {e}")
            return False
    
    async def _row_to_learning_data(self, row) -> Optional[LearningData]:
        """Convert database row to LearningData object."""
        try:
            return LearningData(
                data_id=row[0],
                user_id=row[1],
                agent_id=row[2],
                session_id=row[3],
                data_type=row[4],
                data_content=self._deserialize_data(row[5]),
                privacy_level=DataPrivacyLevel(row[6]),
                quality_score=row[7],
                quality_status=DataQualityStatus(row[8]),
                created_at=datetime.fromisoformat(row[9]),
                updated_at=datetime.fromisoformat(row[10]),
                expires_at=datetime.fromisoformat(row[11]) if row[11] else None,
                metadata=self._deserialize_data(row[12] or '{}')
            )
        except Exception as e:
            self.logger.error(f"Failed to convert row to LearningData: {e}")
            return None
    
    async def _row_to_user_profile(self, row) -> Optional[UserLearningProfile]:
        """Convert database row to UserLearningProfile object."""
        try:
            return UserLearningProfile(
                user_id=row[0],
                preferences=self._deserialize_data(row[1]),
                interaction_patterns=self._deserialize_data(row[2]),
                learning_goals=json.loads(row[3]),
                privacy_settings=self._deserialize_data(row[4]),
                data_retention_days=row[5],
                created_at=datetime.fromisoformat(row[6]),
                updated_at=datetime.fromisoformat(row[7]),
                total_interactions=row[8],
                last_interaction=datetime.fromisoformat(row[9]) if row[9] else None
            )
        except Exception as e:
            self.logger.error(f"Failed to convert row to UserLearningProfile: {e}")
            return None
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._stats_task = asyncio.create_task(self._statistics_loop())
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval * 3600)  # Convert hours to seconds
                await self._cleanup_expired_data()
                await self._cleanup_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _statistics_loop(self) -> None:
        """Background statistics update loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                await self._update_statistics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in statistics loop: {e}")
    
    async def _cleanup_expired_data(self) -> int:
        """Clean up expired data records."""
        try:
            current_time = datetime.now().isoformat()
            
            async with aiosqlite.connect(self._db_path) as db:
                # Count records to be deleted
                async with db.execute("""
                    SELECT COUNT(*) FROM learning_data 
                    WHERE expires_at IS NOT NULL AND expires_at <= ?
                """, (current_time,)) as cursor:
                    row = await cursor.fetchone()
                    count_to_delete = row[0] if row else 0
                
                # Delete expired records
                await db.execute("""
                    DELETE FROM learning_data 
                    WHERE expires_at IS NOT NULL AND expires_at <= ?
                """, (current_time,))
                await db.commit()
            
            if count_to_delete > 0:
                self.logger.info(f"Cleaned up {count_to_delete} expired data records")
                self._stats['last_cleanup'] = datetime.now()
            
            return count_to_delete
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired data: {e}")
            return 0
    
    async def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            # Find expired cache entries (older than 1 hour)
            for key, timestamp in self._cache_timestamps.items():
                if (current_time - timestamp).seconds > 3600:  # 1 hour
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                if key.startswith("profile_"):
                    user_id = key.replace("profile_", "")
                    self._user_profiles_cache.pop(user_id, None)
                else:
                    self._data_cache.pop(key, None)
                
                self._cache_timestamps.pop(key, None)
            
            # Limit cache size
            if len(self._data_cache) > self._cache_size:
                # Remove oldest entries
                sorted_items = sorted(
                    self._cache_timestamps.items(), 
                    key=lambda x: x[1]
                )
                
                items_to_remove = len(self._data_cache) - self._cache_size
                for key, _ in sorted_items[:items_to_remove]:
                    if not key.startswith("profile_"):
                        self._data_cache.pop(key, None)
                        self._cache_timestamps.pop(key, None)
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup cache: {e}")