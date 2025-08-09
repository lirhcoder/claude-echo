"""User Preference Learner - Example Learning Algorithm Implementation

This module demonstrates how to implement a concrete learning algorithm
using the BaseLearner framework. It learns user preferences and interaction
patterns to provide personalized experiences.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np

from loguru import logger

from ..base_learner import BaseLearner, LearningContext, LearningResult, LearningMode
from ..learning_events import LearningEventFactory


class UserPreferenceLearner(BaseLearner):
    """
    Example learner that analyzes user preferences and interaction patterns.
    
    This learner demonstrates:
    - Pattern analysis from user interaction data
    - Preference extraction and scoring
    - Personalization recommendations
    - Model persistence and loading
    """
    
    @property
    def learner_type(self) -> str:
        return "user_preference_learner"
    
    @property
    def supported_learning_modes(self) -> List[LearningMode]:
        return [LearningMode.ONLINE, LearningMode.BATCH]
    
    @property
    def input_data_types(self) -> List[str]:
        return ["user_interaction", "session_update", "task_completion"]
    
    @property
    def output_types(self) -> List[str]:
        return ["user_preferences", "interaction_patterns", "personalization_recommendations"]
    
    async def _initialize_learner(self) -> None:
        """Initialize user preference learner."""
        # Learning parameters
        self.preference_threshold = self.config.get('preference_threshold', 0.6)
        self.pattern_window_days = self.config.get('pattern_window_days', 30)
        self.min_interactions = self.config.get('min_interactions', 5)
        
        # User preference models
        self.user_preferences = {}  # user_id -> preference data
        self.interaction_patterns = {}  # user_id -> pattern data
        self.preference_weights = {
            'task_type': 0.3,
            'interaction_time': 0.2,
            'response_style': 0.2,
            'session_duration': 0.15,
            'error_tolerance': 0.15
        }
        
        self.logger.info("User Preference Learner initialized")
    
    async def _learn_from_data(self, data: List[Dict[str, Any]], 
                             context: LearningContext) -> LearningResult:
        """Learn user preferences from interaction data."""
        try:
            users_analyzed = set()
            preferences_updated = 0
            patterns_detected = 0
            recommendations_generated = 0
            
            # Group data by user
            user_data = defaultdict(list)
            for item in data:
                user_id = item.get('user_id')
                if user_id and user_id != 'anonymous':
                    user_data[user_id].append(item)
            
            # Analyze preferences for each user
            for user_id, user_interactions in user_data.items():
                if len(user_interactions) < self.min_interactions:
                    continue
                
                # Analyze user preferences
                preferences = await self._analyze_user_preferences(user_id, user_interactions)
                
                # Detect interaction patterns
                patterns = await self._detect_interaction_patterns(user_id, user_interactions)
                
                # Generate personalization recommendations
                recommendations = await self._generate_recommendations(user_id, preferences, patterns)
                
                # Update user models
                self.user_preferences[user_id] = preferences
                self.interaction_patterns[user_id] = patterns
                
                users_analyzed.add(user_id)
                preferences_updated += len(preferences)
                patterns_detected += len(patterns)
                recommendations_generated += len(recommendations)
                
                # Emit user preference update event
                if preferences:
                    event = LearningEventFactory.user_pattern_detected(
                        user_id=user_id,
                        pattern_type="user_preferences",
                        pattern_data={
                            "preferences": preferences,
                            "patterns": patterns,
                            "recommendations": recommendations
                        },
                        confidence=self._calculate_confidence(user_interactions)
                    )
                    system_event = event.to_system_event()
                    await self.event_system.emit(system_event)
            
            # Generate learning result
            improvements = []
            if users_analyzed:
                improvements.append(f"Analyzed preferences for {len(users_analyzed)} users")
            if preferences_updated > 0:
                improvements.append(f"Updated {preferences_updated} preference categories")
            if patterns_detected > 0:
                improvements.append(f"Detected {patterns_detected} interaction patterns")
            if recommendations_generated > 0:
                improvements.append(f"Generated {recommendations_generated} personalization recommendations")
            
            metrics = {
                "users_analyzed": len(users_analyzed),
                "preferences_updated": preferences_updated,
                "patterns_detected": patterns_detected,
                "recommendations_generated": recommendations_generated,
                "data_quality": self._assess_data_quality(data)
            }
            
            confidence_score = self._calculate_overall_confidence(user_data)
            
            return LearningResult(
                success=True,
                improvements=improvements,
                metrics=metrics,
                confidence_score=confidence_score,
                data_points_processed=len(data)
            )
            
        except Exception as e:
            self.logger.error(f"User preference learning failed: {e}")
            return LearningResult(
                success=False,
                error_message=str(e),
                data_points_processed=len(data)
            )
    
    async def _analyze_user_preferences(self, user_id: str, 
                                      interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user preferences from interaction data."""
        preferences = {
            'task_types': {},
            'interaction_times': [],
            'response_styles': {},
            'session_durations': [],
            'error_patterns': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Analyze task type preferences
        task_types = []
        for interaction in interactions:
            data_content = interaction.get('data_content', {})
            
            # Extract task type information
            if 'task_data' in data_content:
                task_type = data_content['task_data'].get('type', 'unknown')
                task_types.append(task_type)
                
                # Analyze task success/failure
                task_success = data_content['task_data'].get('success', True)
                if task_type not in preferences['error_patterns']:
                    preferences['error_patterns'][task_type] = {'total': 0, 'failures': 0}
                preferences['error_patterns'][task_type]['total'] += 1
                if not task_success:
                    preferences['error_patterns'][task_type]['failures'] += 1
            
            # Extract interaction timing
            if 'interaction_timestamp' in data_content:
                try:
                    timestamp = datetime.fromisoformat(data_content['interaction_timestamp'])
                    preferences['interaction_times'].append(timestamp.hour)
                except:
                    pass
            
            # Extract session duration patterns
            if 'session_context' in data_content:
                duration = data_content['session_context'].get('session_duration')
                if duration:
                    preferences['session_durations'].append(duration)
        
        # Calculate task type preferences
        task_counter = Counter(task_types)
        total_tasks = sum(task_counter.values())
        for task_type, count in task_counter.items():
            preferences['task_types'][task_type] = {
                'count': count,
                'preference_score': count / total_tasks,
                'frequency': count / len(interactions)
            }
        
        # Analyze preferred interaction times
        if preferences['interaction_times']:
            hour_counter = Counter(preferences['interaction_times'])
            preferences['preferred_hours'] = dict(hour_counter.most_common(5))
            
            # Determine time preference category
            avg_hour = sum(preferences['interaction_times']) / len(preferences['interaction_times'])
            if 6 <= avg_hour < 12:
                preferences['time_preference'] = 'morning'
            elif 12 <= avg_hour < 18:
                preferences['time_preference'] = 'afternoon'
            elif 18 <= avg_hour < 22:
                preferences['time_preference'] = 'evening'
            else:
                preferences['time_preference'] = 'night'
        
        # Analyze session duration preferences
        if preferences['session_durations']:
            avg_duration = sum(preferences['session_durations']) / len(preferences['session_durations'])
            preferences['avg_session_duration'] = avg_duration
            
            if avg_duration < 300:  # 5 minutes
                preferences['session_style'] = 'brief'
            elif avg_duration < 1800:  # 30 minutes
                preferences['session_style'] = 'moderate'
            else:
                preferences['session_style'] = 'extended'
        
        # Calculate error tolerance
        total_errors = sum(ep['failures'] for ep in preferences['error_patterns'].values())
        total_tasks = sum(ep['total'] for ep in preferences['error_patterns'].values())
        if total_tasks > 0:
            error_rate = total_errors / total_tasks
            if error_rate < 0.05:
                preferences['error_tolerance'] = 'low'
            elif error_rate < 0.15:
                preferences['error_tolerance'] = 'moderate'
            else:
                preferences['error_tolerance'] = 'high'
        
        return preferences
    
    async def _detect_interaction_patterns(self, user_id: str, 
                                         interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in user interactions."""
        patterns = {
            'temporal_patterns': {},
            'behavioral_patterns': {},
            'usage_patterns': {},
            'consistency_metrics': {}
        }
        
        # Sort interactions by timestamp
        timestamped_interactions = []
        for interaction in interactions:
            try:
                timestamp_str = interaction.get('timestamp') or interaction.get('data_content', {}).get('interaction_timestamp')
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    timestamped_interactions.append((timestamp, interaction))
            except:
                continue
        
        timestamped_interactions.sort(key=lambda x: x[0])
        
        if len(timestamped_interactions) < 2:
            return patterns
        
        # Analyze temporal patterns
        time_gaps = []
        for i in range(1, len(timestamped_interactions)):
            gap = (timestamped_interactions[i][0] - timestamped_interactions[i-1][0]).total_seconds()
            time_gaps.append(gap)
        
        if time_gaps:
            patterns['temporal_patterns'] = {
                'avg_time_between_interactions': sum(time_gaps) / len(time_gaps),
                'interaction_frequency': len(interactions) / max((timestamped_interactions[-1][0] - timestamped_interactions[0][0]).days, 1),
                'regularity_score': self._calculate_regularity(time_gaps)
            }
        
        # Analyze behavioral patterns
        task_sequences = []
        session_ids = set()
        
        for _, interaction in timestamped_interactions:
            data_content = interaction.get('data_content', {})
            
            # Track task sequences
            if 'task_data' in data_content:
                task_type = data_content['task_data'].get('type')
                if task_type:
                    task_sequences.append(task_type)
            
            # Track unique sessions
            session_id = interaction.get('session_id')
            if session_id:
                session_ids.add(session_id)
        
        if task_sequences:
            # Find common task sequences
            sequence_patterns = self._find_sequence_patterns(task_sequences)
            patterns['behavioral_patterns'] = {
                'common_sequences': sequence_patterns,
                'task_diversity': len(set(task_sequences)) / len(task_sequences) if task_sequences else 0,
                'sequence_consistency': self._calculate_sequence_consistency(task_sequences)
            }
        
        # Analyze usage patterns
        patterns['usage_patterns'] = {
            'total_sessions': len(session_ids),
            'interactions_per_session': len(interactions) / max(len(session_ids), 1),
            'engagement_level': self._calculate_engagement_level(interactions)
        }
        
        # Calculate consistency metrics
        patterns['consistency_metrics'] = {
            'temporal_consistency': patterns['temporal_patterns'].get('regularity_score', 0),
            'behavioral_consistency': patterns['behavioral_patterns'].get('sequence_consistency', 0),
            'overall_consistency': 0  # Will be calculated below
        }
        
        # Calculate overall consistency
        consistency_scores = [
            patterns['consistency_metrics']['temporal_consistency'],
            patterns['consistency_metrics']['behavioral_consistency']
        ]
        patterns['consistency_metrics']['overall_consistency'] = sum(consistency_scores) / len(consistency_scores)
        
        return patterns
    
    async def _generate_recommendations(self, user_id: str, 
                                      preferences: Dict[str, Any],
                                      patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalization recommendations based on user preferences and patterns."""
        recommendations = []
        
        # Task type recommendations
        if 'task_types' in preferences and preferences['task_types']:
            preferred_tasks = sorted(
                preferences['task_types'].items(),
                key=lambda x: x[1]['preference_score'],
                reverse=True
            )[:3]  # Top 3 preferred task types
            
            recommendations.append({
                'type': 'task_prioritization',
                'priority': 'high',
                'recommendation': 'Prioritize preferred task types',
                'details': {
                    'preferred_task_types': [task[0] for task in preferred_tasks],
                    'rationale': 'User shows consistent preference for these task types'
                }
            })
        
        # Timing recommendations
        if 'time_preference' in preferences:
            time_pref = preferences['time_preference']
            recommendations.append({
                'type': 'interaction_timing',
                'priority': 'medium',
                'recommendation': f'Optimize for {time_pref} interactions',
                'details': {
                    'preferred_time': time_pref,
                    'suggested_actions': [
                        f'Pre-load resources during {time_pref} hours',
                        f'Schedule maintenance outside {time_pref} hours'
                    ]
                }
            })
        
        # Session duration recommendations
        if 'session_style' in preferences:
            session_style = preferences['session_style']
            if session_style == 'brief':
                recommendations.append({
                    'type': 'session_optimization',
                    'priority': 'high',
                    'recommendation': 'Optimize for quick, efficient interactions',
                    'details': {
                        'session_style': session_style,
                        'suggested_actions': [
                            'Reduce response verbosity',
                            'Prioritize direct answers',
                            'Minimize confirmation steps'
                        ]
                    }
                })
            elif session_style == 'extended':
                recommendations.append({
                    'type': 'session_optimization',
                    'priority': 'medium',
                    'recommendation': 'Provide detailed, comprehensive responses',
                    'details': {
                        'session_style': session_style,
                        'suggested_actions': [
                            'Include additional context and examples',
                            'Offer related suggestions',
                            'Provide detailed explanations'
                        ]
                    }
                })
        
        # Error handling recommendations
        if 'error_tolerance' in preferences:
            error_tolerance = preferences['error_tolerance']
            if error_tolerance == 'low':
                recommendations.append({
                    'type': 'error_handling',
                    'priority': 'high',
                    'recommendation': 'Implement proactive error prevention',
                    'details': {
                        'error_tolerance': error_tolerance,
                        'suggested_actions': [
                            'Add input validation and confirmation',
                            'Provide clear error messages',
                            'Implement graceful fallbacks'
                        ]
                    }
                })
        
        # Consistency-based recommendations
        consistency = patterns.get('consistency_metrics', {}).get('overall_consistency', 0)
        if consistency > 0.7:
            recommendations.append({
                'type': 'predictive_assistance',
                'priority': 'medium',
                'recommendation': 'Enable predictive assistance based on consistent patterns',
                'details': {
                    'consistency_score': consistency,
                    'suggested_actions': [
                        'Pre-populate common requests',
                        'Suggest next likely actions',
                        'Automate repetitive tasks'
                    ]
                }
            })
        
        # Engagement-based recommendations
        engagement = patterns.get('usage_patterns', {}).get('engagement_level', 0)
        if engagement < 0.5:
            recommendations.append({
                'type': 'engagement_improvement',
                'priority': 'medium',
                'recommendation': 'Improve user engagement',
                'details': {
                    'engagement_level': engagement,
                    'suggested_actions': [
                        'Introduce interactive elements',
                        'Provide progress feedback',
                        'Gamify repetitive tasks'
                    ]
                }
            })
        
        return recommendations
    
    def _calculate_confidence(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on data quality and quantity."""
        if not interactions:
            return 0.0
        
        # Base confidence on quantity
        quantity_score = min(len(interactions) / 50, 1.0)  # Max confidence at 50 interactions
        
        # Adjust for data quality
        complete_interactions = sum(
            1 for interaction in interactions
            if interaction.get('data_content') and 
               len(interaction.get('data_content', {})) >= 3  # At least 3 data fields
        )
        quality_score = complete_interactions / len(interactions)
        
        # Adjust for recency
        recent_interactions = sum(
            1 for interaction in interactions
            if self._is_recent(interaction.get('timestamp'))
        )
        recency_score = recent_interactions / len(interactions)
        
        # Combined confidence score
        confidence = (quantity_score * 0.4 + quality_score * 0.4 + recency_score * 0.2)
        return min(confidence, 1.0)
    
    def _calculate_overall_confidence(self, user_data: Dict[str, List]) -> float:
        """Calculate overall confidence across all users."""
        if not user_data:
            return 0.0
        
        user_confidences = [
            self._calculate_confidence(interactions)
            for interactions in user_data.values()
        ]
        
        return sum(user_confidences) / len(user_confidences)
    
    def _assess_data_quality(self, data: List[Dict[str, Any]]) -> float:
        """Assess overall data quality."""
        if not data:
            return 0.0
        
        quality_factors = []
        
        # Completeness
        complete_records = sum(
            1 for item in data
            if item.get('user_id') and item.get('data_content')
        )
        completeness = complete_records / len(data)
        quality_factors.append(completeness)
        
        # Consistency
        consistent_records = sum(
            1 for item in data
            if self._is_consistent_record(item)
        )
        consistency = consistent_records / len(data)
        quality_factors.append(consistency)
        
        # Recency
        recent_records = sum(
            1 for item in data
            if self._is_recent(item.get('timestamp'))
        )
        recency = recent_records / len(data)
        quality_factors.append(recency)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _calculate_regularity(self, time_gaps: List[float]) -> float:
        """Calculate regularity score from time gaps."""
        if len(time_gaps) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower = more regular)
        mean_gap = sum(time_gaps) / len(time_gaps)
        if mean_gap == 0:
            return 1.0
        
        variance = sum((gap - mean_gap) ** 2 for gap in time_gaps) / len(time_gaps)
        std_dev = variance ** 0.5
        cv = std_dev / mean_gap
        
        # Convert to regularity score (0-1, higher = more regular)
        regularity = max(0, 1 - min(cv, 2) / 2)
        return regularity
    
    def _find_sequence_patterns(self, sequences: List[str]) -> Dict[str, int]:
        """Find common patterns in task sequences."""
        if len(sequences) < 2:
            return {}
        
        # Look for 2-gram and 3-gram patterns
        patterns = Counter()
        
        # 2-grams
        for i in range(len(sequences) - 1):
            pattern = (sequences[i], sequences[i + 1])
            patterns[' -> '.join(pattern)] += 1
        
        # 3-grams (if enough data)
        if len(sequences) >= 3:
            for i in range(len(sequences) - 2):
                pattern = (sequences[i], sequences[i + 1], sequences[i + 2])
                patterns[' -> '.join(pattern)] += 1
        
        # Return only patterns that occur more than once
        return {pattern: count for pattern, count in patterns.items() if count > 1}
    
    def _calculate_sequence_consistency(self, sequences: List[str]) -> float:
        """Calculate consistency in task sequences."""
        if len(sequences) < 2:
            return 0.0
        
        # Calculate how often similar patterns repeat
        pattern_counts = Counter()
        for i in range(len(sequences) - 1):
            pattern = (sequences[i], sequences[i + 1])
            pattern_counts[pattern] += 1
        
        # Consistency is based on pattern repetition
        total_transitions = len(sequences) - 1
        repeated_patterns = sum(count for count in pattern_counts.values() if count > 1)
        
        return repeated_patterns / total_transitions if total_transitions > 0 else 0.0
    
    def _calculate_engagement_level(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate user engagement level."""
        if not interactions:
            return 0.0
        
        engagement_factors = []
        
        # Interaction frequency
        time_span = self._get_time_span(interactions)
        if time_span > 0:
            frequency = len(interactions) / time_span  # interactions per day
            frequency_score = min(frequency / 5, 1.0)  # Normalize to max 5 interactions/day
            engagement_factors.append(frequency_score)
        
        # Task completion rate
        completed_tasks = sum(
            1 for interaction in interactions
            if interaction.get('data_content', {}).get('task_data', {}).get('success', False)
        )
        completion_rate = completed_tasks / len(interactions)
        engagement_factors.append(completion_rate)
        
        # Session diversity
        unique_sessions = len(set(
            interaction.get('session_id')
            for interaction in interactions
            if interaction.get('session_id')
        ))
        session_diversity = unique_sessions / len(interactions)
        engagement_factors.append(session_diversity)
        
        return sum(engagement_factors) / len(engagement_factors) if engagement_factors else 0.0
    
    def _is_recent(self, timestamp_str: Optional[str]) -> bool:
        """Check if timestamp is recent (within last 7 days)."""
        if not timestamp_str:
            return False
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            return (datetime.now() - timestamp).days <= 7
        except:
            return False
    
    def _is_consistent_record(self, record: Dict[str, Any]) -> bool:
        """Check if record has consistent structure."""
        required_fields = ['user_id', 'data_type', 'data_content']
        return all(field in record for field in required_fields)
    
    def _get_time_span(self, interactions: List[Dict[str, Any]]) -> float:
        """Get time span of interactions in days."""
        timestamps = []
        
        for interaction in interactions:
            timestamp_str = interaction.get('timestamp') or interaction.get('data_content', {}).get('interaction_timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    timestamps.append(timestamp)
                except:
                    continue
        
        if len(timestamps) < 2:
            return 1.0  # Default to 1 day
        
        return (max(timestamps) - min(timestamps)).days + 1
    
    async def _save_model(self) -> bool:
        """Save user preference models to persistent storage."""
        try:
            model_data = {
                'user_preferences': self.user_preferences,
                'interaction_patterns': self.interaction_patterns,
                'preference_weights': self.preference_weights,
                'model_version': self._model_version,
                'last_updated': datetime.now().isoformat()
            }
            
            model_file = self._model_storage_path / "user_preference_model.json"
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved user preference model with {len(self.user_preferences)} users")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save user preference model: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """Load existing user preference models."""
        try:
            model_file = self._model_storage_path / "user_preference_model.json"
            if not model_file.exists():
                self.logger.info("No existing user preference model found, starting fresh")
                return True
            
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            self.user_preferences = model_data.get('user_preferences', {})
            self.interaction_patterns = model_data.get('interaction_patterns', {})
            self.preference_weights = model_data.get('preference_weights', self.preference_weights)
            
            self.logger.info(f"Loaded user preference model with {len(self.user_preferences)} users")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load user preference model: {e}")
            return False
    
    async def _cleanup_learner(self) -> None:
        """Cleanup user preference learner resources."""
        await self._save_model()
        self.user_preferences.clear()
        self.interaction_patterns.clear()
        self.logger.info("User preference learner cleanup complete")


# Register the learner
from ..base_learner import LearnerRegistry
LearnerRegistry.register("user_preference_learner", UserPreferenceLearner)