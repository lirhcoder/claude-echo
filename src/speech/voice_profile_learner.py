"""Voice Profile Learner - Learning User Voice Characteristics

Implements learning algorithms for user voice characteristics including pitch patterns,
speech rate, volume levels, and accent markers for personalized speech recognition.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import uuid
from collections import defaultdict, deque

from loguru import logger
import librosa
from scipy import stats, signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..learning.base_learner import BaseLearner, LearningMode, LearningContext, LearningResult
from ..core.event_system import EventSystem
from .learning_types import (
    VoiceCharacteristics, SpeechLearningData, PersonalizedVoiceProfile,
    VoiceFeatureType, AdaptationParameters
)


class VoiceProfileLearner(BaseLearner):
    """
    Learns and maintains personalized voice profiles for users.
    
    Analyzes audio characteristics including:
    - Pitch patterns and variance
    - Speech rate and rhythm
    - Volume levels and dynamics
    - Accent markers and pronunciation style
    - Intonation patterns
    """
    
    def __init__(self, 
                 learner_id: str,
                 event_system: EventSystem,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the voice profile learner."""
        super().__init__(learner_id, event_system, config)
        
        # Voice analysis configuration
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.hop_length = self.config.get('hop_length', 512)
        self.frame_length = self.config.get('frame_length', 2048)
        self.min_audio_duration = self.config.get('min_audio_duration', 1.0)  # seconds
        self.max_audio_duration = self.config.get('max_audio_duration', 30.0)  # seconds
        
        # Learning parameters
        self.feature_smoothing_factor = self.config.get('feature_smoothing_factor', 0.1)
        self.outlier_threshold = self.config.get('outlier_threshold', 2.0)  # standard deviations
        self.min_samples_for_update = self.config.get('min_samples_for_update', 5)
        self.profile_update_interval = self.config.get('profile_update_interval', 100)  # samples
        
        # Data storage
        self.voice_profiles: Dict[str, PersonalizedVoiceProfile] = {}
        self.recent_audio_features: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.feature_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Analysis tools
        self.scaler = StandardScaler()
        self.pitch_detector = None  # Will be initialized in _initialize_learner
        
        # Performance tracking
        self.analysis_stats = {
            'total_audio_analyzed': 0,
            'successful_feature_extractions': 0,
            'profile_updates': 0,
            'average_analysis_time': 0.0
        }
        
        logger.info(f"VoiceProfileLearner initialized for {learner_id}")
    
    @property
    def learner_type(self) -> str:
        return "voice_profile"
    
    @property
    def supported_learning_modes(self) -> List[LearningMode]:
        return [LearningMode.ONLINE, LearningMode.BATCH, LearningMode.UNSUPERVISED]
    
    @property
    def input_data_types(self) -> List[str]:
        return ["speech_audio", "recognition_results", "user_feedback"]
    
    @property
    def output_types(self) -> List[str]:
        return ["voice_characteristics", "adaptation_parameters", "profile_updates"]
    
    async def _initialize_learner(self) -> None:
        """Initialize voice analysis tools and load existing profiles."""
        try:
            # Initialize pitch detection
            self.pitch_detector = self._initialize_pitch_detector()
            
            # Load existing voice profiles
            await self._load_voice_profiles()
            
            # Setup feature extraction pipeline
            await self._setup_feature_pipeline()
            
            self.logger.info("Voice profile learner initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize voice profile learner: {e}")
            raise
    
    async def _learn_from_data(self, data: List[Dict[str, Any]], 
                             context: LearningContext) -> LearningResult:
        """
        Learn voice characteristics from audio and recognition data.
        
        Args:
            data: List of speech learning data points
            context: Learning context information
            
        Returns:
            Learning result with voice profile updates
        """
        start_time = datetime.now()
        result = LearningResult(success=False)
        
        try:
            if not data:
                result.error_message = "No data provided for learning"
                return result
            
            user_id = context.user_id or "default_user"
            improvements = []
            processed_count = 0
            
            # Get or create voice profile
            voice_profile = await self._get_or_create_profile(user_id)
            
            # Process each data point
            for data_point in data:
                try:
                    # Convert to SpeechLearningData if needed
                    if isinstance(data_point, dict):
                        speech_data = self._dict_to_speech_data(data_point)
                    else:
                        speech_data = data_point
                    
                    # Extract voice features
                    features = await self._extract_voice_features(speech_data)
                    if features:
                        # Update voice characteristics
                        updated = await self._update_voice_characteristics(
                            voice_profile.voice_characteristics, features, speech_data
                        )
                        
                        if updated:
                            improvements.append(f"Updated voice characteristics from audio sample")
                            processed_count += 1
                        
                        # Store features for batch analysis
                        self.recent_audio_features[user_id].append(features)
                        self.feature_history[user_id].append({
                            'features': features,
                            'timestamp': datetime.now(),
                            'context': speech_data.context_type
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process data point: {e}")
                    continue
            
            # Perform batch analysis if enough samples
            if len(self.recent_audio_features[user_id]) >= self.min_samples_for_update:
                batch_improvements = await self._perform_batch_analysis(voice_profile)
                improvements.extend(batch_improvements)
            
            # Update adaptation parameters based on learned characteristics
            adaptation_updates = await self._update_adaptation_parameters(voice_profile)
            improvements.extend(adaptation_updates)
            
            # Save updated profile
            await self._save_voice_profile(voice_profile)
            
            # Calculate confidence based on sample size and consistency
            confidence_score = self._calculate_learning_confidence(voice_profile, processed_count)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(voice_profile)
            
            # Update performance metrics
            self.analysis_stats['total_audio_analyzed'] += len(data)
            self.analysis_stats['successful_feature_extractions'] += processed_count
            self.analysis_stats['profile_updates'] += 1
            
            # Create successful result
            result = LearningResult(
                success=True,
                improvements=improvements,
                recommendations=recommendations,
                confidence_score=confidence_score,
                data_points_processed=processed_count,
                metrics={
                    'voice_profile_version': voice_profile.profile_version,
                    'total_utterances': voice_profile.total_utterances,
                    'recognition_accuracy': voice_profile.recognition_accuracy,
                    'adaptation_effectiveness': voice_profile.adaptation_effectiveness
                }
            )
            
            self.logger.info(f"Successfully learned from {processed_count} voice samples for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Voice profile learning failed: {e}")
            result.error_message = str(e)
        
        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result
    
    async def _extract_voice_features(self, speech_data: SpeechLearningData) -> Optional[Dict[str, Any]]:
        """
        Extract voice features from audio data.
        
        Args:
            speech_data: Speech learning data with audio features
            
        Returns:
            Dictionary of extracted voice features
        """
        try:
            if speech_data.audio_features is None:
                self.logger.warning("No audio features available for analysis")
                return None
            
            if speech_data.audio_duration < self.min_audio_duration:
                self.logger.debug(f"Audio too short for analysis: {speech_data.audio_duration}s")
                return None
            
            audio_data = speech_data.audio_features
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            features = {}
            
            # Basic audio properties
            features['duration'] = speech_data.audio_duration
            features['sample_rate'] = self.sample_rate
            
            # Pitch analysis
            pitch_features = await self._analyze_pitch(audio_data)
            features.update(pitch_features)
            
            # Speech rate analysis
            rate_features = await self._analyze_speech_rate(audio_data, speech_data.original_text)
            features.update(rate_features)
            
            # Volume/energy analysis
            energy_features = await self._analyze_energy(audio_data)
            features.update(energy_features)
            
            # Spectral features
            spectral_features = await self._analyze_spectral_features(audio_data)
            features.update(spectral_features)
            
            # Prosodic features
            prosodic_features = await self._analyze_prosodic_features(audio_data)
            features.update(prosodic_features)
            
            # Quality assessment
            features['feature_quality'] = self._assess_feature_quality(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    async def _analyze_pitch(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze pitch characteristics."""
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, 
                sr=self.sample_rate,
                hop_length=self.hop_length,
                threshold=0.1
            )
            
            # Extract fundamental frequency
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_track.append(pitch)
            
            if not pitch_track:
                return {'pitch_mean': 0.0, 'pitch_std': 0.0}
            
            pitch_array = np.array(pitch_track)
            
            return {
                'pitch_mean': float(np.mean(pitch_array)),
                'pitch_std': float(np.std(pitch_array)),
                'pitch_median': float(np.median(pitch_array)),
                'pitch_range': float(np.ptp(pitch_array)),
                'pitch_quartiles': [float(q) for q in np.percentile(pitch_array, [25, 50, 75])],
                'pitch_track_length': len(pitch_track),
                'voiced_frames_ratio': len(pitch_track) / pitches.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Pitch analysis failed: {e}")
            return {'pitch_mean': 0.0, 'pitch_std': 0.0}
    
    async def _analyze_speech_rate(self, audio_data: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze speech rate characteristics."""
        try:
            duration = len(audio_data) / self.sample_rate
            word_count = len(text.split()) if text else 0
            
            if duration == 0 or word_count == 0:
                return {'words_per_minute': 0.0, 'syllables_per_second': 0.0}
            
            # Calculate words per minute
            words_per_minute = (word_count / duration) * 60
            
            # Estimate syllables (simple heuristic)
            syllable_count = sum(self._count_syllables(word) for word in text.split())
            syllables_per_second = syllable_count / duration
            
            # Analyze pause patterns using silence detection
            pause_analysis = await self._analyze_pauses(audio_data)
            
            return {
                'words_per_minute': words_per_minute,
                'syllables_per_second': syllables_per_second,
                'word_count': word_count,
                'syllable_count': syllable_count,
                **pause_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Speech rate analysis failed: {e}")
            return {'words_per_minute': 0.0, 'syllables_per_second': 0.0}
    
    async def _analyze_energy(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze energy/volume characteristics."""
        try:
            # Calculate RMS energy
            rms_energy = librosa.feature.rms(
                y=audio_data, 
                hop_length=self.hop_length,
                frame_length=self.frame_length
            )[0]
            
            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms_energy, ref=np.max)
            
            return {
                'rms_mean': float(np.mean(rms_energy)),
                'rms_std': float(np.std(rms_energy)),
                'rms_db_mean': float(np.mean(rms_db)),
                'rms_db_std': float(np.std(rms_db)),
                'dynamic_range': float(np.ptp(rms_db)),
                'energy_percentiles': [float(p) for p in np.percentile(rms_energy, [10, 25, 50, 75, 90])]
            }
            
        except Exception as e:
            self.logger.error(f"Energy analysis failed: {e}")
            return {'rms_mean': 0.0, 'rms_std': 0.0}
    
    async def _analyze_spectral_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral characteristics."""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, 
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y=audio_data,
                hop_length=self.hop_length
            )[0]
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zcr)),
                'zero_crossing_rate_std': float(np.std(zcr))
            }
            
        except Exception as e:
            self.logger.error(f"Spectral analysis failed: {e}")
            return {}
    
    async def _analyze_prosodic_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze prosodic characteristics."""
        try:
            # Compute chromagram for harmony analysis
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Compute tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            return {
                'tempo': float(tempo),
                'beat_count': len(beats),
                'chroma_variance': float(np.var(chroma)),
                'harmonic_complexity': float(np.mean(np.std(chroma, axis=0)))
            }
            
        except Exception as e:
            self.logger.error(f"Prosodic analysis failed: {e}")
            return {'tempo': 0.0}
    
    async def _analyze_pauses(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze pause patterns in speech."""
        try:
            # Simple silence detection based on energy threshold
            frame_length = 1024
            hop_length = 512
            
            # Calculate frame-wise energy
            energy = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy.append(np.sum(frame ** 2))
            
            energy = np.array(energy)
            
            # Determine silence threshold (10% of max energy)
            silence_threshold = np.max(energy) * 0.1
            
            # Find silent frames
            silent_frames = energy < silence_threshold
            
            # Find pause segments
            pauses = []
            in_pause = False
            pause_start = 0
            
            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_pause:
                    in_pause = True
                    pause_start = i
                elif not is_silent and in_pause:
                    in_pause = False
                    pause_duration = (i - pause_start) * hop_length / self.sample_rate
                    if pause_duration > 0.1:  # Only count pauses longer than 100ms
                        pauses.append(pause_duration)
            
            if pauses:
                return {
                    'pause_count': len(pauses),
                    'total_pause_time': sum(pauses),
                    'average_pause_duration': np.mean(pauses),
                    'pause_duration_std': np.std(pauses),
                    'longest_pause': max(pauses),
                    'pause_rate': len(pauses) / (len(audio_data) / self.sample_rate)
                }
            else:
                return {
                    'pause_count': 0,
                    'total_pause_time': 0.0,
                    'average_pause_duration': 0.0,
                    'pause_duration_std': 0.0,
                    'longest_pause': 0.0,
                    'pause_rate': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Pause analysis failed: {e}")
            return {'pause_count': 0}
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting heuristic."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _assess_feature_quality(self, features: Dict[str, Any]) -> float:
        """Assess the quality of extracted features."""
        try:
            quality_score = 1.0
            
            # Check for missing critical features
            critical_features = ['pitch_mean', 'words_per_minute', 'rms_mean']
            for feature in critical_features:
                if feature not in features or features[feature] == 0:
                    quality_score -= 0.2
            
            # Check for reasonable pitch values
            if 'pitch_mean' in features:
                pitch = features['pitch_mean']
                if pitch < 50 or pitch > 800:  # Outside human speech range
                    quality_score -= 0.1
            
            # Check for reasonable speech rate
            if 'words_per_minute' in features:
                wpm = features['words_per_minute']
                if wpm < 50 or wpm > 400:  # Outside normal speech rate
                    quality_score -= 0.1
            
            return max(0.0, quality_score)
            
        except Exception:
            return 0.5  # Default moderate quality
    
    async def _update_voice_characteristics(self, 
                                          characteristics: VoiceCharacteristics,
                                          features: Dict[str, Any],
                                          speech_data: SpeechLearningData) -> bool:
        """Update voice characteristics with new features."""
        try:
            if features.get('feature_quality', 0) < 0.3:
                return False  # Skip low-quality features
            
            alpha = self.feature_smoothing_factor  # Learning rate
            
            # Update pitch characteristics
            if 'pitch_mean' in features and features['pitch_mean'] > 0:
                if characteristics.average_pitch == 0:
                    characteristics.average_pitch = features['pitch_mean']
                    characteristics.pitch_variance = features.get('pitch_std', 0) ** 2
                else:
                    # Exponential moving average
                    new_pitch = features['pitch_mean']
                    if self._is_not_outlier(new_pitch, characteristics.average_pitch, characteristics.pitch_variance):
                        characteristics.average_pitch = (
                            alpha * new_pitch + (1 - alpha) * characteristics.average_pitch
                        )
                        # Update variance
                        new_variance = features.get('pitch_std', 0) ** 2
                        characteristics.pitch_variance = (
                            alpha * new_variance + (1 - alpha) * characteristics.pitch_variance
                        )
            
            # Update speech rate
            if 'words_per_minute' in features and features['words_per_minute'] > 0:
                if characteristics.average_speech_rate == 0:
                    characteristics.average_speech_rate = features['words_per_minute']
                else:
                    new_rate = features['words_per_minute']
                    if self._is_not_outlier(new_rate, characteristics.average_speech_rate, 
                                          characteristics.speech_rate_variance):
                        characteristics.average_speech_rate = (
                            alpha * new_rate + (1 - alpha) * characteristics.average_speech_rate
                        )
            
            # Update volume characteristics
            if 'rms_mean' in features:
                if characteristics.average_volume == 0:
                    characteristics.average_volume = features['rms_mean']
                    characteristics.volume_variance = features.get('rms_std', 0) ** 2
                else:
                    new_volume = features['rms_mean']
                    if self._is_not_outlier(new_volume, characteristics.average_volume,
                                          characteristics.volume_variance):
                        characteristics.average_volume = (
                            alpha * new_volume + (1 - alpha) * characteristics.average_volume
                        )
                        new_variance = features.get('rms_std', 0) ** 2
                        characteristics.volume_variance = (
                            alpha * new_variance + (1 - alpha) * characteristics.volume_variance
                        )
            
            # Update pause patterns
            if 'pause_count' in features and features['pause_count'] > 0:
                pause_rate = features.get('pause_rate', 0)
                if len(characteristics.pause_patterns) < 100:  # Keep recent patterns
                    characteristics.pause_patterns.append(pause_rate)
                else:
                    # Replace oldest
                    characteristics.pause_patterns.pop(0)
                    characteristics.pause_patterns.append(pause_rate)
            
            # Update sample count and timestamp
            characteristics.sample_count += 1
            characteristics.last_updated = datetime.now()
            
            # Recalculate confidence score
            characteristics.confidence_score = self._calculate_characteristics_confidence(characteristics)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update voice characteristics: {e}")
            return False
    
    def _is_not_outlier(self, value: float, mean: float, variance: float) -> bool:
        """Check if a value is not an outlier based on statistical threshold."""
        if variance <= 0:
            return True  # No variance data, accept the value
        
        std_dev = np.sqrt(variance)
        z_score = abs(value - mean) / std_dev if std_dev > 0 else 0
        return z_score <= self.outlier_threshold
    
    def _calculate_characteristics_confidence(self, characteristics: VoiceCharacteristics) -> float:
        """Calculate confidence score for voice characteristics."""
        try:
            # Base confidence on sample count
            sample_confidence = min(characteristics.sample_count / 100.0, 1.0)
            
            # Factor in data quality
            quality_score = 1.0
            
            # Check if key characteristics are populated
            if characteristics.average_pitch > 0:
                quality_score += 0.2
            if characteristics.average_speech_rate > 0:
                quality_score += 0.2
            if characteristics.average_volume > 0:
                quality_score += 0.2
            if len(characteristics.pause_patterns) > 0:
                quality_score += 0.2
            if len(characteristics.accent_markers) > 0:
                quality_score += 0.2
            
            return min(sample_confidence * quality_score, 1.0)
            
        except Exception:
            return 0.5
    
    async def _perform_batch_analysis(self, voice_profile: PersonalizedVoiceProfile) -> List[str]:
        """Perform batch analysis on accumulated features."""
        improvements = []
        
        try:
            user_id = voice_profile.user_id
            recent_features = list(self.recent_audio_features[user_id])
            
            if len(recent_features) < self.min_samples_for_update:
                return improvements
            
            # Analyze feature consistency
            consistency_analysis = self._analyze_feature_consistency(recent_features)
            if consistency_analysis['is_consistent']:
                improvements.append("Voice characteristics showing good consistency")
            else:
                improvements.append("Detected voice pattern variations - adapting model sensitivity")
            
            # Detect speaking style patterns
            style_patterns = self._detect_speaking_style_patterns(recent_features)
            if style_patterns:
                improvements.extend([f"Learned speaking pattern: {pattern}" for pattern in style_patterns])
            
            # Update accent markers based on spectral analysis
            accent_updates = await self._analyze_accent_markers(recent_features)
            if accent_updates:
                voice_profile.voice_characteristics.accent_markers.update(accent_updates)
                improvements.append(f"Updated accent characteristics ({len(accent_updates)} markers)")
            
            # Clear processed features to avoid reprocessing
            self.recent_audio_features[user_id].clear()
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
        
        return improvements
    
    def _analyze_feature_consistency(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consistency of voice features across samples."""
        try:
            if len(features_list) < 3:
                return {'is_consistent': True, 'variability_score': 0.0}
            
            # Extract key features for consistency analysis
            pitch_values = [f.get('pitch_mean', 0) for f in features_list if f.get('pitch_mean', 0) > 0]
            rate_values = [f.get('words_per_minute', 0) for f in features_list if f.get('words_per_minute', 0) > 0]
            volume_values = [f.get('rms_mean', 0) for f in features_list if f.get('rms_mean', 0) > 0]
            
            variability_scores = []
            
            # Calculate coefficient of variation for each feature
            for values in [pitch_values, rate_values, volume_values]:
                if len(values) >= 3:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / mean_val if mean_val > 0 else 0
                    variability_scores.append(cv)
            
            if variability_scores:
                avg_variability = np.mean(variability_scores)
                is_consistent = avg_variability < 0.3  # 30% coefficient of variation threshold
            else:
                avg_variability = 0.0
                is_consistent = True
            
            return {
                'is_consistent': is_consistent,
                'variability_score': avg_variability,
                'feature_count': len(variability_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Consistency analysis failed: {e}")
            return {'is_consistent': True, 'variability_score': 0.0}
    
    def _detect_speaking_style_patterns(self, features_list: List[Dict[str, Any]]) -> List[str]:
        """Detect patterns in speaking style."""
        patterns = []
        
        try:
            if len(features_list) < 5:
                return patterns
            
            # Analyze pause patterns
            pause_rates = [f.get('pause_rate', 0) for f in features_list]
            if pause_rates:
                avg_pause_rate = np.mean(pause_rates)
                if avg_pause_rate > 0.5:
                    patterns.append("Frequent pauses - reflective speaking style")
                elif avg_pause_rate < 0.1:
                    patterns.append("Minimal pauses - rapid speaking style")
            
            # Analyze speech rate consistency
            speech_rates = [f.get('words_per_minute', 0) for f in features_list if f.get('words_per_minute', 0) > 0]
            if speech_rates:
                rate_std = np.std(speech_rates)
                if rate_std > 50:
                    patterns.append("Variable speech rate - expressive speaking style")
                elif rate_std < 20:
                    patterns.append("Consistent speech rate - measured speaking style")
            
            # Analyze volume dynamics
            volume_ranges = [f.get('dynamic_range', 0) for f in features_list]
            if volume_ranges:
                avg_range = np.mean(volume_ranges)
                if avg_range > 20:  # dB range
                    patterns.append("High volume variation - dynamic speaking style")
                elif avg_range < 5:
                    patterns.append("Low volume variation - monotone speaking style")
            
        except Exception as e:
            self.logger.error(f"Style pattern detection failed: {e}")
        
        return patterns
    
    async def _analyze_accent_markers(self, features_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze accent characteristics from spectral features."""
        accent_markers = {}
        
        try:
            if len(features_list) < 5:
                return accent_markers
            
            # Analyze spectral characteristics that indicate accent
            spectral_centroids = [f.get('spectral_centroid_mean', 0) for f in features_list 
                                if f.get('spectral_centroid_mean', 0) > 0]
            
            if spectral_centroids:
                avg_centroid = np.mean(spectral_centroids)
                # Spectral centroid can indicate accent characteristics
                if avg_centroid > 2000:  # Higher frequency emphasis
                    accent_markers['high_frequency_emphasis'] = (avg_centroid - 2000) / 1000
                elif avg_centroid < 1000:  # Lower frequency emphasis
                    accent_markers['low_frequency_emphasis'] = (1000 - avg_centroid) / 1000
            
            # Analyze bandwidth characteristics
            bandwidths = [f.get('spectral_bandwidth_mean', 0) for f in features_list 
                         if f.get('spectral_bandwidth_mean', 0) > 0]
            
            if bandwidths:
                avg_bandwidth = np.mean(bandwidths)
                if avg_bandwidth > 1500:
                    accent_markers['broad_spectrum'] = (avg_bandwidth - 1500) / 1000
                elif avg_bandwidth < 800:
                    accent_markers['narrow_spectrum'] = (800 - avg_bandwidth) / 800
            
            # Analyze pitch patterns that might indicate tonal characteristics
            pitch_stds = [f.get('pitch_std', 0) for f in features_list if f.get('pitch_std', 0) > 0]
            if pitch_stds:
                avg_pitch_variation = np.mean(pitch_stds)
                if avg_pitch_variation > 50:  # High pitch variation
                    accent_markers['tonal_variation'] = min(avg_pitch_variation / 100, 1.0)
            
        except Exception as e:
            self.logger.error(f"Accent marker analysis failed: {e}")
        
        return accent_markers
    
    async def _update_adaptation_parameters(self, voice_profile: PersonalizedVoiceProfile) -> List[str]:
        """Update adaptation parameters based on learned characteristics."""
        improvements = []
        
        try:
            characteristics = voice_profile.voice_characteristics
            params = voice_profile.adaptation_parameters
            
            # Adjust confidence threshold based on voice consistency
            if characteristics.confidence_score > 0.8:
                # High confidence in voice characteristics - can be more aggressive
                old_threshold = params.minimum_confidence_for_adaptation
                params.minimum_confidence_for_adaptation = max(0.5, old_threshold - 0.05)
                if abs(params.minimum_confidence_for_adaptation - old_threshold) > 0.01:
                    improvements.append("Lowered confidence threshold due to consistent voice profile")
            
            # Adjust learning rate based on sample count
            if characteristics.sample_count > 100:
                # Lots of data - can use lower learning rate for stability
                params.online_learning_rate = max(0.005, params.online_learning_rate * 0.9)
                improvements.append("Reduced learning rate for stability with large sample size")
            
            # Adjust vocabulary boost for speech rate
            if characteristics.average_speech_rate > 200:  # Fast speaker
                params.vocabulary_boost['rapid_speech'] = 1.2
                improvements.append("Added vocabulary boost for rapid speech patterns")
            elif characteristics.average_speech_rate < 100:  # Slow speaker
                params.vocabulary_boost['careful_speech'] = 1.1
                improvements.append("Added vocabulary boost for careful speech patterns")
            
            # Adjust accent adaptation based on markers
            if len(characteristics.accent_markers) > 3:
                params.accent_adaptation_strength = min(1.5, params.accent_adaptation_strength + 0.1)
                improvements.append("Increased accent adaptation strength")
            
        except Exception as e:
            self.logger.error(f"Parameter update failed: {e}")
        
        return improvements
    
    def _calculate_learning_confidence(self, voice_profile: PersonalizedVoiceProfile, 
                                     processed_count: int) -> float:
        """Calculate confidence in learning results."""
        try:
            # Base confidence on processed samples
            sample_confidence = min(processed_count / 10.0, 1.0)
            
            # Factor in voice characteristics confidence
            char_confidence = voice_profile.voice_characteristics.confidence_score
            
            # Factor in profile maturity
            maturity_factor = min(voice_profile.voice_characteristics.sample_count / 100.0, 1.0)
            
            # Combined confidence
            combined_confidence = (
                sample_confidence * 0.4 + 
                char_confidence * 0.4 + 
                maturity_factor * 0.2
            )
            
            return min(combined_confidence, 0.95)  # Cap at 95%
            
        except Exception:
            return 0.5
    
    async def _generate_recommendations(self, voice_profile: PersonalizedVoiceProfile) -> List[str]:
        """Generate recommendations for improving voice recognition."""
        recommendations = []
        
        try:
            characteristics = voice_profile.voice_characteristics
            
            # Recommendations based on sample count
            if characteristics.sample_count < 50:
                recommendations.append("Collect more voice samples to improve personalization accuracy")
            
            # Recommendations based on consistency
            if characteristics.confidence_score < 0.6:
                recommendations.append("Voice patterns show high variability - ensure consistent microphone setup")
            
            # Recommendations based on speech rate
            if characteristics.average_speech_rate > 250:
                recommendations.append("Consider speaking slightly slower for better recognition accuracy")
            elif characteristics.average_speech_rate < 80:
                recommendations.append("Current speech rate is very slow - this is well handled by the system")
            
            # Recommendations based on pause patterns
            if len(characteristics.pause_patterns) > 0:
                avg_pause_rate = np.mean(characteristics.pause_patterns)
                if avg_pause_rate > 1.0:
                    recommendations.append("Frequent pauses detected - consider more fluid speech for better recognition")
            
            # Recommendations based on accent markers
            if len(characteristics.accent_markers) > 5:
                recommendations.append("Strong accent characteristics detected - pronunciation adaptation is active")
            
            # Recommendations based on recognition accuracy
            if voice_profile.recognition_accuracy < 0.8:
                recommendations.append("Recognition accuracy could be improved with more training data")
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
        
        return recommendations
    
    async def _get_or_create_profile(self, user_id: str) -> PersonalizedVoiceProfile:
        """Get existing profile or create new one."""
        if user_id in self.voice_profiles:
            return self.voice_profiles[user_id]
        
        # Create new profile
        profile = PersonalizedVoiceProfile(user_id=user_id)
        self.voice_profiles[user_id] = profile
        
        return profile
    
    def _dict_to_speech_data(self, data_dict: Dict[str, Any]) -> SpeechLearningData:
        """Convert dictionary to SpeechLearningData."""
        return SpeechLearningData(
            data_id=data_dict.get('data_id', str(uuid.uuid4())),
            user_id=data_dict.get('user_id', 'unknown'),
            session_id=data_dict.get('session_id'),
            audio_features=data_dict.get('audio_features'),
            audio_duration=data_dict.get('audio_duration', 0.0),
            original_text=data_dict.get('original_text', ''),
            corrected_text=data_dict.get('corrected_text'),
            confidence_scores=data_dict.get('confidence_scores', []),
            context_type=data_dict.get('context_type'),
            user_correction=data_dict.get('user_correction'),
            user_satisfaction_rating=data_dict.get('user_satisfaction_rating')
        )
    
    def _initialize_pitch_detector(self):
        """Initialize pitch detection tools."""
        # This is a placeholder for more sophisticated pitch detection
        # In a full implementation, you might use specialized libraries
        return None
    
    async def _setup_feature_pipeline(self) -> None:
        """Setup the feature extraction pipeline."""
        pass  # Implementation depends on specific audio processing requirements
    
    async def _load_voice_profiles(self) -> None:
        """Load existing voice profiles from storage."""
        # Implementation would load from persistent storage
        pass
    
    async def _save_voice_profile(self, voice_profile: PersonalizedVoiceProfile) -> bool:
        """Save voice profile to persistent storage."""
        try:
            # In a full implementation, this would save to database
            # For now, keep in memory
            self.voice_profiles[voice_profile.user_id] = voice_profile
            return True
        except Exception as e:
            self.logger.error(f"Failed to save voice profile: {e}")
            return False
    
    async def _save_model(self) -> bool:
        """Save the voice profile models."""
        try:
            # Implementation would save profiles to persistent storage
            self.logger.info(f"Saved {len(self.voice_profiles)} voice profiles")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save voice profile models: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """Load voice profile models."""
        try:
            # Implementation would load from persistent storage
            self.logger.info("Voice profile models loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load voice profile models: {e}")
            return False
    
    async def _cleanup_learner(self) -> None:
        """Cleanup learner resources."""
        self.voice_profiles.clear()
        self.recent_audio_features.clear()
        self.feature_history.clear()
        
    # Public API methods
    
    async def get_voice_profile(self, user_id: str) -> Optional[PersonalizedVoiceProfile]:
        """Get voice profile for a user."""
        return self.voice_profiles.get(user_id)
    
    async def update_user_feedback(self, user_id: str, recognition_result: str, 
                                 user_correction: Optional[str], satisfaction: Optional[int]) -> None:
        """Update profile with user feedback."""
        try:
            profile = await self._get_or_create_profile(user_id)
            
            # Update satisfaction metrics
            if satisfaction is not None:
                profile.update_performance_metrics(
                    accuracy=1.0 if user_correction is None else 0.7,
                    satisfaction=satisfaction / 5.0
                )
            
            self.logger.debug(f"Updated user feedback for {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update user feedback: {e}")
    
    def get_adaptation_parameters(self, user_id: str) -> Optional[AdaptationParameters]:
        """Get adaptation parameters for a user."""
        profile = self.voice_profiles.get(user_id)
        return profile.adaptation_parameters if profile else None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.analysis_stats,
            'active_profiles': len(self.voice_profiles),
            'total_feature_samples': sum(len(features) for features in self.recent_audio_features.values())
        }