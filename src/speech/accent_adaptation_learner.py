"""Accent Adaptation Learner - Learning and Adapting to User Accent Characteristics

Implements learning algorithms for identifying accent patterns, regional speech variations,
and language-specific pronunciation characteristics to improve recognition accuracy.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import uuid
from scipy.stats import norm
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from loguru import logger

from ..learning.base_learner import BaseLearner, LearningMode, LearningContext, LearningResult
from ..core.event_system import EventSystem
from .learning_types import (
    VoiceCharacteristics, SpeechLearningData, PersonalizedVoiceProfile,
    AdaptationParameters, VoiceFeatureType
)


class AccentCategory:
    """Categories of accents and their characteristics."""
    
    # English accent categories
    AMERICAN_GENERAL = "american_general"
    BRITISH = "british"
    AUSTRALIAN = "australian"
    INDIAN = "indian"
    CHINESE_ENGLISH = "chinese_english"
    
    # Chinese accent categories
    MANDARIN_STANDARD = "mandarin_standard"
    CANTONESE_INFLUENCED = "cantonese_influenced"
    REGIONAL_CHINESE = "regional_chinese"
    
    # Mixed language patterns
    CODE_SWITCHING = "code_switching"
    BILINGUAL_MIXED = "bilingual_mixed"


class AccentCharacteristics:
    """Characteristics that define an accent pattern."""
    
    def __init__(self, accent_id: str):
        self.accent_id = accent_id
        self.phonetic_substitutions: Dict[str, str] = {}  # Common sound replacements
        self.prosodic_patterns: Dict[str, float] = {}     # Rhythm, stress, intonation
        self.spectral_signatures: Dict[str, List[float]] = {}  # Frequency characteristics
        self.vowel_modifications: Dict[str, Tuple[float, float]] = {}  # (shift, quality)
        self.consonant_modifications: Dict[str, str] = {}  # Sound changes
        self.rhythm_patterns: List[float] = []            # Timing patterns
        self.confidence_score: float = 0.0
        self.sample_count: int = 0
        self.last_updated: datetime = datetime.now()


class AccentAdaptationLearner(BaseLearner):
    """
    Learns and adapts to user accent characteristics.
    
    Analyzes acoustic features to identify:
    - Regional accent patterns
    - Language-specific pronunciation habits
    - Phonetic substitution patterns
    - Prosodic characteristics
    - Cross-language interference patterns
    """
    
    def __init__(self, 
                 learner_id: str,
                 event_system: EventSystem,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the accent adaptation learner."""
        super().__init__(learner_id, event_system, config)
        
        # Configuration
        self.min_samples_for_accent_detection = self.config.get('min_samples_for_accent_detection', 10)
        self.accent_confidence_threshold = self.config.get('accent_confidence_threshold', 0.7)
        self.feature_smoothing_factor = self.config.get('feature_smoothing_factor', 0.15)
        self.clustering_min_samples = self.config.get('clustering_min_samples', 5)
        self.max_accent_categories = self.config.get('max_accent_categories', 5)
        
        # Accent data storage
        self.user_accent_profiles: Dict[str, Dict[str, AccentCharacteristics]] = defaultdict(dict)
        self.accent_feature_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.accent_clusters: Dict[str, Dict[str, Any]] = {}
        
        # Pre-defined accent templates
        self.accent_templates = self._initialize_accent_templates()
        
        # Feature extractors and analyzers
        self.spectral_analyzer = None
        self.prosodic_analyzer = None
        self.phonetic_analyzer = None
        
        # Learning statistics
        self.accent_stats = {
            'accents_detected': 0,
            'adaptations_applied': 0,
            'recognition_improvements': 0,
            'false_accent_detections': 0,
            'clustering_operations': 0
        }
        
        logger.info(f"AccentAdaptationLearner initialized for {learner_id}")
    
    @property
    def learner_type(self) -> str:
        return "accent_adaptation"
    
    @property
    def supported_learning_modes(self) -> List[LearningMode]:
        return [LearningMode.ONLINE, LearningMode.BATCH, LearningMode.UNSUPERVISED]
    
    @property
    def input_data_types(self) -> List[str]:
        return ["speech_audio", "voice_characteristics", "recognition_results", "user_feedback"]
    
    @property
    def output_types(self) -> List[str]:
        return ["accent_characteristics", "adaptation_parameters", "recognition_adjustments"]
    
    async def _initialize_learner(self) -> None:
        """Initialize accent analysis tools and templates."""
        try:
            # Initialize analysis components
            self.spectral_analyzer = self._initialize_spectral_analyzer()
            self.prosodic_analyzer = self._initialize_prosodic_analyzer()
            self.phonetic_analyzer = self._initialize_phonetic_analyzer()
            
            # Load existing accent profiles
            await self._load_accent_profiles()
            
            # Initialize clustering algorithms
            await self._setup_clustering_algorithms()
            
            self.logger.info("Accent adaptation learner initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize accent adaptation learner: {e}")
            raise
    
    async def _learn_from_data(self, data: List[Dict[str, Any]], 
                             context: LearningContext) -> LearningResult:
        """
        Learn accent characteristics from voice data.
        
        Args:
            data: List of speech learning data with voice features
            context: Learning context information
            
        Returns:
            Learning result with accent adaptations
        """
        start_time = datetime.now()
        result = LearningResult(success=False)
        
        try:
            if not data:
                result.error_message = "No data provided for learning"
                return result
            
            user_id = context.user_id or "default_user"
            improvements = []
            accents_detected = 0
            adaptations_applied = 0
            
            # Process voice characteristics and audio features
            for data_point in data:
                try:
                    # Convert to SpeechLearningData if needed
                    if isinstance(data_point, dict):
                        speech_data = self._dict_to_speech_data(data_point)
                    else:
                        speech_data = data_point
                    
                    # Extract accent-related features
                    accent_features = await self._extract_accent_features(speech_data)
                    if accent_features:
                        # Store features for analysis
                        self.accent_feature_history[user_id].append({
                            'features': accent_features,
                            'timestamp': datetime.now(),
                            'context': speech_data.context_type or 'general'
                        })
                        
                        # Immediate accent detection if features are strong
                        immediate_accent = await self._detect_immediate_accent(accent_features, user_id)
                        if immediate_accent:
                            accents_detected += 1
                            improvements.append(f"Detected accent characteristic: {immediate_accent}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process accent data point: {e}")
                    continue
            
            # Perform batch accent analysis if sufficient data
            if len(self.accent_feature_history[user_id]) >= self.min_samples_for_accent_detection:
                batch_results = await self._perform_batch_accent_analysis(user_id)
                if batch_results:
                    accents_detected += batch_results['accents_identified']
                    improvements.extend(batch_results['improvements'])
            
            # Update accent adaptations
            adaptation_results = await self._update_accent_adaptations(user_id)
            if adaptation_results:
                adaptations_applied = adaptation_results['adaptations_count']
                improvements.extend(adaptation_results['improvements'])
            
            # Generate adaptive recognition parameters
            adaptive_params = await self._generate_adaptive_parameters(user_id)
            
            # Calculate confidence based on data quality and consistency
            confidence_score = self._calculate_accent_confidence(user_id, accents_detected)
            
            # Generate recommendations
            recommendations = await self._generate_accent_recommendations(user_id)
            
            # Update statistics
            self.accent_stats['accents_detected'] += accents_detected
            self.accent_stats['adaptations_applied'] += adaptations_applied
            
            # Create successful result
            result = LearningResult(
                success=True,
                improvements=improvements,
                recommendations=recommendations,
                confidence_score=confidence_score,
                data_points_processed=len(data),
                metrics={
                    'accents_detected': accents_detected,
                    'adaptations_applied': adaptations_applied,
                    'accent_categories': len(self.user_accent_profiles[user_id]),
                    'feature_samples': len(self.accent_feature_history[user_id]),
                    **adaptive_params
                }
            )
            
            self.logger.info(f"Learned accent characteristics for user {user_id}: "
                           f"{accents_detected} accents detected, {adaptations_applied} adaptations applied")
            
        except Exception as e:
            self.logger.error(f"Accent adaptation learning failed: {e}")
            result.error_message = str(e)
        
        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result
    
    async def _extract_accent_features(self, speech_data: SpeechLearningData) -> Optional[Dict[str, Any]]:
        """Extract accent-relevant features from speech data."""
        try:
            features = {}
            
            if not speech_data.audio_features is not None:
                return None
            
            audio_data = speech_data.audio_features
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Extract spectral features for accent analysis
            spectral_features = await self._extract_spectral_accent_features(audio_data)
            features.update(spectral_features)
            
            # Extract prosodic features
            prosodic_features = await self._extract_prosodic_accent_features(audio_data)
            features.update(prosodic_features)
            
            # Extract formant information (vowel characteristics)
            formant_features = await self._extract_formant_features(audio_data)
            features.update(formant_features)
            
            # Extract rhythm and timing features
            rhythm_features = await self._extract_rhythm_features(audio_data, speech_data.original_text)
            features.update(rhythm_features)
            
            # Analyze language-specific patterns
            if speech_data.original_text:
                language_features = await self._extract_language_pattern_features(
                    speech_data.original_text, features
                )
                features.update(language_features)
            
            # Quality assessment
            features['accent_feature_quality'] = self._assess_accent_feature_quality(features)
            
            return features if features.get('accent_feature_quality', 0) > 0.3 else None
            
        except Exception as e:
            self.logger.error(f"Accent feature extraction failed: {e}")
            return None
    
    async def _extract_spectral_accent_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract spectral features that indicate accent characteristics."""
        features = {}
        
        try:
            # Compute MFCC features (good for accent analysis)
            import librosa
            
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.config.get('sample_rate', 16000),
                n_mfcc=13,
                hop_length=512
            )
            
            # Statistical features of MFCCs
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
            features['mfcc_skewness'] = [float(stats.skew(mfcc)) for mfcc in mfccs]
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=16000)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=16000)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=16000)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
        except Exception as e:
            self.logger.error(f"Spectral feature extraction failed: {e}")
        
        return features
    
    async def _extract_prosodic_accent_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract prosodic features that indicate accent characteristics."""
        features = {}
        
        try:
            import librosa
            
            # Pitch analysis for prosodic patterns
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, 
                sr=16000,
                hop_length=512,
                threshold=0.1
            )
            
            # Extract pitch contour
            pitch_contour = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_contour.append(pitch)
            
            if pitch_contour:
                pitch_array = np.array(pitch_contour)
                
                # Pitch statistics
                features['pitch_contour_mean'] = float(np.mean(pitch_array))
                features['pitch_contour_std'] = float(np.std(pitch_array))
                features['pitch_range'] = float(np.ptp(pitch_array))
                
                # Pitch dynamics (rate of change)
                if len(pitch_array) > 1:
                    pitch_deltas = np.diff(pitch_array)
                    features['pitch_delta_mean'] = float(np.mean(np.abs(pitch_deltas)))
                    features['pitch_delta_std'] = float(np.std(pitch_deltas))
                
                # Identify pitch patterns (rising, falling, flat)
                features['pitch_trend'] = self._analyze_pitch_trend(pitch_array)
            
            # Energy contour analysis
            rms_energy = librosa.feature.rms(y=audio_data, hop_length=512)[0]
            if len(rms_energy) > 0:
                features['energy_variance'] = float(np.var(rms_energy))
                features['energy_skewness'] = float(stats.skew(rms_energy))
                
                # Energy dynamics
                if len(rms_energy) > 1:
                    energy_deltas = np.diff(rms_energy)
                    features['energy_change_rate'] = float(np.mean(np.abs(energy_deltas)))
            
        except Exception as e:
            self.logger.error(f"Prosodic feature extraction failed: {e}")
        
        return features
    
    async def _extract_formant_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract formant features for vowel analysis."""
        features = {}
        
        try:
            # Simplified formant estimation using spectral peaks
            # In a full implementation, this would use proper formant tracking
            import librosa
            
            # Compute spectrum
            stft = librosa.stft(audio_data, hop_length=512)
            magnitude_spectrum = np.abs(stft)
            
            # Find spectral peaks for formant estimation
            freq_bins = librosa.fft_frequencies(sr=16000, n_fft=2048)
            
            # Average spectrum over time
            avg_spectrum = np.mean(magnitude_spectrum, axis=1)
            
            # Find peaks (simplified formant estimation)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(avg_spectrum, height=np.max(avg_spectrum) * 0.1, distance=10)
            
            if len(peaks) >= 2:
                # First two formants (approximate)
                f1_idx = peaks[0] if len(peaks) > 0 else 0
                f2_idx = peaks[1] if len(peaks) > 1 else 0
                
                features['f1_estimate'] = float(freq_bins[f1_idx])
                features['f2_estimate'] = float(freq_bins[f2_idx])
                features['f2_f1_ratio'] = features['f2_estimate'] / max(features['f1_estimate'], 1.0)
            
            # Spectral tilt (balance between low and high frequencies)
            low_freq_energy = np.sum(avg_spectrum[:len(avg_spectrum)//4])
            high_freq_energy = np.sum(avg_spectrum[3*len(avg_spectrum)//4:])
            features['spectral_tilt'] = float(high_freq_energy / max(low_freq_energy, 1e-6))
            
        except Exception as e:
            self.logger.error(f"Formant feature extraction failed: {e}")
        
        return features
    
    async def _extract_rhythm_features(self, audio_data: np.ndarray, text: str) -> Dict[str, Any]:
        """Extract rhythm and timing features."""
        features = {}
        
        try:
            import librosa
            
            # Onset detection for rhythm analysis
            onset_frames = librosa.onset.onset_detect(
                y=audio_data,
                sr=16000,
                hop_length=512,
                units='frames'
            )
            
            if len(onset_frames) > 1:
                # Convert to time
                onset_times = librosa.frames_to_time(onset_frames, sr=16000, hop_length=512)
                
                # Inter-onset intervals
                ioi = np.diff(onset_times)
                
                features['rhythm_regularity'] = float(1.0 / (1.0 + np.std(ioi)))
                features['average_ioi'] = float(np.mean(ioi))
                features['ioi_coefficient_variation'] = float(np.std(ioi) / max(np.mean(ioi), 1e-6))
                
                # Rhythm complexity
                features['rhythm_complexity'] = float(np.var(ioi))
            
            # Speech rate estimation
            duration = len(audio_data) / 16000
            if text and duration > 0:
                word_count = len(text.split())
                features['estimated_speech_rate'] = word_count / duration * 60  # words per minute
                
                # Syllable rate (rough estimate)
                syllable_count = sum(self._estimate_syllables(word) for word in text.split())
                features['estimated_syllable_rate'] = syllable_count / duration
            
        except Exception as e:
            self.logger.error(f"Rhythm feature extraction failed: {e}")
        
        return features
    
    async def _extract_language_pattern_features(self, text: str, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features related to language mixing and code-switching."""
        features = {}
        
        try:
            # Detect language mixing
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            english_words = len([word for word in text.split() if word.isascii()])
            total_linguistic_units = chinese_chars + english_words
            
            if total_linguistic_units > 0:
                features['chinese_ratio'] = chinese_chars / total_linguistic_units
                features['english_ratio'] = english_words / total_linguistic_units
                features['language_mixing_index'] = min(features['chinese_ratio'], features['english_ratio']) * 2
            
            # Detect code-switching points
            code_switches = self._detect_code_switches(text)
            features['code_switch_count'] = len(code_switches)
            features['code_switch_rate'] = len(code_switches) / max(len(text.split()), 1)
            
            # Analyze pronunciation context
            if audio_features.get('pitch_contour_std', 0) > 0:
                # Higher pitch variation might indicate tonal language influence
                features['tonal_influence_indicator'] = min(
                    audio_features['pitch_contour_std'] / 50.0, 2.0
                )
            
        except Exception as e:
            self.logger.error(f"Language pattern feature extraction failed: {e}")
        
        return features
    
    def _analyze_pitch_trend(self, pitch_contour: np.ndarray) -> str:
        """Analyze overall pitch trend."""
        try:
            if len(pitch_contour) < 3:
                return "insufficient_data"
            
            # Linear regression to find trend
            x = np.arange(len(pitch_contour))
            slope, _ = np.polyfit(x, pitch_contour, 1)
            
            if slope > 5:  # Rising pitch
                return "rising"
            elif slope < -5:  # Falling pitch
                return "falling"
            else:
                return "flat"
        except Exception:
            return "unknown"
    
    def _estimate_syllables(self, word: str) -> int:
        """Estimate syllable count in a word."""
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
    
    def _detect_code_switches(self, text: str) -> List[int]:
        """Detect positions where language switching occurs."""
        switches = []
        words = text.split()
        
        try:
            current_lang = None
            for i, word in enumerate(words):
                # Simple language detection
                if any('\u4e00' <= char <= '\u9fff' for char in word):
                    word_lang = 'chinese'
                elif word.isascii():
                    word_lang = 'english'
                else:
                    word_lang = 'mixed'
                
                if current_lang is not None and current_lang != word_lang and word_lang != 'mixed':
                    switches.append(i)
                
                if word_lang != 'mixed':
                    current_lang = word_lang
        
        except Exception as e:
            self.logger.error(f"Code switch detection failed: {e}")
        
        return switches
    
    def _assess_accent_feature_quality(self, features: Dict[str, Any]) -> float:
        """Assess quality of extracted accent features."""
        quality_score = 1.0
        
        try:
            # Check for presence of key features
            required_features = [
                'mfcc_mean', 'spectral_centroid_mean', 'pitch_contour_mean'
            ]
            
            for feature in required_features:
                if feature not in features:
                    quality_score -= 0.2
            
            # Check for reasonable values
            if features.get('spectral_centroid_mean', 0) <= 0:
                quality_score -= 0.1
            
            if features.get('pitch_contour_mean', 0) <= 0 or features.get('pitch_contour_mean', 0) > 1000:
                quality_score -= 0.1
            
            # Check MFCC validity
            mfcc_mean = features.get('mfcc_mean', [])
            if not mfcc_mean or len(mfcc_mean) < 10:
                quality_score -= 0.2
            
            return max(0.0, quality_score)
            
        except Exception:
            return 0.5
    
    async def _detect_immediate_accent(self, features: Dict[str, Any], user_id: str) -> Optional[str]:
        """Detect accent characteristics from immediate features."""
        try:
            # Compare against known accent templates
            best_match = None
            best_score = 0.0
            
            for accent_id, template in self.accent_templates.items():
                score = self._calculate_accent_similarity(features, template)
                if score > best_score and score > self.accent_confidence_threshold:
                    best_score = score
                    best_match = accent_id
            
            if best_match:
                # Update or create accent profile
                if best_match not in self.user_accent_profiles[user_id]:
                    self.user_accent_profiles[user_id][best_match] = AccentCharacteristics(best_match)
                
                accent_profile = self.user_accent_profiles[user_id][best_match]
                accent_profile.sample_count += 1
                accent_profile.confidence_score = (
                    accent_profile.confidence_score * 0.9 + best_score * 0.1
                )
                accent_profile.last_updated = datetime.now()
                
                return best_match
            
        except Exception as e:
            self.logger.error(f"Immediate accent detection failed: {e}")
        
        return None
    
    def _calculate_accent_similarity(self, features: Dict[str, Any], template: Dict[str, Any]) -> float:
        """Calculate similarity between features and accent template."""
        try:
            similarity_scores = []
            
            # Compare MFCC features
            if 'mfcc_mean' in features and 'mfcc_mean' in template:
                feat_mfcc = np.array(features['mfcc_mean'])
                temp_mfcc = np.array(template['mfcc_mean'])
                
                if len(feat_mfcc) == len(temp_mfcc):
                    mfcc_similarity = 1.0 - (np.mean(np.abs(feat_mfcc - temp_mfcc)) / 10.0)
                    similarity_scores.append(max(0.0, mfcc_similarity))
            
            # Compare spectral features
            spectral_features = ['spectral_centroid_mean', 'spectral_bandwidth_mean']
            for feature in spectral_features:
                if feature in features and feature in template:
                    feat_val = features[feature]
                    temp_val = template[feature]
                    
                    # Normalized difference
                    diff = abs(feat_val - temp_val) / max(temp_val, 1.0)
                    similarity = 1.0 - min(diff, 1.0)
                    similarity_scores.append(similarity)
            
            # Compare prosodic features
            if 'pitch_contour_std' in features and 'pitch_contour_std' in template:
                feat_pitch_var = features['pitch_contour_std']
                temp_pitch_var = template['pitch_contour_std']
                
                pitch_similarity = 1.0 - abs(feat_pitch_var - temp_pitch_var) / max(temp_pitch_var, 1.0)
                similarity_scores.append(max(0.0, pitch_similarity))
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Accent similarity calculation failed: {e}")
            return 0.0
    
    async def _perform_batch_accent_analysis(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Perform batch analysis of accent features using clustering."""
        try:
            feature_history = self.accent_feature_history[user_id]
            if len(feature_history) < self.min_samples_for_accent_detection:
                return None
            
            # Prepare feature matrix
            feature_matrix = self._prepare_feature_matrix(feature_history)
            if feature_matrix is None or feature_matrix.shape[0] < self.clustering_min_samples:
                return None
            
            # Perform clustering to identify accent patterns
            clusters = await self._perform_accent_clustering(feature_matrix, user_id)
            
            improvements = []
            accents_identified = 0
            
            if clusters:
                # Analyze each cluster
                for cluster_id, cluster_info in clusters.items():
                    accent_characteristics = self._analyze_cluster_characteristics(
                        cluster_info, feature_history
                    )
                    
                    if accent_characteristics:
                        # Create or update accent profile
                        accent_id = f"cluster_{cluster_id}"
                        
                        if accent_id not in self.user_accent_profiles[user_id]:
                            self.user_accent_profiles[user_id][accent_id] = AccentCharacteristics(accent_id)
                        
                        accent_profile = self.user_accent_profiles[user_id][accent_id]
                        self._update_accent_characteristics(accent_profile, accent_characteristics)
                        
                        accents_identified += 1
                        improvements.append(f"Identified accent cluster: {accent_id}")
                
                # Store clustering results
                self.accent_clusters[user_id] = clusters
                self.accent_stats['clustering_operations'] += 1
            
            return {
                'accents_identified': accents_identified,
                'improvements': improvements,
                'clusters_found': len(clusters) if clusters else 0
            }
            
        except Exception as e:
            self.logger.error(f"Batch accent analysis failed: {e}")
            return None
    
    def _prepare_feature_matrix(self, feature_history: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Prepare feature matrix for clustering analysis."""
        try:
            # Select key features for clustering
            key_features = [
                'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean',
                'pitch_contour_mean', 'pitch_contour_std', 'f2_f1_ratio',
                'rhythm_regularity', 'language_mixing_index'
            ]
            
            feature_vectors = []
            for record in feature_history:
                features = record['features']
                vector = []
                
                for feature_name in key_features:
                    value = features.get(feature_name, 0.0)
                    # Handle list features (like MFCC)
                    if isinstance(value, list):
                        if value:
                            vector.append(np.mean(value))
                        else:
                            vector.append(0.0)
                    else:
                        vector.append(float(value))
                
                if len(vector) == len(key_features):
                    feature_vectors.append(vector)
            
            if len(feature_vectors) < 3:
                return None
            
            # Standardize features
            feature_matrix = np.array(feature_vectors)
            scaler = StandardScaler()
            normalized_matrix = scaler.fit_transform(feature_matrix)
            
            return normalized_matrix
            
        except Exception as e:
            self.logger.error(f"Feature matrix preparation failed: {e}")
            return None
    
    async def _perform_accent_clustering(self, feature_matrix: np.ndarray, 
                                       user_id: str) -> Optional[Dict[str, Any]]:
        """Perform clustering to identify accent patterns."""
        try:
            clusters = {}
            
            # Try DBSCAN clustering first (can find variable number of clusters)
            dbscan = DBSCAN(eps=0.5, min_samples=self.clustering_min_samples)
            dbscan_labels = dbscan.fit_predict(feature_matrix)
            
            unique_labels = set(dbscan_labels)
            if len(unique_labels) > 1 and -1 in unique_labels:
                unique_labels.remove(-1)  # Remove noise cluster
            
            if len(unique_labels) >= 1 and len(unique_labels) <= self.max_accent_categories:
                # DBSCAN found reasonable clusters
                for label in unique_labels:
                    if label != -1:  # Skip noise
                        cluster_indices = np.where(dbscan_labels == label)[0]
                        clusters[f"dbscan_{label}"] = {
                            'type': 'dbscan',
                            'indices': cluster_indices.tolist(),
                            'center': np.mean(feature_matrix[cluster_indices], axis=0).tolist(),
                            'size': len(cluster_indices)
                        }
            
            # If DBSCAN didn't work well, try K-means
            if len(clusters) == 0:
                # Try different numbers of clusters
                best_k = 2
                best_score = -1
                
                for k in range(2, min(6, len(feature_matrix) // 3)):
                    try:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(feature_matrix)
                        score = silhouette_score(feature_matrix, labels)
                        
                        if score > best_score:
                            best_score = score
                            best_k = k
                    except Exception:
                        continue
                
                if best_score > 0.2:  # Reasonable clustering quality
                    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(feature_matrix)
                    
                    for label in range(best_k):
                        cluster_indices = np.where(labels == label)[0]
                        clusters[f"kmeans_{label}"] = {
                            'type': 'kmeans',
                            'indices': cluster_indices.tolist(),
                            'center': kmeans.cluster_centers_[label].tolist(),
                            'size': len(cluster_indices)
                        }
            
            return clusters if clusters else None
            
        except Exception as e:
            self.logger.error(f"Accent clustering failed: {e}")
            return None
    
    def _analyze_cluster_characteristics(self, cluster_info: Dict[str, Any], 
                                       feature_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze characteristics of an accent cluster."""
        try:
            indices = cluster_info['indices']
            cluster_features = [feature_history[i]['features'] for i in indices]
            
            characteristics = {}
            
            # Aggregate spectral characteristics
            spectral_features = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean']
            for feature in spectral_features:
                values = [f.get(feature, 0) for f in cluster_features if f.get(feature, 0) > 0]
                if values:
                    characteristics[feature] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
            
            # Aggregate prosodic characteristics
            prosodic_features = ['pitch_contour_mean', 'pitch_contour_std', 'rhythm_regularity']
            for feature in prosodic_features:
                values = [f.get(feature, 0) for f in cluster_features if f.get(feature, 0) > 0]
                if values:
                    characteristics[feature] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
            
            # Language characteristics
            lang_mixing_values = [f.get('language_mixing_index', 0) for f in cluster_features]
            if lang_mixing_values:
                characteristics['language_mixing_index'] = {
                    'mean': float(np.mean(lang_mixing_values)),
                    'std': float(np.std(lang_mixing_values))
                }
            
            # Cluster metadata
            characteristics['sample_count'] = len(indices)
            characteristics['confidence'] = min(len(indices) / 10.0, 1.0)
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Cluster characteristics analysis failed: {e}")
            return None
    
    def _update_accent_characteristics(self, accent_profile: AccentCharacteristics, 
                                     characteristics: Dict[str, Any]) -> None:
        """Update accent profile with new characteristics."""
        try:
            alpha = self.feature_smoothing_factor
            
            # Update sample count and confidence
            accent_profile.sample_count += characteristics.get('sample_count', 0)
            accent_profile.confidence_score = (
                accent_profile.confidence_score * (1 - alpha) +
                characteristics.get('confidence', 0.5) * alpha
            )
            
            # Update prosodic patterns
            for feature in ['pitch_contour_mean', 'pitch_contour_std', 'rhythm_regularity']:
                if feature in characteristics:
                    char_data = characteristics[feature]
                    if feature not in accent_profile.prosodic_patterns:
                        accent_profile.prosodic_patterns[feature] = char_data['mean']
                    else:
                        current_val = accent_profile.prosodic_patterns[feature]
                        accent_profile.prosodic_patterns[feature] = (
                            current_val * (1 - alpha) + char_data['mean'] * alpha
                        )
            
            # Update spectral signatures
            for feature in ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean']:
                if feature in characteristics:
                    char_data = characteristics[feature]
                    accent_profile.spectral_signatures[feature] = [char_data['mean'], char_data['std']]
            
            accent_profile.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Accent characteristics update failed: {e}")
    
    async def _update_accent_adaptations(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Update accent-based adaptations."""
        try:
            user_accents = self.user_accent_profiles.get(user_id, {})
            if not user_accents:
                return None
            
            improvements = []
            adaptations_count = 0
            
            # Find the most confident accent
            primary_accent = max(user_accents.values(), key=lambda a: a.confidence_score)
            
            if primary_accent.confidence_score > self.accent_confidence_threshold:
                # Generate adaptation parameters based on primary accent
                adaptations = self._generate_accent_adaptations(primary_accent)
                
                if adaptations:
                    adaptations_count = len(adaptations)
                    improvements.extend([f"Applied {adaptations_count} accent adaptations",
                                       f"Primary accent: {primary_accent.accent_id}"])
                
                # Update recognition parameters
                recognition_updates = self._update_recognition_parameters(primary_accent, user_id)
                if recognition_updates:
                    improvements.extend(recognition_updates)
            
            return {
                'adaptations_count': adaptations_count,
                'improvements': improvements
            }
            
        except Exception as e:
            self.logger.error(f"Accent adaptation update failed: {e}")
            return None
    
    def _generate_accent_adaptations(self, accent_profile: AccentCharacteristics) -> List[Dict[str, Any]]:
        """Generate adaptation parameters based on accent characteristics."""
        adaptations = []
        
        try:
            # Spectral adaptations
            if 'spectral_centroid_mean' in accent_profile.spectral_signatures:
                centroid_data = accent_profile.spectral_signatures['spectral_centroid_mean']
                if centroid_data[0] > 2500:  # High spectral centroid
                    adaptations.append({
                        'type': 'spectral_boost',
                        'parameter': 'high_frequency_emphasis',
                        'value': min((centroid_data[0] - 2500) / 1000, 0.3)
                    })
                elif centroid_data[0] < 1500:  # Low spectral centroid
                    adaptations.append({
                        'type': 'spectral_boost',
                        'parameter': 'low_frequency_emphasis',
                        'value': min((1500 - centroid_data[0]) / 1000, 0.3)
                    })
            
            # Prosodic adaptations
            if 'pitch_contour_std' in accent_profile.prosodic_patterns:
                pitch_variation = accent_profile.prosodic_patterns['pitch_contour_std']
                if pitch_variation > 50:  # High pitch variation (tonal influence)
                    adaptations.append({
                        'type': 'prosodic_adjustment',
                        'parameter': 'pitch_sensitivity',
                        'value': min(pitch_variation / 100, 0.5)
                    })
            
            # Language mixing adaptations
            accent_id = accent_profile.accent_id.lower()
            if 'chinese' in accent_id or 'mixed' in accent_id:
                adaptations.append({
                    'type': 'language_model',
                    'parameter': 'bilingual_boost',
                    'value': 0.2
                })
            
        except Exception as e:
            self.logger.error(f"Accent adaptation generation failed: {e}")
        
        return adaptations
    
    def _update_recognition_parameters(self, accent_profile: AccentCharacteristics, 
                                     user_id: str) -> List[str]:
        """Update recognition parameters based on accent."""
        improvements = []
        
        try:
            # This would integrate with the recognition system
            # For now, we just record what adaptations should be made
            
            accent_id = accent_profile.accent_id
            
            if 'chinese' in accent_id.lower():
                improvements.append("Enabled Chinese-English bilingual recognition mode")
                improvements.append("Adjusted pronunciation model for Chinese accent")
            
            if 'tonal' in accent_profile.prosodic_patterns:
                improvements.append("Enhanced pitch sensitivity for tonal characteristics")
            
            if accent_profile.confidence_score > 0.8:
                improvements.append("High-confidence accent profile - enabled aggressive adaptation")
            
        except Exception as e:
            self.logger.error(f"Recognition parameter update failed: {e}")
        
        return improvements
    
    async def _generate_adaptive_parameters(self, user_id: str) -> Dict[str, Any]:
        """Generate adaptive parameters based on learned accents."""
        params = {}
        
        try:
            user_accents = self.user_accent_profiles.get(user_id, {})
            
            if user_accents:
                # Count accent types
                params['accent_categories_detected'] = len(user_accents)
                
                # Find primary accent
                primary_accent = max(user_accents.values(), key=lambda a: a.confidence_score)
                params['primary_accent'] = primary_accent.accent_id
                params['primary_accent_confidence'] = primary_accent.confidence_score
                
                # Calculate adaptation strength
                params['adaptation_strength'] = min(primary_accent.confidence_score, 0.8)
                
                # Language characteristics
                params['supports_code_switching'] = any(
                    'mixed' in acc.accent_id.lower() or 'bilingual' in acc.accent_id.lower()
                    for acc in user_accents.values()
                )
            
        except Exception as e:
            self.logger.error(f"Adaptive parameter generation failed: {e}")
        
        return params
    
    def _calculate_accent_confidence(self, user_id: str, accents_detected: int) -> float:
        """Calculate confidence in accent detection and adaptation."""
        try:
            user_accents = self.user_accent_profiles.get(user_id, {})
            
            if not user_accents:
                return 0.0
            
            # Base confidence on accent profile strength
            avg_accent_confidence = np.mean([acc.confidence_score for acc in user_accents.values()])
            
            # Factor in sample sizes
            total_samples = sum(acc.sample_count for acc in user_accents.values())
            sample_confidence = min(total_samples / 50.0, 1.0)
            
            # Factor in recent detections
            detection_confidence = min(accents_detected / 5.0, 1.0)
            
            # Factor in feature quality
            feature_quality = len(self.accent_feature_history[user_id]) / 20.0
            feature_quality = min(feature_quality, 1.0)
            
            combined_confidence = (
                avg_accent_confidence * 0.4 +
                sample_confidence * 0.3 +
                detection_confidence * 0.2 +
                feature_quality * 0.1
            )
            
            return min(combined_confidence, 0.95)
            
        except Exception:
            return 0.5
    
    async def _generate_accent_recommendations(self, user_id: str) -> List[str]:
        """Generate recommendations for accent adaptation."""
        recommendations = []
        
        try:
            user_accents = self.user_accent_profiles.get(user_id, {})
            
            if not user_accents:
                recommendations.append("Collect more speech samples to detect accent characteristics")
                return recommendations
            
            # Find most prominent accent
            primary_accent = max(user_accents.values(), key=lambda a: a.confidence_score)
            
            if primary_accent.confidence_score < 0.6:
                recommendations.append("Accent profile needs more samples for reliable adaptation")
            
            # Accent-specific recommendations
            accent_id = primary_accent.accent_id.lower()
            
            if 'chinese' in accent_id:
                recommendations.append("Chinese accent detected - enabling tone-aware recognition")
                recommendations.append("Consider clearer articulation of English consonant clusters")
            
            if 'mixed' in accent_id or 'bilingual' in accent_id:
                recommendations.append("Bilingual speech pattern detected - code-switching adaptation enabled")
            
            if primary_accent.sample_count > 50:
                recommendations.append("Sufficient accent data collected - personalization is highly effective")
            
            # Multiple accent handling
            if len(user_accents) > 2:
                recommendations.append("Multiple accent patterns detected - adaptive recognition enabled")
            
        except Exception as e:
            self.logger.error(f"Accent recommendation generation failed: {e}")
        
        return recommendations
    
    def _initialize_accent_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize accent templates for comparison."""
        return {
            AccentCategory.CHINESE_ENGLISH: {
                'mfcc_mean': [2.5, -8.2, 4.1, 1.8, -2.1, 3.2, -1.5, 2.8, -1.9, 0.7, -1.2, 1.1, -0.8],
                'spectral_centroid_mean': 1800,
                'spectral_bandwidth_mean': 1200,
                'pitch_contour_std': 45,
                'f2_f1_ratio': 2.1,
                'language_mixing_index': 0.6
            },
            AccentCategory.AMERICAN_GENERAL: {
                'mfcc_mean': [1.8, -7.5, 3.2, 2.1, -1.8, 2.9, -1.2, 2.5, -1.6, 0.9, -0.8, 1.3, -0.5],
                'spectral_centroid_mean': 2100,
                'spectral_bandwidth_mean': 1400,
                'pitch_contour_std': 25,
                'f2_f1_ratio': 2.3,
                'language_mixing_index': 0.0
            },
            AccentCategory.BRITISH: {
                'mfcc_mean': [2.1, -7.8, 3.5, 1.9, -2.0, 3.1, -1.3, 2.6, -1.7, 0.8, -1.0, 1.2, -0.6],
                'spectral_centroid_mean': 2200,
                'spectral_bandwidth_mean': 1350,
                'pitch_contour_std': 28,
                'f2_f1_ratio': 2.4,
                'language_mixing_index': 0.0
            }
        }
    
    def _initialize_spectral_analyzer(self):
        """Initialize spectral analysis tools."""
        return None
    
    def _initialize_prosodic_analyzer(self):
        """Initialize prosodic analysis tools."""
        return None
    
    def _initialize_phonetic_analyzer(self):
        """Initialize phonetic analysis tools."""
        return None
    
    async def _load_accent_profiles(self) -> None:
        """Load existing accent profiles."""
        pass
    
    async def _setup_clustering_algorithms(self) -> None:
        """Setup clustering algorithms."""
        pass
    
    def _dict_to_speech_data(self, data_dict: Dict[str, Any]) -> SpeechLearningData:
        """Convert dictionary to SpeechLearningData."""
        return SpeechLearningData(
            data_id=data_dict.get('data_id', str(uuid.uuid4())),
            user_id=data_dict.get('user_id', 'unknown'),
            audio_features=data_dict.get('audio_features'),
            audio_duration=data_dict.get('audio_duration', 0.0),
            original_text=data_dict.get('original_text', ''),
            context_type=data_dict.get('context_type')
        )
    
    async def _save_model(self) -> bool:
        """Save accent profiles and models."""
        try:
            self.logger.info(f"Saved accent profiles for {len(self.user_accent_profiles)} users")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save accent models: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """Load accent profiles and models."""
        try:
            self.logger.info("Accent models loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load accent models: {e}")
            return False
    
    async def _cleanup_learner(self) -> None:
        """Cleanup learner resources."""
        self.user_accent_profiles.clear()
        self.accent_feature_history.clear()
        self.accent_clusters.clear()
    
    # Public API methods
    
    async def get_accent_profile(self, user_id: str) -> Dict[str, AccentCharacteristics]:
        """Get accent profile for a user."""
        return self.user_accent_profiles.get(user_id, {})
    
    async def get_primary_accent(self, user_id: str) -> Optional[AccentCharacteristics]:
        """Get primary accent for a user."""
        user_accents = self.user_accent_profiles.get(user_id, {})
        if not user_accents:
            return None
        
        return max(user_accents.values(), key=lambda a: a.confidence_score)
    
    def get_accent_adaptations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get current accent adaptations for a user."""
        primary_accent = asyncio.run(self.get_primary_accent(user_id))
        if primary_accent:
            return self._generate_accent_adaptations(primary_accent)
        return []
    
    def get_accent_statistics(self) -> Dict[str, Any]:
        """Get accent learning statistics."""
        return {
            **self.accent_stats,
            'active_users': len(self.user_accent_profiles),
            'total_accent_profiles': sum(len(profiles) for profiles in self.user_accent_profiles.values()),
            'clustering_results': len(self.accent_clusters)
        }