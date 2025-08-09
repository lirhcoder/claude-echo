"""Pronunciation Pattern Learner - Learning Common Pronunciation Errors

Implements learning algorithms for identifying and correcting common pronunciation 
errors, misrecognized words, and speech patterns that consistently cause recognition issues.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import difflib
import uuid
import numpy as np

from loguru import logger
from phonemizer import phonemize
from jiwer import wer, mer, wil, wip

from ..learning.base_learner import BaseLearner, LearningMode, LearningContext, LearningResult
from ..core.event_system import EventSystem
from .learning_types import (
    PronunciationPattern, SpeechLearningData, PersonalizedVoiceProfile,
    AdaptationParameters
)


class PronunciationPatternLearner(BaseLearner):
    """
    Learns pronunciation patterns and common speech recognition errors.
    
    Identifies and learns from:
    - Consistently misrecognized words
    - Phonetic variations in pronunciation
    - Context-dependent pronunciation errors
    - Language mixing patterns (Chinese-English)
    - User-specific pronunciation habits
    """
    
    def __init__(self, 
                 learner_id: str,
                 event_system: EventSystem,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the pronunciation pattern learner."""
        super().__init__(learner_id, event_system, config)
        
        # Configuration
        self.min_error_frequency = self.config.get('min_error_frequency', 3)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.phonetic_distance_threshold = self.config.get('phonetic_distance_threshold', 0.7)
        self.correction_confidence_threshold = self.config.get('correction_confidence_threshold', 0.6)
        self.max_patterns_per_user = self.config.get('max_patterns_per_user', 500)
        
        # Pattern storage
        self.pronunciation_patterns: Dict[str, Dict[str, PronunciationPattern]] = defaultdict(dict)
        self.user_corrections: Dict[str, List[Tuple[str, str, datetime]]] = defaultdict(list)
        self.phonetic_mappings: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Analysis components
        self.phonetic_analyzer = None
        self.word_similarity_analyzer = None
        
        # Learning statistics
        self.learning_stats = {
            'total_patterns_learned': 0,
            'corrections_processed': 0,
            'successful_corrections': 0,
            'phonetic_mappings_created': 0,
            'pattern_applications': 0
        }
        
        # Common programming terms and their phonetic variations
        self.programming_phonetics = self._initialize_programming_phonetics()
        
        # Chinese-English code-switching patterns
        self.code_switching_patterns = self._initialize_code_switching_patterns()
        
        logger.info(f"PronunciationPatternLearner initialized for {learner_id}")
    
    @property
    def learner_type(self) -> str:
        return "pronunciation_pattern"
    
    @property
    def supported_learning_modes(self) -> List[LearningMode]:
        return [LearningMode.ONLINE, LearningMode.SUPERVISED, LearningMode.REINFORCEMENT]
    
    @property
    def input_data_types(self) -> List[str]:
        return ["speech_corrections", "recognition_errors", "user_feedback", "phonetic_data"]
    
    @property
    def output_types(self) -> List[str]:
        return ["pronunciation_patterns", "correction_suggestions", "phonetic_mappings"]
    
    async def _initialize_learner(self) -> None:
        """Initialize pronunciation analysis tools."""
        try:
            # Initialize phonetic analyzer
            self.phonetic_analyzer = self._initialize_phonetic_analyzer()
            
            # Initialize word similarity analyzer
            self.word_similarity_analyzer = self._initialize_similarity_analyzer()
            
            # Load existing patterns
            await self._load_pronunciation_patterns()
            
            # Setup pattern matching algorithms
            await self._setup_pattern_matching()
            
            self.logger.info("Pronunciation pattern learner initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pronunciation pattern learner: {e}")
            raise
    
    async def _learn_from_data(self, data: List[Dict[str, Any]], 
                             context: LearningContext) -> LearningResult:
        """
        Learn pronunciation patterns from speech corrections and errors.
        
        Args:
            data: List of speech learning data with corrections
            context: Learning context information
            
        Returns:
            Learning result with learned patterns
        """
        start_time = datetime.now()
        result = LearningResult(success=False)
        
        try:
            if not data:
                result.error_message = "No data provided for learning"
                return result
            
            user_id = context.user_id or "default_user"
            improvements = []
            patterns_learned = 0
            corrections_processed = 0
            
            # Process each data point
            for data_point in data:
                try:
                    # Convert to SpeechLearningData if needed
                    if isinstance(data_point, dict):
                        speech_data = self._dict_to_speech_data(data_point)
                    else:
                        speech_data = data_point
                    
                    # Process corrections and errors
                    if speech_data.user_correction or speech_data.corrected_text:
                        correction_result = await self._process_correction(speech_data, user_id)
                        if correction_result:
                            patterns_learned += correction_result['patterns_created']
                            corrections_processed += 1
                            improvements.extend(correction_result['improvements'])
                    
                    # Analyze recognition errors
                    error_patterns = await self._analyze_recognition_errors(speech_data, user_id)
                    if error_patterns:
                        patterns_learned += len(error_patterns)
                        improvements.extend([f"Identified pronunciation pattern: {p.target_word}" 
                                           for p in error_patterns])
                    
                    # Learn from context patterns
                    context_patterns = await self._learn_contextual_patterns(speech_data, user_id)
                    if context_patterns:
                        improvements.extend(context_patterns)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process data point: {e}")
                    continue
            
            # Perform pattern consolidation
            consolidation_result = await self._consolidate_patterns(user_id)
            if consolidation_result:
                improvements.extend(consolidation_result['improvements'])
                patterns_learned += consolidation_result['patterns_consolidated']
            
            # Update phonetic mappings
            phonetic_updates = await self._update_phonetic_mappings(user_id)
            if phonetic_updates:
                improvements.extend(phonetic_updates)
            
            # Generate correction suggestions
            suggestions = await self._generate_correction_suggestions(user_id)
            
            # Calculate confidence based on pattern strength and frequency
            confidence_score = self._calculate_pattern_confidence(user_id, patterns_learned)
            
            # Update statistics
            self.learning_stats['total_patterns_learned'] += patterns_learned
            self.learning_stats['corrections_processed'] += corrections_processed
            
            # Create successful result
            result = LearningResult(
                success=True,
                improvements=improvements,
                recommendations=suggestions,
                confidence_score=confidence_score,
                data_points_processed=corrections_processed,
                metrics={
                    'patterns_learned': patterns_learned,
                    'user_patterns_total': len(self.pronunciation_patterns.get(user_id, {})),
                    'corrections_processed': corrections_processed,
                    'phonetic_mappings': len(self.phonetic_mappings.get(user_id, {}))
                }
            )
            
            self.logger.info(f"Learned {patterns_learned} pronunciation patterns for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Pronunciation pattern learning failed: {e}")
            result.error_message = str(e)
        
        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result
    
    async def _process_correction(self, speech_data: SpeechLearningData, 
                                user_id: str) -> Optional[Dict[str, Any]]:
        """Process a user correction to learn pronunciation patterns."""
        try:
            original_text = speech_data.original_text.strip()
            corrected_text = (speech_data.user_correction or speech_data.corrected_text or "").strip()
            
            if not original_text or not corrected_text or original_text == corrected_text:
                return None
            
            # Record the correction
            self.user_corrections[user_id].append((original_text, corrected_text, datetime.now()))
            
            # Analyze word-level differences
            word_patterns = await self._analyze_word_differences(original_text, corrected_text, speech_data)
            patterns_created = 0
            improvements = []
            
            for pattern in word_patterns:
                pattern.user_id = user_id
                
                # Check if pattern already exists
                existing_pattern = self.pronunciation_patterns[user_id].get(pattern.pattern_id)
                if existing_pattern:
                    # Update existing pattern
                    existing_pattern.update_occurrence()
                    existing_pattern.calculate_priority()
                    improvements.append(f"Updated pronunciation pattern: {pattern.target_word}")
                else:
                    # Create new pattern
                    self.pronunciation_patterns[user_id][pattern.pattern_id] = pattern
                    patterns_created += 1
                    improvements.append(f"Learned new pronunciation pattern: {pattern.target_word}")
            
            return {
                'patterns_created': patterns_created,
                'improvements': improvements,
                'word_patterns_analyzed': len(word_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process correction: {e}")
            return None
    
    async def _analyze_word_differences(self, original: str, corrected: str,
                                      speech_data: SpeechLearningData) -> List[PronunciationPattern]:
        """Analyze differences between original and corrected text."""
        patterns = []
        
        try:
            # Split into words
            original_words = original.lower().split()
            corrected_words = corrected.lower().split()
            
            # Use sequence matcher to find differences
            matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    # Word substitution - potential pronunciation pattern
                    for orig_idx in range(i1, i2):
                        for corr_idx in range(j1, j2):
                            if orig_idx < len(original_words) and corr_idx < len(corrected_words):
                                orig_word = original_words[orig_idx]
                                corr_word = corrected_words[corr_idx]
                                
                                if self._are_phonetically_similar(orig_word, corr_word):
                                    pattern = await self._create_pronunciation_pattern(
                                        orig_word, corr_word, speech_data
                                    )
                                    if pattern:
                                        patterns.append(pattern)
                
                elif tag == 'delete':
                    # Words that were incorrectly inserted by recognition
                    for orig_idx in range(i1, i2):
                        if orig_idx < len(original_words):
                            deleted_word = original_words[orig_idx]
                            # This might be a recognition artifact
                            pattern = await self._create_deletion_pattern(deleted_word, speech_data)
                            if pattern:
                                patterns.append(pattern)
                
                elif tag == 'insert':
                    # Words that were missed by recognition
                    for corr_idx in range(j1, j2):
                        if corr_idx < len(corrected_words):
                            inserted_word = corrected_words[corr_idx]
                            pattern = await self._create_insertion_pattern(inserted_word, speech_data)
                            if pattern:
                                patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Word difference analysis failed: {e}")
        
        return patterns
    
    def _are_phonetically_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are phonetically similar."""
        try:
            # Simple heuristics for phonetic similarity
            # In a full implementation, this would use proper phonetic analysis
            
            # Check edit distance
            edit_distance = difflib.SequenceMatcher(None, word1, word2).ratio()
            if edit_distance >= self.similarity_threshold:
                return True
            
            # Check for common phonetic substitutions
            phonetic_subs = {
                ('c', 'k'), ('ph', 'f'), ('gh', 'f'), ('w', 'v'),
                ('th', 'd'), ('th', 't'), ('s', 'z'), ('x', 'ks')
            }
            
            for sub1, sub2 in phonetic_subs:
                if ((sub1 in word1 and sub2 in word2) or 
                    (sub2 in word1 and sub1 in word2)):
                    return True
            
            # Check length similarity (for pronunciation errors)
            length_ratio = min(len(word1), len(word2)) / max(len(word1), len(word2))
            return length_ratio >= 0.6
            
        except Exception:
            return False
    
    async def _create_pronunciation_pattern(self, original_word: str, correct_word: str,
                                          speech_data: SpeechLearningData) -> Optional[PronunciationPattern]:
        """Create a pronunciation pattern from word pair."""
        try:
            pattern_id = f"{original_word}_{correct_word}_{hash(speech_data.session_id or '')}"
            
            # Analyze phonetic deviation
            phonetic_deviation = await self._analyze_phonetic_deviation(original_word, correct_word)
            
            # Determine context
            context_words = speech_data.preceding_context[-3:] if speech_data.preceding_context else []
            language_context = self._detect_language_context(speech_data.original_text)
            
            pattern = PronunciationPattern(
                pattern_id=pattern_id,
                user_id=speech_data.user_id,
                target_word=correct_word,
                actual_pronunciation=original_word,
                expected_pronunciation=correct_word,
                phonetic_deviation=phonetic_deviation,
                context_words=context_words,
                language_context=language_context
            )
            
            # Calculate initial priority
            pattern.calculate_priority()
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Failed to create pronunciation pattern: {e}")
            return None
    
    async def _create_deletion_pattern(self, deleted_word: str, 
                                     speech_data: SpeechLearningData) -> Optional[PronunciationPattern]:
        """Create pattern for words incorrectly recognized (should be deleted)."""
        try:
            pattern_id = f"delete_{deleted_word}_{hash(speech_data.session_id or '')}"
            
            pattern = PronunciationPattern(
                pattern_id=pattern_id,
                user_id=speech_data.user_id,
                target_word="",  # Should not be recognized
                actual_pronunciation=deleted_word,
                expected_pronunciation="",
                context_words=speech_data.preceding_context[-3:] if speech_data.preceding_context else [],
                language_context=self._detect_language_context(speech_data.original_text)
            )
            
            pattern.calculate_priority()
            return pattern
            
        except Exception as e:
            self.logger.error(f"Failed to create deletion pattern: {e}")
            return None
    
    async def _create_insertion_pattern(self, inserted_word: str,
                                      speech_data: SpeechLearningData) -> Optional[PronunciationPattern]:
        """Create pattern for words missed by recognition (should be inserted)."""
        try:
            pattern_id = f"insert_{inserted_word}_{hash(speech_data.session_id or '')}"
            
            pattern = PronunciationPattern(
                pattern_id=pattern_id,
                user_id=speech_data.user_id,
                target_word=inserted_word,
                actual_pronunciation="",  # Was not recognized
                expected_pronunciation=inserted_word,
                context_words=speech_data.preceding_context[-3:] if speech_data.preceding_context else [],
                language_context=self._detect_language_context(speech_data.original_text)
            )
            
            pattern.calculate_priority()
            return pattern
            
        except Exception as e:
            self.logger.error(f"Failed to create insertion pattern: {e}")
            return None
    
    async def _analyze_phonetic_deviation(self, original: str, correct: str) -> List[Tuple[str, str]]:
        """Analyze phonetic differences between words."""
        deviations = []
        
        try:
            # Simple character-level analysis
            # In a full implementation, this would use proper phoneme analysis
            matcher = difflib.SequenceMatcher(None, original, correct)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    orig_part = original[i1:i2]
                    corr_part = correct[j1:j2]
                    if orig_part and corr_part:
                        deviations.append((orig_part, corr_part))
                elif tag == 'delete':
                    deleted_part = original[i1:i2]
                    if deleted_part:
                        deviations.append((deleted_part, ''))
                elif tag == 'insert':
                    inserted_part = correct[j1:j2]
                    if inserted_part:
                        deviations.append('', inserted_part)
            
        except Exception as e:
            self.logger.error(f"Phonetic deviation analysis failed: {e}")
        
        return deviations
    
    def _detect_language_context(self, text: str) -> str:
        """Detect the language context of the text."""
        try:
            # Simple heuristic for Chinese-English detection
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            total_chars = len([c for c in text if c.isalpha() or '\u4e00' <= c <= '\u9fff'])
            
            if total_chars == 0:
                return "mixed"
            
            chinese_ratio = chinese_chars / total_chars
            
            if chinese_ratio > 0.7:
                return "chinese"
            elif chinese_ratio < 0.3:
                return "english"
            else:
                return "mixed"
                
        except Exception:
            return "mixed"
    
    async def _analyze_recognition_errors(self, speech_data: SpeechLearningData, 
                                        user_id: str) -> List[PronunciationPattern]:
        """Analyze recognition errors from confidence scores and alternatives."""
        patterns = []
        
        try:
            if not speech_data.alternative_transcriptions or not speech_data.confidence_scores:
                return patterns
            
            original_text = speech_data.original_text
            alternatives = speech_data.alternative_transcriptions
            confidences = speech_data.confidence_scores
            
            # If primary recognition has low confidence, check alternatives
            if confidences and confidences[0] < self.correction_confidence_threshold:
                for i, alternative in enumerate(alternatives[:3]):  # Check top 3 alternatives
                    if i < len(confidences) and confidences[i] > confidences[0]:
                        # Alternative has higher confidence - might be a pattern
                        pattern = await self._create_pronunciation_pattern(
                            original_text, alternative, speech_data
                        )
                        if pattern:
                            patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Recognition error analysis failed: {e}")
        
        return patterns
    
    async def _learn_contextual_patterns(self, speech_data: SpeechLearningData, 
                                       user_id: str) -> List[str]:
        """Learn patterns that depend on context."""
        improvements = []
        
        try:
            if not speech_data.context_type or not speech_data.original_text:
                return improvements
            
            context_type = speech_data.context_type
            text = speech_data.original_text.lower()
            
            # Learn programming context patterns
            if context_type in ["coding", "programming"]:
                prog_patterns = await self._learn_programming_patterns(text, user_id)
                improvements.extend(prog_patterns)
            
            # Learn file operation patterns
            elif context_type == "file_ops":
                file_patterns = await self._learn_file_operation_patterns(text, user_id)
                improvements.extend(file_patterns)
            
            # Learn system command patterns
            elif context_type == "system":
                sys_patterns = await self._learn_system_command_patterns(text, user_id)
                improvements.extend(sys_patterns)
            
        except Exception as e:
            self.logger.error(f"Contextual pattern learning failed: {e}")
        
        return improvements
    
    async def _learn_programming_patterns(self, text: str, user_id: str) -> List[str]:
        """Learn programming-specific pronunciation patterns."""
        improvements = []
        
        try:
            # Check for programming terms that might be mispronounced
            words = text.split()
            for word in words:
                if word in self.programming_phonetics:
                    # Check if this word commonly gets mispronounced
                    common_errors = self.programming_phonetics[word]
                    for error_pattern in common_errors:
                        # Create a preventive pattern
                        pattern_id = f"prog_{word}_{error_pattern}"
                        
                        if pattern_id not in self.pronunciation_patterns[user_id]:
                            pattern = PronunciationPattern(
                                pattern_id=pattern_id,
                                user_id=user_id,
                                target_word=word,
                                actual_pronunciation=error_pattern,
                                expected_pronunciation=word,
                                language_context="programming"
                            )
                            
                            self.pronunciation_patterns[user_id][pattern_id] = pattern
                            improvements.append(f"Added programming term pattern: {word}")
            
        except Exception as e:
            self.logger.error(f"Programming pattern learning failed: {e}")
        
        return improvements
    
    async def _learn_file_operation_patterns(self, text: str, user_id: str) -> List[str]:
        """Learn file operation pronunciation patterns."""
        improvements = []
        
        try:
            # Common file operation terms that get mispronounced
            file_terms = {
                'mkdir': ['make dir', 'make directory'],
                'ls': ['list', 'l s'],
                'pwd': ['p w d', 'print working directory'],
                'cd': ['c d', 'change directory'],
                'chmod': ['ch mod', 'change mode'],
                'grep': ['g rep', 'global regular expression print']
            }
            
            words = text.split()
            for word in words:
                if word in file_terms:
                    for variant in file_terms[word]:
                        pattern_id = f"file_{word}_{hash(variant)}"
                        
                        if pattern_id not in self.pronunciation_patterns[user_id]:
                            pattern = PronunciationPattern(
                                pattern_id=pattern_id,
                                user_id=user_id,
                                target_word=word,
                                actual_pronunciation=variant,
                                expected_pronunciation=word,
                                language_context="file_operations"
                            )
                            
                            self.pronunciation_patterns[user_id][pattern_id] = pattern
                            improvements.append(f"Added file operation pattern: {word}")
            
        except Exception as e:
            self.logger.error(f"File operation pattern learning failed: {e}")
        
        return improvements
    
    async def _learn_system_command_patterns(self, text: str, user_id: str) -> List[str]:
        """Learn system command pronunciation patterns."""
        improvements = []
        
        try:
            # System commands often mispronounced
            system_commands = {
                'sudo': ['pseudo', 'su do'],
                'ssh': ['s s h', 'secure shell'],
                'wget': ['w get', 'web get'],
                'curl': ['c url'],
                'ps': ['p s', 'process status'],
                'kill': ['kill process']
            }
            
            words = text.split()
            for word in words:
                if word in system_commands:
                    for variant in system_commands[word]:
                        pattern_id = f"sys_{word}_{hash(variant)}"
                        
                        if pattern_id not in self.pronunciation_patterns[user_id]:
                            pattern = PronunciationPattern(
                                pattern_id=pattern_id,
                                user_id=user_id,
                                target_word=word,
                                actual_pronunciation=variant,
                                expected_pronunciation=word,
                                language_context="system_commands"
                            )
                            
                            self.pronunciation_patterns[user_id][pattern_id] = pattern
                            improvements.append(f"Added system command pattern: {word}")
            
        except Exception as e:
            self.logger.error(f"System command pattern learning failed: {e}")
        
        return improvements
    
    async def _consolidate_patterns(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Consolidate similar patterns and remove low-priority ones."""
        try:
            user_patterns = self.pronunciation_patterns.get(user_id, {})
            if not user_patterns:
                return None
            
            # Group patterns by target word
            word_groups = defaultdict(list)
            for pattern in user_patterns.values():
                word_groups[pattern.target_word].append(pattern)
            
            consolidations = 0
            improvements = []
            patterns_removed = 0
            
            for target_word, patterns in word_groups.items():
                if len(patterns) > 1:
                    # Find the most frequent pattern
                    best_pattern = max(patterns, key=lambda p: p.occurrence_count)
                    
                    # Consolidate others into the best pattern
                    for pattern in patterns:
                        if pattern != best_pattern:
                            best_pattern.occurrence_count += pattern.occurrence_count
                            # Remove the consolidated pattern
                            del self.pronunciation_patterns[user_id][pattern.pattern_id]
                            patterns_removed += 1
                    
                    consolidations += 1
                    improvements.append(f"Consolidated {len(patterns)} patterns for '{target_word}'")
            
            # Remove patterns with very low priority or old patterns
            cutoff_date = datetime.now() - timedelta(days=90)  # 3 months
            low_priority_patterns = []
            
            for pattern_id, pattern in list(user_patterns.items()):
                if (pattern.adaptation_priority < 0.1 or 
                    pattern.last_observed < cutoff_date and pattern.occurrence_count < 2):
                    low_priority_patterns.append(pattern_id)
            
            for pattern_id in low_priority_patterns:
                del self.pronunciation_patterns[user_id][pattern_id]
                patterns_removed += 1
            
            if patterns_removed > 0:
                improvements.append(f"Removed {patterns_removed} low-priority patterns")
            
            return {
                'patterns_consolidated': consolidations,
                'patterns_removed': patterns_removed,
                'improvements': improvements
            }
            
        except Exception as e:
            self.logger.error(f"Pattern consolidation failed: {e}")
            return None
    
    async def _update_phonetic_mappings(self, user_id: str) -> List[str]:
        """Update phonetic mappings based on learned patterns."""
        improvements = []
        
        try:
            user_patterns = self.pronunciation_patterns.get(user_id, {})
            phonetic_map = self.phonetic_mappings[user_id]
            
            # Create mappings from high-frequency patterns
            for pattern in user_patterns.values():
                if pattern.occurrence_count >= self.min_error_frequency and pattern.adaptation_priority > 0.5:
                    key = pattern.actual_pronunciation.lower()
                    value = pattern.expected_pronunciation.lower()
                    
                    if key != value and key not in phonetic_map:
                        phonetic_map[key] = value
                        improvements.append(f"Added phonetic mapping: {key} -> {value}")
                        self.learning_stats['phonetic_mappings_created'] += 1
            
        except Exception as e:
            self.logger.error(f"Phonetic mapping update failed: {e}")
        
        return improvements
    
    def _calculate_pattern_confidence(self, user_id: str, patterns_learned: int) -> float:
        """Calculate confidence in learned pronunciation patterns."""
        try:
            user_patterns = self.pronunciation_patterns.get(user_id, {})
            
            if not user_patterns:
                return 0.0
            
            # Base confidence on pattern count and quality
            pattern_count_factor = min(len(user_patterns) / 50.0, 1.0)
            
            # Factor in pattern priorities
            avg_priority = np.mean([p.adaptation_priority for p in user_patterns.values()])
            priority_factor = avg_priority
            
            # Factor in recent learning
            recent_learning_factor = min(patterns_learned / 10.0, 1.0)
            
            # Factor in correction success rates
            success_rates = [p.correction_success_rate for p in user_patterns.values() 
                           if p.correction_success_rate > 0]
            success_factor = np.mean(success_rates) if success_rates else 0.5
            
            confidence = (
                pattern_count_factor * 0.3 +
                priority_factor * 0.25 +
                recent_learning_factor * 0.25 +
                success_factor * 0.2
            )
            
            return min(confidence, 0.95)
            
        except Exception:
            return 0.5
    
    async def _generate_correction_suggestions(self, user_id: str) -> List[str]:
        """Generate suggestions for pronunciation improvements."""
        suggestions = []
        
        try:
            user_patterns = self.pronunciation_patterns.get(user_id, {})
            
            if not user_patterns:
                return suggestions
            
            # Find most problematic patterns
            high_priority_patterns = sorted(
                user_patterns.values(),
                key=lambda p: p.adaptation_priority,
                reverse=True
            )[:5]  # Top 5 most important patterns
            
            for pattern in high_priority_patterns:
                if pattern.occurrence_count >= 3:
                    if pattern.target_word and pattern.actual_pronunciation:
                        suggestions.append(
                            f"Consider practicing '{pattern.target_word}' - "
                            f"often recognized as '{pattern.actual_pronunciation}'"
                        )
                    elif not pattern.target_word:  # Deletion pattern
                        suggestions.append(
                            f"'{pattern.actual_pronunciation}' is often incorrectly recognized - "
                            f"try speaking more clearly"
                        )
                    elif not pattern.actual_pronunciation:  # Insertion pattern
                        suggestions.append(
                            f"'{pattern.target_word}' is sometimes missed - "
                            f"try emphasizing this word"
                        )
            
            # General suggestions based on pattern analysis
            if len(user_patterns) > 20:
                suggestions.append("Multiple pronunciation patterns detected - consider speech training")
            
            # Language mixing suggestions
            mixed_patterns = [p for p in user_patterns.values() if p.language_context == "mixed"]
            if len(mixed_patterns) > 5:
                suggestions.append("Consider clearer language separation when code-switching")
            
        except Exception as e:
            self.logger.error(f"Suggestion generation failed: {e}")
        
        return suggestions
    
    def _initialize_programming_phonetics(self) -> Dict[str, List[str]]:
        """Initialize common programming term pronunciation errors."""
        return {
            'python': ['pie-thon', 'python'],
            'javascript': ['java script', 'j script'],
            'github': ['git hub', 'g hub'],
            'mysql': ['my sequel', 'my s q l'],
            'postgresql': ['postgres', 'post gres'],
            'nginx': ['engine x', 'n ginx'],
            'apache': ['a patch'],
            'jquery': ['j query', 'jay query'],
            'async': ['a sync', 'asynchronous'],
            'boolean': ['bool', 'true false'],
            'integer': ['int', 'whole number'],
            'string': ['text', 'str'],
            'array': ['list', 'collection'],
            'object': ['obj', 'instance'],
            'function': ['func', 'method'],
            'variable': ['var', 'value'],
            'constant': ['const', 'fixed value']
        }
    
    def _initialize_code_switching_patterns(self) -> Dict[str, List[str]]:
        """Initialize Chinese-English code switching patterns."""
        return {
            'file': ['文件', 'wen jian'],
            'folder': ['文件夹', 'wen jian jia'],
            'save': ['保存', 'bao cun'],
            'open': ['打开', 'da kai'],
            'close': ['关闭', 'guan bi'],
            'new': ['新建', 'xin jian'],
            'edit': ['编辑', 'bian ji'],
            'delete': ['删除', 'shan chu'],
            'copy': ['复制', 'fu zhi'],
            'paste': ['粘贴', 'zhan tie']
        }
    
    def _initialize_phonetic_analyzer(self):
        """Initialize phonetic analysis tools."""
        # Placeholder for phonetic analyzer initialization
        # In a full implementation, this would load phonetic models
        return None
    
    def _initialize_similarity_analyzer(self):
        """Initialize word similarity analyzer."""
        # Placeholder for similarity analyzer
        return None
    
    async def _setup_pattern_matching(self) -> None:
        """Setup pattern matching algorithms."""
        pass
    
    async def _load_pronunciation_patterns(self) -> None:
        """Load existing pronunciation patterns."""
        # Implementation would load from persistent storage
        pass
    
    def _dict_to_speech_data(self, data_dict: Dict[str, Any]) -> SpeechLearningData:
        """Convert dictionary to SpeechLearningData."""
        return SpeechLearningData(
            data_id=data_dict.get('data_id', str(uuid.uuid4())),
            user_id=data_dict.get('user_id', 'unknown'),
            session_id=data_dict.get('session_id'),
            original_text=data_dict.get('original_text', ''),
            corrected_text=data_dict.get('corrected_text'),
            user_correction=data_dict.get('user_correction'),
            confidence_scores=data_dict.get('confidence_scores', []),
            alternative_transcriptions=data_dict.get('alternative_transcriptions', []),
            context_type=data_dict.get('context_type'),
            preceding_context=data_dict.get('preceding_context', [])
        )
    
    async def _save_model(self) -> bool:
        """Save pronunciation patterns."""
        try:
            # Implementation would save to persistent storage
            self.logger.info(f"Saved pronunciation patterns for {len(self.pronunciation_patterns)} users")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save pronunciation patterns: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """Load pronunciation patterns."""
        try:
            # Implementation would load from persistent storage
            self.logger.info("Pronunciation patterns loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load pronunciation patterns: {e}")
            return False
    
    async def _cleanup_learner(self) -> None:
        """Cleanup learner resources."""
        self.pronunciation_patterns.clear()
        self.user_corrections.clear()
        self.phonetic_mappings.clear()
    
    # Public API methods
    
    async def get_pronunciation_patterns(self, user_id: str) -> Dict[str, PronunciationPattern]:
        """Get pronunciation patterns for a user."""
        return self.pronunciation_patterns.get(user_id, {})
    
    async def apply_pronunciation_corrections(self, user_id: str, text: str) -> str:
        """Apply learned pronunciation corrections to text."""
        try:
            corrected_text = text
            phonetic_map = self.phonetic_mappings.get(user_id, {})
            
            # Apply phonetic mappings
            words = corrected_text.split()
            corrected_words = []
            
            for word in words:
                word_lower = word.lower()
                if word_lower in phonetic_map:
                    corrected_words.append(phonetic_map[word_lower])
                    self.learning_stats['pattern_applications'] += 1
                else:
                    corrected_words.append(word)
            
            return ' '.join(corrected_words)
            
        except Exception as e:
            self.logger.error(f"Failed to apply pronunciation corrections: {e}")
            return text
    
    async def get_word_alternatives(self, user_id: str, word: str) -> List[str]:
        """Get alternative pronunciations/spellings for a word."""
        alternatives = []
        
        try:
            user_patterns = self.pronunciation_patterns.get(user_id, {})
            
            for pattern in user_patterns.values():
                if pattern.target_word.lower() == word.lower():
                    if pattern.actual_pronunciation:
                        alternatives.append(pattern.actual_pronunciation)
                elif pattern.actual_pronunciation.lower() == word.lower():
                    if pattern.target_word:
                        alternatives.append(pattern.target_word)
            
            # Remove duplicates and original word
            alternatives = list(set(alternatives))
            if word in alternatives:
                alternatives.remove(word)
            
        except Exception as e:
            self.logger.error(f"Failed to get word alternatives: {e}")
        
        return alternatives
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            **self.learning_stats,
            'active_users': len(self.pronunciation_patterns),
            'total_patterns': sum(len(patterns) for patterns in self.pronunciation_patterns.values()),
            'total_phonetic_mappings': sum(len(mappings) for mappings in self.phonetic_mappings.values())
        }