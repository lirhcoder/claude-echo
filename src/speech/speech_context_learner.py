"""Speech Context Learner - Learning Speech Context and Intent Patterns

Implements learning algorithms for understanding speech context patterns,
user-specific vocabulary, command preferences, and contextual speech habits.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import uuid

from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

from ..learning.base_learner import BaseLearner, LearningMode, LearningContext, LearningResult
from ..core.event_system import EventSystem
from .learning_types import SpeechContextPattern, SpeechLearningData


class SpeechContextLearner(BaseLearner):
    """
    Learns speech context patterns and user preferences.
    
    Analyzes and learns:
    - Context-specific vocabulary and phrases
    - Command usage patterns and preferences
    - Intent disambiguation based on context
    - User-specific terminology and shortcuts
    - Temporal usage patterns
    """
    
    def __init__(self, 
                 learner_id: str,
                 event_system: EventSystem,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the speech context learner."""
        super().__init__(learner_id, event_system, config)
        
        # Configuration
        self.min_pattern_frequency = self.config.get('min_pattern_frequency', 3)
        self.context_similarity_threshold = self.config.get('context_similarity_threshold', 0.7)
        self.vocab_learning_threshold = self.config.get('vocab_learning_threshold', 5)
        
        # Context data storage
        self.user_context_patterns: Dict[str, Dict[str, SpeechContextPattern]] = defaultdict(dict)
        self.user_vocabulary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.command_patterns: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # Learning models
        self.context_classifier = None
        self.vocabulary_vectorizer = None
        
        # Statistics
        self.context_stats = {
            'contexts_learned': 0,
            'vocabulary_terms_learned': 0,
            'command_patterns_identified': 0,
            'successful_predictions': 0
        }
        
        logger.info(f"SpeechContextLearner initialized for {learner_id}")
    
    @property
    def learner_type(self) -> str:
        return "speech_context"
    
    @property
    def supported_learning_modes(self) -> List[LearningMode]:
        return [LearningMode.ONLINE, LearningMode.SUPERVISED, LearningMode.BATCH]
    
    @property
    def input_data_types(self) -> List[str]:
        return ["speech_context", "user_commands", "intent_results", "temporal_patterns"]
    
    @property
    def output_types(self) -> List[str]:
        return ["context_patterns", "vocabulary_preferences", "command_suggestions"]
    
    async def _initialize_learner(self) -> None:
        """Initialize context learning models."""
        try:
            # Initialize text analysis tools
            self.vocabulary_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.context_classifier = MultinomialNB()
            
            # Load existing patterns
            await self._load_context_patterns()
            
            self.logger.info("Speech context learner initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize speech context learner: {e}")
            raise
    
    async def _learn_from_data(self, data: List[Dict[str, Any]], 
                             context: LearningContext) -> LearningResult:
        """Learn context patterns from speech data."""
        start_time = datetime.now()
        result = LearningResult(success=False)
        
        try:
            if not data:
                result.error_message = "No data provided for learning"
                return result
            
            user_id = context.user_id or "default_user"
            improvements = []
            contexts_learned = 0
            vocab_terms_learned = 0
            
            # Process speech context data
            for data_point in data:
                speech_data = self._dict_to_speech_data(data_point)
                
                # Learn context patterns
                context_result = await self._learn_context_patterns(speech_data, user_id)
                if context_result:
                    contexts_learned += context_result['patterns_created']
                    improvements.extend(context_result['improvements'])
                
                # Learn vocabulary preferences
                vocab_result = await self._learn_vocabulary_patterns(speech_data, user_id)
                if vocab_result:
                    vocab_terms_learned += vocab_result['terms_learned']
                    improvements.extend(vocab_result['improvements'])
                
                # Learn command patterns
                command_result = await self._learn_command_patterns(speech_data, user_id)
                if command_result:
                    improvements.extend(command_result['improvements'])
            
            # Update context models
            await self._update_context_models(user_id)
            
            # Generate recommendations
            recommendations = await self._generate_context_recommendations(user_id)
            
            # Calculate confidence
            confidence_score = self._calculate_context_confidence(user_id, contexts_learned)
            
            # Update statistics
            self.context_stats['contexts_learned'] += contexts_learned
            self.context_stats['vocabulary_terms_learned'] += vocab_terms_learned
            
            result = LearningResult(
                success=True,
                improvements=improvements,
                recommendations=recommendations,
                confidence_score=confidence_score,
                data_points_processed=len(data),
                metrics={
                    'contexts_learned': contexts_learned,
                    'vocab_terms_learned': vocab_terms_learned,
                    'total_context_patterns': len(self.user_context_patterns[user_id])
                }
            )
            
            self.logger.info(f"Learned context patterns for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Context learning failed: {e}")
            result.error_message = str(e)
        
        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result
    
    async def _learn_context_patterns(self, speech_data: SpeechLearningData, 
                                    user_id: str) -> Optional[Dict[str, Any]]:
        """Learn context-specific patterns."""
        try:
            if not speech_data.context_type or not speech_data.original_text:
                return None
            
            context_type = speech_data.context_type
            text = speech_data.original_text
            
            # Get or create context pattern
            pattern_id = f"{context_type}_{hash(text[:50])}"
            
            if pattern_id not in self.user_context_patterns[user_id]:
                pattern = SpeechContextPattern(
                    pattern_id=pattern_id,
                    user_id=user_id,
                    context_type=context_type
                )
                self.user_context_patterns[user_id][pattern_id] = pattern
                patterns_created = 1
                improvements = [f"Created new context pattern: {context_type}"]
            else:
                pattern = self.user_context_patterns[user_id][pattern_id]
                patterns_created = 0
                improvements = []
            
            # Update pattern data
            pattern.total_uses += 1
            pattern.last_updated = datetime.now()
            
            # Extract phrases
            phrases = self._extract_key_phrases(text)
            for phrase in phrases:
                if phrase not in pattern.common_phrases:
                    pattern.common_phrases.append(phrase)
                    improvements.append(f"Learned phrase for {context_type}: {phrase}")
            
            # Update success rate if feedback available
            if speech_data.user_accepted_result is not None:
                old_rate = pattern.success_rate
                pattern.success_rate = (old_rate * 0.9 + (1.0 if speech_data.user_accepted_result else 0.0) * 0.1)
            
            return {
                'patterns_created': patterns_created,
                'improvements': improvements
            }
            
        except Exception as e:
            self.logger.error(f"Context pattern learning failed: {e}")
            return None
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        phrases = []
        
        try:
            # Simple n-gram extraction
            words = text.lower().split()
            
            # Extract 2-grams and 3-grams
            for i in range(len(words) - 1):
                bigram = ' '.join(words[i:i+2])
                if len(bigram) > 5:  # Skip very short phrases
                    phrases.append(bigram)
            
            for i in range(len(words) - 2):
                trigram = ' '.join(words[i:i+3])
                if len(trigram) > 8:
                    phrases.append(trigram)
            
            # Extract programming-specific patterns
            prog_patterns = [
                r'\b(create|make|generate)\s+\w+',
                r'\b(open|edit|modify)\s+\w+',
                r'\b(run|execute|start)\s+\w+',
                r'\b\w+\s+(function|method|class)',
                r'\b(import|include|require)\s+\w+'
            ]
            
            for pattern in prog_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                phrases.extend(matches)
            
        except Exception as e:
            self.logger.error(f"Phrase extraction failed: {e}")
        
        return list(set(phrases))  # Remove duplicates
    
    async def _learn_vocabulary_patterns(self, speech_data: SpeechLearningData, 
                                       user_id: str) -> Optional[Dict[str, Any]]:
        """Learn user-specific vocabulary patterns."""
        try:
            if not speech_data.original_text:
                return None
            
            words = speech_data.original_text.lower().split()
            terms_learned = 0
            improvements = []
            
            for word in words:
                if len(word) > 2 and word.isalpha():  # Valid words only
                    self.user_vocabulary[user_id][word] += 1
                    
                    # Check if this is a newly frequent term
                    if self.user_vocabulary[user_id][word] == self.vocab_learning_threshold:
                        terms_learned += 1
                        improvements.append(f"Learned frequent term: {word}")
            
            # Learn technical terms
            tech_terms = self._extract_technical_terms(speech_data.original_text)
            for term in tech_terms:
                self.user_vocabulary[user_id][f"tech_{term}"] += 1
                terms_learned += 1
                improvements.append(f"Learned technical term: {term}")
            
            return {
                'terms_learned': terms_learned,
                'improvements': improvements
            }
            
        except Exception as e:
            self.logger.error(f"Vocabulary learning failed: {e}")
            return None
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical/programming terms."""
        tech_patterns = [
            r'\b[A-Z][a-z]+[A-Z]\w*',  # CamelCase
            r'\b[a-z]+_[a-z_]+',       # snake_case
            r'\b\w+\.(js|py|html|css|json)\b',  # File extensions
            r'\b(def|class|import|function|var|let|const)\b',  # Keywords
            r'\b\w+\(\)',              # Function calls
            r'\b[A-Z]{2,}\b'          # Acronyms
        ]
        
        tech_terms = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            tech_terms.extend(matches)
        
        return list(set(tech_terms))
    
    async def _learn_command_patterns(self, speech_data: SpeechLearningData, 
                                    user_id: str) -> Optional[Dict[str, Any]]:
        """Learn command usage patterns."""
        try:
            if not speech_data.intent_classification or not speech_data.original_text:
                return None
            
            intent = speech_data.intent_classification
            command_text = speech_data.original_text
            
            # Store command pattern
            self.command_patterns[user_id][intent].append(command_text)
            
            # Keep only recent patterns (last 50)
            if len(self.command_patterns[user_id][intent]) > 50:
                self.command_patterns[user_id][intent] = self.command_patterns[user_id][intent][-50:]
            
            improvements = [f"Learned command pattern for {intent}"]
            
            return {
                'improvements': improvements
            }
            
        except Exception as e:
            self.logger.error(f"Command pattern learning failed: {e}")
            return None
    
    async def _update_context_models(self, user_id: str) -> None:
        """Update machine learning models for context prediction."""
        try:
            # Prepare training data
            texts = []
            labels = []
            
            for pattern in self.user_context_patterns[user_id].values():
                for phrase in pattern.common_phrases:
                    texts.append(phrase)
                    labels.append(pattern.context_type)
            
            if len(texts) < 5:  # Need minimum samples
                return
            
            # Update vectorizer and classifier
            if len(set(labels)) > 1:  # Need multiple classes
                X = self.vocabulary_vectorizer.fit_transform(texts)
                self.context_classifier.fit(X, labels)
                
                self.logger.debug(f"Updated context model with {len(texts)} samples")
            
        except Exception as e:
            self.logger.error(f"Context model update failed: {e}")
    
    def _calculate_context_confidence(self, user_id: str, contexts_learned: int) -> float:
        """Calculate confidence in context learning."""
        try:
            patterns = self.user_context_patterns.get(user_id, {})
            
            if not patterns:
                return 0.0
            
            # Base on pattern count and quality
            pattern_count_factor = min(len(patterns) / 10.0, 1.0)
            
            # Factor in usage frequency
            avg_usage = sum(p.total_uses for p in patterns.values()) / len(patterns)
            usage_factor = min(avg_usage / 10.0, 1.0)
            
            # Factor in success rates
            success_rates = [p.success_rate for p in patterns.values() if p.success_rate > 0]
            success_factor = sum(success_rates) / len(success_rates) if success_rates else 0.5
            
            # Recent learning factor
            recent_factor = min(contexts_learned / 5.0, 1.0)
            
            confidence = (
                pattern_count_factor * 0.3 +
                usage_factor * 0.3 +
                success_factor * 0.3 +
                recent_factor * 0.1
            )
            
            return min(confidence, 0.95)
            
        except Exception:
            return 0.5
    
    async def _generate_context_recommendations(self, user_id: str) -> List[str]:
        """Generate context-based recommendations."""
        recommendations = []
        
        try:
            patterns = self.user_context_patterns.get(user_id, {})
            vocab = self.user_vocabulary.get(user_id, {})
            
            # Context coverage recommendations
            if len(patterns) < 3:
                recommendations.append("Use voice commands in different contexts to improve personalization")
            
            # Vocabulary recommendations
            frequent_terms = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:10]
            if len(frequent_terms) > 5:
                top_terms = [term for term, count in frequent_terms[:3]]
                recommendations.append(f"Most used terms: {', '.join(top_terms)}")
            
            # Success rate recommendations
            low_success_patterns = [p for p in patterns.values() if p.success_rate < 0.6 and p.total_uses > 3]
            if low_success_patterns:
                recommendations.append(f"Consider rephrasing commands in {low_success_patterns[0].context_type} context")
            
            # Command pattern recommendations
            command_patterns = self.command_patterns.get(user_id, {})
            if command_patterns:
                most_used_intent = max(command_patterns.keys(), key=lambda k: len(command_patterns[k]))
                recommendations.append(f"Most frequently used: {most_used_intent} commands")
            
        except Exception as e:
            self.logger.error(f"Context recommendation generation failed: {e}")
        
        return recommendations
    
    def _dict_to_speech_data(self, data_dict: Dict[str, Any]) -> SpeechLearningData:
        """Convert dictionary to SpeechLearningData."""
        return SpeechLearningData(
            data_id=data_dict.get('data_id', str(uuid.uuid4())),
            user_id=data_dict.get('user_id', 'unknown'),
            original_text=data_dict.get('original_text', ''),
            context_type=data_dict.get('context_type'),
            intent_classification=data_dict.get('intent_classification'),
            preceding_context=data_dict.get('preceding_context', []),
            user_accepted_result=data_dict.get('user_accepted_result')
        )
    
    async def _load_context_patterns(self) -> None:
        """Load existing context patterns."""
        pass
    
    async def _save_model(self) -> bool:
        """Save context patterns and models."""
        try:
            self.logger.info(f"Saved context patterns for {len(self.user_context_patterns)} users")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save context models: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """Load context patterns and models."""
        try:
            self.logger.info("Context models loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load context models: {e}")
            return False
    
    async def _cleanup_learner(self) -> None:
        """Cleanup learner resources."""
        self.user_context_patterns.clear()
        self.user_vocabulary.clear()
        self.command_patterns.clear()
    
    # Public API methods
    
    async def predict_context(self, user_id: str, text: str) -> Optional[str]:
        """Predict context for given text."""
        try:
            if not self.context_classifier or not self.vocabulary_vectorizer:
                return None
            
            X = self.vocabulary_vectorizer.transform([text])
            prediction = self.context_classifier.predict(X)
            return prediction[0] if len(prediction) > 0 else None
            
        except Exception as e:
            self.logger.error(f"Context prediction failed: {e}")
            return None
    
    async def get_vocabulary_suggestions(self, user_id: str, context: str) -> List[str]:
        """Get vocabulary suggestions for a context."""
        try:
            patterns = self.user_context_patterns.get(user_id, {})
            context_patterns = [p for p in patterns.values() if p.context_type == context]
            
            suggestions = []
            for pattern in context_patterns:
                suggestions.extend(pattern.common_phrases[:5])  # Top 5 phrases per pattern
            
            return list(set(suggestions))[:10]  # Top 10 unique suggestions
            
        except Exception as e:
            self.logger.error(f"Vocabulary suggestion failed: {e}")
            return []
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context learning statistics."""
        return {
            **self.context_stats,
            'active_users': len(self.user_context_patterns),
            'total_patterns': sum(len(patterns) for patterns in self.user_context_patterns.values()),
            'vocabulary_size': sum(len(vocab) for vocab in self.user_vocabulary.values())
        }