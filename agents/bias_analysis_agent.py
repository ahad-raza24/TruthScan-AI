import json
import os
import logging
import re
import math
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict

from .base_agent import Agent
from utils.name_entity_recognition import NERProcessor

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from nltk import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

class BiasDetectionAgent(Agent):
    """Agent responsible for detecting bias in news articles with enhanced capabilities."""
    
    def __init__(self, name: str = "BiasDetector", 
                 lexicon_path: str = None, 
                 subjectivity_lexicon_path: str = None,
                 bias_weights: Dict[str, float] = None):
        """Initialize the bias detection agent with expanded lexicons and configurable weights.
        
        Args:
            name: Agent name
            lexicon_path: Optional path to expanded lexicons file in JSON format
            subjectivity_lexicon_path: Optional path to MPQA subjectivity lexicon in TFF format
            bias_weights: Optional dictionary of weights for bias score components
        """
        super().__init__(name)
        
        # Check dependencies
        if not NLTK_AVAILABLE:
            self.log("NLTK not available; tokenization disabled", "error")
            raise ImportError("NLTK is required for BiasDetectionAgent")
        
        # Load spaCy model for NER if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                self.log("Loaded spaCy model for entity analysis")
                self.has_spacy = True
            except Exception as e:
                self.nlp = None
                self.has_spacy = False
                self.log(f"Failed to load spaCy model: {str(e)}", "warning")
        else:
            self.nlp = None
            self.has_spacy = False
            self.log("spaCy not available; entity analysis will use rule-based fallback", "warning")
        
        # Default bias score weights - adjusted for better sensitivity
        default_weights = {
            'political_bias': 0.25,
            'subjectivity_score': 0.20,
            'sentiment_score': 0.15,
            'exaggeration_score': 0.15,
            'loaded_phrase_score': 0.10,
            'clickbait_score': 0.10,
            'citation_density': 0.05
        }
        self.bias_weights = bias_weights if bias_weights else default_weights
        
        # Normalize weights if necessary
        if abs(sum(self.bias_weights.values()) - 1.0) > 0.001:
            self.log(f"Normalizing weights from sum of {sum(self.bias_weights.values())} to 1.0", "warning")
            total = sum(self.bias_weights.values())
            self.bias_weights = {k: v / total for k, v in self.bias_weights.items()}
        
        # Initialize lexicon stats
        self.lexicon_stats = {
            'positive_words': 0,
            'negative_words': 0,
            'subjective_words': 0,
            'left_political': 0,
            'right_political': 0,
            'exaggeration_words': 0,
            'hedging_words': 0,
            'loaded_phrases': 0,
            'clickbait_phrases': 0
        }
        
        # Initialize expanded lexicons with default values first
        self._initialize_default_lexicons()
        
        # Load MPQA subjectivity lexicon if provided
        if subjectivity_lexicon_path and os.path.exists(subjectivity_lexicon_path):
            self._load_mpqa_subjectivity_lexicon(subjectivity_lexicon_path)
        
        # Load additional lexicons from JSON file if provided
        if lexicon_path and os.path.exists(lexicon_path):
            self._load_json_lexicons(lexicon_path)
            
        # Log summary of loaded lexicons
        self.log("Lexicon loading complete. Lexicon sizes:")
        for key, count in self.lexicon_stats.items():
            self.log(f"  - {key}: {count} terms")
    
    def _initialize_default_lexicons(self):
        """Initialize default lexicons with basic values."""
        # Political terms
        self.political_terms = {
            'left': set([
                'progressive', 'liberal', 'democrat', 'socialism', 'welfare', 'regulation',
                'equality', 'diversity', 'inclusive', 'activist', 'reform', 'collective',
                'social justice', 'worker rights', 'union', 'environmental', 'green new deal',
                'medicare for all', 'public option', 'universal healthcare', 'planned parenthood',
                'pro-choice', 'gun control', 'black lives matter', 'lgbtq+', 'living wage',
                'wealth tax', 'democratic socialism', 'defund police', 'immigration reform',
                'climate action', 'equity', 'labor union', 'progressive tax', 'single payer'
            ]),
            'right': set([
                'conservative', 'republican', 'capitalism', 'deregulation', 'traditional',
                'patriot', 'nationalist', 'america first', 'free market', 'liberty', 'freedom',
                'individual', 'originalist', 'constitutional', 'fiscal responsibility',
                'small government', 'pro-life', 'second amendment', 'gun rights', 'family values',
                'religious freedom', 'tough on crime', 'national security', 'military strength',
                'law and order', 'tax cuts', 'border security', 'privatization', 'tough borders',
                'free speech', 'school choice', 'job creator', 'limited government'
            ])
        }
        
        # Expanded positive sentiment lexicon
        self.positive_words = set([
            'good', 'great', 'excellent', 'positive', 'nice', 'correct', 'true', 'authentic',
            'legitimate', 'factual', 'accurate', 'verified', 'proven', 'confirmed', 'valid',
            'right', 'honest', 'reliable', 'trustworthy', 'credible', 'beneficial', 'effective',
            'successful', 'remarkable', 'outstanding', 'impressive', 'exceptional', 'favorable',
            'constructive', 'commendable', 'praiseworthy', 'admirable', 'valuable', 'helpful',
            'promising', 'encouraging', 'optimistic', 'uplifting', 'breakthrough', 'innovative',
            'improving', 'progress', 'advance', 'solution', 'achievement', 'triumph', 'victory'
        ])
        
        # Expanded negative sentiment lexicon
        self.negative_words = set([
            'bad', 'awful', 'terrible', 'negative', 'wrong', 'false', 'fake', 'hoax',
            'misleading', 'deceptive', 'fraudulent', 'unverified', 'untrue', 'bogus', 'fabricated',
            'inaccurate', 'erroneous', 'dishonest', 'unreliable', 'questionable', 'dubious',
            'suspicious', 'harmful', 'failed', 'poor', 'disappointing', 'inferior', 'baseless',
            'problematic', 'flawed', 'mistaken', 'exaggerated', 'misrepresented', 'dangerous',
            'misguided', 'biased', 'distorted', 'disinformation', 'propaganda', 'conspiracy',
            'corrupt', 'scandal', 'crisis', 'disaster', 'catastrophe', 'failure', 'damaging'
        ])
        
        # Expanded subjective expression lexicon
        self.subjective_words = set([
            'think', 'believe', 'feel', 'opinion', 'seems', 'appears', 'maybe', 'perhaps',
            'allegedly', 'supposedly', 'apparently', 'rumors', 'claiming', 'reportedly',
            'suggests', 'suspected', 'speculated', 'controversial', 'debatable', 'disputed',
            'contentious', 'unclear', 'uncertain', 'doubtful', 'likely', 'unlikely', 'possible',
            'impossible', 'claimed', 'assumes', 'presumably', 'supposedly', 'ostensibly',
            'purportedly', 'argued', 'contended', 'insisted', 'perspective', 'viewpoint', 
            'interpretation', 'judgment', 'assumption', 'theory', 'speculation', 'assertion'
        ])
        
        # Expanded exaggeration lexicon
        self.exaggeration_words = set([
            'very', 'extremely', 'incredibly', 'absolutely', 'definitely', 'undoubtedly',
            'totally', 'completely', 'utterly', 'entirely', 'overwhelmingly', 'extraordinarily',
            'exceedingly', 'vastly', 'tremendously', 'immensely', 'profoundly', 'fundamentally',
            'drastically', 'radically', 'massively', 'perfectly', 'endless', 'unlimited',
            'unparalleled', 'unprecedented', 'revolutionary', 'groundbreaking', 'earth-shattering',
            'ever', 'never', 'always', 'impossible', 'unimaginable', 'catastrophic', 'disastrous',
            'crisis', 'emergency', 'tsunami', 'explosion', 'skyrocket', 'plummet', 'collapse',
            'huge', 'enormous', 'giant', 'colossal', 'major', 'critical', 'essential', 'vital'
        ])
        
        # Expanded hedging expressions lexicon
        self.hedging_words = set([
            'may', 'might', 'could', 'possibly', 'potentially', 'conceivably', 'seemingly',
            'somewhat', 'fairly', 'relatively', 'generally', 'usually', 'normally', 'typically',
            'in general', 'for the most part', 'to some extent', 'to a certain degree',
            'in a sense', 'in a way', 'sort of', 'kind of', 'more or less', 'approximately',
            'roughly', 'about', 'around', 'suggest', 'indicate', 'imply', 'hint', 'allude',
            'not necessarily', 'not entirely', 'not always', 'not exactly', 'almost', 'nearly'
        ])
        
        # Expanded loaded phrases lexicon
        self.loaded_phrases = set([
            'so-called', 'self-proclaimed', 'alleged', 'purported', 'supposed', 'claimed',
            'radical', 'extremist', 'staunch', 'hardline', 'fringe', 'militant', 'fanatic',
            'denier', 'apologist', 'enabler', 'elitist', 'establishment', 'mainstream',
            'special interest', 'scandal-plagued', 'corrupt', 'crooked', 'failed', 'disgraced',
            'controversial', 'notorious', 'infamous', 'embattled', 'beleaguered', 'far-right',
            'far-left', 'ultra-conservative', 'ultra-liberal', 'conspiracy theorist',
            'propaganda machine', 'fake news', 'deep state', 'witch hunt', 'hoax'
        ])
        
        # Expanded clickbait phrases
        self.clickbait_phrases = set([
            'you won\'t believe', 'shocking', 'mind-blowing', 'jaw-dropping', 'game-changing',
            'what happens next', 'this is why', 'here\'s why', 'the truth about', 'the real reason',
            'secret', 'hidden', 'revealed', 'exclusive', 'breaking', 'just in', 'urgent',
            'warning', 'must see', 'must read', 'need to know', 'simple trick', 'one weird trick',
            'doctors hate', 'this one thing', 'stunning', 'miraculous', 'unbelievable',
            'will shock you', 'changed forever', 'will never be the same', 'gone wrong',
            'gone viral', 'what they don\'t want you to know', 'signs you might', 'top reasons'
        ])
        
        # Update lexicon stats
        self.lexicon_stats = {
            'positive_words': len(self.positive_words),
            'negative_words': len(self.negative_words),
            'subjective_words': len(self.subjective_words),
            'left_political': len(self.political_terms['left']),
            'right_political': len(self.political_terms['right']),
            'exaggeration_words': len(self.exaggeration_words),
            'hedging_words': len(self.hedging_words),
            'loaded_phrases': len(self.loaded_phrases),
            'clickbait_phrases': len(self.clickbait_phrases)
        }
    
    def _load_mpqa_subjectivity_lexicon(self, lexicon_path: str):
        """Load the MPQA Subjectivity Lexicon from TFF format.
        
        Args:
            lexicon_path: Path to the TFF format lexicon file
        """
        self.log(f"Loading MPQA Subjectivity Lexicon from {lexicon_path}")
        
        # Initialize counters for statistics
        strong_subj_count = 0
        weak_subj_count = 0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        both_count = 0
        
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Skip comments and empty lines
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    # Parse the line into key-value pairs
                    entries = {}
                    for entry in line.strip().split():
                        if '=' in entry:
                            key, value = entry.split('=', 1)
                            entries[key] = value
                    
                    # Extract the necessary information
                    if 'word1' in entries and 'priorpolarity' in entries and 'type' in entries:
                        word = entries['word1'].lower()
                        polarity = entries['priorpolarity']
                        subj_type = entries['type']
                        pos = entries.get('pos1', 'anypos')
                        
                        # Add to subjective words
                        if subj_type == 'strongsubj':
                            self.subjective_words.add(word)
                            strong_subj_count += 1
                        elif subj_type == 'weaksubj':
                            # Optionally add weak subjective words too
                            self.subjective_words.add(word)
                            weak_subj_count += 1
                        
                        # Add to sentiment lexicons based on polarity
                        if polarity == 'positive':
                            self.positive_words.add(word)
                            positive_count += 1
                        elif polarity == 'negative':
                            self.negative_words.add(word)
                            negative_count += 1
                        elif polarity == 'neutral':
                            neutral_count += 1
                        elif polarity == 'both':
                            # Words marked as 'both' have both positive and negative meanings
                            self.positive_words.add(word)
                            self.negative_words.add(word)
                            both_count += 1
            
            # Update lexicon stats
            self.lexicon_stats['positive_words'] = len(self.positive_words)
            self.lexicon_stats['negative_words'] = len(self.negative_words)
            self.lexicon_stats['subjective_words'] = len(self.subjective_words)
            
            self.log(f"MPQA Lexicon loaded successfully: {strong_subj_count} strong subjective, "
                     f"{weak_subj_count} weak subjective, {positive_count} positive, "
                     f"{negative_count} negative, {neutral_count} neutral, {both_count} both")
            
        except Exception as e:
            self.log(f"Error loading MPQA Subjectivity Lexicon: {str(e)}", "error")
    
    def _load_json_lexicons(self, lexicon_path: str):
        """Load additional lexicons from a JSON file.
        
        Args:
            lexicon_path: Path to the JSON lexicon file
        """
        self.log(f"Loading additional lexicons from {lexicon_path}")
        
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                lexicons = json.load(f)
            
            # Process political terms if present
            if 'political_terms' in lexicons:
                if isinstance(lexicons['political_terms'], dict):
                    # Handle dictionary format with 'left' and 'right' keys
                    if 'left' in lexicons['political_terms']:
                        self.political_terms['left'].update(set(lexicons['political_terms']['left']))
                    if 'right' in lexicons['political_terms']:
                        self.political_terms['right'].update(set(lexicons['political_terms']['right']))
                elif isinstance(lexicons['political_terms'], list):
                    # Legacy format handling
                    self.log("Found list format for political terms, attempting to categorize", "warning")
                    for term in lexicons['political_terms']:
                        # Simple heuristic for categorization
                        left_matches = sum(1 for t in self.political_terms['left'] if t in term or term in t)
                        right_matches = sum(1 for t in self.political_terms['right'] if t in term or term in t)
                        
                        if left_matches > right_matches:
                            self.political_terms['left'].add(term)
                        elif right_matches > left_matches:
                            self.political_terms['right'].add(term)
            
            # Process media bias sources if present
            if 'media_bias_sources' in lexicons:
                self.media_bias_sources = lexicons['media_bias_sources']
                self.log(f"Loaded media bias sources: {sum(len(sources) for sources in self.media_bias_sources.values())} sources")
            else:
                # Initialize empty media bias sources
                self.media_bias_sources = {'left_leaning': [], 'right_leaning': [], 'center': []}
            
            # Process other lexicon categories
            lexicon_mapping = {
                'positive_words': self.positive_words,
                'negative_words': self.negative_words,
                'subjective_words': self.subjective_words,
                'exaggeration_words': self.exaggeration_words,
                'hedging_words': self.hedging_words,
                'loaded_phrases': self.loaded_phrases,
                'clickbait_phrases': self.clickbait_phrases
            }
            
            # Special handling for conspiracy and scientific skepticism terms
            if 'conspiracy_terms' in lexicons and isinstance(lexicons['conspiracy_terms'], list):
                if not hasattr(self, 'conspiracy_terms'):
                    self.conspiracy_terms = set()
                self.conspiracy_terms.update(set(lexicons['conspiracy_terms']))
                self.log(f"Loaded {len(self.conspiracy_terms)} conspiracy terms")
            
            if 'scientific_skepticism' in lexicons and isinstance(lexicons['scientific_skepticism'], list):
                if not hasattr(self, 'scientific_skepticism'):
                    self.scientific_skepticism = set()
                self.scientific_skepticism.update(set(lexicons['scientific_skepticism']))
                self.log(f"Loaded {len(self.scientific_skepticism)} scientific skepticism terms")
            
            # Process standard lexicon categories
            for key, target_set in lexicon_mapping.items():
                if key in lexicons and isinstance(lexicons[key], list):
                    target_set.update(set(lexicons[key]))
            
            # Update lexicon stats
            self.lexicon_stats = {
                'positive_words': len(self.positive_words),
                'negative_words': len(self.negative_words),
                'subjective_words': len(self.subjective_words),
                'left_political': len(self.political_terms['left']),
                'right_political': len(self.political_terms['right']),
                'exaggeration_words': len(self.exaggeration_words),
                'hedging_words': len(self.hedging_words),
                'loaded_phrases': len(self.loaded_phrases),
                'clickbait_phrases': len(self.clickbait_phrases)
            }
            
            self.log(f"Successfully loaded lexicons from {lexicon_path}")
            
        except Exception as e:
            self.log(f"Error loading lexicons from JSON file: {str(e)}", "error")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text by normalizing whitespace and converting to lowercase."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text safely."""
        if not text:
            return []
        try:
            return word_tokenize(text)
        except Exception as e:
            self.log(f"Error in tokenization: {str(e)}", "warning")
            return text.split()

    def _get_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Get n-grams from token list."""
        return [' '.join(gram) for gram in ngrams(tokens, n)]
    
    def _extract_entities(self, title: str, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """Extract named entities with improved robustness.
        
        Args:
            title: Article title
            text: Article text
            
        Returns:
            Dictionary of entity types to list of (entity text, entity type) tuples
        """
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geo-political entities (countries, cities, etc.)
            'LOC': [],  # Non-GPE locations
            'PRODUCT': [],
            'EVENT': [],
            'WORK_OF_ART': [],
            'LAW': [],
            'OTHER': []
        }
        
        full_text = f"{title} {text}"
        
        # Try using spaCy for NER if available
        if self.has_spacy and self.nlp:
            try:
                doc = self.nlp(full_text)
                for ent in doc.ents:
                    # Skip very short entities
                    if len(ent.text.strip()) <= 2:
                        continue
                    
                    # Map entity to appropriate category
                    category = ent.label_ if ent.label_ in entities else 'OTHER'
                    entities[category].append((ent.text, ent.label_))
                
                # If we found entities, return them
                if any(entities.values()):
                    return entities
                else:
                    self.log("spaCy NER found no entities, falling back to rule-based extraction", "warning")
            except Exception as e:
                self.log(f"Error in spaCy NER: {str(e)}. Falling back to rule-based extraction", "warning")
        
        # Fallback: Rule-based NER
        self.log("Using rule-based NER fallback", "info")
        
        # Extract potential entity candidates using regex patterns
        person_pattern = r'\b([A-Z][a-z]+ (?:[A-Z][a-z]+ )?[A-Z][a-z]+)\b'
        org_pattern = r'\b([A-Z][A-Za-z]+ (?:Company|Corporation|Inc\.?|Ltd\.?|LLC|University|College|Association|Foundation))\b'
        gpe_pattern = r'\b(United States|U\.S\.|United Kingdom|U\.K\.|Russia|China|Germany|France|[A-Z][a-z]+ (?:City|County|State))\b'
        
        # Extract entities
        for match in re.finditer(person_pattern, full_text):
            entities['PERSON'].append((match.group(1), 'PERSON'))
        
        for match in re.finditer(org_pattern, full_text):
            entities['ORG'].append((match.group(1), 'ORG'))
        
        for match in re.finditer(gpe_pattern, full_text):
            entities['GPE'].append((match.group(1), 'GPE'))
        
        # Extract capitalized phrases as potential entities
        caps_pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b'
        for match in re.finditer(caps_pattern, full_text):
            # Check if already captured by other patterns
            entity_text = match.group(1)
            if not any(entity_text in [e[0] for e in cat] for cat in entities.values()):
                entities['OTHER'].append((entity_text, 'OTHER'))
        
        return entities

    def _calculate_entity_sentiment(self, text: str, entities: Dict[str, List[Tuple[str, str]]]) -> Dict[str, float]:
        """Calculate sentiment for each entity.
        
        Args:
            text: Preprocessed full text
            entities: Dict of entity types to entity tuples
            
        Returns:
            Dictionary mapping entity text to sentiment score
        """
        entity_sentiment = {}
        
        # Flatten entity list
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)
        
        # Get sentences
        sentences = sent_tokenize(text)
        
        # Calculate sentiment for each entity based on sentences containing it
        for entity_tuple in all_entities:
            entity_text, _ = entity_tuple
            entity_lower = entity_text.lower()
            
            # Find sentences containing this entity
            containing_sentences = [s for s in sentences if entity_lower in s.lower()]
            
            if not containing_sentences:
                continue
                
            # Calculate sentiment scores for these sentences
            positive_count = 0
            negative_count = 0
            
            for sentence in containing_sentences:
                tokens = self._tokenize(sentence)
                positive_count += sum(1 for token in tokens if token in self.positive_words)
                negative_count += sum(1 for token in tokens if token in self.negative_words)
            
            # Calculate sentiment score
            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment = 0
                
            entity_sentiment[entity_text] = sentiment
        
        return entity_sentiment

    def _find_phrases(self, text: str, phrase_set: Set[str]) -> int:
        """Find exact phrase matches in text.
        
        Args:
            text: Preprocessed text
            phrase_set: Set of phrases to look for
            
        Returns:
            Count of phrases found
        """
        count = 0
        
        for phrase in phrase_set:
            # Handle multi-word phrases
            if ' ' in phrase:
                pattern = r'\b' + re.escape(phrase) + r'\b'
                matches = re.findall(pattern, text, re.IGNORECASE)
                count += len(matches) * 2  # Weight phrases higher
            # For single words, only count exact matches
            else:
                words = text.split()
                count += sum(1 for word in words if word.lower() == phrase.lower())
                
        return count
    
    def _check_source_bias(self, source: str) -> Tuple[str, float]:
        """Check if the source domain has a known political bias.
        
        Args:
            source: The source domain (e.g., 'cnn.com', 'foxnews.com')
            
        Returns:
            Tuple of (bias_direction, bias_strength)
        """
        if not hasattr(self, 'media_bias_sources'):
            return 'unknown', 0.0
            
        source = source.lower().strip()
        # Extract domain from URL if necessary
        if '/' in source:
            source = source.split('/')[0]
        
        # Remove common prefixes and suffixes
        for prefix in ['www.', 'news.', 'blog.']:
            if source.startswith(prefix):
                source = source[len(prefix):]
                
        for suffix in ['.com', '.org', '.net', '.co.uk', '.gov', '.edu']:
            if source.endswith(suffix):
                source = source[:-len(suffix)]
        
        # Check if source is in our bias database
        for bias, sources in self.media_bias_sources.items():
            for known_source in sources:
                if source in known_source or known_source in source:
                    if bias == 'left_leaning':
                        return 'left', 0.7
                    elif bias == 'right_leaning':
                        return 'right', 0.7
                    elif bias == 'center':
                        return 'center', 0.2
        
        return 'unknown', 0.0

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect bias in news article or claim with enhanced analysis.
        
        Args:
            data: Dictionary containing text to analyze {'title': str, 'text': str, 'source': str}
                
        Returns:
            Dictionary containing enhanced bias analysis
        """
        # Input validation
        if not isinstance(data, dict) or 'text' not in data:
            self.log("Invalid input: 'text' field required", "error")
            return {'bias_score': 0.0, 'error': "Invalid input: 'text' field required"}
        
        title = data.get('title', '')
        text = data.get('text', '')
        source = data.get('source', '')
        
        # Preprocess text
        clean_title = self._preprocess_text(title)
        clean_text = self._preprocess_text(text)
        full_text = f"{clean_title} {clean_text}"
        
        # Tokenize
        tokens = self._tokenize(full_text)
        if not tokens:
            self.log("Warning: No tokens found in text", "warning")
            return {'bias_score': 0.0, 'error': "No tokens found in text"}
        
        self.log(f"Analyzing bias in text from {source} ({len(tokens)} tokens)")
        self.log(f"Sample tokens: {tokens[:10]}")  # Debug: show first 10 tokens
        
        # Get bi-grams for better phrase matching
        bigrams = self._get_ngrams(tokens, 2)
        
        # 1. Political leaning analysis
        left_count = 0
        right_count = 0
        
        # Check single-word political terms
        for token in tokens:
            if token in self.political_terms['left']:
                left_count += 1
            if token in self.political_terms['right']:
                right_count += 1
                
        # Check multi-word political terms (especially important)
        for phrase in bigrams:
            if phrase in self.political_terms['left']:
                left_count += 2  # Weight phrases higher
            if phrase in self.political_terms['right']:
                right_count += 2
                
        # Additional search for specific multi-word phrases
        for phrase in self.political_terms['left']:
            if ' ' in phrase and phrase in full_text:
                left_count += 2
                
        for phrase in self.political_terms['right']:
            if ' ' in phrase and phrase in full_text:
                right_count += 2
        
        # Source bias checking (if we have a source)
        source_leaning, source_bias_strength = 'unknown', 0.0
        if source:
            source_leaning, source_bias_strength = self._check_source_bias(source)
            if source_leaning == 'left':
                left_count += int(source_bias_strength * 5)  # Add weighted bias based on source
            elif source_leaning == 'right':
                right_count += int(source_bias_strength * 5)
        
        # Calculate political leaning
        total_political_terms = left_count + right_count
        if total_political_terms == 0:
            political_leaning = 'center'
            left_ratio = 0.5
        else:
            total_political_terms = max(1, total_political_terms)
            left_ratio = left_count / total_political_terms
            
            # Use more sensitive thresholds
            if left_ratio > 0.6:
                political_leaning = 'left'
            elif left_ratio < 0.4:
                political_leaning = 'right'
            else:
                political_leaning = 'center'
        
        # Calculate political bias strength (how politically charged the text is)
        political_bias_raw = (left_count + right_count) / max(1, len(tokens))
        political_bias = min(1.0, political_bias_raw * 20)  # Increased scaling for better sensitivity
        
        # 2. Sentiment analysis
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)
        
        # Add bigram sentiment (for phrases like "very good")
        for bigram in bigrams:
            bigram_split = bigram.split()
            if len(bigram_split) == 2:
                # Check if first word is intensifier and second is sentiment
                if (bigram_split[0] in self.exaggeration_words and 
                    (bigram_split[1] in self.positive_words or bigram_split[1] in self.negative_words)):
                    # Intensified sentiment counts double
                    if bigram_split[1] in self.positive_words:
                        positive_count += 1
                    elif bigram_split[1] in self.negative_words:
                        negative_count += 1
        
        sentiment_score = ((positive_count - negative_count) / 
                         max(1, positive_count + negative_count)) if positive_count + negative_count > 0 else 0
        
        # 3. Loaded phrases analysis
        loaded_phrase_count = self._find_phrases(full_text, self.loaded_phrases)
        loaded_phrase_score = min(1.0, loaded_phrase_count / max(5, len(tokens) / 30))
        
        # 4. Subjectivity analysis
        subjectivity_count = sum(1 for token in tokens if token in self.subjective_words)
        
        # Check for bigrams containing subjectivity markers
        for bigram in bigrams:
            bigram_words = bigram.split()
            # Check combinations like "I believe", "they think"
            if len(bigram_words) == 2 and bigram_words[1] in self.subjective_words:
                subjectivity_count += 1
                
        subjectivity_score = min(1.0, subjectivity_count / max(1, len(tokens)) * 20)
        
        # 5. Hedging language analysis
        hedging_count = sum(1 for token in tokens if token in self.hedging_words)
        hedging_score = min(1.0, hedging_count / max(1, len(tokens)) * 25)
        
        # 6. Exaggeration analysis
        exaggeration_count = sum(1 for token in tokens if token in self.exaggeration_words)
        exaggeration_score = min(1.0, exaggeration_count / max(1, len(tokens)) * 25)
        
        # 7. Clickbait analysis - important for headlines
        clickbait_score = 0
        
        # Check for clickbait phrases in title (weighted higher)
        for phrase in self.clickbait_phrases:
            if phrase in clean_title:
                clickbait_score += 0.3  # Higher weight for title
        
        # Check for clickbait phrases in full text
        for phrase in self.clickbait_phrases:
            if phrase in full_text and phrase not in clean_title:  # Avoid double counting
                clickbait_score += 0.2
                
        clickbait_score = min(1.0, clickbait_score)  # Cap at 1.0
        
        # 8. Citation analysis
        citation_patterns = [
            r'according to', r'cited by', r'reported by', r'said', r'source', 
            r'reference', r'reports that', r'stated', r'quoted', r'noted', 
            r'mentioned by', r'as per', r'in the words of'
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, full_text))
        
        # Calculate how many citations would be expected based on text length
        expected_citations = max(1, len(tokens) / 100)
        citation_density = min(1.0, citation_count / expected_citations)
        
        # 9. Entity analysis
        entity_dict = self._extract_entities(title, text)
        entity_sentiment = self._calculate_entity_sentiment(full_text, entity_dict)
        
        # Calculate entity bias (contrasting sentiment toward different entities)
        entity_bias = 0
        if len(entity_sentiment) >= 2:
            sentiment_values = list(entity_sentiment.values())
            # Calculate variance in sentiment
            mean_sentiment = sum(sentiment_values) / len(sentiment_values)
            variance = sum((s - mean_sentiment) ** 2 for s in sentiment_values) / len(sentiment_values)
            entity_bias = min(1.0, math.sqrt(variance) * 2)  # Scale up for sensitivity
        
        # 10. Conspiracy language analysis (if available)
        conspiracy_score = 0
        if hasattr(self, 'conspiracy_terms'):
            conspiracy_count = sum(1 for token in tokens if token in self.conspiracy_terms)
            # Check for multi-word conspiracy phrases
            for phrase in self.conspiracy_terms:
                if ' ' in phrase and phrase in full_text:
                    conspiracy_count += 2
            conspiracy_score = min(1.0, conspiracy_count / max(10, len(tokens) / 10))
        
        # 11. Calculate overall bias score with weights
        bias_score = (
            self.bias_weights['political_bias'] * political_bias +
            self.bias_weights['subjectivity_score'] * subjectivity_score +
            self.bias_weights['sentiment_score'] * abs(sentiment_score) +
            self.bias_weights['exaggeration_score'] * exaggeration_score +
            self.bias_weights['loaded_phrase_score'] * loaded_phrase_score +
            self.bias_weights['clickbait_score'] * clickbait_score +
            self.bias_weights['citation_density'] * (1 - citation_density)  # Invert: lower citation density means higher bias
        )
        
        # Apply secondary boost for entity bias
        if entity_bias > 0.3:
            bias_score = bias_score * (1.0 + (entity_bias * 0.2))
            
        # Apply conspiracy language boost if significant
        if conspiracy_score > 0.3:
            bias_score = bias_score * (1.0 + (conspiracy_score * 0.3))
        
        # 12. Generate bias indicators
        bias_indicators = []
        
        # Only add political leaning if it's strong enough
        if political_bias > 0.2:
            if political_leaning != 'center':
                bias_indicators.append(f"Shows {political_leaning}-leaning political terms")
        
        # Add source bias if known
        if source_leaning != 'unknown' and source_bias_strength > 0.3:
            bias_indicators.append(f"Source ({source}) has known {source_leaning}-leaning bias")
        
        # Add sentiment bias indicator if significant
        if abs(sentiment_score) > 0.2:
            if sentiment_score > 0:
                bias_indicators.append("Uses positive loaded language")
            else:
                bias_indicators.append("Uses negative loaded language")
        
        # Other indicators with adjusted thresholds
        if subjectivity_score > 0.2:
            bias_indicators.append("Contains subjective language")
        
        if hedging_score > 0.2:
            bias_indicators.append("Uses hedging language")
        
        if exaggeration_score > 0.2:
            bias_indicators.append("Uses exaggerated language")
        
        if clickbait_score > 0.2:
            bias_indicators.append("Contains clickbait language")
        
        if citation_density < 0.3 and len(tokens) > 50:
            bias_indicators.append("Lacks sufficient citations or sources")
        
        if entity_bias > 0.3 and len(entity_sentiment) >= 2:
            bias_indicators.append("Shows contrasting sentiment toward different entities")
        
        if loaded_phrase_score > 0.2:
            bias_indicators.append("Uses loaded phrases")
            
        if conspiracy_score > 0.2:
            bias_indicators.append("Contains conspiracy-related language")
        
        # 13. Construct final result
        result = {
            'bias_score': min(1.0, max(0.0, bias_score)),
            'political_leaning': political_leaning,
            'political_bias': political_bias,
            'left_count': left_count,
            'right_count': right_count,
            'subjectivity_score': subjectivity_score,
            'sentiment_score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'exaggeration_score': exaggeration_score,
            'hedging_score': hedging_score,
            'loaded_phrase_score': loaded_phrase_score,
            'clickbait_score': clickbait_score,
            'citation_density': citation_density,
            'entity_bias': entity_bias,
            'entity_sentiment': entity_sentiment,
            'bias_indicators': bias_indicators
        }
        
        # Add conspiracy score if available
        if hasattr(self, 'conspiracy_terms'):
            result['conspiracy_score'] = conspiracy_score
            
        # Add source bias info if available
        if source_leaning != 'unknown':
            result['source_bias'] = {
                'leaning': source_leaning,
                'strength': source_bias_strength
            }
        
        self.log(f"Bias analysis complete: score={bias_score:.2f}, leaning={political_leaning}")
        self.log(f"Bias components: political={political_bias:.3f}, subjectivity={subjectivity_score:.3f}, sentiment={abs(sentiment_score):.3f}, exaggeration={exaggeration_score:.3f}")
        return result