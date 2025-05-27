import logging
import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple, Optional, Any

class FakeNewsDetector(nn.Module):
    """Model that combines embeddings, TF-IDF, and other features"""
    
    def __init__(self, embedding_dim=384, tfidf_dim=5000, feature_dim=16, dropout=0.3):
        super(FakeNewsDetector, self).__init__()
        
        self.embedding_branch = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.tfidf_branch = nn.Sequential(
            nn.Linear(tfidf_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.feature_branch = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, embeddings, tfidf, features):
        emb_output = self.embedding_branch(embeddings)
        tfidf_output = self.tfidf_branch(tfidf)
        feat_output = self.feature_branch(features)
        combined = torch.cat((emb_output, tfidf_output, feat_output), dim=1)
        return self.classifier(combined)

class NERProcessor:
    """Utility class for Named Entity Recognition with multiple fallback strategies."""
    
    def __init__(self, use_spacy: bool = True):
        """Initialize the NER processor.
        
        Args:
            use_spacy: Whether to try loading spaCy (recommended for best results)
        """
        self.logger = logging.getLogger("NERProcessor")
        self.has_spacy = False
        self.nlp = None
        
        # Try to load spaCy if requested
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load('en_core_web_sm')
                self.has_spacy = True
                self.logger.info("Loaded spaCy model for NER")
            except Exception as e:
                self.logger.warning(f"Could not load spaCy: {str(e)}")
        
        # Prepare regex patterns for fallback NER
        self.entity_patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Full names
                r'\bPresident [A-Z][a-z]+\b',     # Titles with names
                r'\bDr\. [A-Z][a-z]+\b',
                r'\bMr\. [A-Z][a-z]+\b',
                r'\bMs\. [A-Z][a-z]+\b',
                r'\bMrs\. [A-Z][a-z]+\b'
            ],
            'ORG': [
                r'\b[A-Z][A-Za-z]* (Company|Corporation|Inc\.?|Ltd\.?|LLC)\b',
                r'\b[A-Z][A-Za-z]+ (University|College)\b',
                r'\bUniversity of [A-Z][A-Za-z]+\b',
                r'\b[A-Z][A-Z]+\b'  # Acronyms like NASA, FBI
            ],
            'GPE': [  # Geo-Political Entities
                r'\b[A-Z][a-z]+ (City|County|State)\b',
                r'\bCity of [A-Z][a-z]+\b',
                r'\b(United States|U\.S\.|US|USA|UK|China|Russia|Germany|France|Japan)\b'
            ],
            'DATE': [
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b',
                r'\b\d{4}\b'  # Years
            ],
            'QUANTITY': [
                r'\b\d+ (percent|million|billion|trillion)\b',
                r'\b\$\d+(\.\d+)? (million|billion|trillion)?\b'
            ]
        }
        
        # Load common stopwords that should not be considered entities
        self.stopwords = set([
            'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'I', 'You', 'He', 'She', 
            'We', 'They', 'It', 'There', 'Here', 'Today', 'Tomorrow', 'Yesterday',
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using best available method.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary mapping entity types to lists of entity strings
        """
        # Use spaCy if available
        if self.has_spacy and self.nlp:
            return self._extract_with_spacy(text)
        
        # Fallback to regex patterns
        return self._extract_with_regex(text)
    
    def _extract_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types to entity strings
        """
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'LOC': [],
            'DATE': [],
            'MONEY': [],
            'PRODUCT': [],
            'EVENT': [],
            'WORK_OF_ART': [],
            'MISC': []
        }
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Skip single character entities
                if len(ent.text.strip()) <= 1:
                    continue
                
                # Skip common stopwords
                if ent.text.strip() in self.stopwords:
                    continue
                
                # Add entity to appropriate category
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
                else:
                    entities['MISC'].append(ent.text)
            
            # Add noun chunks as potential entities if they're capitalized
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                
                # Check if starts with capital and not already included
                if (chunk_text and chunk_text[0].isupper() and
                    chunk_text not in self.stopwords and
                    not any(chunk_text in entity_list for entity_list in entities.values())):
                    
                    entities['MISC'].append(chunk_text)
        
        except Exception as e:
            self.logger.error(f"Error in spaCy entity extraction: {str(e)}")
        
        # Remove empty categories and duplicates
        result = {}
        for category, entity_list in entities.items():
            if entity_list:
                result[category] = list(dict.fromkeys(entity_list))  # Remove duplicates while preserving order
        
        return result
    
    def _extract_with_regex(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns when spaCy is unavailable.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types to entity strings
        """
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'DATE': [],
            'QUANTITY': [],
            'MISC': []
        }
        
        try:
            import re
            
            # Extract capitalized multi-word phrases (potential entities)
            capital_phrase_pattern = r'\b([A-Z][a-z]+(\s+[A-Z][a-z]+)+)\b'
            capital_phrases = re.findall(capital_phrase_pattern, text)
            for phrase_tuple in capital_phrases:
                phrase = phrase_tuple[0]
                if phrase not in self.stopwords:
                    entities['MISC'].append(phrase)
            
            # Apply specific patterns for each entity type
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]  # Take first group if it's a tuple
                        
                        if match and match not in self.stopwords:
                            entities[entity_type].append(match)
            
            # Extract capitalized single words not at sentence beginnings
            # (This is a simple heuristic, may produce false positives)
            sentences = text.split('.')
            for sentence in sentences:
                words = sentence.strip().split()
                for i, word in enumerate(words):
                    if (i > 0 and word and word[0].isupper() and 
                        word not in self.stopwords and 
                        all(word not in entity_list for entity_list in entities.values())):
                        
                        entities['MISC'].append(word)
        
        except Exception as e:
            self.logger.error(f"Error in regex entity extraction: {str(e)}")
        
        # Remove empty categories and duplicates
        result = {}
        for category, entity_list in entities.items():
            if entity_list:
                result[category] = list(dict.fromkeys(entity_list))  # Remove duplicates while preserving order
        
        return result
    
    def extract_flat_entities(self, text: str) -> List[str]:
        """Extract entities as a flat list without categories.
        
        Args:
            text: Input text
            
        Returns:
            List of entity strings
        """
        entity_dict = self.extract_entities(text)
        flat_entities = []
        
        for entity_list in entity_dict.values():
            flat_entities.extend(entity_list)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(flat_entities))