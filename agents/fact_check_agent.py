import torch
import os
import pickle
import re
import time
from typing import Dict, List, Any
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer
from scipy import sparse

from .base_agent import Agent
from utils.name_entity_recognition import NERProcessor
from utils.knowledge_graph import KnowledgeGraphConnector
from ml_components.model_training import FakeNewsDetector

class FactCheckAgent(Agent):
    """Agent responsible for verifying facts using language models and knowledge graphs."""
    
    def __init__(self, name: str = "FactChecker", model_path: str = None, 
                 device: str = None, knowledge_graph: KnowledgeGraphConnector = None,
                 enable_kg_verification: bool = True, kg_timeout: float = 8.0):
        """Initialize the fact check agent.
        
        Args:
            name: Agent name
            model_path: Path to the fine-tuned model checkpoint
            device: Computing device (cuda or cpu)
            knowledge_graph: KnowledgeGraphConnector instance
            enable_kg_verification: Whether to use knowledge graph verification
            kg_timeout: Maximum time to spend on knowledge graph verification
        """
        super().__init__(name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.knowledge_graph = knowledge_graph
        self.enable_kg_verification = enable_kg_verification
        self.kg_timeout = kg_timeout
        
        self.model = None
        self.tokenizer = None
        self.tfidf_vectorizer = None
        self.has_kg = knowledge_graph is not None
        
        # Load the fine-tuned model if path is provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self.log("No model path provided or model not found. Running in limited mode.", "warning")

    def _create_embedding_layer(self):
        """Create an embedding layer for the model if it doesn't exist."""
        if hasattr(self, '_embedding_layer'):
            return
            
        try:
            # Get vocabulary size from tokenizer
            vocab_size = self.tokenizer.vocab_size
            
            # Get embedding dimension from model
            if hasattr(self.model, 'embedding_branch'):
                # For DeBERTa and RoBERTa models with custom heads
                embedding_dim = self.model.embedding_branch[0].in_features
            else:
                # Default embedding dimension
                embedding_dim = 768
                
            # Create embedding layer
            self._embedding_layer = torch.nn.Embedding(
                vocab_size, 
                embedding_dim
            ).to(self.device)
            
            # Initialize with random values
            torch.nn.init.normal_(self._embedding_layer.weight, mean=0, std=0.02)
            
            self.log(f"Created embedding layer with vocab size {vocab_size} and dimension {embedding_dim}")
        except Exception as e:
            self.log(f"Error creating embedding layer: {str(e)}", "error")
            import traceback
            traceback.print_exc()
    
    def _load_model(self, model_path: str):
        """Load the fine-tuned model with proper error handling.
        
        Args:
            model_path: Path to model directory or checkpoint
        """
        try:
            self.log(f"Loading model from {model_path}")
            
            # Check if path is a directory or file
            if os.path.isdir(model_path):
                # Find best model checkpoint
                if "deberta" in model_path.lower():
                    checkpoint_path = os.path.join(model_path, "deberta_best.pt")
                    model_type = "deberta"
                else:
                    checkpoint_path = os.path.join(model_path, "roberta_best.pt")
                    model_type = "roberta"
                
                config_path = os.path.join(model_path, "config.pkl")
            else:
                # Direct checkpoint path
                checkpoint_path = model_path
                config_path = os.path.join(os.path.dirname(model_path), "config.pkl")
                model_type = "deberta" if "deberta" in model_path.lower() else "roberta"
            
            # Load model configuration
            if os.path.exists(config_path):
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
                self.log(f"Loaded model config from {config_path}")
            else:
                # Default configuration if config file not found
                self.log("Config file not found, using default parameters", "warning")
                config = {
                    'embedding_dim': 768,
                    'tfidf_dim': 5000,
                    'feature_dim': 16,
                    'dropout': 0.3,
                    'base_dir': '/kaggle/working'
                }
            
            # Initialize model with config
            self.model = FakeNewsDetector(
                embedding_dim=config.get('embedding_dim', 768),
                tfidf_dim=config.get('tfidf_dim', 5000),
                feature_dim=config.get('feature_dim', 16),
                dropout=config.get('dropout', 0.3)
            )
            
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                self.log(f"Model checkpoint not found at {checkpoint_path}", "error")
                return
            
            # Load weights
            self.model.load_state_dict(torch.load(
                checkpoint_path, 
                map_location=self.device,
                weights_only=True  # Add this parameter to fix the warning
            ))
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer based on model type
            model_name = "microsoft/deberta-v3-small" if model_type == "deberta" else "roberta-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_type = model_type
            
            # Try to load TF-IDF vectorizer
            vectorizer_path = os.path.join(config.get('base_dir', '/kaggle/working'), 
                                          'data/processed/tfidf_vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                self.log("Loaded TF-IDF vectorizer")
            else:
                self.log("TF-IDF vectorizer not found at expected path", "warning")
            
            self.log(f"Successfully loaded {model_type.upper()} model and tokenizer")
        except Exception as e:
            self.log(f"Error loading model: {str(e)}", "error")
            import traceback
            traceback.print_exc()
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a claim for fact-checking.
        
        Args:
            data: Dictionary containing claim and evidence articles
                {
                    'claim': 'The claim text to verify',
                    'evidence': [list of evidence articles],
                    'knowledge_graph': optional KnowledgeGraphConnector instance
                }
                
        Returns:
            Dictionary containing verification results
        """
        claim = data.get('claim', '')
        evidence_articles = data.get('evidence', [])
        knowledge_graph = data.get('knowledge_graph') or self.knowledge_graph
        
        self.log(f"Fact checking claim: '{claim}'")
        
        if not claim:
            self.log("No claim provided", "error")
            return {
                'claim': claim,
                'verification_score': 0.0,
                'evidence_matches': [],
                'explanation': "No claim provided for verification"
            }
        
        # 1. Calculate relevance scores between claim and evidence articles
        relevant_evidence = self._find_relevant_evidence(claim, evidence_articles)
        
        # 2. If knowledge graph is available and enabled, try to verify using it (optimized)
        kg_verification = 0.0
        entity_verification = []
        if knowledge_graph and self.enable_kg_verification:
            self.log("Attempting verification using Knowledge Graph")
            kg_start_time = time.time()
            
            try:
                # Extract entities
                entities = self._extract_entities_from_claim(claim)
                
                if entities and len(entities) >= 1:
                    # Limit to top 2 entities and 2 relationships for performance
                    max_entities = min(2, len(entities))
                    relationships = ["related to", "involves"]  # Reduced from 4 to 2 relationships
                    
                    # Try different combinations of entities and relationships
                    for i in range(max_entities):
                        # Check timeout
                        if (time.time() - kg_start_time) > self.kg_timeout:
                            self.log(f"Knowledge Graph verification timeout after {self.kg_timeout}s", "warning")
                            break
                            
                        for j in range(i+1, min(max_entities + 1, len(entities))):
                            entity1 = entities[i]
                            entity2 = entities[j]
                            
                            # Try relationships in order of likelihood
                            for rel in relationships:
                                # Check timeout before each query
                                if (time.time() - kg_start_time) > self.kg_timeout:
                                    self.log(f"Knowledge Graph verification timeout after {self.kg_timeout}s", "warning")
                                    break
                                    
                                fact_result = knowledge_graph.verify_fact_triple(entity1, rel, entity2)
                                if fact_result.get('verified', False):
                                    kg_verification = max(kg_verification, fact_result.get('confidence', 0.7))
                                    self.log(f"Fact verified by Knowledge Graph: {entity1} {rel} {entity2}")
                                    entity_verification.append(fact_result)
                                    # Early termination - stop after first verification
                                    break
                            
                            # Early termination - stop after first successful verification
                            if kg_verification > 0:
                                break
                        
                        # Early termination - stop after first successful verification
                        if kg_verification > 0:
                            break
                
                kg_time = time.time() - kg_start_time
                self.log(f"Knowledge Graph verification completed in {kg_time:.2f}s")
                            
            except Exception as e:
                self.log(f"Error during Knowledge Graph verification: {str(e)}", "warning")
        elif knowledge_graph and not self.enable_kg_verification:
            self.log("Knowledge Graph verification disabled for faster processing")
        
        # 3. Use model to verify if available
        model_verification = 0.0
        if self.model and self.tokenizer:
            model_verification = self._verify_with_model(claim, relevant_evidence)
        else:
            # Fallback to simpler heuristic verification
            model_verification = self._heuristic_verification(claim, relevant_evidence)
        
        # 4. Combine verification scores (KG has higher weight if available)
        if kg_verification > 0:
            verification_score = (0.7 * kg_verification) + (0.3 * model_verification)
        else:
            verification_score = model_verification
        
        # 5. Generate explanation
        explanation = self._generate_explanation(claim, relevant_evidence, verification_score, entity_verification)
        
        result = {
            'claim': claim,
            'verification_score': verification_score,
            'evidence_matches': relevant_evidence[:3],  # Top 3 most relevant evidence
            'explanation': explanation,
            'kg_verification': kg_verification > 0  # Flag if KG was used
        }
        
        self.log(f"Verification score: {verification_score:.4f}")
        return result

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text for knowledge graph queries.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        # Simple implementation - extract capitalized words
        # In a real implementation, use NER
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        return words
    
    def _find_relevant_evidence(self, claim: str, evidence_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find evidence articles relevant to the claim with enhanced similarity.
        
        Args:
            claim: The claim text
            evidence_articles: List of evidence articles
            
        Returns:
            List of evidence articles with relevance scores
        """
        relevant_evidence = []
        
        # Extract key terms from claim
        claim_words = set(word_tokenize(claim.lower()))
        
        for article in evidence_articles:
            # Get article title and text
            article_title = article.get('title', '')
            article_text = article.get('text', '')
            full_text = f"{article_title} {article_text}".lower()
            
            # Calculate word overlap (basic lexical similarity)
            article_words = set(word_tokenize(full_text))
            
            # Calculate overlap
            if len(claim_words) > 0 and len(article_words) > 0:
                word_overlap = len(claim_words.intersection(article_words)) / len(claim_words)
            else:
                word_overlap = 0
            
            # Calculate semantic similarity if model is available
            semantic_similarity = 0
            if self.model and self.tokenizer:
                try:
                    semantic_similarity = self._calculate_semantic_similarity(claim, full_text)
                except Exception as e:
                    self.log(f"Error calculating semantic similarity: {str(e)}", "warning")
            
            # Calculate phrase matching (for exact phrases from claim)
            phrase_match_score = 0
            claim_phrases = self._extract_phrases(claim)
            for phrase in claim_phrases:
                if phrase.lower() in full_text:
                    phrase_match_score += 0.2  # Bonus for each matched phrase
            phrase_match_score = min(1.0, phrase_match_score)
            
            # Combined relevance score
            if semantic_similarity > 0:
                relevance_score = (0.4 * word_overlap) + (0.4 * semantic_similarity) + (0.2 * phrase_match_score)
            else:
                relevance_score = (0.7 * word_overlap) + (0.3 * phrase_match_score)
                
            # Add article with relevance score if it's above threshold
            if relevance_score > 0.05:  # Lower threshold to include more potentially relevant evidence
                relevant_article = article.copy()
                relevant_article['relevance_score'] = relevance_score
                relevant_evidence.append(relevant_article)
        
        # Sort by relevance score
        relevant_evidence.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_evidence
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text.
        
        Args:
            text: Input text
            
        Returns:
            List of phrases
        """
        # Simple phrase extraction based on sentence parsing
        phrases = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Add the entire sentence as a phrase
            if 3 <= len(sentence.split()) <= 12:
                phrases.append(sentence)
                
            # Extract noun phrases (simplified approach)
            tokens = word_tokenize(sentence)
            for i in range(len(tokens) - 1):
                if len(tokens[i]) > 1 and len(tokens[i+1]) > 1:  # Skip single-character tokens
                    phrases.append(f"{tokens[i]} {tokens[i+1]}")
        
        return phrases
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not self.tokenizer or not self.model:
            return 0
            
        try:
            # Tokenize both texts
            inputs = self.tokenizer(
                [text1, text2],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                # First pass inputs through the model embedding layer if it exists
                if hasattr(self.model, 'embedding_branch'):
                    # DeBERTa models need input_ids converted to embedding vectors first
                    # Add a fake embedding layer if needed
                    if not hasattr(self, '_embedding_layer'):
                        vocab_size = self.tokenizer.vocab_size
                        if hasattr(self.model, 'embedding_branch'):
                            embedding_dim = self.model.embedding_branch[0].in_features
                        else:
                            embedding_dim = 768  # Default embedding dimension
                            
                        self._embedding_layer = torch.nn.Embedding(
                            vocab_size, 
                            embedding_dim
                        ).to(self.device)
                        
                        # Initialize with random values
                        torch.nn.init.normal_(self._embedding_layer.weight, mean=0, std=0.02)
                    
                    # Convert input_ids to embeddings
                    embeddings = self._embedding_layer(inputs['input_ids'])
                    
                    # Get mean of token embeddings for each sequence
                    sequence_embeddings = embeddings.mean(dim=1)
                    
                    # Now pass through the model's embedding branch
                    outputs = self.model.embedding_branch(sequence_embeddings)
                else:
                    # For models where embedding_branch isn't available, use the base model
                    outputs = self.model.base_model(inputs['input_ids']).last_hidden_state
                    outputs = outputs.mean(dim=1)  # Take mean of token representations
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                outputs[0].unsqueeze(0),
                outputs[1].unsqueeze(0)
            ).item()
            
            # Normalize to 0-1 range (cosine similarity is between -1 and 1)
            return (similarity + 1) / 2
        except Exception as e:
            self.log(f"Error in semantic similarity calculation: {str(e)}", "warning")
            import traceback
            traceback.print_exc()
            return 0

    def _verify_with_model(self, claim: str, evidence: List[Dict[str, Any]]) -> float:
        """Use the fine-tuned model to verify the claim against evidence.
        
        Args:
            claim: The claim text
            evidence: List of relevant evidence articles
            
        Returns:
            Verification score (0-1, where 1 means verified as true)
        """
        if not self.model or not self.tokenizer:
            return self._heuristic_verification(claim, evidence)
            
        try:
            self.log("Verifying with fine-tuned model")
            
            # Prepare combined text from evidence (top 3 most relevant)
            evidence_text = ""
            for e in evidence[:3]:
                title = e.get('title', '')
                text = e.get('text', '')
                evidence_text += f"{title}. {text} "
            
            # Truncate if too long
            if len(evidence_text) > 5000:
                evidence_text = evidence_text[:5000]
                
            combined_text = f"{claim} {evidence_text}"
            
            # 1. Get embeddings from model
            inputs = self.tokenizer(
                combined_text,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Handle embeddings based on model architecture
            with torch.no_grad():
                if hasattr(self.model, 'embedding_branch'):
                    # Create embedding layer if not already created
                    if not hasattr(self, '_embedding_layer'):
                        vocab_size = self.tokenizer.vocab_size
                        embedding_dim = self.model.embedding_branch[0].in_features
                        
                        self._embedding_layer = torch.nn.Embedding(
                            vocab_size, 
                            embedding_dim
                        ).to(self.device)
                        
                        # Initialize with random values
                        torch.nn.init.normal_(self._embedding_layer.weight, mean=0, std=0.02)
                    
                    # Convert input_ids to embeddings
                    token_embeddings = self._embedding_layer(inputs['input_ids'])
                    
                    # Get mean of token embeddings
                    sequence_embedding = token_embeddings.mean(dim=1)
                    
                    # Get embeddings from model's embedding branch
                    embeddings = self.model.embedding_branch(sequence_embedding)
                else:
                    # For models without embedding_branch
                    outputs = self.model.base_model(inputs['input_ids']).last_hidden_state
                    embeddings = outputs.mean(dim=1)
                
            # 2. Check if we have TF-IDF vectorizer and can run full model
            if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer is not None:
                try:
                    tfidf_features = self.tfidf_vectorizer.transform([combined_text])
                    if sparse.issparse(tfidf_features):
                        tfidf_features = tfidf_features.toarray()
                    tfidf_tensor = torch.tensor(tfidf_features, dtype=torch.float).to(self.device)
                    
                    # 3. Create linguistic features (simplified)
                    feature_dim = self.model.feature_branch[0].in_features
                    features = torch.zeros((1, feature_dim), dtype=torch.float).to(self.device)
                    
                    # 4. Run full model for prediction
                    with torch.no_grad():
                        output = self.model(embeddings, tfidf_tensor, features)
                        prediction = torch.sigmoid(output).item()
                    
                    return prediction
                except Exception as e:
                    self.log(f"Error using TF-IDF vectorizer: {str(e)}", "warning")
                    # Fall back to heuristic verification
                    return self._heuristic_verification(claim, evidence)
            else:
                # Fallback when full model components aren't available
                self.log("TF-IDF vectorizer not available, using heuristic verification", "warning")
                return self._heuristic_verification(claim, evidence)
                
        except Exception as e:
            self.log(f"Error in model verification: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            # Fall back to heuristic verification
            return self._heuristic_verification(claim, evidence)
    
    def _heuristic_verification(self, claim: str, evidence: List[Dict[str, Any]]) -> float:
        """Use enhanced heuristics to verify claim when model is unavailable.
        
        Args:
            claim: The claim text
            evidence: List of relevant evidence
            
        Returns:
            Verification score (0-1)
        """
        self.log("Using enhanced heuristic verification")
        
        # Start with neutral score
        score = 0.5
        
        # Adjust score based on evidence availability
        if not evidence:
            return 0.4  # Slightly higher base confidence without evidence
        
        # Count sources with high relevance
        high_relevance_count = sum(1 for e in evidence if e.get('relevance_score', 0) > 0.5)
        medium_relevance_count = sum(1 for e in evidence if 0.3 <= e.get('relevance_score', 0) <= 0.5)
        
        # Adjust score based on number of high-relevance sources
        if high_relevance_count >= 3:
            score += 0.3
        elif high_relevance_count >= 2:
            score += 0.2
        elif high_relevance_count >= 1:
            score += 0.15
        elif medium_relevance_count >= 3:
            score += 0.1
        elif medium_relevance_count >= 1:
            score += 0.05
        elif len(evidence) <= 1:
            score -= 0.05
            
        # Check for credible sources
        credible_sources = {'reuters', 'associated press', 'bbc', 'npr', 'pbs', 
                           'nyt', 'new york times', 'washington post', 'guardian',
                           'abc', 'cbs', 'nbc', 'financial times', 'economist'}
        
        has_credible = any(
            any(src in e.get('source', '').lower() for src in credible_sources)
            for e in evidence
        )
        
        if has_credible:
            score += 0.1
        
        # Check for consensus among sources
        if len(evidence) >= 3:
            # Calculate relevance-weighted average of top 3 evidence
            top_evidence = evidence[:3]
            total_relevance = sum(e.get('relevance_score', 0) for e in top_evidence)
            if total_relevance > 0:
                weighted_score = sum(e.get('relevance_score', 0) / total_relevance for e in top_evidence)
                score = (score + weighted_score) / 2  # Blend the scores
        
        # Check for contradictions in evidence
        contradiction_detected = False
        positive_terms = {'confirm', 'verify', 'prove', 'support', 'validate', 'true', 'correct'}
        negative_terms = {'deny', 'debunk', 'disprove', 'false', 'incorrect', 'hoax', 'fake'}
        
        # Check for presence of contradicting terms in different evidence pieces
        has_positive = False
        has_negative = False
        
        for e in evidence[:5]:  # Check top 5 evidence pieces
            text = (e.get('title', '') + " " + e.get('text', '')).lower()
            if any(term in text for term in positive_terms):
                has_positive = True
            if any(term in text for term in negative_terms):
                has_negative = True
        
        # If both positive and negative indicators are found
        if has_positive and has_negative:
            contradiction_detected = True
            score = 0.5  # Reset to neutral due to contradiction
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _extract_entities_from_claim(self, claim: str) -> List[str]:
        """Extract entity names from claim text using improved NER.
        
        Args:
            claim: The claim text
            
        Returns:
            List of entity names
        """
        # Use the NERProcessor if available
        if hasattr(self, 'ner_processor') and self.ner_processor:
            return self.ner_processor.extract_flat_entities(claim)
        
        # Fallback to simple extraction
        return self._extract_entities(claim)
        
    def _generate_explanation(self, claim: str, evidence: List[Dict[str, Any]], 
                             score: float, entity_verification: List[Dict[str, Any]] = None) -> str:
        """Generate a detailed explanation for the verification result."""
        # Default empty list if None is provided
        if entity_verification is None:
            entity_verification = []
            
        # Generate an explanation based on the verification score and evidence
        if score >= 0.8:
            reliability = "highly likely to be true"
        elif score >= 0.6:
            reliability = "somewhat likely to be true"
        elif score >= 0.4:
            reliability = "uncertain"
        elif score >= 0.2:
            reliability = "somewhat likely to be false"
        else:
            reliability = "highly likely to be false"
            
        explanation = f"This claim is {reliability}. "
        
        # Add evidence details
        if evidence:
            top_evidence = evidence[:3]  # Use top 3 evidence pieces
            explanation += f"Found {len(evidence)} relevant sources. "
            
            for i, e in enumerate(top_evidence):
                source = e.get('source', 'Unknown source')
                relevance = e.get('relevance_score', 0)
                relevance_desc = "highly relevant" if relevance > 0.7 else "moderately relevant" if relevance > 0.4 else "somewhat relevant"
                explanation += f"Source {i+1}: {source} ({relevance_desc}). "
        else:
            explanation += "No relevant evidence sources were found. "
        
        # Add knowledge graph verification results if available
        if entity_verification:
            explanation += "Knowledge graph verification: "
            verified_count = sum(1 for v in entity_verification if v.get('verified', False))
            total_count = len(entity_verification)
            
            if verified_count > 0:
                explanation += f"{verified_count} out of {total_count} facts were verified. "
                
                # Add details about a top verified fact
                for v in entity_verification:
                    if v.get('verified', False):
                        entity = v.get('subject', '')
                        relation = v.get('predicate', '')
                        object_entity = v.get('object', '')
                        explanation += f"Confirmed: {entity} {relation} {object_entity}. "
                        break
            else:
                explanation += f"None of the {total_count} extracted facts could be verified. "
        
        # Add confidence statement
        if score >= 0.8 or score <= 0.2:
            explanation += "This assessment has high confidence based on the available evidence."
        elif 0.4 <= score <= 0.6:
            explanation += "This assessment has low confidence due to limited or contradictory evidence."
        else:
            explanation += "This assessment has moderate confidence based on the available evidence."
            
        return explanation