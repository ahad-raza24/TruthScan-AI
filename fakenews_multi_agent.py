# Integrated Multi-Agent System for Fake News Detection and Verification
# -----------------------------------------------------------------
import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm
from abc import ABC, abstractmethod
from openai import OpenAI

# Import system configuration
from system_configuration import SystemConfiguration, ProcessingMode

from base_agent import Agent
from name_entity_recognition import NERProcessor
from knowledge_graph import KnowledgeGraphConnector
from news_retrieval_agent import NewsRetrieverAgent
from fact_check_agent import FactCheckAgent
from bias_analysis_agent import BiasDetectionAgent
from llm_agent import LLMDecisionAgent

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt_tab')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FakeNewsSystem")

# Base directory configuration
BASE_DIR = '/Users/ahadraza/Documents/GenAI_Project'
os.makedirs(f'{BASE_DIR}/models', exist_ok=True)
os.makedirs(f'{BASE_DIR}/results', exist_ok=True)

# Model paths (direct paths as provided)
DEBERTA_MODEL_PATH = '/models/deberta_final.pt'
ROBERTA_MODEL_PATH = '/models/roberta_final.pt'


class FakeNewsDetectionSystem:
    """Main system coordinating the multi-agent fact-checking process."""
    
    def __init__(self, model_path: str = None, api_key: str = None, 
             kg_api_key: str = None, lexicon_path: str = None, 
             subjectivity_lexicon_path: str = None, openrouter_api_key: str = None,
             mediastack_key: str = None, gnews_key: str = None,
             enable_kg_verification: bool = True, kg_timeout: float = 6.0,
             config: SystemConfiguration = None):
        """Initialize the fake news detection system.
        
        Args:
            model_path: Path to fine-tuned model
            api_key: News API key
            kg_api_key: Knowledge Graph API key
            lexicon_path: Path to bias lexicon file (JSON format)
            subjectivity_lexicon_path: Path to MPQA subjectivity lexicon (TFF format)
            openrouter_api_key: API key for OpenRouter LLM access
            mediastack_key: MediaStack API key
            gnews_key: GNews API key
            enable_kg_verification: Whether to enable Knowledge Graph verification
            kg_timeout: Maximum time to spend on Knowledge Graph verification
            config: SystemConfiguration instance (optional)
        """
        # Use provided config or create default
        self.config = config or SystemConfiguration()
        
        self.logger = logging.getLogger("FakeNewsSystem")
        self.logger.info("Initializing Fake News Detection System...")
        
        # Initialize shared NER processor
        self.ner_processor = NERProcessor()
        
        # Initialize the knowledge graph first so it can be passed to the fact checker
        self.knowledge_graph = KnowledgeGraphConnector(kg_api_key=kg_api_key)
        
        # Initialize agents with a progress bar
        print("Initializing agents...")
        agents_to_init = ['NewsRetriever', 'FactChecker', 'BiasDetector', 'LLMDecisionMaker']
        agent_pbar = tqdm(total=len(agents_to_init), desc="Initializing Agents")
        
        # Initialize NewsRetrieverAgent with all API keys
        self.news_retriever = NewsRetrieverAgent(
            api_key=api_key,
            mediastack_key=mediastack_key,
            gnews_key=gnews_key
        )
        agent_pbar.update(1)
        
        # Initialize FactCheckAgent with model and optimized knowledge graph settings
        self.fact_checker = FactCheckAgent(
            model_path=model_path, 
            knowledge_graph=self.knowledge_graph,
            enable_kg_verification=enable_kg_verification,
            kg_timeout=kg_timeout
        )
        self.fact_checker.ner_processor = self.ner_processor  # Share NER processor
        agent_pbar.update(1)
        
        # Initialize BiasDetectionAgent with both lexicons
        self.bias_detector = BiasDetectionAgent(
            lexicon_path=lexicon_path, 
            subjectivity_lexicon_path=subjectivity_lexicon_path
        )
        self.bias_detector.ner_processor = self.ner_processor  # Share NER processor
        agent_pbar.update(1)
        
        # Initialize LLMDecisionAgent instead of standard DecisionMakingAgent
        self.decision_maker = LLMDecisionAgent(
            api_key=openrouter_api_key,
            model="deepseek/deepseek-r1-distill-qwen-32b:free",
            site_url="https://fakenewsdetector.org",
            site_name="Fake News Detection System",
            verbose=True
        )
        agent_pbar.update(1)
        
        agent_pbar.close()
        self.logger.info("Fake News Detection System initialized with all agents")
    
    def verify_claim(self, claim: str, source: str = "unknown") -> Dict[str, Any]:
        """Verify a news claim using the multi-agent system."""
        self.logger.info(f"Starting verification of claim: '{claim}'")
        
        # Set up progress bar for verification steps
        verification_steps = ['Retrieving articles', 'Fact checking', 'Bias analysis', 'Making decision']
        verification_pbar = tqdm(total=len(verification_steps), desc="Verification Progress")
        
        # Step 1: Retrieve relevant articles
        verification_pbar.set_description("Retrieving articles")
        evidence_articles = self.news_retriever.search_by_claim(claim)
        verification_pbar.update(1)
        
        # Step 2: Perform fact checking
        verification_pbar.set_description("Fact checking")
        fact_check_input = {
            'claim': claim,
            'evidence': evidence_articles,
            'knowledge_graph': self.knowledge_graph
        }
        fact_check_results = self.fact_checker.process(fact_check_input)
        verification_pbar.update(1)
        
        # Step 3: Analyze bias in the claim
        verification_pbar.set_description("Analyzing bias")
        bias_input = {
            'title': '',
            'text': claim,
            'source': source
        }
        bias_results = self.bias_detector.process(bias_input)
        verification_pbar.update(1)
        
        # Step 4: Make final decision
        verification_pbar.set_description("Making final decision with LLM")
        decision_input = {
            'claim': claim,
            'fact_check_results': fact_check_results,
            'bias_results': bias_results,
            'source': source,
            'evidence': evidence_articles
        }
        
        # Get LLM decision but only log the raw response, don't print it
        llm_decision = self.decision_maker.process(decision_input)
        verification_pbar.update(1)
        verification_pbar.close()
        
        # Only log the decision, don't print it to console
        self.logger.debug(f"LLM decision: {llm_decision}")
        
        # Compile complete results
        results = {
            'claim': claim,
            'source': source,
            'evidence': evidence_articles[:3] if evidence_articles else [],
            'fact_check_results': fact_check_results,
            'bias_analysis': bias_results,
            'decision': llm_decision,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Verification complete. Verdict: {llm_decision.get('verdict', 'unknown')}")
        return results

    def verify_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a full news article.
        
        Args:
            article: Dictionary containing article data
                {
                    'title': article title,
                    'text': article body text,
                    'source': article source,
                    'url': article URL (optional)
                }
                
        Returns:
            Dictionary containing verification results
        """
        title = article.get('title', '')
        text = article.get('text', '')
        source = article.get('source', 'unknown')
        
        # Extract main claim (using title as the main claim)
        main_claim = title
        
        self.logger.info(f"Verifying article: '{title}' from {source}")
        print(f"\nVerifying article: '{title}'")
        
        # Verify the main claim
        claim_verification = self.verify_claim(main_claim, source)
        
        # Analyze bias in the full article with progress bar
        print("Analyzing article bias...")
        bias_input = {
            'title': title,  # Include title for proper bias analysis
            'text': text,
            'source': source
        }
        article_bias = self.bias_detector.process(bias_input)
        
        # Combine results
        results = claim_verification.copy()
        results['article_title'] = title
        results['article_text'] = text[:500] + "..." if len(text) > 500 else text  # Truncate long text
        results['article_bias'] = article_bias
        
        return results

    def verify_user_query(self, user_query: str, source: str = None):
        """
        Verifies a user-provided query using the multi-agent system.
        
        Args:
            user_query (str): The claim or statement to verify
            source (str, optional): Source of the claim if known. Defaults to None.
        
        Returns:
            Dict[str, Any]: Dictionary containing verification results
        """
        # If source is not provided, use a generic value
        if source is None:
            source = "user_query"
        
        print("\n===== VERIFYING USER QUERY =====")
        print(f"QUERY: '{user_query}'")
        if source != "user_query":
            print(f"SOURCE: {source}")
        
        # Create a simple progress bar for the overall process
        with tqdm(total=100, desc="Overall Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as overall_pbar:
            # Step 1: Use the news retriever to extract relevant articles
            print("\n1. EXTRACTING RELEVANT ARTICLES...")
            evidence_articles = self.news_retriever.search_by_claim(user_query)
            overall_pbar.update(25)
            
            # Show the number of articles found
            if evidence_articles:
                print(f"   Found {len(evidence_articles)} relevant articles")
                # Display the top 3 article titles if available
                for i, article in enumerate(evidence_articles[:3]):
                    title = article.get('title', 'Untitled')
                    source = article.get('source', 'Unknown')
                    print(f"   {i+1}. '{title}' from {source}")
            else:
                print("   No relevant articles found")
            
            # Step 2: Use the fact checker with knowledge graph
            print("\n2. CHECKING FACTS AGAINST KNOWLEDGE GRAPH...")
            fact_check_input = {
                'claim': user_query,
                'evidence': evidence_articles,
                'knowledge_graph': self.knowledge_graph
            }
            fact_check_results = self.fact_checker.process(fact_check_input)
            overall_pbar.update(25)
            
            # Display verification score if available
            if 'verification_score' in fact_check_results:
                print(f"   Verification Score: {fact_check_results['verification_score']:.2f}")
            
            # Step 3: Analyze bias in the query
            print("\n3. ANALYZING BIAS...")
            bias_input = {
                'title': '',
                'text': user_query,
                'source': source
            }
            bias_results = self.bias_detector.process(bias_input)
            overall_pbar.update(25)
            
            # Display bias score if available
            if 'bias_score' in bias_results:
                print(f"   Bias Score: {bias_results['bias_score']:.2f}")
            
            # Step 4: Make final decision using LLM
            print("\n4. MAKING FINAL DECISION...")
            decision_input = {
                'claim': user_query,
                'fact_check_results': fact_check_results,
                'bias_results': bias_results,
                'source': source,
                'evidence': evidence_articles
            }
            
            # Show a waiting message for the LLM decision
            print("   Consulting LLM for final verdict (this may take a moment)...")
            llm_decision = self.decision_maker.process(decision_input)
            overall_pbar.update(25)
        
        # Compile results
        results = {
            'claim': user_query,
            'source': source,
            'evidence': evidence_articles[:3] if evidence_articles else [],
            'fact_check_results': fact_check_results,
            'bias_analysis': bias_results,
            'decision': llm_decision,
            'timestamp': datetime.now().isoformat()
        }
        
        # Display final results with clear formatting
        print("\n===== VERIFICATION RESULTS =====")
        print(f"CLAIM: '{user_query}'")
        print(f"VERDICT: {llm_decision['verdict'].upper()}")
        print(f"TRUST SCORE: {llm_decision['trust_score']:.2f}")
        print(f"EXPLANATION: {llm_decision['explanation']}")
        
        # Display bias indicators
        bias_indicators = bias_results.get('bias_indicators', [])
        if bias_indicators:
            print("\nBIAS INDICATORS:")
            for indicator in bias_indicators:
                print(f"- {indicator}")
        
        # Display contributing factors if available
        contributing_factors = llm_decision.get('contributing_factors', [])
        if contributing_factors:
            print("\nCONTRIBUTING FACTORS:")
            for factor in contributing_factors:
                print(f"- {factor}")
        
        # Save the results to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'{BASE_DIR}/results/user_query_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return results

    def run_test_examples(self):
        """Run test examples to demonstrate the system functionality."""
        print("\nRunning test examples...")
        
        # Dictionary to store all test results
        test_results = {
            'claim_verification': [],
            'article_verification': []
        }
        
        # ======= Test claim verification =======
        print("\n========== Testing Claim Verification ==========\n")
        
        test_claims = [
            {
                'claim': 'AI-powered fact-checkers are 95% more accurate than human fact-checkers according to new research.',
                'source': 'technology-insider.org'
            },
            {
                'claim': 'A recent study found that social media algorithms deliberately amplify fake news to increase engagement.',
                'source': 'digital-research-institute.edu'
            },
            {
                'claim': 'Knowledge graphs have been proven to verify news 3 times faster than traditional fact-checking methods.',
                'source': 'ai-research-weekly.com'
            },
            {
                'claim': 'Major tech companies are secretly censoring political content under the guise of fact-checking, according to whistleblowers.',
                'source': 'tech-liberty-network.org'
            }
        ]
        
        # Test each claim with a cleaner output format
        for i, test_case in enumerate(test_claims):
            print(f"\n{i+1}. CLAIM: '{test_case['claim']}'")
            print(f"   SOURCE: {test_case['source']}")
            
            # Process claim with less verbose output
            result = self.verify_claim(test_case['claim'], test_case['source'])
            
            # Print results in a cleaner format
            print(f"   VERDICT: {result['decision']['verdict'].upper()}")
            print(f"   TRUST SCORE: {result['decision']['trust_score']:.2f}")
            print(f"   EXPLANATION: {result['decision']['explanation']}")
            
            # Print bias indicators in a cleaner format
            bias_indicators = result['bias_analysis'].get('bias_indicators', [])
            if bias_indicators:
                print("   BIAS INDICATORS:")
                for indicator in bias_indicators:
                    print(f"   - {indicator}")
            
            test_results['claim_verification'].append(result)
        
        # ======= Test article verification =======
        print("\n========== Testing Article Verification ==========\n")
        
        test_articles = [
            {
                'title': 'New Multi-Agent AI System Achieves 90% Accuracy in Detecting Misinformation',
                'text': 'Researchers have developed a groundbreaking multi-agent AI system that combines large language models with knowledge graphs to detect fake news with unprecedented accuracy. The system, which incorporates specialized agents for fact verification, bias detection, and source credibility analysis, achieved a 90% accuracy rate in preliminary testing. Unlike previous approaches that relied solely on content analysis, this system cross-references claims against multiple verified knowledge sources and evaluates the reliability of news outlets. "This represents a significant advancement in automated fact-checking technology," said lead researcher Dr. Sarah Chen. The team plans to release the system as an open-source toolkit for journalists and social media platforms within the next six months.',
                'source': 'ai-research-journal.org'
            }
        ]
        
        # Test each article with a cleaner output format
        for i, article in enumerate(test_articles):
            print(f"\n{i+1}. ARTICLE: '{article['title']}'")
            print(f"   SOURCE: {article['source']}")
            
            # Print article text preview 
            preview_text = article['text'][:100] + "..." if len(article['text']) > 100 else article['text']
            print(f"   PREVIEW: {preview_text}")
            
            # Process article
            result = self.verify_article(article)
            
            # Print results in a cleaner format
            print(f"   VERDICT: {result['decision']['verdict'].upper()}")
            print(f"   TRUST SCORE: {result['decision']['trust_score']:.2f}")
            print(f"   EXPLANATION: {result['decision']['explanation']}")
            print(f"   BIAS SCORE: {result['article_bias']['bias_score']:.2f}")
            
            # Print bias indicators in a cleaner format
            bias_indicators = result['article_bias'].get('bias_indicators', [])
            if bias_indicators:
                print("   BIAS INDICATORS:")
                for indicator in bias_indicators:
                    print(f"   - {indicator}")
            
            test_results['article_verification'].append(result)
        
        # Save test results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'{BASE_DIR}/results/test_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        return test_results