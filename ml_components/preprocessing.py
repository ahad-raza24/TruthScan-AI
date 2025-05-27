# Standard Library
import os
import re
import json
import glob
import pickle
import warnings
import multiprocessing
from typing import Dict, List, Union, Tuple, Optional, Any
from collections import Counter

# Data Handling and Computation
import numpy as np
import pandas as pd
from scipy import sparse

# Progress and Utility
from tqdm import tqdm

# Natural Language Processing (NLP)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
from textstat import textstat
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer

# Transformers and Deep Learning
import torch
from transformers import AutoTokenizer, AutoModel

# Machine Learning Utilities
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Check if GPU is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Enhanced text processor with more sophisticated NLP features
class TextProcessor:
    """
    Advanced text preprocessing and linguistic analysis toolkit.
    
    Handles text cleaning, normalization, and extraction of linguistic features
    including sentiment, subjectivity, clickbait indicators, and complexity metrics.
    Supports both NLTK and SpaCy backends for NLP operations.
    """
    
    def __init__(self, use_spacy=False):
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        
        # Load SpaCy model for more advanced NLP if requested
        self.use_spacy = use_spacy
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                print("Downloading SpaCy model...")
                os.system('python -m spacy download en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
        
        # Enhanced lexicon for feature extraction
        self.positive_words = set([
            'good', 'great', 'excellent', 'positive', 'nice', 'correct', 'true', 'authentic', 
            'legitimate', 'factual', 'accurate', 'verified', 'proven', 'confirmed', 'valid'
        ])
        
        self.negative_words = set([
            'bad', 'awful', 'terrible', 'negative', 'wrong', 'false', 'fake', 'hoax', 
            'misleading', 'deceptive', 'fraudulent', 'unverified', 'untrue', 'bogus', 'fabricated'
        ])
        
        self.subjective_words = set([
            'think', 'believe', 'feel', 'opinion', 'seems', 'appears', 'maybe', 'perhaps', 
            'probably', 'possibly', 'likely', 'allegedly', 'supposedly', 'apparently', 'rumors', 
            'claiming', 'according', 'sources', 'reports', 'suggests', 'speculates', 'might',
            'could', 'would', 'should', 'must', 'may', 'can', 'will'
        ])
        
        # Clickbait indicators
        self.clickbait_words = set([
            'shocking', 'unbelievable', 'amazing', 'surprising', 'mind-blowing', 'jaw-dropping',
            'incredible', 'you won\'t believe', 'never seen before', 'won\'t believe', 'shocked',
            'this is why', 'secret', 'trick', 'easy way', 'simple trick', 'how to', 'must see',
            'what happens next', 'this happened', 'will shock you', 'simple way', 'best ever'
        ])
        
        # Emotional words
        self.emotional_words = set([
            'angry', 'sad', 'happy', 'joy', 'fear', 'scary', 'terrifying', 'horrifying', 'exciting',
            'thrilling', 'devastating', 'outrageous', 'scandalous', 'shocking', 'distressing',
            'heartbreaking', 'infuriating', 'frustrated', 'thrilled', 'disgusted', 'appalled',
            'worried', 'concerned', 'anxious', 'nervous', 'hopeful', 'optimistic', 'pessimistic'
        ])
        
        # Exaggeration indicators
        self.exaggeration_words = set([
            'always', 'never', 'everyone', 'nobody', 'all', 'none', 'every', 'only', 'best', 'worst',
            'most', 'least', 'greatest', 'perfect', 'complete', 'absolutely', 'totally', 'entirely',
            'literally', 'exactly', 'definitely', 'completely', 'utterly', 'purely', 'thoroughly'
        ])
    
    def remove_urls_and_tags(self, text: str) -> str:
        """Remove URLs, HTML tags, and special characters from text"""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters but keep punctuation for sentence analysis
        text = re.sub(r'[^\w\s\.\,\!\?\"\'\-\:]', '', text)
        # Replace digits with space (keep punctuation structure)
        text = re.sub(r'\d+', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by converting to lowercase and expanding contractions"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Expanded contractions list
        contractions = {
            "n't": " not",
            "'s": " is",
            "'m": " am",
            "'ll": " will",
            "'d": " would",
            "'ve": " have",
            "'re": " are",
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "isn't": "is not",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }
        
        # Replace contractions
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text while preserving structure"""
        if not isinstance(text, str):
            return ""
        
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def preprocess_text(self, text: str, remove_stops: bool = False, preserve_sentence_structure: bool = True) -> str:
        """Complete text preprocessing pipeline with sentence structure preservation option"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
            
        # Remove URLs and HTML tags
        text = self.remove_urls_and_tags(text)
        
        # Normalize text
        text = self.normalize_text(text)
        
        if preserve_sentence_structure:
            # Process each sentence separately to maintain structure
            sentences = sent_tokenize(text)
            processed_sentences = []
            
            for sentence in sentences:
                if remove_stops:
                    tokens = word_tokenize(sentence)
                    filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
                    processed_sentences.append(' '.join(filtered_tokens))
                else: 
                    processed_sentences.append(sentence)
                    
            return ' '.join(processed_sentences)
        else:
            # Standard processing without sentence preservation
            if remove_stops:
                return self.remove_stopwords(text)
            else:
                return text
    
    def get_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score using lexicon approach"""
        if not isinstance(text, str):
            return 0
        
        tokens = word_tokenize(text.lower())
        pos_count = sum(1 for token in tokens if token in self.positive_words)
        neg_count = sum(1 for token in tokens if token in self.negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0
        return (pos_count - neg_count) / total
    
    def get_subjectivity_score(self, text: str) -> float:
        """Calculate subjectivity score using lexicon approach"""
        if not isinstance(text, str):
            return 0
        
        tokens = word_tokenize(text.lower())
        subj_count = sum(1 for token in tokens if token in self.subjective_words)
        
        if len(tokens) == 0:
            return 0
        return subj_count / len(tokens)
    
    def get_clickbait_score(self, text: str) -> float:
        """Calculate clickbait score using lexicon approach"""
        if not isinstance(text, str):
            return 0
            
        # Check for full phrases first
        clickbait_phrases = ['you won\'t believe', 'this is why', 'what happens next', 'will shock you']
        phrase_score = 0
        for phrase in clickbait_phrases:
            if phrase in text.lower():
                phrase_score += 1
        
        # Check for individual words
        tokens = word_tokenize(text.lower())
        clickbait_count = sum(1 for token in tokens if token in self.clickbait_words)
        
        if len(tokens) == 0:
            return 0
            
        # Combine phrase and word scores
        return (clickbait_count + (phrase_score * 2)) / (len(tokens) + 0.1)
    
    def get_emotional_score(self, text: str) -> float:
        """Calculate emotional language score"""
        if not isinstance(text, str):
            return 0
            
        tokens = word_tokenize(text.lower())
        emotional_count = sum(1 for token in tokens if token in self.emotional_words)
        
        if len(tokens) == 0:
            return 0
        return emotional_count / len(tokens)
    
    def get_exaggeration_score(self, text: str) -> float:
        """Calculate exaggeration score based on extreme language"""
        if not isinstance(text, str):
            return 0
            
        tokens = word_tokenize(text.lower())
        exaggeration_count = sum(1 for token in tokens if token in self.exaggeration_words)
        
        if len(tokens) == 0:
            return 0
        return exaggeration_count / len(tokens)
    
    def calculate_complexity(self, text: str) -> Dict[str, float]:
        """Calculate multiple linguistic complexity metrics"""
        if not isinstance(text, str) or len(text) == 0:
            return {
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'readability_score': 0,
                'unique_word_ratio': 0
            }
        
        # Tokenize text
        tokens = word_tokenize(text)
        if not tokens:
            return {
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'readability_score': 0,
                'unique_word_ratio': 0
            }
        
        # Tokenize sentences
        sentences = sent_tokenize(text)
        
        # Average word length
        avg_word_length = sum(len(word) for word in tokens) / len(tokens)
        
        # Average sentence length
        if len(sentences) > 0:
            avg_sentence_length = len(tokens) / len(sentences)
        else:
            avg_sentence_length = 0
        
        # Readability score (using textstat)
        try:
            readability_score = textstat.flesch_kincaid_grade(text) if len(text) > 100 else 0
        except:
            readability_score = 0
        
        # Unique word ratio
        unique_words = len(set([w.lower() for w in tokens]))
        unique_word_ratio = unique_words / len(tokens) if tokens else 0
        
        return {
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'readability_score': readability_score,
            'unique_word_ratio': unique_word_ratio
        }
    
    def extract_linguistic_inconsistency(self, text: str) -> float:
        """Detect linguistic inconsistencies that may indicate fake news"""
        if not isinstance(text, str) or len(text) < 20:
            return 0
            
        # Use spaCy for inconsistency detection if available
        if self.use_spacy:
            doc = self.nlp(text)
            
            # Check for inconsistent tense usage (simplified)
            verb_tenses = [token.tag_ for token in doc if token.pos_ == 'VERB']
            tense_counter = Counter(verb_tenses)
            
            # If multiple tenses are used, calculate inconsistency
            if len(tense_counter) > 1:
                most_common_tense, most_common_count = tense_counter.most_common(1)[0]
                inconsistency = 1 - (most_common_count / len(verb_tenses) if verb_tenses else 0)
                return inconsistency
            return 0
        else:
            # Fallback for when spaCy isn't available
            # Check for sentence structure inconsistencies
            sentences = sent_tokenize(text)
            if len(sentences) <= 1:
                return 0
                
            # Calculate variance in sentence length as a proxy for inconsistency
            sent_lengths = [len(word_tokenize(s)) for s in sentences]
            mean_length = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_length) ** 2 for x in sent_lengths) / len(sent_lengths)
            # Normalize to 0-1 range (higher variance = more inconsistency)
            return min(1.0, variance / 100)

def simple_oversample(minority_features, minority_labels, target_count):
    """Simple oversampling by random duplication with small random noise"""
    current_count = len(minority_labels)
    samples_needed = target_count - current_count
    
    if samples_needed <= 0:
        return minority_features, minority_labels
    
    # Indices to duplicate (with replacement)
    indices = np.random.choice(current_count, samples_needed, replace=True)
    
    # Create new samples with small random noise
    new_features = minority_features[indices].copy()
    if sparse.issparse(new_features):
        # For sparse matrices, we need to handle differently
        new_features = new_features.toarray()
        new_features += np.random.normal(0, 0.01, new_features.shape)
        new_features = sparse.csr_matrix(new_features)
    else:
        # Regular numpy arrays
        new_features += np.random.normal(0, 0.01, new_features.shape)
    
    # Combine original and new samples
    augmented_features = sparse.vstack([minority_features, new_features]) if sparse.issparse(minority_features) else np.vstack([minority_features, new_features])
    augmented_labels = np.concatenate([minority_labels, minority_labels[indices]])
    
    return augmented_features, augmented_labels

class DatasetLoader:
    """
    Kaggle-optimized dataset loader for fake news datasets.
    
    Loads and standardizes LIAR and ISOT datasets from Kaggle input directories.
    Handles data format normalization and creates unified dataset structure
    with consistent column naming and metadata extraction.
    """
    
    def __init__(self, base_dir: str = '/kaggle/working'):
        """Initialize with the base directory for data storage"""
        self.base_dir = base_dir
        
        # Fixed Kaggle input paths for our two datasets
        self.liar_input_dir = '/kaggle/input/liar-dataset'
        self.isot_input_dir = '/kaggle/input/isot-fake-news-dataset'
        
        # Create directories if they don't exist
        os.makedirs(f'{self.base_dir}/data/processed', exist_ok=True)
        os.makedirs(f'{self.base_dir}/data/embeddings', exist_ok=True)
    
    def load_liar_dataset(self) -> pd.DataFrame:
        """Load and preprocess the LIAR dataset from Kaggle input directory"""
        print("Loading LIAR dataset...")
        
        # Check if dataset exists in Kaggle inputs
        if not os.path.exists(self.liar_input_dir):
            print("LIAR dataset not found in Kaggle input directory.")
            return pd.DataFrame()
        
        try:
            # Load train, validation, and test sets
            liar_train = pd.read_csv(f'{self.liar_input_dir}/train.tsv', sep='\t', 
                                    header=None, 
                                    names=['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
                                        'state_info', 'party_affiliation', 'barely_true_counts', 
                                        'false_counts', 'half_true_counts', 'mostly_true_counts', 
                                        'pants_on_fire_counts', 'context'])
            
            liar_val = pd.read_csv(f'{self.liar_input_dir}/valid.tsv', sep='\t', 
                                header=None, 
                                names=['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
                                        'state_info', 'party_affiliation', 'barely_true_counts', 
                                        'false_counts', 'half_true_counts', 'mostly_true_counts', 
                                        'pants_on_fire_counts', 'context'])
            
            liar_test = pd.read_csv(f'{self.liar_input_dir}/test.tsv', sep='\t', 
                                header=None, 
                                names=['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
                                        'state_info', 'party_affiliation', 'barely_true_counts', 
                                        'false_counts', 'half_true_counts', 'mostly_true_counts', 
                                        'pants_on_fire_counts', 'context'])
            
            # Combine all datasets
            liar_df = pd.concat([liar_train, liar_val, liar_test], ignore_index=True)
            
            # Rename columns to standardized names
            liar_df = liar_df.rename(columns={
                'statement': 'text',
                'context': 'context_info',
                'speaker': 'source'
            })
            
            # Add source dataset column
            liar_df['dataset_source'] = 'liar'
            
            # Add title column (LIAR doesn't have titles)
            liar_df['title'] = ''
            
            # Add speaker credibility metrics based on historical truthfulness
            liar_df['source_credibility'] = (
                (liar_df['mostly_true_counts'] + liar_df['half_true_counts']) / 
                (liar_df['mostly_true_counts'] + liar_df['half_true_counts'] + 
                liar_df['false_counts'] + liar_df['barely_true_counts'] + 
                liar_df['pants_on_fire_counts']).clip(lower=1)
            )
            
            # Add political party flag (useful for bias detection)
            liar_df['political_affiliation'] = liar_df['party_affiliation'].fillna('')
            
            # Select and reorder columns
            liar_df = liar_df[['id', 'title', 'text', 'label', 'source', 'context_info', 
                            'subject', 'dataset_source', 'source_credibility', 'political_affiliation']]
            
            print(f"LIAR dataset loaded with {len(liar_df)} samples.")
            return liar_df
            
        except Exception as e:
            print(f"Error loading LIAR dataset: {e}")
            return pd.DataFrame()
    
    def load_isot_dataset(self) -> pd.DataFrame:
        """Load ISOT Fake News dataset from Kaggle input directory"""
        print("Loading ISOT Fake News dataset...")
        
        # Check if dataset exists in Kaggle inputs
        if not os.path.exists(self.isot_input_dir):
            print("ISOT dataset not found in Kaggle input directory.")
            return pd.DataFrame()
        
        try:
            # Load True.csv and Fake.csv (adjust filenames if needed)
            fake_df = pd.read_csv(f'{self.isot_input_dir}/Fake.csv')
            true_df = pd.read_csv(f'{self.isot_input_dir}/True.csv')
            
            # Add label column
            fake_df['label'] = 'fake'
            true_df['label'] = 'real'
            
            # Combine datasets
            isot_df = pd.concat([fake_df, true_df], ignore_index=True)
            
            # Standardize column names
            # Note: In ISOT, the columns should already be 'text' and 'title'
            # This conditional rename handles possible variations
            column_map = {}
            if 'text' not in isot_df.columns and 'content' in isot_df.columns:
                column_map['content'] = 'text'
            if 'title' not in isot_df.columns and 'headline' in isot_df.columns:
                column_map['headline'] = 'title'
                
            if column_map:
                isot_df = isot_df.rename(columns=column_map)
            
            # Add missing columns
            isot_df['id'] = isot_df.index.astype(str)
            isot_df['dataset_source'] = 'isot'
            isot_df['context_info'] = ''
            
            # Handle subject and source columns
            if 'subject' in isot_df.columns:
                isot_df['subject'] = isot_df['subject'].fillna('')
                # Use subject as source if no source column
                if 'source' not in isot_df.columns:
                    isot_df['source'] = isot_df['subject']
            else:
                isot_df['subject'] = ''
                if 'source' not in isot_df.columns:
                    isot_df['source'] = ''
            
            # Add credibility and political affiliation
            isot_df['source_credibility'] = 0.5  # Default credibility score
            isot_df['political_affiliation'] = ''
            
            # Ensure all required columns exist
            required_columns = ['id', 'title', 'text', 'label', 'source', 'context_info', 
                               'subject', 'dataset_source', 'source_credibility', 'political_affiliation']
            
            # Create a new DataFrame with just the required columns
            result_df = pd.DataFrame()
            for col in required_columns:
                if col in isot_df.columns:
                    result_df[col] = isot_df[col]
                else:
                    result_df[col] = ''  # Add empty column if missing
            
            print(f"ISOT dataset loaded with {len(result_df)} samples.")
            return result_df
                
        except Exception as e:
            print(f"Error loading ISOT dataset from Kaggle: {e}")
            return pd.DataFrame()
    
    def load_datasets(self) -> pd.DataFrame:
        """Load and combine both datasets"""
        print("Loading LIAR and ISOT datasets...")
        
        datasets = []
        
        # 1. Load LIAR dataset
        liar_df = self.load_liar_dataset()
        if not liar_df.empty:
            datasets.append(liar_df)
        
        # 2. Load ISOT dataset
        isot_df = self.load_isot_dataset()
        if not isot_df.empty:
            datasets.append(isot_df)
        
        # Combine datasets
        if datasets:
            combined_df = pd.concat(datasets, ignore_index=True)
            print(f"Combined dataset created with {len(combined_df)} samples from {len(datasets)} sources.")
            
            # Display dataset distribution
            dataset_counts = combined_df['dataset_source'].value_counts()
            print("Dataset distribution:")
            for dataset, count in dataset_counts.items():
                print(f"  - {dataset}: {count} samples")
            
            return combined_df
        else:
            print("Error: No datasets could be loaded.")
            return pd.DataFrame()

class EnhancedFeatureExtractor:
    """
    GPU-accelerated feature extraction and embedding generation system.
    
    Combines multiple feature extraction approaches:
    - Transformer-based embeddings (DeBERTa, RoBERTa)
    - TF-IDF vectorization with n-grams
    - Linguistic complexity analysis
    - Semantic and syntactic feature extraction
    """
    
    def __init__(self, use_spacy=False, device: torch.device = DEVICE):
        """Initialize with the device for computation"""
        self.device = device
        self.text_processor = TextProcessor(use_spacy=use_spacy)
        
        # Choose whether to use SentenceTransformer or Hugging Face models
        self.use_huggingface = True
        
        if self.use_huggingface:
            # Load DeBERTa model
            print(f"Loading DeBERTa model on {self.device}...")
            self.deberta_model_name = "microsoft/deberta-v3-small"
            self.deberta_tokenizer = AutoTokenizer.from_pretrained(self.deberta_model_name)
            self.deberta_model = AutoModel.from_pretrained(self.deberta_model_name)
            self.deberta_model = self.deberta_model.to(self.device)
            
            # Load RoBERTa model
            print(f"Loading RoBERTa model on {self.device}...")
            self.roberta_model_name = "roberta-base"
            self.roberta_tokenizer = AutoTokenizer.from_pretrained(self.roberta_model_name)
            self.roberta_model = AutoModel.from_pretrained(self.roberta_model_name)
            self.roberta_model = self.roberta_model.to(self.device)
        else:
            # Load SentenceTransformer model
            print(f"Loading sentence transformer model on {self.device}...")
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            if self.device.type == 'cuda':
                self.sentence_transformer = self.sentence_transformer.to(self.device)
    
    def extract_linguistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced linguistic features from text data with progress bars"""
        print("Extracting linguistic features...")
        
        # Basic text metrics
        df['text_length'] = df['clean_text'].str.len()
        df['word_count'] = df['clean_text'].str.split().str.len()
        df['title_length'] = df['clean_title'].str.len()
        df['title_word_count'] = df['clean_title'].str.split().str.len()
        
        total_samples = len(df)
        print(f"Processing {total_samples} samples...")
        
        # Process text content with progress bars
        print("Calculating sentiment scores...")
        df['sentiment_score'] = [
            self.text_processor.get_sentiment_score(text)
            for text in tqdm(df['clean_text'], total=total_samples, desc="Sentiment analysis", ncols=100)
        ]
        
        print("Calculating subjectivity scores...")
        df['subjectivity_score'] = [
            self.text_processor.get_subjectivity_score(text)
            for text in tqdm(df['clean_text'], total=total_samples, desc="Subjectivity analysis", ncols=100)
        ]
        
        print("Calculating clickbait scores...")
        df['clickbait_score'] = [
            self.text_processor.get_clickbait_score(text)
            for text in tqdm(df['clean_title'], total=total_samples, desc="Clickbait detection", ncols=100)
        ]
        
        print("Calculating emotional language scores...")
        df['emotional_score'] = [
            self.text_processor.get_emotional_score(text)
            for text in tqdm(df['clean_text'], total=total_samples, desc="Emotion analysis", ncols=100)
        ]
        
        print("Calculating exaggeration scores...")
        df['exaggeration_score'] = [
            self.text_processor.get_exaggeration_score(text)
            for text in tqdm(df['clean_text'], total=total_samples, desc="Exaggeration detection", ncols=100)
        ]
        
        print("Calculating linguistic inconsistency scores...")
        df['inconsistency_score'] = [
            self.text_processor.extract_linguistic_inconsistency(text)
            for text in tqdm(df['clean_text'], total=total_samples, desc="Inconsistency detection", ncols=100)
        ]
        
        # Process complexity metrics (returns a dictionary of multiple metrics)
        print("Calculating complexity metrics...")
        complexity_features = [
            self.text_processor.calculate_complexity(text)
            for text in tqdm(df['clean_text'], total=total_samples, desc="Complexity analysis", ncols=100)
        ]
        
        # Convert list of dictionaries to DataFrame for easier processing
        complexity_df = pd.DataFrame(complexity_features)
        
        # Extract complexity features
        df['avg_word_length'] = complexity_df['avg_word_length']
        df['avg_sentence_length'] = complexity_df['avg_sentence_length']
        df['readability_score'] = complexity_df['readability_score']
        df['unique_word_ratio'] = complexity_df['unique_word_ratio']
        
        # Calculate title-specific scores with progress bar
        print("Analyzing titles...")
        title_features = []
        for title in tqdm(df['clean_title'], total=total_samples, desc="Title analysis", ncols=100):
            title_features.append({
                'sentiment': self.text_processor.get_sentiment_score(title),
                'subjectivity': self.text_processor.get_subjectivity_score(title)
            })
        
        title_df = pd.DataFrame(title_features)
        df['title_sentiment_score'] = title_df['sentiment']
        df['title_subjectivity_score'] = title_df['subjectivity']
        
        # Title-content discrepancy score
        df['sentiment_discrepancy'] = (df['title_sentiment_score'] - df['sentiment_score']).abs()
        
        return df

    def extract_tfidf_features(self, train_texts, val_texts=None, test_texts=None):
        """Extract TF-IDF features with progress tracking"""
        print("Extracting TF-IDF features...")
        
        # Create TF-IDF vectorizer with n-grams
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.7,
            strip_accents='unicode',
            ngram_range=(1, 2)
        )
        
        # Fit and transform on training data (with progress bar)
        print(f"Processing {len(train_texts)} training samples...")
        train_texts_with_progress = tqdm(train_texts, desc="TF-IDF fitting", total=len(train_texts), ncols=100)
        train_tfidf = vectorizer.fit_transform(train_texts_with_progress)
        
        # Transform validation data if provided
        if val_texts is not None:
            print(f"Processing {len(val_texts)} validation samples...")
            val_texts_with_progress = tqdm(val_texts, desc="Validation vectorization", total=len(val_texts), ncols=100)
            val_tfidf = vectorizer.transform(val_texts_with_progress)
        else:
            val_tfidf = None
        
        # Transform test data if provided
        if test_texts is not None:
            print(f"Processing {len(test_texts)} test samples...")
            test_texts_with_progress = tqdm(test_texts, desc="Test vectorization", total=len(test_texts), ncols=100)
            test_tfidf = vectorizer.transform(test_texts_with_progress)
        else:
            test_tfidf = None
        
        # Report feature shape
        print(f"TF-IDF features: {train_tfidf.shape[1]} dimensions")
        
        return {
            'vectorizer': vectorizer,
            'train_tfidf': train_tfidf,
            'val_tfidf': val_tfidf,
            'test_tfidf': test_tfidf
        }
    
    def generate_phraser_features(self, texts):
        """Generate phrase-based features using Gensim Phrases"""
        print("Generating phrase features...")
        
        # Tokenize sentences
        sentences = [text.split() for text in texts]
        
        # Train a bigram detector
        bigram = Phrases(sentences, min_count=5, threshold=10)
        bigram_phraser = Phraser(bigram)
        
        # Apply the phraser to transform sentences
        transformed_sentences = [bigram_phraser[sentence] for sentence in sentences]
        
        # Count the number of phrases in each text
        phrase_counts = []
        for original, transformed in zip(sentences, transformed_sentences):
            # Count bigrams (phrases that were joined)
            phrase_count = len(transformed) - len(original)
            phrase_counts.append(abs(phrase_count))  # abs because phrase count should be positive
        
        return phrase_counts
    
    def create_embeddings_with_huggingface(self, texts: List[str], batch_size: int = 8, model_type: str = 'deberta') -> np.ndarray:
        """Create embeddings using Hugging Face models with GPU acceleration"""
        if model_type == 'deberta':
            model_name = self.deberta_model_name
            tokenizer = self.deberta_tokenizer
            model = self.deberta_model
        elif model_type == 'roberta':
            model_name = self.roberta_model_name
            tokenizer = self.roberta_tokenizer
            model = self.roberta_model
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        print(f"Creating embeddings with {model_name}...")
        embeddings = []
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i+batch_size]
                
                # Tokenize the batch
                inputs = tokenizer(
                    batch,
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model outputs
                outputs = model(**inputs)
                
                # Use mean of last hidden state as embeddings
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        return np.vstack(embeddings)
    
    def create_embeddings_with_sentencetransformer(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Create embeddings using SentenceTransformer with GPU acceleration"""
        print("Creating embeddings with SentenceTransformer...")
        embeddings = []
        
        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            # Use GPU acceleration if available
            batch_embeddings = self.sentence_transformer.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        return np.vstack(embeddings)
    
    def create_embeddings(self, texts: List[str], batch_size: int = 16) -> Dict[str, np.ndarray]:
        """Create embeddings using both DeBERTa and RoBERTa models"""
        if self.use_huggingface:
            deberta_embeddings = self.create_embeddings_with_huggingface(texts, batch_size, model_type='deberta')
            roberta_embeddings = self.create_embeddings_with_huggingface(texts, batch_size, model_type='roberta')
            return {
                'deberta': deberta_embeddings,
                'roberta': roberta_embeddings
            }
        else:
            sentencetransformer_embeddings = self.create_embeddings_with_sentencetransformer(texts, batch_size)
            return {
                'sentencetransformer': sentencetransformer_embeddings
            }

class EnhancedDataProcessor:
     """
    Main orchestrator for the complete data preprocessing pipeline.
    
    Coordinates dataset loading, text preprocessing, feature extraction,
    label standardization, and data augmentation. Produces ready-to-use
    feature stores for machine learning model training and evaluation.
    """
    
    def __init__(self, base_dir: str = '/kaggle/working',
                 use_spacy: bool = False, use_oversampling: bool = True):
        """Initialize the data processor with component classes"""
        self.base_dir = base_dir
        self.loader = DatasetLoader(base_dir)  # Updated to match new DatasetLoader
        self.text_processor = TextProcessor(use_spacy=use_spacy)
        self.feature_extractor = EnhancedFeatureExtractor(use_spacy=use_spacy)
        self.use_oversampling = use_oversampling
        
        # Label standardization maps
        self.liar_label_map = {
            'pants-fire': 0, 
            'false': 0, 
            'barely-true': 0.25, 
            'half-true': 0.5, 
            'mostly-true': 0.75, 
            'true': 1
        }
        
        # General label map for all datasets
        self.general_label_map = {
            'fake': 0,
            'false': 0,
            'pants-fire': 0,
            'barely-true': 0, 
            'half-true': 0.5,
            'mostly-true': 1,
            'true': 1,
            'real': 1
        }
    
    def standardize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize labels across datasets with better handling of edge cases"""
        # Create binary and fine-grained labels
        df['binary_label'] = np.nan
        df['fine_grained_label'] = df['label']
        
        # Apply mapping based on dataset source
        liar_mask = df['dataset_source'] == 'liar'
        isot_mask = df['dataset_source'] == 'isot'
        
        # Apply dataset-specific mapping first
        df.loc[liar_mask, 'binary_label'] = df.loc[liar_mask, 'label'].map(self.liar_label_map)
        
        # For other datasets or missing values, apply general mapping
        unknown_mask = df['binary_label'].isna()
        if unknown_mask.any():
            df.loc[unknown_mask, 'binary_label'] = df.loc[unknown_mask, 'label'].map(self.general_label_map)
        
        # Fill any remaining NaN values with default value (0.5 - uncertain)
        df['binary_label'] = df['binary_label'].fillna(0.5)
        
        # Convert to binary (0 or 1) based on threshold
        df['binary_label'] = (df['binary_label'] >= 0.5).astype(int)
        return df
    
    def apply_data_augmentation(self, texts, labels):
        """Apply Easy Data Augmentation (EDA) techniques to increase dataset diversity"""
        print("Applying data augmentation...")
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # Original text
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Only augment longer texts
            if len(text.split()) > 10:
                # Synonym replacement (simplified version)
                words = text.split()
                n_words = max(1, int(len(words) * 0.1))  # Replace 10% of words
                random_indices = np.random.choice(range(len(words)), n_words, replace=False)
                
                # Create augmented text by removing words (simplified augmentation)
                augmented_text = ' '.join([w for i, w in enumerate(words) if i not in random_indices])
                if len(augmented_text) > 0:
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
    
    def process_datasets(self) -> Dict[str, Any]:
        """Complete data processing pipeline with enhanced features"""
        # Load datasets (not download since they're already in Kaggle)
        combined_df = self.loader.load_datasets()  # Use the new method name
        
        print(f"Combined dataset size: {len(combined_df)} samples")
        
        # Check if we have a valid dataset
        if combined_df.empty:
            print("Error: No data could be loaded. Please check the dataset paths.")
            return {}
        
        # Preprocess text
        print("Preprocessing text...")
        combined_df['clean_title'] = combined_df['title'].apply(
            lambda x: self.text_processor.preprocess_text(x, preserve_sentence_structure=True)
        )
        combined_df['clean_text'] = combined_df['text'].apply(
            lambda x: self.text_processor.preprocess_text(x, preserve_sentence_structure=True)
        )
        
        # For longer texts, remove stopwords to reduce dimensionality
        long_text_mask = combined_df['clean_text'].str.split().str.len() > 50
        combined_df.loc[long_text_mask, 'clean_text_no_stops'] = combined_df.loc[long_text_mask, 'clean_text'].apply(
            lambda x: self.text_processor.remove_stopwords(x)
        )
        
        # Extract enhanced linguistic features
        print("Extracting linguistic features...")
        combined_df = self.feature_extractor.extract_linguistic_features(combined_df)
        
        # Standardize labels
        print("Standardizing labels...")
        combined_df = self.standardize_labels(combined_df)
        
        # Create combined text field for embedding
        combined_df['combined_text'] = combined_df['clean_title'] + ' ' + combined_df['clean_text']
        
        # Train/Validation/Test split with stratification
        print("Splitting dataset...")
        train_df, temp_df = train_test_split(
            combined_df, test_size=0.3, stratify=combined_df['binary_label'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['binary_label'], random_state=42
        )
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        # Apply data augmentation if enabled (only to minority class in training set)
        if self.use_oversampling:
            print("Applying simple oversampling...")
            # Prepare data for TF-IDF
            X_text = train_df['combined_text'].tolist()
            y = train_df['binary_label'].values
            
            # Extract TF-IDF features
            tfidf_results = self.feature_extractor.extract_tfidf_features([text for text in X_text])
            X_tfidf = tfidf_results['train_tfidf']
            
            # Get class distributions
            class_counts = np.bincount(y)
            target_count = max(class_counts)
            
            # Identify minority class
            minority_class = 0 if class_counts[0] < class_counts[1] else 1
            minority_mask = y == minority_class
            
            # Apply oversampling only to minority class
            minority_features = X_tfidf[minority_mask]
            minority_labels = y[minority_mask]
            
            # Apply our simple oversampling function
            oversampled_features, oversampled_labels = simple_oversample(
                minority_features, minority_labels, target_count
            )
            
            print(f"After oversampling: {len(oversampled_labels)} minority samples (target: {target_count})")
            
            # For text augmentation (not connected to the TF-IDF features above but useful for text generation)
            minority_texts = train_df[train_df['binary_label'] == minority_class]['combined_text'].tolist()
            minority_labels = [minority_class] * len(minority_texts)
            
            # Apply text-specific augmentation to the minority class
            aug_texts, aug_labels = self.apply_data_augmentation(minority_texts, minority_labels)
            
            print(f"Text augmentation: {len(aug_texts)} texts from original {len(minority_texts)}")
        
        # Save processed datasets
        train_df.to_csv(f'{self.base_dir}/data/processed/train.csv', index=False)
        val_df.to_csv(f'{self.base_dir}/data/processed/validation.csv', index=False)
        test_df.to_csv(f'{self.base_dir}/data/processed/test.csv', index=False)
        
        # Create embeddings
        print("Creating embeddings for train set...")
        train_embeddings = self.feature_extractor.create_embeddings(train_df['combined_text'].tolist())
        
        print("Creating embeddings for validation set...")
        val_embeddings = self.feature_extractor.create_embeddings(val_df['combined_text'].tolist())
        
        print("Creating embeddings for test set...")
        test_embeddings = self.feature_extractor.create_embeddings(test_df['combined_text'].tolist())
        
        # Save embeddings
        # DeBERTa embeddings
        np.save(f'{self.base_dir}/data/embeddings/train_deberta_embeddings.npy', train_embeddings['deberta'])
        np.save(f'{self.base_dir}/data/embeddings/val_deberta_embeddings.npy', val_embeddings['deberta'])
        np.save(f'{self.base_dir}/data/embeddings/test_deberta_embeddings.npy', test_embeddings['deberta'])
        
        # RoBERTa embeddings
        np.save(f'{self.base_dir}/data/embeddings/train_roberta_embeddings.npy', train_embeddings['roberta'])
        np.save(f'{self.base_dir}/data/embeddings/val_roberta_embeddings.npy', val_embeddings['roberta'])
        np.save(f'{self.base_dir}/data/embeddings/test_roberta_embeddings.npy', test_embeddings['roberta'])
        
        # Generate TF-IDF features
        tfidf_results = self.feature_extractor.extract_tfidf_features(
            train_df['combined_text'].tolist(),
            val_df['combined_text'].tolist(),
            test_df['combined_text'].tolist()
        )
        
        # Save TF-IDF vectorizer for future use
        with open(f'{self.base_dir}/data/processed/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_results['vectorizer'], f)
        
        # Save TF-IDF features
        sparse.save_npz(f'{self.base_dir}/data/processed/train_tfidf.npz', tfidf_results['train_tfidf'])
        sparse.save_npz(f'{self.base_dir}/data/processed/val_tfidf.npz', tfidf_results['val_tfidf'])
        sparse.save_npz(f'{self.base_dir}/data/processed/test_tfidf.npz', tfidf_results['test_tfidf'])
        
        # Create feature stores
        print("Creating feature stores...")
        
        # Linguistic features (non-embedding features)
        feature_cols = [
            'text_length', 'word_count', 'sentiment_score', 'subjectivity_score', 
            'clickbait_score', 'emotional_score', 'exaggeration_score', 'inconsistency_score',
            'avg_word_length', 'avg_sentence_length', 'readability_score', 'unique_word_ratio',
            'title_sentiment_score', 'title_subjectivity_score', 'sentiment_discrepancy',
            'source_credibility'
        ]
        
        # Train feature store
        train_feature_store = {
            'metadata': train_df[['id', 'title', 'text', 'label', 'source', 'dataset_source']],
            'features': train_df[['id'] + feature_cols + ['binary_label']],
            'embeddings_path': f'{self.base_dir}/data/embeddings/train_deberta_embeddings.npy',
            'roberta_embeddings_path': f'{self.base_dir}/data/embeddings/train_roberta_embeddings.npy',
            'tfidf_path': f'{self.base_dir}/data/processed/train_tfidf.npz'
        }
        
        # Validation feature store
        val_feature_store = {
            'metadata': val_df[['id', 'title', 'text', 'label', 'source', 'dataset_source']],
            'features': val_df[['id'] + feature_cols + ['binary_label']],
            'embeddings_path': f'{self.base_dir}/data/embeddings/val_deberta_embeddings.npy',
            'roberta_embeddings_path': f'{self.base_dir}/data/embeddings/val_roberta_embeddings.npy',
            'tfidf_path': f'{self.base_dir}/data/processed/val_tfidf.npz'
        }
        
        # Test feature store
        test_feature_store = {
            'metadata': test_df[['id', 'title', 'text', 'label', 'source', 'dataset_source']],
            'features': test_df[['id'] + feature_cols + ['binary_label']],
            'embeddings_path': f'{self.base_dir}/data/embeddings/test_deberta_embeddings.npy',
            'roberta_embeddings_path': f'{self.base_dir}/data/embeddings/test_roberta_embeddings.npy',
            'tfidf_path': f'{self.base_dir}/data/processed/test_tfidf.npz'
        }
        
        # Save feature stores
        with open(f'{self.base_dir}/data/processed/train_feature_store.pkl', 'wb') as f:
            pickle.dump(train_feature_store, f)
        
        with open(f'{self.base_dir}/data/processed/val_feature_store.pkl', 'wb') as f:
            pickle.dump(val_feature_store, f)
        
        with open(f'{self.base_dir}/data/processed/test_feature_store.pkl', 'wb') as f:
            pickle.dump(test_feature_store, f)
        
        # Dataset statistics
        label_counts = combined_df['binary_label'].value_counts()
        source_counts = combined_df['source'].value_counts()
        dataset_source_counts = combined_df['dataset_source'].value_counts()
        
        print("\nDataset Statistics:")
        print(f"Total samples: {len(combined_df)}")
        print(f"Label distribution: {label_counts.to_dict()}")
        print(f"Source distribution (top 5): {source_counts.head(5).to_dict()}")
        print(f"Dataset source distribution: {dataset_source_counts.to_dict()}")
        
        return {
            'train': train_feature_store,
            'validation': val_feature_store,
            'test': test_feature_store,
            'statistics': {
                'label_counts': label_counts.to_dict(),
                'source_counts': source_counts.to_dict(),
                'dataset_source_counts': dataset_source_counts.to_dict()
            }
        }

def main():
    """Main function to execute the data processing pipeline"""
    
    print("Starting the fake news dataset preprocessing pipeline...")
    
    # Set up base directory
    base_dir = '/kaggle/working'
    
    # Check if we're running in Kaggle environment
    if not os.path.exists('/kaggle'):
        print("Not running in Kaggle environment, using local directories...")
        base_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(base_dir, exist_ok=True)
    
    # Configuration parameters
    use_spacy = False  # Set to True for more sophisticated NLP, but requires more dependencies
    use_oversampling = True  # Set to True to balance classes
    
    # Print configuration
    print(f"Configuration:")
    print(f"- Base directory: {base_dir}")
    print(f"- Using SpaCy: {use_spacy}")
    print(f"- Using oversampling: {use_oversampling}")
    print(f"- Using device: {DEVICE}")
    
    # Initialize and run data processor
    try:
        processor = EnhancedDataProcessor(
            base_dir=base_dir,
            use_spacy=use_spacy,
            use_oversampling=use_oversampling
        )
        
        processed_data = processor.process_datasets()
        
        if processed_data:
            print("\nData processing completed successfully!")
            print(f"Processed data is stored in {base_dir}/data/processed/")
            print(f"Embeddings are stored in {base_dir}/data/embeddings/")
            
            # Print some summary statistics
            if 'statistics' in processed_data:
                stats = processed_data['statistics']
                print("\nDataset summary:")
                print(f"- Total fake news samples: {stats['label_counts'].get(0, 0)}")
                print(f"- Total real news samples: {stats['label_counts'].get(1, 0)}")
                
                dataset_sources = stats['dataset_source_counts']
                print("\nDataset sources:")
                for source, count in dataset_sources.items():
                    print(f"- {source}: {count} samples")
        else:
            print("\nError: Data processing did not complete successfully.")
    
    except Exception as e:
        print(f"\nAn error occurred during data processing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPreprocessing pipeline execution finished.")

# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()