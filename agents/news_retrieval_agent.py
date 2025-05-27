import numpy as np
import pandas as pd
import re
import requests
from abc import ABC, abstractmethod
from scipy import sparse
from typing import Dict, List, Union, Tuple, Optional, Any
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModel

from .base_agent import Agent

class NewsRetrieverAgent(Agent):
    """Agent responsible for retrieving news articles from various sources with enhanced capabilities."""
    
    def __init__(self, name: str = "NewsRetriever", api_key: str = None, 
                 mediastack_key: str = None, gnews_key: str = None, 
                 cache_expiry: int = 3600):
        """Initialize the news retriever agent with multiple API support.
        
        Args:
            name: Agent name
            api_key: API key for NewsAPI
            mediastack_key: API key for MediaStack
            gnews_key: API key for GNews
            cache_expiry: Cache expiry time in seconds (default: 1 hour)
        """
        super().__init__(name)
        self.api_keys = {
            "newsapi": api_key,
            "mediastack": mediastack_key,
            "gnews": gnews_key
        }
        
        self.sources = {
            "newsapi": "https://newsapi.org/v2/",
            "mediastack": "http://api.mediastack.com/v1/news",
            "gnews": "https://gnews.io/api/v4/search"
        }
        
        # Enhanced cache with expiry times
        self.cache = {}
        self.cache_expiry = cache_expiry
        
        # Initialize newspaper for content extraction
        try:
            import newspaper
            self.has_newspaper = True
            self.log("Newspaper3k library initialized for article extraction")
        except ImportError:
            self.has_newspaper = False
            self.log("Newspaper3k library not available, full content extraction disabled", "warning")
        
        # Initialize NLTK for better key term extraction
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('punkt')
                nltk.download('stopwords')
            self.has_nltk = True
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
            self.log("NLTK initialized for enhanced term extraction")
        except ImportError:
            self.has_nltk = False
            # Basic English stopwords as fallback
            self.stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                             'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'through',
                             'this', 'that', 'these', 'those', 'it', 'they', 'he', 'she', 'we', 'you',
                             'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
                             'can', 'may', 'might', 'must', 'be', 'been', 'being'}
            self.log("NLTK not available, using basic stopwords list", "warning")
    
    def process(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve news articles related to the query with enhanced source coverage.
        
        Args:
            query: Search query or topic
            
        Returns:
            List of retrieved articles with metadata
        """
        self.log(f"Retrieving news for query: {query}")
        
        # Check cache first with expiry handling
        cache_key = f"query_{query.lower().replace(' ', '_')}"
        if cache_key in self.cache:
            cached_time, cached_articles = self.cache[cache_key]
            current_time = time.time()
            
            # If cache is still valid
            if current_time - cached_time < self.cache_expiry:
                self.log(f"Using cached results for: {query}")
                return cached_articles
            else:
                self.log(f"Cache expired for: {query}")
        
        articles = []
        
        # Try fetching from different sources in order of reliability/quality
        try:
            # Use a shorter timeout for API requests
            api_timeout = 10  # 10 seconds timeout
            
            # 1. NewsAPI (most reliable structured data)
            if self.api_keys.get("newsapi"):
                self.log("Trying NewsAPI...", "info")
                start_time = time.time()
                try:
                    newsapi_articles = self._fetch_from_newsapi(query)
                    articles.extend(newsapi_articles)
                    self.log(f"Retrieved {len(newsapi_articles)} articles from NewsAPI in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    self.log(f"Error with NewsAPI: {str(e)}", "error")
            
            # 2. GNews API (additional coverage)
            if self.api_keys.get("gnews"):
                self.log("Trying GNews API...", "info")
                start_time = time.time()
                try:
                    gnews_articles = self._fetch_from_gnews(query)
                    # Deduplicate before adding
                    unique_gnews = self._deduplicate_articles(gnews_articles, articles)
                    articles.extend(unique_gnews)
                    self.log(f"Retrieved {len(unique_gnews)} unique articles from GNews in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    self.log(f"Error with GNews: {str(e)}", "error")
            
            # 3. MediaStack API (additional coverage)
            if self.api_keys.get("mediastack"):
                self.log("Trying MediaStack API...", "info")
                start_time = time.time()
                try:
                    mediastack_articles = self._fetch_from_mediastack(query)
                    # Deduplicate before adding
                    unique_mediastack = self._deduplicate_articles(mediastack_articles, articles)
                    articles.extend(unique_mediastack)
                    self.log(f"Retrieved {len(unique_mediastack)} unique articles from MediaStack in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    self.log(f"Error with MediaStack: {str(e)}", "error")
            
            # If no articles found, try to return quickly with mock data
            if len(articles) == 0:
                self.log("No articles found, adding mock article for testing", "warning")
                # Add a mock article to ensure we have something to return
                articles.append({
                    "title": f"Information about {query}",
                    "text": f"This is a mock article generated because no real articles were found for the query: {query}",
                    "source": "System Generated",
                    "url": "",
                    "published_at": datetime.now().isoformat(),
                    "retrieved_at": datetime.now().isoformat(),
                    "source_api": "mock"
                })
            
            # Store in cache with current timestamp
            if articles:
                self.cache[cache_key] = (time.time(), articles)
            
            self.log(f"Retrieved a total of {len(articles)} articles")
            return articles
                
        except Exception as e:
            self.log(f"Error retrieving news: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            
            # Return mock data on error to avoid complete failure
            mock_article = {
                "title": f"Information about {query}",
                "text": f"This is a fallback article generated due to an error in news retrieval for the query: {query}",
                "source": "Error Fallback",
                "url": "",
                "published_at": datetime.now().isoformat(),
                "retrieved_at": datetime.now().isoformat(),
                "source_api": "fallback"
            }
            return [mock_article]
    
    def _fetch_from_newsapi(self, query: str) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI with enhanced error handling.
        
        Args:
            query: Search query
            
        Returns:
            List of articles
        """
        self.log(f"Fetching from NewsAPI with query: {query}", "info")
        url = f"{self.sources['newsapi']}everything"
        
        # Get current date in YYYY-MM-DD format
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        params = {
            "q": query,
            "apiKey": self.api_keys.get("newsapi"),
            "language": "en",
            "sortBy": "publishedAt",  # Align with provided format
            "pageSize": 20,
            "from": from_date
        }
        
        try:
            # Use a short timeout to avoid hanging
            start_time = time.time()
            response = requests.get(url, params=params, timeout=5)
            response_time = time.time() - start_time
            self.log(f"NewsAPI response time: {response_time:.2f} seconds", "info")
            
            # Log API response code and size
            self.log(f"NewsAPI response status code: {response.status_code}", "info")
            if response.status_code == 200:
                self.log(f"NewsAPI response size: {len(response.text)} bytes", "info")
                
            data = response.json()
            
            if response.status_code == 200 and data.get("status") == "ok":
                # Transform to standardized format
                articles = []
                for article in data.get("articles", []):
                    # Skip articles with missing essential information
                    if not article.get("title") or not article.get("url"):
                        continue
                        
                    # Combine description and content for better context
                    text = article.get("description", "") or ""
                    if article.get("content"):
                        text += " " + article.get("content")
                    
                    articles.append({
                        "title": article.get("title", ""),
                        "text": text,
                        "source": article.get("source", {}).get("name", "NewsAPI"),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "retrieved_at": datetime.now().isoformat(),
                        "source_api": "newsapi"
                    })
                return articles
            else:
                error_msg = data.get("message", "Unknown error")
                self.log(f"NewsAPI error: {error_msg}", "error")
                
                # Handle rate limiting explicitly
                if "rateLimited" in error_msg or response.status_code == 429:
                    self.log("NewsAPI rate limit exceeded, waiting before retrying", "warning")
                    
                return []
                    
        except requests.exceptions.Timeout:
            self.log("NewsAPI request timed out", "error")
            return []
        except requests.exceptions.RequestException as e:
            self.log(f"Error fetching from NewsAPI: {str(e)}", "error")
            return []
        except Exception as e:
            self.log(f"Unexpected error with NewsAPI: {str(e)}", "error")
            return []    
    
    def _fetch_from_gnews(self, query: str) -> List[Dict[str, Any]]:
        """Fetch news from GNews API."""
        if not self.api_keys.get("gnews"):
            return []
            
        url = self.sources["gnews"]
        params = {
            "q": query,
            "apikey": self.api_keys.get("gnews"),  # Correct parameter name for the API key
            "lang": "en",
            "max": 10,
            "sortby": "publish_date_desc"  # Changed from "relevance" to get newest articles first
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if response.status_code == 200 and "articles" in data:
                articles = []
                for article in data.get("articles", []):
                    articles.append({
                        "title": article.get("title", ""),
                        "text": article.get("description", ""),
                        "source": article.get("source", {}).get("name", "GNews"),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "retrieved_at": datetime.now().isoformat(),
                        "source_api": "gnews"
                    })
                return articles
            else:
                error_msg = "Unknown error"
                if "errors" in data:
                    error_msg = data.get("errors", [])[0] if data.get("errors") else "Unknown error"
                self.log(f"GNews API error: {error_msg}", "error")
                return []
        except Exception as e:
            self.log(f"Error fetching from GNews: {str(e)}", "error")
            return []
        
    def _fetch_from_mediastack(self, query: str) -> List[Dict[str, Any]]:
        """Fetch news from MediaStack API."""
        if not self.api_keys.get("mediastack"):
            return []
    
        url = self.sources["mediastack"]
        params = {
            "access_key": self.api_keys.get("mediastack"),
            "keywords": query,
            "languages": "en",
            "limit": 10,
            "sort": "published_desc",  # Correct value per API docs
            "categories": "-general,-sports"  # Example exclusion
        }
    
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
    
            if response.status_code == 200 and "data" in data:
                articles = []
                for article in data["data"]:
                    articles.append({
                        "title": article.get("title", ""),
                        "text": article.get("description", ""),
                        "source": article.get("source", "MediaStack"),
                        "url": article.get("url", ""),
                        "published_at": article.get("published_at", ""),
                        "retrieved_at": datetime.now().isoformat(),
                        "source_api": "mediastack"
                    })
                return articles
            else:
                error_info = data.get("error", {}).get("info", f"HTTP {response.status_code}")
                self.log(f"MediaStack API error: {error_info}", "error")
                return []
    
        except Exception as e:
            self.log(f"Error fetching from MediaStack: {str(e)}", "error")
            return []
    
    def _scrape_news(self, query: str) -> List[Dict[str, Any]]:
        """Scrape news from search engines using newspaper3k if available.
        
        Args:
            query: Search query
            
        Returns:
            List of scraped articles
        """
        if not self.has_newspaper:
            self.log("Newspaper3k library required for web scraping", "warning")
            return []
        
        try:
            import newspaper
            from newspaper import Article
            import requests
            from bs4 import BeautifulSoup
            
            # Use search engine to find relevant URLs
            articles = []
            
            # Google search scraping (simple approach - not robust for production)
            # Note: In production, use a proper API or more robust scraping approach
            search_url = f"https://www.google.com/search?q={query}+news"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract news URLs (very simplified - Google can change layout)
                urls = []
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href.startswith('/url?q=') and 'google' not in href:
                        # Extract actual URL
                        url = href.split('/url?q=')[1].split('&')[0]
                        if url.startswith('http') and url not in urls:
                            urls.append(url)
                
                # Limit to first 5 URLs to avoid excessive scraping
                for url in urls[:5]:
                    try:
                        article = Article(url)
                        article.download()
                        article.parse()
                        
                        if article.title and article.text:
                            articles.append({
                                "title": article.title,
                                "text": article.text[:2000] + ("..." if len(article.text) > 2000 else ""),
                                "source": article.source_url or url.split('/')[2],
                                "url": url,
                                "published_at": article.publish_date.isoformat() if article.publish_date else "",
                                "retrieved_at": datetime.now().isoformat(),
                                "source_api": "webscrape"
                            })
                    except Exception as e:
                        self.log(f"Error parsing article {url}: {str(e)}", "warning")
                        continue
            
            return articles
                
        except Exception as e:
            self.log(f"Error during web scraping: {str(e)}", "error")
            return []
    
    def _enhance_with_full_content(self, articles: List[Dict[str, Any]]) -> None:
        """Enhance articles with full content using newspaper3k.
        
        Args:
            articles: List of article dictionaries to enhance
        """
        if not self.has_newspaper:
            return
            
        try:
            import newspaper
            from newspaper import Article
            
            for article in articles:
                url = article.get("url")
                if not url or len(article.get("text", "")) > 1000:  # Skip if we already have substantial text
                    continue
                    
                try:
                    # Use newspaper to extract full article
                    news_article = Article(url)
                    news_article.download()
                    news_article.parse()
                    
                    if news_article.text:
                        article["full_text"] = news_article.text
                        # Update text field with better content but keep it reasonably sized
                        article["text"] = news_article.text[:2000] + ("..." if len(news_article.text) > 2000 else "")
                        
                    # Add additional metadata if available
                    if news_article.publish_date and not article.get("published_at"):
                        article["published_at"] = news_article.publish_date.isoformat()
                    if news_article.authors:
                        article["authors"] = news_article.authors
                    if news_article.keywords:
                        article["keywords"] = news_article.keywords
                    
                except Exception as e:
                    self.log(f"Error extracting content for {url}: {str(e)}", "warning")
                    continue
                    
        except Exception as e:
            self.log(f"Error in content enhancement: {str(e)}", "error")
    
    def _deduplicate_articles(self, new_articles: List[Dict[str, Any]], 
                             existing_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on URL or title similarity.
        
        Args:
            new_articles: List of new articles to check
            existing_articles: List of existing articles to check against
            
        Returns:
            List of non-duplicate articles
        """
        existing_urls = set(a.get("url", "") for a in existing_articles)
        existing_titles = set(a.get("title", "").lower() for a in existing_articles)
        
        unique_articles = []
        
        for article in new_articles:
            url = article.get("url", "")
            title = article.get("title", "").lower()
            
            # Skip if URL already exists
            if url and url in existing_urls:
                continue
                
            # Check for very similar titles
            title_match = False
            for existing_title in existing_titles:
                if existing_title and title and self._calculate_similarity(title, existing_title) > 0.8:
                    title_match = True
                    break
                    
            if not title_match:
                unique_articles.append(article)
                # Add to existing sets to avoid duplicates within new articles
                existing_urls.add(url)
                existing_titles.add(title)
        
        return unique_articles
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Jaccard similarity of words.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0
            
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity: intersection / union
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0
            
        return intersection / union
    
    def search_by_claim(self, claim: str) -> List[Dict[str, Any]]:
        """Search for news articles related to a specific claim with improved term extraction.
        
        Args:
            claim: The claim to verify
            
        Returns:
            List of relevant articles
        """
        self.log(f"Starting search_by_claim for: '{claim}'", "info")
        
        try:
            # Extract key terms from the claim with more sophisticated approach
            self.log("Extracting key terms...", "info")
            claim_terms = self._extract_key_terms(claim)
            self.log(f"Extracted terms: {claim_terms}", "info")

            # Named Entity Recognition for better search if NLTK is available
            self.log("Extracting named entities...", "info")
            named_entities = self._extract_named_entities(claim)
            self.log(f"Extracted entities: {named_entities}", "info")

            # Combine regular terms and named entities, prioritizing named entities
            search_terms = []
            search_terms.extend(named_entities)
            for term in claim_terms:
                if term not in search_terms and len(search_terms) < 5:
                    search_terms.append(term)
            
            search_query = " ".join(search_terms[:5])  # Limit to top 5 terms
            
            self.log(f"Final search query: '{search_query}'", "info")
            
            # Set a timeout for the process function
            articles = []
            try:
                # Add a timeout to the process call
                self.log(f"Calling process with search query: '{search_query}'", "info")
                start_time = time.time()
                articles = self.process(search_query)
                elapsed_time = time.time() - start_time
                self.log(f"Retrieved {len(articles)} articles in {elapsed_time:.2f} seconds", "info")
            except Exception as e:
                self.log(f"Error in process call: {str(e)}", "error")
                # Return empty list on failure
                return []
            
            return articles
        except Exception as e:
            self.log(f"Error in search_by_claim: {str(e)}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
            return []
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text with enhanced NLP techniques.
        
        Args:
            text: Input text
            
        Returns:
            List of key terms
        """
        # Extract words with better regex handling
        words = re.findall(r'\b[a-zA-Z]\w+\b', text.lower())
        
        # Filter out stopwords and very short words
        key_terms = [word for word in words if word not in self.stopwords and len(word) > 2]
        
        # Use frequency to identify important terms
        from collections import Counter
        word_counts = Counter(key_terms)
        
        # Get most common terms
        important_terms = [word for word, count in word_counts.most_common(15)]
        
        # Add important bigrams (two-word phrases)
        bigrams = self._extract_bigrams(text)
        
        # Add important trigrams (three-word phrases) for better context
        trigrams = self._extract_trigrams(text)
        
        # Combine phrases and single terms, prioritizing longer phrases
        combined_terms = []
        
        # Add trigrams first (most specific)
        for trigram in trigrams[:2]:
            combined_terms.append(trigram)
            
        # Add bigrams
        for bigram in bigrams[:4]:
            # Check if not already covered by trigrams
            already_included = any(bigram in existing for existing in combined_terms)
            if not already_included:
                combined_terms.append(bigram)
            
        # Add single terms that aren't part of selected phrases
        for term in important_terms:
            already_included = any(term in existing for existing in combined_terms)
            
            if not already_included and len(combined_terms) < 8:
                combined_terms.append(term)
        
        return combined_terms
    
    def _extract_bigrams(self, text: str) -> List[str]:
        """Extract important bigrams (two-word phrases) from text.
        
        Args:
            text: Input text
            
        Returns:
            List of bigram strings
        """
        # Simple bigram extraction without NLTK
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]\w+\b', text)]
        
        # Filter out stopwords
        filtered_words = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        # Create bigrams
        bigrams = []
        for i in range(len(filtered_words) - 1):
            bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
            bigrams.append(bigram)
        
        # Return most frequent bigrams
        from collections import Counter
        return [bigram for bigram, count in Counter(bigrams).most_common(8)]
    
    def _extract_trigrams(self, text: str) -> List[str]:
        """Extract important trigrams (three-word phrases) from text.
        
        Args:
            text: Input text
            
        Returns:
            List of trigram strings
        """
        # Simple trigram extraction
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]\w+\b', text)]
        
        # Filter out stopwords
        filtered_words = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        # Create trigrams
        trigrams = []
        for i in range(len(filtered_words) - 2):
            trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
            trigrams.append(trigram)
        
        # Return most frequent trigrams
        from collections import Counter
        return [trigram for trigram, count in Counter(trigrams).most_common(4)]
    
    def _extract_named_entities(self, text: str) -> List[str]:
        """Extract named entities for better search queries.
        
        Args:
            text: Input text
            
        Returns:
            List of named entities
        """
        entities = []
        
        # Use NLTK if available
        if self.has_nltk:
            try:
                import nltk
                # Try to use named entity recognition if available
                try:
                    nltk.data.find('chunkers/maxent_ne_chunker')
                    nltk.data.find('taggers/maxent_treebank_pos_tagger')
                except LookupError:
                    nltk.download('maxent_ne_chunker')
                    nltk.download('words')
                    nltk.download('maxent_treebank_pos_tagger')
                
                words = nltk.word_tokenize(text)
                pos_tags = nltk.pos_tag(words)
                named_entities = nltk.ne_chunk(pos_tags)
                
                # Extract named entities
                current_entity = []
                current_type = None
                
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
                        # This is a named entity
                        entity_text = ' '.join([c[0] for c in chunk])
                        if entity_text and len(entity_text) > 1:
                            entities.append(entity_text)
                
            except Exception as e:
                self.log(f"Error in NLTK named entity extraction: {str(e)}", "warning")
        
        # Fallback: simple capitalized phrases
        if not entities:
            # Find sequences of capitalized words
            cap_phrases = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*', text)
            for phrase in cap_phrases:
                if len(phrase) > 2 and phrase.lower() not in self.stopwords:
                    entities.append(phrase)
        
        return entities