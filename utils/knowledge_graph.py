import logging
import torch
import torch.nn as nn
import numpy as np
import requests
import time
from typing import Dict, List, Union, Tuple, Optional, Any 

class KnowledgeGraphConnector:
    """Connector for external knowledge graphs to enhance fact-checking."""
    
    def __init__(self, kg_api_key: str = None):
        """Initialize the knowledge graph connector.
        
        Args:
            kg_api_key: API key for knowledge graph services that require it
        """
        self.logger = logging.getLogger("KnowledgeGraph")
        self.kg_api_key = kg_api_key
        
        # Knowledge graph endpoints
        self.kg_endpoints = {
            'wikidata': 'https://query.wikidata.org/sparql',
            'dbpedia': 'https://dbpedia.org/sparql',
            'google_kg': 'https://kgsearch.googleapis.com/v1/entities:search'
        }
        
        # Cache to avoid repeated requests
        self.cache = {}
        
        # Performance settings
        self.request_timeout = 3.0  # 3 second timeout for HTTP requests
        self.max_query_time = 5.0   # 5 second max per knowledge graph query
        self.enable_fallback = True  # Enable fast fallback methods
    
    def query_entity(self, entity: str) -> Dict[str, Any]:
        """Query knowledge graphs for entity information with timeout protection.
        
        Args:
            entity: Entity name to query
            
        Returns:
            Dictionary containing entity data from multiple knowledge graphs
        """
        cache_key = f"entity_{entity.lower().replace(' ', '_')}"
        if cache_key in self.cache:
            self.logger.info(f"Using cached data for entity: {entity}")
            return self.cache[cache_key]
            
        self.logger.info(f"Querying knowledge graphs for entity: {entity}")
        start_time = time.time()
        
        # Try multiple knowledge graphs and combine results
        entity_data = {
            'entity': entity,
            'found': False,
            'sources': [],
            'attributes': {},
            'description': '',
            'aliases': []
        }
        
        # Try Google Knowledge Graph first (fastest and most reliable)
        if self.kg_api_key and (time.time() - start_time) < self.max_query_time:
            google_result = self._query_google_kg(entity)
            if google_result.get('found', False):
                entity_data['found'] = True
                entity_data['sources'].append('google_kg')
                entity_data['attributes'].update(google_result.get('attributes', {}))
                entity_data['description'] = google_result.get('description', '')
                entity_data['aliases'].extend(google_result.get('aliases', []))
                
                # If we found good data from Google KG, we can skip the slower sources
                if entity_data['description'] and len(entity_data['attributes']) > 0:
                    self.cache[cache_key] = entity_data
                    return entity_data
        
        # Try Wikidata with timeout protection
        if (time.time() - start_time) < self.max_query_time:
            try:
                wikidata_result = self._query_wikidata_fast(entity)
                if wikidata_result.get('found', False):
                    entity_data['found'] = True
                    entity_data['sources'].append('wikidata')
                    entity_data['attributes'].update(wikidata_result.get('attributes', {}))
                    if not entity_data['description']:
                        entity_data['description'] = wikidata_result.get('description', '')
                    entity_data['aliases'].extend(wikidata_result.get('aliases', []))
            except Exception as e:
                self.logger.warning(f"Wikidata query failed for {entity}: {str(e)}")
        
        # Only try DBpedia if we still need more info and have time
        if (not entity_data['found'] or not entity_data['description']) and (time.time() - start_time) < self.max_query_time:
            try:
                dbpedia_result = self._query_dbpedia_fast(entity)
                if dbpedia_result.get('found', False):
                    entity_data['found'] = True
                    entity_data['sources'].append('dbpedia')
                    entity_data['attributes'].update(dbpedia_result.get('attributes', {}))
                    if not entity_data['description']:
                        entity_data['description'] = dbpedia_result.get('description', '')
                    entity_data['aliases'].extend(dbpedia_result.get('aliases', []))
            except Exception as e:
                self.logger.warning(f"DBpedia query failed for {entity}: {str(e)}")
        
        # Remove duplicates from aliases
        entity_data['aliases'] = list(set(entity_data['aliases']))
        
        # Cache the result
        self.cache[cache_key] = entity_data
        
        query_time = time.time() - start_time
        self.logger.info(f"Entity query for '{entity}' completed in {query_time:.2f}s")
        
        return entity_data
    
    def _query_wikidata_fast(self, entity: str) -> Dict[str, Any]:
        """Fast Wikidata query with simplified SPARQL and timeout protection.
        
        Args:
            entity: Entity name to query
            
        Returns:
            Dictionary containing Wikidata entity data
        """
        try:
            # Simplified query that's much faster
            query = f"""
            SELECT ?item ?itemLabel ?itemDescription
            WHERE {{
              ?item rdfs:label "{entity}"@en.
              OPTIONAL {{ ?item schema:description ?itemDescription. FILTER(LANG(?itemDescription) = "en") }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 1
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'TruthScan/2.0'
            }
            
            response = requests.get(
                self.kg_endpoints['wikidata'],
                params={'query': query, 'format': 'json'},
                headers=headers,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                
                if results:
                    entity_data = {
                        'found': True,
                        'attributes': {},
                        'description': results[0].get('itemDescription', {}).get('value', ''),
                        'aliases': []
                    }
                    
                    # Get entity URI for basic info
                    if results[0].get('item', {}).get('value'):
                        entity_uri = results[0]['item']['value']
                        entity_data['attributes']['wikidata_uri'] = entity_uri
                    
                    return entity_data
                
            return {'found': False}
                
        except requests.Timeout:
            self.logger.warning(f"Wikidata query timeout for entity: {entity}")
            return {'found': False, 'error': 'timeout'}
        except Exception as e:
            self.logger.error(f"Error querying Wikidata: {str(e)}")
            return {'found': False, 'error': str(e)}
    
    def _query_dbpedia_fast(self, entity: str) -> Dict[str, Any]:
        """Fast DBpedia query with simplified approach and timeout protection.
        
        Args:
            entity: Entity name to query
            
        Returns:
            Dictionary containing DBpedia entity data
        """
        try:
            # Format entity name for DBpedia
            formatted_entity = '_'.join(word.capitalize() for word in entity.split())
            
            # Simplified query
            query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbr: <http://dbpedia.org/resource/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?abstract
            WHERE {{
              dbr:{formatted_entity} dbo:abstract ?abstract.
              FILTER(LANG(?abstract) = "en")
            }}
            LIMIT 1
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'TruthScan/2.0'
            }
            
            response = requests.get(
                self.kg_endpoints['dbpedia'],
                params={'query': query, 'format': 'json'},
                headers=headers,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                
                if results and results[0].get('abstract', {}).get('value'):
                    entity_data = {
                        'found': True,
                        'attributes': {'dbpedia_uri': f"http://dbpedia.org/resource/{formatted_entity}"},
                        'description': results[0]['abstract']['value'],
                        'aliases': []
                    }
                    return entity_data
            
            return {'found': False}
                
        except requests.Timeout:
            self.logger.warning(f"DBpedia query timeout for entity: {entity}")
            return {'found': False, 'error': 'timeout'}
        except Exception as e:
            self.logger.error(f"Error querying DBpedia: {str(e)}")
            return {'found': False, 'error': str(e)}
    
    def _query_wikidata(self, entity: str) -> Dict[str, Any]:
        """Query Wikidata for entity information.
        
        Args:
            entity: Entity name to query
            
        Returns:
            Dictionary containing Wikidata entity data
        """
        try:
            import requests
            
            # SPARQL query to get basic entity information
            query = f"""
            SELECT ?item ?itemLabel ?itemDescription ?alias
            WHERE {{
              ?item rdfs:label ?label.
              ?item schema:description ?itemDescription.
              OPTIONAL {{ ?item skos:altLabel ?alias. FILTER(LANG(?alias) = "en") }}
              FILTER(LANG(?label) = "en" && REGEX(?label, "^{entity}$", "i"))
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 5
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'FakeNewsDetector/1.0 (your-email@example.com)'
            }
            
            response = requests.get(
                self.kg_endpoints['wikidata'],
                params={'query': query, 'format': 'json'},
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                
                if results:
                    entity_data = {
                        'found': True,
                        'attributes': {},
                        'description': '',
                        'aliases': []
                    }
                    
                    # Get the first result's description
                    if results[0].get('itemDescription', {}).get('value'):
                        entity_data['description'] = results[0]['itemDescription']['value']
                    
                    # Collect aliases
                    for result in results:
                        if result.get('alias', {}).get('value'):
                            entity_data['aliases'].append(result['alias']['value'])
                    
                    # Get entity URI for further queries
                    if results[0].get('item', {}).get('value'):
                        entity_uri = results[0]['item']['value']
                        entity_data['attributes']['wikidata_uri'] = entity_uri
                        
                        # Get additional attributes with a second query
                        property_query = f"""
                        SELECT ?property ?propertyLabel ?value ?valueLabel
                        WHERE {{
                          <{entity_uri}> ?property ?value.
                          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                          FILTER(STRSTARTS(STR(?property), "http://www.wikidata.org/prop/direct/"))
                        }}
                        LIMIT 50
                        """
                        
                        prop_response = requests.get(
                            self.kg_endpoints['wikidata'],
                            params={'query': property_query, 'format': 'json'},
                            headers=headers
                        )
                        
                        if prop_response.status_code == 200:
                            prop_data = prop_response.json()
                            prop_results = prop_data.get('results', {}).get('bindings', [])
                            
                            # Process property results
                            for prop in prop_results:
                                if prop.get('propertyLabel', {}).get('value') and prop.get('valueLabel', {}).get('value'):
                                    prop_name = prop['propertyLabel']['value']
                                    prop_value = prop['valueLabel']['value']
                                    entity_data['attributes'][prop_name] = prop_value
                    
                    return entity_data
                
            return {'found': False}
                
        except Exception as e:
            self.logger.error(f"Error querying Wikidata: {str(e)}")
            return {'found': False, 'error': str(e)}
    
    def _query_dbpedia(self, entity: str) -> Dict[str, Any]:
        """Query DBpedia for entity information.
        
        Args:
            entity: Entity name to query
            
        Returns:
            Dictionary containing DBpedia entity data
        """
        try:
            import requests
            
            # Format entity name for DBpedia (capitalize first letter of each word)
            formatted_entity = '_'.join(word.capitalize() for word in entity.split())
            
            # Query to get entity information
            query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbr: <http://dbpedia.org/resource/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?abstract ?label
            WHERE {{
              OPTIONAL {{ dbr:{formatted_entity} dbo:abstract ?abstract. FILTER(LANG(?abstract) = "en") }}
              OPTIONAL {{ dbr:{formatted_entity} rdfs:label ?label. FILTER(LANG(?label) = "en") }}
            }}
            LIMIT 1
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'FakeNewsDetector/1.0 (your-email@example.com)'
            }
            
            response = requests.get(
                self.kg_endpoints['dbpedia'],
                params={'query': query, 'format': 'json'},
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                
                if results and (results[0].get('abstract', {}).get('value') or results[0].get('label', {}).get('value')):
                    entity_data = {
                        'found': True,
                        'attributes': {'dbpedia_uri': f"http://dbpedia.org/resource/{formatted_entity}"},
                        'description': results[0].get('abstract', {}).get('value', ''),
                        'aliases': []
                    }
                    
                    if results[0].get('label', {}).get('value'):
                        entity_data['aliases'].append(results[0]['label']['value'])
                    
                    # Get additional properties
                    property_query = f"""
                    PREFIX dbr: <http://dbpedia.org/resource/>
                    
                    SELECT ?property ?value
                    WHERE {{
                      dbr:{formatted_entity} ?property ?value.
                      FILTER(LANG(?value) = "en" || LANG(?value) = "")
                      FILTER(STRSTARTS(STR(?property), "http://dbpedia.org/ontology/"))
                    }}
                    LIMIT 30
                    """
                    
                    prop_response = requests.get(
                        self.kg_endpoints['dbpedia'],
                        params={'query': property_query, 'format': 'json'},
                        headers=headers
                    )
                    
                    if prop_response.status_code == 200:
                        prop_data = prop_response.json()
                        prop_results = prop_data.get('results', {}).get('bindings', [])
                        
                        # Process property results
                        for prop in prop_results:
                            if prop.get('property', {}).get('value') and prop.get('value', {}).get('value'):
                                # Extract property name from URI
                                prop_uri = prop['property']['value']
                                prop_name = prop_uri.split('/')[-1]
                                prop_value = prop['value']['value']
                                entity_data['attributes'][prop_name] = prop_value
                    
                    return entity_data
            
            return {'found': False}
                
        except Exception as e:
            self.logger.error(f"Error querying DBpedia: {str(e)}")
            return {'found': False, 'error': str(e)}
    
    def _query_google_kg(self, entity: str) -> Dict[str, Any]:
        """Query Google Knowledge Graph for entity information.
        
        Args:
            entity: Entity name to query
            
        Returns:
            Dictionary containing Google KG entity data
        """
        if not self.kg_api_key:
            return {'found': False, 'error': 'No Google KG API key provided'}
            
        try:
            import requests
            
            params = {
                'query': entity,
                'limit': 1,
                'indent': True,
                'key': self.kg_api_key,
                'languages': 'en'
            }
            
            response = requests.get(self.kg_endpoints['google_kg'], params=params)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('itemListElement', [])
                
                if items and items[0].get('result'):
                    result = items[0]['result']
                    entity_data = {
                        'found': True,
                        'attributes': {},
                        'description': result.get('description', ''),
                        'aliases': result.get('name', [])
                    }
                    
                    # Add types
                    if result.get('@type'):
                        entity_data['attributes']['types'] = result['@type']
                    
                    # Add detailed description
                    if result.get('detailedDescription'):
                        entity_data['attributes']['detailed_description'] = result['detailedDescription'].get('articleBody', '')
                        entity_data['attributes']['url'] = result['detailedDescription'].get('url', '')
                    
                    return entity_data
            
            return {'found': False}
                
        except Exception as e:
            self.logger.error(f"Error querying Google Knowledge Graph: {str(e)}")
            return {'found': False, 'error': str(e)}
    
    def verify_fact_triple(self, subject: str, predicate: str, object_entity: str) -> Dict[str, Any]:
        """Verify a fact triple against knowledge graphs with timeout protection.
        
        Args:
            subject: Subject entity
            predicate: Relation predicate
            object_entity: Object entity
            
        Returns:
            Dictionary containing verification result
        """
        self.logger.info(f"Verifying fact: {subject} {predicate} {object_entity}")
        start_time = time.time()
        
        cache_key = f"fact_{subject.lower().replace(' ', '_')}_{predicate.lower().replace(' ', '_')}_{object_entity.lower().replace(' ', '_')}"
        if cache_key in self.cache:
            self.logger.info(f"Using cached verification result for fact: {subject} {predicate} {object_entity}")
            return self.cache[cache_key]
        
        # Initialize verification result
        verification_result = {
            'subject': subject,
            'predicate': predicate,
            'object': object_entity,
            'verified': False,
            'confidence': 0.0,
            'sources': []
        }
        
        try:
            # Quick heuristic check first (very fast)
            if self._quick_heuristic_check(subject, predicate, object_entity):
                verification_result['verified'] = True
                verification_result['confidence'] = 0.5  # Medium confidence for heuristic
                verification_result['sources'].append('heuristic_match')
                self.cache[cache_key] = verification_result
                return verification_result
            
            # Only do expensive KG queries if we have time and haven't found a match
            if (time.time() - start_time) < self.max_query_time:
                # Get basic information about the subject (with timeout)
                subject_data = self.query_entity(subject)
                
                if subject_data['found']:
                    # Check if the object appears in any of the subject's attributes
                    fact_verified = self._check_entity_attributes(subject_data, predicate, object_entity)
                    
                    if fact_verified:
                        verification_result['verified'] = True
                        verification_result['confidence'] = fact_verified['confidence']
                        verification_result['sources'] = fact_verified['sources']
                    
                    # If still not verified and we have time, try one simplified SPARQL query
                    elif not fact_verified and (time.time() - start_time) < (self.max_query_time - 1.0):
                        try:
                            wikidata_verification = self._verify_fact_with_wikidata_fast(subject, predicate, object_entity)
                            if wikidata_verification.get('verified', False):
                                verification_result['verified'] = True
                                verification_result['confidence'] = wikidata_verification['confidence']
                                verification_result['sources'].append('wikidata_sparql')
                        except Exception as e:
                            self.logger.warning(f"Fast Wikidata verification failed: {str(e)}")
            
            # Cache the result
            self.cache[cache_key] = verification_result
            
            query_time = time.time() - start_time
            self.logger.info(f"Fact verification completed in {query_time:.2f}s")
            
            return verification_result
                
        except Exception as e:
            self.logger.error(f"Error verifying fact triple: {str(e)}")
            verification_result['error'] = str(e)
            return verification_result
    
    def _quick_heuristic_check(self, subject: str, predicate: str, object_entity: str) -> bool:
        """Quick heuristic check for obvious fact relationships.
        
        Args:
            subject: Subject entity
            predicate: Relation predicate  
            object_entity: Object entity
            
        Returns:
            Boolean indicating if fact seems plausible
        """
        # Convert to lowercase for comparison
        subj_lower = subject.lower()
        pred_lower = predicate.lower()
        obj_lower = object_entity.lower()
        
        # Check for obvious relationships
        if pred_lower in ['related to', 'involves', 'mentions', 'about']:
            # These are very general relationships - check for word overlap
            subj_words = set(subj_lower.split())
            obj_words = set(obj_lower.split())
            
            # If there's significant word overlap, it's likely related
            overlap = subj_words.intersection(obj_words)
            if len(overlap) > 0 and len(overlap) >= min(len(subj_words), len(obj_words)) * 0.3:
                return True
                
            # Check for common entity patterns
            if any(word in subj_lower for word in ['election', 'president', 'government']) and \
               any(word in obj_lower for word in ['election', 'president', 'government', 'vote', 'fraud']):
                return True
                
            if any(word in subj_lower for word in ['covid', 'vaccine', 'virus']) and \
               any(word in obj_lower for word in ['covid', 'vaccine', 'virus', 'health', 'medical']):
                return True
        
        return False
    
    def _check_entity_attributes(self, subject_data: Dict[str, Any], predicate: str, object_entity: str) -> Optional[Dict[str, Any]]:
        """Check if object appears in subject's attributes.
        
        Args:
            subject_data: Subject entity data
            predicate: Relation predicate
            object_entity: Object entity
            
        Returns:
            Dictionary with verification result or None if not found
        """
        # Normalize predicate
        normalized_predicate = predicate.lower().replace(' ', '_')
        
        # Check if any attribute values match the object
        for attr_name, attr_value in subject_data['attributes'].items():
            # Check for exact match of attribute name to predicate
            if attr_name.lower() == normalized_predicate or attr_name.lower().replace('_', ' ') == predicate.lower():
                if isinstance(attr_value, str) and self._are_entities_similar(attr_value, object_entity):
                    return {
                        'confidence': 0.9,
                        'sources': [f"direct match in {' '.join(subject_data['sources'])}"]
                    }
                    
            # Check for object in attribute value (for list-type attributes)
            elif isinstance(attr_value, list):
                for val in attr_value:
                    if isinstance(val, str) and self._are_entities_similar(val, object_entity):
                        return {
                            'confidence': 0.8,
                            'sources': [f"list match in {' '.join(subject_data['sources'])}"]
                        }
                        
            # Check if attribute value contains object as substring (lower confidence)
            elif isinstance(attr_value, str) and object_entity.lower() in attr_value.lower():
                return {
                    'confidence': 0.6,
                    'sources': [f"substring match in {' '.join(subject_data['sources'])}"]
                }
        
        return None
    
    def _verify_fact_with_wikidata_fast(self, subject: str, predicate: str, object_entity: str) -> Dict[str, Any]:
        """Fast Wikidata verification with simplified query and timeout protection.
        
        Args:
            subject: Subject entity
            predicate: Relation predicate
            object_entity: Object entity
            
        Returns:
            Dictionary containing verification result
        """
        try:
            # Very simple query that just checks if both entities exist and are related
            query = f"""
            SELECT ?s ?o
            WHERE {{
              ?s rdfs:label "{subject}"@en.
              ?o rdfs:label "{object_entity}"@en.
              ?s ?p ?o.
            }}
            LIMIT 1
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'TruthScan/2.0'
            }
            
            response = requests.get(
                self.kg_endpoints['wikidata'],
                params={'query': query, 'format': 'json'},
                headers=headers,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                
                if results:
                    return {
                        'verified': True,
                        'confidence': 0.7,  # Good confidence from direct SPARQL query
                        'source': 'wikidata'
                    }
            
            return {'verified': False, 'confidence': 0.0}
                
        except requests.Timeout:
            self.logger.warning(f"Wikidata verification timeout for: {subject} {predicate} {object_entity}")
            return {'verified': False, 'confidence': 0.0, 'error': 'timeout'}
        except Exception as e:
            self.logger.error(f"Error in fast Wikidata verification: {str(e)}")
            return {'verified': False, 'confidence': 0.0, 'error': str(e)}
    
    def _verify_fact_with_wikidata(self, subject: str, predicate: str, object_entity: str) -> Dict[str, Any]:
        """Verify a fact using direct SPARQL query to Wikidata.
        
        Args:
            subject: Subject entity
            predicate: Relation predicate
            object_entity: Object entity
            
        Returns:
            Dictionary containing verification result
        """
        try:
            import requests
            
            # Map common predicates to Wikidata properties
            predicate_map = {
                'born in': 'wdt:P19',  # place of birth
                'died in': 'wdt:P20',  # place of death
                'founded': 'wdt:P571',  # inception date
                'capital': 'wdt:P36',  # capital
                'population': 'wdt:P1082',  # population
                'spouse': 'wdt:P26',  # spouse
                'occupation': 'wdt:P106',  # occupation
                'nationality': 'wdt:P27',  # country of citizenship
                'author': 'wdt:P50',  # author
                'director': 'wdt:P57',  # director
                'located in': 'wdt:P131'  # located in administrative entity
            }
            
            # Default to a generic "has property" relation if no mapping exists
            wdt_predicate = predicate_map.get(predicate.lower(), 'wdt:P')
            
            # Query to find entities that match the subject and have the object as a property value
            query = f"""
            SELECT ?subject ?object ?objectLabel
            WHERE {{
              ?subject rdfs:label ?subjectLabel.
              ?subject {wdt_predicate} ?object.
              ?object rdfs:label ?objectLabel.
              
              FILTER(LANG(?subjectLabel) = "en" && REGEX(?subjectLabel, "^{subject}$", "i"))
              FILTER(LANG(?objectLabel) = "en" && REGEX(?objectLabel, "^{object_entity}$", "i"))
              
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 5
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'FakeNewsDetector/1.0 (your-email@example.com)'
            }
            
            response = requests.get(
                self.kg_endpoints['wikidata'],
                params={'query': query, 'format': 'json'},
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                
                if results:
                    return {
                        'verified': True,
                        'confidence': 0.85,  # High confidence from direct SPARQL query
                        'source': 'wikidata'
                    }
            
            # Try a more flexible query if the first one fails
            fuzzy_query = f"""
            SELECT ?subject ?predicate ?object ?objectLabel
            WHERE {{
              ?subject rdfs:label ?subjectLabel.
              ?subject ?predicate ?object.
              ?object rdfs:label ?objectLabel.
              
              FILTER(LANG(?subjectLabel) = "en" && REGEX(?subjectLabel, "^{subject}$", "i"))
              FILTER(LANG(?objectLabel) = "en" && REGEX(?objectLabel, "{object_entity}", "i"))
              
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 10
            """
            
            fuzzy_response = requests.get(
                self.kg_endpoints['wikidata'],
                params={'query': fuzzy_query, 'format': 'json'},
                headers=headers
            )
            
            if fuzzy_response.status_code == 200:
                fuzzy_data = fuzzy_response.json()
                fuzzy_results = fuzzy_data.get('results', {}).get('bindings', [])
                
                if fuzzy_results:
                    return {
                        'verified': True,
                        'confidence': 0.6,  # Lower confidence for fuzzy match
                        'source': 'wikidata'
                    }
            
            return {'verified': False, 'confidence': 0.0}
                
        except Exception as e:
            self.logger.error(f"Error verifying with Wikidata SPARQL: {str(e)}")
            return {'verified': False, 'confidence': 0.0, 'error': str(e)}
    
    def _are_entities_similar(self, entity1: str, entity2: str) -> bool:
        """Check if two entity strings are similar enough to be considered a match.
        
        Args:
            entity1: First entity string
            entity2: Second entity string
            
        Returns:
            Boolean indicating similarity
        """
        # Simple normalization and comparison
        e1 = entity1.lower().strip()
        e2 = entity2.lower().strip()
        
        # Check for exact match after normalization
        if e1 == e2:
            return True
            
        # Check for substring match
        if e1 in e2 or e2 in e1:
            return True
            
        # Check for significant word overlap
        words1 = set(e1.split())
        words2 = set(e2.split())
        overlap = words1.intersection(words2)
        
        if len(overlap) >= min(len(words1), len(words2)) / 2:
            return True
            
        return False
    
    def get_entity_relations(self, entity: str, max_relations: int = 10) -> List[Dict[str, Any]]:
        """Get important relations for an entity from knowledge graphs.
        
        Args:
            entity: Entity name to query
            max_relations: Maximum number of relations to return
            
        Returns:
            List of relation dictionaries containing subject, predicate, object
        """
        self.logger.info(f"Getting relations for entity: {entity}")
        
        try:
            import requests
            
            # First, get entity information to get its Wikidata ID
            entity_data = self.query_entity(entity)
            
            if not entity_data['found'] or 'wikidata_uri' not in entity_data['attributes']:
                return []
                
            wikidata_uri = entity_data['attributes']['wikidata_uri']
            
            # Query to get important relations
            query = f"""
            SELECT ?predicate ?predicateLabel ?object ?objectLabel
            WHERE {{
              <{wikidata_uri}> ?predicate ?object.
              
              # Filter for direct properties
              FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/direct/"))
              
              # Only get objects with labels
              ?object rdfs:label ?objectLabel.
              FILTER(LANG(?objectLabel) = "en")
              
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT {max_relations}
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'FakeNewsDetector/1.0 (your-email@example.com)'
            }
            
            response = requests.get(
                self.kg_endpoints['wikidata'],
                params={'query': query, 'format': 'json'},
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                
                relations = []
                for result in results:
                    if result.get('predicateLabel', {}).get('value') and result.get('objectLabel', {}).get('value'):
                        relation = {
                            'subject': entity,
                            'predicate': result['predicateLabel']['value'],
                            'object': result['objectLabel']['value'],
                            'source': 'wikidata'
                        }
                        relations.append(relation)
                
                return relations
            
            return []
                
        except Exception as e:
            self.logger.error(f"Error getting entity relations: {str(e)}")
            return []