"""
TruthScan Web Application
========================

A Flask-based web application for fake news detection and verification.
This application provides a clean, modern interface for users to verify claims,
analyze articles, and get comprehensive fact-checking results.

Author: TruthScan Development Team
Version: 2.0
Last Updated: 2025
"""

import os
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
import logging

# Import our new system configuration
from system_configuration import (
    SystemConfiguration, 
    ProcessingMode, 
    config as default_config
)
from fakenews_multi_agent import FakeNewsDetectionSystem

# Load environment variables
load_dotenv()


class TruthScanApp:
    """
    Main TruthScan Flask application class.
    
    This class encapsulates the Flask application and provides a clean interface
    for managing the fake news detection system and handling web requests.
    """
    
    def __init__(self, processing_mode: ProcessingMode = ProcessingMode.BALANCED):
        """
        Initialize the TruthScan application.
        
        Args:
            processing_mode: The processing mode to use (FAST, BALANCED, or ACCURATE)
        """
        self.config = SystemConfiguration(mode=processing_mode)
        self.config.setup_logging()
        self.logger = logging.getLogger("TruthScanApp")
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        static_folder='static', 
                        template_folder='templates')
        
        # Set up directories
        self._setup_directories()
        
        # Initialize the detection system
        self.detection_system: Optional[FakeNewsDetectionSystem] = None
        
        # Register routes
        self._register_routes()
        self._register_error_handlers()
        
        self.logger.info(f"TruthScan application initialized in {processing_mode.value} mode")
    
    def _setup_directories(self) -> None:
        """Ensure all required directories exist."""
        for directory in self.config.file_paths.values():
            if directory.endswith(('_directory', 'results', 'logs')):
                os.makedirs(directory, exist_ok=True)
    
    def _initialize_detection_system(self) -> bool:
        """
        Initialize the fake news detection system.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self.detection_system is not None:
            return True
            
        self.logger.info("Initializing FakeNewsDetectionSystem...")
        
        try:
            # Log API key availability (without exposing the keys)
            api_status = {
                service: bool(self.config.get_api_key(service))
                for service in ['news_api', 'knowledge_graph_api', 'openrouter_api', 
                              'mediastack_api', 'gnews_api']
            }
            self.logger.info(f"API key status: {api_status}")
            
            # Initialize the detection system with configuration
            self.detection_system = FakeNewsDetectionSystem(
                model_path=None,  # Use fallback mode
                api_key=self.config.get_api_key('news_api'),
                kg_api_key=self.config.get_api_key('knowledge_graph_api'),
                lexicon_path=self.config.file_paths['lexicon_file'],
                subjectivity_lexicon_path=self.config.file_paths['subjectivity_lexicon_file'],
                openrouter_api_key=self.config.get_api_key('openrouter_api'),
                mediastack_key=self.config.get_api_key('mediastack_api'),
                gnews_key=self.config.get_api_key('gnews_api'),
                enable_kg_verification=self.config.knowledge_graph.enabled,
                kg_timeout=self.config.knowledge_graph.timeout_seconds,
                config=self.config  # Pass the configuration instance
            )
            
            self.logger.info("FakeNewsDetectionSystem initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing FakeNewsDetectionSystem: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _register_routes(self) -> None:
        """Register all Flask routes."""
        
        @self.app.route('/')
        def home():
            """Serve the main page."""
            return render_template('index.html')
        
        @self.app.route('/static/<path:path>')
        def serve_static(path):
            """Serve static files."""
            return send_from_directory('static', path)
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'mode': self.config.mode.value,
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'detection_system': self.detection_system is not None,
                    'knowledge_graph': self.config.is_service_enabled('knowledge_graph'),
                    'news_retrieval': self.config.is_service_enabled('news_retrieval'),
                    'llm_decision': self.config.is_service_enabled('llm_decision')
                }
            })
        
        @self.app.route('/api/config')
        def get_config():
            """Get current configuration (without sensitive data)."""
            config_dict = self.config.to_dict()
            # Remove sensitive information
            config_dict.pop('api_keys', None)
            return jsonify(config_dict)
        
        @self.app.route('/detect', methods=['POST'])
        def detect():
            """Main endpoint for claim verification."""
            return self._handle_claim_verification()
        
        @self.app.route('/verify_article', methods=['POST'])
        def verify_article():
            """Endpoint for full article verification."""
            return self._handle_article_verification()
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file uploads for verification."""
            return self._handle_file_upload()
        
        @self.app.route('/download_report/<path:filename>')
        def download_report(filename):
            """Download verification report."""
            return send_from_directory(
                self.config.file_paths['results_directory'], 
                filename, 
                as_attachment=True
            )
    
    def _register_error_handlers(self) -> None:
        """Register error handlers."""
        
        @self.app.errorhandler(404)
        def page_not_found(e):
            """Custom 404 page."""
            return render_template('404.html'), 404
        
        @self.app.errorhandler(500)
        def server_error(e):
            """Custom 500 page."""
            self.logger.error(f"Server error: {str(e)}")
            return render_template('500.html'), 500
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            """Handle unexpected exceptions."""
            self.logger.error(f"Unhandled exception: {str(e)}")
            self.logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': 'An unexpected error occurred. Please try again.'
            }), 500
    
    def _handle_claim_verification(self) -> Dict[str, Any]:
        """Handle claim verification requests."""
        if not self._initialize_detection_system():
            return jsonify({
                'success': False,
                'error': 'Failed to initialize verification system. Check logs for details.'
            }), 500
        
        # Extract request data
        claim = request.form.get('claim', '').strip()
        source = request.form.get('source', '').strip()
        
        if not claim:
            return jsonify({
                'success': False,
                'error': 'No claim provided for verification'
            }), 400
        
        # Validate claim length
        if len(claim) > self.config.security.max_claim_length:
            return jsonify({
                'success': False,
                'error': f'Claim too long. Maximum length is {self.config.security.max_claim_length} characters.'
            }), 400
        
        self.logger.info(f"Processing claim verification: '{claim[:100]}...'")
        
        try:
            # Create response structure
            response = self._create_verification_response(claim, source)
            
            # Step 1: Retrieve articles
            response = self._process_news_retrieval(response, claim)
            if not response['success']:
                return jsonify(response), 500
            
            # Step 2: Fact checking
            response = self._process_fact_checking(response, claim)
            if not response['success']:
                return jsonify(response), 500
            
            # Step 3: Bias analysis
            response = self._process_bias_analysis(response, claim, source)
            if not response['success']:
                return jsonify(response), 500
            
            # Step 4: Final decision
            response = self._process_final_decision(response, claim, source)
            if not response['success']:
                return jsonify(response), 500
            
            # Save results
            self._save_verification_results(response)
            
            self.logger.info(f"Verification complete. Verdict: {response['final_result']['verdict']}")
            return jsonify(response)
            
        except Exception as e:
            self.logger.error(f"Error during verification: {str(e)}")
            self.logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Error during verification: {str(e)}'
            }), 500
    
    def _create_verification_response(self, claim: str, source: str) -> Dict[str, Any]:
        """Create the initial verification response structure."""
        return {
            'success': True,
            'verification_id': datetime.now().strftime("%Y%m%d%H%M%S"),
            'claim': claim,
            'source': source,
            'steps': {
                'step1': {'status': 'pending', 'message': 'Retrieving articles...'},
                'step2': {'status': 'pending', 'message': 'Fact checking...'},
                'step3': {'status': 'pending', 'message': 'Bias analysis...'},
                'step4': {'status': 'pending', 'message': 'Decision making...'}
            }
        }
    
    def _process_news_retrieval(self, response: Dict[str, Any], claim: str) -> Dict[str, Any]:
        """Process news article retrieval."""
        self.logger.info("Step 1: Retrieving news articles...")
        response['steps']['step1']['status'] = 'processing'
        
        try:
            articles = self.detection_system.news_retriever.search_by_claim(claim)
            
            if not articles:
                self.logger.warning("No articles found for claim")
                articles = [{
                    "title": "No articles found",
                    "text": "The system could not find any relevant articles for this claim.",
                    "source": "System",
                    "url": "",
                    "published_at": datetime.now().isoformat()
                }]
            
            # Limit articles based on configuration
            max_articles = self.config.news_retrieval.max_articles_per_query
            articles = articles[:max_articles]
            
            response['steps']['step1'] = {
                'status': 'complete',
                'message': f'Found {len(articles)} relevant articles',
                'data': {'articles': articles}
            }
            response['articles'] = articles
            
        except Exception as e:
            self.logger.error(f"Error in news retrieval: {str(e)}")
            response['steps']['step1'] = {
                'status': 'error',
                'message': 'Error retrieving articles',
                'error': str(e)
            }
            response['success'] = False
        
        return response
    
    def _process_fact_checking(self, response: Dict[str, Any], claim: str) -> Dict[str, Any]:
        """Process fact checking."""
        self.logger.info("Step 2: Performing fact checking...")
        response['steps']['step2']['status'] = 'processing'
        
        try:
            fact_check_input = {
                'claim': claim,
                'evidence': response['articles'],
                'knowledge_graph': self.detection_system.knowledge_graph
            }
            
            fact_check_results = self.detection_system.fact_checker.process(fact_check_input)
            
            response['steps']['step2'] = {
                'status': 'complete',
                'message': 'Fact checking complete',
                'data': fact_check_results
            }
            response['fact_check_results'] = fact_check_results
            
        except Exception as e:
            self.logger.error(f"Error in fact checking: {str(e)}")
            response['steps']['step2'] = {
                'status': 'error',
                'message': 'Error during fact checking',
                'error': str(e)
            }
            response['success'] = False
        
        return response
    
    def _process_bias_analysis(self, response: Dict[str, Any], claim: str, source: str) -> Dict[str, Any]:
        """Process bias analysis."""
        self.logger.info("Step 3: Analyzing bias...")
        response['steps']['step3']['status'] = 'processing'
        
        try:
            bias_input = {
                'title': '',
                'text': claim,
                'source': source
            }
            
            bias_results = self.detection_system.bias_detector.process(bias_input)
            
            response['steps']['step3'] = {
                'status': 'complete',
                'message': 'Bias analysis complete',
                'data': bias_results
            }
            response['bias_results'] = bias_results
            
        except Exception as e:
            self.logger.error(f"Error in bias analysis: {str(e)}")
            response['steps']['step3'] = {
                'status': 'error',
                'message': 'Error during bias analysis',
                'error': str(e)
            }
            response['success'] = False
        
        return response
    
    def _process_final_decision(self, response: Dict[str, Any], claim: str, source: str) -> Dict[str, Any]:
        """Process final decision making."""
        self.logger.info("Step 4: Making final decision...")
        response['steps']['step4']['status'] = 'processing'
        
        try:
            decision_input = {
                'claim': claim,
                'fact_check_results': response['fact_check_results'],
                'bias_results': response['bias_results'],
                'source': source,
                'evidence': response['articles']
            }
            
            decision = self.detection_system.decision_maker.process(decision_input)
            
            response['steps']['step4'] = {
                'status': 'complete',
                'message': 'Decision complete',
                'data': decision
            }
            
            # Add final result summary
            response['final_result'] = {
                'verdict': decision.get('verdict', 'unknown'),
                'trust_score': decision.get('trust_score', 0.5),
                'confidence': decision.get('confidence', decision.get('trust_score', 0.5) * 100),
                'explanation': decision.get('explanation', 'No explanation provided'),
                'contributing_factors': decision.get('contributing_factors', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error in decision making: {str(e)}")
            response['steps']['step4'] = {
                'status': 'error',
                'message': 'Error during decision making',
                'error': str(e)
            }
            response['success'] = False
        
        return response
    
    def _save_verification_results(self, response: Dict[str, Any]) -> None:
        """Save verification results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.config.file_paths['results_directory'],
                f'verification_{timestamp}.json'
            )
            
            # Create a clean results structure for saving
            full_results = {
                'claim': response['claim'],
                'source': response['source'],
                'evidence': response.get('articles', []),
                'fact_check_results': response.get('fact_check_results', {}),
                'bias_analysis': response.get('bias_results', {}),
                'decision': response.get('final_result', {}),
                'timestamp': datetime.now().isoformat(),
                'configuration': {
                    'mode': self.config.mode.value,
                    'knowledge_graph_enabled': self.config.knowledge_graph.enabled
                }
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            
            response['results_file'] = os.path.basename(results_file)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def _handle_article_verification(self) -> Dict[str, Any]:
        """Handle full article verification requests."""
        if not self._initialize_detection_system():
            return jsonify({
                'success': False,
                'error': 'Failed to initialize verification system.'
            }), 500
        
        # Extract article data
        title = request.form.get('title', '').strip()
        text = request.form.get('text', '').strip()
        source = request.form.get('source', '').strip()
        
        if not title and not text:
            return jsonify({
                'success': False,
                'error': 'No article content provided for verification'
            }), 400
        
        try:
            article = {'title': title, 'text': text, 'source': source}
            results = self.detection_system.verify_article(article)
            
            return jsonify({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            self.logger.error(f"Error during article verification: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error during verification: {str(e)}'
            }), 500
    
    def _handle_file_upload(self) -> Dict[str, Any]:
        """Handle file upload requests."""
        if 'docfile' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['docfile']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        claim = request.form.get('claim', '').strip()
        
        try:
            text_content = self._extract_file_content(file)
            full_text = f"{claim}\n{text_content}" if claim else text_content
            
            # Truncate for preview
            preview_text = full_text[:1000] + '...' if len(full_text) > 1000 else full_text
            
            return jsonify({
                'success': True,
                'content': preview_text,
                'message': f'Successfully extracted content from {file.filename}'
            })
            
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error processing file: {str(e)}'
            }), 500
    
    def _extract_file_content(self, file) -> str:
        """Extract text content from uploaded file."""
        filename = file.filename.lower()
        text_content = ""
        
        if filename.endswith('.pdf'):
            try:
                import pdfplumber
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += '\n' + page_text
            except ImportError:
                raise Exception("PDF processing requires pdfplumber. Please install it.")
                
        elif filename.endswith('.docx'):
            try:
                import io
                import mammoth
                docx_buf = io.BytesIO(file.read())
                result = mammoth.extract_raw_text(docx_buf)
                text_content = result.value
            except ImportError:
                raise Exception("DOCX processing requires mammoth. Please install it.")
                
        elif filename.endswith('.txt'):
            text_content = file.read().decode('utf-8')
        else:
            raise Exception(f"Unsupported file type: {filename}")
        
        return text_content
    
    def run(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = True) -> None:
        """
        Run the Flask application.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Whether to run in debug mode
        """
        if self._initialize_detection_system():
            self.logger.info(f"Starting TruthScan application server on {host}:{port}")
            self.app.run(host=host, port=port, debug=debug)
        else:
            self.logger.error("Failed to initialize the verification system. Application cannot start.")


# Create the application instance
def create_app(mode: ProcessingMode = ProcessingMode.BALANCED) -> TruthScanApp:
    """
    Create and configure the TruthScan application.
    
    Args:
        mode: Processing mode to use
        
    Returns:
        Configured TruthScanApp instance
    """
    return TruthScanApp(processing_mode=mode)


# For backward compatibility and direct execution
app_instance = create_app()
app = app_instance.app

if __name__ == '__main__':
    # Allow mode selection via environment variable
    mode_str = os.getenv('TRUTHSCAN_MODE', 'balanced').upper()
    try:
        mode = ProcessingMode(mode_str.lower())
    except ValueError:
        mode = ProcessingMode.BALANCED
    
    # Create and run the application
    truthscan_app = create_app(mode)
    truthscan_app.run()