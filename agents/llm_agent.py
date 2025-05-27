import os
import re
import json
from openai import OpenAI
from typing import Dict, List, Any
from dotenv import load_dotenv

from .base_agent import Agent

class LLMDecisionAgent(Agent):
    """Agent that uses LLM (via OpenRouter) to make final decisions about news claims."""
    
    def __init__(self, name: str = "LLMDecisionMaker", 
                 api_key: str = None, 
                 model: str = "deepseek/deepseek-r1-distill-qwen-32b:free",
                 site_url: str = "https://fakenewsdetector.org",
                 site_name: str = "Fake News Detection System",
                 verbose: bool = True):
        """Initialize the LLM decision agent."""
        super().__init__(name)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.verbose = verbose
        self.client = None
        
        if not self.api_key:
            self.log("No OpenRouter API key provided. Agent will run in limited mode.", "warning")
        else:
            # Only initialize the client if we have an API key
            try:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                )
                self.log(f"Successfully initialized OpenAI client with OpenRouter API")
            except Exception as e:
                self.log(f"Error initializing OpenAI client: {str(e)}", "error")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and make a decision using LLM."""
        # Extract input data
        claim = input_data.get('claim', '')
        fact_check_results = input_data.get('fact_check_results', {})
        bias_results = input_data.get('bias_results', {})
        source = input_data.get('source', 'unknown')
        evidence = input_data.get('evidence', [])
        
        self.log(f"Making decision for claim: '{claim}' from {source}")
        
        # If no API key or client initialization failed, fall back to rule-based decision
        if not self.api_key or not self.client:
            self.log("No API key or client unavailable, using fallback decision logic", "warning")
            return self._fallback_decision(input_data)
        
        # Construct prompt for LLM
        prompt = self._construct_prompt(claim, fact_check_results, bias_results, source, evidence)
        
        try:
            # Call OpenRouter API - only show a simple message, not model details
            if self.verbose:
                print(f"   Processing claim with AI decision system...")
            
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a specialized news verification AI that makes final decisions about the veracity of news claims based on multiple sources of evidence and analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent outputs
                max_tokens=1024
            )
            
            # Extract response
            response = completion.choices[0].message.content
            
            # Remove verbose output - don't print the raw LLM response
            # Instead, log it at debug level so it's only in the logs
            self.log(f"LLM raw response: {response}", level="debug")
            
            # Parse response to extract structured decision
            decision = self._parse_llm_response(response, claim)
            
            # Log the parsed decision at debug level, don't print to console
            self.log(f"Parsed decision: {decision}", level="debug")
            
            # Return the LLM's decision
            return decision
            
        except Exception as e:
            self.log(f"Error calling LLM API: {str(e)}", "error")
            # Fall back to rule-based decision
            return self._fallback_decision(input_data)
    
    def _parse_llm_response(self, response: str, claim: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured decision."""
        try:
            # Extract JSON from response
            json_start = response.find('```json')
            json_end = response.rfind('```')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                # Extract JSON string
                json_str = response[json_start + 7:json_end].strip()
                # Parse JSON
                decision = json.loads(json_str)
                
                # Ensure required fields exist
                if not all(k in decision for k in ['verdict', 'trust_score', 'explanation']):
                    raise ValueError("Response missing required fields")
                
                # Normalize verdict to ensure consistent formatting
                verdict_map = {
                    'TRUE': 'true',
                    'MOSTLY TRUE': 'mostly true',
                    'PARTIALLY TRUE': 'partially true',
                    'UNVERIFIED': 'unverified',
                    'MOSTLY FALSE': 'mostly false',
                    'FALSE': 'false'
                }
                
                verdict = decision.get('verdict', '').upper()
                decision['verdict'] = verdict_map.get(verdict, 'unverified')
                
                # Ensure trust score is between 0 and 1
                decision['trust_score'] = max(0, min(1, float(decision.get('trust_score', 0.5))))
                
                # CRITICAL FIX: Make sure we log the parsed decision
                if self.verbose:
                    print(f"\nParsed decision: {decision}")
                
                return decision
            else:
                # Attempt to extract information with regex
                # Try to extract verdict
                verdict_match = re.search(r'VERDICT:\s*(TRUE|MOSTLY TRUE|PARTIALLY TRUE|UNVERIFIED|MOSTLY FALSE|FALSE)', response, re.IGNORECASE)
                verdict = verdict_match.group(1).lower() if verdict_match else 'unverified'
                
                # Try to extract trust score
                score_match = re.search(r'TRUST_SCORE:\s*(0\.\d+|\d+)', response)
                trust_score = float(score_match.group(1)) if score_match else 0.5
                
                # Extract explanation - everything between EXPLANATION: and the next heading
                explanation_match = re.search(r'EXPLANATION:(.*?)(?:CONTRIBUTING_FACTORS:|$)', response, re.DOTALL)
                explanation = explanation_match.group(1).strip() if explanation_match else "Could not determine explanation from LLM response."
                
                # Extract contributing factors
                factors_match = re.search(r'CONTRIBUTING_FACTORS:(.*?)(?:$)', response, re.DOTALL)
                factors_text = factors_match.group(1) if factors_match else ""
                
                # Extract individual factors using bullet points or numbered lists
                factors = re.findall(r'[-*\d+]\s*(.*?)(?:\n|$)', factors_text)
                if not factors:
                    # Try splitting by commas or semicolons
                    factors = [f.strip() for f in re.split(r'[,;]', factors_text) if f.strip()]
                
                decision = {
                    'verdict': verdict,
                    'trust_score': max(0, min(1, trust_score)),
                    'explanation': explanation,
                    'contributing_factors': factors or ["Based on evidence analysis", "Source credibility", "Fact-checking results"]
                }
                
                # CRITICAL FIX: Make sure we log the regex-parsed decision
                if self.verbose:
                    print(f"\nRegex-parsed decision: {decision}")
                
                return decision
        except Exception as e:
            self.log(f"Error parsing LLM response: {str(e)}", "error")
            # Provide a fallback decision
            return {
                'verdict': 'unverified',
                'trust_score': 0.5,
                'explanation': f"Could not determine veracity of claim: '{claim}'. The LLM response could not be properly parsed.",
                'contributing_factors': ["Insufficient evidence", "Processing error"]
            }
    
    def _fallback_decision(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback decision-making when LLM is unavailable."""
        self.log("Using fallback decision logic", "warning")
        
        # Extract key inputs
        claim = input_data.get('claim', '')
        fact_check_results = input_data.get('fact_check_results', {})
        bias_results = input_data.get('bias_results', {})
        
        # Get verification score from fact checking
        verification_score = fact_check_results.get('verification_score', 0.5)
        
        # Get bias score from bias analysis
        bias_score = bias_results.get('bias_score', 0.5) if isinstance(bias_results, dict) else 0.5
        
        # Adjust verification score based on bias
        # High bias reduces trust, but not too aggressively
        adjusted_score = verification_score * (1 - bias_score * 0.3)
        
        # Determine verdict based on score with more balanced thresholds
        if adjusted_score >= 0.75:
            verdict = "true"
        elif adjusted_score >= 0.55:
            verdict = "mostly true"
        elif adjusted_score >= 0.35:
            verdict = "partially true"
        elif adjusted_score >= 0.15:
            verdict = "mostly false"
        else:
            verdict = "false"
        
        # Generate explanation
        if verification_score >= 0.7 and bias_score <= 0.3:
            explanation = f"The claim appears to be accurate based on factual evidence. Multiple sources confirm key aspects of the claim."
        elif verification_score >= 0.7 and bias_score > 0.3:
            explanation = f"While factually supported, the claim shows significant bias in presentation."
        elif verification_score >= 0.4:
            explanation = f"The claim contains some accurate elements but may be missing context or includes some inaccuracies."
        else:
            explanation = f"The claim is contradicted by available evidence or lacks sufficient supporting evidence."
        
        # Add bias information to explanation
        if bias_score >= 0.7:
            explanation += " The claim exhibits strong bias in its presentation."
        elif bias_score >= 0.4:
            explanation += " The claim shows moderate bias in its framing."
        
        decision = {
            'verdict': verdict,
            'trust_score': adjusted_score,
            'explanation': explanation,
            'contributing_factors': [
                "Fact verification results",
                "Source credibility analysis",
                "Bias detection"
            ]
        }
        
        # CRITICAL FIX: Make sure we log when fallback is used
        if self.verbose:
            print(f"\nUsing fallback decision: {decision}")
            
        return decision
    
    def _construct_prompt(self, claim: str, fact_check_results: Dict[str, Any], 
                          bias_results: Dict[str, Any], source: str, 
                          evidence: List[Dict[str, Any]]) -> str:
        """Construct prompt for the LLM."""
        # Format evidence articles
        evidence_text = ""
        for i, article in enumerate(evidence[:5]):  # Increased to top 5 articles for better context
            title = article.get('title', 'Untitled')
            text = article.get('text', '')
            article_source = article.get('source', 'Unknown source')
            evidence_text += f"ARTICLE {i+1}:\nTitle: {title}\nSource: {article_source}\nExcerpt: {text[:300]}...\n\n"
        
        # Format fact-checking results
        fc_score = fact_check_results.get('verification_score', 0)
        fc_explanation = fact_check_results.get('explanation', 'No explanation available')
        
        # Format bias results
        bias_score = bias_results.get('bias_score', 0) if isinstance(bias_results, dict) else 0
        bias_indicators = bias_results.get('bias_indicators', []) if isinstance(bias_results, dict) else []
        bias_indicators_text = '\n'.join([f"- {indicator}" for indicator in bias_indicators[:5]])
        if not bias_indicators_text:
            bias_indicators_text = "None detected"
        
        # Construct the full prompt
        prompt = f"""# CLAIM VERIFICATION TASK

        Please analyze the following claim and all available evidence to determine its veracity.
        
        ## CLAIM:
        "{claim}"
        Source: {source}
        
        ## EVIDENCE ARTICLES:
        {evidence_text or "No evidence articles available."}
        
        ## FACT-CHECKING RESULTS:
        Verification Score: {fc_score:.2f} (0-1 scale, higher means more likely to be true)
        Explanation: {fc_explanation}
        
        ## BIAS ANALYSIS:
        Bias Score: {bias_score:.2f} (0-1 scale, higher means more biased)
        Bias Indicators:
        {bias_indicators_text}
        
        ## YOUR TASK:
        Based on all the information above, provide a comprehensive assessment of the claim's veracity. Include:
        
        1. VERDICT: Choose one of [TRUE, MOSTLY TRUE, PARTIALLY TRUE, UNVERIFIED, MOSTLY FALSE, FALSE]
        2. TRUST_SCORE: A numerical score between 0 and 1 indicating confidence in the claim's truthfulness
        3. EXPLANATION: A detailed explanation of your reasoning
        4. CONTRIBUTING_FACTORS: Key factors that influenced your decision
        
        Format your response in JSON as follows:
        ```json
        {{
          "verdict": "VERDICT",
          "trust_score": SCORE,
          "explanation": "Your detailed explanation here",
          "contributing_factors": ["Factor 1", "Factor 2", "Factor 3"]
        }}
        
        Ensure your assessment is balanced, unbiased, and based strictly on the evidence provided.
        """
        return prompt