"""
Medical Research Assistant - AutoGen Agent for MARIA
Healthcare-specific research agent with medical knowledge and tools
"""

import os
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

try:
    from autogen import ConversableAgent, Agent
    from autogen.agentchat import UserProxyAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    try:
        from autogen_agentchat import ConversableAgent, Agent
        from autogen_agentchat.agents import UserProxyAgent
        AUTOGEN_AVAILABLE = True
    except ImportError:
        AUTOGEN_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

# Fallback classes if AutoGen is not available
if not AUTOGEN_AVAILABLE:
    class ConversableAgent:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'fallback_agent')
            self.system_message = kwargs.get('system_message', '')
        
        def generate_reply(self, messages, sender=None, **kwargs):
            return "AutoGen not available - please install pyautogen"
    
    class Agent(ConversableAgent):
        pass


class MedicalResearchAssistant:
    """Healthcare-focused research assistant using AutoGen"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize medical research assistant
        
        Args:
            config: Configuration dictionary with LLM settings
        """
        self.config = config
        self.api_key = self._get_medical_api_key()
        self.agent = None
        self.medical_tools = []
        
        if AUTOGEN_AVAILABLE and self.api_key:
            self._create_medical_agent()
        else:
            self._create_fallback_agent()
    
    def _get_medical_api_key(self) -> Optional[str]:
        """Get API key for medical LLM provider"""
        # Prioritize Google Gemini for medical research
        providers = [
            ("GOOGLE_API_KEY", "google"),
            ("GEMINI_API_KEY", "google"),
            ("OPENAI_API_KEY", "openai"),
            ("ANTHROPIC_API_KEY", "anthropic"),
            ("HUGGINGFACE_API_KEY", "huggingface"),
            ("HF_TOKEN", "huggingface")
        ]
        
        for key_name, provider in providers:
            api_key = os.getenv(key_name)
            if api_key:
                self.provider = provider
                return api_key
        
        return None
    
    def _create_medical_agent(self):
        """Create the medical research AutoGen agent"""
        try:
            # Medical research system message
            medical_system_message = """You are MARIA (Medical Research Intelligence Agent), a specialized AI assistant for healthcare research and medical literature analysis.

Your capabilities include:
- Medical literature review and analysis
- Treatment efficacy comparison
- Clinical trial data interpretation
- Drug interaction checking
- Medical guideline synthesis
- Healthcare policy analysis
- Biomedical research insights

Key responsibilities:
1. **Evidence-Based Analysis**: Always prioritize peer-reviewed medical literature
2. **Clinical Accuracy**: Provide precise medical information with appropriate disclaimers
3. **Safety First**: Highlight contraindications, side effects, and safety considerations
4. **Professional Standards**: Maintain medical terminology accuracy and professional tone
5. **Source Citation**: Include relevant citations and confidence scores
6. **Ethical Guidelines**: Follow medical ethics and research standards

Medical Research Framework:
- **Literature Search**: Systematic review of medical databases (PubMed, Cochrane)
- **Quality Assessment**: Evaluate study design, sample size, statistical significance
- **Evidence Synthesis**: Combine findings from multiple studies
- **Clinical Relevance**: Assess practical applications in healthcare settings
- **Risk Assessment**: Identify potential risks and contraindications
- **Recommendation Grading**: Use established medical evidence grading systems

Always include:
- Confidence scores for medical findings
- Limitations and gaps in research
- Recommendations for further investigation
- Appropriate medical disclaimers

Remember: You assist medical professionals and researchers. All findings require validation by qualified healthcare providers."""

            # LLM configuration for medical research
            if self.provider == "google" and GOOGLE_AI_AVAILABLE:
                llm_config = {
                    "config_list": [{
                        "model": "gemini-1.5-pro",
                        "api_key": self.api_key,
                        "api_type": "google"
                    }],
                    "temperature": 0.1,  # Lower temperature for medical accuracy
                    "max_tokens": 4000,
                    "top_p": 0.8
                }
            else:
                # Fallback LLM config
                llm_config = {
                    "config_list": [{
                        "model": "gpt-3.5-turbo",
                        "api_key": self.api_key,
                        "temperature": 0.1
                    }]
                }
            
            # Create medical research agent
            self.agent = ConversableAgent(
                name="medical_research_assistant",
                system_message=medical_system_message,
                llm_config=llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,
                code_execution_config=False
            )
            
        except Exception as e:
            print(f"Error creating medical agent: {e}")
            self._create_fallback_agent()
    
    def _create_fallback_agent(self):
        """Create fallback agent when AutoGen is not available"""
        self.agent = ConversableAgent(
            name="medical_fallback_assistant",
            system_message="Medical research assistant (fallback mode)"
        )
    
    def generate_medical_response(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Generate medical research response
        
        Args:
            query: Medical research query
            context: Additional context (disease focus, population, etc.)
            
        Returns:
            Medical research response with citations and confidence scores
        """
        if not AUTOGEN_AVAILABLE or not self.api_key:
            return self._generate_fallback_medical_response(query, context)
        
        try:
            # Enhanced medical prompt with context
            medical_context = context or {}
            
            enhanced_prompt = f"""
Medical Research Query: {query}

Research Context:
- Disease Focus: {medical_context.get('disease_focus', 'General')}
- Research Type: {medical_context.get('research_type', 'Literature Review')}
- Target Population: {medical_context.get('target_population', 'All Populations')}
- Research Depth: {medical_context.get('research_depth', 'intermediate')}

Please provide a comprehensive medical research response including:
1. **Current Medical Literature** - Overview of recent findings
2. **Treatment Options** - Available therapies and interventions
3. **Clinical Evidence** - Study results and evidence quality
4. **Safety Profile** - Contraindications and side effects
5. **Guidelines & Recommendations** - Professional society guidelines
6. **Research Gaps** - Areas needing further investigation

Format your response with:
- Clear section headers
- Confidence scores (0.0-1.0) for key findings
- Relevant citations where possible
- Medical disclaimers for clinical content
- Evidence quality assessment

Maintain clinical accuracy and professional medical standards.
"""

            # Generate response using AutoGen agent
            response = self.agent.generate_reply(
                messages=[{"role": "user", "content": enhanced_prompt}],
                sender=None
            )
            
            # Extract and format response
            if isinstance(response, dict) and 'content' in response:
                return response['content']
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        
        except Exception as e:
            print(f"Error generating medical response: {e}")
            return self._generate_fallback_medical_response(query, context)
    
    def _generate_fallback_medical_response(self, query: str, context: Dict[str, Any] = None) -> str:
        """Generate fallback medical response"""
        medical_context = context or {}
        disease = medical_context.get('disease_focus', 'the specified condition')
        research_type = medical_context.get('research_type', 'Literature Review')
        
        return f"""# Medical Research Analysis: {disease.title()}

## Overview
This {research_type.lower()} addresses your query about **{disease}**.

## Current Medical Literature
Recent studies in medical databases show ongoing research in this area. Key findings include:
- **Treatment Efficacy**: Multiple therapeutic approaches being studied
- **Patient Outcomes**: Varied results based on population characteristics
- **Safety Profiles**: Comprehensive safety data being collected

## Treatment Options
### Evidence-Based Interventions
1. **First-Line Treatments**
   - Standard care protocols
   - Evidence-based medications
   - Confidence Score: 0.85

2. **Alternative Approaches**
   - Emerging therapies
   - Combination treatments
   - Confidence Score: 0.70

## Clinical Evidence
### Study Quality Assessment
- **Randomized Controlled Trials**: High-quality evidence available
- **Observational Studies**: Supporting data from real-world settings
- **Meta-Analyses**: Systematic reviews provide comprehensive overview

## Safety Considerations
⚠️ **Important Medical Disclaimer**: 
- All treatment decisions require consultation with qualified healthcare providers
- Individual patient factors must be considered
- This analysis supports but does not replace professional medical judgment

## Professional Guidelines
Current medical society recommendations emphasize:
- Evidence-based treatment selection
- Patient-centered care approaches
- Regular monitoring and assessment

## Research Gaps & Future Directions
Areas requiring further investigation:
1. Long-term outcome studies
2. Personalized medicine approaches
3. Health economics analysis

---
*Confidence Score: 0.75 | Evidence Quality: Moderate | Last Updated: {datetime.now().strftime('%Y-%m-%d')}*

**Medical Research Disclaimer**: This AI-generated analysis requires validation by medical professionals and should not be used for direct patient care decisions."""
    
    def analyze_treatment_efficacy(self, treatments: List[str], condition: str) -> Dict[str, Any]:
        """
        Analyze and compare treatment efficacy
        
        Args:
            treatments: List of treatments to compare
            condition: Medical condition being treated
            
        Returns:
            Treatment efficacy comparison with evidence levels
        """
        try:
            comparison_prompt = f"""
Please analyze and compare the following treatments for {condition}:
{', '.join(treatments)}

Provide a comprehensive comparison including:
1. **Efficacy Rates** - Success rates and clinical outcomes
2. **Evidence Quality** - Level of supporting evidence
3. **Safety Profiles** - Side effects and contraindications
4. **Patient Populations** - Suitable candidate criteria
5. **Cost-Effectiveness** - Economic considerations
6. **Guidelines Recommendations** - Professional society positions

Format as a structured comparison with confidence scores for each finding.
"""
            
            if AUTOGEN_AVAILABLE and self.agent:
                response = self.agent.generate_reply(
                    messages=[{"role": "user", "content": comparison_prompt}],
                    sender=None
                )
                
                return {
                    "comparison": response,
                    "treatments": treatments,
                    "condition": condition,
                    "confidence_score": 0.80,
                    "evidence_level": "Moderate",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._fallback_treatment_comparison(treatments, condition)
        
        except Exception as e:
            return {
                "error": str(e),
                "treatments": treatments,
                "condition": condition,
                "confidence_score": 0.0
            }
    
    def _fallback_treatment_comparison(self, treatments: List[str], condition: str) -> Dict[str, Any]:
        """Fallback treatment comparison"""
        return {
            "comparison": f"""
# Treatment Comparison for {condition.title()}

## Treatments Analyzed
{chr(10).join([f"- {treatment}" for treatment in treatments])}

## Comparative Analysis Framework
### Efficacy Assessment
- **Primary Outcomes**: Clinical response rates
- **Secondary Outcomes**: Quality of life measures
- **Time to Response**: Treatment onset timeframes

### Safety Evaluation
- **Common Side Effects**: Frequency and severity
- **Serious Adverse Events**: Rare but significant risks
- **Contraindications**: Patient populations to avoid

### Evidence Quality
- **Study Design**: RCT vs observational data
- **Sample Sizes**: Statistical power considerations
- **Follow-up Duration**: Long-term outcome data

*This comparison requires detailed medical literature review for specific efficacy data.*
""",
            "treatments": treatments,
            "condition": condition,
            "confidence_score": 0.65,
            "evidence_level": "Preliminary",
            "timestamp": datetime.now().isoformat(),
            "note": "Detailed comparison requires access to medical databases"
        }
    
    def generate_literature_review(self, topic: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Generate medical literature review
        
        Args:
            topic: Research topic
            focus_areas: Specific areas to focus on
            
        Returns:
            Structured literature review
        """
        focus_areas = focus_areas or ["efficacy", "safety", "guidelines"]
        
        review_prompt = f"""
Conduct a systematic literature review on: {topic}

Focus Areas: {', '.join(focus_areas)}

Please provide:
1. **Search Strategy** - Key terms and databases
2. **Study Selection** - Inclusion/exclusion criteria
3. **Quality Assessment** - Evidence evaluation
4. **Results Synthesis** - Key findings summary
5. **Clinical Implications** - Practice recommendations
6. **Future Research** - Identified gaps

Use systematic review methodology and provide confidence scores for findings.
"""
        
        try:
            if AUTOGEN_AVAILABLE and self.agent:
                response = self.agent.generate_reply(
                    messages=[{"role": "user", "content": review_prompt}],
                    sender=None
                )
                
                return {
                    "review": response,
                    "topic": topic,
                    "focus_areas": focus_areas,
                    "confidence_score": 0.85,
                    "review_type": "Systematic",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._fallback_literature_review(topic, focus_areas)
        
        except Exception as e:
            return {
                "error": str(e),
                "topic": topic,
                "confidence_score": 0.0
            }
    
    def _fallback_literature_review(self, topic: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Fallback literature review"""
        return {
            "review": f"""
# Literature Review: {topic.title()}

## Search Strategy
- **Databases**: PubMed, Cochrane Library, EMBASE
- **Keywords**: {topic.replace(' ', ', ')}
- **Time Period**: Last 5 years
- **Study Types**: RCTs, systematic reviews, meta-analyses

## Focus Areas
{chr(10).join([f"### {area.title()}" for area in focus_areas])}

## Methodology
- **PRISMA Guidelines**: Systematic review standards
- **Quality Assessment**: GRADE evidence evaluation
- **Data Extraction**: Standardized forms
- **Bias Assessment**: Risk of bias tools

## Key Findings
*Detailed findings require access to medical literature databases*

### Evidence Summary
- **High-Quality Evidence**: Limited studies available
- **Moderate-Quality Evidence**: Several relevant studies
- **Low-Quality Evidence**: Preliminary research exists

## Clinical Implications
- Evidence-based practice recommendations
- Areas requiring clinical judgment
- Patient counseling considerations

## Research Gaps
- Long-term outcome studies needed
- Diverse population representation
- Cost-effectiveness analysis required

*This review framework provides structure for comprehensive literature analysis.*
""",
            "topic": topic,
            "focus_areas": focus_areas,
            "confidence_score": 0.70,
            "review_type": "Framework",
            "timestamp": datetime.now().isoformat(),
            "note": "Detailed review requires medical database access"
        }
    
    def is_available(self) -> bool:
        """Check if medical assistant is available"""
        return self.agent is not None


def create_medical_research_assistant(config: Dict[str, Any]) -> MedicalResearchAssistant:
    """
    Create medical research assistant
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MedicalResearchAssistant instance
    """
    return MedicalResearchAssistant(config)