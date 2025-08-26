"""
Medical Configuration for MARIA
Healthcare-specific AutoGen and LLM configuration
"""

import os
from typing import Dict, List, Any, Optional

def get_medical_autogen_config() -> Dict[str, Any]:
    """
    Get AutoGen configuration for medical research
    
    Returns:
        Medical AutoGen configuration dictionary
    """
    # Medical LLM provider priority (prioritize reliable providers)
    providers = [
        {
            "name": "google",
            "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
            "medical_optimized": True
        },
        {
            "name": "openai", 
            "api_key": os.getenv("OPENAI_API_KEY"),
            "models": ["gpt-4", "gpt-3.5-turbo"],
            "medical_optimized": True
        },
        {
            "name": "anthropic",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
            "medical_optimized": True
        },
        {
            "name": "huggingface",
            "api_key": os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN"),
            "models": ["microsoft/BioGPT-Large", "dmis-lab/biobert-base-cased-v1.2"],
            "medical_optimized": True
        }
    ]
    
    # Find first available provider
    active_provider = None
    for provider in providers:
        if provider["api_key"]:
            active_provider = provider
            break
    
    if not active_provider:
        # Fallback configuration
        return {
            "provider": "fallback",
            "config_list": [],
            "temperature": 0.1,
            "max_tokens": 2000,
            "medical_settings": get_medical_settings()
        }
    
    # Create LLM config list
    config_list = []
    
    if active_provider["name"] == "google":
        config_list = [{
            "model": "gemini-1.5-pro",
            "api_key": active_provider["api_key"],
            "api_type": "google",
            "temperature": 0.1,  # Low temperature for medical accuracy
            "max_tokens": 4000,
            "top_p": 0.8
        }]
    elif active_provider["name"] == "openai":
        config_list = [{
            "model": "gpt-4",
            "api_key": active_provider["api_key"],
            "api_type": "openai",
            "temperature": 0.1,
            "max_tokens": 3000
        }]
    elif active_provider["name"] == "anthropic":
        config_list = [{
            "model": "claude-3-5-sonnet-20241022",
            "api_key": active_provider["api_key"],
            "api_type": "anthropic",
            "temperature": 0.1,
            "max_tokens": 3000
        }]
    elif active_provider["name"] == "huggingface":
        config_list = [{
            "model": "microsoft/BioGPT-Large",
            "api_key": active_provider["api_key"],
            "api_type": "huggingface",
            "temperature": 0.2,
            "max_tokens": 2000
        }]
    
    return {
        "provider": active_provider["name"],
        "config_list": config_list,
        "temperature": 0.1,  # Conservative temperature for medical content
        "max_tokens": 4000,
        "timeout": 120,  # 2 minute timeout for medical queries
        "medical_settings": get_medical_settings(),
        "cache_seed": None,  # Disable caching for medical content
        "use_docker": False
    }


def get_medical_settings() -> Dict[str, Any]:
    """
    Get medical-specific settings
    
    Returns:
        Medical settings dictionary
    """
    return {
        # Medical content validation
        "require_medical_disclaimer": True,
        "confidence_threshold": 0.8,
        "require_citations": True,
        "enable_safety_checks": True,
        
        # Research standards
        "evidence_levels": ["systematic_review", "rct", "cohort", "case_series", "expert_opinion"],
        "required_evidence_level": "cohort",  # Minimum evidence level
        "max_research_depth": "comprehensive",
        "enable_peer_review_filter": True,
        
        # Clinical safety
        "flag_high_risk_content": True,
        "require_human_approval": True,
        "enable_drug_interaction_check": True,
        "contraindication_warnings": True,
        
        # Professional standards
        "medical_terminology_check": True,
        "clinical_accuracy_validation": True,
        "ethics_compliance_check": True,
        "hipaa_compliance": True,
        
        # Research databases
        "primary_databases": ["pubmed", "cochrane", "embase"],
        "secondary_databases": ["clinicaltrials.gov", "who_ictrp"],
        "enable_real_time_search": False,  # Would require API integrations
        
        # Output formatting
        "include_confidence_scores": True,
        "structured_medical_format": True,
        "enable_medical_citations": True,
        "prisma_compliance": True,
        
        # Quality assurance
        "double_validation": True,
        "bias_assessment": True,
        "statistical_significance_check": True,
        "clinical_relevance_assessment": True
    }


def get_medical_model_preferences() -> Dict[str, List[str]]:
    """
    Get medical model preferences by task
    
    Returns:
        Dictionary mapping medical tasks to preferred models
    """
    return {
        "literature_review": [
            "gemini-1.5-pro",
            "gpt-4", 
            "claude-3-5-sonnet-20241022",
            "microsoft/BioGPT-Large"
        ],
        "clinical_analysis": [
            "gpt-4",
            "gemini-1.5-pro",
            "claude-3-5-sonnet-20241022"
        ],
        "drug_information": [
            "gemini-1.5-pro",
            "gpt-4",
            "microsoft/BioGPT-Large"
        ],
        "treatment_comparison": [
            "gpt-4",
            "claude-3-5-sonnet-20241022",
            "gemini-1.5-pro"
        ],
        "safety_assessment": [
            "gpt-4",
            "gemini-1.5-pro",
            "claude-3-5-sonnet-20241022"
        ],
        "guideline_synthesis": [
            "claude-3-5-sonnet-20241022",
            "gpt-4",
            "gemini-1.5-pro"
        ]
    }


def get_medical_prompt_templates() -> Dict[str, str]:
    """
    Get medical prompt templates
    
    Returns:
        Dictionary of medical prompt templates
    """
    return {
        "literature_review": """
Conduct a systematic literature review on: {topic}

Medical Research Requirements:
- Search Strategy: Define key terms and databases
- Inclusion Criteria: Peer-reviewed medical literature
- Quality Assessment: Evaluate study design and evidence level
- Data Extraction: Key findings and clinical outcomes
- Bias Assessment: Risk of bias evaluation
- Clinical Relevance: Applicability to practice

Focus Areas: {focus_areas}
Target Population: {population}
Time Period: {time_period}

Please provide:
1. **Systematic Search Strategy**
2. **Evidence Quality Assessment** 
3. **Clinical Findings Summary**
4. **Safety and Contraindications**
5. **Professional Guidelines**
6. **Research Gaps and Limitations**

Include confidence scores and appropriate medical disclaimers.
""",
        
        "treatment_analysis": """
Analyze treatment options for: {condition}

Treatment Comparison:
- Treatments: {treatments}
- Population: {population}
- Outcomes: {outcomes}

Medical Analysis Framework:
1. **Efficacy Assessment**: Clinical effectiveness data
2. **Safety Profile**: Adverse events and contraindications
3. **Evidence Quality**: Study design and statistical significance
4. **Guidelines Recommendations**: Professional society positions
5. **Cost-Effectiveness**: Economic considerations
6. **Patient Selection**: Appropriate candidate criteria

Provide structured comparison with confidence scores for each finding.
""",
        
        "clinical_assessment": """
Clinical Assessment Request: {query}

Medical Context:
- Condition: {condition}
- Population: {population}
- Clinical Setting: {setting}

Assessment Requirements:
1. **Clinical Evidence Review**
2. **Diagnostic Considerations**
3. **Treatment Options Analysis**
4. **Risk-Benefit Assessment**
5. **Monitoring Requirements**
6. **Professional Guidelines**

Maintain clinical accuracy and include appropriate medical disclaimers.
""",
        
        "safety_review": """
Safety Review for: {intervention}

Safety Assessment Framework:
- Intervention: {intervention}
- Population: {population}
- Duration: {duration}

Safety Analysis:
1. **Adverse Event Profile**: Common and serious side effects
2. **Contraindications**: Absolute and relative contraindications
3. **Drug Interactions**: Potential medication conflicts
4. **Special Populations**: Pregnancy, pediatric, elderly considerations
5. **Monitoring Requirements**: Laboratory and clinical monitoring
6. **Risk Mitigation**: Strategies to minimize risks

Include safety warnings and contraindication alerts.
"""
    }


def get_medical_validation_rules() -> Dict[str, Any]:
    """
    Get medical content validation rules
    
    Returns:
        Medical validation rules dictionary
    """
    return {
        "content_safety": {
            "prohibited_claims": [
                "cure",
                "guaranteed treatment",
                "medical diagnosis",
                "specific dosing recommendations"
            ],
            "required_disclaimers": [
                "consult healthcare provider",
                "not medical advice",
                "for research purposes",
                "professional validation required"
            ],
            "risk_keywords": [
                "dosage", "dose", "prescription", "treatment",
                "diagnosis", "contraindication", "side effect"
            ]
        },
        
        "evidence_standards": {
            "minimum_evidence_level": "case_series",
            "require_peer_review": True,
            "confidence_threshold": 0.7,
            "citation_required": True,
            "bias_assessment": True
        },
        
        "clinical_accuracy": {
            "medical_terminology_check": True,
            "fact_verification": True,
            "guideline_compliance": True,
            "statistical_validation": True
        },
        
        "ethical_compliance": {
            "patient_privacy": True,
            "informed_consent": True,
            "research_ethics": True,
            "professional_standards": True
        }
    }


def get_medical_api_config() -> Dict[str, Any]:
    """
    Get medical API configuration
    
    Returns:
        Medical API configuration
    """
    return {
        "pubmed": {
            "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            "api_key": os.getenv("PUBMED_API_KEY") or os.getenv("PubMed"),
            "rate_limit": 10,  # requests per second
            "enabled": bool(os.getenv("PUBMED_API_KEY") or os.getenv("PubMed"))
        },
        
        "clinicaltrials": {
            "base_url": "https://clinicaltrials.gov/api/",
            "api_key": None,  # Public API
            "rate_limit": 5,
            "enabled": True
        },
        
        "who_ictrp": {
            "base_url": "https://www.who.int/ictrp/",
            "api_key": None,
            "rate_limit": 2,
            "enabled": False  # Requires manual integration
        },
        
        "fda_api": {
            "base_url": "https://api.fda.gov/",
            "api_key": os.getenv("FDA_API_KEY"),
            "rate_limit": 40,  # requests per minute
            "enabled": bool(os.getenv("FDA_API_KEY"))
        }
    }


def validate_medical_config() -> Dict[str, Any]:
    """
    Validate medical configuration
    
    Returns:
        Configuration validation results
    """
    config = get_medical_autogen_config()
    settings = get_medical_settings()
    api_config = get_medical_api_config()
    
    validation = {
        "config_valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    # Check LLM provider availability
    if config["provider"] == "fallback":
        validation["errors"].append("No LLM provider available - medical research will be limited")
        validation["config_valid"] = False
    
    # Check medical API availability
    enabled_apis = [api for api, conf in api_config.items() if conf.get("enabled")]
    if not enabled_apis:
        validation["warnings"].append("No medical research APIs configured - external data access limited")
    
    # Check safety settings
    if not settings.get("require_medical_disclaimer"):
        validation["warnings"].append("Medical disclaimers not required - consider enabling for safety")
    
    if settings.get("confidence_threshold", 0) < 0.7:
        validation["warnings"].append("Low confidence threshold for medical content")
    
    # Recommendations
    if config["provider"] != "google":
        validation["recommendations"].append("Consider using Google Gemini for medical research")
    
    if not api_config["pubmed"]["enabled"]:
        validation["recommendations"].append("Enable PubMed API for medical literature access")
    
    return validation