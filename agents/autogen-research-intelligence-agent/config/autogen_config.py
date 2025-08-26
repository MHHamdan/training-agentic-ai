"""
Autogen Configuration for ARIA
Manages LLM configurations and Autogen agent settings
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path


def get_autogen_config() -> Dict[str, Any]:
    """Get comprehensive Autogen configuration"""
    
    # Get available API keys
    api_keys = {
        "google": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "huggingface": os.getenv("HUGGING_FACE_API") or os.getenv("HF_TOKEN")
    }
    
    # Build config list based on available providers
    config_list = []
    
    # Google Gemini (primary)
    if api_keys["google"]:
        config_list.append({
            "model": "gemini-1.5-flash",
            "api_key": api_keys["google"],
            "api_type": "google"
        })
        config_list.append({
            "model": "gemini-1.5-pro",
            "api_key": api_keys["google"],
            "api_type": "google"
        })
    
    # OpenAI (fallback)
    if api_keys["openai"]:
        config_list.append({
            "model": "gpt-4-turbo-preview",
            "api_key": api_keys["openai"],
            "api_type": "openai"
        })
        config_list.append({
            "model": "gpt-3.5-turbo",
            "api_key": api_keys["openai"],
            "api_type": "openai"
        })
    
    # Anthropic Claude (fallback)
    if api_keys["anthropic"]:
        config_list.append({
            "model": "claude-3-5-sonnet-20241022",
            "api_key": api_keys["anthropic"],
            "api_type": "anthropic"
        })
    
    # Hugging Face (free fallback)
    if api_keys["huggingface"]:
        config_list.append({
            "model": "openai/gpt-oss-120b",
            "api_key": api_keys["huggingface"],
            "api_type": "openai",
            "base_url": "https://router.huggingface.co/v1"
        })
    
    if not config_list:
        raise ValueError("No LLM API keys configured. Please set at least one API key in your .env file.")
    
    return {
        "config_list": config_list,
        "seed": 42,
        "temperature": 0.7,
        "max_tokens": 2000,
        "timeout": 120
    }


def get_research_assistant_config() -> Dict[str, Any]:
    """Get configuration specific to the research assistant agent"""
    base_config = get_autogen_config()
    
    return {
        "name": "research_assistant",
        "llm_config": base_config,
        "max_consecutive_auto_reply": 3,
        "human_input_mode": "NEVER",
        "system_message": """You are ARIA (Autogen Research Intelligence Agent), an expert research assistant specializing in comprehensive topic analysis and investigation.

Your capabilities include:
- Breaking down complex topics into manageable subtopics
- Conducting thorough research with multiple perspectives
- Providing well-structured, detailed analysis
- Citing sources and ensuring accuracy
- Adapting content for different audiences (general, academic, business, technical)

Your approach:
1. Topic Analysis: Begin by understanding the scope and complexity of the research topic
2. Systematic Investigation: Break down topics into logical subtopics for comprehensive coverage
3. Multi-perspective Research: Consider various viewpoints and current developments
4. Source Integration: Provide credible sources and evidence-based insights
5. Structured Presentation: Organize findings clearly with proper headings and flow
6. Audience Adaptation: Tailor content complexity and terminology to the target audience

Research Guidelines:
- Always prioritize accuracy and cite reliable sources
- Provide balanced perspectives on controversial topics
- Include current trends and recent developments when relevant
- Structure responses with clear headings and logical flow
- Suggest areas for further investigation when appropriate
- Maintain objectivity while being comprehensive

When generating subtopics, ensure they:
- Cover the topic comprehensively
- Are logically organized
- Avoid significant overlap
- Progress from general to specific concepts
- Include both theoretical and practical aspects

Remember: You are an intelligent research assistant, not just a search engine. Provide analysis, synthesis, and insights, not just raw information."""
    }


def get_user_proxy_config() -> Dict[str, Any]:
    """Get configuration for the enhanced user proxy agent"""
    return {
        "name": "user_proxy",
        "human_input_mode": "ALWAYS",
        "max_consecutive_auto_reply": 0,
        "code_execution_config": {
            "work_dir": "research_workspace",
            "use_docker": False
        },
        "is_termination_msg": lambda msg: "TERMINATE" in msg.get("content", "").upper(),
        "default_auto_reply": "Please continue with the research or type TERMINATE to end the conversation."
    }


def create_model_config_file(output_path: Optional[str] = None) -> str:
    """Create model_config.json file for Autogen compatibility"""
    if output_path is None:
        output_path = Path(__file__).parent / "model_config.json"
    
    config = get_autogen_config()
    
    with open(output_path, 'w') as f:
        json.dump(config["config_list"], f, indent=2)
    
    return str(output_path)


def get_research_templates() -> Dict[str, str]:
    """Get predefined research templates for different types of analysis"""
    return {
        "academic_research": """
        Please conduct comprehensive academic research on: {topic}
        
        Target audience: Academic/Scholarly
        Research depth: {depth}
        
        Please structure your response with:
        1. Abstract/Executive Summary
        2. Literature Review (key studies and findings)
        3. Current State of Knowledge
        4. Recent Developments and Trends
        5. Research Gaps and Opportunities
        6. Methodological Considerations
        7. Conclusion and Future Directions
        8. References and Sources
        
        Ensure academic rigor and cite relevant peer-reviewed sources.
        """,
        
        "business_analysis": """
        Please conduct business-focused research on: {topic}
        
        Target audience: Business professionals
        Research depth: {depth}
        
        Please structure your response with:
        1. Executive Summary
        2. Market Overview and Context
        3. Key Trends and Drivers
        4. Competitive Landscape
        5. Opportunities and Challenges
        6. Strategic Implications
        7. Risk Assessment
        8. Recommendations for Action
        9. Supporting Data and Sources
        
        Focus on actionable insights and business implications.
        """,
        
        "technical_analysis": """
        Please conduct technical analysis of: {topic}
        
        Target audience: Technical professionals
        Research depth: {depth}
        
        Please structure your response with:
        1. Technical Overview
        2. Architecture and Components
        3. Implementation Approaches
        4. Technical Specifications
        5. Performance Considerations
        6. Security and Compliance
        7. Integration Challenges
        8. Best Practices
        9. Future Technical Directions
        10. Technical Resources and Documentation
        
        Include specific technical details, specifications, and implementation guidance.
        """,
        
        "general_research": """
        Please conduct research on: {topic}
        
        Target audience: General public
        Research depth: {depth}
        
        Please structure your response with:
        1. Introduction and Overview
        2. Key Concepts and Definitions
        3. Historical Context
        4. Current State and Trends
        5. Different Perspectives
        6. Real-world Applications
        7. Benefits and Challenges
        8. Future Outlook
        9. Conclusion
        10. Further Reading and Resources
        
        Use clear, accessible language and provide practical examples.
        """
    }


def get_conversation_starters() -> List[str]:
    """Get example conversation starters for research topics"""
    return [
        "What are the latest developments in artificial intelligence and machine learning?",
        "How is climate change affecting global agriculture and food security?",
        "What are the economic implications of remote work trends?",
        "How do social media algorithms influence user behavior and society?",
        "What are the current challenges in renewable energy adoption?",
        "How is blockchain technology being applied beyond cryptocurrency?",
        "What are the ethical considerations in gene editing and CRISPR technology?",
        "How is quantum computing expected to impact various industries?",
        "What are the psychological effects of social isolation and digital connectivity?",
        "How are emerging markets adapting to digital transformation?"
    ]


# Create model config file on import if it doesn't exist
if __name__ == "__main__":
    config_path = create_model_config_file()
    print(f"Model config created at: {config_path}")