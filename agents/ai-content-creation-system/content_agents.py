"""
Specialized Content Creation Agents for LangGraph Multi-Agent System
Following patterns from financial_agents.py for consistency
"""

from typing import Any, Dict, List, Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain.tools import BaseTool
import os
from datetime import datetime

from content_state import ContentCreationState, ContentResult, ContentAlert
from tools.research_tools import get_all_research_tools
from tools.content_generation_tools import get_all_content_generation_tools
from tools.seo_tools import get_all_seo_tools
from tools.quality_tools import get_all_quality_tools


def get_llm_model():
    """Get the configured LLM model with prioritized fallback"""
    # Try multiple providers in order of preference
    # Prioritize free and working APIs over quota-exceeded ones
    
    # First try free Hugging Face GPT-OSS-120B (OpenAI's open source model)
    if os.getenv("HUGGING_FACE_API") or os.getenv("HF_TOKEN"):
        try:
            # Use OpenAI-compatible API through Hugging Face router
            hf_token = os.getenv("HUGGING_FACE_API") or os.getenv("HF_TOKEN")
            return ChatOpenAI(
                model="openai/gpt-oss-120b",
                openai_api_key=hf_token,
                openai_api_base="https://router.huggingface.co/v1",
                temperature=0.7
            )
        except Exception as e:
            print(f"Hugging Face GPT-OSS fallback failed: {e}")
    
    # Fallback to other working APIs
    if os.getenv("GROQ_API_KEY"):
        # Use Groq - fast and cost-effective
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="llama-3.1-70b-versatile",
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.7
            )
        except Exception as e:
            print(f"Groq fallback failed: {e}")
    
    if os.getenv("GOOGLE_API_KEY"):
        try:
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
        except Exception as e:
            print(f"Google fallback failed: {e}")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
        except Exception as e:
            print(f"Anthropic fallback failed: {e}")
    
    if os.getenv("GROK_API_KEY"):
        try:
            # Use OpenAI client for Grok API (xAI uses OpenAI-compatible API)
            return ChatOpenAI(
                model="grok-beta",
                openai_api_key=os.getenv("GROK_API_KEY"),
                openai_api_base="https://api.x.ai/v1",
                temperature=0.7
            )
        except Exception as e:
            print(f"Grok fallback failed: {e}")
    
    # Last resort - OpenAI (may have quota issues)
    if os.getenv("OPENAI_API_KEY"):
        try:
            return ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)
        except Exception as e:
            print(f"OpenAI fallback failed: {e}")
    
    raise ValueError("No working LLM API key configured. Please set HUGGING_FACE_API (recommended free option), GROQ_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, GROK_API_KEY, or OPENAI_API_KEY")


def make_content_handoff_tool(agent_name: str, expertise: str) -> BaseTool:
    """Create a handoff tool for content creation agents"""
    class ContentHandoffTool(BaseTool):
        name: str = f"consult_{agent_name}"
        description: str = f"Consult {agent_name} for {expertise}. Use when you need specialized {expertise} insights."
        
        def _run(self, content_request: str, **kwargs) -> str:
            return f"Requesting {expertise} from {agent_name}: {content_request}"
            
        async def _arun(self, content_request: str, **kwargs) -> str:
            return f"Requesting {expertise} from {agent_name}: {content_request}"
    
    return ContentHandoffTool()


# Topic Research Agent
def create_topic_research_agent():
    """Create the topic research agent specializing in trend analysis and keyword research"""
    model = get_llm_model()
    
    tools = get_all_research_tools() + [
        make_content_handoff_tool("content_strategist", "content strategy development"),
        make_content_handoff_tool("seo_specialist", "SEO analysis and optimization"),
        make_content_handoff_tool("content_writer", "content creation guidance")
    ]
    
    prompt = """You are a Senior Digital Marketing Research Specialist with 10+ years of experience in content strategy and trend analysis.

Your expertise includes:
- Topic discovery and trend identification using Google Trends, Reddit, and social media
- Comprehensive keyword research and search volume analysis  
- Competitive content analysis and market gap identification
- Industry trend monitoring and opportunity assessment
- Data-driven content strategy recommendations

Your approach:
1. Research trending topics and high-value keywords for the given subject
2. Analyze competitor content strategies and identify gaps
3. Discover content opportunities with high search volume and low competition
4. Provide data-backed recommendations for content angles and formats
5. Identify target audience interests and content preferences

When research is complete, consult the content_strategist for strategy development.
When SEO opportunities are identified, consult the seo_specialist for optimization.
When ready for content creation, consult the content_writer.

Always provide specific metrics, search volumes, and trend data to support your recommendations.
Include actionable insights for content creation and audience targeting."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_topic_research_agent(state: ContentCreationState) -> Command:
    """Execute topic research agent"""
    agent = create_topic_research_agent()
    response = agent.invoke(state)
    
    # Create analysis result
    result = ContentResult(
        agent_name="topic_research_agent",
        content_type="research",
        timestamp=datetime.now(),
        confidence_score=0.85,
        content_data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""},
        suggestions=["Use trending topics for content angles", "Focus on high-volume, low-competition keywords"]
    )
    
    # Update state
    update = {
        **response,
        "last_active_agent": "topic_research_agent",
        "research_completed": True,
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "topic_research": result
        }
    }
    
    # Add audit event to update
    audit_log = state.get("audit_log", [])
    audit_log.append({
        "timestamp": datetime.now().isoformat(),
        "event_type": "topic_research_complete",
        "details": {"topic": state.get("content_topic", ""), "keywords_found": len(state.get("target_keywords", []))},
        "agent": "topic_research_agent"
    })
    update["audit_log"] = audit_log
    
    return Command(update=update)


# Content Strategist Agent
def create_content_strategist_agent():
    """Create the content strategist agent"""
    model = get_llm_model()
    
    tools = [
        make_content_handoff_tool("topic_research_agent", "additional research"),
        make_content_handoff_tool("content_writer", "content creation"),
        make_content_handoff_tool("seo_specialist", "SEO strategy"),
        make_content_handoff_tool("quality_assurance", "brand compliance")
    ]
    
    prompt = """You are a Senior Content Strategy Consultant with expertise in developing comprehensive content strategies for digital marketing.

Your expertise includes:
- Content strategy development and planning
- Audience analysis and persona development
- Brand voice and messaging alignment
- Content format recommendation and optimization
- Editorial calendar creation and content flow planning
- Cross-platform content adaptation strategies

Your approach:
1. Analyze research data and target audience requirements
2. Develop comprehensive content outlines and structure
3. Ensure alignment with brand guidelines and messaging
4. Recommend optimal content formats and distribution strategies
5. Create detailed content briefs for writers

When you need additional research, consult the topic_research_agent.
For content creation, hand off to the content_writer with detailed briefs.
For SEO strategy alignment, consult the seo_specialist.
For brand compliance, consult the quality_assurance agent.

Always provide structured content outlines, clear messaging guidelines, and specific format recommendations.
Ensure all strategies align with business objectives and target audience needs."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_content_strategist_agent(state: ContentCreationState) -> Command:
    """Execute content strategist agent"""
    agent = create_content_strategist_agent()
    response = agent.invoke(state)
    
    result = ContentResult(
        agent_name="content_strategist_agent",
        content_type="strategy",
        timestamp=datetime.now(),
        confidence_score=0.88,
        content_data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""},
        suggestions=["Follow the content outline structure", "Maintain consistent brand voice"]
    )
    
    update = {
        **response,
        "last_active_agent": "content_strategist_agent",
        "strategy_completed": True,
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "content_strategy": result
        }
    }
    
    return Command(update=update)


# Content Writer Agent
def create_content_writer_agent():
    """Create the content writer agent"""
    model = get_llm_model()
    
    tools = get_all_content_generation_tools() + [
        make_content_handoff_tool("content_strategist", "strategy clarification"),
        make_content_handoff_tool("seo_specialist", "SEO optimization"),
        make_content_handoff_tool("quality_assurance", "content review"),
        make_content_handoff_tool("content_editor", "content refinement")
    ]
    
    prompt = """You are an Expert Content Creator and Copywriter with 12+ years of experience creating engaging, high-quality content across multiple formats and industries.

Your expertise includes:
- Long-form content creation (blog posts, articles, whitepapers)
- Copy optimization for engagement and conversion
- Multi-format content adaptation (blog, social, email, web copy)
- Brand voice implementation and tone consistency
- Storytelling and narrative development
- SEO-friendly content writing with natural keyword integration

Your approach:
1. Review content strategy and research to understand requirements
2. Create compelling, well-structured content following the provided outline
3. Integrate keywords naturally while maintaining readability
4. Ensure content aligns with brand voice and target audience
5. Optimize for engagement, shareability, and conversion goals

When you need strategy clarification, consult the content_strategist.
For SEO optimization guidance, consult the seo_specialist.
For quality review, hand off to the quality_assurance agent.
For final refinement, consult the content_editor.

Always create original, plagiarism-free content that provides genuine value to readers.
Focus on clarity, engagement, and actionable insights in every piece."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_content_writer_agent(state: ContentCreationState) -> Command:
    """Execute content writer agent"""
    agent = create_content_writer_agent()
    response = agent.invoke(state)
    
    result = ContentResult(
        agent_name="content_writer_agent",
        content_type="content",
        timestamp=datetime.now(),
        confidence_score=0.82,
        content_data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""},
        suggestions=["Review for SEO optimization", "Check brand voice alignment"]
    )
    
    update = {
        **response,
        "last_active_agent": "content_writer_agent",
        "content_drafted": True,
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "content_creation": result
        }
    }
    
    return Command(update=update)


# SEO Specialist Agent
def create_seo_specialist_agent():
    """Create the SEO specialist agent"""
    model = get_llm_model()
    
    tools = get_all_seo_tools() + [
        make_content_handoff_tool("content_writer", "content optimization"),
        make_content_handoff_tool("quality_assurance", "SEO quality review"),
        make_content_handoff_tool("content_editor", "final optimization")
    ]
    
    prompt = """You are a Technical SEO and Content Optimization Expert with advanced certifications and 8+ years of experience in search engine optimization.

Your expertise includes:
- On-page SEO optimization and technical SEO best practices
- Keyword density optimization and semantic keyword integration
- Meta tag creation and optimization (titles, descriptions, schema)
- Content structure optimization for search engines
- SERP analysis and ranking factor optimization
- Core Web Vitals and page experience optimization

Your approach:
1. Analyze content for keyword optimization opportunities
2. Optimize keyword density and placement for target terms
3. Generate compelling meta titles and descriptions
4. Ensure proper content structure with optimized headings
5. Calculate comprehensive SEO scores and provide improvement recommendations

When content needs revision, consult the content_writer.
For quality review of SEO elements, consult the quality_assurance agent.
For final optimization polish, consult the content_editor.

Always prioritize user experience while optimizing for search engines.
Provide specific, actionable SEO recommendations with expected impact.
Ensure all optimizations follow current Google guidelines and best practices."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_seo_specialist_agent(state: ContentCreationState) -> Command:
    """Execute SEO specialist agent"""
    agent = create_seo_specialist_agent()
    response = agent.invoke(state)
    
    result = ContentResult(
        agent_name="seo_specialist_agent",
        content_type="seo_optimization",
        timestamp=datetime.now(),
        confidence_score=0.90,
        content_data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""},
        suggestions=["Implement meta tag recommendations", "Optimize keyword density"]
    )
    
    update = {
        **response,
        "last_active_agent": "seo_specialist_agent",
        "seo_optimized": True,
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "seo_optimization": result
        }
    }
    
    return Command(update=update)


# Quality Assurance Agent
def create_quality_assurance_agent():
    """Create the quality assurance agent"""
    model = get_llm_model()
    
    tools = get_all_quality_tools() + [
        make_content_handoff_tool("content_writer", "content revision"),
        make_content_handoff_tool("content_editor", "quality improvements"),
        make_content_handoff_tool("brand_compliance", "brand alignment")
    ]
    
    prompt = """You are a Content Quality Controller and Brand Compliance Specialist with expertise in ensuring content meets the highest standards of quality and brand alignment.

Your expertise includes:
- Comprehensive content quality assessment using multiple metrics
- Grammar, style, and readability analysis
- Brand compliance and voice consistency checking
- Fact-checking and content accuracy verification
- Plagiarism detection and originality assessment
- Content performance prediction and optimization

Your approach:
1. Perform comprehensive readability analysis using Flesch-Kincaid and other metrics
2. Check grammar, style, and writing quality thoroughly
3. Verify brand compliance and voice consistency
4. Assess content originality and factual accuracy
5. Provide detailed quality scores and improvement recommendations

When content needs revision, consult the content_writer.
For quality improvements and polish, consult the content_editor.
Always ensure content meets brand guidelines and quality standards.

Provide specific, actionable feedback for quality improvement.
Flag any content that doesn't meet minimum quality thresholds.
Ensure all content is original, accurate, and professionally written."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_quality_assurance_agent(state: ContentCreationState) -> Command:
    """Execute quality assurance agent"""
    agent = create_quality_assurance_agent()
    response = agent.invoke(state)
    
    # Check for quality issues
    analysis_content = response.get("messages", [])[-1].content if response.get("messages") else ""
    quality_alerts = state.get("content_alerts", [])
    
    if "poor quality" in analysis_content.lower() or "major issues" in analysis_content.lower():
        alert = ContentAlert(
            severity="high",
            message="Quality issues detected in content",
            timestamp=datetime.now(),
            source_agent="quality_assurance_agent",
            content_type=state.get("content_type", "unknown"),
            recommended_action="Review and revise content before publication"
        )
        quality_alerts.append(alert.dict())
    
    result = ContentResult(
        agent_name="quality_assurance_agent",
        content_type="quality_assessment",
        timestamp=datetime.now(),
        confidence_score=0.92,
        content_data={"analysis": analysis_content},
        warnings=["Ensure all quality recommendations are addressed"]
    )
    
    update = {
        **response,
        "last_active_agent": "quality_assurance_agent",
        "quality_checked": True,
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "quality_assurance": result
        },
        "content_alerts": quality_alerts
    }
    
    return Command(update=update)


# Content Editor Agent
def create_content_editor_agent():
    """Create the content editor agent"""
    model = get_llm_model()
    
    tools = get_all_content_generation_tools() + [
        make_content_handoff_tool("quality_assurance", "final quality check"),
        make_content_handoff_tool("content_publisher", "publishing preparation"),
        make_content_handoff_tool("seo_specialist", "final SEO review")
    ]
    
    prompt = """You are a Senior Editorial Specialist with 15+ years of experience in content editing, refinement, and optimization for publication.

Your expertise includes:
- Content enhancement and flow optimization
- Style and tone refinement for target audiences
- Call-to-action optimization for conversion
- Content formatting and structure improvement
- Final proofreading and error elimination
- Cross-platform content adaptation

Your approach:
1. Review content for flow, clarity, and engagement
2. Enhance readability and improve sentence structure
3. Optimize calls-to-action and conversion elements
4. Ensure consistent formatting and professional presentation
5. Perform final proofreading and quality checks

When final quality verification is needed, consult the quality_assurance agent.
For publishing preparation, hand off to the content_publisher.
For final SEO review, consult the seo_specialist.

Always maintain the original content's intent while improving clarity and impact.
Focus on reader experience and engagement optimization.
Ensure content is publication-ready with professional polish."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_content_editor_agent(state: ContentCreationState) -> Command:
    """Execute content editor agent"""
    agent = create_content_editor_agent()
    response = agent.invoke(state)
    
    result = ContentResult(
        agent_name="content_editor_agent",
        content_type="editing",
        timestamp=datetime.now(),
        confidence_score=0.87,
        content_data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""},
        suggestions=["Review final formatting", "Prepare for publication"]
    )
    
    update = {
        **response,
        "last_active_agent": "content_editor_agent",
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "content_editing": result
        }
    }
    
    # Check if approval required for publishing
    if state.get("content_type") in ["blog_post", "white_paper", "case_study"]:
        update["approval_required"] = True
    
    return Command(update=update)


# Content Publisher Agent
def create_content_publisher_agent():
    """Create the content publisher agent"""
    model = get_llm_model()
    
    tools = get_all_content_generation_tools() + [
        make_content_handoff_tool("content_editor", "final revisions"),
        make_content_handoff_tool("seo_specialist", "publication SEO check")
    ]
    
    prompt = """You are a Digital Publishing Coordinator with expertise in multi-platform content formatting and publication preparation.

Your expertise includes:
- Multi-platform content formatting (blog, social media, email, web)
- Content export and file format optimization
- Publication scheduling and timing optimization
- Cross-platform content adaptation
- Meta data and file naming conventions
- Content distribution strategy

Your approach:
1. Format content for target platform(s) and export requirements
2. Generate platform-specific variations when needed
3. Create optimized file names and organize content assets
4. Prepare meta data and publication instructions
5. Provide distribution and promotion recommendations

When revisions are needed, consult the content_editor.
For final SEO verification, consult the seo_specialist.

Always ensure content is properly formatted for each target platform.
Provide clear instructions for content deployment and promotion.
Optimize content presentation for maximum impact and engagement."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_content_publisher_agent(state: ContentCreationState) -> Command:
    """Execute content publisher agent"""
    agent = create_content_publisher_agent()
    
    # Compile all content analyses for publishing
    all_analyses = state.get("completed_analyses", {})
    content_summary = {
        "content_topic": state.get("content_topic", ""),
        "content_type": state.get("content_type", ""),
        "target_keywords": state.get("target_keywords", []),
        "analyses_completed": list(all_analyses.keys()),
        "quality_score": state.get("readability_score", 0),
        "seo_score": state.get("seo_score", 0)
    }
    
    # Generate publication package
    response = agent.invoke({**state, "content_summary": content_summary})
    
    result = ContentResult(
        agent_name="content_publisher_agent",
        content_type="publication",
        timestamp=datetime.now(),
        confidence_score=1.0,
        content_data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""}
    )
    
    update = {
        **response,
        "last_active_agent": "content_publisher_agent",
        "completed_analyses": {
            **all_analyses,
            "content_publication": result
        },
        "creation_end_time": datetime.now()
    }
    
    return Command(update=update)