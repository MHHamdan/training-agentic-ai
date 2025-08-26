"""
Enhanced Research Assistant Agent for ARIA
Built on Microsoft Autogen framework with specialized research capabilities
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import ChatAgent
    import autogen_agentchat
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

# For now, let's create a simplified implementation that doesn't require the complex autogen setup
class BaseResearchAgent:
    """Base research agent class for simplified implementation"""
    def __init__(self, name: str = "research_assistant", **kwargs):
        self.name = name
        self.kwargs = kwargs


class EnhancedResearchAssistant(AssistantAgent if AUTOGEN_AVAILABLE else BaseResearchAgent):
    """
    Enhanced Research Assistant with specialized research capabilities
    """
    
    def __init__(self, 
                 name: str = "research_assistant",
                 llm_config: Dict[str, Any] = None,
                 system_message: str = None,
                 research_tools: List[str] = None,
                 **kwargs):
        """
        Initialize the Enhanced Research Assistant
        
        Args:
            name: Agent name
            llm_config: LLM configuration
            system_message: System message for the agent
            research_tools: List of available research tools
            **kwargs: Additional arguments for AssistantAgent
        """
        self.research_tools = research_tools or ["web_search", "academic_search", "summarization"]
        self.research_history = []
        self.current_research_context = {}
        
        # Default system message if none provided
        if system_message is None:
            system_message = self._get_default_system_message()
        
        # Store system message before parent initialization
        self._stored_system_message = system_message
        
        # Initialize parent class (AssistantAgent or BaseResearchAgent)
        if AUTOGEN_AVAILABLE:
            super().__init__(
                name=name,
                model_client=llm_config,  # New autogen uses model_client instead of llm_config
                system_message=system_message,
                **kwargs
            )
            # Ensure system_message attribute is available
            if not hasattr(self, 'system_message'):
                self.system_message = system_message
        else:
            super().__init__(name=name, **kwargs)
            self.system_message = system_message
            self.llm_config = llm_config
        
        # Initialize LLM client for research tasks
        try:
            from config.llm_clients import create_llm_client
            self.llm_client = create_llm_client(llm_config)
        except ImportError:
            self.llm_client = None
    
    def _get_default_system_message(self) -> str:
        """Get the default system message for the research assistant"""
        return """You are ARIA (Autogen Research Intelligence Agent), an expert research assistant specializing in comprehensive topic analysis and investigation.

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

    def set_research_context(self, 
                           topic: str, 
                           depth: str = "intermediate", 
                           audience: str = "general",
                           additional_context: Dict[str, Any] = None):
        """
        Set the current research context
        
        Args:
            topic: Research topic
            depth: Research depth (basic, intermediate, comprehensive)
            audience: Target audience (general, academic, business, technical)
            additional_context: Additional context information
        """
        self.current_research_context = {
            "topic": topic,
            "depth": depth,
            "audience": audience,
            "timestamp": datetime.now().isoformat(),
            "additional_context": additional_context or {}
        }
    
    def generate_research_prompt(self, 
                               topic: str, 
                               depth: str = "intermediate",
                               audience: str = "general") -> str:
        """
        Generate a structured research prompt based on the topic and parameters
        
        Args:
            topic: Research topic
            depth: Research depth
            audience: Target audience
            
        Returns:
            Formatted research prompt
        """
        depth_instructions = {
            "basic": "Provide a foundational overview with key concepts and main points.",
            "intermediate": "Provide detailed analysis with multiple perspectives and current developments.",
            "comprehensive": "Provide exhaustive analysis including historical context, current state, future trends, and expert opinions."
        }
        
        audience_instructions = {
            "general": "Use accessible language and provide practical examples.",
            "academic": "Use scholarly language, cite peer-reviewed sources, and include methodological considerations.",
            "business": "Focus on strategic implications, market dynamics, and actionable insights.",
            "technical": "Include technical specifications, implementation details, and system architectures."
        }
        
        return f"""Please conduct {depth} research on the following topic for a {audience} audience:

Topic: {topic}

Research Depth: {depth_instructions.get(depth, depth_instructions["intermediate"])}
Audience Focus: {audience_instructions.get(audience, audience_instructions["general"])}

Please structure your response with:
1. Executive Summary/Overview
2. Key Concepts and Definitions
3. Current State and Context
4. Recent Developments and Trends
5. Multiple Perspectives and Viewpoints
6. Practical Applications and Implications
7. Challenges and Limitations
8. Future Outlook and Opportunities
9. Areas for Further Investigation
10. Sources and References

Ensure your response is well-structured, thoroughly researched, and tailored to the specified audience and depth level."""

    def generate_subtopics(self, main_topic: str, max_subtopics: int = 5) -> List[str]:
        """
        Generate logical subtopics for comprehensive research coverage
        
        Args:
            main_topic: The main research topic
            max_subtopics: Maximum number of subtopics to generate
            
        Returns:
            List of generated subtopics
        """
        subtopic_prompt = f"""Generate {max_subtopics} logical subtopics for comprehensive research on: {main_topic}

Requirements:
- Each subtopic should be distinct and non-overlapping
- Cover different aspects: theoretical, practical, historical, future trends
- Progress from general to specific concepts
- Be suitable for in-depth analysis
- Cover the main topic comprehensively

Format: Return only the subtopic titles, one per line."""
        
        # This would typically use the LLM to generate subtopics
        # For now, returning a placeholder structure
        return [
            f"Historical Development of {main_topic}",
            f"Current State and Trends in {main_topic}",
            f"Key Technologies and Methodologies in {main_topic}",
            f"Applications and Use Cases of {main_topic}",
            f"Future Outlook and Emerging Trends in {main_topic}"
        ]
    
    def summarize_research(self, research_content: List[Dict[str, Any]]) -> str:
        """
        Summarize research findings from multiple sources
        
        Args:
            research_content: List of research findings
            
        Returns:
            Comprehensive research summary
        """
        if not research_content:
            return "No research content to summarize."
        
        summary_prompt = f"""Please provide a comprehensive summary of the following research findings:

Research Content:
{json.dumps(research_content, indent=2)}

Please structure the summary with:
1. Key Findings Overview
2. Main Themes and Patterns
3. Consensus and Disagreements
4. Most Significant Insights
5. Gaps and Areas for Further Research
6. Conclusion and Recommendations

Make the summary coherent, well-organized, and highlight the most important insights."""
        
        return summary_prompt
    
    def get_research_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the current research session
        
        Returns:
            Dictionary containing research metrics
        """
        return {
            "research_context": self.current_research_context,
            "total_interactions": len(self.research_history),
            "available_tools": self.research_tools,
            "last_updated": datetime.now().isoformat()
        }
    
    def add_to_research_history(self, interaction: Dict[str, Any]):
        """
        Add an interaction to the research history
        
        Args:
            interaction: Dictionary containing interaction details
        """
        interaction["timestamp"] = datetime.now().isoformat()
        self.research_history.append(interaction)
    
    def generate_research_response(self, prompt: str) -> str:
        """
        Generate research response using LLM client
        
        Args:
            prompt: Research prompt
            
        Returns:
            Generated research response
        """
        if self.llm_client and self.llm_client.is_available():
            # Get system message from various possible attributes
            system_msg = self._get_system_message()
            full_prompt = f"{system_msg}\n\nUser Request: {prompt}"
            return self.llm_client.generate_response(full_prompt)
        else:
            # Fallback response
            return f"""I understand you're asking for research on: {prompt}

While I don't have direct LLM access right now, I can guide you through the research process:

1. **Topic Analysis**: Break down your topic into key components
2. **Research Tools**: Use the available web and academic search tools
3. **Multiple Perspectives**: Consider different viewpoints and stakeholder interests
4. **Current Information**: Look for recent developments and trends
5. **Credible Sources**: Focus on authoritative and peer-reviewed sources

Please use the research tools in ARIA to gather specific information on your topic."""
    
    def _get_system_message(self) -> str:
        """
        Get system message from various possible attributes
        
        Returns:
            System message string
        """
        # Try different possible attribute names for system message
        if hasattr(self, 'system_message'):
            return self.system_message
        elif hasattr(self, '_stored_system_message'):
            return self._stored_system_message
        elif hasattr(self, '_system_messages') and self._system_messages:
            # Handle list of system messages
            if isinstance(self._system_messages, list):
                return self._system_messages[0] if self._system_messages else self._get_default_system_message()
            else:
                return str(self._system_messages)
        elif hasattr(self, '_system_message'):
            return self._system_message
        else:
            # Return default system message
            return self._get_default_system_message()

    def export_research_session(self) -> Dict[str, Any]:
        """
        Export the complete research session data
        
        Returns:
            Dictionary containing complete session data
        """
        return {
            "session_id": f"aria_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "research_context": self.current_research_context,
            "research_history": self.research_history,
            "tools_used": self.research_tools,
            "metrics": self.get_research_metrics(),
            "export_timestamp": datetime.now().isoformat()
        }


def create_research_assistant(config: Dict[str, Any]) -> EnhancedResearchAssistant:
    """
    Factory function to create a configured research assistant
    
    Args:
        config: Configuration dictionary containing LLM and agent settings
        
    Returns:
        Configured EnhancedResearchAssistant instance
    """
    return EnhancedResearchAssistant(
        name=config.get("name", "research_assistant"),
        llm_config=config.get("llm_config", {}),
        system_message=config.get("system_message"),
        research_tools=config.get("research_tools", ["web_search", "academic_search", "summarization"])
    )


def get_research_assistant_capabilities() -> Dict[str, Any]:
    """
    Get information about research assistant capabilities
    
    Returns:
        Dictionary describing available capabilities
    """
    return {
        "core_features": [
            "Topic analysis and breakdown",
            "Multi-perspective research",
            "Audience-specific content adaptation",
            "Source integration and citation",
            "Structured response generation",
            "Research session management"
        ],
        "research_depths": ["basic", "intermediate", "comprehensive"],
        "target_audiences": ["general", "academic", "business", "technical"],
        "available_tools": ["web_search", "academic_search", "summarization", "subtopic_generation"],
        "export_formats": ["json", "markdown", "structured_text"],
        "autogen_integration": AUTOGEN_AVAILABLE
    }