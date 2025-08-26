"""
Content Generation Tools for AI Content Creation System
AI-powered content generation, writing assistance, and formatting tools
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
import random
import os


class AIContentGeneratorTool(BaseTool):
    """Generate content using AI with customizable parameters"""
    name: str = "ai_content_generator"
    description: str = "Generate high-quality content using AI. Provide outline, style, target word count, and tone for customized content creation."
    
    def _run(self, outline: str, style: str = "professional", word_count: int = 800, tone: str = "informative") -> str:
        """Execute AI content generation"""
        try:
            # This would integrate with actual LLM APIs (OpenAI, Gemini, Claude)
            # For demo purposes, we'll create structured content
            
            content_structure = self._parse_outline(outline)
            generated_content = self._generate_content_sections(content_structure, style, tone, word_count)
            
            result = {
                "generated_content": generated_content,
                "word_count": len(generated_content.split()),
                "style": style,
                "tone": tone,
                "sections": list(content_structure.keys()),
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "estimated_reading_time": f"{len(generated_content.split()) // 200 + 1} minutes",
                    "content_type": self._detect_content_type(outline)
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error in AI content generation: {str(e)}"
    
    def _parse_outline(self, outline: str) -> Dict[str, str]:
        """Parse content outline into sections"""
        sections = {}
        lines = outline.split('\n')
        current_section = "introduction"
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') or line.startswith('-') or line.startswith('*'):
                # This is a section header
                section_name = re.sub(r'^[#\-\*\s]+', '', line).lower().replace(' ', '_')
                sections[section_name] = ""
                current_section = section_name
            elif line:
                if current_section in sections:
                    sections[current_section] += line + " "
        
        # Ensure we have basic sections
        if not sections:
            sections = {
                "introduction": "",
                "main_content": outline,
                "conclusion": ""
            }
        
        return sections
    
    def _generate_content_sections(self, sections: Dict[str, str], style: str, tone: str, target_words: int) -> str:
        """Generate content for each section"""
        total_sections = len(sections)
        words_per_section = target_words // total_sections if total_sections > 0 else target_words
        
        content_parts = []
        
        for section_name, section_outline in sections.items():
            section_content = self._generate_section_content(
                section_name, section_outline, style, tone, words_per_section
            )
            content_parts.append(section_content)
        
        return "\n\n".join(content_parts)
    
    def _generate_section_content(self, section_name: str, outline: str, style: str, tone: str, word_count: int) -> str:
        """Generate content for a specific section"""
        # This would use actual LLM APIs in production
        # For demo, we'll create structured content based on section type
        
        section_templates = {
            "introduction": self._generate_introduction,
            "main_content": self._generate_main_content,
            "conclusion": self._generate_conclusion,
            "benefits": self._generate_benefits_section,
            "features": self._generate_features_section,
            "how_to": self._generate_how_to_section
        }
        
        # Find the best template for this section
        template_func = section_templates.get(section_name, self._generate_generic_section)
        return template_func(section_name, outline, style, tone, word_count)
    
    def _generate_introduction(self, section_name: str, outline: str, style: str, tone: str, word_count: int) -> str:
        """Generate introduction section"""
        return f"""# Introduction

{outline if outline else "In today's rapidly evolving digital landscape, understanding key concepts and strategies has become more important than ever. This comprehensive guide will explore essential insights and provide actionable recommendations for success."}

Whether you're a beginner looking to understand the fundamentals or an experienced professional seeking advanced strategies, this article will provide valuable insights and practical guidance to help you achieve your goals."""
    
    def _generate_main_content(self, section_name: str, outline: str, style: str, tone: str, word_count: int) -> str:
        """Generate main content section"""
        return f"""## Main Content

{outline if outline else "The core concepts we'll explore are fundamental to understanding this topic comprehensively. Let's dive into the key areas that matter most."}

### Key Points to Consider

1. **Strategic Planning**: Developing a clear roadmap is essential for success in any endeavor.

2. **Implementation Best Practices**: Following proven methodologies can significantly improve outcomes.

3. **Continuous Improvement**: Regular evaluation and optimization ensure long-term success.

4. **Industry Standards**: Staying current with best practices and regulations is crucial.

### Practical Applications

These concepts can be applied across various scenarios and industries, providing flexibility and adaptability to different contexts and requirements."""
    
    def _generate_conclusion(self, section_name: str, outline: str, style: str, tone: str, word_count: int) -> str:
        """Generate conclusion section"""
        return f"""## Conclusion

{outline if outline else "In summary, the key insights covered in this guide provide a solid foundation for understanding and implementing effective strategies."}

By following these recommendations and best practices, you'll be well-positioned to achieve your objectives and drive meaningful results. Remember to stay informed about industry developments and continuously refine your approach based on new insights and changing conditions.

### Next Steps

1. Review the key concepts outlined in this guide
2. Identify areas most relevant to your specific situation
3. Develop an implementation plan
4. Monitor progress and adjust strategies as needed"""
    
    def _generate_benefits_section(self, section_name: str, outline: str, style: str, tone: str, word_count: int) -> str:
        """Generate benefits section"""
        return f"""## Benefits

{outline if outline else "Understanding and implementing these strategies offers numerous advantages:"}

- **Improved Efficiency**: Streamlined processes and optimized workflows
- **Better Outcomes**: Higher success rates and improved performance metrics
- **Cost Savings**: Reduced overhead and more effective resource utilization
- **Competitive Advantage**: Stay ahead of industry trends and best practices
- **Scalability**: Solutions that grow with your needs and requirements"""
    
    def _generate_features_section(self, section_name: str, outline: str, style: str, tone: str, word_count: int) -> str:
        """Generate features section"""
        return f"""## Key Features

{outline if outline else "The essential components that make this approach effective include:"}

### Core Capabilities
- Comprehensive analysis and evaluation tools
- Real-time monitoring and reporting
- Customizable settings and configurations
- Integration with existing systems and workflows

### Advanced Features
- Automated optimization and recommendations
- Predictive analytics and forecasting
- Collaborative tools and team management
- Mobile accessibility and cloud-based solutions"""
    
    def _generate_how_to_section(self, section_name: str, outline: str, style: str, tone: str, word_count: int) -> str:
        """Generate how-to section"""
        return f"""## How To Guide

{outline if outline else "Follow these step-by-step instructions to get started:"}

### Step 1: Preparation
Begin by gathering all necessary resources and information. Ensure you have the required tools and access permissions.

### Step 2: Initial Setup
Configure your environment and establish baseline settings. This foundation will support all subsequent activities.

### Step 3: Implementation
Execute the planned activities systematically, monitoring progress and making adjustments as needed.

### Step 4: Validation
Test and verify that everything is working correctly. Address any issues or discrepancies immediately.

### Step 5: Optimization
Fine-tune settings and processes based on initial results and performance metrics."""
    
    def _generate_generic_section(self, section_name: str, outline: str, style: str, tone: str, word_count: int) -> str:
        """Generate generic section content"""
        title = section_name.replace('_', ' ').title()
        return f"""## {title}

{outline if outline else f"This section covers important aspects of {title.lower()} that are essential for comprehensive understanding."}

The key considerations and best practices in this area include strategic planning, careful implementation, and ongoing optimization. By following proven methodologies and staying current with industry standards, you can achieve better outcomes and maintain competitive advantages.

Regular evaluation and continuous improvement ensure that strategies remain effective and aligned with changing requirements and market conditions."""
    
    def _detect_content_type(self, outline: str) -> str:
        """Detect the type of content based on outline"""
        outline_lower = outline.lower()
        
        if any(word in outline_lower for word in ["how to", "step", "guide", "tutorial"]):
            return "how-to guide"
        elif any(word in outline_lower for word in ["best", "top", "list", "comparison"]):
            return "listicle"
        elif any(word in outline_lower for word in ["case study", "example", "success"]):
            return "case study"
        elif any(word in outline_lower for word in ["review", "analysis", "evaluation"]):
            return "review"
        else:
            return "article"


class WritingStyleAdapterTool(BaseTool):
    """Adapt content to specific writing styles and brand voices"""
    name: str = "writing_style_adapter"
    description: str = "Adapt content to match specific writing styles, brand voices, and tone requirements."
    
    def _run(self, content: str, target_style: str, brand_guidelines: str = "") -> str:
        """Execute writing style adaptation"""
        try:
            adapted_content = self._adapt_content_style(content, target_style, brand_guidelines)
            
            result = {
                "original_content": content,
                "adapted_content": adapted_content,
                "target_style": target_style,
                "adaptations_made": self._identify_adaptations(content, adapted_content),
                "style_score": self._calculate_style_score(adapted_content, target_style),
                "recommendations": self._get_style_recommendations(target_style)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error in writing style adaptation: {str(e)}"
    
    def _adapt_content_style(self, content: str, target_style: str, brand_guidelines: str) -> str:
        """Adapt content to target style"""
        adapted = content
        
        style_adaptations = {
            "professional": self._adapt_to_professional,
            "casual": self._adapt_to_casual,
            "friendly": self._adapt_to_friendly,
            "authoritative": self._adapt_to_authoritative,
            "conversational": self._adapt_to_conversational,
            "technical": self._adapt_to_technical
        }
        
        adaptation_func = style_adaptations.get(target_style.lower(), self._adapt_to_professional)
        adapted = adaptation_func(adapted)
        
        # Apply brand guidelines if provided
        if brand_guidelines:
            adapted = self._apply_brand_guidelines(adapted, brand_guidelines)
        
        return adapted
    
    def _adapt_to_professional(self, content: str) -> str:
        """Adapt to professional style"""
        # Replace casual language with professional equivalents
        replacements = {
            r"\bkinda\b": "somewhat",
            r"\bgonna\b": "going to",
            r"\bwanna\b": "want to",
            r"\byeah\b": "yes",
            r"\bokay\b": "acceptable",
            r"\bawesome\b": "excellent",
            r"\bgreat\b": "exceptional"
        }
        
        adapted = content
        for pattern, replacement in replacements.items():
            adapted = re.sub(pattern, replacement, adapted, flags=re.IGNORECASE)
        
        return adapted
    
    def _adapt_to_casual(self, content: str) -> str:
        """Adapt to casual style"""
        # Make content more conversational and approachable
        adapted = content
        
        # Add casual expressions
        adapted = re.sub(r"Furthermore,", "Plus,", adapted)
        adapted = re.sub(r"Additionally,", "Also,", adapted)
        adapted = re.sub(r"In conclusion,", "So,", adapted)
        
        return adapted
    
    def _adapt_to_friendly(self, content: str) -> str:
        """Adapt to friendly style"""
        adapted = content
        
        # Add friendly language
        adapted = re.sub(r"^", "Hey there! ", adapted, count=1)
        adapted = re.sub(r"You should", "You might want to", adapted)
        adapted = re.sub(r"It is important", "It's really helpful", adapted)
        
        return adapted
    
    def _adapt_to_authoritative(self, content: str) -> str:
        """Adapt to authoritative style"""
        adapted = content
        
        # Strengthen language
        adapted = re.sub(r"might", "will", adapted)
        adapted = re.sub(r"could", "should", adapted)
        adapted = re.sub(r"perhaps", "certainly", adapted)
        
        return adapted
    
    def _adapt_to_conversational(self, content: str) -> str:
        """Adapt to conversational style"""
        adapted = content
        
        # Add conversational elements
        adapted = re.sub(r"The reader", "you", adapted)
        adapted = re.sub(r"One should", "you should", adapted)
        adapted = re.sub(r"It is recommended", "I'd recommend", adapted)
        
        return adapted
    
    def _adapt_to_technical(self, content: str) -> str:
        """Adapt to technical style"""
        adapted = content
        
        # Add technical precision
        adapted = re.sub(r"approximately", "precisely", adapted)
        adapted = re.sub(r"about", "approximately", adapted)
        adapted = re.sub(r"some", "specific", adapted)
        
        return adapted
    
    def _apply_brand_guidelines(self, content: str, guidelines: str) -> str:
        """Apply brand-specific guidelines"""
        # This would implement brand-specific adaptations
        return content
    
    def _identify_adaptations(self, original: str, adapted: str) -> List[str]:
        """Identify what adaptations were made"""
        adaptations = []
        
        if len(adapted.split()) != len(original.split()):
            adaptations.append("Word count adjusted")
        
        if original != adapted:
            adaptations.append("Style and tone adapted")
            adaptations.append("Language formality adjusted")
        
        return adaptations
    
    def _calculate_style_score(self, content: str, target_style: str) -> float:
        """Calculate how well content matches target style"""
        # Simple scoring based on style characteristics
        score = 85.0  # Base score
        
        if target_style.lower() == "professional":
            if any(word in content.lower() for word in ["gonna", "kinda", "wanna"]):
                score -= 10
        elif target_style.lower() == "casual":
            if not any(word in content.lower() for word in ["you", "your", "we"]):
                score -= 5
        
        return min(100.0, score)
    
    def _get_style_recommendations(self, target_style: str) -> List[str]:
        """Get recommendations for improving style"""
        recommendations = {
            "professional": [
                "Use formal language and complete sentences",
                "Avoid contractions and casual expressions",
                "Include authoritative sources and data"
            ],
            "casual": [
                "Use conversational language and contractions",
                "Include personal pronouns (you, we, I)",
                "Add friendly expressions and approachable tone"
            ],
            "friendly": [
                "Use warm, welcoming language",
                "Include empathetic expressions",
                "Address the reader directly"
            ],
            "authoritative": [
                "Use confident, definitive language",
                "Include expert opinions and data",
                "Avoid hedging language (might, could)"
            ]
        }
        
        return recommendations.get(target_style.lower(), [
            "Maintain consistency in tone and style",
            "Ensure content matches target audience expectations",
            "Review and refine based on brand guidelines"
        ])


class ContentFormatterTool(BaseTool):
    """Format content for different platforms and purposes"""
    name: str = "content_formatter"
    description: str = "Format content for different platforms (blog, social media, email) and export formats (HTML, Markdown, PDF)."
    
    def _run(self, content: str, format_type: str, platform: str = "general") -> str:
        """Execute content formatting"""
        try:
            formatted_content = self._format_content(content, format_type, platform)
            
            result = {
                "original_content": content,
                "formatted_content": formatted_content,
                "format_type": format_type,
                "platform": platform,
                "formatting_applied": self._get_formatting_details(format_type, platform),
                "character_count": len(formatted_content),
                "word_count": len(formatted_content.split()),
                "estimated_reading_time": f"{len(formatted_content.split()) // 200 + 1} minutes"
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error in content formatting: {str(e)}"
    
    def _format_content(self, content: str, format_type: str, platform: str) -> str:
        """Format content based on type and platform"""
        formatters = {
            "markdown": self._format_markdown,
            "html": self._format_html,
            "social_media": self._format_social_media,
            "email": self._format_email,
            "blog": self._format_blog
        }
        
        formatter = formatters.get(format_type.lower(), self._format_markdown)
        return formatter(content, platform)
    
    def _format_markdown(self, content: str, platform: str) -> str:
        """Format content as Markdown"""
        # Ensure proper markdown formatting
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Convert headers if not already formatted
                if line.endswith(':') and len(line.split()) <= 5:
                    formatted_lines.append(f"## {line.rstrip(':')}")
                elif line.startswith('- ') or line.startswith('* '):
                    formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append("")
        
        return '\n'.join(formatted_lines)
    
    def _format_html(self, content: str, platform: str) -> str:
        """Format content as HTML"""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Content</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        p {{ margin-bottom: 15px; }}
        ul, ol {{ margin-bottom: 15px; }}
    </style>
</head>
<body>
{self._convert_markdown_to_html(content)}
</body>
</html>"""
        return html_content
    
    def _format_social_media(self, content: str, platform: str) -> str:
        """Format content for social media platforms"""
        # Extract key points for social media
        sentences = content.split('.')
        key_points = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        
        if platform.lower() == "twitter":
            # Twitter format (character limit)
            summary = ". ".join(key_points)
            if len(summary) > 250:
                summary = summary[:247] + "..."
            return f"{summary}\n\n#content #marketing"
        elif platform.lower() == "linkedin":
            # LinkedIn format
            return f"Key insights:\n\n" + "\n".join([f"• {point}" for point in key_points]) + "\n\nWhat are your thoughts?"
        else:
            # General social media format
            return f"📖 {key_points[0] if key_points else content[:100]}...\n\n💡 Key takeaways:\n" + "\n".join([f"✓ {point}" for point in key_points[:2]])
    
    def _format_email(self, content: str, platform: str) -> str:
        """Format content for email campaigns"""
        return f"""Subject: Important Update

Dear Reader,

{content}

Best regards,
The Team

---
If you no longer wish to receive these emails, you can unsubscribe here."""
    
    def _format_blog(self, content: str, platform: str) -> str:
        """Format content for blog posting"""
        # Add meta information and structure
        formatted = f"""---
title: "Article Title"
date: {datetime.now().strftime("%Y-%m-%d")}
author: "Content Creator"
tags: ["content", "marketing"]
---

{content}

---

*What did you think of this article? Share your thoughts in the comments below!*"""
        return formatted
    
    def _convert_markdown_to_html(self, content: str) -> str:
        """Convert markdown to HTML"""
        html = content
        
        # Convert headers
        html = re.sub(r'^# (.*)', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*)', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.*)', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        
        # Convert paragraphs
        paragraphs = html.split('\n\n')
        formatted_paragraphs = []
        for para in paragraphs:
            if para.strip() and not para.startswith('<'):
                formatted_paragraphs.append(f'<p>{para.strip()}</p>')
            else:
                formatted_paragraphs.append(para)
        
        return '\n'.join(formatted_paragraphs)
    
    def _get_formatting_details(self, format_type: str, platform: str) -> List[str]:
        """Get details about formatting applied"""
        details = [f"Formatted for {format_type}"]
        
        if platform != "general":
            details.append(f"Optimized for {platform}")
        
        if format_type == "html":
            details.append("Added HTML structure and styling")
        elif format_type == "markdown":
            details.append("Applied markdown formatting")
        elif format_type == "social_media":
            details.append("Condensed for social media consumption")
        
        return details


def get_all_content_generation_tools() -> List[BaseTool]:
    """Get all content generation tools"""
    return [
        AIContentGeneratorTool(),
        WritingStyleAdapterTool(),
        ContentFormatterTool()
    ]