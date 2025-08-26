"""
SEO Tools for Content Creation System
SEO optimization, analysis, and meta tag generation tools
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
import math
from collections import Counter
import random


class KeywordDensityOptimizerTool(BaseTool):
    """Optimize keyword density and placement in content"""
    name: str = "keyword_density_optimizer"
    description: str = "Analyze and optimize keyword density, placement, and distribution throughout content for better SEO performance."
    
    def _run(self, content: str, primary_keyword: str, secondary_keywords: str = "") -> str:
        """Execute keyword density optimization"""
        try:
            secondary_list = [k.strip() for k in secondary_keywords.split(',')] if secondary_keywords else []
            
            analysis = self._analyze_keyword_density(content, primary_keyword, secondary_list)
            optimized_content = self._optimize_keyword_placement(content, primary_keyword, secondary_list)
            recommendations = self._get_optimization_recommendations(analysis)
            
            result = {
                "original_content": content,
                "optimized_content": optimized_content,
                "keyword_analysis": analysis,
                "optimization_recommendations": recommendations,
                "seo_score": self._calculate_seo_score(analysis),
                "keyword_opportunities": self._identify_keyword_opportunities(content, primary_keyword, secondary_list)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error in keyword density optimization: {str(e)}"
    
    def _analyze_keyword_density(self, content: str, primary_keyword: str, secondary_keywords: List[str]) -> Dict[str, Any]:
        """Analyze keyword density and distribution"""
        words = content.lower().split()
        total_words = len(words)
        
        # Analyze primary keyword
        primary_count = content.lower().count(primary_keyword.lower())
        primary_density = (primary_count / total_words) * 100 if total_words > 0 else 0
        
        # Analyze secondary keywords
        secondary_analysis = {}
        for keyword in secondary_keywords:
            count = content.lower().count(keyword.lower())
            density = (count / total_words) * 100 if total_words > 0 else 0
            secondary_analysis[keyword] = {"count": count, "density": density}
        
        # Analyze keyword placement
        placement_analysis = self._analyze_keyword_placement(content, primary_keyword)
        
        return {
            "total_words": total_words,
            "primary_keyword": {
                "keyword": primary_keyword,
                "count": primary_count,
                "density": primary_density,
                "placement": placement_analysis
            },
            "secondary_keywords": secondary_analysis,
            "keyword_distribution": self._analyze_keyword_distribution(content, primary_keyword, secondary_keywords)
        }
    
    def _analyze_keyword_placement(self, content: str, keyword: str) -> Dict[str, bool]:
        """Analyze keyword placement in important positions"""
        content_lower = content.lower()
        keyword_lower = keyword.lower()
        
        # Check placement in important positions
        first_paragraph = content.split('\n\n')[0] if '\n\n' in content else content[:200]
        last_paragraph = content.split('\n\n')[-1] if '\n\n' in content else content[-200:]
        
        return {
            "in_title": keyword_lower in content_lower[:100],
            "in_first_paragraph": keyword_lower in first_paragraph.lower(),
            "in_last_paragraph": keyword_lower in last_paragraph.lower(),
            "in_headings": self._check_keyword_in_headings(content, keyword),
            "in_meta_content": keyword_lower in content_lower[:300]
        }
    
    def _check_keyword_in_headings(self, content: str, keyword: str) -> bool:
        """Check if keyword appears in headings"""
        heading_patterns = [r'^#+\s+(.+)', r'^(.+):$']
        keyword_lower = keyword.lower()
        
        for pattern in heading_patterns:
            headings = re.findall(pattern, content, re.MULTILINE)
            for heading in headings:
                if keyword_lower in heading.lower():
                    return True
        return False
    
    def _analyze_keyword_distribution(self, content: str, primary_keyword: str, secondary_keywords: List[str]) -> Dict[str, float]:
        """Analyze how keywords are distributed throughout the content"""
        # Split content into sections
        sections = content.split('\n\n')
        section_count = len(sections)
        
        if section_count == 0:
            return {"distribution_score": 0.0}
        
        # Check keyword presence in each section
        primary_sections = sum(1 for section in sections if primary_keyword.lower() in section.lower())
        primary_distribution = (primary_sections / section_count) * 100
        
        secondary_distribution = {}
        for keyword in secondary_keywords:
            keyword_sections = sum(1 for section in sections if keyword.lower() in section.lower())
            secondary_distribution[keyword] = (keyword_sections / section_count) * 100
        
        return {
            "primary_distribution": primary_distribution,
            "secondary_distribution": secondary_distribution,
            "distribution_score": min(100, primary_distribution + sum(secondary_distribution.values()) / len(secondary_distribution) if secondary_distribution else 0)
        }
    
    def _optimize_keyword_placement(self, content: str, primary_keyword: str, secondary_keywords: List[str]) -> str:
        """Optimize keyword placement in content"""
        optimized = content
        
        # Ensure primary keyword appears in first paragraph if not already
        paragraphs = optimized.split('\n\n')
        if paragraphs and primary_keyword.lower() not in paragraphs[0].lower():
            # Try to naturally integrate the keyword
            first_para = paragraphs[0]
            if not first_para.endswith('.'):
                first_para += '.'
            first_para += f" Understanding {primary_keyword} is essential for success in this area."
            paragraphs[0] = first_para
        
        # Ensure keyword appears in conclusion if not already
        if len(paragraphs) > 1 and primary_keyword.lower() not in paragraphs[-1].lower():
            last_para = paragraphs[-1]
            if not last_para.endswith('.'):
                last_para += '.'
            last_para += f" These insights about {primary_keyword} will help you achieve better results."
            paragraphs[-1] = last_para
        
        optimized = '\n\n'.join(paragraphs)
        return optimized
    
    def _get_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Get recommendations for keyword optimization"""
        recommendations = []
        
        primary_density = analysis["primary_keyword"]["density"]
        placement = analysis["primary_keyword"]["placement"]
        
        # Density recommendations
        if primary_density < 1.0:
            recommendations.append("Increase primary keyword density (target: 1-2%)")
        elif primary_density > 3.0:
            recommendations.append("Reduce primary keyword density to avoid over-optimization")
        
        # Placement recommendations
        if not placement["in_first_paragraph"]:
            recommendations.append("Include primary keyword in the first paragraph")
        if not placement["in_last_paragraph"]:
            recommendations.append("Include primary keyword in the conclusion")
        if not placement["in_headings"]:
            recommendations.append("Include primary keyword in at least one heading")
        
        # Distribution recommendations
        distribution_score = analysis["keyword_distribution"]["distribution_score"]
        if distribution_score < 30:
            recommendations.append("Improve keyword distribution throughout the content")
        
        return recommendations
    
    def _calculate_seo_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall SEO score based on keyword analysis"""
        score = 0.0
        
        # Density score (0-30 points)
        primary_density = analysis["primary_keyword"]["density"]
        if 1.0 <= primary_density <= 2.5:
            score += 30
        elif 0.5 <= primary_density < 1.0 or 2.5 < primary_density <= 3.5:
            score += 20
        elif primary_density > 0:
            score += 10
        
        # Placement score (0-40 points)
        placement = analysis["primary_keyword"]["placement"]
        placement_score = sum([
            10 if placement["in_first_paragraph"] else 0,
            10 if placement["in_last_paragraph"] else 0,
            10 if placement["in_headings"] else 0,
            10 if placement["in_meta_content"] else 0
        ])
        score += placement_score
        
        # Distribution score (0-30 points)
        distribution_score = analysis["keyword_distribution"]["distribution_score"]
        score += (distribution_score / 100) * 30
        
        return min(100.0, score)
    
    def _identify_keyword_opportunities(self, content: str, primary_keyword: str, secondary_keywords: List[str]) -> List[str]:
        """Identify opportunities for keyword optimization"""
        opportunities = []
        
        # Check for related keywords that could be added
        related_terms = self._generate_related_keywords(primary_keyword)
        for term in related_terms:
            if term.lower() not in content.lower():
                opportunities.append(f"Consider adding related term: '{term}'")
        
        # Check for long-tail opportunities
        long_tail = self._generate_long_tail_keywords(primary_keyword)
        for phrase in long_tail[:3]:
            if phrase.lower() not in content.lower():
                opportunities.append(f"Consider adding long-tail phrase: '{phrase}'")
        
        return opportunities[:5]
    
    def _generate_related_keywords(self, primary_keyword: str) -> List[str]:
        """Generate related keywords"""
        related = [
            f"{primary_keyword} guide",
            f"{primary_keyword} tips",
            f"best {primary_keyword}",
            f"{primary_keyword} strategies",
            f"{primary_keyword} tools"
        ]
        return related
    
    def _generate_long_tail_keywords(self, primary_keyword: str) -> List[str]:
        """Generate long-tail keyword suggestions"""
        long_tail = [
            f"how to improve {primary_keyword}",
            f"best practices for {primary_keyword}",
            f"{primary_keyword} for beginners",
            f"advanced {primary_keyword} techniques",
            f"{primary_keyword} case studies"
        ]
        return long_tail


class MetaTagGeneratorTool(BaseTool):
    """Generate optimized meta tags for SEO"""
    name: str = "meta_tag_generator"
    description: str = "Generate optimized meta titles, descriptions, and other meta tags based on content and target keywords."
    
    def _run(self, content: str, primary_keyword: str, page_type: str = "article") -> str:
        """Generate meta tags"""
        try:
            meta_tags = self._generate_all_meta_tags(content, primary_keyword, page_type)
            
            result = {
                "meta_tags": meta_tags,
                "optimization_tips": self._get_meta_optimization_tips(),
                "character_counts": self._get_character_counts(meta_tags),
                "seo_recommendations": self._get_meta_seo_recommendations(meta_tags)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error generating meta tags: {str(e)}"
    
    def _generate_all_meta_tags(self, content: str, primary_keyword: str, page_type: str) -> Dict[str, Any]:
        """Generate all meta tags"""
        # Generate multiple title options
        title_options = self._generate_title_options(content, primary_keyword, page_type)
        
        # Generate multiple description options
        description_options = self._generate_description_options(content, primary_keyword)
        
        # Generate other meta tags
        keywords = self._extract_keywords_from_content(content, primary_keyword)
        
        return {
            "title_options": title_options,
            "description_options": description_options,
            "keywords": keywords,
            "og_tags": self._generate_og_tags(title_options[0], description_options[0]),
            "twitter_tags": self._generate_twitter_tags(title_options[0], description_options[0]),
            "additional_tags": self._generate_additional_tags(page_type)
        }
    
    def _generate_title_options(self, content: str, primary_keyword: str, page_type: str) -> List[str]:
        """Generate multiple title options"""
        # Extract main topic from content
        first_paragraph = content.split('\n\n')[0] if '\n\n' in content else content[:200]
        
        # Generate title variations
        titles = []
        
        if page_type == "article":
            titles = [
                f"Complete Guide to {primary_keyword.title()}",
                f"How to Master {primary_keyword.title()}: Expert Tips",
                f"{primary_keyword.title()}: Everything You Need to Know",
                f"The Ultimate {primary_keyword.title()} Guide for 2024",
                f"Beginner's Guide to {primary_keyword.title()}"
            ]
        elif page_type == "product":
            titles = [
                f"Best {primary_keyword.title()} - Premium Quality",
                f"{primary_keyword.title()} - Professional Solution",
                f"High-Quality {primary_keyword.title()} for Your Needs",
                f"Professional {primary_keyword.title()} Services",
                f"Top-Rated {primary_keyword.title()}"
            ]
        elif page_type == "service":
            titles = [
                f"Professional {primary_keyword.title()} Services",
                f"Expert {primary_keyword.title()} Solutions",
                f"{primary_keyword.title()} Services - Get Results",
                f"Trusted {primary_keyword.title()} Experts",
                f"Premium {primary_keyword.title()} Services"
            ]
        else:
            titles = [
                f"{primary_keyword.title()} - Professional Resource",
                f"Learn About {primary_keyword.title()}",
                f"{primary_keyword.title()} Information & Tips",
                f"Understanding {primary_keyword.title()}",
                f"{primary_keyword.title()} Guide & Resources"
            ]
        
        # Ensure titles are within optimal length (50-60 characters)
        optimized_titles = []
        for title in titles:
            if len(title) > 60:
                title = title[:57] + "..."
            optimized_titles.append(title)
        
        return optimized_titles[:5]
    
    def _generate_description_options(self, content: str, primary_keyword: str) -> List[str]:
        """Generate multiple meta description options"""
        # Extract key points from content
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        key_points = sentences[:3]
        
        descriptions = [
            f"Discover everything about {primary_keyword}. {key_points[0] if key_points else 'Learn essential tips and strategies.'} Expert insights and practical advice.",
            f"Complete guide to {primary_keyword}. Learn best practices, tips, and strategies from industry experts. Get started today.",
            f"Master {primary_keyword} with our comprehensive guide. {key_points[0] if key_points else 'Professional insights and actionable advice.'} Start improving now.",
            f"Learn {primary_keyword} from experts. Practical tips, strategies, and insights to help you succeed. Read our complete guide.",
            f"Everything you need to know about {primary_keyword}. Expert advice, proven strategies, and practical tips for success."
        ]
        
        # Ensure descriptions are within optimal length (150-160 characters)
        optimized_descriptions = []
        for desc in descriptions:
            if len(desc) > 160:
                desc = desc[:157] + "..."
            optimized_descriptions.append(desc)
        
        return optimized_descriptions
    
    def _extract_keywords_from_content(self, content: str, primary_keyword: str) -> List[str]:
        """Extract relevant keywords from content"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = Counter(words)
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'against'}
        
        keywords = [primary_keyword]
        for word, freq in word_freq.most_common(10):
            if len(word) > 3 and word not in stop_words and word != primary_keyword.lower():
                keywords.append(word)
        
        return keywords[:10]
    
    def _generate_og_tags(self, title: str, description: str) -> Dict[str, str]:
        """Generate Open Graph tags"""
        return {
            "og:title": title,
            "og:description": description,
            "og:type": "article",
            "og:url": "https://example.com/article",
            "og:image": "https://example.com/images/og-image.jpg",
            "og:site_name": "Your Site Name"
        }
    
    def _generate_twitter_tags(self, title: str, description: str) -> Dict[str, str]:
        """Generate Twitter Card tags"""
        return {
            "twitter:card": "summary_large_image",
            "twitter:title": title,
            "twitter:description": description,
            "twitter:image": "https://example.com/images/twitter-card.jpg",
            "twitter:site": "@yoursite"
        }
    
    def _generate_additional_tags(self, page_type: str) -> Dict[str, str]:
        """Generate additional meta tags"""
        tags = {
            "robots": "index, follow",
            "author": "Content Creator",
            "viewport": "width=device-width, initial-scale=1.0",
            "language": "en"
        }
        
        if page_type == "article":
            tags["article:published_time"] = datetime.now().isoformat()
            tags["article:author"] = "Content Creator"
        
        return tags
    
    def _get_character_counts(self, meta_tags: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Get character counts for meta tags"""
        return {
            "titles": {
                f"Option {i+1}": len(title) 
                for i, title in enumerate(meta_tags["title_options"])
            },
            "descriptions": {
                f"Option {i+1}": len(desc) 
                for i, desc in enumerate(meta_tags["description_options"])
            }
        }
    
    def _get_meta_optimization_tips(self) -> List[str]:
        """Get meta tag optimization tips"""
        return [
            "Keep titles between 50-60 characters for optimal display",
            "Keep meta descriptions between 150-160 characters",
            "Include primary keyword in both title and description",
            "Make descriptions compelling and action-oriented",
            "Ensure each page has unique meta tags",
            "Include emotional triggers and benefits in descriptions"
        ]
    
    def _get_meta_seo_recommendations(self, meta_tags: Dict[str, Any]) -> List[str]:
        """Get SEO recommendations for meta tags"""
        recommendations = []
        
        # Check title optimization
        for i, title in enumerate(meta_tags["title_options"]):
            if len(title) > 60:
                recommendations.append(f"Title option {i+1} is too long ({len(title)} chars)")
            elif len(title) < 30:
                recommendations.append(f"Title option {i+1} might be too short ({len(title)} chars)")
        
        # Check description optimization
        for i, desc in enumerate(meta_tags["description_options"]):
            if len(desc) > 160:
                recommendations.append(f"Description option {i+1} is too long ({len(desc)} chars)")
            elif len(desc) < 120:
                recommendations.append(f"Description option {i+1} might be too short ({len(desc)} chars)")
        
        return recommendations


class SEOScoreCalculatorTool(BaseTool):
    """Calculate comprehensive SEO score for content"""
    name: str = "seo_score_calculator"
    description: str = "Calculate comprehensive SEO score based on multiple factors including keywords, content structure, readability, and technical SEO elements."
    
    def _run(self, content: str, target_keywords: str, meta_title: str = "", meta_description: str = "") -> str:
        """Calculate SEO score"""
        try:
            keyword_list = [k.strip() for k in target_keywords.split(',')] if target_keywords else []
            
            # Calculate individual scores
            keyword_score = self._calculate_keyword_score(content, keyword_list)
            content_score = self._calculate_content_structure_score(content)
            readability_score = self._calculate_readability_score(content)
            technical_score = self._calculate_technical_seo_score(content, meta_title, meta_description)
            
            # Calculate overall score
            overall_score = (keyword_score * 0.3 + content_score * 0.25 + 
                           readability_score * 0.25 + technical_score * 0.2)
            
            result = {
                "overall_seo_score": round(overall_score, 1),
                "score_breakdown": {
                    "keyword_optimization": round(keyword_score, 1),
                    "content_structure": round(content_score, 1),
                    "readability": round(readability_score, 1),
                    "technical_seo": round(technical_score, 1)
                },
                "detailed_analysis": {
                    "word_count": len(content.split()),
                    "heading_structure": self._analyze_heading_structure(content),
                    "keyword_density": self._calculate_keyword_densities(content, keyword_list),
                    "readability_metrics": self._get_readability_metrics(content)
                },
                "improvement_suggestions": self._get_improvement_suggestions(
                    keyword_score, content_score, readability_score, technical_score
                ),
                "ranking_potential": self._assess_ranking_potential(overall_score)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error calculating SEO score: {str(e)}"
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """Calculate keyword optimization score"""
        if not keywords:
            return 0.0
        
        total_words = len(content.split())
        score = 0.0
        
        for keyword in keywords:
            keyword_count = content.lower().count(keyword.lower())
            density = (keyword_count / total_words) * 100 if total_words > 0 else 0
            
            # Optimal density is 1-2%
            if 1.0 <= density <= 2.0:
                score += 25
            elif 0.5 <= density < 1.0 or 2.0 < density <= 3.0:
                score += 15
            elif density > 0:
                score += 5
        
        return min(100.0, score)
    
    def _calculate_content_structure_score(self, content: str) -> float:
        """Calculate content structure score"""
        score = 0.0
        
        # Word count (0-25 points)
        word_count = len(content.split())
        if 800 <= word_count <= 2000:
            score += 25
        elif 500 <= word_count < 800 or 2000 < word_count <= 3000:
            score += 20
        elif 300 <= word_count < 500:
            score += 15
        elif word_count >= 300:
            score += 10
        
        # Heading structure (0-25 points)
        headings = re.findall(r'^#+\s+(.+)', content, re.MULTILINE)
        if len(headings) >= 3:
            score += 25
        elif len(headings) >= 2:
            score += 20
        elif len(headings) >= 1:
            score += 15
        
        # Paragraph structure (0-25 points)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        if 50 <= avg_para_length <= 150:
            score += 25
        elif 30 <= avg_para_length < 50 or 150 < avg_para_length <= 200:
            score += 20
        elif avg_para_length > 0:
            score += 10
        
        # Lists and formatting (0-25 points)
        list_count = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        if list_count >= 2:
            score += 25
        elif list_count >= 1:
            score += 15
        else:
            score += 5
        
        return score
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score (simplified Flesch-Kincaid)"""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Calculate average syllables per word (simplified)
        total_syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Simplified Flesch Reading Ease score
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Convert to 0-100 scale where higher is better
        if flesch_score >= 90:
            return 100.0
        elif flesch_score >= 80:
            return 90.0
        elif flesch_score >= 70:
            return 80.0
        elif flesch_score >= 60:
            return 70.0
        elif flesch_score >= 50:
            return 60.0
        elif flesch_score >= 30:
            return 50.0
        else:
            return 30.0
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(1, syllables)
    
    def _calculate_technical_seo_score(self, content: str, meta_title: str, meta_description: str) -> float:
        """Calculate technical SEO score"""
        score = 0.0
        
        # Meta title (0-30 points)
        if meta_title:
            title_length = len(meta_title)
            if 50 <= title_length <= 60:
                score += 30
            elif 40 <= title_length < 50 or 60 < title_length <= 70:
                score += 25
            elif title_length > 0:
                score += 15
        
        # Meta description (0-30 points)
        if meta_description:
            desc_length = len(meta_description)
            if 150 <= desc_length <= 160:
                score += 30
            elif 140 <= desc_length < 150 or 160 < desc_length <= 170:
                score += 25
            elif desc_length > 0:
                score += 15
        
        # URL structure (simulated - 0-20 points)
        score += 15  # Assume good URL structure
        
        # Internal linking opportunities (0-20 points)
        link_patterns = re.findall(r'\[([^\]]+)\]\([^)]+\)', content)
        if len(link_patterns) >= 3:
            score += 20
        elif len(link_patterns) >= 1:
            score += 15
        else:
            score += 5
        
        return score
    
    def _analyze_heading_structure(self, content: str) -> Dict[str, int]:
        """Analyze heading structure"""
        h1_count = len(re.findall(r'^#\s+(.+)', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+(.+)', content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s+(.+)', content, re.MULTILINE))
        
        return {
            "h1_count": h1_count,
            "h2_count": h2_count,
            "h3_count": h3_count,
            "total_headings": h1_count + h2_count + h3_count
        }
    
    def _calculate_keyword_densities(self, content: str, keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword densities"""
        total_words = len(content.split())
        densities = {}
        
        for keyword in keywords:
            count = content.lower().count(keyword.lower())
            density = (count / total_words) * 100 if total_words > 0 else 0
            densities[keyword] = round(density, 2)
        
        return densities
    
    def _get_readability_metrics(self, content: str) -> Dict[str, Any]:
        """Get detailed readability metrics"""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": round(len(words) / len(sentences), 1) if sentences else 0,
            "avg_sentence_length": "Good" if 15 <= len(words) / len(sentences) <= 20 else "Needs improvement"
        }
    
    def _get_improvement_suggestions(self, keyword_score: float, content_score: float, 
                                   readability_score: float, technical_score: float) -> List[str]:
        """Get specific improvement suggestions"""
        suggestions = []
        
        if keyword_score < 70:
            suggestions.append("Improve keyword optimization - increase keyword density to 1-2%")
        if content_score < 70:
            suggestions.append("Enhance content structure - add more headings and improve paragraph length")
        if readability_score < 70:
            suggestions.append("Improve readability - use shorter sentences and simpler words")
        if technical_score < 70:
            suggestions.append("Optimize technical SEO - improve meta tags and internal linking")
        
        return suggestions
    
    def _assess_ranking_potential(self, overall_score: float) -> str:
        """Assess ranking potential based on overall score"""
        if overall_score >= 90:
            return "Excellent - High ranking potential"
        elif overall_score >= 80:
            return "Good - Strong ranking potential"
        elif overall_score >= 70:
            return "Fair - Moderate ranking potential"
        elif overall_score >= 60:
            return "Poor - Low ranking potential"
        else:
            return "Very Poor - Needs significant improvement"


def get_all_seo_tools() -> List[BaseTool]:
    """Get all SEO tools"""
    return [
        KeywordDensityOptimizerTool(),
        MetaTagGeneratorTool(),
        SEOScoreCalculatorTool()
    ]