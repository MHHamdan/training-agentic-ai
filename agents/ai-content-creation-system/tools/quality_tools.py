"""
Quality Assurance Tools for Content Creation System
Content quality analysis, grammar checking, and brand compliance tools
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from collections import Counter
import random


class ReadabilityAnalyzerTool(BaseTool):
    """Analyze content readability using multiple metrics"""
    name: str = "readability_analyzer"
    description: str = "Analyze content readability using Flesch-Kincaid, sentence complexity, and other readability metrics."
    
    def _run(self, content: str, target_audience: str = "general") -> str:
        """Execute readability analysis"""
        try:
            # Calculate various readability metrics
            flesch_score = self._calculate_flesch_score(content)
            fog_index = self._calculate_fog_index(content)
            sentence_analysis = self._analyze_sentences(content)
            word_analysis = self._analyze_words(content)
            
            # Get target audience recommendations
            audience_recommendations = self._get_audience_recommendations(target_audience)
            
            # Calculate overall readability grade
            overall_grade = self._calculate_overall_grade(flesch_score, fog_index, sentence_analysis)
            
            result = {
                "readability_scores": {
                    "flesch_reading_ease": flesch_score,
                    "flesch_grade_level": self._flesch_to_grade_level(flesch_score),
                    "fog_index": fog_index,
                    "overall_grade": overall_grade
                },
                "sentence_analysis": sentence_analysis,
                "word_analysis": word_analysis,
                "target_audience": target_audience,
                "audience_recommendations": audience_recommendations,
                "improvement_suggestions": self._get_improvement_suggestions(
                    flesch_score, fog_index, sentence_analysis, word_analysis, target_audience
                ),
                "readability_verdict": self._get_readability_verdict(overall_grade, target_audience)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error in readability analysis: {str(e)}"
    
    def _calculate_flesch_score(self, content: str) -> float:
        """Calculate Flesch Reading Ease score"""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Calculate metrics
        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula
        if total_sentences > 0 and total_words > 0:
            avg_sentence_length = total_words / total_sentences
            avg_syllables_per_word = total_syllables / total_words
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, flesch_score))
        
        return 0.0
    
    def _calculate_fog_index(self, content: str) -> float:
        """Calculate Gunning Fog Index"""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Count complex words (3+ syllables)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        
        # Calculate Fog Index
        avg_sentence_length = len(words) / len(sentences)
        percentage_complex = (complex_words / len(words)) * 100
        
        fog_index = 0.4 * (avg_sentence_length + percentage_complex)
        return round(fog_index, 1)
    
    def _analyze_sentences(self, content: str) -> Dict[str, Any]:
        """Analyze sentence structure and complexity"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {}
        
        sentence_lengths = [len(s.split()) for s in sentences]
        
        return {
            "total_sentences": len(sentences),
            "avg_sentence_length": round(sum(sentence_lengths) / len(sentence_lengths), 1),
            "shortest_sentence": min(sentence_lengths),
            "longest_sentence": max(sentence_lengths),
            "sentences_over_20_words": sum(1 for length in sentence_lengths if length > 20),
            "sentences_under_10_words": sum(1 for length in sentence_lengths if length < 10),
            "sentence_variety_score": self._calculate_sentence_variety(sentence_lengths)
        }
    
    def _analyze_words(self, content: str) -> Dict[str, Any]:
        """Analyze word usage and complexity"""
        words = re.findall(r'\b\w+\b', content.lower())
        
        if not words:
            return {}
        
        # Calculate word metrics
        word_lengths = [len(word) for word in words]
        syllable_counts = [self._count_syllables(word) for word in words]
        
        # Count complex words
        complex_words = sum(1 for syllables in syllable_counts if syllables >= 3)
        very_complex_words = sum(1 for syllables in syllable_counts if syllables >= 4)
        
        return {
            "total_words": len(words),
            "unique_words": len(set(words)),
            "avg_word_length": round(sum(word_lengths) / len(word_lengths), 1),
            "avg_syllables_per_word": round(sum(syllable_counts) / len(syllable_counts), 1),
            "complex_words": complex_words,
            "complex_word_percentage": round((complex_words / len(words)) * 100, 1),
            "very_complex_words": very_complex_words,
            "lexical_diversity": round(len(set(words)) / len(words), 2)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower().strip()
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for i, char in enumerate(word):
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        # Handle special cases
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllables += 1
        
        return max(1, syllables)
    
    def _calculate_sentence_variety(self, sentence_lengths: List[int]) -> float:
        """Calculate sentence variety score"""
        if len(sentence_lengths) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((x - mean_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        std_dev = variance ** 0.5
        
        coefficient_of_variation = (std_dev / mean_length) * 100 if mean_length > 0 else 0
        
        # Convert to 0-100 scale where higher is better variety
        return min(100, coefficient_of_variation * 2)
    
    def _flesch_to_grade_level(self, flesch_score: float) -> str:
        """Convert Flesch score to grade level"""
        if flesch_score >= 90:
            return "5th grade"
        elif flesch_score >= 80:
            return "6th grade"
        elif flesch_score >= 70:
            return "7th grade"
        elif flesch_score >= 60:
            return "8th-9th grade"
        elif flesch_score >= 50:
            return "10th-12th grade"
        elif flesch_score >= 30:
            return "College level"
        else:
            return "Graduate level"
    
    def _calculate_overall_grade(self, flesch_score: float, fog_index: float, sentence_analysis: Dict) -> str:
        """Calculate overall readability grade"""
        # Average the different metrics
        flesch_grade = self._flesch_score_to_number(flesch_score)
        fog_grade = min(16, fog_index)  # Cap at 16th grade
        
        avg_grade = (flesch_grade + fog_grade) / 2
        
        if avg_grade <= 6:
            return "Elementary"
        elif avg_grade <= 8:
            return "Middle School"
        elif avg_grade <= 12:
            return "High School"
        elif avg_grade <= 16:
            return "College"
        else:
            return "Graduate"
    
    def _flesch_score_to_number(self, flesch_score: float) -> float:
        """Convert Flesch score to numeric grade level"""
        if flesch_score >= 90:
            return 5
        elif flesch_score >= 80:
            return 6
        elif flesch_score >= 70:
            return 7
        elif flesch_score >= 60:
            return 8.5
        elif flesch_score >= 50:
            return 11
        elif flesch_score >= 30:
            return 14
        else:
            return 16
    
    def _get_audience_recommendations(self, target_audience: str) -> Dict[str, Any]:
        """Get recommendations based on target audience"""
        audience_configs = {
            "general": {
                "target_grade": "8th-9th grade",
                "max_sentence_length": 20,
                "flesch_target": "60-70",
                "recommendations": [
                    "Use conversational tone",
                    "Avoid jargon and technical terms",
                    "Keep sentences under 20 words"
                ]
            },
            "professionals": {
                "target_grade": "College level",
                "max_sentence_length": 25,
                "flesch_target": "40-60",
                "recommendations": [
                    "Technical terms are acceptable",
                    "Use industry-specific language",
                    "Longer sentences are okay for complex ideas"
                ]
            },
            "academics": {
                "target_grade": "Graduate level",
                "max_sentence_length": 30,
                "flesch_target": "30-50",
                "recommendations": [
                    "Complex vocabulary is expected",
                    "Detailed explanations are valued",
                    "Formal academic tone"
                ]
            },
            "beginners": {
                "target_grade": "6th-7th grade",
                "max_sentence_length": 15,
                "flesch_target": "70-80",
                "recommendations": [
                    "Use simple, clear language",
                    "Explain all technical terms",
                    "Short sentences and paragraphs"
                ]
            }
        }
        
        return audience_configs.get(target_audience.lower(), audience_configs["general"])
    
    def _get_improvement_suggestions(self, flesch_score: float, fog_index: float, 
                                   sentence_analysis: Dict, word_analysis: Dict, 
                                   target_audience: str) -> List[str]:
        """Get specific improvement suggestions"""
        suggestions = []
        
        # Flesch score suggestions
        if flesch_score < 50:
            suggestions.append("Simplify vocabulary and reduce sentence length to improve readability")
        elif flesch_score < 60 and target_audience in ["general", "beginners"]:
            suggestions.append("Consider using simpler words and shorter sentences for better accessibility")
        
        # Fog index suggestions
        if fog_index > 12:
            suggestions.append("Reduce complex words and sentence length (Fog Index too high)")
        
        # Sentence length suggestions
        avg_length = sentence_analysis.get("avg_sentence_length", 0)
        if avg_length > 25:
            suggestions.append("Break up long sentences for better readability")
        elif avg_length < 10:
            suggestions.append("Consider combining short sentences for better flow")
        
        # Word complexity suggestions
        complex_percentage = word_analysis.get("complex_word_percentage", 0)
        if complex_percentage > 20:
            suggestions.append("Reduce complex words (3+ syllables) for easier reading")
        
        # Sentence variety suggestions
        variety_score = sentence_analysis.get("sentence_variety_score", 0)
        if variety_score < 30:
            suggestions.append("Vary sentence lengths for better reading rhythm")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _get_readability_verdict(self, overall_grade: str, target_audience: str) -> str:
        """Get overall readability verdict"""
        audience_targets = {
            "general": ["Elementary", "Middle School", "High School"],
            "professionals": ["High School", "College"],
            "academics": ["College", "Graduate"],
            "beginners": ["Elementary", "Middle School"]
        }
        
        target_grades = audience_targets.get(target_audience.lower(), ["High School"])
        
        if overall_grade in target_grades:
            return f"✅ Good readability for {target_audience} audience"
        elif overall_grade == "Graduate" and target_audience not in ["academics", "professionals"]:
            return f"⚠️ Too complex for {target_audience} audience"
        elif overall_grade == "Elementary" and target_audience == "academics":
            return f"⚠️ Too simple for {target_audience} audience"
        else:
            return f"📊 Acceptable readability for {target_audience} audience"


class GrammarStyleCheckerTool(BaseTool):
    """Check grammar, style, and writing quality"""
    name: str = "grammar_style_checker"
    description: str = "Check content for grammar errors, style issues, and writing quality problems."
    
    def _run(self, content: str, style_guide: str = "general") -> str:
        """Execute grammar and style checking"""
        try:
            # Perform various checks
            grammar_issues = self._check_grammar(content)
            style_issues = self._check_style(content, style_guide)
            punctuation_issues = self._check_punctuation(content)
            consistency_issues = self._check_consistency(content)
            
            # Calculate overall score
            total_issues = len(grammar_issues) + len(style_issues) + len(punctuation_issues) + len(consistency_issues)
            word_count = len(content.split())
            quality_score = max(0, 100 - (total_issues / word_count * 1000)) if word_count > 0 else 0
            
            result = {
                "quality_score": round(quality_score, 1),
                "total_issues": total_issues,
                "grammar_issues": grammar_issues,
                "style_issues": style_issues,
                "punctuation_issues": punctuation_issues,
                "consistency_issues": consistency_issues,
                "improvement_suggestions": self._get_writing_suggestions(
                    grammar_issues, style_issues, punctuation_issues, consistency_issues
                ),
                "style_guide": style_guide,
                "overall_verdict": self._get_quality_verdict(quality_score)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error in grammar and style checking: {str(e)}"
    
    def _check_grammar(self, content: str) -> List[Dict[str, str]]:
        """Check for common grammar issues"""
        issues = []
        
        # Check for common grammar mistakes
        grammar_patterns = [
            (r'\bit\'s\s+a\s+it\b', "it's a it", "Redundant 'it's a it' - choose either 'it's' or 'it is'"),
            (r'\bthere\s+are\s+a\b', "there are a", "Use 'there is a' for singular nouns"),
            (r'\bless\s+\w+s\b', "less with countable", "Use 'fewer' with countable nouns, 'less' with uncountable"),
            (r'\bwho\s+are\s+\w+ing\b', "who are + -ing", "Consider 'who' followed by simple verb form"),
            (r'\band\s+and\b', "and and", "Duplicate 'and' - remove one"),
            (r'\bthe\s+the\b', "the the", "Duplicate 'the' - remove one"),
            (r'\bof\s+of\b', "of of", "Duplicate 'of' - remove one")
        ]
        
        for pattern, issue_type, suggestion in grammar_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append({
                    "type": "grammar",
                    "issue": issue_type,
                    "text": match.group(),
                    "position": match.start(),
                    "suggestion": suggestion
                })
        
        return issues[:10]  # Limit to 10 issues
    
    def _check_style(self, content: str, style_guide: str) -> List[Dict[str, str]]:
        """Check for style issues"""
        issues = []
        
        # Common style issues
        style_patterns = [
            (r'\bvery\s+\w+', "very + adjective", "Consider using a stronger adjective instead of 'very'"),
            (r'\bthat\s+is\s+\w+', "that is", "Consider removing 'that is' for conciseness"),
            (r'\bin\s+order\s+to\b', "in order to", "Consider using 'to' instead of 'in order to'"),
            (r'\bdue\s+to\s+the\s+fact\s+that\b', "due to the fact that", "Consider using 'because' instead"),
            (r'\bit\s+is\s+important\s+to\s+note\s+that\b', "it is important to note", "Consider more direct phrasing"),
            (r'\bthere\s+are\s+many\b', "there are many", "Be more specific than 'many'"),
            (r'\ba\s+lot\s+of\b', "a lot of", "Consider 'many', 'much', or 'numerous' for formal writing")
        ]
        
        # Apply style guide specific checks
        if style_guide == "academic":
            style_patterns.extend([
                (r'\bdon\'t\b', "contractions", "Avoid contractions in academic writing"),
                (r'\bcan\'t\b', "contractions", "Avoid contractions in academic writing"),
                (r'\bwon\'t\b', "contractions", "Avoid contractions in academic writing"),
                (r'\bi\s+think\b', "personal opinion", "Use more objective language in academic writing")
            ])
        elif style_guide == "business":
            style_patterns.extend([
                (r'\bkinda\b', "informal language", "Use more professional language"),
                (r'\bsorta\b', "informal language", "Use more professional language"),
                (r'\bawesome\b', "informal language", "Consider 'excellent' or 'outstanding'")
            ])
        
        for pattern, issue_type, suggestion in style_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append({
                    "type": "style",
                    "issue": issue_type,
                    "text": match.group(),
                    "position": match.start(),
                    "suggestion": suggestion
                })
        
        return issues[:10]  # Limit to 10 issues
    
    def _check_punctuation(self, content: str) -> List[Dict[str, str]]:
        """Check for punctuation issues"""
        issues = []
        
        # Punctuation patterns
        punctuation_patterns = [
            (r'\s+,', "space before comma", "Remove space before comma"),
            (r'\s+\.', "space before period", "Remove space before period"),
            (r',,', "double comma", "Remove duplicate comma"),
            (r'\.\.', "double period", "Remove duplicate period"),
            (r'\?\?', "double question mark", "Use single question mark"),
            (r'!{2,}', "multiple exclamation", "Use single exclamation mark"),
            (r'[A-Za-z]\.[A-Za-z]', "missing space after period", "Add space after period"),
            (r'[A-Za-z],[A-Za-z]', "missing space after comma", "Add space after comma")
        ]
        
        for pattern, issue_type, suggestion in punctuation_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                issues.append({
                    "type": "punctuation",
                    "issue": issue_type,
                    "text": match.group(),
                    "position": match.start(),
                    "suggestion": suggestion
                })
        
        return issues[:10]  # Limit to 10 issues
    
    def _check_consistency(self, content: str) -> List[Dict[str, str]]:
        """Check for consistency issues"""
        issues = []
        
        # Check for spelling variations
        variations = [
            (r'\b(color|colour)\b', "color/colour", "Use consistent spelling throughout"),
            (r'\b(organize|organise)\b', "organize/organise", "Use consistent spelling throughout"),
            (r'\b(realize|realise)\b', "realize/realise", "Use consistent spelling throughout"),
            (r'\b(center|centre)\b', "center/centre", "Use consistent spelling throughout")
        ]
        
        for pattern, issue_type, suggestion in variations:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if len(matches) > 1:
                # Check if different spellings are used
                spellings = set(match.group().lower() for match in matches)
                if len(spellings) > 1:
                    issues.append({
                        "type": "consistency",
                        "issue": issue_type,
                        "text": f"Found: {', '.join(spellings)}",
                        "position": matches[0].start(),
                        "suggestion": suggestion
                    })
        
        # Check for capitalization consistency
        sentences = re.split(r'[.!?]+', content)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                issues.append({
                    "type": "consistency",
                    "issue": "capitalization",
                    "text": sentence[:20] + "...",
                    "position": 0,
                    "suggestion": "Capitalize first letter of sentence"
                })
        
        return issues[:5]  # Limit to 5 issues
    
    def _get_writing_suggestions(self, grammar_issues: List, style_issues: List, 
                               punctuation_issues: List, consistency_issues: List) -> List[str]:
        """Get overall writing improvement suggestions"""
        suggestions = []
        
        if grammar_issues:
            suggestions.append(f"Fix {len(grammar_issues)} grammar issues for better clarity")
        
        if style_issues:
            suggestions.append(f"Address {len(style_issues)} style issues for more professional writing")
        
        if punctuation_issues:
            suggestions.append(f"Correct {len(punctuation_issues)} punctuation errors")
        
        if consistency_issues:
            suggestions.append(f"Resolve {len(consistency_issues)} consistency issues")
        
        # General suggestions
        suggestions.extend([
            "Proofread content carefully before publishing",
            "Consider using writing tools for additional checking",
            "Read content aloud to catch rhythm and flow issues"
        ])
        
        return suggestions[:5]
    
    def _get_quality_verdict(self, quality_score: float) -> str:
        """Get overall quality verdict"""
        if quality_score >= 90:
            return "✅ Excellent writing quality"
        elif quality_score >= 80:
            return "✅ Good writing quality"
        elif quality_score >= 70:
            return "⚠️ Fair writing quality - minor improvements needed"
        elif quality_score >= 60:
            return "⚠️ Poor writing quality - significant improvements needed"
        else:
            return "❌ Very poor writing quality - major revision required"


class BrandComplianceCheckerTool(BaseTool):
    """Check content compliance with brand guidelines"""
    name: str = "brand_compliance_checker"
    description: str = "Check content for compliance with brand guidelines, tone, voice, and messaging requirements."
    
    def _run(self, content: str, brand_guidelines: str = "", brand_voice: str = "professional") -> str:
        """Execute brand compliance checking"""
        try:
            # Parse brand guidelines
            guidelines = self._parse_brand_guidelines(brand_guidelines)
            
            # Perform compliance checks
            tone_analysis = self._analyze_tone_compliance(content, brand_voice)
            keyword_compliance = self._check_brand_keywords(content, guidelines)
            messaging_compliance = self._check_messaging_compliance(content, guidelines)
            style_compliance = self._check_style_compliance(content, guidelines)
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(
                tone_analysis, keyword_compliance, messaging_compliance, style_compliance
            )
            
            result = {
                "compliance_score": compliance_score,
                "tone_analysis": tone_analysis,
                "keyword_compliance": keyword_compliance,
                "messaging_compliance": messaging_compliance,
                "style_compliance": style_compliance,
                "recommendations": self._get_compliance_recommendations(
                    tone_analysis, keyword_compliance, messaging_compliance, style_compliance
                ),
                "brand_alignment_verdict": self._get_alignment_verdict(compliance_score)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error in brand compliance checking: {str(e)}"
    
    def _parse_brand_guidelines(self, guidelines: str) -> Dict[str, List[str]]:
        """Parse brand guidelines into structured format"""
        if not guidelines:
            return {
                "required_keywords": [],
                "avoid_words": [],
                "brand_values": [],
                "messaging_pillars": []
            }
        
        # Simple parsing - in production, this would be more sophisticated
        lines = guidelines.split('\n')
        parsed = {
            "required_keywords": [],
            "avoid_words": [],
            "brand_values": [],
            "messaging_pillars": []
        }
        
        current_section = None
        for line in lines:
            line = line.strip().lower()
            if 'required' in line or 'keywords' in line:
                current_section = "required_keywords"
            elif 'avoid' in line or 'forbidden' in line:
                current_section = "avoid_words"
            elif 'values' in line:
                current_section = "brand_values"
            elif 'messaging' in line or 'pillars' in line:
                current_section = "messaging_pillars"
            elif line and current_section:
                parsed[current_section].append(line.strip('- '))
        
        return parsed
    
    def _analyze_tone_compliance(self, content: str, brand_voice: str) -> Dict[str, Any]:
        """Analyze tone compliance with brand voice"""
        content_lower = content.lower()
        
        # Define tone indicators for different brand voices
        tone_indicators = {
            "professional": {
                "positive": ["expertise", "professional", "quality", "reliable", "experienced"],
                "negative": ["awesome", "super", "totally", "kinda", "sorta"],
                "target_score": 80
            },
            "friendly": {
                "positive": ["welcome", "help", "support", "community", "together"],
                "negative": ["corporate", "enterprise", "systematic", "compliance"],
                "target_score": 75
            },
            "authoritative": {
                "positive": ["proven", "leader", "expert", "authority", "certified"],
                "negative": ["maybe", "perhaps", "possibly", "might"],
                "target_score": 85
            },
            "casual": {
                "positive": ["easy", "simple", "fun", "cool", "awesome"],
                "negative": ["formal", "systematic", "comprehensive", "enterprise"],
                "target_score": 70
            }
        }
        
        indicators = tone_indicators.get(brand_voice, tone_indicators["professional"])
        
        # Count positive and negative indicators
        positive_count = sum(1 for word in indicators["positive"] if word in content_lower)
        negative_count = sum(1 for word in indicators["negative"] if word in content_lower)
        
        # Calculate tone score
        total_words = len(content.split())
        positive_ratio = (positive_count / total_words) * 100 if total_words > 0 else 0
        negative_ratio = (negative_count / total_words) * 100 if total_words > 0 else 0
        
        tone_score = max(0, 100 - (negative_ratio * 10) + (positive_ratio * 5))
        
        return {
            "brand_voice": brand_voice,
            "tone_score": round(tone_score, 1),
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "meets_target": tone_score >= indicators["target_score"],
            "tone_assessment": self._get_tone_assessment(tone_score, brand_voice)
        }
    
    def _check_brand_keywords(self, content: str, guidelines: Dict[str, List[str]]) -> Dict[str, Any]:
        """Check for required and forbidden keywords"""
        content_lower = content.lower()
        
        required_keywords = guidelines.get("required_keywords", [])
        avoid_words = guidelines.get("avoid_words", [])
        
        # Check required keywords
        missing_required = [kw for kw in required_keywords if kw not in content_lower]
        present_required = [kw for kw in required_keywords if kw in content_lower]
        
        # Check forbidden words
        found_forbidden = [word for word in avoid_words if word in content_lower]
        
        compliance_score = 100
        if required_keywords:
            compliance_score -= (len(missing_required) / len(required_keywords)) * 50
        compliance_score -= len(found_forbidden) * 10
        
        return {
            "required_keywords": {
                "total": len(required_keywords),
                "present": len(present_required),
                "missing": missing_required
            },
            "forbidden_words": {
                "found": found_forbidden,
                "count": len(found_forbidden)
            },
            "keyword_compliance_score": max(0, compliance_score)
        }
    
    def _check_messaging_compliance(self, content: str, guidelines: Dict[str, List[str]]) -> Dict[str, Any]:
        """Check messaging compliance with brand pillars"""
        brand_values = guidelines.get("brand_values", [])
        messaging_pillars = guidelines.get("messaging_pillars", [])
        
        content_lower = content.lower()
        
        # Check alignment with brand values
        values_mentioned = sum(1 for value in brand_values if value in content_lower)
        pillars_mentioned = sum(1 for pillar in messaging_pillars if pillar in content_lower)
        
        total_brand_elements = len(brand_values) + len(messaging_pillars)
        total_mentioned = values_mentioned + pillars_mentioned
        
        messaging_score = (total_mentioned / total_brand_elements * 100) if total_brand_elements > 0 else 100
        
        return {
            "brand_values_alignment": {
                "total_values": len(brand_values),
                "mentioned": values_mentioned,
                "percentage": round((values_mentioned / len(brand_values) * 100), 1) if brand_values else 100
            },
            "messaging_pillars_alignment": {
                "total_pillars": len(messaging_pillars),
                "mentioned": pillars_mentioned,
                "percentage": round((pillars_mentioned / len(messaging_pillars) * 100), 1) if messaging_pillars else 100
            },
            "messaging_compliance_score": round(messaging_score, 1)
        }
    
    def _check_style_compliance(self, content: str, guidelines: Dict[str, List[str]]) -> Dict[str, Any]:
        """Check style compliance"""
        # Check basic style elements
        style_checks = {
            "proper_capitalization": self._check_capitalization(content),
            "consistent_punctuation": self._check_punctuation_consistency(content),
            "appropriate_length": self._check_content_length(content),
            "paragraph_structure": self._check_paragraph_structure(content)
        }
        
        passed_checks = sum(1 for check in style_checks.values() if check)
        style_score = (passed_checks / len(style_checks)) * 100
        
        return {
            "style_checks": style_checks,
            "style_compliance_score": round(style_score, 1),
            "style_issues": [name for name, passed in style_checks.items() if not passed]
        }
    
    def _check_capitalization(self, content: str) -> bool:
        """Check proper capitalization"""
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                return False
        return True
    
    def _check_punctuation_consistency(self, content: str) -> bool:
        """Check punctuation consistency"""
        # Simple check for double punctuation
        return not re.search(r'[.!?]{2,}', content)
    
    def _check_content_length(self, content: str) -> bool:
        """Check if content length is appropriate"""
        word_count = len(content.split())
        return 300 <= word_count <= 3000  # Reasonable range
    
    def _check_paragraph_structure(self, content: str) -> bool:
        """Check paragraph structure"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if not paragraphs:
            return False
        
        # Check if paragraphs are reasonable length
        avg_para_words = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
        return 30 <= avg_para_words <= 200
    
    def _calculate_compliance_score(self, tone_analysis: Dict, keyword_compliance: Dict, 
                                  messaging_compliance: Dict, style_compliance: Dict) -> float:
        """Calculate overall compliance score"""
        scores = [
            tone_analysis.get("tone_score", 0) * 0.3,
            keyword_compliance.get("keyword_compliance_score", 0) * 0.3,
            messaging_compliance.get("messaging_compliance_score", 0) * 0.2,
            style_compliance.get("style_compliance_score", 0) * 0.2
        ]
        
        return round(sum(scores), 1)
    
    def _get_tone_assessment(self, tone_score: float, brand_voice: str) -> str:
        """Get tone assessment"""
        if tone_score >= 80:
            return f"✅ Tone aligns well with {brand_voice} brand voice"
        elif tone_score >= 60:
            return f"⚠️ Tone somewhat aligns with {brand_voice} brand voice"
        else:
            return f"❌ Tone does not align with {brand_voice} brand voice"
    
    def _get_compliance_recommendations(self, tone_analysis: Dict, keyword_compliance: Dict, 
                                     messaging_compliance: Dict, style_compliance: Dict) -> List[str]:
        """Get compliance improvement recommendations"""
        recommendations = []
        
        if tone_analysis.get("tone_score", 0) < 70:
            recommendations.append(f"Adjust tone to better match {tone_analysis.get('brand_voice')} brand voice")
        
        missing_keywords = keyword_compliance.get("required_keywords", {}).get("missing", [])
        if missing_keywords:
            recommendations.append(f"Include required brand keywords: {', '.join(missing_keywords[:3])}")
        
        forbidden_found = keyword_compliance.get("forbidden_words", {}).get("found", [])
        if forbidden_found:
            recommendations.append(f"Remove forbidden words: {', '.join(forbidden_found)}")
        
        if messaging_compliance.get("messaging_compliance_score", 0) < 70:
            recommendations.append("Better align content with brand values and messaging pillars")
        
        style_issues = style_compliance.get("style_issues", [])
        if style_issues:
            recommendations.append(f"Address style issues: {', '.join(style_issues[:2])}")
        
        return recommendations[:5]
    
    def _get_alignment_verdict(self, compliance_score: float) -> str:
        """Get overall brand alignment verdict"""
        if compliance_score >= 90:
            return "✅ Excellent brand alignment"
        elif compliance_score >= 80:
            return "✅ Good brand alignment"
        elif compliance_score >= 70:
            return "⚠️ Fair brand alignment - minor adjustments needed"
        elif compliance_score >= 60:
            return "⚠️ Poor brand alignment - significant improvements needed"
        else:
            return "❌ Very poor brand alignment - major revision required"


def get_all_quality_tools() -> List[BaseTool]:
    """Get all quality assurance tools"""
    return [
        ReadabilityAnalyzerTool(),
        GrammarStyleCheckerTool(),
        BrandComplianceCheckerTool()
    ]