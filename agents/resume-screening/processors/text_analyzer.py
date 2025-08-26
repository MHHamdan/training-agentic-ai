import logging
import re
from typing import Dict, List, Any, Optional
from langsmith import traceable
import asyncio

from models.prompts import PromptTemplates

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        self.prompt_templates = PromptTemplates()
        self.skill_patterns = self._compile_skill_patterns()
    
    def _compile_skill_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "programming_languages": re.compile(
                r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|R|MATLAB|Scala|Perl)\b',
                re.IGNORECASE
            ),
            "frameworks": re.compile(
                r'\b(React|Angular|Vue|Django|Flask|Spring|Express|FastAPI|Rails|Laravel|ASP\.NET|Node\.js|TensorFlow|PyTorch|Scikit-learn)\b',
                re.IGNORECASE
            ),
            "databases": re.compile(
                r'\b(MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|SQL Server|SQLite|DynamoDB|Elasticsearch|Neo4j)\b',
                re.IGNORECASE
            ),
            "cloud": re.compile(
                r'\b(AWS|Azure|GCP|Google Cloud|Docker|Kubernetes|Terraform|CloudFormation|Jenkins|CI/CD|DevOps)\b',
                re.IGNORECASE
            ),
            "tools": re.compile(
                r'\b(Git|GitHub|GitLab|Jira|Confluence|Slack|VS Code|IntelliJ|Eclipse|Postman|Swagger|GraphQL|REST API)\b',
                re.IGNORECASE
            )
        }
    
    @traceable(name="analyze_resume", metadata={"component": "text_analyzer"})
    async def analyze_resume(
        self,
        resume_text: str,
        job_requirements: str,
        model: Any
    ) -> Dict[str, Any]:
        try:
            # First, try to get analysis from the model
            prompt = self.prompt_templates.RESUME_ANALYSIS.format(
                resume_text=resume_text[:3000],
                job_requirements=job_requirements[:1000]
            )
            
            analysis = None
            model_failed = False
            
            try:
                # Temporarily force content-based analysis for debugging
                logger.info("Forcing content-based analysis for debugging")
                raise Exception("Debug mode: forcing content-based analysis")
                
                # Original code (disabled for debugging)
                # if hasattr(model, 'analyze_resume'):
                #     analysis = await model.analyze_resume(
                #         resume_text, job_requirements, prompt
                #     )
                # else:
                #     response = await model.generate(prompt)
                #     analysis = self._parse_analysis_response(response)
                    
            except Exception as model_error:
                logger.warning(f"Model analysis failed: {str(model_error)}. Falling back to content-based analysis.")
                model_failed = True
            
            # If model failed or returned invalid analysis, use content-based analysis
            if model_failed or not analysis or "error" in analysis:
                logger.info("Using content-based analysis as fallback")
                analysis = await self._content_based_analysis(resume_text, job_requirements)
            
            # Extract additional metadata
            skills = await self.extract_skills(resume_text)
            experience_years = self._extract_experience_years(resume_text)
            education = self._extract_education(resume_text)
            
            analysis.update({
                "extracted_skills": skills,
                "experience_years": experience_years,
                "education": education
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}")
            # Even in error case, provide content-based analysis
            return await self._content_based_analysis(resume_text, job_requirements)
    
    @traceable(name="extract_skills", metadata={"component": "text_analyzer"})
    async def extract_skills(self, text: str) -> Dict[str, List[str]]:
        skills = {}
        
        for category, pattern in self.skill_patterns.items():
            matches = pattern.findall(text)
            skills[category] = list(set(matches))
        
        return skills
    
    def _extract_experience_years(self, text: str) -> Optional[int]:
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience:\s*(\d+)\+?\s*years?',
            r'(\d+)\s*years?\s*in\s*',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except:
                    pass
        
        year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        years = year_pattern.findall(text)
        if len(years) >= 2:
            try:
                min_year = min(int(y) for y in years)
                max_year = max(int(y) for y in years)
                return max_year - min_year
            except:
                pass
        
        return None
    
    def _extract_education(self, text: str) -> List[str]:
        education = []
        
        degree_patterns = [
            r"(?:Bachelor|Master|PhD|Ph\.D\.|Doctorate|Associate|MBA|MS|MA|BS|BA|B\.S\.|B\.A\.|M\.S\.|M\.A\.)[^.]*",
            r"(?:Computer Science|Engineering|Business|Mathematics|Physics|Chemistry|Biology)[^.]*(?:degree|Degree)",
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education.extend(matches)
        
        return list(set(education))[:5]
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        try:
            # Try to extract scores from model response first
            scores = {}
            score_patterns = {
                "technical_skills": r"technical\s*skills?:?\s*(\d+)",
                "experience_relevance": r"experience\s*relevance?:?\s*(\d+)",
                "cultural_fit": r"cultural\s*fit?:?\s*(\d+)",
                "growth_potential": r"growth\s*potential?:?\s*(\d+)",
                "risk_assessment": r"risk\s*assessment?:?\s*(\d+)",
                "overall": r"overall?:?\s*(\d+)"
            }
            
            for key, pattern in score_patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        scores[key] = min(100, max(0, int(match.group(1))))
                    except:
                        pass
            
            # If no scores were extracted from response, return error for fallback handling
            if not scores:
                logger.warning("No scores found in model response, triggering fallback analysis")
                raise ValueError("No parseable scores in model response")
            
            # Fill in missing scores with reasonable defaults
            default_scores = {
                "technical_skills": 60,
                "experience_relevance": 65,
                "cultural_fit": 70,
                "growth_potential": 75,
                "risk_assessment": 80,
                "overall": 65
            }
            
            for key, default_value in default_scores.items():
                if key not in scores:
                    scores[key] = default_value
            
            insights = []
            lines = response.split('\n')
            for line in lines:
                if line.strip() and not any(word in line.lower() for word in ['score', 'technical', 'experience']):
                    if len(line.strip()) > 20:
                        insights.append(line.strip())
            
            return {
                "scores": scores,
                "insights": insights[:10],
                "raw_response": response[:1000]
            }
            
        except Exception as e:
            logger.error(f"Error parsing response, will use content-based analysis: {str(e)}")
            # Don't return default scores here - let the calling function handle fallback
            raise e
    
    @traceable(name="calculate_skill_match", metadata={"component": "text_analyzer"})
    async def calculate_skill_match(
        self,
        resume_skills: Dict[str, List[str]],
        required_skills: List[str]
    ) -> float:
        all_resume_skills = []
        for skills_list in resume_skills.values():
            all_resume_skills.extend([s.lower() for s in skills_list])
        
        required_lower = [s.lower() for s in required_skills]
        
        if not required_lower:
            return 0.0
        
        matched = sum(1 for skill in required_lower if skill in all_resume_skills)
        return round((matched / len(required_lower)) * 100, 2)
    
    async def _content_based_analysis(self, resume_text: str, job_requirements: str) -> Dict[str, Any]:
        """Fallback content-based analysis when model analysis fails"""
        try:
            # Extract skills and calculate technical score
            skills = await self.extract_skills(resume_text)
            
            # Extract job requirements skills
            job_skills = await self.extract_skills(job_requirements)
            all_job_skills = []
            for skill_list in job_skills.values():
                all_job_skills.extend(skill_list)
            
            # Calculate technical skills score based on skill matching
            skill_match = await self.calculate_skill_match(skills, all_job_skills)
            technical_score = min(100, max(20, skill_match * 1.2))  # Scale and bound
            
            # Calculate experience relevance
            experience_years = self._extract_experience_years(resume_text)
            experience_score = self._calculate_experience_score(experience_years, job_requirements)
            
            # Calculate cultural fit based on content analysis
            cultural_score = self._calculate_cultural_fit_score(resume_text)
            
            # Calculate growth potential based on education and career progression
            education = self._extract_education(resume_text)
            growth_score = self._calculate_growth_potential_score(resume_text, education, experience_years)
            
            # Calculate risk assessment
            risk_score = self._calculate_risk_score(resume_text, experience_years)
            
            # Calculate overall score as weighted average
            overall_score = round(
                (technical_score * 0.3 + 
                 experience_score * 0.25 + 
                 cultural_score * 0.2 + 
                 growth_score * 0.15 + 
                 risk_score * 0.1), 1
            )
            
            # Generate insights based on analysis
            insights = self._generate_insights(
                resume_text, job_requirements, skills, experience_years, 
                technical_score, experience_score, cultural_score
            )
            
            return {
                "scores": {
                    "technical_skills": round(technical_score, 1),
                    "experience_relevance": round(experience_score, 1),
                    "cultural_fit": round(cultural_score, 1),
                    "growth_potential": round(growth_score, 1),
                    "risk_assessment": round(risk_score, 1),
                    "overall": round(overall_score, 1)
                },
                "insights": insights,
                "raw_response": "Content-based analysis (model unavailable)",
                "analysis_type": "content_based"
            }
            
        except Exception as e:
            logger.error(f"Error in content-based analysis: {str(e)}")
            return {
                "scores": {
                    "technical_skills": 30,
                    "experience_relevance": 40,
                    "cultural_fit": 50,
                    "growth_potential": 45,
                    "risk_assessment": 60,
                    "overall": 42
                },
                "insights": [f"Analysis error: {str(e)}", "Scores are estimated based on limited information"],
                "error": str(e),
                "analysis_type": "fallback"
            }
    
    def _calculate_experience_score(self, experience_years: Optional[int], job_requirements: str) -> float:
        """Calculate experience relevance score"""
        if experience_years is None:
            return 40.0  # Default for unknown experience
        
        # Extract required experience from job requirements
        req_exp_match = re.search(r'(\d+)\+?\s*years?\s*(?:of\s*)?experience', job_requirements, re.IGNORECASE)
        required_years = 3  # Default requirement
        if req_exp_match:
            required_years = int(req_exp_match.group(1))
        
        if experience_years >= required_years:
            # Scale score: exact match = 80, more experience = higher score up to 95
            excess_years = experience_years - required_years
            return min(95, 80 + (excess_years * 2))
        else:
            # Penalty for less experience: scale down from 70
            shortage = required_years - experience_years
            return max(20, 70 - (shortage * 10))
    
    def _calculate_cultural_fit_score(self, resume_text: str) -> float:
        """Calculate cultural fit based on keywords and phrases"""
        cultural_indicators = {
            "collaboration": ["team", "collaborate", "partnership", "cross-functional", "worked with"],
            "leadership": ["led", "managed", "mentored", "guided", "supervised", "directed"],
            "communication": ["presented", "communication", "documented", "explained", "trained"],
            "innovation": ["innovative", "created", "developed", "designed", "implemented"],
            "adaptability": ["adapted", "learned", "flexible", "agile", "diverse", "various"]
        }
        
        text_lower = resume_text.lower()
        total_score = 0
        
        for category, keywords in cultural_indicators.items():
            category_score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    category_score += 1
            # Scale category score to 0-20 range
            category_score = min(20, category_score * 3)
            total_score += category_score
        
        return max(30, min(85, total_score))
    
    def _calculate_growth_potential_score(self, resume_text: str, education: List[str], experience_years: Optional[int]) -> float:
        """Calculate growth potential score"""
        score = 50  # Base score
        
        # Education bonus
        if education:
            for edu in education:
                if any(degree in edu.lower() for degree in ["master", "phd", "mba", "doctorate"]):
                    score += 15
                elif any(degree in edu.lower() for degree in ["bachelor", "bs", "ba"]):
                    score += 10
        
        # Learning indicators
        learning_keywords = ["certification", "course", "training", "learned", "studied", "bootcamp"]
        text_lower = resume_text.lower()
        
        for keyword in learning_keywords:
            if keyword in text_lower:
                score += 3
        
        # Career progression (if we can detect it)
        if experience_years and experience_years > 2:
            # Look for progression indicators
            progression_keywords = ["promoted", "advanced", "senior", "lead", "principal", "manager"]
            for keyword in progression_keywords:
                if keyword in text_lower:
                    score += 5
        
        return max(40, min(90, score))
    
    def _calculate_risk_score(self, resume_text: str, experience_years: Optional[int]) -> float:
        """Calculate risk assessment score (higher is better/lower risk)"""
        score = 70  # Base score (moderate risk)
        
        # Check for employment gaps (basic check)
        years_mentioned = re.findall(r'\b(20\d{2}|19\d{2})\b', resume_text)
        if len(years_mentioned) >= 4:  # Multiple years suggest consistent employment
            years_int = [int(y) for y in years_mentioned]
            year_gaps = []
            sorted_years = sorted(set(years_int))
            for i in range(1, len(sorted_years)):
                gap = sorted_years[i] - sorted_years[i-1]
                if gap > 2:  # Gap of more than 2 years
                    year_gaps.append(gap)
            
            if not year_gaps:
                score += 10  # No significant gaps
            elif len(year_gaps) == 1 and year_gaps[0] <= 3:
                score -= 5   # One small gap
            else:
                score -= 15  # Multiple or large gaps
        
        # Job hopping check (rough estimate)
        if experience_years and experience_years > 5:
            # Count job-related terms
            job_indicators = resume_text.lower().count("company") + resume_text.lower().count("position")
            if job_indicators > experience_years / 2:  # Many jobs relative to experience
                score -= 10
        
        # Stability indicators
        stability_keywords = ["years", "established", "consistent", "stable", "long-term"]
        for keyword in stability_keywords:
            if keyword in resume_text.lower():
                score += 3
        
        return max(40, min(85, score))
    
    def _generate_insights(self, resume_text: str, job_requirements: str, skills: Dict[str, List[str]], 
                          experience_years: Optional[int], technical_score: float, 
                          experience_score: float, cultural_score: float) -> List[str]:
        """Generate meaningful insights based on the analysis"""
        insights = []
        
        # Technical skills insights
        total_skills = sum(len(skill_list) for skill_list in skills.values())
        if total_skills > 15:
            insights.append(f"Strong technical breadth with {total_skills} identified skills")
        elif total_skills > 8:
            insights.append(f"Good technical foundation with {total_skills} relevant skills")
        else:
            insights.append(f"Limited technical skills identified ({total_skills} skills)")
        
        # Experience insights
        if experience_years:
            if experience_score > 80:
                insights.append(f"Excellent experience match with {experience_years} years in field")
            elif experience_score > 60:
                insights.append(f"Good experience level with {experience_years} years background")
            else:
                insights.append(f"Experience gap noted - {experience_years} years may not fully meet requirements")
        
        # Cultural fit insights
        if cultural_score > 70:
            insights.append("Strong cultural fit indicators - collaborative and adaptable profile")
        elif cultural_score > 50:
            insights.append("Moderate cultural fit - some team collaboration experience evident")
        else:
            insights.append("Limited cultural fit indicators - may need assessment of soft skills")
        
        # Add specific skill highlights
        for category, skill_list in skills.items():
            if len(skill_list) > 3:
                insights.append(f"Strong {category.replace('_', ' ')} background: {', '.join(skill_list[:3])}")
        
        return insights[:8]  # Limit to top insights