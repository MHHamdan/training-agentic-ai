class PromptTemplates:
    
    RESUME_ANALYSIS = """Analyze the following resume against the job requirements and provide a comprehensive assessment.

RESUME:
{resume_text}

JOB REQUIREMENTS:
{job_requirements}

Please provide your analysis in the following format:

SCORES (0-100):
Technical Skills: [score]
Experience Relevance: [score]
Cultural Fit: [score]
Growth Potential: [score]
Risk Assessment: [score]
Overall: [score]

KEY STRENGTHS:
- [strength 1]
- [strength 2]
- [strength 3]

GAPS/CONCERNS:
- [gap 1]
- [gap 2]

RECOMMENDATION:
[Your hiring recommendation and reasoning]
"""

    SKILL_EXTRACTION = """Extract all technical and soft skills from the following resume:

{resume_text}

List the skills in these categories:
1. Programming Languages:
2. Frameworks/Libraries:
3. Tools/Technologies:
4. Soft Skills:
5. Domain Expertise:
"""

    EXPERIENCE_MATCHING = """Compare the candidate's experience with the job requirements:

CANDIDATE EXPERIENCE:
{resume_text}

JOB REQUIREMENTS:
{job_requirements}

Provide:
1. Matching experiences (with relevance score 0-100)
2. Missing experiences
3. Transferable skills
4. Overall experience match percentage
"""

    CULTURAL_FIT_ASSESSMENT = """Assess the cultural fit based on the resume content:

{resume_text}

Look for indicators of:
- Team collaboration
- Leadership potential
- Communication style
- Work ethic
- Innovation mindset
- Adaptability

Provide a cultural fit score (0-100) with justification.
"""

    RISK_ASSESSMENT = """Identify potential risks or red flags in this resume:

{resume_text}

Consider:
- Employment gaps
- Job hopping patterns
- Overqualification/Underqualification
- Inconsistencies
- Missing critical information

Provide a risk score (0-100) where higher means lower risk.
"""

    COMPARATIVE_ANALYSIS = """Compare this candidate against typical requirements for the role:

RESUME:
{resume_text}

ROLE REQUIREMENTS:
{job_requirements}

Provide:
1. How this candidate compares to typical candidates
2. Unique value propositions
3. Areas where candidate exceeds requirements
4. Critical gaps that need addressing
"""

    GROWTH_POTENTIAL = """Assess the candidate's growth potential:

{resume_text}

Evaluate:
- Learning agility indicators
- Career progression pattern
- Skill acquisition rate
- Leadership potential
- Adaptability to new technologies

Provide a growth potential score (0-100) with explanation.
"""

    @classmethod
    def get_template(cls, template_name: str) -> str:
        return getattr(cls, template_name.upper(), cls.RESUME_ANALYSIS)