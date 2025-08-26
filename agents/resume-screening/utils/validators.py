import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.rtf'}
        self.min_text_length = 100
        self.max_text_length = 50000
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            path = Path(file_path)
            
            if not path.exists():
                result["valid"] = False
                result["errors"].append(f"File does not exist: {file_path}")
                return result
            
            if not path.is_file():
                result["valid"] = False
                result["errors"].append(f"Path is not a file: {file_path}")
                return result
            
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                result["valid"] = False
                result["errors"].append(f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum ({self.max_file_size / 1024 / 1024}MB)")
            
            if file_size == 0:
                result["valid"] = False
                result["errors"].append("File is empty")
            
            extension = path.suffix.lower()
            if extension not in self.allowed_extensions:
                result["valid"] = False
                result["errors"].append(f"File type {extension} not supported. Allowed: {self.allowed_extensions}")
            
            if not os.access(path, os.R_OK):
                result["valid"] = False
                result["errors"].append("File is not readable")
            
            if result["valid"]:
                result["message"] = "File validation successful"
            else:
                result["message"] = f"File validation failed: {'; '.join(result['errors'])}"
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Validation error: {str(e)}")
            result["message"] = f"Validation error: {str(e)}"
        
        return result
    
    def validate_text(self, text: str) -> Dict[str, Any]:
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not text or not text.strip():
            result["valid"] = False
            result["errors"].append("Text is empty")
            return result
        
        text_length = len(text)
        
        if text_length < self.min_text_length:
            result["warnings"].append(f"Text is very short ({text_length} chars)")
        
        if text_length > self.max_text_length:
            result["warnings"].append(f"Text is very long ({text_length} chars), may be truncated")
        
        if not re.search(r'[a-zA-Z]', text):
            result["valid"] = False
            result["errors"].append("Text contains no alphabetic characters")
        
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'onclick=',
            r'onerror='
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result["warnings"].append(f"Suspicious pattern detected: {pattern}")
        
        return result
    
    def validate_job_requirements(self, requirements: str) -> Dict[str, Any]:
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "extracted_requirements": {}
        }
        
        if not requirements or not requirements.strip():
            result["valid"] = False
            result["errors"].append("Job requirements are empty")
            return result
        
        if len(requirements) < 50:
            result["warnings"].append("Job requirements are very brief")
        
        skills = self._extract_skills(requirements)
        if not skills:
            result["warnings"].append("No specific skills mentioned in requirements")
        else:
            result["extracted_requirements"]["skills"] = skills
        
        experience = self._extract_experience_requirement(requirements)
        if experience:
            result["extracted_requirements"]["experience_years"] = experience
        
        education = self._extract_education_requirement(requirements)
        if education:
            result["extracted_requirements"]["education"] = education
        
        return result
    
    def _extract_skills(self, text: str) -> List[str]:
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Node\.js)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Terraform)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch)\b',
            r'\b(?:Machine Learning|Deep Learning|NLP|Computer Vision)\b',
            r'\b(?:Agile|Scrum|DevOps|CI/CD|Git)\b'
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))
    
    def _extract_experience_requirement(self, text: str) -> Optional[int]:
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'minimum\s*(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except:
                    pass
        
        return None
    
    def _extract_education_requirement(self, text: str) -> List[str]:
        education_patterns = [
            r"(?:Bachelor|Master|PhD|Ph\.D\.|Doctorate|MBA|MS|MA|BS|BA)(?:'s)?(?:\s+degree)?",
            r"(?:Computer Science|Engineering|Business|Mathematics|related field)"
        ]
        
        education = []
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education.extend(matches)
        
        return list(set(education))
    
    def sanitize_input(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        
        text = re.sub(r'[^\w\s\-.,!?@#$%^&*()_+=\[\]{};:\'"\\|/`~]', '', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()