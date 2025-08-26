import unittest
import asyncio
import sys
import os
from pathlib import Path
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.document_processor import DocumentProcessor
from processors.text_analyzer import TextAnalyzer
from processors.vector_store import VectorStoreManager

class TestDocumentProcessor(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.processor = DocumentProcessor()
    
    def test_supported_formats(self):
        self.assertIn('.pdf', self.processor.supported_formats)
        self.assertIn('.docx', self.processor.supported_formats)
        self.assertIn('.txt', self.processor.supported_formats)
    
    def test_extract_text_from_txt(self):
        async def run_test():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Test resume content\nWith multiple lines\nAnd skills: Python, JavaScript")
                temp_path = f.name
            
            try:
                text = await self.processor.extract_text(temp_path)
                return text
            finally:
                os.unlink(temp_path)
        
        text = asyncio.run(run_test())
        self.assertIsNotNone(text)
        self.assertIn("Test resume content", text)
        self.assertIn("Python", text)
    
    def test_sanitize_text(self):
        async def run_test():
            text = "John Doe\nEmail: john@example.com\nPhone: 555-123-4567"
            sanitized = await self.processor.sanitize_text(text)
            return sanitized
        
        sanitized = asyncio.run(run_test())
        self.assertIn("[EMAIL]", sanitized)
        self.assertIn("[PHONE]", sanitized)
        self.assertNotIn("john@example.com", sanitized)

class TestTextAnalyzer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.analyzer = TextAnalyzer()
    
    def test_extract_skills(self):
        async def run_test():
            text = """
            Experienced developer with expertise in Python, Java, React, and AWS.
            Proficient in Docker, Kubernetes, PostgreSQL, and MongoDB.
            Strong knowledge of Git, CI/CD, and Agile methodologies.
            """
            skills = await self.analyzer.extract_skills(text)
            return skills
        
        skills = asyncio.run(run_test())
        self.assertIn("programming_languages", skills)
        self.assertIn("Python", skills["programming_languages"])
        self.assertIn("Java", skills["programming_languages"])
        self.assertIn("cloud", skills)
        self.assertIn("AWS", skills["cloud"])
    
    def test_extract_experience_years(self):
        test_cases = [
            ("I have 5 years of experience", 5),
            ("10+ years experience in software", 10),
            ("Experience: 3 years", 3),
            ("No experience mentioned", None)
        ]
        
        for text, expected in test_cases:
            result = self.analyzer._extract_experience_years(text)
            self.assertEqual(result, expected)
    
    def test_extract_education(self):
        text = """
        Education:
        - Bachelor's degree in Computer Science
        - Master's in Software Engineering
        - PhD in Machine Learning
        """
        
        education = self.analyzer._extract_education(text)
        self.assertTrue(len(education) > 0)
        self.assertTrue(any("Bachelor" in e for e in education))
        self.assertTrue(any("Master" in e for e in education))
    
    def test_calculate_skill_match(self):
        async def run_test():
            resume_skills = {
                "programming_languages": ["Python", "Java"],
                "frameworks": ["Django", "React"],
                "databases": ["PostgreSQL"]
            }
            required_skills = ["Python", "Django", "AWS", "Docker"]
            
            match_score = await self.analyzer.calculate_skill_match(
                resume_skills, required_skills
            )
            return match_score
        
        match_score = asyncio.run(run_test())
        self.assertIsInstance(match_score, float)
        self.assertTrue(0 <= match_score <= 100)
        self.assertEqual(match_score, 50.0)  # 2 out of 4 skills matched

class TestVectorStoreManager(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.vector_store = VectorStoreManager(persist_directory="/tmp/test_chroma_db")
    
    def test_store_and_retrieve_analysis(self):
        async def run_test():
            test_analysis = {
                "file_path": "test_resume.pdf",
                "timestamp": "2024-01-01T12:00:00",
                "comprehensive_score": {"overall": 75.5},
                "recommendation": "RECOMMEND",
                "processing_time": 2.5,
                "status": "success"
            }
            
            success = await self.vector_store.store_analysis(test_analysis)
            self.assertTrue(success)
            
            history = await self.vector_store.get_history(limit=1)
            return history
        
        history = asyncio.run(run_test())
        self.assertIsInstance(history, list)
    
    def test_search_similar(self):
        async def run_test():
            query = "Python developer with Django experience"
            results = await self.vector_store.search_similar(query, n_results=3)
            return results
        
        results = asyncio.run(run_test())
        self.assertIsInstance(results, list)
    
    def test_get_statistics(self):
        async def run_test():
            stats = await self.vector_store.get_statistics()
            return stats
        
        stats = asyncio.run(run_test())
        self.assertIsInstance(stats, dict)
        if "total_analyses" in stats:
            self.assertIsInstance(stats["total_analyses"], int)

if __name__ == "__main__":
    unittest.main()