import unittest
import asyncio
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ResumeScreeningAgent
from config import config

class TestResumeScreeningAgent(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.agent = ResumeScreeningAgent()
        cls.test_resume_text = """
        John Doe
        Senior Software Engineer
        
        Experience:
        - 8 years of Python development
        - Expert in Django, Flask, FastAPI
        - AWS, Docker, Kubernetes experience
        - Led team of 5 developers
        
        Education:
        - Bachelor's in Computer Science
        - Master's in Software Engineering
        
        Skills:
        Python, JavaScript, React, PostgreSQL, MongoDB, Git, CI/CD
        """
        
        cls.test_job_requirements = """
        Looking for a Senior Software Engineer with:
        - 5+ years of Python experience
        - Strong knowledge of web frameworks (Django/Flask)
        - Cloud platform experience (AWS preferred)
        - Leadership experience
        - Bachelor's degree in Computer Science or related field
        """
    
    def test_agent_initialization(self):
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.config.agent_id, "agent-12")
        self.assertEqual(self.agent.config.agent_name, "Resume Screening Agent")
    
    def test_health_check(self):
        health = self.agent.health_check()
        self.assertIn("status", health)
        self.assertEqual(health["status"], "healthy")
        self.assertIn("agent_id", health)
        self.assertIn("version", health)
    
    def test_analyze_resume_with_models(self):
        async def run_test():
            analyses = await self.agent.analyze_resume_with_models(
                self.test_resume_text,
                self.test_job_requirements,
                [self.agent.config.default_model]
            )
            return analyses
        
        analyses = asyncio.run(run_test())
        self.assertIsInstance(analyses, dict)
        self.assertTrue(len(analyses) > 0)
    
    def test_calculate_comprehensive_score(self):
        async def run_test():
            mock_analyses = {
                "model1": {
                    "status": "success",
                    "analysis": {
                        "scores": {
                            "technical_skills": 85,
                            "experience_relevance": 90,
                            "cultural_fit": 75,
                            "growth_potential": 80,
                            "risk_assessment": 85,
                            "overall": 83
                        }
                    }
                }
            }
            
            scores = await self.agent.calculate_comprehensive_score(mock_analyses)
            return scores
        
        scores = asyncio.run(run_test())
        self.assertIn("technical_skills", scores)
        self.assertIn("overall", scores)
        self.assertIsInstance(scores["overall"], float)
        self.assertTrue(0 <= scores["overall"] <= 100)
    
    def test_generate_recommendation(self):
        test_cases = [
            ({"overall": 85}, "STRONGLY RECOMMEND"),
            ({"overall": 75}, "RECOMMEND"),
            ({"overall": 65}, "CONSIDER"),
            ({"overall": 55}, "MAYBE"),
            ({"overall": 45}, "NOT RECOMMENDED")
        ]
        
        for scores, expected_keyword in test_cases:
            recommendation = self.agent._generate_recommendation(scores)
            self.assertIn(expected_keyword, recommendation)
    
    def test_export_results(self):
        async def run_test():
            test_results = {
                "file_path": "test.pdf",
                "comprehensive_score": {"overall": 75},
                "status": "success"
            }
            
            json_export = await self.agent.export_results(test_results, "json")
            csv_export = await self.agent.export_results(test_results, "csv")
            
            return json_export, csv_export
        
        json_export, csv_export = asyncio.run(run_test())
        
        self.assertIsInstance(json_export, str)
        self.assertIn("file_path", json_export)
        
        self.assertIsInstance(csv_export, str)
        self.assertIn("Key", csv_export)
    
    def test_model_manager_integration(self):
        available_models = self.agent.model_manager.get_available_models()
        self.assertIsInstance(available_models, dict)
        self.assertTrue(len(available_models) > 0)
        
        for category, models in available_models.items():
            self.assertIsInstance(models, list)
            self.assertTrue(len(models) > 0)

if __name__ == "__main__":
    unittest.main()