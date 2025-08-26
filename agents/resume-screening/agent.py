import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import config
from models.model_manager import ModelManager
from processors.document_processor import DocumentProcessor
from processors.text_analyzer import TextAnalyzer
from processors.vector_store import VectorStoreManager
from utils.observability import setup_langsmith, log_performance
from utils.validators import InputValidator
from utils.metrics import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeScreeningAgent:
    def __init__(self):
        self.config = config
        self.model_manager = ModelManager()
        self.document_processor = DocumentProcessor()
        self.text_analyzer = TextAnalyzer()
        self.vector_store_manager = VectorStoreManager()
        self.input_validator = InputValidator()
        self.metrics_collector = MetricsCollector()
        
        setup_langsmith()
        
        logger.info(f"Initialized {self.config.agent_name} v{self.config.version}")
        
    @traceable(name="process_resume", metadata={"agent": "resume-screening"})
    async def process_resume(
        self,
        file_path: str,
        job_requirements: str,
        model_selection: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        try:
            start_time = datetime.now()
            
            validation_result = self.input_validator.validate_file(file_path)
            if not validation_result["valid"]:
                return {"error": validation_result["message"], "status": "failed"}
            
            resume_text = await self.document_processor.extract_text(file_path)
            
            if not resume_text:
                return {"error": "Failed to extract text from resume", "status": "failed"}
            
            analyses = await self.analyze_resume_with_models(
                resume_text, job_requirements, model_selection
            )
            
            comprehensive_score = await self.calculate_comprehensive_score(analyses)
            
            result = {
                "file_path": file_path,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "resume_text": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
                "analyses": analyses,
                "comprehensive_score": comprehensive_score,
                "recommendation": self._generate_recommendation(comprehensive_score),
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            await self.vector_store_manager.store_analysis(result)
            
            self.metrics_collector.record_processing(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @traceable(name="analyze_resume_multi_model", metadata={"agent": "resume-screening"})
    async def analyze_resume_with_models(
        self,
        resume_text: str,
        job_requirements: str,
        selected_models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not selected_models:
            selected_models = [self.config.default_model]
        
        tasks = []
        for model_name in selected_models[:self.config.max_concurrent_models]:
            task = self._analyze_with_single_model(
                resume_text, job_requirements, model_name
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        analyses = {}
        for model_name, result in zip(selected_models, results):
            if isinstance(result, Exception):
                analyses[model_name] = {
                    "error": str(result),
                    "status": "failed"
                }
            else:
                analyses[model_name] = result
        
        return analyses
    
    async def _analyze_with_single_model(
        self,
        resume_text: str,
        job_requirements: str,
        model_name: str
    ) -> Dict[str, Any]:
        try:
            model = self.model_manager.get_model(model_name)
            
            analysis = await self.text_analyzer.analyze_resume(
                resume_text, job_requirements, model
            )
            
            return {
                "model": model_name,
                "analysis": analysis,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error with model {model_name}: {str(e)}")
            return {
                "model": model_name,
                "error": str(e),
                "status": "failed"
            }
    
    @traceable(name="calculate_comprehensive_score", metadata={"agent": "resume-screening"})
    async def calculate_comprehensive_score(
        self,
        analyses: Dict[str, Any]
    ) -> Dict[str, float]:
        scores = {
            "technical_skills": 0.0,
            "experience_relevance": 0.0,
            "cultural_fit": 0.0,
            "growth_potential": 0.0,
            "risk_assessment": 0.0,
            "overall": 0.0
        }
        
        valid_analyses = [
            a for a in analyses.values()
            if a.get("status") == "success" and "analysis" in a
        ]
        
        if not valid_analyses:
            return scores
        
        for analysis in valid_analyses:
            if "scores" in analysis.get("analysis", {}):
                model_scores = analysis["analysis"]["scores"]
                for key in scores.keys():
                    if key in model_scores:
                        scores[key] += float(model_scores[key])
        
        num_models = len(valid_analyses)
        for key in scores.keys():
            scores[key] = round(scores[key] / num_models, 2) if num_models > 0 else 0.0
        
        scores["overall"] = round(
            (scores["technical_skills"] * 0.3 +
             scores["experience_relevance"] * 0.3 +
             scores["cultural_fit"] * 0.15 +
             scores["growth_potential"] * 0.15 +
             scores["risk_assessment"] * 0.1),
            2
        )
        
        return scores
    
    def _generate_recommendation(self, scores: Dict[str, float]) -> str:
        overall_score = scores.get("overall", 0)
        
        if overall_score >= 80:
            return "STRONGLY RECOMMEND - Excellent match for the position"
        elif overall_score >= 70:
            return "RECOMMEND - Good match with strong potential"
        elif overall_score >= 60:
            return "CONSIDER - Moderate match, review specific areas"
        elif overall_score >= 50:
            return "MAYBE - Some relevant skills, significant gaps exist"
        else:
            return "NOT RECOMMENDED - Poor match for requirements"
    
    @traceable(name="batch_process", metadata={"agent": "resume-screening"})
    async def batch_process(
        self,
        file_paths: List[str],
        job_requirements: str,
        model_selection: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        tasks = []
        for file_path in file_paths:
            task = self.process_resume(file_path, job_requirements, model_selection)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for file_path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                processed_results.append({
                    "file_path": file_path,
                    "error": str(result),
                    "status": "failed"
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    @traceable(name="get_analysis_history", metadata={"agent": "resume-screening"})
    async def get_analysis_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        return await self.vector_store_manager.get_history(session_id, limit)
    
    @traceable(name="compare_models", metadata={"agent": "resume-screening"})
    async def compare_models(
        self,
        resume_text: str,
        job_requirements: str
    ) -> Dict[str, Any]:
        try:
            all_models = []
            for category, models in self.config.available_models.items():
                if models:  # Ensure models list is not empty
                    all_models.extend([m.name for m in models[:2] if m and m.name])
            
            if not all_models:
                logger.warning("No models available for comparison")
                return {
                    "models": {},
                    "best_model": None,
                    "consensus_score": 0.0,
                    "error": "No models available"
                }
            
            analyses = await self.analyze_resume_with_models(
                resume_text, job_requirements, all_models[:self.config.max_concurrent_models]
            )
            
            comparison = {
                "models": {},
                "best_model": None,
                "consensus_score": 0.0
            }
            
            best_score = 0.0
            valid_analyses = 0
            
            for model_name, analysis in analyses.items():
                if model_name and analysis and analysis.get("status") == "success":
                    try:
                        score = analysis.get("analysis", {}).get("scores", {}).get("overall", 0)
                        processing_time = analysis.get("processing_time", 0)
                        
                        # Ensure score is a valid number
                        if isinstance(score, (int, float)) and score >= 0:
                            comparison["models"][model_name] = {
                                "score": float(score),
                                "processing_time": float(processing_time) if processing_time else 0.0
                            }
                            
                            if score > best_score:
                                best_score = score
                                comparison["best_model"] = model_name
                            
                            valid_analyses += 1
                    except Exception as e:
                        logger.error(f"Error processing model {model_name}: {str(e)}")
                        continue
            
            if comparison["models"] and valid_analyses > 0:
                scores = [m["score"] for m in comparison["models"].values() if isinstance(m.get("score"), (int, float))]
                if scores:
                    comparison["consensus_score"] = round(sum(scores) / len(scores), 2)
            
            logger.info(f"Model comparison completed: {valid_analyses} valid analyses from {len(all_models)} models")
            return comparison
            
        except Exception as e:
            logger.error(f"Error in compare_models: {str(e)}")
            return {
                "models": {},
                "best_model": None,
                "consensus_score": 0.0,
                "error": str(e)
            }
    
    @traceable(name="export_results", metadata={"agent": "resume-screening"})
    async def export_results(
        self,
        results: Dict[str, Any],
        format: str = "json"
    ) -> str:
        if format == "json":
            return json.dumps(results, indent=2, default=str)
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(["Category", "Metric", "Value"])
            
            # Write comprehensive scores
            if "comprehensive_score" in results:
                scores = results["comprehensive_score"]
                writer.writerow(["Overall Analysis", "Overall Score", scores.get("overall", 0)])
                writer.writerow(["Overall Analysis", "Technical Skills", scores.get("technical_skills", 0)])
                writer.writerow(["Overall Analysis", "Experience Relevance", scores.get("experience_relevance", 0)])
                writer.writerow(["Overall Analysis", "Cultural Fit", scores.get("cultural_fit", 0)])
                writer.writerow(["Overall Analysis", "Growth Potential", scores.get("growth_potential", 0)])
                writer.writerow(["Overall Analysis", "Risk Assessment", scores.get("risk_assessment", 0)])
            
            # Write recommendation
            if "recommendation" in results:
                writer.writerow(["Overall Analysis", "Recommendation", results["recommendation"]])
            
            # Write processing metrics
            writer.writerow(["Metrics", "Processing Time (seconds)", results.get("processing_time", 0)])
            writer.writerow(["Metrics", "File Path", results.get("file_path", "")])
            writer.writerow(["Metrics", "Timestamp", results.get("timestamp", "")])
            
            # Write model-specific analyses if available
            if "analyses" in results:
                for model_name, analysis in results.get("analyses", {}).items():
                    if analysis.get("status") == "success" and "analysis" in analysis:
                        model_scores = analysis["analysis"].get("scores", {})
                        for score_type, score_value in model_scores.items():
                            writer.writerow([f"Model: {model_name}", score_type.replace("_", " ").title(), score_value])
            
            return output.getvalue()
        elif format == "pdf":
            # PDF export with comprehensive formatting
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            import io
            from datetime import datetime
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2E4057'),
                alignment=TA_CENTER
            )
            story.append(Paragraph("Resume Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Date and metadata
            metadata_style = ParagraphStyle(
                'Metadata',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#666666')
            )
            story.append(Paragraph(f"Generated: {results.get('timestamp', datetime.now().isoformat())}", metadata_style))
            story.append(Paragraph(f"File: {results.get('file_path', 'Unknown')}", metadata_style))
            story.append(Spacer(1, 20))
            
            # Overall Scores Section
            section_style = ParagraphStyle(
                'SectionHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#34495E'),
                spaceAfter=12
            )
            story.append(Paragraph("Overall Analysis Scores", section_style))
            
            if "comprehensive_score" in results:
                scores = results["comprehensive_score"]
                scores_data = [
                    ["Metric", "Score (0-100)"],
                    ["Overall Score", f"{scores.get('overall', 0):.1f}"],
                    ["Technical Skills", f"{scores.get('technical_skills', 0):.1f}"],
                    ["Experience Relevance", f"{scores.get('experience_relevance', 0):.1f}"],
                    ["Cultural Fit", f"{scores.get('cultural_fit', 0):.1f}"],
                    ["Growth Potential", f"{scores.get('growth_potential', 0):.1f}"],
                    ["Risk Assessment", f"{scores.get('risk_assessment', 0):.1f}"]
                ]
                
                scores_table = Table(scores_data, colWidths=[3*inch, 2*inch])
                scores_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(scores_table)
            
            story.append(Spacer(1, 20))
            
            # Recommendation
            if "recommendation" in results:
                story.append(Paragraph("Recommendation", section_style))
                rec_style = ParagraphStyle(
                    'Recommendation',
                    parent=styles['Normal'],
                    fontSize=12,
                    leading=14
                )
                story.append(Paragraph(results["recommendation"], rec_style))
                story.append(Spacer(1, 20))
            
            # Model Analyses (if available)
            if "analyses" in results:
                story.append(Paragraph("Model-Specific Analyses", section_style))
                for model_name, analysis in results.get("analyses", {}).items():
                    if analysis.get("status") == "success" and "analysis" in analysis:
                        model_style = ParagraphStyle(
                            'ModelName',
                            parent=styles['Heading3'],
                            fontSize=12,
                            textColor=colors.HexColor('#555555')
                        )
                        story.append(Paragraph(f"Model: {model_name}", model_style))
                        
                        if "scores" in analysis["analysis"]:
                            model_scores = analysis["analysis"]["scores"]
                            model_data = [[k.replace("_", " ").title(), f"{v:.1f}"] for k, v in model_scores.items()]
                            if model_data:
                                model_table = Table([["Metric", "Score"]] + model_data, colWidths=[3*inch, 2*inch])
                                model_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                                ]))
                                story.append(model_table)
                                story.append(Spacer(1, 10))
            
            # Processing Information
            story.append(Spacer(1, 20))
            story.append(Paragraph("Processing Information", section_style))
            proc_data = [
                ["Metric", "Value"],
                ["Processing Time", f"{results.get('processing_time', 0):.2f} seconds"],
                ["Status", results.get('status', 'Unknown')]
            ]
            proc_table = Table(proc_data, colWidths=[3*inch, 2*inch])
            proc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95A5A6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(proc_table)
            
            # Build PDF
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return pdf_data
        else:
            return str(results)
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "agent_id": self.config.agent_id,
            "agent_name": self.config.agent_name,
            "version": self.config.version,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_available": len([m for cat in self.config.available_models.values() for m in cat]),
            "langsmith_enabled": self.config.langchain_tracing
        }