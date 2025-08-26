import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langsmith import traceable
import hashlib

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = "resume_analyses"
        self._initialize_collection()
    
    def _initialize_collection(self):
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Resume screening analysis results"}
            )
            logger.info(f"Initialized collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            self.collection = None
    
    @traceable(name="store_resume_analysis", metadata={"component": "vector_store"})
    async def store_analysis(self, analysis_result: Dict[str, Any]) -> bool:
        try:
            if not self.collection:
                return False
            
            document_id = self._generate_document_id(analysis_result)
            
            document = json.dumps({
                "file_path": analysis_result.get("file_path", ""),
                "timestamp": analysis_result.get("timestamp", datetime.now().isoformat()),
                "comprehensive_score": analysis_result.get("comprehensive_score", {}),
                "recommendation": analysis_result.get("recommendation", ""),
                "processing_time": analysis_result.get("processing_time", 0)
            })
            
            metadata = {
                "file_path": analysis_result.get("file_path", ""),
                "timestamp": analysis_result.get("timestamp", datetime.now().isoformat()),
                "overall_score": float(analysis_result.get("comprehensive_score", {}).get("overall", 0)),
                "status": analysis_result.get("status", "unknown")
            }
            
            embedding_text = self._create_embedding_text(analysis_result)
            
            self.collection.upsert(
                ids=[document_id],
                documents=[embedding_text],
                metadatas=[metadata]
            )
            
            logger.info(f"Stored analysis with ID: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing analysis: {str(e)}")
            return False
    
    @traceable(name="search_similar_resumes", metadata={"component": "vector_store"})
    async def search_similar(
        self,
        query_text: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            if not self.collection:
                return []
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            similar_resumes = []
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    similar_resumes.append({
                        "id": doc_id,
                        "metadata": metadata,
                        "similarity_score": 1 - distance
                    })
            
            return similar_resumes
            
        except Exception as e:
            logger.error(f"Error searching similar resumes: {str(e)}")
            return []
    
    @traceable(name="get_analysis_history", metadata={"component": "vector_store"})
    async def get_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            if not self.collection:
                return []
            
            where_clause = {}
            if session_id:
                where_clause["session_id"] = session_id
            
            results = self.collection.get(
                limit=limit,
                where=where_clause if where_clause else None
            )
            
            history = []
            if results and results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    document = results['documents'][i] if results['documents'] else ""
                    
                    history.append({
                        "id": doc_id,
                        "metadata": metadata,
                        "document": document
                    })
            
            return sorted(history, key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting history: {str(e)}")
            return []
    
    @traceable(name="delete_analysis", metadata={"component": "vector_store"})
    async def delete_analysis(self, document_id: str) -> bool:
        try:
            if not self.collection:
                return False
            
            self.collection.delete(ids=[document_id])
            logger.info(f"Deleted analysis with ID: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting analysis: {str(e)}")
            return False
    
    def _generate_document_id(self, analysis_result: Dict[str, Any]) -> str:
        content = f"{analysis_result.get('file_path', '')}_{analysis_result.get('timestamp', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_embedding_text(self, analysis_result: Dict[str, Any]) -> str:
        parts = []
        
        if "resume_text" in analysis_result:
            parts.append(analysis_result["resume_text"][:1000])
        
        if "comprehensive_score" in analysis_result:
            scores = analysis_result["comprehensive_score"]
            parts.append(f"Scores: {json.dumps(scores)}")
        
        if "analyses" in analysis_result:
            for model_name, analysis in analysis_result["analyses"].items():
                if isinstance(analysis, dict) and "analysis" in analysis:
                    insights = analysis["analysis"].get("insights", [])
                    if insights:
                        parts.append(f"{model_name}: {' '.join(insights[:3])}")
        
        return " ".join(parts)
    
    @traceable(name="get_statistics", metadata={"component": "vector_store"})
    async def get_statistics(self) -> Dict[str, Any]:
        try:
            if not self.collection:
                return {}
            
            count = self.collection.count()
            
            all_results = self.collection.get(limit=count)
            
            if all_results and all_results['metadatas']:
                scores = [m.get('overall_score', 0) for m in all_results['metadatas']]
                avg_score = sum(scores) / len(scores) if scores else 0
                
                return {
                    "total_analyses": count,
                    "average_score": round(avg_score, 2),
                    "highest_score": max(scores) if scores else 0,
                    "lowest_score": min(scores) if scores else 0
                }
            
            return {"total_analyses": count}
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}