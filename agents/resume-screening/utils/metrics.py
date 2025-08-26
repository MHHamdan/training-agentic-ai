import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.processing_times = []
        self.model_performance = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.score_distributions = []
        self.start_time = datetime.now()
    
    def record_processing(self, result: Dict[str, Any]):
        try:
            if "processing_time" in result:
                self.processing_times.append(result["processing_time"])
            
            if result.get("status") == "success":
                self.success_counts["total"] += 1
                
                if "comprehensive_score" in result:
                    self.score_distributions.append(result["comprehensive_score"])
                
                if "analyses" in result:
                    for model_name, analysis in result["analyses"].items():
                        if analysis.get("status") == "success":
                            self.success_counts[model_name] += 1
                            if "processing_time" in analysis:
                                self.model_performance[model_name].append({
                                    "time": analysis["processing_time"],
                                    "success": True
                                })
                        else:
                            self.error_counts[model_name] += 1
                            self.model_performance[model_name].append({
                                "success": False
                            })
            else:
                self.error_counts["total"] += 1
                
        except Exception as e:
            logger.error(f"Error recording metrics: {str(e)}")
    
    def get_summary(self) -> Dict[str, Any]:
        total_processed = self.success_counts["total"] + self.error_counts["total"]
        
        summary = {
            "uptime": str(datetime.now() - self.start_time),
            "total_processed": total_processed,
            "success_rate": (self.success_counts["total"] / total_processed * 100) if total_processed > 0 else 0,
            "error_rate": (self.error_counts["total"] / total_processed * 100) if total_processed > 0 else 0,
            "average_processing_time": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            "min_processing_time": min(self.processing_times) if self.processing_times else 0,
            "max_processing_time": max(self.processing_times) if self.processing_times else 0
        }
        
        if self.score_distributions:
            all_scores = []
            for score_dict in self.score_distributions:
                if "overall" in score_dict:
                    all_scores.append(score_dict["overall"])
            
            if all_scores:
                summary["average_score"] = sum(all_scores) / len(all_scores)
                summary["min_score"] = min(all_scores)
                summary["max_score"] = max(all_scores)
        
        model_summaries = {}
        for model_name, performances in self.model_performance.items():
            successful = [p for p in performances if p.get("success")]
            model_summaries[model_name] = {
                "total_calls": len(performances),
                "success_rate": (len(successful) / len(performances) * 100) if performances else 0,
                "average_time": sum(p.get("time", 0) for p in successful) / len(successful) if successful else 0
            }
        
        summary["model_performance"] = model_summaries
        
        return summary
    
    def get_performance_trends(self, window_minutes: int = 60) -> Dict[str, Any]:
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_times = [t for t in self.processing_times[-100:]]
        
        if not recent_times:
            return {"message": "No recent data available"}
        
        trend = {
            "window_minutes": window_minutes,
            "recent_average": sum(recent_times) / len(recent_times),
            "recent_min": min(recent_times),
            "recent_max": max(recent_times),
            "sample_count": len(recent_times)
        }
        
        if len(self.processing_times) > len(recent_times):
            older_times = self.processing_times[:-len(recent_times)]
            trend["previous_average"] = sum(older_times) / len(older_times)
            
            if trend["previous_average"] > 0:
                improvement = ((trend["previous_average"] - trend["recent_average"]) / 
                             trend["previous_average"] * 100)
                trend["performance_change"] = f"{improvement:+.1f}%"
        
        return trend
    
    def export_metrics(self, format: str = "json") -> str:
        metrics_data = {
            "summary": self.get_summary(),
            "trends": self.get_performance_trends(),
            "timestamp": datetime.now().isoformat()
        }
        
        if format == "json":
            return json.dumps(metrics_data, indent=2, default=str)
        elif format == "text":
            lines = []
            lines.append("=== Resume Screening Agent Metrics ===")
            lines.append(f"Generated at: {metrics_data['timestamp']}")
            lines.append("")
            
            summary = metrics_data["summary"]
            lines.append("SUMMARY:")
            for key, value in summary.items():
                if key != "model_performance":
                    lines.append(f"  {key}: {value}")
            
            lines.append("")
            lines.append("MODEL PERFORMANCE:")
            for model, perf in summary.get("model_performance", {}).items():
                lines.append(f"  {model}:")
                for metric, value in perf.items():
                    lines.append(f"    {metric}: {value}")
            
            return "\n".join(lines)
        else:
            return str(metrics_data)
    
    def reset(self):
        self.processing_times = []
        self.model_performance = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.score_distributions = []
        self.start_time = datetime.now()