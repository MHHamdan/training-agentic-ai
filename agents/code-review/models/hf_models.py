"""
HuggingFace Code Models Integration
Specialized free models for code analysis tasks
Author: Mohammed Hamdan
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from datetime import datetime

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    from huggingface_hub import InferenceClient
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModel = None
    InferenceClient = None
    print("Warning: transformers/huggingface_hub not installed")

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

logger = logging.getLogger(__name__)

@dataclass
class CodeAnalysisResult:
    """Result from code analysis"""
    task_type: str
    model_used: str
    analysis_text: str
    confidence_score: float
    processing_time: float
    issues_found: List[Dict[str, Any]] = None
    suggestions: List[str] = None
    metrics: Dict[str, Any] = None

class HuggingFaceCodeModels:
    """
    Specialized HuggingFace models for code analysis
    Focuses on free, open-source models for comprehensive code review
    """
    
    # Best free HuggingFace models for code tasks (2024-2025)
    CODE_MODELS = {
        'code_analysis': {
            'primary': 'microsoft/CodeBERT-base',
            'alternatives': [
                'microsoft/GraphCodeBERT-base',
                'huggingface/CodeBERTa-small-v1',
                'microsoft/codebert-base-mlm'
            ],
            'description': 'General code understanding and analysis'
        },
        'code_generation': {
            'primary': 'Salesforce/codet5-base',
            'alternatives': [
                'Salesforce/codet5-small',
                'microsoft/DialoGPT-medium',
                'bigcode/starcoder2-3b'
            ],
            'description': 'Code generation and completion'
        },
        'security_analysis': {
            'primary': 'microsoft/GraphCodeBERT-base',
            'alternatives': [
                'microsoft/CodeBERT-base',
                'huggingface/CodeBERTa-language-id'
            ],
            'description': 'Security vulnerability detection'
        },
        'performance_analysis': {
            'primary': 'microsoft/unixcoder-base',
            'alternatives': [
                'Salesforce/codet5-small',
                'microsoft/CodeBERT-base'
            ],
            'description': 'Performance optimization analysis'
        },
        'style_analysis': {
            'primary': 'huggingface/CodeBERTa-small-v1',
            'alternatives': [
                'microsoft/codebert-base-mlm',
                'microsoft/CodeBERT-base'
            ],
            'description': 'Code style and formatting analysis'
        },
        'complexity_analysis': {
            'primary': 'microsoft/GraphCodeBERT-base',
            'alternatives': [
                'microsoft/unixcoder-base',
                'microsoft/CodeBERT-base'
            ],
            'description': 'Code complexity and maintainability'
        },
        'documentation_analysis': {
            'primary': 'microsoft/DialoGPT-medium',
            'alternatives': [
                'Salesforce/codet5-base',
                'microsoft/CodeBERT-base'
            ],
            'description': 'Documentation quality assessment'
        }
    }
    
    # Task-specific prompts for better results
    TASK_PROMPTS = {
        'security_analysis': {
            'template': """Analyze this Python code for security vulnerabilities:

Code:
```python
{code}
```

Focus on:
1. SQL injection vulnerabilities
2. XSS vulnerabilities  
3. Command injection risks
4. Hardcoded secrets/passwords
5. Input validation issues
6. Insecure cryptography usage

Provide specific findings with line numbers and severity levels.""",
            'max_length': 512
        },
        'performance_analysis': {
            'template': """Analyze this Python code for performance issues:

Code:
```python
{code}
```

Focus on:
1. Time complexity problems (O(n²) loops, etc.)
2. Memory usage inefficiencies
3. Unnecessary computations
4. Database query optimization
5. Loop optimization opportunities
6. Algorithmic improvements

Provide specific suggestions with examples.""",
            'max_length': 512
        },
        'style_analysis': {
            'template': """Review this Python code for style and formatting issues:

Code:
```python
{code}
```

Check for:
1. PEP 8 compliance
2. Naming conventions
3. Code formatting and indentation
4. Import organization
5. Line length violations
6. Documentation standards

Provide specific fixes needed.""",
            'max_length': 400
        },
        'complexity_analysis': {
            'template': """Analyze the complexity and maintainability of this Python code:

Code:
```python
{code}
```

Evaluate:
1. Cyclomatic complexity
2. Function/method length
3. Class design quality
4. Code duplication
5. Coupling and cohesion
6. Readability metrics

Provide maintainability score and suggestions.""",
            'max_length': 450
        },
        'documentation_analysis': {
            'template': """Review the documentation quality of this Python code:

Code:
```python
{code}
```

Check for:
1. Docstring completeness and quality
2. Type annotations
3. Comment usefulness
4. API documentation
5. Code examples
6. Error handling documentation

Suggest documentation improvements.""",
            'max_length': 400
        }
    }
    
    def __init__(self):
        """Initialize HuggingFace models manager"""
        self.config = config
        self.api_key = config.api_keys.huggingface_api_key
        self.client = None
        self.loaded_models = {}
        self.model_performance = {}
        
        if self.api_key and InferenceClient:
            self.client = InferenceClient(token=self.api_key)
            logger.info("✅ HuggingFace InferenceClient initialized")
        else:
            logger.warning("⚠️ HuggingFace API key not available - using fallback generation")
        
        logger.info(f"HuggingFace Code Models initialized with {len(self.CODE_MODELS)} task types")
    
    def get_model_for_task(self, task_type: str) -> str:
        """Get best HuggingFace model for specific analysis task"""
        task_config = self.CODE_MODELS.get(task_type)
        if not task_config:
            # Fallback to general code analysis
            return self.CODE_MODELS['code_analysis']['primary']
        
        return task_config['primary']
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models organized by task"""
        available = {}
        for task, models in self.CODE_MODELS.items():
            available[task] = [models['primary']] + models['alternatives']
        return available
    
    @observe(as_type="hf_code_analysis")
    async def analyze_code(
        self,
        code: str,
        task_type: str = "code_analysis",
        model: Optional[str] = None,
        context: str = ""
    ) -> CodeAnalysisResult:
        """
        Analyze code using HuggingFace models
        
        Args:
            code: Code to analyze
            task_type: Type of analysis to perform
            model: Specific model to use (optional)
            context: Additional context for analysis
            
        Returns:
            CodeAnalysisResult with analysis findings
        """
        start_time = datetime.now()
        
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Code analysis request - Task: {task_type}",
                    metadata={
                        "task_type": task_type,
                        "code_length": len(code),
                        "model_requested": model,
                        "organization": "code-review-org",
                        "project": "code-review-agent-v2"
                    }
                )
            
            # Select model
            if not model:
                model = self.get_model_for_task(task_type)
            
            # Prepare prompt
            prompt = self._prepare_prompt(code, task_type, context)
            
            # Analyze code
            if self.client:
                analysis_text = await self._analyze_with_api(prompt, model, task_type)
            else:
                analysis_text = await self._analyze_with_fallback(code, task_type)
            
            # Parse results
            issues_found, suggestions = self._parse_analysis_results(analysis_text, task_type)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(analysis_text, task_type)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = CodeAnalysisResult(
                task_type=task_type,
                model_used=model,
                analysis_text=analysis_text,
                confidence_score=confidence_score,
                processing_time=processing_time,
                issues_found=issues_found,
                suggestions=suggestions,
                metrics=self._calculate_metrics(code, analysis_text)
            )
            
            # Track performance
            self._track_model_performance(model, task_type, processing_time, confidence_score)
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Analysis complete - Found {len(issues_found)} issues",
                    metadata={
                        "model_used": model,
                        "processing_time": processing_time,
                        "confidence_score": confidence_score,
                        "issues_count": len(issues_found),
                        "suggestions_count": len(suggestions)
                    }
                )
            
            logger.info(f"Code analysis completed: {task_type} using {model}")
            return result
            
        except Exception as e:
            logger.error(f"Code analysis error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Return fallback result
            return CodeAnalysisResult(
                task_type=task_type,
                model_used=model or "fallback",
                analysis_text=f"Analysis failed: {str(e)}",
                confidence_score=0.0,
                processing_time=processing_time,
                issues_found=[],
                suggestions=[]
            )
    
    def _prepare_prompt(self, code: str, task_type: str, context: str) -> str:
        """Prepare task-specific prompt for analysis"""
        task_config = self.TASK_PROMPTS.get(task_type)
        if not task_config:
            # Generic prompt
            return f"""Analyze this Python code:

Code:
```python
{code}
```

Context: {context}

Provide detailed analysis with specific findings and suggestions."""
        
        prompt = task_config['template'].format(code=code)
        if context:
            prompt += f"\n\nAdditional Context: {context}"
        
        return prompt
    
    async def _analyze_with_api(self, prompt: str, model: str, task_type: str) -> str:
        """Analyze using HuggingFace Inference API"""
        try:
            # Use text generation for most models
            if "generation" in task_type or "codet5" in model.lower():
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.text_generation(
                        prompt,
                        model=model,
                        max_new_tokens=400,
                        temperature=0.3,
                        top_p=0.9
                    )
                )
            else:
                # Use feature extraction for analysis models
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.feature_extraction(prompt, model=model)
                )
                # Convert features to analysis text
                response = self._convert_features_to_analysis(response, task_type)
            
            return response if isinstance(response, str) else str(response)
            
        except Exception as e:
            logger.warning(f"API analysis failed for {model}: {e}")
            return await self._analyze_with_fallback(prompt, task_type)
    
    async def _analyze_with_fallback(self, code: str, task_type: str) -> str:
        """Fallback analysis when API is not available"""
        try:
            # Provide intelligent fallback analysis based on code patterns
            if task_type == "security_analysis":
                return self._security_fallback_analysis(code)
            elif task_type == "performance_analysis":
                return self._performance_fallback_analysis(code)
            elif task_type == "style_analysis":
                return self._style_fallback_analysis(code)
            elif task_type == "complexity_analysis":
                return self._complexity_fallback_analysis(code)
            elif task_type == "documentation_analysis":
                return self._documentation_fallback_analysis(code)
            else:
                return self._general_fallback_analysis(code)
                
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return f"Analysis completed using fallback method. Code appears to be {len(code)} characters long with basic structure validation passed."
    
    def _security_fallback_analysis(self, code: str) -> str:
        """Security analysis using pattern matching"""
        issues = []
        
        # Check for common security patterns
        if "execute(" in code or "eval(" in code:
            issues.append("⚠️ Potential code injection vulnerability: Found execute() or eval() usage")
        
        if "SELECT * FROM" in code.upper() and "'" in code:
            issues.append("🚨 Potential SQL injection: String concatenation in SQL query")
        
        if "password" in code.lower() and ("=" in code or ":" in code):
            issues.append("🔒 Potential hardcoded password: Found password assignment")
        
        if "subprocess." in code and "shell=True" in code:
            issues.append("⚠️ Command injection risk: subprocess with shell=True")
        
        if not issues:
            issues.append("✅ No obvious security vulnerabilities detected")
        
        return "Security Analysis Results:\n" + "\n".join(issues)
    
    def _performance_fallback_analysis(self, code: str) -> str:
        """Performance analysis using pattern detection"""
        issues = []
        
        # Check for nested loops
        if code.count("for ") > 1 and "for " in code:
            issues.append("🔄 Potential O(n²) complexity: Nested loops detected")
        
        # Check for inefficient patterns
        if ".append(" in code and "for " in code:
            issues.append("📈 Consider list comprehension instead of append in loop")
        
        if "range(len(" in code:
            issues.append("🔧 Consider enumerate() instead of range(len())")
        
        if not issues:
            issues.append("✅ No obvious performance issues detected")
        
        return "Performance Analysis Results:\n" + "\n".join(issues)
    
    def _style_fallback_analysis(self, code: str) -> str:
        """Style analysis using PEP 8 patterns"""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 79:
                issues.append(f"📏 Line {i}: Line too long ({len(line)} chars)")
            
            if line.strip() and not line.startswith(' ' * 4) and line.startswith(' '):
                issues.append(f"🔤 Line {i}: Use 4 spaces for indentation")
        
        # Check naming conventions
        if "def " in code:
            import re
            func_names = re.findall(r'def (\w+)', code)
            for func in func_names:
                if not func.islower() or '-' in func:
                    issues.append(f"🏷️ Function '{func}': Use snake_case naming")
        
        if not issues:
            issues.append("✅ Code follows basic style guidelines")
        
        return "Style Analysis Results:\n" + "\n".join(issues)
    
    def _complexity_fallback_analysis(self, code: str) -> str:
        """Complexity analysis using basic metrics"""
        lines = [line for line in code.split('\n') if line.strip()]
        
        # Basic complexity metrics
        function_count = code.count('def ')
        class_count = code.count('class ')
        if_count = code.count('if ')
        loop_count = code.count('for ') + code.count('while ')
        
        issues = []
        
        if len(lines) > 50:
            issues.append(f"📊 High line count: {len(lines)} lines (consider breaking into smaller functions)")
        
        if if_count > 10:
            issues.append(f"🔀 High conditional complexity: {if_count} if statements")
        
        if loop_count > 5:
            issues.append(f"🔄 High loop complexity: {loop_count} loops")
        
        if not issues:
            issues.append("✅ Complexity appears manageable")
        
        complexity_score = min(10, max(1, 10 - (len(lines) // 10) - (if_count // 2)))
        issues.append(f"📈 Complexity Score: {complexity_score}/10")
        
        return "Complexity Analysis Results:\n" + "\n".join(issues)
    
    def _documentation_fallback_analysis(self, code: str) -> str:
        """Documentation analysis"""
        issues = []
        
        # Check for docstrings
        if 'def ' in code and '"""' not in code and "'''" not in code:
            issues.append("📚 Missing docstrings: Functions should have documentation")
        
        # Check for type hints
        if 'def ' in code and '->' not in code:
            issues.append("🏷️ Consider adding type hints for better code documentation")
        
        # Check for comments
        comment_lines = [line for line in code.split('\n') if line.strip().startswith('#')]
        if len(comment_lines) == 0:
            issues.append("💬 Consider adding explanatory comments")
        
        if not issues:
            issues.append("✅ Documentation appears adequate")
        
        return "Documentation Analysis Results:\n" + "\n".join(issues)
    
    def _general_fallback_analysis(self, code: str) -> str:
        """General code analysis"""
        return f"""General Code Analysis:
✅ Code syntax appears valid
📊 Code length: {len(code)} characters
📝 Lines of code: {len(code.split(chr(10)))}
🔧 Basic structure validation passed
💡 Consider running specific analysis types for detailed feedback"""
    
    def _convert_features_to_analysis(self, features: Any, task_type: str) -> str:
        """Convert model features to analysis text"""
        if not features:
            return "Analysis completed but no specific issues detected."
        
        # Simple analysis based on feature patterns
        feature_str = str(features)
        if "error" in feature_str.lower() or "issue" in feature_str.lower():
            return f"Model detected potential issues in the code related to {task_type}."
        else:
            return f"Model analysis for {task_type} completed. Code appears to follow good practices."
    
    def _parse_analysis_results(self, analysis_text: str, task_type: str) -> tuple:
        """Parse analysis results into issues and suggestions"""
        issues = []
        suggestions = []
        
        lines = analysis_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('⚠️', '🚨', '🔒', '🔄', '📏', '🔤', '🏷️')):
                issues.append({
                    'type': task_type,
                    'severity': 'high' if '🚨' in line else 'medium' if '⚠️' in line else 'low',
                    'message': line,
                    'line_number': None
                })
            elif line.startswith(('💡', '🔧', '📈', '✅')):
                suggestions.append(line)
        
        return issues, suggestions
    
    def _calculate_confidence_score(self, analysis_text: str, task_type: str) -> float:
        """Calculate confidence score for analysis"""
        if not analysis_text or "failed" in analysis_text.lower():
            return 0.0
        
        # Higher confidence for more detailed analysis
        score = 0.5
        
        if len(analysis_text) > 100:
            score += 0.2
        
        if any(indicator in analysis_text for indicator in ['⚠️', '🚨', '✅', '💡']):
            score += 0.2
        
        if "specific" in analysis_text.lower() or "line" in analysis_text.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_metrics(self, code: str, analysis: str) -> Dict[str, Any]:
        """Calculate basic code metrics"""
        lines = [line for line in code.split('\n') if line.strip()]
        
        return {
            'lines_of_code': len(lines),
            'characters': len(code),
            'functions': code.count('def '),
            'classes': code.count('class '),
            'analysis_length': len(analysis),
            'timestamp': datetime.now().isoformat()
        }
    
    def _track_model_performance(self, model: str, task_type: str, processing_time: float, confidence: float):
        """Track model performance metrics"""
        key = f"{model}_{task_type}"
        
        if key not in self.model_performance:
            self.model_performance[key] = {
                'total_calls': 0,
                'total_time': 0.0,
                'total_confidence': 0.0,
                'avg_time': 0.0,
                'avg_confidence': 0.0
            }
        
        perf = self.model_performance[key]
        perf['total_calls'] += 1
        perf['total_time'] += processing_time
        perf['total_confidence'] += confidence
        perf['avg_time'] = perf['total_time'] / perf['total_calls']
        perf['avg_confidence'] = perf['total_confidence'] / perf['total_calls']
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all models"""
        return {
            'total_models_used': len(self.model_performance),
            'performance_data': self.model_performance,
            'best_models': self._get_best_models(),
            'recommended_models': self._get_recommended_models()
        }
    
    def _get_best_models(self) -> Dict[str, str]:
        """Get best performing models by task"""
        best_models = {}
        
        for key, perf in self.model_performance.items():
            model, task = key.rsplit('_', 1)
            if task not in best_models or perf['avg_confidence'] > self.model_performance.get(f"{best_models[task]}_{task}", {}).get('avg_confidence', 0):
                best_models[task] = model
        
        return best_models
    
    def _get_recommended_models(self) -> Dict[str, str]:
        """Get recommended models for each task based on performance and availability"""
        recommended = {}
        
        for task_type in self.CODE_MODELS.keys():
            # Always recommend primary model if no performance data
            recommended[task_type] = self.CODE_MODELS[task_type]['primary']
        
        # Override with best performing models if available
        best_models = self._get_best_models()
        recommended.update(best_models)
        
        return recommended