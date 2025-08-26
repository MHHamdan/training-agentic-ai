"""
Performance Analysis for Code Review
Detects performance issues and optimization opportunities
Author: Mohammed Hamdan
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None

logger = logging.getLogger(__name__)

class PerformanceIssueType(Enum):
    """Types of performance issues"""
    TIME_COMPLEXITY = "time_complexity"
    MEMORY_USAGE = "memory_usage"
    LOOP_INEFFICIENCY = "loop_inefficiency"
    DATABASE_QUERY = "database_query"
    IO_OPERATIONS = "io_operations"
    ALGORITHMIC = "algorithmic"
    DATA_STRUCTURE = "data_structure"
    CACHING = "caching"

class ImpactLevel(Enum):
    """Performance impact levels"""
    CRITICAL = "critical"  # Major performance bottleneck
    HIGH = "high"         # Significant impact
    MEDIUM = "medium"     # Moderate impact
    LOW = "low"          # Minor optimization
    INFO = "info"        # Informational

@dataclass
class PerformanceIssue:
    """Performance issue finding"""
    issue_type: PerformanceIssueType
    impact: ImpactLevel
    title: str
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    complexity_before: Optional[str] = None
    complexity_after: Optional[str] = None
    confidence: float = 1.0

@dataclass
class PerformanceAnalysisResult:
    """Complete performance analysis result"""
    issues: List[PerformanceIssue]
    performance_score: float
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    analysis_time: float
    optimizations: List[str]
    complexity_metrics: Dict[str, Any]

class PerformanceAnalyzer:
    """
    Performance analyzer for code optimization
    Detects inefficiencies and suggests improvements
    """
    
    # Performance patterns
    PERFORMANCE_PATTERNS = {
        PerformanceIssueType.TIME_COMPLEXITY: [
            {
                'pattern': r'for\s+\w+\s+in.*:\s*\n.*for\s+\w+\s+in',
                'description': 'Nested loops may indicate O(n²) complexity',
                'impact': ImpactLevel.HIGH,
                'recommendation': 'Consider using hash maps, sets, or more efficient algorithms'
            },
            {
                'pattern': r'while.*:\s*\n.*while.*:',
                'description': 'Nested while loops can cause performance issues',
                'impact': ImpactLevel.HIGH,
                'recommendation': 'Review loop logic and consider optimization'
            }
        ],
        
        PerformanceIssueType.LOOP_INEFFICIENCY: [
            {
                'pattern': r'for\s+\w+\s+in\s+range\(len\(',
                'description': 'Using range(len()) instead of enumerate()',
                'impact': ImpactLevel.LOW,
                'recommendation': 'Use enumerate() for better readability and slight performance gain'
            },
            {
                'pattern': r'\.append\(.*\)\s*\n.*for.*in',
                'description': 'List append in loop - consider list comprehension',
                'impact': ImpactLevel.MEDIUM,
                'recommendation': 'Use list comprehension for better performance'
            },
            {
                'pattern': r'len\(.*\)\s*==\s*0',
                'description': 'Inefficient empty check',
                'impact': ImpactLevel.LOW,
                'recommendation': 'Use "not list_name" instead of "len(list_name) == 0"'
            }
        ],
        
        PerformanceIssueType.DATABASE_QUERY: [
            {
                'pattern': r'for.*in.*:\s*\n.*\.execute\(',
                'description': 'Database query in loop (N+1 problem)',
                'impact': ImpactLevel.CRITICAL,
                'recommendation': 'Use batch queries or join operations'
            },
            {
                'pattern': r'SELECT\s+\*\s+FROM',
                'description': 'SELECT * queries can be inefficient',
                'impact': ImpactLevel.MEDIUM,
                'recommendation': 'Select only required columns'
            }
        ],
        
        PerformanceIssueType.MEMORY_USAGE: [
            {
                'pattern': r'\.read\(\)\s*$',
                'description': 'Reading entire file into memory',
                'impact': ImpactLevel.HIGH,
                'recommendation': 'Consider reading file in chunks for large files'
            },
            {
                'pattern': r'\[\]\s*\+.*for.*in',
                'description': 'List concatenation in loop',
                'impact': ImpactLevel.MEDIUM,
                'recommendation': 'Use extend() or list comprehension'
            }
        ],
        
        PerformanceIssueType.DATA_STRUCTURE: [
            {
                'pattern': r'if\s+.*\s+in\s+\[.*\]:',
                'description': 'Membership test on list instead of set',
                'impact': ImpactLevel.MEDIUM,
                'recommendation': 'Use set for O(1) membership testing'
            },
            {
                'pattern': r'\.count\(.*\)\s*>\s*0',
                'description': 'Using count() for existence check',
                'impact': ImpactLevel.LOW,
                'recommendation': 'Use "in" operator for existence checks'
            }
        ]
    }
    
    # Performance recommendations
    OPTIMIZATION_SUGGESTIONS = {
        PerformanceIssueType.TIME_COMPLEXITY: [
            "Consider using hash maps (dict) for O(1) lookups",
            "Implement caching for repeated calculations",
            "Use more efficient sorting algorithms when needed",
            "Consider parallel processing for independent operations"
        ],
        PerformanceIssueType.LOOP_INEFFICIENCY: [
            "Use list comprehensions instead of append loops",
            "Consider using enumerate() instead of range(len())",
            "Break early from loops when possible",
            "Cache loop invariants outside the loop"
        ],
        PerformanceIssueType.DATABASE_QUERY: [
            "Use batch operations instead of individual queries",
            "Implement proper indexing strategies",
            "Use joins instead of multiple queries",
            "Consider query result caching"
        ],
        PerformanceIssueType.MEMORY_USAGE: [
            "Process large files in chunks",
            "Use generators for memory-efficient iteration",
            "Implement object pooling for frequent allocations",
            "Consider using slots for classes with many instances"
        ]
    }
    
    def __init__(self):
        """Initialize performance analyzer"""
        self.analysis_stats = {
            'total_analyses': 0,
            'issues_found': 0,
            'avg_performance_score': 0.0
        }
        logger.info("Performance Analyzer initialized")
    
    @observe(as_type="performance_analysis")
    async def analyze_performance(
        self,
        code: str,
        context: str = "",
        include_ai_analysis: bool = True
    ) -> PerformanceAnalysisResult:
        """
        Comprehensive performance analysis
        
        Args:
            code: Source code to analyze
            context: Additional context
            include_ai_analysis: Whether to include AI analysis
            
        Returns:
            PerformanceAnalysisResult with findings
        """
        start_time = datetime.now()
        
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Performance analysis - {len(code)} characters",
                    metadata={
                        "analysis_type": "performance",
                        "code_length": len(code),
                        "ai_analysis_enabled": include_ai_analysis,
                        "organization": "code-review-org",
                        "project": "code-review-agent-v2"
                    }
                )
            
            issues = []
            
            # 1. Pattern-based analysis
            pattern_issues = await self._analyze_patterns(code)
            issues.extend(pattern_issues)
            
            # 2. AST-based analysis
            ast_issues = await self._analyze_ast(code)
            issues.extend(ast_issues)
            
            # 3. Complexity analysis
            complexity_issues = await self._analyze_complexity(code)
            issues.extend(complexity_issues)
            
            # 4. AI-powered analysis
            if include_ai_analysis:
                ai_issues = await self._analyze_with_ai(code, context)
                issues.extend(ai_issues)
            
            # 5. Compile results
            result = self._compile_results(issues, start_time, code)
            
            # 6. Update statistics
            self._update_stats(result)
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Performance analysis complete - {result.total_issues} issues",
                    metadata={
                        "performance_score": result.performance_score,
                        "total_issues": result.total_issues,
                        "critical_issues": result.critical_issues,
                        "analysis_time": result.analysis_time
                    }
                )
            
            logger.info(f"Performance analysis completed: {result.total_issues} issues found")
            return result
            
        except Exception as e:
            logger.error(f"Performance analysis error: {e}")
            return self._create_error_result(start_time, str(e))
    
    async def _analyze_patterns(self, code: str) -> List[PerformanceIssue]:
        """Analyze using performance patterns"""
        issues = []
        lines = code.split('\n')
        
        for issue_type, patterns in self.PERFORMANCE_PATTERNS.items():
            for pattern_config in patterns:
                # Single line pattern matching
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern_config['pattern'], line, re.IGNORECASE):
                        issue = PerformanceIssue(
                            issue_type=issue_type,
                            impact=pattern_config['impact'],
                            title=f"{issue_type.value.replace('_', ' ').title()} Issue",
                            description=pattern_config['description'],
                            line_number=line_num,
                            code_snippet=line.strip(),
                            recommendation=pattern_config['recommendation'],
                            confidence=0.8
                        )
                        issues.append(issue)
                
                # Multi-line pattern matching
                if re.search(pattern_config['pattern'], code, re.MULTILINE | re.DOTALL):
                    issue = PerformanceIssue(
                        issue_type=issue_type,
                        impact=pattern_config['impact'],
                        title=f"{issue_type.value.replace('_', ' ').title()} Issue",
                        description=pattern_config['description'],
                        recommendation=pattern_config['recommendation'],
                        confidence=0.8
                    )
                    issues.append(issue)
        
        return issues
    
    async def _analyze_ast(self, code: str) -> List[PerformanceIssue]:
        """AST-based performance analysis"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            class PerformanceVisitor(ast.NodeVisitor):
                def __init__(self, issues_list):
                    self.issues = issues_list
                    self.loop_depth = 0
                    self.function_complexity = {}
                
                def visit_For(self, node):
                    self.loop_depth += 1
                    
                    # Check for nested loops
                    if self.loop_depth > 2:
                        self.issues.append(PerformanceIssue(
                            issue_type=PerformanceIssueType.TIME_COMPLEXITY,
                            impact=ImpactLevel.HIGH,
                            title="Deep Loop Nesting",
                            description=f"Loop nesting depth: {self.loop_depth}",
                            line_number=getattr(node, 'lineno', None),
                            recommendation="Consider flattening loops or using more efficient algorithms",
                            complexity_before="O(n³) or higher",
                            complexity_after="O(n²) or better",
                            confidence=0.9
                        ))
                    
                    self.generic_visit(node)
                    self.loop_depth -= 1
                
                def visit_While(self, node):
                    self.loop_depth += 1
                    self.generic_visit(node)
                    self.loop_depth -= 1
                
                def visit_ListComp(self, node):
                    # List comprehensions are generally good
                    # But check for nested comprehensions
                    if any(isinstance(generator.iter, ast.ListComp) for generator in node.generators):
                        self.issues.append(PerformanceIssue(
                            issue_type=PerformanceIssueType.MEMORY_USAGE,
                            impact=ImpactLevel.MEDIUM,
                            title="Nested List Comprehension",
                            description="Nested list comprehensions can consume excessive memory",
                            line_number=getattr(node, 'lineno', None),
                            recommendation="Consider using generator expressions or breaking into separate operations",
                            confidence=0.7
                        ))
                    
                    self.generic_visit(node)
                
                def visit_Call(self, node):
                    # Check for inefficient function calls
                    if isinstance(node.func, ast.Attribute):
                        attr_name = node.func.attr
                        
                        # Check for .count() usage
                        if attr_name == 'count' and len(node.args) == 1:
                            self.issues.append(PerformanceIssue(
                                issue_type=PerformanceIssueType.DATA_STRUCTURE,
                                impact=ImpactLevel.LOW,
                                title="Inefficient Count Usage",
                                description="Using .count() for existence check",
                                line_number=getattr(node, 'lineno', None),
                                recommendation="Use 'in' operator for existence checks",
                                confidence=0.8
                            ))
                    
                    self.generic_visit(node)
            
            visitor = PerformanceVisitor(issues)
            visitor.visit(tree)
            
        except SyntaxError as e:
            logger.warning(f"Could not parse code for AST analysis: {e}")
        
        return issues
    
    async def _analyze_complexity(self, code: str) -> List[PerformanceIssue]:
        """Analyze computational complexity"""
        issues = []
        lines = code.split('\n')
        
        # Count nested structures
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.strip()
            if any(keyword in stripped for keyword in ['for ', 'while ', 'if ']):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif stripped in ['', 'pass'] or stripped.startswith(('return', 'break', 'continue')):
                current_depth = max(0, current_depth - 1)
        
        if max_depth > 4:
            issues.append(PerformanceIssue(
                issue_type=PerformanceIssueType.TIME_COMPLEXITY,
                impact=ImpactLevel.HIGH,
                title="High Computational Complexity",
                description=f"Maximum nesting depth: {max_depth}",
                recommendation="Consider breaking complex logic into smaller functions",
                complexity_before=f"O(n^{max_depth})",
                complexity_after="O(n) or O(n log n)",
                confidence=0.7
            ))
        
        return issues
    
    async def _analyze_with_ai(self, code: str, context: str) -> List[PerformanceIssue]:
        """AI-powered performance analysis"""
        issues = []
        
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models.hf_models import HuggingFaceCodeModels
            
            hf_models = HuggingFaceCodeModels()
            result = await hf_models.analyze_code(
                code=code,
                task_type="performance_analysis",
                context=f"Performance optimization context: {context}"
            )
            
            if result.issues_found:
                for issue in result.issues_found:
                    impact_map = {
                        'critical': ImpactLevel.CRITICAL,
                        'high': ImpactLevel.HIGH,
                        'medium': ImpactLevel.MEDIUM,
                        'low': ImpactLevel.LOW
                    }
                    
                    perf_issue = PerformanceIssue(
                        issue_type=PerformanceIssueType.ALGORITHMIC,
                        impact=impact_map.get(issue.get('severity', 'medium'), ImpactLevel.MEDIUM),
                        title="AI-Detected Performance Issue",
                        description=issue.get('message', 'Performance issue detected by AI'),
                        line_number=issue.get('line_number'),
                        confidence=result.confidence_score,
                        recommendation="Review AI suggestions for optimization opportunities"
                    )
                    issues.append(perf_issue)
        
        except Exception as e:
            logger.warning(f"AI performance analysis failed: {e}")
        
        return issues
    
    def _compile_results(
        self, 
        issues: List[PerformanceIssue], 
        start_time: datetime,
        code: str
    ) -> PerformanceAnalysisResult:
        """Compile performance analysis results"""
        
        # Count issues by impact
        critical_count = sum(1 for i in issues if i.impact == ImpactLevel.CRITICAL)
        high_count = sum(1 for i in issues if i.impact == ImpactLevel.HIGH)
        medium_count = sum(1 for i in issues if i.impact == ImpactLevel.MEDIUM)
        low_count = sum(1 for i in issues if i.impact == ImpactLevel.LOW)
        
        total_issues = len(issues)
        
        # Calculate performance score (0-10)
        if total_issues == 0:
            performance_score = 10.0
        else:
            impact_weights = {
                ImpactLevel.CRITICAL: 3.0,
                ImpactLevel.HIGH: 2.0,
                ImpactLevel.MEDIUM: 1.0,
                ImpactLevel.LOW: 0.5
            }
            
            total_weight = sum(impact_weights.get(issue.impact, 1.0) for issue in issues)
            performance_score = max(0.0, 10.0 - (total_weight * 0.3))
        
        # Generate optimizations
        optimizations = []
        issue_types = set(issue.issue_type for issue in issues)
        for issue_type in issue_types:
            optimizations.extend(self.OPTIMIZATION_SUGGESTIONS.get(issue_type, []))
        
        optimizations = list(set(optimizations))[:8]  # Limit to 8 suggestions
        
        # Calculate complexity metrics
        complexity_metrics = self._calculate_complexity_metrics(code)
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        return PerformanceAnalysisResult(
            issues=issues,
            performance_score=performance_score,
            total_issues=total_issues,
            critical_issues=critical_count,
            high_issues=high_count,
            medium_issues=medium_count,
            low_issues=low_count,
            analysis_time=analysis_time,
            optimizations=optimizations,
            complexity_metrics=complexity_metrics
        )
    
    def _calculate_complexity_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate basic complexity metrics"""
        lines = [line for line in code.split('\n') if line.strip()]
        
        # Count various constructs
        loop_count = code.count('for ') + code.count('while ')
        condition_count = code.count('if ') + code.count('elif ')
        function_count = code.count('def ')
        class_count = code.count('class ')
        
        # Estimate cyclomatic complexity
        cyclomatic_complexity = 1 + condition_count + loop_count
        
        return {
            'lines_of_code': len(lines),
            'loop_count': loop_count,
            'condition_count': condition_count,
            'function_count': function_count,
            'class_count': class_count,
            'cyclomatic_complexity': cyclomatic_complexity,
            'complexity_level': self._get_complexity_level(cyclomatic_complexity)
        }
    
    def _get_complexity_level(self, complexity: int) -> str:
        """Get complexity level description"""
        if complexity <= 10:
            return "Simple"
        elif complexity <= 20:
            return "Moderate"
        elif complexity <= 50:
            return "Complex"
        else:
            return "Very Complex"
    
    def _create_error_result(self, start_time: datetime, error_msg: str) -> PerformanceAnalysisResult:
        """Create error result"""
        return PerformanceAnalysisResult(
            issues=[],
            performance_score=0.0,
            total_issues=1,
            critical_issues=0,
            high_issues=0,
            medium_issues=0,
            low_issues=1,
            analysis_time=(datetime.now() - start_time).total_seconds(),
            optimizations=[f"Analysis failed: {error_msg}"],
            complexity_metrics={}
        )
    
    def _update_stats(self, result: PerformanceAnalysisResult):
        """Update analyzer statistics"""
        self.analysis_stats['total_analyses'] += 1
        self.analysis_stats['issues_found'] += result.total_issues
        
        # Update average performance score
        current_avg = self.analysis_stats['avg_performance_score']
        total_analyses = self.analysis_stats['total_analyses']
        
        if total_analyses == 1:
            self.analysis_stats['avg_performance_score'] = result.performance_score
        else:
            self.analysis_stats['avg_performance_score'] = (
                (current_avg * (total_analyses - 1) + result.performance_score) / total_analyses
            )
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get performance analyzer statistics"""
        return {
            **self.analysis_stats,
            'supported_issue_types': len(self.PERFORMANCE_PATTERNS),
            'total_patterns': sum(len(patterns) for patterns in self.PERFORMANCE_PATTERNS.values())
        }