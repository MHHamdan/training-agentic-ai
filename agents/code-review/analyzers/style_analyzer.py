"""
Code Style and Quality Analyzer
PEP8 compliance, naming conventions, and formatting analysis
Author: Mohammed Hamdan
"""

import re
import ast
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

class StyleIssueType(Enum):
    """Types of style issues"""
    PEP8_VIOLATION = "pep8_violation"
    NAMING_CONVENTION = "naming_convention"
    CODE_FORMATTING = "code_formatting"
    IMPORT_ORGANIZATION = "import_organization"
    LINE_LENGTH = "line_length"
    WHITESPACE = "whitespace"
    DOCSTRING = "docstring"
    TYPE_ANNOTATION = "type_annotation"
    COMMENT_QUALITY = "comment_quality"

class StyleSeverity(Enum):
    """Style issue severity levels"""
    ERROR = "error"        # Must fix
    WARNING = "warning"    # Should fix
    INFO = "info"         # Consider fixing
    SUGGESTION = "suggestion"  # Optional improvement

@dataclass
class StyleIssue:
    """Style issue finding"""
    issue_type: StyleIssueType
    severity: StyleSeverity
    title: str
    description: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    code_snippet: Optional[str] = None
    fix_suggestion: Optional[str] = None
    rule_code: Optional[str] = None  # PEP8 rule code
    confidence: float = 1.0

@dataclass
class StyleAnalysisResult:
    """Complete style analysis result"""
    issues: List[StyleIssue]
    style_score: float
    total_issues: int
    error_count: int
    warning_count: int
    info_count: int
    suggestion_count: int
    analysis_time: float
    pep8_compliance: float
    improvements: List[str]
    good_practices: List[str]

class StyleAnalyzer:
    """
    Code style and quality analyzer
    Focuses on PEP8 compliance and Python best practices
    """
    
    # PEP8 and style patterns
    STYLE_PATTERNS = {
        StyleIssueType.LINE_LENGTH: [
            {
                'check': lambda line: len(line) > 79,
                'description': 'Line too long (>79 characters)',
                'severity': StyleSeverity.WARNING,
                'rule_code': 'E501',
                'fix': 'Break line using parentheses or backslash'
            }
        ],
        
        StyleIssueType.WHITESPACE: [
            {
                'pattern': r'\s+$',
                'description': 'Trailing whitespace',
                'severity': StyleSeverity.WARNING,
                'rule_code': 'W291',
                'fix': 'Remove trailing whitespace'
            },
            {
                'pattern': r'\t',
                'description': 'Tab character used (use 4 spaces)',
                'severity': StyleSeverity.ERROR,
                'rule_code': 'W191',
                'fix': 'Replace tabs with 4 spaces'
            },
            {
                'pattern': r'  +def |  +class |  +if |  +for |  +while ',
                'description': 'Inconsistent indentation (not 4 spaces)',
                'severity': StyleSeverity.ERROR,
                'rule_code': 'E111',
                'fix': 'Use exactly 4 spaces for indentation'
            }
        ],
        
        StyleIssueType.CODE_FORMATTING: [
            {
                'pattern': r'[^=!<>]==[^=]|[^=!<>]!=[^=]',
                'description': 'Missing spaces around comparison operators',
                'severity': StyleSeverity.WARNING,
                'rule_code': 'E225',
                'fix': 'Add spaces around == and !='
            },
            {
                'pattern': r'[a-zA-Z0-9]\+[a-zA-Z0-9]|[a-zA-Z0-9]-[a-zA-Z0-9]',
                'description': 'Missing spaces around arithmetic operators',
                'severity': StyleSeverity.WARNING,
                'rule_code': 'E226',
                'fix': 'Add spaces around +, -, *, /'
            },
            {
                'pattern': r',\S',
                'description': 'Missing space after comma',
                'severity': StyleSeverity.WARNING,
                'rule_code': 'E231',
                'fix': 'Add space after comma'
            },
            {
                'pattern': r':\S',
                'description': 'Missing space after colon',
                'severity': StyleSeverity.WARNING,
                'rule_code': 'E231',
                'fix': 'Add space after colon'
            }
        ],
        
        StyleIssueType.NAMING_CONVENTION: [
            {
                'pattern': r'def [A-Z][a-zA-Z]*\(',
                'description': 'Function name should be lowercase with underscores',
                'severity': StyleSeverity.WARNING,
                'rule_code': 'N802',
                'fix': 'Use snake_case for function names'
            },
            {
                'pattern': r'class [a-z][a-zA-Z]*[:\(]',
                'description': 'Class name should use CapWords convention',
                'severity': StyleSeverity.WARNING,
                'rule_code': 'N801',
                'fix': 'Use PascalCase for class names'
            },
            {
                'pattern': r'[A-Z]{2,}[a-z]',
                'description': 'Avoid acronyms in mixed case',
                'severity': StyleSeverity.INFO,
                'rule_code': 'N806',
                'fix': 'Use consistent case for acronyms'
            }
        ],
        
        StyleIssueType.IMPORT_ORGANIZATION: [
            {
                'pattern': r'from .* import \*',
                'description': 'Avoid wildcard imports',
                'severity': StyleSeverity.WARNING,
                'rule_code': 'F403',
                'fix': 'Import specific names or use module import'
            },
            {
                'pattern': r'import [a-zA-Z0-9_]+, [a-zA-Z0-9_]+',
                'description': 'Multiple imports on one line',
                'severity': StyleSeverity.INFO,
                'rule_code': 'E401',
                'fix': 'Put each import on separate line'
            }
        ]
    }
    
    # Best practice recommendations
    STYLE_RECOMMENDATIONS = {
        StyleIssueType.PEP8_VIOLATION: [
            "Follow PEP 8 style guide consistently",
            "Use automatic code formatters like Black",
            "Configure IDE/editor for PEP 8 compliance",
            "Run flake8 or similar linters regularly"
        ],
        StyleIssueType.NAMING_CONVENTION: [
            "Use snake_case for functions and variables",
            "Use PascalCase for class names",
            "Use UPPER_CASE for constants",
            "Choose descriptive and meaningful names"
        ],
        StyleIssueType.DOCSTRING: [
            "Add docstrings to all public functions and classes",
            "Follow Google or NumPy docstring conventions",
            "Include parameter types and return values",
            "Provide usage examples for complex functions"
        ],
        StyleIssueType.TYPE_ANNOTATION: [
            "Add type hints to function parameters and returns",
            "Use typing module for complex types",
            "Consider using mypy for type checking",
            "Document expected types in docstrings"
        ]
    }
    
    def __init__(self):
        """Initialize style analyzer"""
        self.analysis_stats = {
            'total_analyses': 0,
            'issues_found': 0,
            'avg_style_score': 0.0
        }
        logger.info("Style Analyzer initialized")
    
    @observe(as_type="style_analysis")
    async def analyze_style(
        self,
        code: str,
        context: str = "",
        include_ai_analysis: bool = True
    ) -> StyleAnalysisResult:
        """
        Comprehensive style analysis
        
        Args:
            code: Source code to analyze
            context: Additional context
            include_ai_analysis: Whether to include AI analysis
            
        Returns:
            StyleAnalysisResult with findings
        """
        start_time = datetime.now()
        
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Style analysis - {len(code)} characters",
                    metadata={
                        "analysis_type": "style",
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
            
            # 3. Docstring analysis
            docstring_issues = await self._analyze_docstrings(code)
            issues.extend(docstring_issues)
            
            # 4. Type annotation analysis
            type_issues = await self._analyze_type_annotations(code)
            issues.extend(type_issues)
            
            # 5. AI-powered analysis
            if include_ai_analysis:
                ai_issues = await self._analyze_with_ai(code, context)
                issues.extend(ai_issues)
            
            # 6. Compile results
            result = self._compile_results(issues, start_time, code)
            
            # 7. Update statistics
            self._update_stats(result)
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Style analysis complete - {result.total_issues} issues",
                    metadata={
                        "style_score": result.style_score,
                        "pep8_compliance": result.pep8_compliance,
                        "total_issues": result.total_issues,
                        "analysis_time": result.analysis_time
                    }
                )
            
            logger.info(f"Style analysis completed: {result.total_issues} issues found")
            return result
            
        except Exception as e:
            logger.error(f"Style analysis error: {e}")
            return self._create_error_result(start_time, str(e))
    
    async def _analyze_patterns(self, code: str) -> List[StyleIssue]:
        """Pattern-based style analysis"""
        issues = []
        lines = code.split('\n')
        
        for issue_type, patterns in self.STYLE_PATTERNS.items():
            for pattern_config in patterns:
                if 'check' in pattern_config:
                    # Custom check function
                    check_func = pattern_config['check']
                    for line_num, line in enumerate(lines, 1):
                        if check_func(line):
                            issue = StyleIssue(
                                issue_type=issue_type,
                                severity=pattern_config['severity'],
                                title=f"{issue_type.value.replace('_', ' ').title()}",
                                description=pattern_config['description'],
                                line_number=line_num,
                                code_snippet=line.rstrip(),
                                fix_suggestion=pattern_config['fix'],
                                rule_code=pattern_config.get('rule_code'),
                                confidence=0.9
                            )
                            issues.append(issue)
                
                elif 'pattern' in pattern_config:
                    # Regex pattern
                    pattern = pattern_config['pattern']
                    for line_num, line in enumerate(lines, 1):
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            issue = StyleIssue(
                                issue_type=issue_type,
                                severity=pattern_config['severity'],
                                title=f"{issue_type.value.replace('_', ' ').title()}",
                                description=pattern_config['description'],
                                line_number=line_num,
                                column_number=match.start(),
                                code_snippet=line.rstrip(),
                                fix_suggestion=pattern_config['fix'],
                                rule_code=pattern_config.get('rule_code'),
                                confidence=0.9
                            )
                            issues.append(issue)
        
        return issues
    
    async def _analyze_ast(self, code: str) -> List[StyleIssue]:
        """AST-based style analysis"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            class StyleVisitor(ast.NodeVisitor):
                def __init__(self, issues_list):
                    self.issues = issues_list
                
                def visit_FunctionDef(self, node):
                    # Check function naming
                    func_name = node.name
                    if not self._is_snake_case(func_name) and not func_name.startswith('__'):
                        self.issues.append(StyleIssue(
                            issue_type=StyleIssueType.NAMING_CONVENTION,
                            severity=StyleSeverity.WARNING,
                            title="Function Naming Convention",
                            description=f"Function '{func_name}' should use snake_case",
                            line_number=getattr(node, 'lineno', None),
                            fix_suggestion=f"Rename to '{self._to_snake_case(func_name)}'",
                            rule_code="N802",
                            confidence=0.9
                        ))
                    
                    # Check for missing docstrings
                    if not ast.get_docstring(node) and not func_name.startswith('_'):
                        self.issues.append(StyleIssue(
                            issue_type=StyleIssueType.DOCSTRING,
                            severity=StyleSeverity.INFO,
                            title="Missing Function Docstring",
                            description=f"Public function '{func_name}' lacks docstring",
                            line_number=getattr(node, 'lineno', None),
                            fix_suggestion="Add docstring describing function purpose",
                            rule_code="D100",
                            confidence=0.8
                        ))
                    
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    # Check class naming
                    class_name = node.name
                    if not self._is_pascal_case(class_name):
                        self.issues.append(StyleIssue(
                            issue_type=StyleIssueType.NAMING_CONVENTION,
                            severity=StyleSeverity.WARNING,
                            title="Class Naming Convention",
                            description=f"Class '{class_name}' should use PascalCase",
                            line_number=getattr(node, 'lineno', None),
                            fix_suggestion=f"Rename to '{self._to_pascal_case(class_name)}'",
                            rule_code="N801",
                            confidence=0.9
                        ))
                    
                    # Check for missing docstrings
                    if not ast.get_docstring(node):
                        self.issues.append(StyleIssue(
                            issue_type=StyleIssueType.DOCSTRING,
                            severity=StyleSeverity.INFO,
                            title="Missing Class Docstring",
                            description=f"Class '{class_name}' lacks docstring",
                            line_number=getattr(node, 'lineno', None),
                            fix_suggestion="Add docstring describing class purpose",
                            rule_code="D101",
                            confidence=0.8
                        ))
                    
                    self.generic_visit(node)
                
                def visit_Assign(self, node):
                    # Check constant naming
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            if self._looks_like_constant(node.value) and not var_name.isupper():
                                self.issues.append(StyleIssue(
                                    issue_type=StyleIssueType.NAMING_CONVENTION,
                                    severity=StyleSeverity.INFO,
                                    title="Constant Naming Convention",
                                    description=f"Constant '{var_name}' should be UPPER_CASE",
                                    line_number=getattr(node, 'lineno', None),
                                    fix_suggestion=f"Rename to '{var_name.upper()}'",
                                    rule_code="N806",
                                    confidence=0.7
                                ))
                    
                    self.generic_visit(node)
                
                def _is_snake_case(self, name: str) -> bool:
                    return re.match(r'^[a-z_][a-z0-9_]*$', name) is not None
                
                def _is_pascal_case(self, name: str) -> bool:
                    return re.match(r'^[A-Z][a-zA-Z0-9]*$', name) is not None
                
                def _to_snake_case(self, name: str) -> str:
                    # Simple conversion to snake_case
                    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
                
                def _to_pascal_case(self, name: str) -> str:
                    return ''.join(word.capitalize() for word in name.split('_'))
                
                def _looks_like_constant(self, value_node) -> bool:
                    return isinstance(value_node, (ast.Constant, ast.Num, ast.Str))
            
            visitor = StyleVisitor(issues)
            visitor.visit(tree)
            
        except SyntaxError as e:
            logger.warning(f"Could not parse code for AST analysis: {e}")
        
        return issues
    
    async def _analyze_docstrings(self, code: str) -> List[StyleIssue]:
        """Analyze docstring quality"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Check docstring quality
                        if len(docstring.strip()) < 10:
                            issues.append(StyleIssue(
                                issue_type=StyleIssueType.DOCSTRING,
                                severity=StyleSeverity.INFO,
                                title="Short Docstring",
                                description=f"Docstring for '{node.name}' is very short",
                                line_number=getattr(node, 'lineno', None),
                                fix_suggestion="Provide more detailed description",
                                confidence=0.7
                            ))
                        
                        # Check for triple quotes
                        if '"""' not in code and "'''" not in code:
                            issues.append(StyleIssue(
                                issue_type=StyleIssueType.DOCSTRING,
                                severity=StyleSeverity.WARNING,
                                title="Docstring Format",
                                description="Use triple quotes for docstrings",
                                line_number=getattr(node, 'lineno', None),
                                fix_suggestion="Use \"\"\" for docstring formatting",
                                confidence=0.8
                            ))
        
        except SyntaxError:
            pass
        
        return issues
    
    async def _analyze_type_annotations(self, code: str) -> List[StyleIssue]:
        """Analyze type annotation usage"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for missing return annotation
                    if not node.returns and not node.name.startswith('_'):
                        issues.append(StyleIssue(
                            issue_type=StyleIssueType.TYPE_ANNOTATION,
                            severity=StyleSeverity.INFO,
                            title="Missing Return Type Annotation",
                            description=f"Function '{node.name}' lacks return type annotation",
                            line_number=getattr(node, 'lineno', None),
                            fix_suggestion="Add return type annotation (e.g., -> str)",
                            confidence=0.6
                        ))
                    
                    # Check for missing parameter annotations
                    for arg in node.args.args:
                        if not arg.annotation and arg.arg != 'self':
                            issues.append(StyleIssue(
                                issue_type=StyleIssueType.TYPE_ANNOTATION,
                                severity=StyleSeverity.INFO,
                                title="Missing Parameter Type Annotation",
                                description=f"Parameter '{arg.arg}' lacks type annotation",
                                line_number=getattr(node, 'lineno', None),
                                fix_suggestion=f"Add type annotation for '{arg.arg}'",
                                confidence=0.6
                            ))
        
        except SyntaxError:
            pass
        
        return issues
    
    async def _analyze_with_ai(self, code: str, context: str) -> List[StyleIssue]:
        """AI-powered style analysis"""
        issues = []
        
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models.hf_models import HuggingFaceCodeModels
            
            hf_models = HuggingFaceCodeModels()
            result = await hf_models.analyze_code(
                code=code,
                task_type="style_analysis",
                context=f"Style analysis context: {context}"
            )
            
            if result.issues_found:
                for issue in result.issues_found:
                    severity_map = {
                        'critical': StyleSeverity.ERROR,
                        'high': StyleSeverity.WARNING,
                        'medium': StyleSeverity.INFO,
                        'low': StyleSeverity.SUGGESTION
                    }
                    
                    style_issue = StyleIssue(
                        issue_type=StyleIssueType.PEP8_VIOLATION,
                        severity=severity_map.get(issue.get('severity', 'info'), StyleSeverity.INFO),
                        title="AI-Detected Style Issue",
                        description=issue.get('message', 'Style issue detected by AI'),
                        line_number=issue.get('line_number'),
                        confidence=result.confidence_score,
                        fix_suggestion="Review AI suggestions for style improvements"
                    )
                    issues.append(style_issue)
        
        except Exception as e:
            logger.warning(f"AI style analysis failed: {e}")
        
        return issues
    
    def _compile_results(
        self, 
        issues: List[StyleIssue], 
        start_time: datetime,
        code: str
    ) -> StyleAnalysisResult:
        """Compile style analysis results"""
        
        # Count issues by severity
        error_count = sum(1 for i in issues if i.severity == StyleSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == StyleSeverity.WARNING)
        info_count = sum(1 for i in issues if i.severity == StyleSeverity.INFO)
        suggestion_count = sum(1 for i in issues if i.severity == StyleSeverity.SUGGESTION)
        
        total_issues = len(issues)
        
        # Calculate style score (0-10)
        if total_issues == 0:
            style_score = 10.0
        else:
            severity_weights = {
                StyleSeverity.ERROR: 2.0,
                StyleSeverity.WARNING: 1.0,
                StyleSeverity.INFO: 0.5,
                StyleSeverity.SUGGESTION: 0.2
            }
            
            total_weight = sum(severity_weights.get(issue.severity, 0.5) for issue in issues)
            lines_of_code = len([line for line in code.split('\n') if line.strip()])
            
            # Normalize by code length
            normalized_weight = total_weight / max(1, lines_of_code / 10)
            style_score = max(0.0, 10.0 - normalized_weight)
        
        # Calculate PEP8 compliance
        pep8_issues = [i for i in issues if i.rule_code and i.rule_code.startswith(('E', 'W'))]
        total_lines = len(code.split('\n'))
        pep8_compliance = max(0.0, 1.0 - (len(pep8_issues) / max(1, total_lines / 10)))
        
        # Generate improvements
        improvements = []
        issue_types = set(issue.issue_type for issue in issues)
        for issue_type in issue_types:
            improvements.extend(self.STYLE_RECOMMENDATIONS.get(issue_type, []))
        
        improvements = list(set(improvements))[:6]  # Limit to 6
        
        # Generate good practices
        good_practices = []
        if error_count == 0:
            good_practices.append("No critical style errors detected")
        if warning_count < 3:
            good_practices.append("Good adherence to style guidelines")
        if total_issues == 0:
            good_practices.append("Code follows Python style conventions")
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        return StyleAnalysisResult(
            issues=issues,
            style_score=style_score,
            total_issues=total_issues,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            suggestion_count=suggestion_count,
            analysis_time=analysis_time,
            pep8_compliance=pep8_compliance,
            improvements=improvements,
            good_practices=good_practices
        )
    
    def _create_error_result(self, start_time: datetime, error_msg: str) -> StyleAnalysisResult:
        """Create error result"""
        return StyleAnalysisResult(
            issues=[],
            style_score=0.0,
            total_issues=1,
            error_count=0,
            warning_count=0,
            info_count=1,
            suggestion_count=0,
            analysis_time=(datetime.now() - start_time).total_seconds(),
            pep8_compliance=0.0,
            improvements=[f"Analysis failed: {error_msg}"],
            good_practices=[]
        )
    
    def _update_stats(self, result: StyleAnalysisResult):
        """Update analyzer statistics"""
        self.analysis_stats['total_analyses'] += 1
        self.analysis_stats['issues_found'] += result.total_issues
        
        # Update average style score
        current_avg = self.analysis_stats['avg_style_score']
        total_analyses = self.analysis_stats['total_analyses']
        
        if total_analyses == 1:
            self.analysis_stats['avg_style_score'] = result.style_score
        else:
            self.analysis_stats['avg_style_score'] = (
                (current_avg * (total_analyses - 1) + result.style_score) / total_analyses
            )
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get style analyzer statistics"""
        return {
            **self.analysis_stats,
            'supported_issue_types': len(self.STYLE_PATTERNS),
            'total_patterns': sum(len(patterns) for patterns in self.STYLE_PATTERNS.values())
        }