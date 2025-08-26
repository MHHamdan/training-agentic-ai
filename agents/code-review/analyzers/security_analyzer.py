"""
Security Vulnerability Analyzer
Comprehensive security analysis with pattern detection and AI assistance
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

class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss_vulnerability"
    COMMAND_INJECTION = "command_injection"
    HARDCODED_SECRET = "hardcoded_secret"
    INSECURE_CRYPTO = "insecure_crypto"
    PATH_TRAVERSAL = "path_traversal"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    WEAK_RANDOM = "weak_random"
    INPUT_VALIDATION = "input_validation"
    AUTHENTICATION_BYPASS = "authentication_bypass"

class SeverityLevel(Enum):
    """Security vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class SecurityVulnerability:
    """Security vulnerability finding"""
    vuln_type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None
    confidence: float = 1.0

@dataclass
class SecurityAnalysisResult:
    """Complete security analysis result"""
    vulnerabilities: List[SecurityVulnerability]
    security_score: float
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    analysis_time: float
    recommendations: List[str]
    compliant_practices: List[str]

class SecurityAnalyzer:
    """
    Comprehensive security vulnerability analyzer
    Combines pattern matching, AST analysis, and AI assistance
    """
    
    # Security patterns for vulnerability detection
    SECURITY_PATTERNS = {
        VulnerabilityType.SQL_INJECTION: [
            {
                'pattern': r'(execute|exec)\s*\(\s*["\'].*%.*["\']',
                'description': 'SQL string formatting in execute statement',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-89'
            },
            {
                'pattern': r'cursor\.execute\s*\(\s*[f"\'].*\{.*\}.*["\']',
                'description': 'f-string SQL injection vulnerability',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-89'
            },
            {
                'pattern': r'SELECT.*FROM.*WHERE.*\+.*',
                'description': 'SQL concatenation vulnerability',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-89'
            }
        ],
        
        VulnerabilityType.XSS: [
            {
                'pattern': r'\.innerHTML\s*=.*\+',
                'description': 'Potential XSS via innerHTML concatenation',
                'severity': SeverityLevel.MEDIUM,
                'cwe': 'CWE-79'
            },
            {
                'pattern': r'render_template_string\s*\(.*\+.*\)',
                'description': 'Template injection vulnerability',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-79'
            }
        ],
        
        VulnerabilityType.COMMAND_INJECTION: [
            {
                'pattern': r'subprocess\.(call|run|Popen).*shell\s*=\s*True',
                'description': 'Command injection via subprocess with shell=True',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-78'
            },
            {
                'pattern': r'os\.system\s*\(.*\+.*\)',
                'description': 'Command injection via os.system',
                'severity': SeverityLevel.CRITICAL,
                'cwe': 'CWE-78'
            },
            {
                'pattern': r'eval\s*\(.*input.*\)',
                'description': 'Code injection via eval with user input',
                'severity': SeverityLevel.CRITICAL,
                'cwe': 'CWE-95'
            }
        ],
        
        VulnerabilityType.HARDCODED_SECRET: [
            {
                'pattern': r'(password|pwd|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                'description': 'Hardcoded password or secret',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-798'
            },
            {
                'pattern': r'(api_key|apikey|access_key)\s*=\s*["\'][A-Za-z0-9+/]{20,}["\']',
                'description': 'Hardcoded API key',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-798'
            },
            {
                'pattern': r'(sk-[a-zA-Z0-9]{32,}|xoxb-[a-zA-Z0-9-]+)',
                'description': 'Hardcoded service token',
                'severity': SeverityLevel.CRITICAL,
                'cwe': 'CWE-798'
            }
        ],
        
        VulnerabilityType.INSECURE_CRYPTO: [
            {
                'pattern': r'hashlib\.(md5|sha1)\s*\(',
                'description': 'Weak cryptographic hash function',
                'severity': SeverityLevel.MEDIUM,
                'cwe': 'CWE-327'
            },
            {
                'pattern': r'DES\.new\s*\(',
                'description': 'Weak encryption algorithm (DES)',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-327'
            },
            {
                'pattern': r'random\.random\s*\(\)',
                'description': 'Weak random number generator for security',
                'severity': SeverityLevel.MEDIUM,
                'cwe': 'CWE-330'
            }
        ],
        
        VulnerabilityType.PATH_TRAVERSAL: [
            {
                'pattern': r'open\s*\(.*\+.*["\']\.\.\/["\']',
                'description': 'Path traversal vulnerability',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-22'
            },
            {
                'pattern': r'os\.path\.join\s*\(.*input.*\)',
                'description': 'Potential path traversal via user input',
                'severity': SeverityLevel.MEDIUM,
                'cwe': 'CWE-22'
            }
        ],
        
        VulnerabilityType.UNSAFE_DESERIALIZATION: [
            {
                'pattern': r'pickle\.loads?\s*\(',
                'description': 'Unsafe deserialization with pickle',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-502'
            },
            {
                'pattern': r'yaml\.load\s*\(',
                'description': 'Unsafe YAML deserialization',
                'severity': SeverityLevel.HIGH,
                'cwe': 'CWE-502'
            }
        ]
    }
    
    # Secure coding recommendations
    SECURITY_RECOMMENDATIONS = {
        VulnerabilityType.SQL_INJECTION: [
            "Use parameterized queries or prepared statements",
            "Implement input validation and sanitization",
            "Use ORM frameworks when possible",
            "Apply principle of least privilege for database access"
        ],
        VulnerabilityType.COMMAND_INJECTION: [
            "Avoid shell=True in subprocess calls",
            "Use subprocess with explicit argument lists",
            "Validate and sanitize all user inputs",
            "Consider using safer alternatives to os.system()"
        ],
        VulnerabilityType.HARDCODED_SECRET: [
            "Use environment variables for secrets",
            "Implement proper secret management systems",
            "Rotate secrets regularly",
            "Never commit secrets to version control"
        ],
        VulnerabilityType.INSECURE_CRYPTO: [
            "Use SHA-256 or stronger hash functions",
            "Implement proper encryption with AES-256",
            "Use cryptographically secure random generators",
            "Follow current cryptographic best practices"
        ]
    }
    
    def __init__(self):
        """Initialize security analyzer"""
        self.findings = []
        self.analysis_stats = {
            'total_scans': 0,
            'vulnerabilities_found': 0,
            'avg_security_score': 0.0
        }
        logger.info("Security Analyzer initialized")
    
    @observe(as_type="security_analysis")
    async def analyze_security(
        self,
        code: str,
        context: str = "",
        include_ai_analysis: bool = True
    ) -> SecurityAnalysisResult:
        """
        Comprehensive security analysis of code
        
        Args:
            code: Source code to analyze
            context: Additional context about the code
            include_ai_analysis: Whether to include AI-powered analysis
            
        Returns:
            SecurityAnalysisResult with all findings
        """
        start_time = datetime.now()
        
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Security analysis request - {len(code)} characters",
                    metadata={
                        "analysis_type": "security_scan",
                        "code_length": len(code),
                        "context_provided": bool(context),
                        "ai_analysis_enabled": include_ai_analysis,
                        "organization": "code-review-org",
                        "project": "code-review-agent-v2"
                    }
                )
            
            vulnerabilities = []
            
            # 1. Pattern-based analysis
            pattern_vulns = await self._analyze_patterns(code)
            vulnerabilities.extend(pattern_vulns)
            
            # 2. AST-based analysis
            ast_vulns = await self._analyze_ast(code)
            vulnerabilities.extend(ast_vulns)
            
            # 3. Context-specific analysis
            context_vulns = await self._analyze_context(code, context)
            vulnerabilities.extend(context_vulns)
            
            # 4. AI-powered analysis (if enabled and available)
            if include_ai_analysis:
                ai_vulns = await self._analyze_with_ai(code, context)
                vulnerabilities.extend(ai_vulns)
            
            # 5. Calculate security metrics
            result = self._compile_results(vulnerabilities, start_time)
            
            # 6. Update statistics
            self._update_stats(result)
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Security analysis complete - {result.total_issues} issues found",
                    metadata={
                        "security_score": result.security_score,
                        "total_vulnerabilities": result.total_issues,
                        "critical_issues": result.critical_issues,
                        "high_issues": result.high_issues,
                        "analysis_time": result.analysis_time
                    }
                )
            
            logger.info(f"Security analysis completed: {result.total_issues} issues found")
            return result
            
        except Exception as e:
            logger.error(f"Security analysis error: {e}")
            # Return empty result on error
            return SecurityAnalysisResult(
                vulnerabilities=[],
                security_score=0.0,
                total_issues=1,
                critical_issues=0,
                high_issues=0,
                medium_issues=0,
                low_issues=1,
                analysis_time=(datetime.now() - start_time).total_seconds(),
                recommendations=[f"Analysis failed: {str(e)}"],
                compliant_practices=[]
            )
    
    async def _analyze_patterns(self, code: str) -> List[SecurityVulnerability]:
        """Analyze code using security patterns"""
        vulnerabilities = []
        lines = code.split('\n')
        
        for vuln_type, patterns in self.SECURITY_PATTERNS.items():
            for pattern_config in patterns:
                pattern = pattern_config['pattern']
                
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    
                    for match in matches:
                        vulnerability = SecurityVulnerability(
                            vuln_type=vuln_type,
                            severity=pattern_config['severity'],
                            title=f"{vuln_type.value.replace('_', ' ').title()} Detected",
                            description=pattern_config['description'],
                            line_number=line_num,
                            column_number=match.start(),
                            code_snippet=line.strip(),
                            cwe_id=pattern_config.get('cwe'),
                            confidence=0.9,  # High confidence for pattern matches
                            recommendation=self._get_recommendation(vuln_type)
                        )
                        vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    async def _analyze_ast(self, code: str) -> List[SecurityVulnerability]:
        """Analyze code using Abstract Syntax Tree"""
        vulnerabilities = []
        
        try:
            tree = ast.parse(code)
            
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self, vulnerabilities_list):
                    self.vulnerabilities = vulnerabilities_list
                
                def visit_Call(self, node):
                    # Check for dangerous function calls
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        # eval() calls
                        if func_name == 'eval':
                            self.vulnerabilities.append(SecurityVulnerability(
                                vuln_type=VulnerabilityType.COMMAND_INJECTION,
                                severity=SeverityLevel.CRITICAL,
                                title="Dangerous eval() Usage",
                                description="Use of eval() function can lead to code injection",
                                line_number=getattr(node, 'lineno', None),
                                cwe_id='CWE-95',
                                confidence=0.95,
                                recommendation="Avoid eval(). Use safer alternatives like literal_eval() for data parsing"
                            ))
                        
                        # exec() calls
                        elif func_name == 'exec':
                            self.vulnerabilities.append(SecurityVulnerability(
                                vuln_type=VulnerabilityType.COMMAND_INJECTION,
                                severity=SeverityLevel.CRITICAL,
                                title="Dangerous exec() Usage",
                                description="Use of exec() function can lead to code injection",
                                line_number=getattr(node, 'lineno', None),
                                cwe_id='CWE-95',
                                confidence=0.95,
                                recommendation="Avoid exec(). Consider safer alternatives for dynamic code execution"
                            ))
                    
                    self.generic_visit(node)
                
                def visit_Assign(self, node):
                    # Check for hardcoded secrets in assignments
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        value = node.value.value
                        
                        # Check for potential secrets
                        if len(value) > 20 and any(char.isalnum() for char in value):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    var_name = target.id.lower()
                                    if any(secret_word in var_name for secret_word in 
                                          ['password', 'key', 'secret', 'token', 'pwd']):
                                        self.vulnerabilities.append(SecurityVulnerability(
                                            vuln_type=VulnerabilityType.HARDCODED_SECRET,
                                            severity=SeverityLevel.HIGH,
                                            title="Hardcoded Secret Detected",
                                            description=f"Variable '{target.id}' appears to contain a hardcoded secret",
                                            line_number=getattr(node, 'lineno', None),
                                            cwe_id='CWE-798',
                                            confidence=0.8,
                                            recommendation="Use environment variables or secure configuration for secrets"
                                        ))
                    
                    self.generic_visit(node)
            
            visitor = SecurityVisitor(vulnerabilities)
            visitor.visit(tree)
            
        except SyntaxError as e:
            logger.warning(f"Could not parse code for AST analysis: {e}")
        
        return vulnerabilities
    
    async def _analyze_context(self, code: str, context: str) -> List[SecurityVulnerability]:
        """Context-specific security analysis"""
        vulnerabilities = []
        
        # Web application context
        if any(framework in code.lower() for framework in ['flask', 'django', 'fastapi']):
            # Check for missing CSRF protection
            if 'csrf' not in code.lower() and 'form' in code.lower():
                vulnerabilities.append(SecurityVulnerability(
                    vuln_type=VulnerabilityType.INPUT_VALIDATION,
                    severity=SeverityLevel.MEDIUM,
                    title="Potential CSRF Vulnerability",
                    description="Web forms should include CSRF protection",
                    confidence=0.6,
                    recommendation="Implement CSRF tokens for all forms"
                ))
        
        # Database context
        if any(db_lib in code.lower() for db_lib in ['sqlite3', 'psycopg2', 'mysql']):
            # Check for SQL injection patterns
            if any(pattern in code for pattern in ['%s', '+ ', 'format(']):
                vulnerabilities.append(SecurityVulnerability(
                    vuln_type=VulnerabilityType.SQL_INJECTION,
                    severity=SeverityLevel.HIGH,
                    title="Potential SQL Injection",
                    description="Database queries should use parameterized statements",
                    confidence=0.7,
                    recommendation="Use parameterized queries instead of string concatenation"
                ))
        
        # Authentication context
        if 'password' in code.lower() and 'hash' not in code.lower():
            vulnerabilities.append(SecurityVulnerability(
                vuln_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                severity=SeverityLevel.HIGH,
                title="Password Security Issue",
                description="Passwords should be properly hashed",
                confidence=0.6,
                recommendation="Use secure password hashing (bcrypt, scrypt, or Argon2)"
            ))
        
        return vulnerabilities
    
    async def _analyze_with_ai(self, code: str, context: str) -> List[SecurityVulnerability]:
        """AI-powered security analysis"""
        vulnerabilities = []
        
        try:
            # Import here to avoid circular imports
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models.hf_models import HuggingFaceCodeModels
            
            hf_models = HuggingFaceCodeModels()
            
            # Perform AI analysis
            result = await hf_models.analyze_code(
                code=code,
                task_type="security_analysis",
                context=f"Security analysis context: {context}"
            )
            
            # Parse AI findings into vulnerabilities
            if result.issues_found:
                for issue in result.issues_found:
                    # Map AI findings to vulnerability format
                    severity_map = {
                        'high': SeverityLevel.HIGH,
                        'medium': SeverityLevel.MEDIUM,
                        'low': SeverityLevel.LOW,
                        'critical': SeverityLevel.CRITICAL
                    }
                    
                    vulnerability = SecurityVulnerability(
                        vuln_type=VulnerabilityType.INPUT_VALIDATION,  # Default type
                        severity=severity_map.get(issue.get('severity', 'medium'), SeverityLevel.MEDIUM),
                        title="AI-Detected Security Issue",
                        description=issue.get('message', 'Security issue detected by AI analysis'),
                        line_number=issue.get('line_number'),
                        confidence=result.confidence_score,
                        recommendation="Review AI recommendations and apply appropriate security measures"
                    )
                    vulnerabilities.append(vulnerability)
            
        except Exception as e:
            logger.warning(f"AI security analysis failed: {e}")
        
        return vulnerabilities
    
    def _get_recommendation(self, vuln_type: VulnerabilityType) -> str:
        """Get security recommendation for vulnerability type"""
        recommendations = self.SECURITY_RECOMMENDATIONS.get(vuln_type, [])
        return recommendations[0] if recommendations else "Review and apply security best practices"
    
    def _compile_results(self, vulnerabilities: List[SecurityVulnerability], start_time: datetime) -> SecurityAnalysisResult:
        """Compile final security analysis results"""
        
        # Count vulnerabilities by severity
        critical_count = sum(1 for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for v in vulnerabilities if v.severity == SeverityLevel.HIGH)
        medium_count = sum(1 for v in vulnerabilities if v.severity == SeverityLevel.MEDIUM)
        low_count = sum(1 for v in vulnerabilities if v.severity == SeverityLevel.LOW)
        
        total_issues = len(vulnerabilities)
        
        # Calculate security score (0-10, higher is better)
        if total_issues == 0:
            security_score = 10.0
        else:
            # Weighted scoring based on severity
            severity_weight = {
                SeverityLevel.CRITICAL: 4.0,
                SeverityLevel.HIGH: 2.0,
                SeverityLevel.MEDIUM: 1.0,
                SeverityLevel.LOW: 0.5
            }
            
            total_weight = sum(severity_weight.get(v.severity, 1.0) for v in vulnerabilities)
            security_score = max(0.0, 10.0 - (total_weight * 0.5))
        
        # Generate recommendations
        recommendations = []
        vuln_types = set(v.vuln_type for v in vulnerabilities)
        for vuln_type in vuln_types:
            recommendations.extend(self.SECURITY_RECOMMENDATIONS.get(vuln_type, []))
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        # Generate compliant practices
        compliant_practices = []
        if critical_count == 0:
            compliant_practices.append("No critical security vulnerabilities detected")
        if high_count == 0:
            compliant_practices.append("No high-severity security issues found")
        if total_issues == 0:
            compliant_practices.append("Code passes basic security validation")
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        return SecurityAnalysisResult(
            vulnerabilities=vulnerabilities,
            security_score=security_score,
            total_issues=total_issues,
            critical_issues=critical_count,
            high_issues=high_count,
            medium_issues=medium_count,
            low_issues=low_count,
            analysis_time=analysis_time,
            recommendations=recommendations[:10],  # Limit to top 10
            compliant_practices=compliant_practices
        )
    
    def _update_stats(self, result: SecurityAnalysisResult):
        """Update analyzer statistics"""
        self.analysis_stats['total_scans'] += 1
        self.analysis_stats['vulnerabilities_found'] += result.total_issues
        
        # Update average security score
        current_avg = self.analysis_stats['avg_security_score']
        total_scans = self.analysis_stats['total_scans']
        
        if total_scans == 1:
            self.analysis_stats['avg_security_score'] = result.security_score
        else:
            self.analysis_stats['avg_security_score'] = (
                (current_avg * (total_scans - 1) + result.security_score) / total_scans
            )
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get security analyzer statistics"""
        return {
            **self.analysis_stats,
            'supported_vulnerability_types': len(self.SECURITY_PATTERNS),
            'total_patterns': sum(len(patterns) for patterns in self.SECURITY_PATTERNS.values())
        }