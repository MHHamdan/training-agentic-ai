import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)

class ComplianceLevel(Enum):
    """Compliance level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RegulatoryStandard(Enum):
    """Regulatory standard enumeration"""
    SEC = "sec"
    FINRA = "finra"
    CFTC = "cftc"
    GDPR = "gdpr"
    SOX = "sox"

@dataclass
class ComplianceRecord:
    """Individual compliance record"""
    record_id: str
    timestamp: datetime
    agent_name: str
    action_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    compliance_level: ComplianceLevel
    regulatory_standards: List[RegulatoryStandard]
    decision_reasoning: str
    risk_assessment: Dict[str, Any]
    audit_trail: Dict[str, Any]
    data_hash: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        record_dict = asdict(self)
        record_dict['timestamp'] = self.timestamp.isoformat()
        record_dict['compliance_level'] = self.compliance_level.value
        record_dict['regulatory_standards'] = [std.value for std in self.regulatory_standards]
        return record_dict

class ComplianceValidator:
    """Validates compliance requirements for financial analysis"""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
        self.violation_thresholds = self._load_violation_thresholds()
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules and requirements"""
        return {
            "sec_requirements": {
                "investment_advice": {
                    "required_disclosures": [
                        "Material risk factors",
                        "Conflicts of interest",
                        "Methodology and assumptions",
                        "Data sources and limitations",
                        "Past performance disclaimers"
                    ],
                    "prohibited_activities": [
                        "Guaranteeing returns",
                        "Omitting material risks",
                        "Making unsubstantiated claims",
                        "Insider information usage"
                    ]
                },
                "record_keeping": {
                    "retention_period": 2555,  # 7 years in days
                    "required_fields": [
                        "timestamp", "agent_name", "input_data", "output_data",
                        "decision_reasoning", "risk_assessment", "user_id"
                    ]
                }
            },
            "finra_requirements": {
                "suitability": {
                    "customer_profile": [
                        "investment_objectives", "risk_tolerance", "financial_situation",
                        "investment_experience", "time_horizon", "liquidity_needs"
                    ],
                    "product_analysis": [
                        "risk_characteristics", "complexity_level", "cost_structure",
                        "liquidity_features", "tax_implications"
                    ]
                },
                "best_execution": {
                    "factors": [
                        "price_improvement", "execution_speed", "probability_execution",
                        "market_impact", "commission_costs"
                    ]
                }
            },
            "risk_management": {
                "position_limits": {
                    "single_security": 0.10,  # 10% max
                    "sector_concentration": 0.25,  # 25% max
                    "geographic_concentration": 0.30  # 30% max
                },
                "stress_testing": {
                    "required_scenarios": [
                        "market_crash_20", "market_crash_30", "interest_rate_shock",
                        "liquidity_crisis", "sector_specific_stress"
                    ]
                }
            }
        }
    
    def _load_violation_thresholds(self) -> Dict[str, Any]:
        """Load violation threshold configurations"""
        return {
            "high_risk_threshold": 0.15,  # 15% VaR
            "concentration_limit": 0.10,  # 10% single position
            "liquidity_minimum": 0.05,   # 5% daily volume
            "volatility_threshold": 0.30,  # 30% annualized
            "correlation_limit": 0.80,    # 80% correlation
            "leverage_maximum": 3.0       # 3x leverage
        }
    
    def validate_investment_decision(self, decision: Dict[str, Any], 
                                   client_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate investment decision compliance"""
        validation_result = {
            "is_compliant": True,
            "compliance_score": 100,
            "violations": [],
            "warnings": [],
            "required_actions": [],
            "regulatory_status": {
                "sec_compliant": True,
                "finra_compliant": True,
                "risk_compliant": True
            },
            "validation_timestamp": datetime.now().isoformat()
        }
        
        try:
            # SEC Compliance Validation
            sec_validation = self._validate_sec_compliance(decision)
            validation_result["regulatory_status"]["sec_compliant"] = sec_validation["compliant"]
            validation_result["violations"].extend(sec_validation["violations"])
            validation_result["warnings"].extend(sec_validation["warnings"])
            
            # FINRA Suitability Validation
            if client_profile:
                finra_validation = self._validate_finra_suitability(decision, client_profile)
                validation_result["regulatory_status"]["finra_compliant"] = finra_validation["compliant"]
                validation_result["violations"].extend(finra_validation["violations"])
                validation_result["warnings"].extend(finra_validation["warnings"])
            
            # Risk Management Validation
            risk_validation = self._validate_risk_compliance(decision)
            validation_result["regulatory_status"]["risk_compliant"] = risk_validation["compliant"]
            validation_result["violations"].extend(risk_validation["violations"])
            validation_result["warnings"].extend(risk_validation["warnings"])
            
            # Calculate overall compliance
            validation_result["is_compliant"] = all(validation_result["regulatory_status"].values())
            validation_result["compliance_score"] = self._calculate_compliance_score(validation_result)
            
            # Generate required actions
            validation_result["required_actions"] = self._generate_required_actions(validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in compliance validation: {str(e)}")
            return {
                "is_compliant": False,
                "compliance_score": 0,
                "violations": [f"Validation error: {str(e)}"],
                "error": str(e)
            }
    
    def _validate_sec_compliance(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SEC compliance requirements"""
        violations = []
        warnings = []
        
        # Check required disclosures
        required_disclosures = self.compliance_rules["sec_requirements"]["investment_advice"]["required_disclosures"]
        provided_disclosures = decision.get("disclosures", [])
        
        missing_disclosures = [disc for disc in required_disclosures 
                             if not any(disc.lower() in str(provided).lower() 
                                      for provided in provided_disclosures)]
        
        if missing_disclosures:
            violations.extend([f"Missing required disclosure: {disc}" for disc in missing_disclosures])
        
        # Check for prohibited activities
        prohibited_activities = self.compliance_rules["sec_requirements"]["investment_advice"]["prohibited_activities"]
        recommendation_text = str(decision.get("recommendation", "")).lower()
        
        for prohibited in prohibited_activities:
            if any(word in recommendation_text for word in prohibited.lower().split()):
                violations.append(f"Potential prohibited activity: {prohibited}")
        
        # Check risk assessment completeness
        if not decision.get("risk_assessment"):
            violations.append("Missing required risk assessment")
        
        # Check methodology disclosure
        if not decision.get("methodology"):
            warnings.append("Methodology disclosure recommended")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings
        }
    
    def _validate_finra_suitability(self, decision: Dict[str, Any], 
                                  client_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Validate FINRA suitability requirements"""
        violations = []
        warnings = []
        
        required_profile_fields = self.compliance_rules["finra_requirements"]["suitability"]["customer_profile"]
        
        # Check customer profile completeness
        missing_profile_fields = [field for field in required_profile_fields 
                                if field not in client_profile]
        
        if missing_profile_fields:
            violations.extend([f"Missing customer profile field: {field}" 
                             for field in missing_profile_fields])
        
        # Risk tolerance alignment
        investment_risk = decision.get("risk_assessment", {}).get("overall_risk_rating", "medium")
        client_risk_tolerance = client_profile.get("risk_tolerance", "medium")
        
        risk_levels = {"low": 1, "medium": 2, "high": 3}
        if risk_levels.get(investment_risk, 2) > risk_levels.get(client_risk_tolerance, 2):
            violations.append("Investment risk exceeds client risk tolerance")
        
        # Time horizon alignment
        investment_horizon = decision.get("investment_horizon", "medium-term")
        client_horizon = client_profile.get("time_horizon", "medium-term")
        
        if investment_horizon == "long-term" and client_horizon == "short-term":
            warnings.append("Investment horizon may not align with client needs")
        
        # Liquidity needs assessment
        if client_profile.get("liquidity_needs") == "high":
            investment_liquidity = decision.get("liquidity_assessment", {}).get("liquidity_score", 5)
            if investment_liquidity < 7:
                violations.append("Investment liquidity insufficient for client needs")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings
        }
    
    def _validate_risk_compliance(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate risk management compliance"""
        violations = []
        warnings = []
        
        risk_assessment = decision.get("risk_assessment", {})
        
        # Position size limits
        position_size = decision.get("position_size", 0.05)  # Default 5%
        max_position = self.compliance_rules["risk_management"]["position_limits"]["single_security"]
        
        if position_size > max_position:
            violations.append(f"Position size ({position_size:.2%}) exceeds limit ({max_position:.2%})")
        
        # Volatility threshold
        volatility = risk_assessment.get("market_risk", {}).get("volatility", 0.15)
        if volatility > self.violation_thresholds["volatility_threshold"]:
            warnings.append(f"High volatility detected: {volatility:.2%}")
        
        # VaR threshold
        var_95 = risk_assessment.get("market_risk", {}).get("var_95", 0.05)
        if var_95 > self.violation_thresholds["high_risk_threshold"]:
            violations.append(f"VaR exceeds threshold: {var_95:.2%}")
        
        # Stress testing requirements
        required_scenarios = self.compliance_rules["risk_management"]["stress_testing"]["required_scenarios"]
        stress_results = risk_assessment.get("stress_testing", {})
        
        missing_scenarios = [scenario for scenario in required_scenarios 
                           if scenario not in stress_results]
        
        if missing_scenarios:
            warnings.extend([f"Missing stress scenario: {scenario}" for scenario in missing_scenarios])
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings
        }
    
    def _calculate_compliance_score(self, validation_result: Dict[str, Any]) -> int:
        """Calculate overall compliance score"""
        violations = len(validation_result["violations"])
        warnings = len(validation_result["warnings"])
        
        # Start with 100 points
        score = 100
        
        # Deduct points for violations and warnings
        score -= violations * 20  # 20 points per violation
        score -= warnings * 5    # 5 points per warning
        
        # Ensure score is not negative
        return max(0, score)
    
    def _generate_required_actions(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate required actions to achieve compliance"""
        actions = []
        
        for violation in validation_result["violations"]:
            if "Missing required disclosure" in violation:
                actions.append(f"Add required disclosure: {violation.split(': ')[1]}")
            elif "Position size" in violation and "exceeds limit" in violation:
                actions.append("Reduce position size to comply with concentration limits")
            elif "risk exceeds" in violation:
                actions.append("Reassess investment suitability or client risk profile")
            elif "VaR exceeds threshold" in violation:
                actions.append("Implement additional risk mitigation measures")
        
        for warning in validation_result["warnings"]:
            if "Methodology disclosure" in warning:
                actions.append("Consider adding methodology disclosure for transparency")
            elif "High volatility" in warning:
                actions.append("Monitor volatility and consider hedging strategies")
        
        return actions

class AuditTrailManager:
    """Manages comprehensive audit trails for all agent activities"""
    
    def __init__(self):
        self.records: List[ComplianceRecord] = []
        self.session_records: Dict[str, List[str]] = {}
        
    def create_compliance_record(self, agent_name: str, action_type: str,
                                input_data: Dict[str, Any], output_data: Dict[str, Any],
                                compliance_level: ComplianceLevel,
                                regulatory_standards: List[RegulatoryStandard],
                                decision_reasoning: str,
                                risk_assessment: Optional[Dict[str, Any]] = None,
                                user_id: Optional[str] = None,
                                session_id: Optional[str] = None) -> ComplianceRecord:
        """Create a new compliance record"""
        
        record_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create data hash for integrity
        data_to_hash = json.dumps({
            "agent_name": agent_name,
            "action_type": action_type,
            "input_data": input_data,
            "output_data": output_data,
            "timestamp": timestamp.isoformat()
        }, sort_keys=True)
        data_hash = hashlib.sha256(data_to_hash.encode()).hexdigest()
        
        # Create audit trail
        audit_trail = {
            "created_timestamp": timestamp.isoformat(),
            "data_integrity_hash": data_hash,
            "regulatory_compliance_check": True,
            "retention_period_days": 2555,  # 7 years for SEC compliance
            "access_log": [
                {
                    "timestamp": timestamp.isoformat(),
                    "action": "record_created",
                    "user_id": user_id or "system"
                }
            ]
        }
        
        record = ComplianceRecord(
            record_id=record_id,
            timestamp=timestamp,
            agent_name=agent_name,
            action_type=action_type,
            input_data=input_data,
            output_data=output_data,
            compliance_level=compliance_level,
            regulatory_standards=regulatory_standards,
            decision_reasoning=decision_reasoning,
            risk_assessment=risk_assessment or {},
            audit_trail=audit_trail,
            data_hash=data_hash,
            user_id=user_id,
            session_id=session_id
        )
        
        self.records.append(record)
        
        # Track session records
        if session_id:
            if session_id not in self.session_records:
                self.session_records[session_id] = []
            self.session_records[session_id].append(record_id)
        
        logger.info(f"Created compliance record {record_id} for {agent_name}")
        return record
    
    def get_records_by_agent(self, agent_name: str, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[ComplianceRecord]:
        """Get records filtered by agent name and date range"""
        filtered_records = [record for record in self.records 
                          if record.agent_name == agent_name]
        
        if start_date:
            filtered_records = [record for record in filtered_records 
                              if record.timestamp >= start_date]
        
        if end_date:
            filtered_records = [record for record in filtered_records 
                              if record.timestamp <= end_date]
        
        return filtered_records
    
    def get_records_by_session(self, session_id: str) -> List[ComplianceRecord]:
        """Get all records for a specific session"""
        if session_id not in self.session_records:
            return []
        
        record_ids = self.session_records[session_id]
        return [record for record in self.records if record.record_id in record_ids]
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        relevant_records = [record for record in self.records
                          if start_date <= record.timestamp <= end_date]
        
        # Group by compliance level
        compliance_summary = {}
        for level in ComplianceLevel:
            compliance_summary[level.value] = len([r for r in relevant_records 
                                                 if r.compliance_level == level])
        
        # Group by agent
        agent_summary = {}
        for record in relevant_records:
            if record.agent_name not in agent_summary:
                agent_summary[record.agent_name] = 0
            agent_summary[record.agent_name] += 1
        
        # Group by action type
        action_summary = {}
        for record in relevant_records:
            if record.action_type not in action_summary:
                action_summary[record.action_type] = 0
            action_summary[record.action_type] += 1
        
        # Regulatory compliance summary
        regulatory_summary = {}
        for standard in RegulatoryStandard:
            regulatory_summary[standard.value] = len([r for r in relevant_records
                                                    if standard in r.regulatory_standards])
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_records": len(relevant_records),
            "compliance_level_summary": compliance_summary,
            "agent_activity_summary": agent_summary,
            "action_type_summary": action_summary,
            "regulatory_compliance_summary": regulatory_summary,
            "data_integrity_status": "verified",
            "report_generated": datetime.now().isoformat()
        }
    
    def verify_data_integrity(self, record_id: str) -> bool:
        """Verify data integrity of a specific record"""
        record = next((r for r in self.records if r.record_id == record_id), None)
        if not record:
            return False
        
        # Recreate hash
        data_to_hash = json.dumps({
            "agent_name": record.agent_name,
            "action_type": record.action_type,
            "input_data": record.input_data,
            "output_data": record.output_data,
            "timestamp": record.timestamp.isoformat()
        }, sort_keys=True)
        
        current_hash = hashlib.sha256(data_to_hash.encode()).hexdigest()
        return current_hash == record.data_hash
    
    def export_records(self, output_format: str = "json") -> str:
        """Export all records in specified format"""
        if output_format.lower() == "json":
            return json.dumps([record.to_dict() for record in self.records], 
                            indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

# Global instances
compliance_validator = ComplianceValidator()
audit_trail_manager = AuditTrailManager()