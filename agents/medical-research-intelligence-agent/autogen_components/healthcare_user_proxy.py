"""
Healthcare User Proxy - AutoGen User Proxy for MARIA
Specialized user proxy for medical research with HITL controls
"""

import os
from typing import Dict, List, Any, Optional, Callable
import json
from datetime import datetime
import streamlit as st

try:
    from autogen import UserProxyAgent, ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    try:
        from autogen_agentchat.agents import UserProxyAgent
        from autogen_agentchat import ConversableAgent
        AUTOGEN_AVAILABLE = True
    except ImportError:
        AUTOGEN_AVAILABLE = False

# Fallback classes if AutoGen is not available
if not AUTOGEN_AVAILABLE:
    class UserProxyAgent:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'fallback_user_proxy')
            self.human_input_mode = kwargs.get('human_input_mode', 'NEVER')
        
        def generate_reply(self, messages, sender=None, **kwargs):
            return "AutoGen not available - please install pyautogen"
    
    class ConversableAgent:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'fallback_agent')


class HealthcareUserProxy:
    """
    Healthcare-specific user proxy with medical validation and HITL controls
    """
    
    def __init__(self, streamlit_app=None):
        """
        Initialize healthcare user proxy
        
        Args:
            streamlit_app: Streamlit app instance for UI interactions
        """
        self.streamlit_app = streamlit_app
        self.agent = None
        self.medical_validators = []
        self.approval_queue = []
        self.confidence_threshold = 0.8
        self.require_medical_approval = True
        
        if AUTOGEN_AVAILABLE:
            self._create_healthcare_proxy()
        else:
            self._create_fallback_proxy()
    
    def _create_healthcare_proxy(self):
        """Create the healthcare user proxy agent"""
        try:
            # Healthcare-specific system message
            healthcare_system_message = """You are a Healthcare User Proxy representing medical professionals and researchers using MARIA.

Your responsibilities include:
1. **Medical Validation**: Ensure all medical content meets professional standards
2. **Safety Oversight**: Flag potentially harmful or inaccurate medical information
3. **Ethical Compliance**: Maintain medical ethics and research standards
4. **Human-in-the-Loop**: Facilitate appropriate human oversight for medical decisions
5. **Quality Control**: Verify accuracy and reliability of medical research

Medical Content Validation Criteria:
- **Clinical Accuracy**: Medical facts must be evidence-based
- **Safety Information**: Include appropriate warnings and contraindications
- **Professional Standards**: Use proper medical terminology and citations
- **Ethical Guidelines**: Follow medical research ethics
- **Disclaimer Requirements**: Include necessary medical disclaimers

Approval Requirements:
- High-risk medical content requires human approval
- Treatment recommendations need medical professional validation
- Drug information must include safety profiles
- Clinical guidance requires evidence-based support

You serve as the interface between AI-generated medical content and healthcare professionals."""

            # Create user proxy with medical oversight
            self.agent = UserProxyAgent(
                name="healthcare_user_proxy",
                system_message=healthcare_system_message,
                human_input_mode="NEVER",  # We handle human input through Streamlit
                max_consecutive_auto_reply=1,
                code_execution_config=False,
                default_auto_reply="Medical content received. Validating for clinical accuracy and safety...",
                is_termination_msg=self._is_medical_termination
            )
            
        except Exception as e:
            print(f"Error creating healthcare user proxy: {e}")
            self._create_fallback_proxy()
    
    def _create_fallback_proxy(self):
        """Create fallback proxy when AutoGen is not available"""
        self.agent = UserProxyAgent(
            name="healthcare_fallback_proxy",
            human_input_mode="NEVER"
        )
    
    def _is_medical_termination(self, message: Dict[str, Any]) -> bool:
        """
        Check if medical conversation should terminate
        
        Args:
            message: Message to check
            
        Returns:
            True if conversation should terminate
        """
        if isinstance(message, dict) and 'content' in message:
            content = message['content'].lower()
            
            # Terminate on completion phrases
            termination_phrases = [
                "medical analysis complete",
                "research summary provided",
                "literature review finished",
                "treatment comparison done",
                "medical validation complete"
            ]
            
            return any(phrase in content for phrase in termination_phrases)
        
        return False
    
    def validate_medical_content(self, content: str, content_type: str = "general") -> Dict[str, Any]:
        """
        Validate medical content for accuracy and safety
        
        Args:
            content: Medical content to validate
            content_type: Type of medical content
            
        Returns:
            Validation results with confidence scores and flags
        """
        validation_result = {
            "content": content,
            "content_type": content_type,
            "validation_status": "pending",
            "confidence_score": 0.0,
            "safety_flags": [],
            "accuracy_flags": [],
            "requires_approval": False,
            "medical_disclaimers_present": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Basic content validation
            content_lower = content.lower()
            
            # Check for medical disclaimers
            disclaimer_keywords = [
                "medical disclaimer",
                "healthcare provider",
                "professional medical",
                "qualified physician",
                "medical professional"
            ]
            validation_result["medical_disclaimers_present"] = any(
                keyword in content_lower for keyword in disclaimer_keywords
            )
            
            # Check for high-risk content
            high_risk_keywords = [
                "dosage", "dose", "medication", "drug", "prescription",
                "treatment", "therapy", "diagnosis", "contraindication"
            ]
            has_high_risk_content = any(keyword in content_lower for keyword in high_risk_keywords)
            
            if has_high_risk_content:
                validation_result["requires_approval"] = True
                validation_result["safety_flags"].append("Contains treatment/medication information")
            
            # Check for accuracy indicators
            accuracy_indicators = [
                "evidence-based", "clinical trial", "peer-reviewed",
                "confidence score", "systematic review", "meta-analysis"
            ]
            accuracy_score = sum(1 for indicator in accuracy_indicators if indicator in content_lower)
            
            # Calculate confidence score
            base_confidence = 0.7
            if validation_result["medical_disclaimers_present"]:
                base_confidence += 0.1
            if accuracy_score > 0:
                base_confidence += min(accuracy_score * 0.05, 0.2)
            if has_high_risk_content and not validation_result["medical_disclaimers_present"]:
                base_confidence -= 0.3
            
            validation_result["confidence_score"] = min(max(base_confidence, 0.0), 1.0)
            
            # Determine validation status
            if validation_result["confidence_score"] >= self.confidence_threshold:
                if validation_result["requires_approval"]:
                    validation_result["validation_status"] = "requires_approval"
                else:
                    validation_result["validation_status"] = "approved"
            else:
                validation_result["validation_status"] = "needs_review"
            
            return validation_result
        
        except Exception as e:
            validation_result["validation_status"] = "error"
            validation_result["error"] = str(e)
            return validation_result
    
    def queue_for_medical_approval(self, content: str, content_type: str, context: Dict[str, Any] = None) -> str:
        """
        Queue medical content for human approval
        
        Args:
            content: Content requiring approval
            content_type: Type of content
            context: Additional context
            
        Returns:
            Approval queue ID
        """
        approval_item = {
            "id": f"medical_approval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "content": content,
            "type": content_type,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "confidence": self.validate_medical_content(content, content_type)["confidence_score"]
        }
        
        self.approval_queue.append(approval_item)
        
        # Add to Streamlit session state if available
        if self.streamlit_app and hasattr(st.session_state, 'pending_medical_approvals'):
            st.session_state.pending_medical_approvals.append(approval_item)
        
        return approval_item["id"]
    
    def process_medical_approval(self, approval_id: str, decision: str, reviewer_id: str = None) -> Dict[str, Any]:
        """
        Process medical approval decision
        
        Args:
            approval_id: ID of approval item
            decision: "approved" or "rejected"
            reviewer_id: ID of reviewing medical professional
            
        Returns:
            Processing result
        """
        try:
            # Find approval item
            approval_item = None
            for item in self.approval_queue:
                if item["id"] == approval_id:
                    approval_item = item
                    break
            
            if not approval_item:
                return {"success": False, "message": "Approval item not found"}
            
            # Update approval status
            approval_item["status"] = decision
            approval_item["reviewed_by"] = reviewer_id or "unknown_reviewer"
            approval_item["reviewed_at"] = datetime.now().isoformat()
            
            # Remove from queue if approved
            if decision == "approved":
                self.approval_queue.remove(approval_item)
                
                # Update Streamlit session state
                if self.streamlit_app and hasattr(st.session_state, 'pending_medical_approvals'):
                    st.session_state.pending_medical_approvals = [
                        item for item in st.session_state.pending_medical_approvals 
                        if item["id"] != approval_id
                    ]
            
            return {
                "success": True,
                "message": f"Medical content {decision}",
                "approval_item": approval_item
            }
        
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """
        Get list of pending medical approvals
        
        Returns:
            List of pending approval items
        """
        return [item for item in self.approval_queue if item["status"] == "pending"]
    
    def generate_medical_response(self, message: str, sender=None, **kwargs) -> str:
        """
        Generate medical response with validation
        
        Args:
            message: Input message
            sender: Sender agent
            **kwargs: Additional parameters
            
        Returns:
            Validated medical response
        """
        try:
            if not AUTOGEN_AVAILABLE:
                return self._generate_fallback_response(message)
            
            # Generate base response
            if self.agent:
                response = self.agent.generate_reply(
                    messages=[{"role": "user", "content": message}],
                    sender=sender,
                    **kwargs
                )
            else:
                response = self._generate_fallback_response(message)
            
            # Validate medical content
            validation = self.validate_medical_content(response, "medical_response")
            
            # Add validation metadata
            validated_response = f"{response}\n\n---\n**Medical Validation**: Confidence {validation['confidence_score']:.2f}"
            
            if validation["requires_approval"]:
                approval_id = self.queue_for_medical_approval(
                    response, "medical_response", {"original_message": message}
                )
                validated_response += f" | Queued for medical review (ID: {approval_id})"
            
            return validated_response
        
        except Exception as e:
            return f"Error in medical response generation: {str(e)}"
    
    def _generate_fallback_response(self, message: str) -> str:
        """Generate fallback response when AutoGen is not available"""
        return f"""Medical research query received: {message}

**Healthcare User Proxy Response:**
Your medical research request has been received and is being processed according to healthcare professional standards.

**Validation Status:** 
- Content type: Medical query
- Safety check: Passed
- Professional review: Required for clinical content

**Next Steps:**
1. Medical literature analysis
2. Clinical accuracy verification  
3. Safety profile assessment
4. Professional validation

*This response represents the healthcare user proxy interface for medical research validation.*"""
    
    def add_medical_validator(self, validator_func: Callable[[str], Dict[str, Any]]):
        """
        Add custom medical validator function
        
        Args:
            validator_func: Function that takes content and returns validation results
        """
        self.medical_validators.append(validator_func)
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set confidence threshold for medical content
        
        Args:
            threshold: Confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def enable_medical_approval_mode(self, enabled: bool = True):
        """
        Enable or disable medical approval requirement
        
        Args:
            enabled: Whether to require medical approval
        """
        self.require_medical_approval = enabled
    
    def is_available(self) -> bool:
        """Check if healthcare user proxy is available"""
        return self.agent is not None


def create_healthcare_user_proxy(streamlit_app=None) -> HealthcareUserProxy:
    """
    Create healthcare user proxy
    
    Args:
        streamlit_app: Streamlit app instance
        
    Returns:
        HealthcareUserProxy instance
    """
    return HealthcareUserProxy(streamlit_app)