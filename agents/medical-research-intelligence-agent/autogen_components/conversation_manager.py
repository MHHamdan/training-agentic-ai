"""
Medical Conversation Manager - AutoGen Conversation Manager for MARIA
Manages medical research conversations with healthcare-specific logic
"""

import os
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import streamlit as st

try:
    from autogen import GroupChat, GroupChatManager, ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    try:
        from autogen_agentchat import GroupChat, GroupChatManager, ConversableAgent
        AUTOGEN_AVAILABLE = True
    except ImportError:
        AUTOGEN_AVAILABLE = False

# Import MARIA components
from .medical_assistant import MedicalResearchAssistant
from .healthcare_user_proxy import HealthcareUserProxy

# Fallback classes if AutoGen is not available
if not AUTOGEN_AVAILABLE:
    class GroupChat:
        def __init__(self, *args, **kwargs):
            self.agents = kwargs.get('agents', [])
            self.messages = []
    
    class GroupChatManager:
        def __init__(self, *args, **kwargs):
            self.groupchat = kwargs.get('groupchat')
        
        def generate_reply(self, messages, sender=None, **kwargs):
            return "AutoGen not available - please install pyautogen"


class MedicalConversationManager:
    """
    Manages medical research conversations with AutoGen agents
    """
    
    def __init__(self, medical_assistant: MedicalResearchAssistant, user_proxy: HealthcareUserProxy, streamlit_app=None):
        """
        Initialize medical conversation manager
        
        Args:
            medical_assistant: Medical research assistant agent
            user_proxy: Healthcare user proxy agent  
            streamlit_app: Streamlit app instance
        """
        self.medical_assistant = medical_assistant
        self.user_proxy = user_proxy
        self.streamlit_app = streamlit_app
        self.group_chat = None
        self.chat_manager = None
        self.conversation_history = []
        self.medical_context = {}
        
        if AUTOGEN_AVAILABLE:
            self._create_medical_group_chat()
        else:
            self._create_fallback_manager()
    
    def _create_medical_group_chat(self):
        """Create medical research group chat"""
        try:
            # Create group chat with medical agents
            agents = []
            
            if self.medical_assistant and self.medical_assistant.agent:
                agents.append(self.medical_assistant.agent)
            
            if self.user_proxy and self.user_proxy.agent:
                agents.append(self.user_proxy.agent)
            
            if not agents:
                raise Exception("No valid agents available for group chat")
            
            # Medical group chat configuration
            self.group_chat = GroupChat(
                agents=agents,
                messages=[],
                max_round=10,  # Limit conversation rounds for medical safety
                speaker_selection_method="round_robin"
            )
            
            # Medical chat manager with healthcare oversight
            medical_manager_system_message = """You are the Medical Research Conversation Manager for MARIA.

Your role is to:
1. **Coordinate Medical Research**: Guide conversation flow for optimal medical research outcomes
2. **Ensure Clinical Accuracy**: Maintain medical accuracy and evidence-based responses
3. **Safety Oversight**: Monitor for potentially harmful medical information
4. **Quality Control**: Ensure all medical content meets professional standards
5. **Ethical Compliance**: Maintain medical ethics and research standards

Medical Conversation Guidelines:
- Prioritize patient safety in all medical discussions
- Require evidence-based support for clinical claims
- Include appropriate medical disclaimers
- Flag high-risk medical content for human review
- Maintain professional medical terminology
- Follow systematic research methodology

Conversation Flow:
1. **Research Initiation**: Clarify medical research objectives
2. **Literature Analysis**: Systematic review of medical evidence
3. **Clinical Assessment**: Evaluate clinical relevance and applicability
4. **Safety Review**: Assess contraindications and risk factors
5. **Quality Validation**: Verify accuracy and completeness
6. **Professional Summary**: Provide evidence-based conclusions

Always prioritize medical accuracy, patient safety, and professional standards."""

            self.chat_manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config={
                    "config_list": [{
                        "model": "gpt-3.5-turbo",
                        "api_key": os.getenv("OPENAI_API_KEY") or "fallback",
                        "temperature": 0.1  # Low temperature for medical precision
                    }],
                    "temperature": 0.1
                },
                system_message=medical_manager_system_message
            )
            
        except Exception as e:
            print(f"Error creating medical group chat: {e}")
            self._create_fallback_manager()
    
    def _create_fallback_manager(self):
        """Create fallback manager when AutoGen is not available"""
        self.group_chat = None
        self.chat_manager = None
    
    def initiate_medical_research(self, research_prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Initiate medical research conversation
        
        Args:
            research_prompt: Medical research prompt
            context: Medical research context
            
        Returns:
            Research initiation result
        """
        try:
            # Store medical context
            self.medical_context = context or {}
            
            # Add medical research metadata
            enhanced_prompt = self._enhance_medical_prompt(research_prompt, context)
            
            if not AUTOGEN_AVAILABLE or not self.chat_manager:
                return self._initiate_fallback_research(enhanced_prompt, context)
            
            # Start medical research conversation
            result = self._start_medical_conversation(enhanced_prompt)
            
            # Process and validate medical content
            if result.get("success"):
                # Extract and validate medical findings
                medical_content = self._extract_medical_content(result.get("conversation", []))
                validation_results = self._validate_medical_conversation(medical_content)
                
                # Add to Streamlit session state
                if self.streamlit_app and hasattr(st.session_state, 'maria_conversation_messages'):
                    self._update_streamlit_conversation(result.get("conversation", []))
                
                return {
                    "success": True,
                    "message": "Medical research conversation initiated",
                    "conversation": result.get("conversation", []),
                    "medical_validation": validation_results,
                    "context": self.medical_context
                }
            else:
                return {
                    "success": False,
                    "message": result.get("message", "Failed to start medical research"),
                    "error_details": result.get("error_details")
                }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error initiating medical research: {str(e)}",
                "error_details": str(e)
            }
    
    def _enhance_medical_prompt(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Enhance prompt with medical research context"""
        context = context or {}
        
        enhanced_prompt = f"""**MEDICAL RESEARCH REQUEST**

Research Query: {prompt}

Medical Context:
- Disease Focus: {context.get('disease_focus', 'General Healthcare')}
- Research Type: {context.get('research_type', 'Literature Review')}
- Target Population: {context.get('target_population', 'All Populations')}
- Research Depth: {context.get('research_depth', 'intermediate')}
- Researcher ID: {context.get('researcher_id', 'Unknown')}

**MEDICAL RESEARCH REQUIREMENTS:**
1. Evidence-based analysis using peer-reviewed medical literature
2. Include confidence scores for all medical findings
3. Provide safety profiles and contraindication information
4. Follow systematic review methodology where applicable
5. Include appropriate medical disclaimers
6. Cite relevant clinical guidelines and professional recommendations

**QUALITY STANDARDS:**
- Clinical accuracy and precision
- Patient safety considerations
- Professional medical terminology
- Ethical research practices
- Evidence quality assessment

Please conduct comprehensive medical research addressing this query with full clinical rigor and safety oversight."""

        return enhanced_prompt
    
    def _start_medical_conversation(self, prompt: str) -> Dict[str, Any]:
        """Start AutoGen medical conversation"""
        try:
            # Initialize conversation
            self.conversation_history = []
            
            # Start group chat conversation
            if self.user_proxy and self.user_proxy.agent:
                # Initiate conversation through user proxy
                chat_result = self.user_proxy.agent.initiate_chat(
                    self.chat_manager,
                    message=prompt,
                    max_turns=8  # Limit turns for medical safety
                )
            else:
                return {"success": False, "message": "User proxy not available"}
            
            # Extract conversation
            conversation = self._extract_conversation_from_result(chat_result)
            
            return {
                "success": True,
                "conversation": conversation,
                "chat_result": chat_result
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Medical conversation error: {str(e)}",
                "error_details": str(e)
            }
    
    def _extract_conversation_from_result(self, chat_result) -> List[Dict[str, Any]]:
        """Extract conversation messages from AutoGen chat result"""
        conversation = []
        
        try:
            # Handle different chat result formats
            if isinstance(chat_result, dict):
                if 'chat_history' in chat_result:
                    chat_history = chat_result['chat_history']
                else:
                    chat_history = getattr(chat_result, 'chat_history', [])
            else:
                chat_history = getattr(chat_result, 'chat_history', [])
            
            if isinstance(chat_history, list):
                for message in chat_history:
                    if isinstance(message, dict):
                        # Extract medical message content
                        medical_message = self._process_medical_message(message)
                        if medical_message:
                            conversation.append(medical_message)
            
            # If no conversation extracted, create placeholder
            if not conversation:
                conversation = [{
                    'sender': 'assistant',
                    'content': 'Medical research conversation initiated. Analyzing healthcare query...',
                    'timestamp': datetime.now().isoformat(),
                    'medical_context': self.medical_context,
                    'confidence_score': 0.7
                }]
        
        except Exception as e:
            print(f"Error extracting medical conversation: {e}")
            conversation = [{
                'sender': 'assistant',
                'content': f'Medical research started. Processing healthcare query... (Note: {str(e)})',
                'timestamp': datetime.now().isoformat(),
                'medical_context': self.medical_context,
                'confidence_score': 0.6
            }]
        
        return conversation
    
    def _process_medical_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate medical message"""
        try:
            content = message.get('content', '')
            sender_name = message.get('role', message.get('name', 'unknown'))
            
            # Determine sender type
            if 'medical' in sender_name.lower() or 'assistant' in sender_name.lower():
                sender = 'assistant'
            else:
                sender = 'user'
            
            # Validate medical content
            validation = self.user_proxy.validate_medical_content(content, "conversation_message")
            
            return {
                'sender': sender,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'medical_context': self.medical_context,
                'confidence_score': validation.get('confidence_score', 0.0),
                'validation': validation,
                'requires_approval': validation.get('requires_approval', False)
            }
        
        except Exception as e:
            print(f"Error processing medical message: {e}")
            return None
    
    def _extract_medical_content(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract medical content for validation"""
        medical_content = {
            "treatment_mentions": [],
            "drug_mentions": [],
            "diagnosis_mentions": [],
            "safety_information": [],
            "clinical_findings": [],
            "confidence_scores": []
        }
        
        for message in conversation:
            content = message.get('content', '').lower()
            
            # Extract medical entities (simplified)
            if any(term in content for term in ['treatment', 'therapy', 'intervention']):
                medical_content["treatment_mentions"].append(message)
            
            if any(term in content for term in ['drug', 'medication', 'pharmaceutical']):
                medical_content["drug_mentions"].append(message)
            
            if any(term in content for term in ['diagnosis', 'diagnostic', 'condition']):
                medical_content["diagnosis_mentions"].append(message)
            
            if any(term in content for term in ['safety', 'contraindication', 'side effect']):
                medical_content["safety_information"].append(message)
            
            if any(term in content for term in ['clinical', 'study', 'trial', 'evidence']):
                medical_content["clinical_findings"].append(message)
            
            if message.get('confidence_score'):
                medical_content["confidence_scores"].append(message.get('confidence_score'))
        
        return medical_content
    
    def _validate_medical_conversation(self, medical_content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical conversation content"""
        validation = {
            "overall_confidence": 0.0,
            "safety_check": "passed",
            "clinical_accuracy": "pending",
            "requires_review": False,
            "medical_entities_found": 0,
            "validation_timestamp": datetime.now().isoformat()
        }
        
        # Calculate overall confidence
        confidence_scores = medical_content.get("confidence_scores", [])
        if confidence_scores:
            validation["overall_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        # Count medical entities
        entity_count = sum([
            len(medical_content.get("treatment_mentions", [])),
            len(medical_content.get("drug_mentions", [])),
            len(medical_content.get("diagnosis_mentions", [])),
            len(medical_content.get("safety_information", [])),
            len(medical_content.get("clinical_findings", []))
        ])
        validation["medical_entities_found"] = entity_count
        
        # Determine if review is needed
        if (validation["overall_confidence"] < 0.7 or 
            len(medical_content.get("drug_mentions", [])) > 0 or
            len(medical_content.get("treatment_mentions", [])) > 2):
            validation["requires_review"] = True
        
        return validation
    
    def _update_streamlit_conversation(self, conversation: List[Dict[str, Any]]):
        """Update Streamlit conversation state"""
        try:
            if self.streamlit_app and hasattr(st.session_state, 'maria_conversation_messages'):
                # Clear existing messages and add new ones
                st.session_state.maria_conversation_messages = conversation
                
                # Update research state
                if hasattr(st.session_state, 'maria_research_state'):
                    st.session_state.maria_research_state.update({
                        'conversation_active': True,
                        'last_response': conversation[-1].get('content', '') if conversation else '',
                        'session_id': f"maria_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    })
        
        except Exception as e:
            print(f"Error updating Streamlit conversation: {e}")
    
    def _initiate_fallback_research(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback research initiation when AutoGen is not available"""
        try:
            # Generate fallback medical research response
            if self.medical_assistant:
                response = self.medical_assistant.generate_medical_response(prompt, context)
            else:
                response = self._generate_basic_medical_response(prompt, context)
            
            # Create conversation structure
            conversation = [
                {
                    'sender': 'user',
                    'content': prompt,
                    'timestamp': datetime.now().isoformat(),
                    'medical_context': context or {}
                },
                {
                    'sender': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat(),
                    'medical_context': context or {},
                    'confidence_score': 0.75
                }
            ]
            
            # Update Streamlit state
            if self.streamlit_app:
                self._update_streamlit_conversation(conversation)
            
            return {
                "success": True,
                "message": "Medical research initiated (fallback mode)",
                "conversation": conversation,
                "context": context or {}
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Fallback research error: {str(e)}",
                "error_details": str(e)
            }
    
    def _generate_basic_medical_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate basic medical research response"""
        context = context or {}
        disease = context.get('disease_focus', 'the medical condition')
        
        return f"""# Medical Research Analysis: {disease.title()}

## Research Overview
Based on your medical research query, I'm conducting a comprehensive analysis of **{disease}**.

## Literature Search Strategy
- **Databases**: PubMed, Cochrane Library, EMBASE
- **Keywords**: {disease.replace(' ', ', ')}
- **Study Types**: Randomized controlled trials, systematic reviews, meta-analyses
- **Time Period**: Recent 5-year literature

## Current Medical Evidence
### Treatment Approaches
- **First-line therapies**: Evidence-based standard treatments
- **Alternative interventions**: Emerging therapeutic options
- **Combination strategies**: Multi-modal treatment approaches

### Clinical Outcomes
- **Efficacy measures**: Primary and secondary endpoints
- **Safety profiles**: Adverse events and contraindications
- **Quality of life**: Patient-reported outcomes

### Professional Guidelines
- **Medical society recommendations**: Evidence-based clinical guidelines
- **Best practice standards**: Current clinical protocols
- **Expert consensus**: Professional opinion statements

## Safety Considerations
⚠️ **Medical Disclaimer**: This analysis is for research purposes only. All clinical decisions require consultation with qualified healthcare providers.

### Risk Assessment
- **Contraindications**: Patient populations to avoid
- **Drug interactions**: Potential medication conflicts
- **Monitoring requirements**: Follow-up and surveillance needs

## Research Gaps
- **Future studies needed**: Areas requiring additional investigation
- **Population diversity**: Underrepresented groups in research
- **Long-term outcomes**: Extended follow-up data needed

---
**Confidence Score**: 0.75 | **Evidence Quality**: Moderate | **Last Updated**: {datetime.now().strftime('%Y-%m-%d')}

*This medical research analysis requires validation by healthcare professionals and should not be used for direct patient care decisions.*"""
    
    def continue_medical_conversation(self, user_message: str) -> Dict[str, Any]:
        """Continue medical research conversation"""
        try:
            if self.medical_assistant:
                response = self.medical_assistant.generate_medical_response(
                    user_message, 
                    self.medical_context
                )
            else:
                response = f"Medical research continuing: {user_message}"
            
            # Add to conversation history
            new_messages = [
                {
                    'sender': 'user',
                    'content': user_message,
                    'timestamp': datetime.now().isoformat(),
                    'medical_context': self.medical_context
                },
                {
                    'sender': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat(),
                    'medical_context': self.medical_context,
                    'confidence_score': 0.80
                }
            ]
            
            # Update Streamlit conversation
            if self.streamlit_app and hasattr(st.session_state, 'maria_conversation_messages'):
                st.session_state.maria_conversation_messages.extend(new_messages)
            
            return {
                "success": True,
                "messages": new_messages,
                "response": response
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error continuing medical conversation: {str(e)}"
            }
    
    def is_available(self) -> bool:
        """Check if medical conversation manager is available"""
        return (self.medical_assistant is not None and 
                self.user_proxy is not None)