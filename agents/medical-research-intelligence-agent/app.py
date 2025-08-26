"""
MARIA - Medical Research Intelligence Agent
AI-powered healthcare research assistant with AutoGen framework
Built for MediSyn Labs - Specialized in medical literature analysis and treatment comparison
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import MARIA components
try:
    from autogen_components.medical_assistant import create_medical_research_assistant
    from autogen_components.healthcare_user_proxy import create_healthcare_user_proxy
    from autogen_components.conversation_manager import MedicalConversationManager
    from tools.medical_research_tools import get_medical_research_tools
    from tools.export_tools import MedicalReportExporter
    from ui.medical_interface import create_medical_research_interface
    from config.medical_config import get_medical_autogen_config
    MARIA_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ MARIA components not available. Error: {str(e)}")
    MARIA_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="MARIA - Medical Research Intelligence Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for MARIA medical interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #dc2626;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #dc2626 0%, #7c2d12 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .medical-container {
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(220,38,38,0.1);
    }
    .medical-conversation-bubble {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .medical-agent-response {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .medical-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .status-researching {
        background-color: #dc2626;
        color: #ffffff;
    }
    .status-analyzing {
        background-color: #ea580c;
        color: #ffffff;
    }
    .status-complete {
        background-color: #16a34a;
        color: #ffffff;
    }
    .status-pending {
        background-color: #94a3b8;
        color: #1e293b;
    }
    .medical-warning {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


class MARIAApp:
    """Main MARIA application class"""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize medical research session state variables"""
        if 'maria_conversation_messages' not in st.session_state:
            st.session_state.maria_conversation_messages = []
        if 'maria_research_state' not in st.session_state:
            st.session_state.maria_research_state = {
                'researcher_id': 'researcher_001',
                'project_id': '',
                'disease_focus': '',
                'research_type': 'Literature Review',
                'target_population': 'All Populations',
                'current_topic': '',
                'research_depth': 'intermediate',
                'subtopics_generated': [],
                'literature_reviewed': [],
                'treatments_compared': {},
                'conversation_active': False,
                'last_response': '',
                'session_id': None,
                'confidence_scores': {},
                'validated_sources': []
            }
        if 'maria_export_data' not in st.session_state:
            st.session_state.maria_export_data = {}
        if 'pending_medical_approvals' not in st.session_state:
            st.session_state.pending_medical_approvals = []
    
    def run(self):
        """Main application entry point"""
        # Header with medical disclaimer
        st.markdown('<h1 class="main-header">🏥 MARIA - Medical Research Intelligence Agent</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Healthcare Research Assistant • AutoGen Framework • Medical Literature Analysis</p>', unsafe_allow_html=True)
        
        # Medical disclaimer
        self.display_medical_disclaimer()
        
        # Check for API keys
        self.check_medical_api_configuration()
        
        # Main interface
        if MARIA_AVAILABLE:
            self.display_medical_research_interface()
        else:
            self.display_error_state()
    
    def display_medical_disclaimer(self):
        """Display important medical disclaimer"""
        st.markdown("""
        <div class="medical-warning">
            ⚠️ <strong>Medical Research Disclaimer:</strong> MARIA is a research assistance tool for medical professionals and researchers. 
            All AI-generated content requires validation by qualified medical professionals. This tool supports research workflows 
            and does not provide direct medical advice for patient care decisions.
        </div>
        """, unsafe_allow_html=True)
    
    def check_medical_api_configuration(self):
        """Check and display medical API configuration status"""
        api_keys = {
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY"),
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "PUBMED_API_KEY": os.getenv("PUBMED_API_KEY") or os.getenv("PubMed")
        }
        
        available_providers = [k for k, v in api_keys.items() if v]
        
        if not available_providers:
            st.error("❌ No medical research API keys configured. Please add at least one API key to your .env file")
        else:
            # Show primary LLM provider
            if api_keys.get("GOOGLE_API_KEY") or api_keys.get("GEMINI_API_KEY"):
                st.success("🧠 Using **Google Gemini** as primary medical LLM provider")
            elif api_keys.get("OPENAI_API_KEY"):
                st.info("🤖 Using **OpenAI** as medical LLM provider")
            elif api_keys.get("ANTHROPIC_API_KEY"):
                st.info("🧠 Using **Anthropic Claude** as medical LLM provider")
            elif api_keys.get("HUGGINGFACE_API_KEY") or api_keys.get("HF_TOKEN"):
                st.info("🤗 Using **Hugging Face Medical Models** as LLM provider")
            
            # Show medical research APIs
            medical_apis = []
            if api_keys.get("PUBMED_API_KEY"):
                medical_apis.append("PubMed")
            if medical_apis:
                st.success(f"🏥 Medical Research APIs: {', '.join(medical_apis)}")
            else:
                st.info("🏥 Medical Research APIs: Using demo mode (add PubMed API key for real data)")
    
    def display_medical_research_interface(self):
        """Display the main medical research interface"""
        # Sidebar configuration
        with st.sidebar:
            st.header("🔬 Medical Research Configuration")
            
            # Researcher information
            researcher_id = st.text_input(
                "Researcher ID",
                value=st.session_state.maria_research_state['researcher_id'],
                help="Enter your researcher identification"
            )
            
            # Project selection
            project_id = st.selectbox(
                "Research Project",
                ["COVID-19 Treatments", "Cancer Therapies", "Diabetes Management", 
                 "Cardiovascular Research", "Neurological Studies", "Custom Project"],
                help="Select your current research project"
            )
            
            if project_id == "Custom Project":
                project_id = st.text_input("Custom Project Name")
            
            # Medical research focus
            disease_focus = st.selectbox(
                "Primary Disease Focus",
                ["COVID-19", "Cancer", "Diabetes", "Cardiovascular Disease", 
                 "Neurological Disorders", "Infectious Diseases", "Autoimmune Disorders", "Other"],
                help="Select the primary disease or condition being researched"
            )
            
            if disease_focus == "Other":
                disease_focus = st.text_input("Specify Disease/Condition")
            
            # Research type
            research_type = st.selectbox(
                "Research Type",
                ["Literature Review", "Treatment Efficacy Analysis", "Drug Outcome Comparison", 
                 "Clinical Trial Analysis", "Meta-Analysis", "Systematic Review"],
                help="Select the type of medical research"
            )
            
            # Target population
            target_population = st.selectbox(
                "Target Population",
                ["All Populations", "Pediatric (0-18 years)", "Adult (18-65 years)", 
                 "Elderly (65+ years)", "Pregnancy/Lactation", "Immunocompromised"],
                help="Select the target patient population"
            )
            
            # Research depth
            research_depth = st.selectbox(
                "Research Depth",
                ["basic", "intermediate", "comprehensive", "systematic"],
                index=1,
                help="Select the depth of medical literature analysis"
            )
            
            # Advanced medical options
            with st.expander("⚙️ Advanced Medical Options"):
                max_literature = st.slider("Max Literature Sources", 5, 50, 15)
                enable_drug_interaction = st.checkbox("Enable Drug Interaction Checking", value=True)
                enable_clinical_trials = st.checkbox("Include Clinical Trials", value=True)
                confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8)
                require_peer_review = st.checkbox("Require Peer-Reviewed Sources Only", value=True)
            
            # Update session state
            st.session_state.maria_research_state.update({
                'researcher_id': researcher_id,
                'project_id': project_id,
                'disease_focus': disease_focus,
                'research_type': research_type,
                'target_population': target_population,
                'research_depth': research_depth
            })
            
            st.divider()
            
            # Medical action buttons
            col1, col2 = st.columns(2)
            with col1:
                start_research = st.button("🔍 Start Medical Research", type="primary", use_container_width=True)
            with col2:
                reset_session = st.button("🔄 Reset Session", use_container_width=True)
            
            generate_literature = st.button("📚 Generate Literature Review", use_container_width=True)
            compare_treatments = st.button("⚖️ Compare Treatments", use_container_width=True)
            validate_sources = st.button("✅ Validate Medical Sources", use_container_width=True)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Medical research conversation area
            st.markdown('<div class="medical-container">', unsafe_allow_html=True)
            st.header("💬 Medical Research Conversation")
            
            # Display medical conversation
            if st.session_state.maria_conversation_messages:
                self.display_medical_conversation()
            else:
                st.info("💡 Start medical research by entering a healthcare topic and clicking 'Start Medical Research'")
            
            # Medical research input area
            research_query = st.text_area(
                "Medical Research Query:",
                placeholder="e.g., What are the latest treatment outcomes for mRNA COVID-19 vaccines in elderly populations with cardiovascular comorbidities?",
                height=120,
                disabled=not st.session_state.maria_research_state['conversation_active']
            )
            
            if st.button("📤 Submit Medical Query") and research_query:
                self.handle_medical_query(research_query)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Medical research status and controls
            st.header("📊 Medical Research Status")
            self.display_medical_research_status()
            
            # HITL Medical Approval System
            st.header("🔍 Medical Review & Approval")
            self.display_medical_approval_system()
            
            # Export medical reports
            if st.session_state.maria_conversation_messages:
                st.header("📁 Medical Report Export")
                self.display_medical_export_options()
        
        # Handle button actions
        if start_research and (disease_focus or research_query):
            self.start_medical_research(disease_focus, research_type, target_population, research_depth)
        
        if generate_literature:
            self.generate_literature_review()
        
        if compare_treatments:
            self.compare_medical_treatments()
        
        if validate_sources:
            self.validate_medical_sources()
        
        if reset_session:
            self.reset_medical_research_session()
    
    def display_medical_conversation(self):
        """Display the medical research conversation"""
        for i, message in enumerate(st.session_state.maria_conversation_messages):
            if message['sender'] == 'user':
                st.markdown(f'<div class="medical-conversation-bubble"><strong>Researcher:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                confidence = message.get('confidence_score', 0.0)
                confidence_badge = f" (Confidence: {confidence:.2f})" if confidence > 0 else ""
                st.markdown(f'<div class="medical-agent-response"><strong>MARIA:</strong> {message["content"]}{confidence_badge}</div>', unsafe_allow_html=True)
        
        # Auto-scroll to bottom
        if st.session_state.maria_conversation_messages:
            st.empty()
    
    def display_medical_research_status(self):
        """Display current medical research status"""
        state = st.session_state.maria_research_state
        
        # Current research context
        if state['disease_focus']:
            st.markdown(f"**Disease Focus:** {state['disease_focus']}")
        if state['research_type']:
            st.markdown(f"**Research Type:** {state['research_type']}")
        if state['target_population']:
            st.markdown(f"**Population:** {state['target_population']}")
        
        # Medical research progress
        status_items = [
            ("Research Active", state['conversation_active']),
            ("Literature Reviewed", len(state['literature_reviewed']) > 0),
            ("Sources Validated", len(state['validated_sources']) > 0),
            ("Treatments Compared", len(state['treatments_compared']) > 0)
        ]
        
        for item, completed in status_items:
            status_class = "status-complete" if completed else "status-pending"
            icon = "✅" if completed else "⏳"
            st.markdown(f'<span class="medical-status {status_class}">{icon} {item}</span>', unsafe_allow_html=True)
        
        # Literature sources
        if state['literature_reviewed']:
            st.markdown("**Literature Sources Reviewed:**")
            for source in state['literature_reviewed'][:3]:  # Show first 3
                st.markdown(f"• {source}")
            if len(state['literature_reviewed']) > 3:
                st.markdown(f"• ... and {len(state['literature_reviewed']) - 3} more")
    
    def display_medical_approval_system(self):
        """Display HITL medical approval system"""
        if st.session_state.pending_medical_approvals:
            st.warning(f"⚠️ {len(st.session_state.pending_medical_approvals)} items pending medical review")
            
            for i, approval in enumerate(st.session_state.pending_medical_approvals[:2]):  # Show first 2
                with st.expander(f"Review #{i+1}: {approval.get('type', 'Medical Content')}"):
                    st.write(f"**Content**: {approval.get('content', '')[:200]}...")
                    st.write(f"**Confidence**: {approval.get('confidence', 0.0):.2f}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Approve #{i+1}", key=f"approve_med_{i}"):
                            self.approve_medical_content(i)
                    with col2:
                        if st.button(f"❌ Reject #{i+1}", key=f"reject_med_{i}"):
                            self.reject_medical_content(i)
        else:
            st.success("✅ No pending medical approvals")
    
    def display_medical_export_options(self):
        """Display medical report export options"""
        export_data = {
            'conversation': st.session_state.maria_conversation_messages,
            'research_state': st.session_state.maria_research_state,
            'timestamp': datetime.now().isoformat(),
            'medical_disclaimer': 'This is AI-generated medical research content that requires professional validation.'
        }
        
        # Medical export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Clinical Report PDF"):
                self.export_medical_research("clinical_pdf", export_data)
            if st.button("📊 Research Data CSV"):
                self.export_medical_research("research_csv", export_data)
        
        with col2:
            if st.button("📝 Literature Review"):
                self.export_medical_research("literature_review", export_data)
            if st.button("📋 PRISMA Report"):
                self.export_medical_research("prisma_report", export_data)
    
    def start_medical_research(self, disease: str, research_type: str, population: str, depth: str):
        """Start a new medical research conversation"""
        if not MARIA_AVAILABLE:
            st.error("MARIA medical components not available")
            return
        
        # Show progress indicator
        with st.spinner(f"🏥 Starting medical research on: {disease}..."):
            try:
                # Initialize medical AutoGen agents
                config = get_medical_autogen_config()
                assistant = create_medical_research_assistant(config)
                user_proxy = create_healthcare_user_proxy(st)
                
                # Create medical conversation manager
                conv_manager = MedicalConversationManager(assistant, user_proxy, st)
                
                # Start medical research conversation
                medical_prompt = f"""
                Please conduct {depth} medical research on the following healthcare topic:
                
                Disease/Condition: {disease}
                Research Type: {research_type}
                Target Population: {population}
                
                Please provide:
                1. Current medical literature overview
                2. Treatment options and efficacy data
                3. Clinical trial findings
                4. Safety profiles and contraindications
                5. Guidelines and recommendations
                6. Gaps in current research
                
                Focus on peer-reviewed medical literature and evidence-based medicine.
                Include relevant citations and confidence scores for all findings.
                """
                
                # Update medical session state
                st.session_state.maria_research_state.update({
                    'conversation_active': True,
                    'current_topic': disease,
                    'research_depth': depth,
                    'session_id': f"maria_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                })
                
                # Start medical conversation
                result = conv_manager.initiate_medical_research(medical_prompt)
                
                if result.get("success"):
                    st.success("🏥 Medical research conversation started!")
                    st.rerun()
                else:
                    st.error(f"Failed to start medical research: {result.get('message', 'Unknown error')}")
                    if result.get('error_details'):
                        st.text("Error details:")
                        st.code(result['error_details'])
                
            except Exception as e:
                st.error(f"Error starting medical research: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    
    def handle_medical_query(self, query: str):
        """Handle medical research query input"""
        if not st.session_state.maria_research_state['conversation_active']:
            st.warning("Please start a medical research conversation first")
            return
        
        with st.spinner("🧠 Processing medical query..."):
            try:
                # Add query to conversation
                st.session_state.maria_conversation_messages.append({
                    'sender': 'user',
                    'content': query,
                    'timestamp': datetime.now().isoformat(),
                    'medical_context': self.extract_medical_context(query)
                })
                
                # Generate medical response
                config = get_medical_autogen_config()
                assistant = create_medical_research_assistant(config)
                
                response = assistant.generate_medical_response(query)
                
                # Add MARIA response
                st.session_state.maria_conversation_messages.append({
                    'sender': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat(),
                    'confidence_score': 0.85  # Would be calculated by medical model
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing medical query: {str(e)}")
                st.rerun()
    
    def extract_medical_context(self, content: str) -> Dict[str, Any]:
        """Extract medical entities and context from content"""
        # Simple medical context extraction (would use medical NLP models)
        medical_terms = ["treatment", "therapy", "drug", "medication", "clinical", "trial", "patient", "diagnosis"]
        found_terms = [term for term in medical_terms if term.lower() in content.lower()]
        
        return {
            "medical_terms": found_terms,
            "has_medical_content": len(found_terms) > 0,
            "confidence": min(len(found_terms) / 5.0, 1.0)
        }
    
    def approve_medical_content(self, approval_index: int):
        """Approve medical content after human review"""
        approval = st.session_state.pending_medical_approvals[approval_index]
        
        # Move to validated sources
        st.session_state.maria_research_state['validated_sources'].append({
            **approval,
            "status": "approved",
            "approved_by": st.session_state.maria_research_state['researcher_id'],
            "approved_at": datetime.now().isoformat()
        })
        
        # Remove from pending
        st.session_state.pending_medical_approvals.pop(approval_index)
        st.success("Medical content approved and validated!")
        st.rerun()
    
    def reject_medical_content(self, approval_index: int):
        """Reject medical content after human review"""
        st.session_state.pending_medical_approvals.pop(approval_index)
        st.warning("Medical content rejected and removed.")
        st.rerun()
    
    def generate_literature_review(self):
        """Generate medical literature review"""
        st.info("🔄 Generating literature review... (Feature in development)")
    
    def compare_medical_treatments(self):
        """Compare medical treatments"""
        st.info("⚖️ Comparing treatments... (Feature in development)")
    
    def validate_medical_sources(self):
        """Validate medical sources"""
        st.info("✅ Validating medical sources... (Feature in development)")
    
    def export_medical_research(self, format_type: str, data: Dict):
        """Export medical research in specified format"""
        st.info(f"📁 Exporting {format_type}... (Feature in development)")
    
    def reset_medical_research_session(self):
        """Reset the medical research session"""
        st.session_state.maria_conversation_messages = []
        st.session_state.maria_research_state = {
            'researcher_id': 'researcher_001',
            'project_id': '',
            'disease_focus': '',
            'research_type': 'Literature Review',
            'target_population': 'All Populations',
            'current_topic': '',
            'research_depth': 'intermediate',
            'subtopics_generated': [],
            'literature_reviewed': [],
            'treatments_compared': {},
            'conversation_active': False,
            'last_response': '',
            'session_id': None,
            'confidence_scores': {},
            'validated_sources': []
        }
        st.session_state.maria_export_data = {}
        st.session_state.pending_medical_approvals = []
        st.success("🔄 Medical research session reset!")
        st.rerun()
    
    def display_error_state(self):
        """Display error state when MARIA components are not available"""
        st.error("⚠️ MARIA medical components not properly installed")
        st.markdown("""
        ### Installation Required
        
        Please install the required medical research dependencies:
        
        ```bash
        pip install pyautogen>=0.2.0
        pip install google-generativeai>=0.8.0
        pip install transformers>=4.35.0
        pip install torch>=2.0.0
        pip install bio-bert
        pip install clinical-bert
        ```
        
        And ensure your medical API keys are configured in the .env file.
        """)


def main():
    """Main entry point"""
    app = MARIAApp()
    app.run()


if __name__ == "__main__":
    main()