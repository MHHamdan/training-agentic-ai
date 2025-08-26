"""
ARIA - Autogen Research Intelligence Agent
Advanced AI-powered research assistant with human-in-the-loop control
Built with Microsoft Autogen framework and Streamlit
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

# Import ARIA components
try:
    from autogen_components.research_assistant import create_research_assistant
    from autogen_components.user_proxy import create_enhanced_user_proxy
    from autogen_components.conversation_manager import AutogenConversationManager
    from tools.research_tools import get_research_tools
    from tools.export_tools import ResearchExporter
    from ui.streamlit_interface import create_research_interface
    from config.autogen_config import get_autogen_config
    ARIA_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ ARIA components not available. Error: {str(e)}")
    ARIA_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="ARIA - Autogen Research Intelligence Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ARIA interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1e40af;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .research-container {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .conversation-bubble {
        background: #f1f5f9;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .agent-response {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .research-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .status-active {
        background-color: #fbbf24;
        color: #78350f;
    }
    .status-complete {
        background-color: #34d399;
        color: #064e3b;
    }
    .status-pending {
        background-color: #94a3b8;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


class ARIAApp:
    """Main ARIA application class"""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'aria_conversation_messages' not in st.session_state:
            st.session_state.aria_conversation_messages = []
        if 'aria_research_state' not in st.session_state:
            st.session_state.aria_research_state = {
                'current_topic': '',
                'research_depth': 'intermediate',
                'target_audience': 'general',
                'subtopics_generated': [],
                'research_completed': {},
                'conversation_active': False,
                'last_response': '',
                'session_id': None
            }
        if 'aria_export_data' not in st.session_state:
            st.session_state.aria_export_data = {}
    
    def run(self):
        """Main application entry point"""
        # Header
        st.markdown('<h1 class="main-header">🔬 ARIA - Autogen Research Intelligence Agent</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced AI Research Assistant • Human-in-the-Loop Control • Powered by Microsoft Autogen</p>', unsafe_allow_html=True)
        
        # Check for API keys
        self.check_api_configuration()
        
        # Main interface
        if ARIA_AVAILABLE:
            self.display_research_interface()
        else:
            self.display_error_state()
    
    def check_api_configuration(self):
        """Check and display API configuration status"""
        api_keys = {
            "HUGGING_FACE_API": os.getenv("HUGGING_FACE_API"),
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
        }
        
        available_providers = [k for k, v in api_keys.items() if v]
        
        if not available_providers:
            st.error("❌ No LLM API keys configured. Please add at least one API key to your .env file")
        else:
            # Show primary provider (prioritize working providers)
            if api_keys.get("GOOGLE_API_KEY") or api_keys.get("GEMINI_API_KEY"):
                st.success("🧠 Using **Google Gemini** as primary LLM provider for ARIA")
            elif api_keys.get("OPENAI_API_KEY"):
                st.info("🤖 Using **OpenAI** as LLM provider")
            elif api_keys.get("ANTHROPIC_API_KEY"):
                st.info("🧠 Using **Anthropic Claude** as LLM provider")
            elif api_keys.get("HUGGING_FACE_API") or api_keys.get("HF_TOKEN"):
                st.info("🤗 Using **Hugging Face** as fallback LLM provider")
            
            # Show fallback options
            if len(available_providers) > 1:
                fallbacks = [k.replace("_API_KEY", "").replace("HUGGING_FACE_API", "HUGGINGFACE").replace("HF_TOKEN", "HUGGINGFACE") for k in available_providers[1:]]
                st.caption(f"📋 Fallback options: {', '.join(fallbacks)}")
    
    def display_research_interface(self):
        """Display the main research interface"""
        # Sidebar configuration
        with st.sidebar:
            st.header("🔧 Research Configuration")
            
            # Research topic
            research_topic = st.text_area(
                "Research Topic",
                value=st.session_state.aria_research_state['current_topic'],
                height=100,
                help="Enter your research topic or question",
                placeholder="e.g., Impact of artificial intelligence on healthcare diagnostics"
            )
            
            # Research depth
            research_depth = st.selectbox(
                "Research Depth",
                options=["basic", "intermediate", "comprehensive"],
                index=["basic", "intermediate", "comprehensive"].index(
                    st.session_state.aria_research_state['research_depth']
                ),
                help="Select the depth of research analysis"
            )
            
            # Target audience
            target_audience = st.selectbox(
                "Target Audience",
                options=["general", "academic", "business", "technical"],
                index=["general", "academic", "business", "technical"].index(
                    st.session_state.aria_research_state['target_audience']
                ),
                help="Select your target audience"
            )
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                max_subtopics = st.slider("Max Subtopics", 3, 10, 5)
                enable_web_search = st.checkbox("Enable Web Search", value=True)
                enable_academic_search = st.checkbox("Enable Academic Search", value=True)
                auto_summarize = st.checkbox("Auto Summarize", value=True)
            
            # Update session state
            st.session_state.aria_research_state.update({
                'current_topic': research_topic,
                'research_depth': research_depth,
                'target_audience': target_audience
            })
            
            st.divider()
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                start_research = st.button("🔍 Start Research", type="primary", use_container_width=True)
            with col2:
                reset_session = st.button("🔄 Reset Session", use_container_width=True)
            
            generate_subtopics = st.button("📋 Generate Subtopics", use_container_width=True)
            summarize_results = st.button("📝 Summarize Results", use_container_width=True)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Research conversation area
            st.markdown('<div class="research-container">', unsafe_allow_html=True)
            st.header("💬 Research Conversation")
            
            # Display conversation messages
            if st.session_state.aria_conversation_messages:
                self.display_conversation()
            else:
                st.info("💡 Start a research conversation by entering a topic and clicking 'Start Research'")
            
            # User input area
            user_input = st.text_input(
                "Continue conversation:",
                placeholder="Ask follow-up questions or provide feedback...",
                disabled=not st.session_state.aria_research_state['conversation_active']
            )
            
            if st.button("📤 Send Message") and user_input:
                self.handle_user_message(user_input)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Research status and controls
            st.header("📊 Research Status")
            self.display_research_status()
            
            # Export options
            if st.session_state.aria_conversation_messages:
                st.header("📁 Export Options")
                self.display_export_options()
        
        # Handle button actions
        if start_research and research_topic:
            self.start_research_conversation(research_topic, research_depth, target_audience)
        
        if generate_subtopics:
            self.generate_subtopics()
        
        if summarize_results:
            self.summarize_research()
        
        if reset_session:
            self.reset_research_session()
    
    def display_conversation(self):
        """Display the research conversation"""
        for i, message in enumerate(st.session_state.aria_conversation_messages):
            if message['sender'] == 'user':
                st.markdown(f'<div class="conversation-bubble"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-response"><strong>ARIA:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Auto-scroll to bottom
        if st.session_state.aria_conversation_messages:
            st.empty()
    
    def display_research_status(self):
        """Display current research status"""
        state = st.session_state.aria_research_state
        
        # Current topic
        if state['current_topic']:
            st.markdown(f"**Current Topic:** {state['current_topic'][:100]}...")
        
        # Research progress
        status_items = [
            ("Topic Analysis", state['current_topic'] != ''),
            ("Conversation Active", state['conversation_active']),
            ("Subtopics Generated", len(state['subtopics_generated']) > 0),
            ("Research Complete", len(state['research_completed']) > 0)
        ]
        
        for item, completed in status_items:
            status_class = "status-complete" if completed else "status-pending"
            icon = "✅" if completed else "⏳"
            st.markdown(f'<span class="research-status {status_class}">{icon} {item}</span>', unsafe_allow_html=True)
        
        # Subtopics
        if state['subtopics_generated']:
            st.markdown("**Generated Subtopics:**")
            for subtopic in state['subtopics_generated']:
                st.markdown(f"• {subtopic}")
    
    def display_export_options(self):
        """Display export options for research results"""
        export_data = {
            'conversation': st.session_state.aria_conversation_messages,
            'research_state': st.session_state.aria_research_state,
            'timestamp': datetime.now().isoformat()
        }
        
        # Export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export PDF"):
                self.export_research("pdf", export_data)
            if st.button("📊 Export CSV"):
                self.export_research("csv", export_data)
        
        with col2:
            if st.button("📝 Export Word"):
                self.export_research("docx", export_data)
            if st.button("📋 Export Markdown"):
                self.export_research("markdown", export_data)
    
    def start_research_conversation(self, topic: str, depth: str, audience: str):
        """Start a new research conversation"""
        if not ARIA_AVAILABLE:
            st.error("ARIA components not available")
            return
        
        # Show progress indicator
        with st.spinner(f"🔍 Starting research on: {topic}..."):
            try:
                # Initialize Autogen agents
                config = get_autogen_config()
                assistant = create_research_assistant(config)
                user_proxy = create_enhanced_user_proxy(st)
                
                # Create conversation manager
                conv_manager = AutogenConversationManager(assistant, user_proxy, st)
                
                # Start research conversation
                research_prompt = f"""
                Please conduct {depth} research on the following topic for a {audience} audience:
                
                Topic: {topic}
                
                Please provide:
                1. An overview of the topic
                2. Key concepts and definitions
                3. Current trends and developments
                4. Relevant applications or implications
                5. Potential areas for further investigation
                
                Structure your response clearly and cite relevant sources where possible.
                """
                
                # Update session state before starting conversation
                st.session_state.aria_research_state.update({
                    'conversation_active': True,
                    'current_topic': topic,
                    'research_depth': depth,
                    'target_audience': audience,
                    'session_id': f"aria_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                })
                
                # Start conversation
                result = conv_manager.initiate_research(research_prompt)
                
                if result.get("success"):
                    st.success("🔍 Research conversation started!")
                    # Force immediate rerun to show the conversation
                    st.rerun()
                else:
                    st.error(f"Failed to start research: {result.get('message', 'Unknown error')}")
                    if result.get('error_details'):
                        st.text("Error details:")
                        st.code(result['error_details'])
                
            except Exception as e:
                st.error(f"Error starting research: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    
    def generate_subtopics(self):
        """Generate subtopics for the current research"""
        if not st.session_state.aria_research_state['current_topic']:
            st.warning("Please start a research conversation first")
            return
        
        topic = st.session_state.aria_research_state['current_topic']
        
        with st.spinner("🔄 Generating subtopics..."):
            try:
                # Create research assistant for subtopic generation
                config = get_autogen_config()
                assistant = create_research_assistant(config)
                
                # Generate subtopics
                subtopics = assistant.generate_subtopics(topic, max_subtopics=5)
                
                # Update session state
                st.session_state.aria_research_state['subtopics_generated'] = subtopics
                
                # Display generated subtopics
                st.success("✅ Subtopics generated successfully!")
                st.write("**Generated Subtopics:**")
                for i, subtopic in enumerate(subtopics, 1):
                    st.write(f"{i}. {subtopic}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating subtopics: {str(e)}")
    
    def summarize_research(self):
        """Summarize the current research results"""
        if not st.session_state.aria_conversation_messages:
            st.warning("Please start a research conversation first")
            return
        
        with st.spinner("📝 Summarizing research..."):
            try:
                # Create research assistant for summarization
                config = get_autogen_config()
                assistant = create_research_assistant(config)
                
                # Prepare research content for summarization
                research_content = []
                for msg in st.session_state.aria_conversation_messages:
                    if msg['sender'] == 'assistant':
                        research_content.append({
                            'content': msg['content'],
                            'timestamp': msg['timestamp']
                        })
                
                if not research_content:
                    st.warning("No research content to summarize yet.")
                    return
                
                # Generate summary
                summary = assistant.summarize_research(research_content)
                
                # Add summary to conversation
                st.session_state.aria_conversation_messages.append({
                    'sender': 'assistant',
                    'content': f"**Research Summary:**\n\n{summary}",
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update research state
                st.session_state.aria_research_state['research_completed']['summary'] = summary
                
                st.success("✅ Research summary generated!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error summarizing research: {str(e)}")
    
    def handle_user_message(self, message: str):
        """Handle user input in the conversation"""
        if not st.session_state.aria_research_state['conversation_active']:
            st.warning("Please start a research conversation first")
            return
        
        with st.spinner("🤖 Processing your message..."):
            try:
                # Add user message to conversation
                st.session_state.aria_conversation_messages.append({
                    'sender': 'user',
                    'content': message,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Create research assistant for response
                config = get_autogen_config()
                assistant = create_research_assistant(config)
                
                # Generate response
                response = assistant.generate_research_response(message)
                
                # Add assistant response to conversation
                st.session_state.aria_conversation_messages.append({
                    'sender': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
                st.rerun()
    
    def export_research(self, format_type: str, data: Dict):
        """Export research results in specified format"""
        try:
            exporter = ResearchExporter(data)
            
            if format_type == "pdf":
                pdf_data = exporter.export_to_pdf()
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_data,
                    file_name=f"aria_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            elif format_type == "docx":
                docx_data = exporter.export_to_word()
                st.download_button(
                    label="📝 Download Word",
                    data=docx_data,
                    file_name=f"aria_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            elif format_type == "csv":
                csv_data = exporter.export_to_csv()
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"aria_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif format_type == "markdown":
                md_data = exporter.export_to_markdown()
                st.download_button(
                    label="📋 Download Markdown",
                    data=md_data,
                    file_name=f"aria_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        except Exception as e:
            st.error(f"Export error: {str(e)}")
    
    def reset_research_session(self):
        """Reset the research session"""
        st.session_state.aria_conversation_messages = []
        st.session_state.aria_research_state = {
            'current_topic': '',
            'research_depth': 'intermediate',
            'target_audience': 'general',
            'subtopics_generated': [],
            'research_completed': {},
            'conversation_active': False,
            'last_response': '',
            'session_id': None
        }
        st.session_state.aria_export_data = {}
        st.success("🔄 Research session reset!")
        st.rerun()
    
    def display_error_state(self):
        """Display error state when ARIA components are not available"""
        st.error("⚠️ ARIA components not properly installed")
        st.markdown("""
        ### Installation Required
        
        Please install the required Autogen dependencies:
        
        ```bash
        pip install pyautogen>=0.2.0
        pip install google-generativeai>=0.8.0
        pip install fpdf2>=2.7.0
        ```
        
        And ensure your API keys are configured in the .env file.
        """)


def main():
    """Main entry point"""
    app = ARIAApp()
    app.run()


if __name__ == "__main__":
    main()