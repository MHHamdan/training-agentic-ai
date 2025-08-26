"""
Streamlit Interface Components for ARIA
Advanced UI components for research interactions
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


def create_research_interface():
    """
    Create the main research interface
    
    Returns:
        Dictionary containing interface state
    """
    interface_state = {
        "research_topic": "",
        "research_depth": "intermediate",
        "target_audience": "general",
        "conversation_active": False,
        "current_session": None
    }
    
    # Research configuration sidebar
    with st.sidebar:
        st.header("🔧 Research Configuration")
        
        # Topic input
        interface_state["research_topic"] = st.text_area(
            "Research Topic",
            height=100,
            help="Enter your research topic or question",
            placeholder="e.g., Impact of artificial intelligence on healthcare diagnostics"
        )
        
        # Research parameters
        interface_state["research_depth"] = st.selectbox(
            "Research Depth",
            options=["basic", "intermediate", "comprehensive"],
            index=1,
            help="Select the depth of research analysis"
        )
        
        interface_state["target_audience"] = st.selectbox(
            "Target Audience",
            options=["general", "academic", "business", "technical"],
            help="Select your target audience"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            max_subtopics = st.slider("Max Subtopics", 3, 10, 5)
            enable_web_search = st.checkbox("Enable Web Search", value=True)
            enable_academic_search = st.checkbox("Enable Academic Search", value=True)
            auto_summarize = st.checkbox("Auto Summarize", value=True)
        
        interface_state.update({
            "max_subtopics": max_subtopics,
            "enable_web_search": enable_web_search,
            "enable_academic_search": enable_academic_search,
            "auto_summarize": auto_summarize
        })
    
    return interface_state


def create_conversation_display(messages: List[Dict[str, Any]]):
    """
    Create conversation display component
    
    Args:
        messages: List of conversation messages
    """
    if not messages:
        st.info("💡 Start a research conversation by entering a topic and clicking 'Start Research'")
        return
    
    st.markdown("### 💬 Research Conversation")
    
    for i, message in enumerate(messages):
        sender = message.get('sender', 'unknown')
        content = message.get('content', '')
        timestamp = message.get('timestamp', '')
        
        if sender == 'user':
            st.markdown(f"""
            <div style="background: #f1f5f9; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                <strong>You ({timestamp}):</strong><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #ecfdf5; border-left: 4px solid #10b981; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                <strong>ARIA ({timestamp}):</strong><br>
                {content}
            </div>
            """, unsafe_allow_html=True)


def create_research_status_panel(research_state: Dict[str, Any]):
    """
    Create research status panel
    
    Args:
        research_state: Current research state
    """
    st.markdown("### 📊 Research Status")
    
    # Current topic
    if research_state.get('current_topic'):
        topic = research_state['current_topic']
        display_topic = topic[:100] + "..." if len(topic) > 100 else topic
        st.markdown(f"**Current Topic:** {display_topic}")
    
    # Status indicators
    status_items = [
        ("Topic Analysis", bool(research_state.get('current_topic'))),
        ("Conversation Active", research_state.get('conversation_active', False)),
        ("Subtopics Generated", len(research_state.get('subtopics_generated', [])) > 0),
        ("Research Complete", len(research_state.get('research_completed', {})) > 0)
    ]
    
    for item, completed in status_items:
        status_class = "status-complete" if completed else "status-pending"
        icon = "✅" if completed else "⏳"
        st.markdown(f"""
        <span style="display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; 
                    font-size: 0.875rem; font-weight: 600; margin: 0.25rem;
                    background-color: {'#34d399' if completed else '#94a3b8'};
                    color: {'#064e3b' if completed else '#1e293b'};">
            {icon} {item}
        </span>
        """, unsafe_allow_html=True)
    
    # Subtopics
    subtopics = research_state.get('subtopics_generated', [])
    if subtopics:
        st.markdown("**Generated Subtopics:**")
        for subtopic in subtopics:
            st.markdown(f"• {subtopic}")


def create_export_panel(conversation_messages: List[Dict], research_state: Dict[str, Any]):
    """
    Create export options panel
    
    Args:
        conversation_messages: Conversation messages
        research_state: Research state
    """
    if not conversation_messages:
        return
    
    st.markdown("### 📁 Export Options")
    
    export_data = {
        'conversation': conversation_messages,
        'research_state': research_state,
        'timestamp': datetime.now().isoformat()
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Export PDF", use_container_width=True):
            try:
                from tools.export_tools import ResearchExporter
                exporter = ResearchExporter(export_data)
                pdf_data = exporter.export_to_pdf()
                
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_data,
                    file_name=f"aria_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF export error: {str(e)}")
        
        if st.button("📊 Export CSV", use_container_width=True):
            try:
                from tools.export_tools import ResearchExporter
                exporter = ResearchExporter(export_data)
                csv_data = exporter.export_to_csv()
                
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"aria_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"CSV export error: {str(e)}")
    
    with col2:
        if st.button("📝 Export Word", use_container_width=True):
            try:
                from tools.export_tools import ResearchExporter
                exporter = ResearchExporter(export_data)
                docx_data = exporter.export_to_word()
                
                st.download_button(
                    label="📝 Download Word",
                    data=docx_data,
                    file_name=f"aria_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"Word export error: {str(e)}")
        
        if st.button("📋 Export Markdown", use_container_width=True):
            try:
                from tools.export_tools import ResearchExporter
                exporter = ResearchExporter(export_data)
                md_data = exporter.export_to_markdown()
                
                st.download_button(
                    label="📋 Download Markdown",
                    data=md_data,
                    file_name=f"aria_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Markdown export error: {str(e)}")


def create_research_metrics_display(conversation_messages: List[Dict]):
    """
    Create research metrics display
    
    Args:
        conversation_messages: Conversation messages
    """
    if not conversation_messages:
        return
    
    st.markdown("### 📈 Research Metrics")
    
    # Calculate metrics
    total_messages = len(conversation_messages)
    user_messages = len([m for m in conversation_messages if m.get('sender') == 'user'])
    assistant_messages = len([m for m in conversation_messages if m.get('sender') == 'assistant'])
    
    # Create metrics columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Messages", total_messages)
    
    with col2:
        st.metric("User Questions", user_messages)
    
    with col3:
        st.metric("AI Responses", assistant_messages)
    
    # Message timeline
    if len(conversation_messages) > 1:
        try:
            # Create a simple timeline visualization
            timestamps = []
            senders = []
            
            for msg in conversation_messages:
                if 'timestamp' in msg:
                    timestamps.append(datetime.fromisoformat(msg['timestamp']))
                    senders.append(msg.get('sender', 'unknown'))
            
            if timestamps:
                df = pd.DataFrame({
                    'Time': timestamps,
                    'Sender': senders
                })
                
                # Group by sender and count
                sender_counts = df['Sender'].value_counts()
                
                st.markdown("**Message Distribution:**")
                for sender, count in sender_counts.items():
                    percentage = (count / total_messages) * 100
                    st.write(f"• {sender.title()}: {count} messages ({percentage:.1f}%)")
                
        except Exception as e:
            st.caption(f"Timeline visualization error: {str(e)}")


def create_subtopic_generator_interface():
    """
    Create subtopic generator interface
    
    Returns:
        Generated subtopics or None
    """
    st.markdown("### 📋 Subtopic Generator")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic_for_subtopics = st.text_input(
            "Topic for subtopic generation:",
            placeholder="Enter a topic to generate subtopics"
        )
    
    with col2:
        max_subtopics = st.number_input("Max subtopics", min_value=3, max_value=10, value=5)
    
    if st.button("🔍 Generate Subtopics"):
        if topic_for_subtopics:
            # Placeholder for subtopic generation
            # In a real implementation, this would use the research assistant
            subtopics = [
                f"Historical Development of {topic_for_subtopics}",
                f"Current State and Trends in {topic_for_subtopics}",
                f"Key Technologies in {topic_for_subtopics}",
                f"Applications of {topic_for_subtopics}",
                f"Future Outlook for {topic_for_subtopics}"
            ][:max_subtopics]
            
            st.success("✅ Subtopics generated!")
            for i, subtopic in enumerate(subtopics, 1):
                st.write(f"{i}. {subtopic}")
            
            return subtopics
        else:
            st.warning("Please enter a topic first")
    
    return None


def create_research_tools_panel():
    """
    Create research tools panel
    """
    st.markdown("### 🛠️ Research Tools")
    
    tab1, tab2, tab3 = st.tabs(["Web Search", "Academic Search", "Content Analysis"])
    
    with tab1:
        st.markdown("#### 🔍 Web Search")
        web_query = st.text_input("Search query:", placeholder="Enter search terms")
        if st.button("Search Web") and web_query:
            with st.spinner("Searching..."):
                # Placeholder for web search
                st.info(f"Searching for: {web_query}")
                st.write("Search results would appear here")
    
    with tab2:
        st.markdown("#### 🎓 Academic Search")
        academic_query = st.text_input("Academic search:", placeholder="Enter academic search terms")
        source = st.selectbox("Source", ["arXiv", "PubMed", "Crossref"])
        if st.button("Search Academic") and academic_query:
            with st.spinner("Searching academic sources..."):
                st.info(f"Searching {source} for: {academic_query}")
                st.write("Academic results would appear here")
    
    with tab3:
        st.markdown("#### 📊 Content Analysis")
        content_to_analyze = st.text_area("Content to analyze:", height=100)
        analysis_type = st.selectbox("Analysis type", ["Basic", "Comprehensive", "Summary"])
        if st.button("Analyze Content") and content_to_analyze:
            with st.spinner("Analyzing content..."):
                st.info(f"Performing {analysis_type.lower()} analysis")
                st.write("Analysis results would appear here")


def create_conversation_controls():
    """
    Create conversation control buttons
    
    Returns:
        Dictionary with button states
    """
    st.markdown("### 🎮 Conversation Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_research = st.button("🔍 Start Research", type="primary", use_container_width=True)
    
    with col2:
        generate_summary = st.button("📝 Generate Summary", use_container_width=True)
    
    with col3:
        end_conversation = st.button("🔚 End Conversation", use_container_width=True)
    
    return {
        "start_research": start_research,
        "generate_summary": generate_summary,
        "end_conversation": end_conversation
    }


def create_api_status_display():
    """
    Create API status display
    """
    import os
    
    st.markdown("### 🔑 API Status")
    
    api_keys = {
        "Google/Gemini": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Hugging Face": os.getenv("HUGGING_FACE_API") or os.getenv("HF_TOKEN")
    }
    
    for provider, key in api_keys.items():
        status = "✅ Connected" if key else "❌ Not configured"
        color = "green" if key else "red"
        st.markdown(f"**{provider}:** :{color}[{status}]")
    
    # Show primary provider
    available_providers = [k for k, v in api_keys.items() if v]
    if available_providers:
        st.success(f"🎯 Primary provider: **{available_providers[0]}**")
    else:
        st.error("⚠️ No API keys configured")


def create_session_management_panel():
    """
    Create session management panel
    """
    st.markdown("### 📂 Session Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Session", use_container_width=True):
            # Placeholder for session saving
            st.success("Session saved!")
    
    with col2:
        if st.button("📁 Load Session", use_container_width=True):
            # Placeholder for session loading
            st.info("Session loading not implemented")
    
    # Session info
    if 'aria_research_state' in st.session_state:
        session_id = st.session_state.aria_research_state.get('session_id')
        if session_id:
            st.caption(f"Current session: {session_id}")


def display_help_section():
    """
    Display help and usage information
    """
    with st.expander("❓ Help & Usage"):
        st.markdown("""
        ### How to Use ARIA
        
        1. **Configure Research**: Enter your research topic and select depth/audience in the sidebar
        2. **Start Research**: Click "Start Research" to begin the conversation
        3. **Interact**: Ask follow-up questions or request specific information
        4. **Export Results**: Use the export options to save your research in various formats
        
        ### Research Depths
        - **Basic**: Foundational overview with key concepts
        - **Intermediate**: Detailed analysis with multiple perspectives
        - **Comprehensive**: Exhaustive analysis with historical context and trends
        
        ### Target Audiences
        - **General**: Accessible language with practical examples
        - **Academic**: Scholarly language with peer-reviewed sources
        - **Business**: Strategic focus with actionable insights
        - **Technical**: Detailed specifications and implementation guidance
        
        ### Available Tools
        - Web search for current information
        - Academic search for scholarly articles
        - Content analysis and summarization
        - Multi-format export (PDF, Word, CSV, Markdown)
        """)


def create_footer():
    """
    Create application footer
    """
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🔬 ARIA - Autogen Research Intelligence Agent<br>
        <small>Powered by Microsoft Autogen • Built with Streamlit</small>
    </div>
    """, unsafe_allow_html=True)