#!/usr/bin/env python3
"""
Simple test to verify basic Streamlit functionality
"""

import streamlit as st
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.workflow_manager import workflow_manager

st.set_page_config(page_title="Simple Research Test", page_icon="🧪")

st.title("🧪 Simple Research Test")

# Test basic Streamlit functionality
st.write("✅ Streamlit is working")

# Test session state
if "test_counter" not in st.session_state:
    st.session_state.test_counter = 0

if st.button("Test Button"):
    st.session_state.test_counter += 1
    st.success(f"Button clicked {st.session_state.test_counter} times!")

st.write(f"Session state counter: {st.session_state.test_counter}")

# Test async research
if st.button("🔬 Test Research"):
    st.write("🔄 Starting research...")
    
    try:
        async def simple_research():
            result = await workflow_manager.run_research(
                query="simple test query",
                user_preferences={"depth": "quick", "use_free_models": True}
            )
            return result
        
        with st.spinner("Running research..."):
            result = asyncio.run(simple_research())
        
        if result:
            st.success("✅ Research completed!")
            st.write(f"Query: {result.get('query', 'N/A')}")
            st.write(f"Quality Score: {result.get('quality_score', 'N/A')}")
            st.write(f"Phase: {result.get('phase', 'N/A')}")
            st.write(f"Results: {len(result.get('search_results', []))} sources")
        else:
            st.error("❌ Research returned None")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

st.write("---")
st.write("If you can see this page and the button works, Streamlit is functioning correctly.")