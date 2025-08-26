import streamlit as st
import asyncio
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ResumeScreeningAgent
from components import UIComponents
from config import config

st.set_page_config(
    page_title="Resume Screening Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_agent():
    return ResumeScreeningAgent()

def main():
    st.title("🎯 AI-Powered Resume Screening Agent")
    st.markdown("### Agent 12 - Production-Ready Resume Analysis with Multi-Model Support")
    
    agent = initialize_agent()
    ui = UIComponents()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Configuration Section
        st.subheader("🔑 API Keys (Optional)")
        st.info("💡 **Add your API keys to unlock premium cloud models**")
        
        with st.expander("🔒 Configure Cloud Provider APIs", expanded=False):
            st.markdown("**Enter your API keys to enable premium models:**")
            
            # Store API keys in session state
            if 'api_keys' not in st.session_state:
                st.session_state.api_keys = {}
            
            # OpenAI API Key
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Get your API key from https://platform.openai.com/api-keys"
            )
            if openai_key:
                st.session_state.api_keys['OPENAI_API_KEY'] = openai_key
                os.environ['OPENAI_API_KEY'] = openai_key
            
            # Anthropic API Key
            anthropic_key = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
                help="Get your API key from https://console.anthropic.com/"
            )
            if anthropic_key:
                st.session_state.api_keys['ANTHROPIC_API_KEY'] = anthropic_key
                os.environ['ANTHROPIC_API_KEY'] = anthropic_key
            
            # Google API Key
            google_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                placeholder="AI...",
                help="Get your API key from https://makersuite.google.com/app/apikey"
            )
            if google_key:
                st.session_state.api_keys['GOOGLE_API_KEY'] = google_key
                os.environ['GOOGLE_API_KEY'] = google_key
            
            # Cohere API Key
            cohere_key = st.text_input(
                "Cohere API Key",
                type="password",
                placeholder="...",
                help="Get your API key from https://dashboard.cohere.ai/"
            )
            if cohere_key:
                st.session_state.api_keys['COHERE_API_KEY'] = cohere_key
                os.environ['COHERE_API_KEY'] = cohere_key
            
            # Groq API Key
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Get your API key from https://console.groq.com/keys"
            )
            if groq_key:
                st.session_state.api_keys['GROQ_API_KEY'] = groq_key
                os.environ['GROQ_API_KEY'] = groq_key
            
            # Update agent configuration when API keys change
            if st.session_state.api_keys:
                st.success(f"✅ {len(st.session_state.api_keys)} API key(s) configured")
                if st.button("🔄 Refresh Models"):
                    st.rerun()
        
        st.divider()
        
        st.subheader("🤖 Model Selection")
        available_models = agent.model_manager.get_available_models()
        
        # Display provider information
        free_providers = [k for k in available_models.keys() if "alternative" in k or "free" in k or k in ["mistral", "deepseek", "qwen", "cerebras", "ollama_compatible"]]
        cloud_providers = [k for k in available_models.keys() if k not in free_providers]
        
        if free_providers:
            st.info(f"🆓 **Free Models Available:** {len(free_providers)} categories with {sum(len(available_models[k]) for k in free_providers)} models")
        
        if cloud_providers:
            st.success(f"☁️ **Cloud APIs Connected:** {len(cloud_providers)} providers with {sum(len(available_models[k]) for k in cloud_providers)} premium models")
        else:
            st.warning("⚠️ **No Cloud APIs Connected** - Add API keys above to unlock premium models")
        
        selected_category = st.selectbox(
            "Model Category",
            list(available_models.keys()),
            help="🆓 = Free Hugging Face models, ☁️ = Premium cloud APIs",
            format_func=lambda x: f"{'☁️' if x in cloud_providers else '🆓'} {x.replace('_', ' ').title()}"
        )
        
        selected_models = st.multiselect(
            "Select Models for Analysis",
            available_models[selected_category],
            default=[available_models[selected_category][0]] if available_models[selected_category] else [],
            help="Choose one or more models for comparative analysis"
        )
        
        st.divider()
        
        st.subheader("📊 Analytics")
        if st.button("🔄 Refresh Metrics"):
            st.rerun()
        
        health = agent.health_check()
        st.success(f"Status: {health['status'].upper()}")
        st.info(f"Version: {health['version']}")
        st.metric("Available Models", health['models_available'])
        
        if config.langchain_tracing:
            st.success("✅ LangSmith Observability Active")
        else:
            st.warning("⚠️ LangSmith Observability Disabled")
    
    tabs = st.tabs(["📤 Upload & Analyze", "📊 Model Comparison", "📈 Analytics", "🔍 Search History", "⚙️ Settings"])
    
    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Resume Upload")
            
            uploaded_file = st.file_uploader(
                "Choose a resume file",
                type=['pdf', 'docx', 'doc', 'txt'],
                help="Upload a resume in PDF, DOCX, DOC, or TXT format"
            )
            
            if uploaded_file:
                temp_path = Path(f"/tmp/{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.info(f"Size: {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            st.subheader("📋 Job Requirements")
            
            # Sample job requirements
            sample_jobs = {
                "Select a sample job...": "",
                "Senior Software Engineer": """We are looking for a Senior Software Engineer to join our team.

Required Qualifications:
- 5+ years of experience in software development
- Strong proficiency in Python, Java, or JavaScript
- Experience with cloud platforms (AWS, Azure, or GCP)
- Solid understanding of data structures and algorithms
- Experience with RESTful APIs and microservices architecture
- Proficiency with Git and CI/CD pipelines
- Bachelor's degree in Computer Science or related field

Preferred Skills:
- Experience with Docker and Kubernetes
- Knowledge of machine learning frameworks (TensorFlow, PyTorch)
- Experience with agile development methodologies
- Strong problem-solving and analytical skills
- Excellent written and verbal communication skills""",
                
                "Data Scientist": """Join our data science team to build cutting-edge ML solutions.

Required Qualifications:
- 3+ years of experience in data science or machine learning
- Strong proficiency in Python and SQL
- Experience with machine learning libraries (scikit-learn, TensorFlow, PyTorch)
- Solid understanding of statistical analysis and modeling
- Experience with data visualization tools (Matplotlib, Seaborn, Tableau)
- Knowledge of big data technologies (Spark, Hadoop)
- Master's degree in Data Science, Statistics, or related field

Preferred Skills:
- Experience with deep learning and neural networks
- Knowledge of NLP techniques and libraries
- Experience with cloud ML platforms (AWS SageMaker, Azure ML)
- Published research papers or kaggle competition experience
- Strong presentation and storytelling skills""",
                
                "Full Stack Developer": """We need a Full Stack Developer to build modern web applications.

Required Qualifications:
- 4+ years of full stack development experience
- Frontend: React, Angular, or Vue.js expertise
- Backend: Node.js, Python (Django/Flask), or Ruby on Rails
- Database experience with PostgreSQL, MySQL, or MongoDB
- Experience with responsive design and CSS frameworks
- Understanding of web security best practices
- Bachelor's degree in Computer Science or equivalent experience

Preferred Skills:
- Experience with TypeScript
- Knowledge of GraphQL
- Experience with serverless architecture
- Mobile development experience (React Native, Flutter)
- DevOps skills with Docker and CI/CD
- Contribution to open source projects""",
                
                "DevOps Engineer": """Looking for a DevOps Engineer to optimize our infrastructure.

Required Qualifications:
- 4+ years of DevOps or SRE experience
- Strong experience with AWS, Azure, or GCP
- Expertise in Infrastructure as Code (Terraform, CloudFormation)
- Proficiency with Docker and Kubernetes
- Experience with CI/CD tools (Jenkins, GitLab CI, GitHub Actions)
- Strong scripting skills (Python, Bash, PowerShell)
- Understanding of networking and security principles

Preferred Skills:
- Experience with monitoring tools (Prometheus, Grafana, ELK stack)
- Knowledge of service mesh (Istio, Linkerd)
- Experience with configuration management (Ansible, Puppet)
- Certification in cloud platforms
- Experience with GitOps practices
- Strong troubleshooting and problem-solving abilities""",
                
                "Machine Learning Engineer": """Join us to build and deploy ML models at scale.

Required Qualifications:
- 3+ years of ML engineering experience
- Strong programming skills in Python
- Experience with ML frameworks (TensorFlow, PyTorch, scikit-learn)
- Experience deploying ML models to production
- Knowledge of MLOps practices and tools
- Understanding of software engineering best practices
- Bachelor's degree in Computer Science, ML, or related field

Preferred Skills:
- Experience with model serving (TensorFlow Serving, TorchServe)
- Knowledge of distributed training frameworks
- Experience with feature stores and data pipelines
- Understanding of A/B testing and model monitoring
- Experience with edge deployment and model optimization
- Publications or contributions to ML research""",
                
                "Product Manager": """We're seeking a Product Manager to lead product strategy.

Required Qualifications:
- 5+ years of product management experience
- Experience with agile development methodologies
- Strong analytical and data-driven decision making skills
- Excellent communication and stakeholder management
- Experience with product analytics tools
- Understanding of UX/UI principles
- Bachelor's degree required, MBA preferred

Technical Skills Preferred:
- Basic understanding of APIs and databases
- Experience with A/B testing platforms
- Knowledge of SQL for data analysis
- Familiarity with design tools (Figma, Sketch)
- Understanding of web technologies
- Experience with JIRA and Confluence""",
                
                "Frontend Engineer": """Looking for a Frontend Engineer to create amazing user experiences.

Required Qualifications:
- 3+ years of frontend development experience
- Expert knowledge of JavaScript, HTML5, and CSS3
- Strong experience with React, Angular, or Vue.js
- Experience with state management (Redux, MobX, Vuex)
- Understanding of responsive design principles
- Experience with modern build tools (Webpack, Vite)
- Knowledge of web performance optimization

Preferred Skills:
- Experience with TypeScript
- Knowledge of CSS-in-JS libraries
- Experience with testing frameworks (Jest, Cypress)
- Understanding of accessibility standards (WCAG)
- Experience with Progressive Web Apps
- Familiarity with design systems and component libraries"""
            }
            
            selected_job = st.selectbox(
                "Choose a sample job or write your own:",
                options=list(sample_jobs.keys()),
                help="Select a pre-filled job description or write your own below"
            )
            
            # Pre-fill the text area if a sample is selected
            default_text = sample_jobs[selected_job] if selected_job != "Select a sample job..." else ""
            
            job_requirements = st.text_area(
                "Job requirements",
                value=default_text,
                height=250,
                placeholder="""Enter the job requirements here or select a sample above...""",
                help="Provide detailed job requirements including skills, experience, and qualifications"
            )
        
        if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
            if uploaded_file and job_requirements:
                with st.spinner("🔍 Analyzing resume with selected models..."):
                    try:
                        result = asyncio.run(
                            agent.process_resume(
                                str(temp_path),
                                job_requirements,
                                selected_models if selected_models else None
                            )
                        )
                        
                        if result.get("status") == "success":
                            st.success("✅ Analysis Complete!")
                            
                            scores = result.get("comprehensive_score", {})
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                ui.create_score_gauge("Overall Score", scores.get("overall", 0))
                            with col2:
                                ui.create_score_gauge("Technical Skills", scores.get("technical_skills", 0))
                            with col3:
                                ui.create_score_gauge("Experience", scores.get("experience_relevance", 0))
                            
                            st.divider()
                            
                            recommendation = result.get("recommendation", "")
                            if "STRONGLY RECOMMEND" in recommendation:
                                st.success(f"✅ {recommendation}")
                            elif "RECOMMEND" in recommendation:
                                st.info(f"👍 {recommendation}")
                            elif "CONSIDER" in recommendation:
                                st.warning(f"🤔 {recommendation}")
                            else:
                                st.error(f"❌ {recommendation}")
                            
                            with st.expander("📊 Detailed Scores"):
                                scores_df = pd.DataFrame([scores]).T
                                scores_df.columns = ["Score"]
                                st.dataframe(scores_df, use_container_width=True)
                            
                            with st.expander("🔍 Model-Specific Analyses"):
                                for model_name, analysis in result.get("analyses", {}).items():
                                    if analysis.get("status") == "success":
                                        st.subheader(f"Model: {model_name}")
                                        model_analysis = analysis.get("analysis", {})
                                        
                                        if "insights" in model_analysis:
                                            st.write("**Insights:**")
                                            for insight in model_analysis["insights"][:5]:
                                                st.write(f"• {insight}")
                                        
                                        if "extracted_skills" in model_analysis:
                                            st.write("**Extracted Skills:**")
                                            skills = model_analysis["extracted_skills"]
                                            for category, skill_list in skills.items():
                                                if skill_list:
                                                    st.write(f"*{category}:* {', '.join(skill_list)}")
                            
                            with st.expander("💾 Export Results"):
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    json_data = asyncio.run(agent.export_results(result, "json"))
                                    st.download_button(
                                        "📥 Download JSON",
                                        json_data,
                                        f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        "application/json",
                                        help="Download complete analysis data in JSON format"
                                    )
                                
                                with col2:
                                    csv_data = asyncio.run(agent.export_results(result, "csv"))
                                    st.download_button(
                                        "📥 Download CSV",
                                        csv_data,
                                        f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        "text/csv",
                                        help="Download analysis scores in CSV format for Excel"
                                    )
                                
                                with col3:
                                    pdf_data = asyncio.run(agent.export_results(result, "pdf"))
                                    st.download_button(
                                        "📥 Download PDF",
                                        pdf_data,
                                        f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        "application/pdf",
                                        help="Download formatted PDF report with all scores"
                                    )
                                
                                with col4:
                                    st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                        
                        else:
                            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please upload a resume and enter job requirements")
    
    with tabs[1]:
        st.subheader("🤖 Model Comparison Dashboard")
        
        if st.button("🔬 Run Model Comparison"):
            if uploaded_file and job_requirements:
                with st.spinner("Comparing models..."):
                    try:
                        comparison = asyncio.run(
                            agent.compare_models(
                                open(temp_path, 'r', errors='ignore').read()[:5000],
                                job_requirements
                            )
                        )
                        
                        if comparison.get("models"):
                            models_data = []
                            for model_name, metrics in comparison["models"].items():
                                if model_name and metrics:  # Ensure both model_name and metrics are not None
                                    # Safe model name extraction
                                    display_name = model_name.split("/")[-1] if "/" in str(model_name) else str(model_name)
                                    models_data.append({
                                        "Model": display_name,
                                        "Score": metrics.get("score", 0) if isinstance(metrics, dict) else 0,
                                        "Time (s)": metrics.get("processing_time", 0) if isinstance(metrics, dict) else 0
                                    })
                            
                            if models_data:  # Only proceed if we have valid data
                                df = pd.DataFrame(models_data)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig = px.bar(
                                        df,
                                        x="Model",
                                        y="Score",
                                        title="Model Performance Comparison",
                                        color="Score",
                                        color_continuous_scale="viridis"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    fig = px.scatter(
                                        df,
                                        x="Time (s)",
                                        y="Score",
                                        text="Model",
                                        title="Performance vs Speed",
                                        size="Score"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                st.divider()
                                
                                # Safe best model name extraction
                                best_model_raw = comparison.get("best_model", "")
                                best_model = ""
                                if best_model_raw:
                                    best_model = best_model_raw.split("/")[-1] if "/" in str(best_model_raw) else str(best_model_raw)
                                
                                consensus = comparison.get("consensus_score", 0)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("🏆 Best Model", best_model or "N/A")
                                with col2:
                                    st.metric("📊 Consensus Score", f"{consensus:.1f}")
                                
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.warning("⚠️ No valid model comparison data available")
                        else:
                            st.warning("⚠️ No models available for comparison")
                    
                    except Exception as e:
                        st.error(f"Error in model comparison: {str(e)}")
            else:
                st.warning("⚠️ Please upload a resume and enter job requirements first")
    
    with tabs[2]:
        st.subheader("📈 Analytics Dashboard")
        
        metrics = agent.metrics_collector.get_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Processed", metrics.get("total_processed", 0))
        with col2:
            st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
        with col3:
            st.metric("Avg Processing Time", f"{metrics.get('average_processing_time', 0):.2f}s")
        with col4:
            st.metric("Avg Score", f"{metrics.get('average_score', 0):.1f}")
        
        st.divider()
        
        if metrics.get("model_performance"):
            st.subheader("Model Performance Metrics")
            
            model_perf_data = []
            for model, perf in metrics["model_performance"].items():
                model_perf_data.append({
                    "Model": model.split("/")[-1],
                    "Calls": perf["total_calls"],
                    "Success Rate": perf["success_rate"],
                    "Avg Time": perf["average_time"]
                })
            
            if model_perf_data:
                df = pd.DataFrame(model_perf_data)
                
                fig = px.bar(
                    df,
                    x="Model",
                    y="Success Rate",
                    title="Model Success Rates",
                    color="Success Rate",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df, use_container_width=True)
        
        trends = agent.metrics_collector.get_performance_trends()
        if trends and "recent_average" in trends:
            st.subheader("Performance Trends")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recent Avg Time", f"{trends['recent_average']:.2f}s")
            with col2:
                st.metric("Min Time", f"{trends['recent_min']:.2f}s")
            with col3:
                st.metric("Max Time", f"{trends['recent_max']:.2f}s")
            
            if "performance_change" in trends:
                st.info(f"Performance Change: {trends['performance_change']}")
    
    with tabs[3]:
        st.subheader("🔍 Analysis History")
        
        limit = st.slider("Number of records to display", 5, 50, 10)
        
        if st.button("📜 Load History"):
            with st.spinner("Loading history..."):
                try:
                    history = asyncio.run(agent.get_analysis_history(limit=limit))
                    
                    if history:
                        for item in history:
                            metadata = item.get("metadata", {})
                            
                            with st.expander(f"📄 {metadata.get('file_path', 'Unknown')} - {metadata.get('timestamp', '')}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Overall Score", f"{metadata.get('overall_score', 0):.1f}")
                                
                                with col2:
                                    st.metric("Status", metadata.get("status", "unknown"))
                                
                                if item.get("document"):
                                    st.json(item["document"])
                    else:
                        st.info("No history available")
                
                except Exception as e:
                    st.error(f"Error loading history: {str(e)}")
        
        st.divider()
        
        st.subheader("🔎 Vector Search")
        
        search_query = st.text_input("Search for similar resumes", placeholder="Enter skills or keywords...")
        
        if st.button("🔍 Search") and search_query:
            with st.spinner("Searching..."):
                try:
                    results = asyncio.run(
                        agent.vector_store_manager.search_similar(search_query, n_results=5)
                    )
                    
                    if results:
                        for result in results:
                            with st.expander(f"Match: {result['similarity_score']:.2f}"):
                                st.json(result["metadata"])
                    else:
                        st.info("No similar resumes found")
                
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
    
    with tabs[4]:
        st.subheader("⚙️ Settings & Configuration")
        
        with st.expander("🔧 Agent Configuration"):
            st.json({
                "agent_id": config.agent_id,
                "agent_name": config.agent_name,
                "version": config.version,
                "max_file_size_mb": config.max_file_size_mb,
                "processing_timeout_seconds": config.processing_timeout_seconds,
                "max_concurrent_models": config.max_concurrent_models
            })
        
        with st.expander("🤖 Available Models"):
            for category, models in available_models.items():
                st.write(f"**{category.upper()}:**")
                for model in models:
                    st.write(f"• {model}")
        
        with st.expander("📊 System Metrics"):
            metrics_export = agent.metrics_collector.export_metrics("text")
            st.text(metrics_export)
            
            st.download_button(
                "📥 Export Metrics",
                metrics_export,
                f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )
        
        with st.expander("🔄 System Controls"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Reset Metrics"):
                    agent.metrics_collector.reset()
                    st.success("Metrics reset successfully")
            
            with col2:
                if st.button("🧹 Clear Cache"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Cache cleared successfully")

if __name__ == "__main__":
    main()