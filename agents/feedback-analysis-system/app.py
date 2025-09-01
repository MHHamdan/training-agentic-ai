import streamlit as st
import pandas as pd
import os
from datetime import datetime
import json
from typing import List, Dict, Any
import time
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Feedback Analysis System",
    page_icon="📊",
    layout="wide"
)

class FeedbackAnalysisSystem:
    def __init__(self):
        self.setup_llm()
        self.setup_agents()
        self.data_dir = "agents/feedback-analysis-system/data"
        self.output_dir = "agents/feedback-analysis-system/outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_llm(self):
        """Initialize the LLM with HuggingFace models"""
        # Get HuggingFace token from environment
        hf_token = os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HF_TOKEN')
        
        if not hf_token:
            st.error("⚠️ HuggingFace token not found in environment variables. Please add HF_TOKEN or HUGGINGFACE_API_KEY to your .env file")
            st.stop()
        
        # Model selection - using powerful free models from HuggingFace
        model_options = {
            "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
            "Mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Zephyr-7B": "HuggingFaceH4/zephyr-7b-beta",
            "Falcon-7B": "tiiuae/falcon-7b-instruct",
            "Microsoft Phi-2": "microsoft/phi-2",
            "Google Flan-T5": "google/flan-t5-xxl",
            "Meta Llama-2-7B": "meta-llama/Llama-2-7b-chat-hf"
        }
        
        # Get selected model from session state or default
        selected_model = st.session_state.get('selected_hf_model', "mistralai/Mistral-7B-Instruct-v0.2")
        
        try:
            # Use HuggingFaceEndpoint for inference API
            self.llm = HuggingFaceEndpoint(
                repo_id=selected_model,
                huggingfacehub_api_token=hf_token,
                temperature=0.3,
                max_new_tokens=512,
                top_p=0.95,
                repetition_penalty=1.1,
                streaming=False
            )
            st.session_state['llm_initialized'] = True
            st.session_state['current_model'] = selected_model
        except Exception as e:
            st.error(f"Error initializing HuggingFace model: {str(e)}")
            # Fallback to a simpler model
            try:
                self.llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    huggingfacehub_api_token=hf_token,
                    model_kwargs={"temperature": 0.3, "max_length": 512}
                )
                st.session_state['current_model'] = "google/flan-t5-large (fallback)"
            except Exception as fallback_error:
                st.error(f"Failed to initialize fallback model: {str(fallback_error)}")
                st.stop()
    
    def setup_agents(self):
        """Initialize all agents for the feedback analysis system"""
        
        # CSV Reader Agent
        self.csv_reader_agent = Agent(
            role="CSV Data Reader",
            goal="Read and parse feedback data from CSV files accurately",
            backstory="""You are a data extraction specialist who reads CSV files 
            containing customer feedback from app stores and support emails. You ensure 
            all data is properly formatted and ready for analysis.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Feedback Classifier Agent  
        self.classifier_agent = Agent(
            role="Feedback Classifier",
            goal="Categorize feedback into Bug, Feature Request, Praise, Complaint, or Spam",
            backstory="""You are an NLP expert who specializes in understanding customer 
            feedback. You can accurately identify the intent and category of user messages 
            based on keywords, sentiment, and context.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Bug Analysis Agent
        self.bug_analyst_agent = Agent(
            role="Bug Analysis Specialist",
            goal="Extract technical details from bug reports including steps to reproduce, platform info, and severity",
            backstory="""You are a QA engineer who excels at identifying critical technical 
            information from bug reports. You extract device info, OS versions, app versions, 
            reproduction steps, and assess severity levels.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Feature Extractor Agent
        self.feature_agent = Agent(
            role="Feature Request Analyst",
            goal="Identify feature requests and estimate user impact/demand",
            backstory="""You are a product manager who understands user needs and can 
            identify valuable feature requests. You assess the potential impact and 
            demand for each feature based on user feedback.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Ticket Creator Agent
        self.ticket_creator_agent = Agent(
            role="Ticket Creation Specialist",
            goal="Generate structured tickets with proper formatting and priority",
            backstory="""You are a project management expert who creates well-structured 
            tickets for development teams. You ensure each ticket has clear titles, 
            descriptions, priorities, and all necessary metadata.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Quality Critic Agent
        self.quality_agent = Agent(
            role="Quality Assurance Critic",
            goal="Review generated tickets for completeness, accuracy, and consistency",
            backstory="""You are a senior QA lead who reviews tickets before they reach 
            the development team. You ensure tickets are complete, accurate, properly 
            prioritized, and follow company standards.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def process_feedback(self, reviews_df: pd.DataFrame, emails_df: pd.DataFrame) -> Dict[str, Any]:
        """Process feedback data through the multi-agent system"""
        
        # Combine feedback sources
        feedback_data = self._combine_feedback_sources(reviews_df, emails_df)
        
        # For HuggingFace models, we'll use a simplified processing approach
        # due to token limits and API constraints
        results = self._process_with_hf_models(feedback_data)
        
        return results
    
    def _process_with_hf_models(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Process feedback using HuggingFace models with simplified approach"""
        tickets = []
        metrics = {
            'total_feedback': len(feedback_data),
            'bugs': 0,
            'features': 0,
            'praise': 0,
            'complaints': 0,
            'spam': 0,
            'tickets_created': 0
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, item in enumerate(feedback_data):
            progress = (idx + 1) / len(feedback_data)
            progress_bar.progress(progress)
            status_text.text(f"Processing feedback {idx + 1} of {len(feedback_data)}...")
            
            # Classify the feedback
            content = item.get('content', '')
            if not content:
                category = 'Complaint'  # Default for empty content
            else:
                category = self._classify_feedback_hf(str(content))
            
            # Ensure category is valid
            if not category:
                category = 'Complaint'
            
            # Fix plural form for 'praise' - it doesn't add 's'
            if category.lower() == 'praise':
                metrics['praise'] += 1
            else:
                metrics[category.lower() + 's'] += 1
            
            # Create ticket for bugs and features
            if category in ['Bug', 'Feature']:
                ticket = self._create_ticket(item, category)
                tickets.append(ticket)
                metrics['tickets_created'] += 1
        
        progress_bar.empty()
        status_text.empty()
        
        # Save tickets to CSV
        if tickets:
            tickets_df = pd.DataFrame(tickets)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tickets_file = os.path.join(self.output_dir, f'generated_tickets_{timestamp}.csv')
            tickets_df.to_csv(tickets_file, index=False)
            st.success(f"✅ Saved {len(tickets)} tickets to {tickets_file}")
        
        return {
            'tickets': tickets,
            'metrics': metrics,
            'processing_complete': True
        }
    
    def _classify_feedback_hf(self, content: str) -> str:
        """Classify feedback using HuggingFace model"""
        try:
            # Skip HuggingFace API if content is empty
            if not content or len(content.strip()) == 0:
                return self._simple_classify("")
            
            # Create a simpler classification prompt for better model compatibility
            prompt = f"Classify as Bug, Feature, Praise, Complaint, or Spam: {content[:300]}"
            
            # Get classification from model with timeout
            try:
                response = self.llm.invoke(prompt)
            except Exception:
                # If model fails, use keyword classification immediately
                return self._simple_classify(content.lower())
            
            # Handle None or empty response
            if not response:
                return self._simple_classify(content.lower())
            
            # Parse the response safely
            if isinstance(response, str):
                response_lower = response.lower()
            else:
                # Try to convert to string if it's another type
                try:
                    response_lower = str(response).lower()
                except:
                    return self._simple_classify(content.lower())
            
            # Check for category keywords in response
            if 'bug' in response_lower or 'error' in response_lower or 'crash' in response_lower:
                return 'Bug'
            elif 'feature' in response_lower or 'request' in response_lower or 'add' in response_lower:
                return 'Feature'
            elif 'praise' in response_lower or 'good' in response_lower or 'love' in response_lower:
                return 'Praise'
            elif 'spam' in response_lower:
                return 'Spam'
            elif 'complaint' in response_lower or 'bad' in response_lower:
                return 'Complaint'
            else:
                # Fallback to keyword-based classification
                return self._simple_classify(content.lower())
                
        except Exception as e:
            # Always fallback to simple classification if anything fails
            if content:
                return self._simple_classify(content.lower())
            else:
                return 'Complaint'  # Default for empty content
    
    def _combine_feedback_sources(self, reviews_df: pd.DataFrame, emails_df: pd.DataFrame) -> List[Dict]:
        """Combine app store reviews and support emails into unified format"""
        feedback_list = []
        
        # Process app store reviews
        if not reviews_df.empty:
            for _, row in reviews_df.iterrows():
                feedback_list.append({
                    'id': row.get('review_id', f'R{_}'),
                    'source': 'app_store',
                    'platform': row.get('platform', 'Unknown'),
                    'type': 'review',
                    'rating': row.get('rating', None),
                    'title': f"App Store Review - Rating: {row.get('rating', 'N/A')}",
                    'content': row.get('review_text', ''),
                    'user': row.get('user_name', 'Anonymous'),
                    'date': row.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'version': row.get('app_version', 'Unknown'),
                    'priority_hint': row.get('priority', None)
                })
        
        # Process support emails
        if not emails_df.empty:
            for _, row in emails_df.iterrows():
                feedback_list.append({
                    'id': row.get('email_id', f'E{_}'),
                    'source': 'support_email',
                    'platform': 'Email',
                    'type': 'email',
                    'rating': None,
                    'title': row.get('subject', 'No Subject'),
                    'content': row.get('body', ''),
                    'user': row.get('sender_email', 'unknown@email.com'),
                    'date': row.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'version': 'N/A',
                    'priority_hint': row.get('priority', None)
                })
        
        return feedback_list
    
    def _simple_classify(self, content: str) -> str:
        """Simple keyword-based classification as fallback"""
        # Handle None or empty content
        if not content:
            return 'Complaint'  # Default for empty content
        
        # Ensure content is string and lowercase
        try:
            content = str(content).lower()
        except:
            return 'Complaint'
        
        bug_keywords = ['crash', 'error', 'bug', 'broken', 'not working', 'fail', 'issue', 'problem', 'stuck', 'freeze']
        feature_keywords = ['please add', 'would love', 'feature request', 'suggestion', 'could you', 'wish', 'want', 'need', 'missing']
        praise_keywords = ['amazing', 'love', 'great', 'perfect', 'excellent', 'best', 'awesome', 'fantastic', 'wonderful', 'good']
        spam_keywords = ['scam', 'check out', 'profile', 'qwerty', 'asdfgh', 'spam', 'fake', 'xxx', 'click here']
        complaint_keywords = ['expensive', 'slow', 'poor', 'bad', 'terrible', 'worst', 'hate', 'disappointed', 'frustrat']
        
        # Check keywords in order of priority
        if any(keyword in content for keyword in bug_keywords):
            return 'Bug'
        elif any(keyword in content for keyword in feature_keywords):
            return 'Feature'
        elif any(keyword in content for keyword in praise_keywords):
            return 'Praise'
        elif any(keyword in content for keyword in spam_keywords):
            return 'Spam'
        elif any(keyword in content for keyword in complaint_keywords):
            return 'Complaint'
        else:
            # Try to infer from rating if available (handled elsewhere)
            return 'Complaint'  # Default category
    
    def _create_ticket(self, item: Dict, category: str) -> Dict:
        """Create a structured ticket from feedback item"""
        
        # Determine priority
        priority = self._determine_priority(item, category)
        
        # Generate unique ticket ID
        ticket_id = f"TKT-{category[:3].upper()}-{hashlib.md5(item['id'].encode()).hexdigest()[:8].upper()}"
        
        return {
            'ticket_id': ticket_id,
            'title': self._generate_title(item, category),
            'category': category,
            'priority': priority,
            'description': item['content'][:1000],  # Limit description length
            'source': item['source'],
            'platform': item['platform'],
            'user': item['user'],
            'date_reported': item['date'],
            'app_version': item.get('version', 'Unknown'),
            'rating': item.get('rating', 'N/A'),
            'status': 'Open',
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _determine_priority(self, item: Dict, category: str) -> str:
        """Determine ticket priority based on content and metadata"""
        content_lower = item['content'].lower()
        
        # Critical keywords
        if any(word in content_lower for word in ['critical', 'urgent', 'data loss', 'security', 'crash']):
            return 'Critical'
        
        # High priority indicators
        if category == 'Bug' and item.get('rating', 5) <= 2:
            return 'High'
        
        if item.get('priority_hint', '').lower() in ['high', 'critical']:
            return 'High'
        
        # Medium priority
        if category == 'Feature' or item.get('rating', 5) == 3:
            return 'Medium'
        
        return 'Low'
    
    def _generate_title(self, item: Dict, category: str) -> str:
        """Generate a concise title for the ticket"""
        if item['type'] == 'email' and item.get('title'):
            return item['title'][:100]
        
        # Extract key issue from content
        content = item['content'][:100] if item['content'] else "No content"
        
        if category == 'Bug':
            return f"Bug: {content.split('.')[0]}"
        elif category == 'Feature':
            return f"Feature Request: {content.split('.')[0]}"
        else:
            return f"{category}: {content.split('.')[0]}"

def main():
    st.title("🎯 Intelligent Feedback Analysis System")
    st.markdown("Multi-Agent AI System for Customer Feedback Processing")
    st.info("🤗 Using FREE HuggingFace Models - No API costs!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 HuggingFace Model Selection")
        
        model_options = {
            "Mistral-7B (Recommended)": "mistralai/Mistral-7B-Instruct-v0.2",
            "Mixtral-8x7B (Powerful)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Zephyr-7B (Fast)": "HuggingFaceH4/zephyr-7b-beta",
            "Falcon-7B": "tiiuae/falcon-7b-instruct",
            "Flan-T5-XXL": "google/flan-t5-xxl",
            "Flan-T5-Large (Light)": "google/flan-t5-large"
        }
        
        selected_model_name = st.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            help="Select a free HuggingFace model for processing"
        )
        
        st.session_state['selected_hf_model'] = model_options[selected_model_name]
        
        # Display current model
        if 'current_model' in st.session_state:
            st.success(f"✅ Using: {st.session_state['current_model'].split('/')[-1]}")
        
        st.divider()
        
        # Processing options
        st.subheader("Processing Options")
        batch_size = st.slider("Batch size", 5, 50, 10)
        use_simple_mode = st.checkbox("Use Simple Classification (Faster)", value=True, 
                                     help="Uses keyword-based classification for faster processing")
        
        # Priority thresholds
        st.subheader("Priority Thresholds")
        critical_keywords = st.text_area("Critical Keywords", 
                                        value="critical, urgent, data loss, security, crash",
                                        help="Keywords that trigger Critical priority")
        high_keywords = st.text_area("High Priority Keywords",
                                    value="error, broken, fail, cannot, stuck",
                                    help="Keywords that trigger High priority")
        
        st.divider()
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **Mistral-7B**: Best balance of quality and speed
        - **Flan-T5**: Fastest but simpler responses
        - **Simple Mode**: Uses keywords for instant classification
        """)
    
    # Main interface
    tabs = st.tabs(["📥 Data Input", "🔄 Process", "📊 Analytics", "🎫 Tickets", "📋 Logs"])
    
    with tabs[0]:  # Data Input
        st.header("Upload Feedback Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("App Store Reviews")
            reviews_file = st.file_uploader("Upload CSV file", type=['csv'], key="reviews")
            
            if reviews_file:
                st.session_state['reviews_df'] = pd.read_csv(reviews_file)
                st.success(f"✅ Loaded {len(st.session_state['reviews_df'])} reviews")
            
            if st.button("Load Sample Reviews", key="load_reviews"):
                sample_path = "agents/feedback-analysis-system/data/app_store_reviews.csv"
                if os.path.exists(sample_path):
                    st.session_state['reviews_df'] = pd.read_csv(sample_path)
                    st.success("Sample reviews loaded!")
                else:
                    st.error("Sample file not found")
        
        with col2:
            st.subheader("Support Emails")
            emails_file = st.file_uploader("Upload CSV file", type=['csv'], key="emails")
            
            if emails_file:
                st.session_state['emails_df'] = pd.read_csv(emails_file)
                st.success(f"✅ Loaded {len(st.session_state['emails_df'])} emails")
            
            if st.button("Load Sample Emails", key="load_emails"):
                sample_path = "agents/feedback-analysis-system/data/support_emails.csv"
                if os.path.exists(sample_path):
                    st.session_state['emails_df'] = pd.read_csv(sample_path)
                    st.success("Sample emails loaded!")
                else:
                    st.error("Sample file not found")
        
        # Display loaded data
        if 'reviews_df' in st.session_state:
            st.subheader("📱 Loaded Reviews Preview")
            st.dataframe(st.session_state['reviews_df'].head())
            st.caption(f"Total: {len(st.session_state['reviews_df'])} reviews")
        
        if 'emails_df' in st.session_state:
            st.subheader("📧 Loaded Emails Preview")
            st.dataframe(st.session_state['emails_df'].head())
            st.caption(f"Total: {len(st.session_state['emails_df'])} emails")
    
    with tabs[1]:  # Process
        st.header("Process Feedback")
        
        # Check if HuggingFace token is configured
        hf_token = os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HF_TOKEN')
        if not hf_token:
            st.error("⚠️ HuggingFace token not found! Please add HF_TOKEN to your .env file")
            st.code("HF_TOKEN=your_huggingface_token_here", language="bash")
            st.stop()
        else:
            st.success("✅ HuggingFace token configured")
        
        if st.button("🚀 Start Processing", type="primary"):
            if 'reviews_df' not in st.session_state and 'emails_df' not in st.session_state:
                st.error("Please load feedback data first!")
            else:
                with st.spinner("Processing feedback with HuggingFace models..."):
                    # Initialize system
                    system = FeedbackAnalysisSystem()
                    
                    # Get dataframes
                    reviews_df = st.session_state.get('reviews_df', pd.DataFrame())
                    emails_df = st.session_state.get('emails_df', pd.DataFrame())
                    
                    # Process feedback
                    try:
                        results = system.process_feedback(reviews_df, emails_df)
                        st.session_state['processing_results'] = results
                        
                        # Display summary
                        st.success(f"✅ Processing complete! Created {results['metrics']['tickets_created']} tickets")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🐛 Bugs Found", results['metrics']['bugs'])
                        with col2:
                            st.metric("✨ Feature Requests", results['metrics']['features'])
                        with col3:
                            st.metric("📝 Total Processed", results['metrics']['total_feedback'])
                        
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        st.info("💡 Try using a simpler model like Flan-T5-Large or enable Simple Classification mode")
        
        # Quick Process with Simple Mode
        if st.button("⚡ Quick Process (Keyword-based)", help="Fast processing using keyword matching"):
            if 'reviews_df' not in st.session_state and 'emails_df' not in st.session_state:
                st.error("Please load feedback data first!")
            else:
                system = FeedbackAnalysisSystem()
                reviews_df = st.session_state.get('reviews_df', pd.DataFrame())
                emails_df = st.session_state.get('emails_df', pd.DataFrame())
                
                feedback_data = system._combine_feedback_sources(reviews_df, emails_df)
                
                tickets = []
                metrics = {
                    'total_feedback': len(feedback_data),
                    'bugs': 0,
                    'features': 0,
                    'praise': 0,
                    'complaints': 0,
                    'spam': 0,
                    'tickets_created': 0
                }
                
                for item in feedback_data:
                    # Safely get content
                    content = item.get('content', '')
                    if content:
                        category = system._simple_classify(content.lower())
                    else:
                        category = 'Complaint'  # Default for missing content
                    
                    # Fix plural form for 'praise' - it doesn't add 's'
                    if category.lower() == 'praise':
                        metrics['praise'] += 1
                    else:
                        metrics[category.lower() + 's'] += 1
                    
                    if category in ['Bug', 'Feature']:
                        ticket = system._create_ticket(item, category)
                        tickets.append(ticket)
                        metrics['tickets_created'] += 1
                
                st.session_state['processing_results'] = {
                    'tickets': tickets,
                    'metrics': metrics
                }
                
                st.success(f"✅ Quick processing complete! Created {metrics['tickets_created']} tickets")
    
    with tabs[2]:  # Analytics
        st.header("📊 Analytics Dashboard")
        
        if 'processing_results' in st.session_state:
            results = st.session_state['processing_results']
            metrics = results['metrics']
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Feedback", metrics['total_feedback'], delta=None)
            with col2:
                st.metric("Tickets Created", metrics['tickets_created'], 
                         delta=f"{(metrics['tickets_created']/metrics['total_feedback']*100):.1f}%")
            with col3:
                bug_rate = (metrics['bugs'] / metrics['total_feedback'] * 100) if metrics['total_feedback'] > 0 else 0
                st.metric("Bug Rate", f"{bug_rate:.1f}%", 
                         delta="High" if bug_rate > 30 else "Normal")
            with col4:
                feature_rate = (metrics['features'] / metrics['total_feedback'] * 100) if metrics['total_feedback'] > 0 else 0
                st.metric("Feature Requests", f"{feature_rate:.1f}%")
            
            # Category distribution
            st.subheader("Feedback Categories")
            
            import plotly.express as px
            import plotly.graph_objects as go
            
            categories = ['Bugs', 'Features', 'Praise', 'Complaints', 'Spam']
            values = [metrics['bugs'], metrics['features'], metrics['praise'], 
                     metrics['complaints'], metrics['spam']]
            colors = ['#ff4444', '#44ff44', '#4444ff', '#ffaa44', '#888888']
            
            fig = go.Figure(data=[go.Pie(labels=categories, values=values, 
                                        marker=dict(colors=colors),
                                        hovertemplate='%{label}: %{value}<br>%{percent}<extra></extra>')])
            fig.update_layout(title="Feedback Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Priority distribution for tickets
            if results.get('tickets'):
                tickets_df = pd.DataFrame(results['tickets'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Priority Distribution")
                    priority_counts = tickets_df['priority'].value_counts()
                    
                    fig2 = px.bar(x=priority_counts.index, y=priority_counts.values,
                                 labels={'x': 'Priority', 'y': 'Count'},
                                 color=priority_counts.index,
                                 color_discrete_map={'Critical': '#ff0000', 'High': '#ff8800',
                                                    'Medium': '#ffcc00', 'Low': '#00ff00'})
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    st.subheader("Source Distribution")
                    source_counts = tickets_df['source'].value_counts()
                    
                    fig3 = px.pie(values=source_counts.values, names=source_counts.index,
                                title="Feedback Sources")
                    st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No analytics data available. Please process feedback first.")
    
    with tabs[3]:  # Tickets
        st.header("🎫 Generated Tickets")
        
        if 'processing_results' in st.session_state and st.session_state['processing_results'].get('tickets'):
            tickets = st.session_state['processing_results']['tickets']
            tickets_df = pd.DataFrame(tickets)
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                category_filter = st.selectbox("Category", ['All'] + list(tickets_df['category'].unique()))
            with col2:
                priority_filter = st.selectbox("Priority", ['All'] + list(tickets_df['priority'].unique()))
            with col3:
                status_filter = st.selectbox("Status", ['All', 'Open', 'In Progress', 'Closed'])
            
            # Apply filters
            filtered_df = tickets_df.copy()
            if category_filter != 'All':
                filtered_df = filtered_df[filtered_df['category'] == category_filter]
            if priority_filter != 'All':
                filtered_df = filtered_df[filtered_df['priority'] == priority_filter]
            if status_filter != 'All':
                filtered_df = filtered_df[filtered_df['status'] == status_filter]
            
            # Display tickets
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv,
                    file_name=f"tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Convert to JIRA format
                jira_df = filtered_df[['title', 'description', 'priority', 'category']].copy()
                jira_df.columns = ['Summary', 'Description', 'Priority', 'Issue Type']
                jira_csv = jira_df.to_csv(index=False)
                st.download_button(
                    label="📧 Export for JIRA",
                    data=jira_csv,
                    file_name=f"jira_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # JSON export
                json_data = json.dumps(tickets, indent=2)
                st.download_button(
                    label="📋 Export as JSON",
                    data=json_data,
                    file_name=f"tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("No tickets generated yet. Please process feedback first.")
            st.markdown("""
            ### 📝 How to use:
            1. Go to **Data Input** tab
            2. Load sample data or upload your CSV files
            3. Come back to **Process** tab
            4. Click **Start Processing** or **Quick Process**
            5. View generated tickets here
            """)
    
    with tabs[4]:  # Logs
        st.header("📋 Processing Logs")
        
        # Check for output files
        output_dir = "agents/feedback-analysis-system/outputs"
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
            if files:
                st.subheader("📁 Generated Files")
                
                # Sort files by modification time
                files_with_time = [(f, os.path.getmtime(os.path.join(output_dir, f))) for f in files]
                files_with_time.sort(key=lambda x: x[1], reverse=True)
                
                for file, mtime in files_with_time[:10]:  # Show latest 10 files
                    file_path = os.path.join(output_dir, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    with col1:
                        st.text(f"📄 {file}")
                    with col2:
                        st.text(datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M'))
                    with col3:
                        st.text(f"{file_size:.2f} KB")
                    with col4:
                        with open(file_path, 'r') as f:
                            st.download_button(
                                label="⬇",
                                data=f.read(),
                                file_name=file,
                                key=f"download_{file}"
                            )
            else:
                st.info("No output files generated yet.")
        
        # Model Information
        st.subheader("🤖 Model Information")
        if 'current_model' in st.session_state:
            st.info(f"Current Model: {st.session_state['current_model']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🆓 Free Models Available:
            - **Mistral-7B**: Advanced instruction model
            - **Mixtral-8x7B**: Mixture of experts model
            - **Zephyr-7B**: Fine-tuned for helpfulness
            - **Falcon-7B**: Efficient transformer model
            - **Flan-T5**: Google's text-to-text model
            """)
        
        with col2:
            st.markdown("""
            ### 💡 Performance Tips:
            - Use **Simple Classification** for faster results
            - **Flan-T5-Large** is fastest but less accurate
            - **Mistral-7B** offers best quality/speed balance
            - Process in smaller batches to avoid timeouts
            - Consider time of day (less traffic = faster)
            """)

if __name__ == "__main__":
    main()