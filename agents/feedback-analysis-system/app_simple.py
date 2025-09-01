import streamlit as st
import pandas as pd
import os
from datetime import datetime
import json
from typing import List, Dict, Any
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

class SimpleFeedbackAnalysisSystem:
    def __init__(self):
        self.data_dir = "agents/feedback-analysis-system/data"
        self.output_dir = "agents/feedback-analysis-system/outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_feedback(self, reviews_df: pd.DataFrame, emails_df: pd.DataFrame) -> Dict[str, Any]:
        """Process feedback data using keyword-based classification"""
        
        # Combine feedback sources
        feedback_data = self._combine_feedback_sources(reviews_df, emails_df)
        
        # Process with keyword classification
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
            
            # Classify using keywords
            content = str(item.get('content', '')).lower() if item.get('content') else ''
            category = self._classify_by_keywords(content, item.get('rating'))
            
            # Update metrics - handle special plurals
            category_lower = category.lower()
            if category_lower == 'praise':
                metrics['praise'] += 1
            elif category_lower == 'spam':
                metrics['spam'] += 1
            else:
                metrics[category_lower + 's'] += 1
            
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
                    'rating': row.get('rating'),
                    'title': f"App Store Review - Rating: {row.get('rating', 'N/A')}",
                    'content': str(row.get('review_text', '')),
                    'user': row.get('user_name', 'Anonymous'),
                    'date': row.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'version': row.get('app_version', 'Unknown'),
                    'priority_hint': row.get('priority')
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
                    'content': str(row.get('body', '')),
                    'user': row.get('sender_email', 'unknown@email.com'),
                    'date': row.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'version': 'N/A',
                    'priority_hint': row.get('priority')
                })
        
        return feedback_list
    
    def _classify_by_keywords(self, content: str, rating=None) -> str:
        """Classify feedback using keywords and rating"""
        
        if not content:
            # Use rating if available
            if rating and isinstance(rating, (int, float)):
                if rating <= 2:
                    return 'Bug'
                elif rating >= 4:
                    return 'Praise'
            return 'Complaint'
        
        # Keywords for classification
        bug_keywords = ['crash', 'error', 'bug', 'broken', 'not working', 'fail', 'issue', 
                       'problem', 'stuck', 'freeze', 'slow', 'lag', 'glitch', 'fault']
        
        feature_keywords = ['please add', 'would love', 'feature request', 'suggestion', 
                          'could you', 'wish', 'want', 'need', 'missing', 'should have',
                          'would be great', 'improvement', 'enhance', 'better if']
        
        praise_keywords = ['amazing', 'love', 'great', 'perfect', 'excellent', 'best', 
                         'awesome', 'fantastic', 'wonderful', 'good', 'helpful', 'useful',
                         'thank', 'appreciate', 'brilliant', 'outstanding']
        
        spam_keywords = ['scam', 'check out', 'profile', 'qwerty', 'asdfgh', 'spam', 
                        'fake', 'xxx', 'click here', 'visit', 'download now', 'free money']
        
        complaint_keywords = ['expensive', 'poor', 'bad', 'terrible', 'worst', 'hate', 
                            'disappointed', 'frustrat', 'annoying', 'useless', 'waste',
                            'refund', 'cancel', 'uninstall']
        
        # Priority order for checking
        if any(kw in content for kw in spam_keywords):
            return 'Spam'
        elif any(kw in content for kw in bug_keywords):
            return 'Bug'
        elif any(kw in content for kw in feature_keywords):
            return 'Feature'
        elif any(kw in content for kw in praise_keywords):
            return 'Praise'
        elif any(kw in content for kw in complaint_keywords):
            return 'Complaint'
        else:
            # Use rating as fallback
            if rating and isinstance(rating, (int, float)):
                if rating <= 2:
                    return 'Complaint'
                elif rating >= 4:
                    return 'Praise'
            return 'Complaint'
    
    def _create_ticket(self, item: Dict, category: str) -> Dict:
        """Create a structured ticket from feedback item"""
        
        # Determine priority
        priority = self._determine_priority(item, category)
        
        # Generate unique ticket ID
        ticket_id = f"TKT-{category[:3].upper()}-{hashlib.md5(str(item['id']).encode()).hexdigest()[:8].upper()}"
        
        return {
            'ticket_id': ticket_id,
            'title': self._generate_title(item, category),
            'category': category,
            'priority': priority,
            'description': str(item.get('content', 'No description'))[:1000],
            'source': item.get('source', 'Unknown'),
            'platform': item.get('platform', 'Unknown'),
            'user': item.get('user', 'Unknown'),
            'date_reported': item.get('date', datetime.now().strftime('%Y-%m-%d')),
            'app_version': item.get('version', 'Unknown'),
            'rating': item.get('rating', 'N/A'),
            'status': 'Open',
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _determine_priority(self, item: Dict, category: str) -> str:
        """Determine ticket priority"""
        content_lower = str(item.get('content', '')).lower()
        
        # Critical keywords
        if any(word in content_lower for word in ['critical', 'urgent', 'data loss', 'security', 'crash', 'cannot use']):
            return 'Critical'
        
        # High priority
        if category == 'Bug':
            rating = item.get('rating')
            if rating and isinstance(rating, (int, float)) and rating <= 2:
                return 'High'
            return 'Medium'
        
        # Feature requests are usually medium
        if category == 'Feature':
            return 'Medium'
        
        return 'Low'
    
    def _generate_title(self, item: Dict, category: str) -> str:
        """Generate ticket title"""
        if item['type'] == 'email' and item.get('title'):
            return str(item['title'])[:100]
        
        content = str(item.get('content', 'No content'))[:100]
        
        if '.' in content:
            content = content.split('.')[0]
        
        return f"{category}: {content}"

def main():
    st.title("🎯 Intelligent Feedback Analysis System")
    st.markdown("**Simple & Reliable Keyword-Based Processing**")
    st.success("✅ No API dependencies - 100% reliable processing!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.info("This is the simplified version that uses keyword-based classification for maximum reliability.")
        
        # Processing options
        st.subheader("Processing Options")
        show_details = st.checkbox("Show processing details", value=False)
        save_to_file = st.checkbox("Save results to file", value=True)
        
        st.divider()
        
        st.markdown("### 📊 Classification Rules")
        st.markdown("""
        **Bug**: crash, error, broken, fail
        **Feature**: request, add, wish, need
        **Praise**: amazing, love, great, perfect
        **Complaint**: bad, terrible, hate, disappointed
        **Spam**: scam, fake, spam
        """)
    
    # Main tabs
    tabs = st.tabs(["📥 Data Input", "🔄 Process", "📊 Analytics", "🎫 Tickets"])
    
    with tabs[0]:  # Data Input
        st.header("Upload Feedback Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("App Store Reviews")
            
            if st.button("📱 Load Sample Reviews"):
                sample_path = "agents/feedback-analysis-system/data/app_store_reviews.csv"
                if os.path.exists(sample_path):
                    st.session_state['reviews_df'] = pd.read_csv(sample_path)
                    st.success(f"Loaded {len(st.session_state['reviews_df'])} reviews")
                else:
                    st.error("Sample file not found")
            
            reviews_file = st.file_uploader("Or upload CSV", type=['csv'], key="reviews")
            if reviews_file:
                st.session_state['reviews_df'] = pd.read_csv(reviews_file)
                st.success(f"✅ Loaded {len(st.session_state['reviews_df'])} reviews")
        
        with col2:
            st.subheader("Support Emails")
            
            if st.button("📧 Load Sample Emails"):
                sample_path = "agents/feedback-analysis-system/data/support_emails.csv"
                if os.path.exists(sample_path):
                    st.session_state['emails_df'] = pd.read_csv(sample_path)
                    st.success(f"Loaded {len(st.session_state['emails_df'])} emails")
                else:
                    st.error("Sample file not found")
            
            emails_file = st.file_uploader("Or upload CSV", type=['csv'], key="emails")
            if emails_file:
                st.session_state['emails_df'] = pd.read_csv(emails_file)
                st.success(f"✅ Loaded {len(st.session_state['emails_df'])} emails")
        
        # Display loaded data
        if 'reviews_df' in st.session_state:
            with st.expander("📱 Review Data Preview"):
                st.dataframe(st.session_state['reviews_df'].head())
        
        if 'emails_df' in st.session_state:
            with st.expander("📧 Email Data Preview"):
                st.dataframe(st.session_state['emails_df'].head())
    
    with tabs[1]:  # Process
        st.header("Process Feedback")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Reviews", len(st.session_state.get('reviews_df', [])))
        with col2:
            st.metric("Emails", len(st.session_state.get('emails_df', [])))
        with col3:
            total = len(st.session_state.get('reviews_df', [])) + len(st.session_state.get('emails_df', []))
            st.metric("Total", total)
        
        if st.button("🚀 Process All Feedback", type="primary", disabled=(total == 0)):
            if 'reviews_df' not in st.session_state and 'emails_df' not in st.session_state:
                st.error("Please load feedback data first!")
            else:
                # Process feedback
                system = SimpleFeedbackAnalysisSystem()
                reviews_df = st.session_state.get('reviews_df', pd.DataFrame())
                emails_df = st.session_state.get('emails_df', pd.DataFrame())
                
                with st.spinner("Processing feedback..."):
                    results = system.process_feedback(reviews_df, emails_df)
                    st.session_state['results'] = results
                
                # Show results
                st.success(f"✅ Processed {results['metrics']['total_feedback']} items")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🐛 Bugs", results['metrics']['bugs'])
                with col2:
                    st.metric("✨ Features", results['metrics']['features'])
                with col3:
                    st.metric("👍 Praise", results['metrics']['praise'])
                with col4:
                    st.metric("🎫 Tickets", results['metrics']['tickets_created'])
    
    with tabs[2]:  # Analytics
        st.header("📊 Analytics Dashboard")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            metrics = results['metrics']
            
            # Pie chart of categories
            import plotly.graph_objects as go
            
            categories = ['Bugs', 'Features', 'Praise', 'Complaints', 'Spam']
            values = [metrics['bugs'], metrics['features'], metrics['praise'], 
                     metrics['complaints'], metrics['spam']]
            colors = ['#ff4444', '#44ff44', '#4444ff', '#ffaa44', '#888888']
            
            fig = go.Figure(data=[go.Pie(
                labels=categories, 
                values=values,
                marker=dict(colors=colors),
                hole=0.3
            )])
            fig.update_layout(title="Feedback Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Processing Summary")
                st.write(f"- Total Processed: **{metrics['total_feedback']}**")
                st.write(f"- Tickets Created: **{metrics['tickets_created']}**")
                st.write(f"- Conversion Rate: **{(metrics['tickets_created']/max(metrics['total_feedback'],1)*100):.1f}%**")
            
            with col2:
                st.subheader("🏆 Top Categories")
                sorted_cats = sorted(
                    [(k.replace('s','').title(), v) for k, v in metrics.items() 
                     if k not in ['total_feedback', 'tickets_created']],
                    key=lambda x: x[1], reverse=True
                )
                for cat, count in sorted_cats[:3]:
                    st.write(f"- {cat}: **{count}**")
        else:
            st.info("No results yet. Process feedback first!")
    
    with tabs[3]:  # Tickets
        st.header("🎫 Generated Tickets")
        
        if 'results' in st.session_state and st.session_state['results'].get('tickets'):
            tickets_df = pd.DataFrame(st.session_state['results']['tickets'])
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                cat_filter = st.selectbox("Filter by Category", 
                                         ['All'] + list(tickets_df['category'].unique()))
            with col2:
                pri_filter = st.selectbox("Filter by Priority",
                                         ['All'] + list(tickets_df['priority'].unique()))
            
            # Apply filters
            filtered = tickets_df.copy()
            if cat_filter != 'All':
                filtered = filtered[filtered['category'] == cat_filter]
            if pri_filter != 'All':
                filtered = filtered[filtered['priority'] == pri_filter]
            
            # Display
            st.dataframe(filtered, use_container_width=True)
            
            # Export
            csv = filtered.to_csv(index=False)
            st.download_button(
                "📥 Download Tickets CSV",
                csv,
                f"tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.info("No tickets generated yet. Process feedback first!")

if __name__ == "__main__":
    main()