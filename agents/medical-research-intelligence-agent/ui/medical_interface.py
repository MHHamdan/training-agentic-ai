"""
Medical Research Interface Components for MARIA
Healthcare-specific UI components and visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

def create_medical_research_interface():
    """Create medical research interface components"""
    return {
        "medical_dashboard": create_medical_dashboard,
        "research_metrics": create_research_metrics_display,
        "literature_analysis": create_literature_analysis_view,
        "clinical_trials": create_clinical_trials_view,
        "drug_interaction": create_drug_interaction_view,
        "safety_alerts": create_safety_alerts_display,
        "evidence_quality": create_evidence_quality_indicator,
        "medical_timeline": create_medical_timeline_view
    }


def create_medical_dashboard(research_data: Dict[str, Any]) -> None:
    """
    Create medical research dashboard
    
    Args:
        research_data: Medical research data
    """
    try:
        st.subheader("🏥 Medical Research Dashboard")
        
        # Research overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_analyses = len(research_data.get('conversation', []))
            st.metric("Medical Analyses", total_analyses, delta=None)
        
        with col2:
            research_state = research_data.get('research_state', {})
            confidence_scores = research_state.get('confidence_scores', {})
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}", delta=None)
        
        with col3:
            validated_sources = len(research_state.get('validated_sources', []))
            st.metric("Validated Sources", validated_sources, delta=None)
        
        with col4:
            pending_approvals = len(research_state.get('pending_approvals', []))
            if pending_approvals > 0:
                st.metric("Pending Reviews", pending_approvals, delta=pending_approvals, delta_color="inverse")
            else:
                st.metric("Pending Reviews", "0", delta=None)
        
        # Medical research progress
        st.subheader("📊 Research Progress")
        
        # Create progress chart
        if research_data.get('conversation'):
            progress_data = _create_research_progress_data(research_data)
            
            if progress_data:
                progress_df = pd.DataFrame(progress_data)
                
                fig = px.line(
                    progress_df, 
                    x='timestamp', 
                    y='confidence_score',
                    title='Medical Research Confidence Over Time',
                    labels={'confidence_score': 'Confidence Score', 'timestamp': 'Analysis Timeline'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Medical entities analysis
        st.subheader("🔬 Medical Entities Analysis")
        
        medical_entities = _extract_medical_entities(research_data)
        if medical_entities:
            _display_medical_entities_chart(medical_entities)
        else:
            st.info("No medical entities detected in current analysis")
        
    except Exception as e:
        st.error(f"Error creating medical dashboard: {str(e)}")


def create_research_metrics_display(research_data: Dict[str, Any]) -> None:
    """
    Create research metrics display
    
    Args:
        research_data: Medical research data
    """
    try:
        st.subheader("📈 Research Metrics")
        
        # Research quality metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Evidence Quality Distribution**")
            evidence_levels = _analyze_evidence_levels(research_data)
            
            if evidence_levels:
                evidence_df = pd.DataFrame(
                    list(evidence_levels.items()), 
                    columns=['Evidence Level', 'Count']
                )
                
                fig = px.pie(
                    evidence_df, 
                    values='Count', 
                    names='Evidence Level',
                    title='Evidence Quality Distribution'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No evidence quality data available")
        
        with col2:
            st.markdown("**Medical Safety Indicators**")
            safety_metrics = _analyze_safety_indicators(research_data)
            
            # Safety score gauge
            safety_score = safety_metrics.get('overall_safety_score', 0.5)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = safety_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Safety Score"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("**Detailed Research Metrics**")
        
        metrics_data = {
            "Metric": [
                "Total Medical Analyses",
                "Average Confidence Score", 
                "High-Confidence Findings",
                "Safety Flags Identified",
                "Professional Guidelines Referenced",
                "Peer-Reviewed Sources"
            ],
            "Value": [
                len(research_data.get('conversation', [])),
                f"{_calculate_average_confidence(research_data):.2f}",
                _count_high_confidence_findings(research_data),
                safety_metrics.get('safety_flags_count', 0),
                safety_metrics.get('guidelines_referenced', 0),
                safety_metrics.get('peer_reviewed_sources', 0)
            ],
            "Target": [
                "≥5",
                "≥0.80",
                "≥70%",
                "0",
                "≥3",
                "≥80%"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating research metrics display: {str(e)}")


def create_literature_analysis_view(literature_data: List[Dict[str, Any]]) -> None:
    """
    Create literature analysis view
    
    Args:
        literature_data: Literature search results
    """
    try:
        st.subheader("📚 Literature Analysis")
        
        if not literature_data:
            st.info("No literature data available for analysis")
            return
        
        # Literature summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Articles", len(literature_data))
        
        with col2:
            avg_confidence = sum(article.get('confidence_score', 0) for article in literature_data) / len(literature_data)
            st.metric("Avg Quality Score", f"{avg_confidence:.2f}")
        
        with col3:
            recent_articles = sum(1 for article in literature_data 
                                if _is_recent_article(article.get('publication_date', '')))
            st.metric("Recent Articles (5yr)", recent_articles)
        
        # Literature timeline
        st.markdown("**Publication Timeline**")
        
        timeline_data = _create_literature_timeline_data(literature_data)
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            
            fig = px.histogram(
                timeline_df,
                x='year',
                title='Literature Publication Timeline',
                labels={'year': 'Publication Year', 'count': 'Number of Articles'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Literature quality distribution
        st.markdown("**Literature Quality Analysis**")
        
        quality_data = _analyze_literature_quality(literature_data)
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            
            fig = px.box(
                quality_df,
                y='confidence_score',
                title='Literature Quality Score Distribution'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top articles table
        st.markdown("**Top Quality Articles**")
        
        top_articles = sorted(literature_data, 
                            key=lambda x: x.get('confidence_score', 0), 
                            reverse=True)[:10]
        
        if top_articles:
            articles_df = pd.DataFrame([
                {
                    "Title": article.get('title', 'N/A')[:60] + "...",
                    "Journal": article.get('journal', 'N/A'),
                    "Year": article.get('publication_date', 'N/A')[:4],
                    "Quality Score": f"{article.get('confidence_score', 0):.2f}",
                    "PMID": article.get('pmid', 'N/A')
                }
                for article in top_articles
            ])
            
            st.dataframe(articles_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating literature analysis view: {str(e)}")


def create_clinical_trials_view(trials_data: List[Dict[str, Any]]) -> None:
    """
    Create clinical trials view
    
    Args:
        trials_data: Clinical trials data
    """
    try:
        st.subheader("🧪 Clinical Trials Analysis")
        
        if not trials_data:
            st.info("No clinical trials data available")
            return
        
        # Trials overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trials", len(trials_data))
        
        with col2:
            active_trials = sum(1 for trial in trials_data 
                              if trial.get('status', '').lower() in ['recruiting', 'active'])
            st.metric("Active Trials", active_trials)
        
        with col3:
            completed_trials = sum(1 for trial in trials_data 
                                 if trial.get('status', '').lower() == 'completed')
            st.metric("Completed", completed_trials)
        
        with col4:
            total_enrollment = sum(int(str(trial.get('enrollment', '0')).replace(',', '')) 
                                 for trial in trials_data if trial.get('enrollment'))
            st.metric("Total Enrollment", f"{total_enrollment:,}")
        
        # Trial status distribution
        st.markdown("**Trial Status Distribution**")
        
        status_data = _analyze_trial_status(trials_data)
        if status_data:
            status_df = pd.DataFrame(
                list(status_data.items()),
                columns=['Status', 'Count']
            )
            
            fig = px.bar(
                status_df,
                x='Status',
                y='Count',
                title='Clinical Trial Status Distribution'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Phase distribution
        st.markdown("**Trial Phase Distribution**")
        
        phase_data = _analyze_trial_phases(trials_data)
        if phase_data:
            phase_df = pd.DataFrame(
                list(phase_data.items()),
                columns=['Phase', 'Count']
            )
            
            fig = px.pie(
                phase_df,
                values='Count',
                names='Phase',
                title='Clinical Trial Phase Distribution'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trials table
        st.markdown("**Clinical Trials Details**")
        
        if trials_data:
            trials_df = pd.DataFrame([
                {
                    "NCT ID": trial.get('nct_id', 'N/A'),
                    "Title": trial.get('title', 'N/A')[:50] + "...",
                    "Status": trial.get('status', 'N/A'),
                    "Phase": trial.get('phase', 'N/A'),
                    "Enrollment": trial.get('enrollment', 'N/A'),
                    "Sponsor": trial.get('sponsor', 'N/A')[:30] + "..."
                }
                for trial in trials_data[:20]  # Show first 20
            ])
            
            st.dataframe(trials_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating clinical trials view: {str(e)}")


def create_drug_interaction_view(interaction_data: Dict[str, Any]) -> None:
    """
    Create drug interaction view
    
    Args:
        interaction_data: Drug interaction analysis data
    """
    try:
        st.subheader("💊 Drug Interaction Analysis")
        
        if not interaction_data or not interaction_data.get('success'):
            st.info("No drug interaction data available")
            return
        
        # Interaction overview
        interactions = interaction_data.get('interactions', [])
        medications = interaction_data.get('medications_checked', [])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Medications Checked", len(medications))
        
        with col2:
            st.metric("Interactions Found", len(interactions))
        
        with col3:
            high_severity = sum(1 for interaction in interactions 
                              if interaction.get('severity') == 'high')
            if high_severity > 0:
                st.metric("High Severity", high_severity, delta_color="inverse")
            else:
                st.metric("High Severity", "0")
        
        # Medications list
        if medications:
            st.markdown("**Medications Analyzed**")
            medications_str = ", ".join(medications)
            st.info(f"Analyzed medications: {medications_str}")
        
        # Interactions found
        if interactions:
            st.markdown("**⚠️ Drug Interactions Detected**")
            
            for i, interaction in enumerate(interactions):
                with st.expander(f"Interaction {i+1}: {interaction.get('drug1')} ↔ {interaction.get('drug2')}"):
                    st.write(f"**Severity:** {interaction.get('severity', 'Unknown').title()}")
                    st.write(f"**Description:** {interaction.get('description', 'No description available')}")
                    st.write(f"**Recommendation:** {interaction.get('recommendation', 'Consult healthcare provider')}")
        else:
            st.success("✅ No significant drug interactions detected")
        
        # Interaction network (if multiple interactions)
        if len(interactions) > 1:
            st.markdown("**Interaction Network**")
            _create_interaction_network_chart(medications, interactions)
        
        # Safety warnings
        warnings = interaction_data.get('warnings', [])
        if warnings:
            st.markdown("**⚠️ Safety Warnings**")
            for warning in warnings:
                st.warning(warning)
        
        # Disclaimer
        disclaimer = interaction_data.get('disclaimer')
        if disclaimer:
            st.info(f"📋 **Disclaimer:** {disclaimer}")
        
    except Exception as e:
        st.error(f"Error creating drug interaction view: {str(e)}")


def create_safety_alerts_display(safety_data: Dict[str, Any]) -> None:
    """
    Create safety alerts display
    
    Args:
        safety_data: Safety analysis data
    """
    try:
        st.subheader("🚨 Medical Safety Alerts")
        
        # Safety overview
        high_risk_flags = safety_data.get('high_risk_flags', [])
        contraindications = safety_data.get('contraindications', [])
        warnings = safety_data.get('warnings', [])
        
        if not any([high_risk_flags, contraindications, warnings]):
            st.success("✅ No critical safety alerts identified")
            return
        
        # Alert summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if high_risk_flags:
                st.metric("High Risk Flags", len(high_risk_flags), delta_color="inverse")
            else:
                st.metric("High Risk Flags", "0")
        
        with col2:
            if contraindications:
                st.metric("Contraindications", len(contraindications), delta_color="inverse")
            else:
                st.metric("Contraindications", "0")
        
        with col3:
            if warnings:
                st.metric("Safety Warnings", len(warnings), delta_color="inverse")
            else:
                st.metric("Safety Warnings", "0")
        
        # Display alerts
        if high_risk_flags:
            st.markdown("**🔴 High Risk Flags**")
            for flag in high_risk_flags:
                st.error(f"⚠️ {flag}")
        
        if contraindications:
            st.markdown("**🚫 Contraindications**")
            for contraindication in contraindications:
                st.warning(f"🚫 {contraindication}")
        
        if warnings:
            st.markdown("**⚠️ Safety Warnings**")
            for warning in warnings:
                st.info(f"⚠️ {warning}")
        
    except Exception as e:
        st.error(f"Error creating safety alerts display: {str(e)}")


def create_evidence_quality_indicator(evidence_data: Dict[str, Any]) -> None:
    """
    Create evidence quality indicator
    
    Args:
        evidence_data: Evidence quality data
    """
    try:
        st.subheader("📊 Evidence Quality Assessment")
        
        overall_quality = evidence_data.get('overall_quality_score', 0.0)
        
        # Quality score gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = overall_quality * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Evidence Quality Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 85], 'color': "orange"},
                    {'range': [85, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality breakdown
        quality_components = evidence_data.get('quality_components', {})
        
        if quality_components:
            st.markdown("**Quality Component Breakdown**")
            
            components_df = pd.DataFrame([
                {
                    "Component": component.replace('_', ' ').title(),
                    "Score": f"{score:.2f}",
                    "Weight": f"{quality_components.get(f'{component}_weight', 1.0):.1f}"
                }
                for component, score in quality_components.items()
                if not component.endswith('_weight')
            ])
            
            st.dataframe(components_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating evidence quality indicator: {str(e)}")


def create_medical_timeline_view(timeline_data: List[Dict[str, Any]]) -> None:
    """
    Create medical timeline view
    
    Args:
        timeline_data: Medical timeline data
    """
    try:
        st.subheader("📅 Medical Research Timeline")
        
        if not timeline_data:
            st.info("No timeline data available")
            return
        
        # Create timeline chart
        timeline_df = pd.DataFrame(timeline_data)
        
        if 'timestamp' in timeline_df.columns and 'event' in timeline_df.columns:
            timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
            
            fig = px.scatter(
                timeline_df,
                x='timestamp',
                y='event',
                title='Medical Research Timeline',
                hover_data=['confidence_score'] if 'confidence_score' in timeline_df.columns else None
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline details
        st.markdown("**Timeline Details**")
        
        display_df = timeline_df.copy()
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating medical timeline view: {str(e)}")


# Helper functions

def _create_research_progress_data(research_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create research progress data for visualization"""
    conversation = research_data.get('conversation', [])
    progress_data = []
    
    for i, message in enumerate(conversation):
        if message.get('sender') == 'assistant':
            progress_data.append({
                'timestamp': message.get('timestamp', ''),
                'confidence_score': message.get('confidence_score', 0.0),
                'analysis_number': i + 1
            })
    
    return progress_data


def _extract_medical_entities(research_data: Dict[str, Any]) -> Dict[str, int]:
    """Extract medical entities from research data"""
    entities = {
        'treatments': 0,
        'medications': 0,
        'conditions': 0,
        'procedures': 0,
        'guidelines': 0
    }
    
    conversation = research_data.get('conversation', [])
    
    for message in conversation:
        content = message.get('content', '').lower()
        
        # Simple keyword-based entity extraction
        if any(word in content for word in ['treatment', 'therapy', 'intervention']):
            entities['treatments'] += 1
        
        if any(word in content for word in ['drug', 'medication', 'pharmaceutical']):
            entities['medications'] += 1
        
        if any(word in content for word in ['condition', 'disease', 'disorder']):
            entities['conditions'] += 1
        
        if any(word in content for word in ['procedure', 'surgery', 'operation']):
            entities['procedures'] += 1
        
        if any(word in content for word in ['guideline', 'recommendation', 'protocol']):
            entities['guidelines'] += 1
    
    return entities


def _display_medical_entities_chart(entities: Dict[str, int]) -> None:
    """Display medical entities chart"""
    entities_df = pd.DataFrame(
        list(entities.items()),
        columns=['Entity Type', 'Count']
    )
    
    fig = px.bar(
        entities_df,
        x='Entity Type',
        y='Count',
        title='Medical Entities Detected'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)


def _analyze_evidence_levels(research_data: Dict[str, Any]) -> Dict[str, int]:
    """Analyze evidence levels in research data"""
    # This would analyze actual evidence levels from content
    # For now, returning sample data
    return {
        'High Quality': 3,
        'Moderate Quality': 5,
        'Low Quality': 2,
        'Very Low Quality': 1
    }


def _analyze_safety_indicators(research_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze safety indicators"""
    # This would perform actual safety analysis
    # For now, returning sample metrics
    return {
        'overall_safety_score': 0.85,
        'safety_flags_count': 2,
        'guidelines_referenced': 4,
        'peer_reviewed_sources': 12
    }


def _calculate_average_confidence(research_data: Dict[str, Any]) -> float:
    """Calculate average confidence score"""
    conversation = research_data.get('conversation', [])
    scores = [msg.get('confidence_score', 0) for msg in conversation if msg.get('confidence_score')]
    return sum(scores) / len(scores) if scores else 0.0


def _count_high_confidence_findings(research_data: Dict[str, Any]) -> int:
    """Count high confidence findings"""
    conversation = research_data.get('conversation', [])
    return sum(1 for msg in conversation if msg.get('confidence_score', 0) >= 0.8)


def _is_recent_article(pub_date: str) -> bool:
    """Check if article is recent (within 5 years)"""
    try:
        if len(pub_date) >= 4:
            year = int(pub_date[:4])
            current_year = datetime.now().year
            return current_year - year <= 5
    except:
        pass
    return False


def _create_literature_timeline_data(literature_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create literature timeline data"""
    timeline_data = []
    
    for article in literature_data:
        pub_date = article.get('publication_date', '')
        if len(pub_date) >= 4:
            try:
                year = int(pub_date[:4])
                timeline_data.append({'year': year})
            except:
                continue
    
    return timeline_data


def _analyze_literature_quality(literature_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze literature quality"""
    return [
        {'confidence_score': article.get('confidence_score', 0)}
        for article in literature_data
        if article.get('confidence_score') is not None
    ]


def _analyze_trial_status(trials_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze trial status distribution"""
    status_counts = {}
    
    for trial in trials_data:
        status = trial.get('status', 'Unknown').title()
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return status_counts


def _analyze_trial_phases(trials_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze trial phase distribution"""
    phase_counts = {}
    
    for trial in trials_data:
        phase = trial.get('phase', 'Unknown')
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    return phase_counts


def _create_interaction_network_chart(medications: List[str], interactions: List[Dict[str, Any]]) -> None:
    """Create drug interaction network chart"""
    st.info("Interaction network visualization would be displayed here (requires network graph library)")