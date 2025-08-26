import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any

class UIComponents:
    
    @staticmethod
    def create_score_gauge(title: str, value: float, max_value: float = 100):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': UIComponents._get_color_for_score(value)},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _get_color_for_score(score: float) -> str:
        if score >= 80:
            return "green"
        elif score >= 70:
            return "yellowgreen"
        elif score >= 60:
            return "gold"
        elif score >= 50:
            return "orange"
        else:
            return "red"
    
    @staticmethod
    def create_radar_chart(scores: Dict[str, float], title: str = "Skills Assessment"):
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Score'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title=title
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_skills_matrix(skills: Dict[str, List[str]]):
        data = []
        for category, skill_list in skills.items():
            for skill in skill_list:
                data.append({
                    "Category": category.replace("_", " ").title(),
                    "Skill": skill,
                    "Present": "✅"
                })
        
        if data:
            df = pd.DataFrame(data)
            pivot_df = df.pivot_table(
                index="Skill",
                columns="Category",
                values="Present",
                aggfunc='first',
                fill_value=""
            )
            st.dataframe(pivot_df, use_container_width=True)
        else:
            st.info("No skills detected")
    
    @staticmethod
    def create_timeline_chart(history: List[Dict[str, Any]]):
        if not history:
            st.info("No history data available")
            return
        
        timeline_data = []
        for item in history:
            metadata = item.get("metadata", {})
            timeline_data.append({
                "Timestamp": metadata.get("timestamp", ""),
                "Score": metadata.get("overall_score", 0),
                "File": metadata.get("file_path", "").split("/")[-1] if metadata.get("file_path") else "Unknown"
            })
        
        df = pd.DataFrame(timeline_data)
        
        if not df.empty:
            fig = px.line(
                df,
                x="Timestamp",
                y="Score",
                title="Score Trends Over Time",
                markers=True,
                hover_data=["File"]
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Overall Score",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_comparison_table(analyses: Dict[str, Any]):
        comparison_data = []
        
        for model_name, analysis in analyses.items():
            if analysis.get("status") == "success":
                scores = analysis.get("analysis", {}).get("scores", {})
                comparison_data.append({
                    "Model": model_name.split("/")[-1],
                    "Technical": scores.get("technical_skills", 0),
                    "Experience": scores.get("experience_relevance", 0),
                    "Cultural": scores.get("cultural_fit", 0),
                    "Growth": scores.get("growth_potential", 0),
                    "Risk": scores.get("risk_assessment", 0),
                    "Overall": scores.get("overall", 0)
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            styled_df = df.style.background_gradient(
                subset=["Technical", "Experience", "Cultural", "Growth", "Risk", "Overall"],
                cmap="RdYlGn",
                vmin=0,
                vmax=100
            )
            
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("No comparison data available")
    
    @staticmethod
    def create_progress_indicator(current: int, total: int, label: str = "Progress"):
        progress = current / total if total > 0 else 0
        st.progress(progress, text=f"{label}: {current}/{total} ({progress*100:.1f}%)")
    
    @staticmethod
    def create_metric_card(title: str, value: Any, delta: Any = None, delta_color: str = "normal"):
        st.metric(label=title, value=value, delta=delta, delta_color=delta_color)
    
    @staticmethod
    def create_info_card(title: str, content: str, type: str = "info"):
        if type == "info":
            st.info(f"**{title}**\n\n{content}")
        elif type == "success":
            st.success(f"**{title}**\n\n{content}")
        elif type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif type == "error":
            st.error(f"**{title}**\n\n{content}")
    
    @staticmethod
    def create_download_section(data: Dict[str, Any], filename_prefix: str = "export"):
        import json
        from datetime import datetime
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_str = json.dumps(data, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            if isinstance(data, dict) and any(isinstance(v, (list, dict)) for v in data.values()):
                flattened = UIComponents._flatten_dict(data)
                df = pd.DataFrame([flattened])
            else:
                df = pd.DataFrame([data])
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            text_content = UIComponents._dict_to_text(data)
            st.download_button(
                label="📥 Download TXT",
                data=text_content,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(UIComponents._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def _dict_to_text(d: Dict, indent: int = 0) -> str:
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{'  ' * indent}{key}:")
                lines.append(UIComponents._dict_to_text(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{'  ' * indent}{key}:")
                for item in value:
                    lines.append(f"{'  ' * (indent + 1)}- {item}")
            else:
                lines.append(f"{'  ' * indent}{key}: {value}")
        return '\n'.join(lines)