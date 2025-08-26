"""Workflow orchestration manager for coordinating multi-agent analysis"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from crewai import Crew, Task, Agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.core.risk_assessor import RiskAssessmentAgent
from agents.core.sentiment_analyzer import SentimentAnalysisAgent
from agents.core.technical_analyst import TechnicalAnalysisAgent
from agents.base import AgentResult
from config.settings import settings


class WorkflowType(Enum):
    """Types of analysis workflows"""
    QUICK_SCAN = "quick_scan"
    COMPREHENSIVE = "comprehensive"
    RISK_FOCUSED = "risk_focused"
    TECHNICAL_FOCUSED = "technical_focused"
    SENTIMENT_FOCUSED = "sentiment_focused"
    PORTFOLIO = "portfolio"
    REAL_TIME = "real_time"


class ExecutionMode(Enum):
    """Execution modes for workflows"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    HIERARCHICAL = "hierarchical"


@dataclass
class WorkflowState:
    """State management for workflow execution"""
    workflow_id: str
    workflow_type: WorkflowType
    ticker: str
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    final_report: Optional[Dict[str, Any]] = None


class WorkflowOrchestrator:
    """Main orchestrator for multi-agent workflows"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize workflow definitions
        self.workflows = self._define_workflows()
        
        # Execution pool for parallel tasks
        self.executor = ThreadPoolExecutor(max_workers=settings.agents.max_parallel_agents)
        
        # Memory for stateful workflows
        self.memory = MemorySaver()
        
        # Active workflows tracking
        self.active_workflows: Dict[str, WorkflowState] = {}
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all available agents"""
        return {
            'risk_assessor': RiskAssessmentAgent(),
            'sentiment_analyzer': SentimentAnalysisAgent(),
            'technical_analyst': TechnicalAnalysisAgent(),
            # Additional agents would be initialized here
        }
    
    def _define_workflows(self) -> Dict[WorkflowType, Dict[str, Any]]:
        """Define workflow configurations"""
        return {
            WorkflowType.QUICK_SCAN: {
                'agents': ['technical_analyst', 'sentiment_analyzer'],
                'mode': ExecutionMode.PARALLEL,
                'timeout': 60
            },
            WorkflowType.COMPREHENSIVE: {
                'agents': ['risk_assessor', 'sentiment_analyzer', 'technical_analyst'],
                'mode': ExecutionMode.PARALLEL,
                'timeout': 300
            },
            WorkflowType.RISK_FOCUSED: {
                'agents': ['risk_assessor', 'technical_analyst'],
                'mode': ExecutionMode.SEQUENTIAL,
                'timeout': 180
            },
            WorkflowType.TECHNICAL_FOCUSED: {
                'agents': ['technical_analyst'],
                'mode': ExecutionMode.SEQUENTIAL,
                'timeout': 120
            },
            WorkflowType.SENTIMENT_FOCUSED: {
                'agents': ['sentiment_analyzer'],
                'mode': ExecutionMode.SEQUENTIAL,
                'timeout': 120
            }
        }
    
    async def execute_workflow(
        self,
        ticker: str,
        workflow_type: WorkflowType = WorkflowType.COMPREHENSIVE,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """Execute a complete analysis workflow"""
        import uuid
        
        # Create workflow state
        workflow_id = str(uuid.uuid4())
        state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            ticker=ticker,
            start_time=datetime.now()
        )
        
        self.active_workflows[workflow_id] = state
        
        try:
            # Get workflow configuration
            workflow_config = self.workflows.get(workflow_type)
            if not workflow_config:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            # Update state
            state.status = "running"
            state.metadata['config'] = workflow_config
            
            # Execute based on mode
            mode = workflow_config['mode']
            agents_to_run = workflow_config['agents']
            
            if mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(state, agents_to_run, ticker, custom_params)
            elif mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(state, agents_to_run, ticker, custom_params)
            elif mode == ExecutionMode.CONDITIONAL:
                await self._execute_conditional(state, agents_to_run, ticker, custom_params)
            else:
                await self._execute_hierarchical(state, agents_to_run, ticker, custom_params)
            
            # Generate final report
            state.final_report = await self._generate_final_report(state)
            
            # Update completion status
            state.status = "completed"
            state.end_time = datetime.now()
            
            self.logger.info(f"Workflow {workflow_id} completed successfully")
            
        except Exception as e:
            state.status = "failed"
            state.errors.append(str(e))
            state.end_time = datetime.now()
            self.logger.error(f"Workflow {workflow_id} failed: {str(e)}")
        
        finally:
            # Clean up active workflow
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
        
        return state
    
    async def _execute_parallel(
        self,
        state: WorkflowState,
        agents: List[str],
        ticker: str,
        params: Optional[Dict[str, Any]]
    ):
        """Execute agents in parallel"""
        tasks = []
        
        for agent_name in agents:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                input_data = {
                    'ticker': ticker,
                    'task_id': f"{state.workflow_id}_{agent_name}",
                    **(params or {})
                }
                tasks.append(agent.execute(input_data))
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store results
        for agent_name, result in zip(agents, results):
            if isinstance(result, Exception):
                state.errors.append(f"{agent_name}: {str(result)}")
                state.agent_results[agent_name] = AgentResult(
                    agent_name=agent_name,
                    task_id=f"{state.workflow_id}_{agent_name}",
                    status='failed',
                    errors=[str(result)]
                )
            else:
                state.agent_results[agent_name] = result
    
    async def _execute_sequential(
        self,
        state: WorkflowState,
        agents: List[str],
        ticker: str,
        params: Optional[Dict[str, Any]]
    ):
        """Execute agents sequentially"""
        previous_result = None
        
        for agent_name in agents:
            if agent_name not in self.agents:
                continue
            
            agent = self.agents[agent_name]
            
            # Prepare input with previous results if available
            input_data = {
                'ticker': ticker,
                'task_id': f"{state.workflow_id}_{agent_name}",
                **(params or {})
            }
            
            if previous_result:
                input_data['previous_analysis'] = previous_result.data
            
            try:
                result = await agent.execute(input_data)
                state.agent_results[agent_name] = result
                previous_result = result
            except Exception as e:
                state.errors.append(f"{agent_name}: {str(e)}")
                state.agent_results[agent_name] = AgentResult(
                    agent_name=agent_name,
                    task_id=f"{state.workflow_id}_{agent_name}",
                    status='failed',
                    errors=[str(e)]
                )
                # Continue with next agent even if one fails
    
    async def _execute_conditional(
        self,
        state: WorkflowState,
        agents: List[str],
        ticker: str,
        params: Optional[Dict[str, Any]]
    ):
        """Execute agents with conditional branching"""
        # Start with technical analysis
        if 'technical_analyst' in agents and 'technical_analyst' in self.agents:
            tech_result = await self.agents['technical_analyst'].execute({
                'ticker': ticker,
                'task_id': f"{state.workflow_id}_technical"
            })
            state.agent_results['technical_analyst'] = tech_result
            
            # Check technical signals
            if tech_result.status == 'completed' and tech_result.data:
                signals = tech_result.data.get('trading_signals', {})
                recommendation = signals.get('recommendation', 'HOLD')
                
                # If strong signal, run risk assessment
                if 'STRONG' in recommendation:
                    if 'risk_assessor' in self.agents:
                        risk_result = await self.agents['risk_assessor'].execute({
                            'ticker': ticker,
                            'task_id': f"{state.workflow_id}_risk",
                            'triggered_by': 'strong_signal'
                        })
                        state.agent_results['risk_assessor'] = risk_result
                
                # Always run sentiment for additional confirmation
                if 'sentiment_analyzer' in self.agents:
                    sent_result = await self.agents['sentiment_analyzer'].execute({
                        'ticker': ticker,
                        'task_id': f"{state.workflow_id}_sentiment"
                    })
                    state.agent_results['sentiment_analyzer'] = sent_result
    
    async def _execute_hierarchical(
        self,
        state: WorkflowState,
        agents: List[str],
        ticker: str,
        params: Optional[Dict[str, Any]]
    ):
        """Execute agents in hierarchical structure with supervisor"""
        # This would implement a more complex hierarchical execution
        # For now, fallback to parallel execution
        await self._execute_parallel(state, agents, ticker, params)
    
    async def _generate_final_report(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate comprehensive final report from all agent results"""
        report = {
            'workflow_id': state.workflow_id,
            'ticker': state.ticker,
            'workflow_type': state.workflow_type.value,
            'execution_time': (state.end_time - state.start_time).total_seconds() if state.end_time and state.start_time else None,
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_analysis': {},
            'recommendations': [],
            'risk_warnings': [],
            'confidence_score': 0
        }
        
        # Aggregate results from all agents
        total_confidence = 0
        confidence_count = 0
        
        for agent_name, result in state.agent_results.items():
            if result.status == 'completed' and result.data:
                report['detailed_analysis'][agent_name] = result.data
                
                # Extract key information based on agent type
                if agent_name == 'risk_assessor':
                    risk_score = result.data.get('risk_score', 50)
                    report['summary']['risk_score'] = risk_score
                    report['risk_warnings'].extend(result.data.get('recommendations', []))
                    
                elif agent_name == 'sentiment_analyzer':
                    sentiment = result.data.get('composite_sentiment', {})
                    report['summary']['sentiment_score'] = sentiment.get('score', 0)
                    report['summary']['sentiment_category'] = sentiment.get('category', 'NEUTRAL')
                    
                elif agent_name == 'technical_analyst':
                    signals = result.data.get('trading_signals', {})
                    report['summary']['technical_recommendation'] = signals.get('recommendation', 'HOLD')
                    report['recommendations'].append(f"Technical Analysis: {signals.get('recommendation', 'HOLD')}")
                
                # Calculate confidence
                if 'confidence' in result.data:
                    conf_value = result.data['confidence']
                    if isinstance(conf_value, str):
                        confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
                        total_confidence += confidence_map.get(conf_value, 0.5)
                    else:
                        total_confidence += float(conf_value)
                    confidence_count += 1
        
        # Calculate overall confidence
        if confidence_count > 0:
            report['confidence_score'] = round(total_confidence / confidence_count, 2)
        
        # Generate executive summary
        report['executive_summary'] = self._generate_executive_summary(report)
        
        return report
    
    def _generate_executive_summary(self, report: Dict[str, Any]) -> str:
        """Generate executive summary from report data"""
        summary_parts = []
        
        # Overall assessment
        summary_parts.append(f"Analysis for {report['ticker']} completed.")
        
        # Risk assessment
        if 'risk_score' in report['summary']:
            risk = report['summary']['risk_score']
            if risk > 70:
                summary_parts.append("HIGH RISK detected.")
            elif risk > 40:
                summary_parts.append("MODERATE RISK profile.")
            else:
                summary_parts.append("LOW RISK profile.")
        
        # Sentiment
        if 'sentiment_category' in report['summary']:
            summary_parts.append(f"Market sentiment is {report['summary']['sentiment_category']}.")
        
        # Technical recommendation
        if 'technical_recommendation' in report['summary']:
            summary_parts.append(f"Technical indicators suggest: {report['summary']['technical_recommendation']}.")
        
        # Confidence
        if report['confidence_score'] > 0.7:
            summary_parts.append("High confidence in analysis.")
        elif report['confidence_score'] > 0.4:
            summary_parts.append("Moderate confidence in analysis.")
        else:
            summary_parts.append("Low confidence - additional research recommended.")
        
        return " ".join(summary_parts)
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running or completed workflow"""
        if workflow_id in self.active_workflows:
            state = self.active_workflows[workflow_id]
            return {
                'workflow_id': workflow_id,
                'status': state.status,
                'ticker': state.ticker,
                'workflow_type': state.workflow_type.value,
                'start_time': state.start_time.isoformat() if state.start_time else None,
                'agents_completed': len(state.agent_results),
                'errors': state.errors
            }
        return None
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id in self.active_workflows:
            state = self.active_workflows[workflow_id]
            state.status = "cancelled"
            state.end_time = datetime.now()
            del self.active_workflows[workflow_id]
            return True
        return False
    
    def get_active_workflows(self) -> List[str]:
        """Get list of active workflow IDs"""
        return list(self.active_workflows.keys())
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)