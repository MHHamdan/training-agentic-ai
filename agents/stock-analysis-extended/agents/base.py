"""Base agent class for the extended stock analysis system"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime
import logging
from crewai import Agent, LLM
from pydantic import BaseModel, Field
import asyncio

from config.settings import settings


class AgentState(TypedDict):
    """Base state for all agents"""
    task_id: str
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    errors: List[str]
    metadata: Dict[str, Any]
    status: str  # 'pending', 'processing', 'completed', 'failed'


class AgentResult(BaseModel):
    """Standard result format for all agents"""
    agent_name: str
    task_id: str
    status: str
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = None


class BaseStockAgent(ABC):
    """Abstract base class for all stock analysis agents"""
    
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[Any]] = None,
        llm: Optional[LLM] = None,
        verbose: bool = True,
        allow_delegation: bool = False,
        max_iterations: int = 5,
        memory: bool = True
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.max_iterations = max_iterations
        self.memory = memory
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Setup LLM
        self.llm = llm or self._setup_default_llm()
        
        # Create CrewAI agent
        self.agent = self._create_agent()
        
        # Performance metrics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0
    
    def _setup_default_llm(self) -> LLM:
        """Setup default LLM based on configuration"""
        provider = settings.agents.default_llm_provider
        
        if provider == 'openai' and settings.api.openai_api_key:
            return LLM(
                model="openai/gpt-4-turbo-preview",
                api_key=settings.api.openai_api_key,
                temperature=settings.agents.temperature,
                max_tokens=settings.agents.max_tokens
            )
        elif provider == 'google' and settings.api.google_api_key:
            return LLM(
                model="gemini/gemini-2.0-flash",
                api_key=settings.api.google_api_key,
                temperature=settings.agents.temperature,
                max_tokens=settings.agents.max_tokens
            )
        elif provider == 'anthropic' and settings.api.anthropic_api_key:
            return LLM(
                model="anthropic/claude-3-sonnet-20240229",
                api_key=settings.api.anthropic_api_key,
                temperature=settings.agents.temperature,
                max_tokens=settings.agents.max_tokens
            )
        else:
            # Fallback to a default model
            self.logger.warning(f"No API key found for {provider}, using fallback")
            return LLM(
                model="openai/gpt-3.5-turbo",
                temperature=settings.agents.temperature,
                max_tokens=settings.agents.max_tokens
            )
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent instance"""
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=self.verbose,
            allow_delegation=self.allow_delegation,
            tools=self.tools,
            llm=self.llm,
            max_iter=self.max_iterations,
            memory=self.memory
        )
    
    @abstractmethod
    async def analyze(self, input_data: Dict[str, Any]) -> AgentResult:
        """Abstract method for agent analysis - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for the agent"""
        pass
    
    async def execute(self, input_data: Dict[str, Any], task_id: str = None) -> AgentResult:
        """Execute the agent's analysis with error handling and metrics"""
        import uuid
        from time import time
        
        task_id = task_id or str(uuid.uuid4())
        start_time = time()
        
        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError(f"Invalid input data for {self.name}")
            
            # Log execution start
            self.logger.info(f"Starting analysis for task {task_id}")
            
            # Execute analysis
            result = await self.analyze(input_data)
            
            # Update metrics
            self.total_tasks += 1
            self.successful_tasks += 1
            processing_time = time() - start_time
            self.total_processing_time += processing_time
            
            # Update result with processing time
            result.processing_time = processing_time
            
            self.logger.info(f"Completed task {task_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Log error
            self.logger.error(f"Error in task {task_id}: {str(e)}")
            
            # Update metrics
            self.total_tasks += 1
            self.failed_tasks += 1
            processing_time = time() - start_time
            
            # Return error result
            return AgentResult(
                agent_name=self.name,
                task_id=task_id,
                status='failed',
                errors=[str(e)],
                processing_time=processing_time
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        success_rate = (
            self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0
        )
        avg_processing_time = (
            self.total_processing_time / self.total_tasks if self.total_tasks > 0 else 0
        )
        
        return {
            'agent_name': self.name,
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': success_rate,
            'average_processing_time': avg_processing_time,
            'total_processing_time': self.total_processing_time
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', role='{self.role}')"