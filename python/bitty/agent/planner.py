"""
Planner Agent for Bitty
Responsible for task decomposition, milestone creation, and roadmap management
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from bitty.agent import BaseAgent


@dataclass
class Task:
    """Represents a development task"""
    id: str
    title: str
    description: str
    priority: int
    dependencies: List[str]
    estimated_effort: str  # small, medium, large
    tags: List[str]


class PlannerAgent(BaseAgent):
    """Plans development work by breaking goals into tasks"""
    
    async def plan(self, goal: str) -> List[Task]:
        """Break down a high-level goal into actionable tasks"""
        self.logger.info(f"Planning for goal: {goal}")
        
        # This is where the AI would break down the goal
        # For now, we'll return a structured template
        
        tasks = [
            Task(
                id="task-1",
                title="Analyze requirements",
                description=f"Analyze the requirements for: {goal}",
                priority=1,
                dependencies=[],
                estimated_effort="medium",
                tags=["analysis"]
            ),
            Task(
                id="task-2",
                title="Design architecture",
                description="Create system architecture design",
                priority=2,
                dependencies=["task-1"],
                estimated_effort="large",
                tags=["architecture"]
            ),
            Task(
                id="task-3",
                title="Implement core functionality",
                description="Implement the main features",
                priority=3,
                dependencies=["task-2"],
                estimated_effort="large",
                tags=["implementation"]
            ),
            Task(
                id="task-4",
                title="Write tests",
                description="Create comprehensive test suite",
                priority=4,
                dependencies=["task-3"],
                estimated_effort="medium",
                tags=["testing"]
            ),
            Task(
                id="task-5",
                title="Optimize and review",
                description="Performance optimization and code review",
                priority=5,
                dependencies=["task-4"],
                estimated_effort="medium",
                tags=["optimization", "review"]
            ),
        ]
        
        # Store plan in memory
        self.memory.store(f"plan_{goal}", tasks)
        
        return tasks
    
    async def prioritize(self, tasks: List[Task]) -> List[Task]:
        """Prioritize tasks based on dependencies and business value"""
        # Sort by priority, respecting dependencies
        return sorted(tasks, key=lambda t: t.priority)