"""
Bitty - Autonomous AI Development System
Main entry point and orchestration layer
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from bitty.config import Config
from bitty.memory import MemorySystem
from bitty.agent.planner import PlannerAgent
from bitty.agent.architect import ArchitectAgent
from bitty.agent.developer import DeveloperAgent
from bitty.agent.tester import TesterAgent
from bitty.agent.optimizer import OptimizerAgent


class Bitty:
    """Main Bitty orchestrator coordinating all agents"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config.load(config_path) if config_path else Config.default()
        self.logger = self._setup_logging()
        self.memory = MemorySystem(self.config.memory_path)
        
        # Initialize agents
        self.planner = PlannerAgent(self.config, self.memory, self.logger)
        self.architect = ArchitectAgent(self.config, self.memory, self.logger)
        self.developer = DeveloperAgent(self.config, self.memory, self.logger)
        self.tester = TesterAgent(self.config, self.memory, self.logger)
        self.optimizer = OptimizerAgent(self.config, self.memory, self.logger)
        
        self.running = False
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("bitty")
        logger.setLevel(self.config.log_level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    async def initialize(self):
        """Initialize all systems"""
        self.logger.info("Initializing Bitty Autonomous AI Development System")
        await self.memory.initialize()
        self.logger.info("Memory system initialized")
        
    async def run_development_cycle(self, goal: str):
        """Run a single development cycle"""
        self.logger.info(f"Starting development cycle for goal: {goal}")
        
        # 1. Planner: Break down goal into tasks
        tasks = await self.planner.plan(goal)
        self.logger.info(f"Planner created {len(tasks)} tasks")
        
        # 2. Architect: Design system for tasks
        design = await self.architect.design(tasks)
        self.logger.info("Architect completed system design")
        
        # 3. Developer: Implement code
        implementation = await self.developer.implement(design)
        self.logger.info("Developer completed implementation")
        
        # 4. Tester: Create and run tests
        test_results = await self.tester.test(implementation)
        self.logger.info(f"Tests completed: {test_results.passed}/{test_results.total} passed")
        
        # 5. Optimizer: Profile and optimize
        optimized = await self.optimizer.optimize(implementation, test_results)
        self.logger.info("Optimization complete")
        
        # 6. Update memory with results
        await self.memory.store_cycle(goal, tasks, design, implementation, test_results, optimized)
        
        return optimized
    
    async def run_continuous(self, goals: list[str]):
        """Run continuous development loop"""
        self.running = True
        self.logger.info("Starting continuous development loop")
        
        for goal in goals:
            if not self.running:
                break
            try:
                await self.run_development_cycle(goal)
            except Exception as e:
                self.logger.error(f"Error in development cycle: {e}")
                await self.memory.store_error(goal, str(e))
        
        self.logger.info("Development loop completed")
    
    def stop(self):
        """Stop the development loop"""
        self.running = False
        self.logger.info("Stopping development loop")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bitty - Autonomous AI Development System")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--goal", help="Single goal to execute")
    parser.add_argument("--goals", nargs="+", help="Multiple goals for continuous mode")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    
    args = parser.parse_args()
    
    bitty = Bitty(args.config)
    await bitty.initialize()
    
    if args.goal:
        await bitty.run_development_cycle(args.goal)
    elif args.goals and args.continuous:
        await bitty.run_continuous(args.goals)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())