"""
Base Agent class for Bitty
All agents inherit from this base class
"""

import logging
from typing import Optional
from bitty.config import Config
from bitty.memory import MemorySystem


class BaseAgent:
    """Base class for all Bitty agents"""
    
    def __init__(self, config: Config, memory: MemorySystem, logger: logging.Logger):
        self.config = config
        self.memory = memory
        self.logger = logger
        
    async def initialize(self):
        """Initialize agent-specific resources"""
        pass