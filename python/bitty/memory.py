"""
Memory System for Bitty
Handles knowledge storage and retrieval for the autonomous development system
"""

import os
from pathlib import Path
import pickle
from typing import Dict, Any


class MemorySystem:
    """Memory system for Bitty"""
    
    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self._data: Dict[str, Any] = {}
        self.load()
        
    def load(self) -> None:
        """Load memory from file"""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'rb') as f:
                self._data = pickle.load(f)
        
    def save(self) -> None:
        """Save memory to file"""
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, 'wb') as f:
            pickle.dump(self._data, f)
        
    def store(self, key: str, value: Any) -> None:
        """Store data in memory"""
        self._data[key] = value
        self.save()
        
    def retrieve(self, key: str) -> Any:
        """Retrieve data from memory"""
        return self._data.get(key)
        
    def initialize(self):
        """Initialize memory"""
        self.store("initialized", True)
