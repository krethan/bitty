"""
Bitty Configuration System
Handles all configuration for the autonomous development system
"""

import os
from typing import Optional, Dict, Any


def _yaml():
    """Import PyYAML lazily so `import bitty` works without it."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load or save YAML config files; "
            "install it with `pip install PyYAML` (or `pip install bitty[yaml]`)"
        ) from exc
    return yaml


class Config:
    """Configuration manager for Bitty"""
    
    def __init__(self, config_data: Dict[str, Any]):
        self._data = config_data
        
    @property
    def log_level(self) -> str:
        return self._data.get("log_level", "INFO")
    
    @property
    def memory_path(self) -> str:
        return os.path.expanduser(self._data.get("memory_path", "~/.bitty/memory"))
    
    @property
    def agent_configs(self) -> Dict[str, Dict[str, Any]]:
        return self._data.get("agents", {})
    
    @property
    def model_configs(self) -> Dict[str, Dict[str, Any]]:
        return self._data.get("models", {})
    
    @property
    def hardware(self) -> Dict[str, Any]:
        return self._data.get("hardware", {})
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from file"""
        if config_path is None:
            config_path = os.path.expanduser("~/.bitty/config.yaml")
        
        if not os.path.exists(config_path):
            return cls.default()
        
        with open(config_path, 'r') as f:
            config_data = _yaml().safe_load(f)
        
        return cls(config_data)
    
    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration"""
        return cls({
            "log_level": "INFO",
            "memory_path": "~/.bitty/memory",
            "agents": {
                "planner": {"enabled": True},
                "architect": {"enabled": True},
                "developer": {"enabled": True},
                "tester": {"enabled": True},
                "optimizer": {"enabled": True}
            },
            "models": {
                "default": {
                    "provider": "ollama",
                    "model": "llama3.1",
                    "quantization": "q4_0"
                },
                "local": {
                    "provider": "llama.cpp",
                    "model": "mistral-7b-q4_0.gguf",
                    "quantization": "q4_0"
                }
            },
            "hardware": {
                "gpu": {
                    "type": "auto",  # auto, nvidia, amd, cpu
                    "memory": "auto"
                },
                "quantization": {
                    "default": "q4_0",
                    "max_bits": 8
                }
            }
        })
    
    def save(self, config_path: Optional[str] = None):
        """Save configuration to file"""
        if config_path is None:
            config_path = os.path.expanduser("~/.bitty/config.yaml")
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            _yaml().dump(self._data, f)