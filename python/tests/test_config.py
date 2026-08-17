"""Tests for the Config class."""

import os
import tempfile

import pytest

from bitty.config import Config


def test_default_config_has_required_keys():
    cfg = Config.default()
    assert cfg.log_level == "INFO"
    assert cfg.memory_path.startswith("~") or cfg.memory_path.startswith("/")
    assert "planner" in cfg.agent_configs
    assert "default" in cfg.model_configs
    assert "gpu" in cfg.hardware


def test_default_config_memory_path_expanded():
    cfg = Config.default()
    assert "~" not in cfg.memory_path, "memory_path should be expanded"


def test_load_nonexistent_returns_default(tmp_path):
    """Loading a config that doesn't exist returns the default."""
    cfg = Config.load(str(tmp_path / "missing.yaml"))
    assert cfg.log_level == "INFO"


def test_save_and_load_roundtrip(tmp_path):
    """A saved YAML config should load back with the same data."""
    config_path = str(tmp_path / "config.yaml")
    cfg = Config.default()
    cfg.save(config_path)

    reloaded = Config.load(config_path)
    assert reloaded.log_level == cfg.log_level
    assert reloaded.memory_path == cfg.memory_path
    assert "planner" in reloaded.agent_configs


def test_load_missing_yaml_raises_import_error(tmp_path, monkeypatch):
    """If PyYAML is unavailable, loading a real YAML config should raise a clear ImportError."""
    # Save a config so the file actually exists and load() reaches the YAML parser.
    config_path = str(tmp_path / "config.yaml")
    Config.default().save(config_path)

    # Block PyYAML import.
    import importlib
    import sys

    # Remove yaml from sys.modules if cached.
    monkeypatch.delitem(sys.modules, "yaml", raising=False)

    # Patch _yaml to raise the same ImportError as the real code path.
    from bitty import config as config_module

    def fake_yaml():
        raise ImportError(
            "PyYAML is required to load or save YAML config files; "
            "install it with `pip install PyYAML` (or `pip install bitty[yaml]`)"
        )

    monkeypatch.setattr(config_module, "_yaml", fake_yaml)

    with pytest.raises(ImportError, match="PyYAML is required"):
        Config.load(config_path)


def test_save_missing_yaml_raises_import_error(tmp_path, monkeypatch):
    """If PyYAML is unavailable, saving a config should raise a clear ImportError."""
    from bitty import config as config_module

    def fake_yaml():
        raise ImportError(
            "PyYAML is required to load or save YAML config files; "
            "install it with `pip install PyYAML` (or `pip install bitty[yaml]`)"
        )

    monkeypatch.setattr(config_module, "_yaml", fake_yaml)

    with pytest.raises(ImportError, match="PyYAML is required"):
        Config.default().save(str(tmp_path / "config.yaml"))
