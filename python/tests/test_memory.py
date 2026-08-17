"""Tests for the MemorySystem class."""

import asyncio
import inspect
import os
import pickle
import tempfile

import pytest

from bitty.memory import MemorySystem


@pytest.fixture
def tmp_memory_path(tmp_path):
    """Provide a temporary memory path under pytest's tmp_path."""
    return str(tmp_path / "memory.pkl")


def test_memory_init_loads_empty(tmp_memory_path):
    mem = MemorySystem(tmp_memory_path)
    assert mem._data == {}


def test_memory_init_loads_existing(tmp_path):
    path = tmp_path / "memory.pkl"
    data = {"key": "value", "cycle_1": {"foo": "bar"}}
    path.write_bytes(pickle.dumps(data))

    mem = MemorySystem(str(path))
    assert mem._data == data


def test_store_and_retrieve(tmp_memory_path):
    mem = MemorySystem(tmp_memory_path)
    mem.store("greeting", "hello")
    assert mem.retrieve("greeting") == "hello"


def test_store_persists_to_disk(tmp_memory_path):
    mem = MemorySystem(tmp_memory_path)
    mem.store("key", "value")

    reloaded = MemorySystem(tmp_memory_path)
    assert reloaded.retrieve("key") == "value"


def test_retrieve_missing_returns_none(tmp_memory_path):
    mem = MemorySystem(tmp_memory_path)
    assert mem.retrieve("never_set") is None


def test_save_creates_parent_directory(tmp_path):
    nested_path = tmp_path / "subdir" / "deeper" / "memory.pkl"
    mem = MemorySystem(str(nested_path))
    mem.store("test", 42)

    assert nested_path.exists()
    reloaded = MemorySystem(str(nested_path))
    assert reloaded.retrieve("test") == 42


def test_initialize_marks_initialized(tmp_memory_path):
    mem = MemorySystem(tmp_memory_path)

    async def run():
        await mem.initialize()

    asyncio.run(run())
    assert mem.retrieve("initialized") is True


def test_initialize_is_async(tmp_memory_path):
    """initialize() must be awaitable (the orchestrator awaits it)."""
    mem = MemorySystem(tmp_memory_path)
    assert inspect.iscoroutinefunction(mem.initialize)


def test_store_cycle_is_async(tmp_memory_path):
    """store_cycle() must be awaitable."""
    mem = MemorySystem(tmp_memory_path)
    assert inspect.iscoroutinefunction(mem.store_cycle)


def test_store_error_is_async(tmp_memory_path):
    """store_error() must be awaitable."""
    mem = MemorySystem(tmp_memory_path)
    assert inspect.iscoroutinefunction(mem.store_error)


def test_store_cycle_persists(tmp_memory_path):
    mem = MemorySystem(tmp_memory_path)

    async def run():
        await mem.store_cycle(
            goal="test goal",
            tasks=["task1"],
            design={"arch": "v1"},
            implementation="code",
            test_results={"passed": 3, "total": 3},
            optimized="final",
        )

    asyncio.run(run())

    reloaded = MemorySystem(tmp_memory_path)
    cycle = reloaded.retrieve("cycle_test goal")
    assert cycle is not None
    assert cycle["goal"] == "test goal"
    assert cycle["tasks"] == ["task1"]
    assert cycle["design"] == {"arch": "v1"}


def test_store_error_persists(tmp_memory_path):
    mem = MemorySystem(tmp_memory_path)

    async def run():
        await mem.store_error("broken goal", "RuntimeError: boom")

    asyncio.run(run())

    reloaded = MemorySystem(tmp_memory_path)
    assert reloaded.retrieve("error_broken goal") == "RuntimeError: boom"
    last_error = reloaded.retrieve("last_error")
    assert last_error["goal"] == "broken goal"
    assert last_error["error"] == "RuntimeError: boom"
