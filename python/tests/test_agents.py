"""Tests for the Bitty orchestrator and its agents."""

import asyncio
import os
import tempfile

import pytest

from bitty import Bitty
from bitty.agent.architect import ArchitectAgent
from bitty.agent.developer import DeveloperAgent
from bitty.agent.optimizer import OptimizerAgent
from bitty.agent.planner import PlannerAgent, Task
from bitty.agent.tester import TesterAgent, TestResult
from bitty.config import Config
from bitty.memory import MemorySystem


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Redirect HOME so memory persistence doesn't pollute the real filesystem."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _orchestrator(tmp_home):
    """Create a Bitty instance pointing at the test memory location."""
    return Bitty()


def test_bitty_creates_all_agents(tmp_home):
    bitty = _orchestrator(tmp_home)
    assert isinstance(bitty.planner, PlannerAgent)
    assert isinstance(bitty.architect, ArchitectAgent)
    assert isinstance(bitty.developer, DeveloperAgent)
    assert isinstance(bitty.tester, TesterAgent)
    assert isinstance(bitty.optimizer, OptimizerAgent)


def test_bitty_uses_default_config_when_none_provided(tmp_home):
    bitty = _orchestrator(tmp_home)
    assert bitty.config.log_level == "INFO"


def test_initialize_initializes_memory(tmp_home):
    bitty = _orchestrator(tmp_home)

    async def run():
        await bitty.initialize()

    asyncio.run(run())
    assert bitty.memory.retrieve("initialized") is True


def test_run_development_cycle_completes(tmp_home):
    bitty = _orchestrator(tmp_home)

    async def run():
        await bitty.initialize()
        result = await bitty.run_development_cycle("Implement fast tokenizer")
        return result

    result = asyncio.run(run())
    assert result is not None
    # Cycle was persisted.
    cycle = bitty.memory.retrieve("cycle_Implement fast tokenizer")
    assert cycle is not None
    assert cycle["goal"] == "Implement fast tokenizer"


def test_run_development_cycle_calls_each_agent(tmp_home):
    """Verify all five agents ran by checking each persisted something."""
    bitty = _orchestrator(tmp_home)

    async def run():
        await bitty.initialize()
        await bitty.run_development_cycle("Build X")

    asyncio.run(run())
    assert bitty.memory.retrieve("plan_Build X") is not None
    assert bitty.memory.retrieve("design") is not None
    assert bitty.memory.retrieve("implementation") is not None
    assert bitty.memory.retrieve("test_results") is not None
    assert bitty.memory.retrieve("optimized") is not None


def test_run_continuous_handles_multiple_goals(tmp_home):
    bitty = _orchestrator(tmp_home)

    async def run():
        await bitty.initialize()
        await bitty.run_continuous(["Goal A", "Goal B", "Goal C"])

    asyncio.run(run())
    assert bitty.memory.retrieve("cycle_Goal A") is not None
    assert bitty.memory.retrieve("cycle_Goal B") is not None
    assert bitty.memory.retrieve("cycle_Goal C") is not None


def test_run_continuous_records_errors(tmp_home):
    bitty = _orchestrator(tmp_home)

    async def run():
        await bitty.initialize()
        # Inject a failing planner to simulate an error mid-cycle.
        bitty.planner.plan = lambda goal: (_ for _ in ()).throw(
            RuntimeError("simulated failure")
        )
        await bitty.run_continuous(["bad goal"])

    asyncio.run(run())
    assert bitty.memory.retrieve("error_bad goal") == "simulated failure"


def test_run_development_cycle_returns_optimized_output(tmp_home):
    from bitty.agent.developer import Implementation

    bitty = _orchestrator(tmp_home)

    async def run():
        await bitty.initialize()
        return await bitty.run_development_cycle("Some goal")

    result = asyncio.run(run())
    # The current scaffold returns whatever DeveloperAgent produced.
    assert result is not None
    assert isinstance(result, Implementation)


def test_stop_halts_continuous_loop(tmp_home):
    """stop() called mid-loop halts before processing remaining goals."""
    bitty = _orchestrator(tmp_home)

    async def run():
        await bitty.initialize()

        # Monkey-patch the planner to call stop() after the first cycle so we
        # can verify the loop exits between goals.
        original_plan = bitty.planner.plan
        stop_after_first = {"done": False}

        async def stop_after_first_goal(goal):
            result = await original_plan(goal)
            if not stop_after_first["done"]:
                stop_after_first["done"] = True
                bitty.stop()
            return result

        bitty.planner.plan = stop_after_first_goal
        await bitty.run_continuous(["a", "b", "c"])

    asyncio.run(run())
    # The second and third goals should never have been processed.
    assert bitty.memory.retrieve("cycle_a") is not None
    assert bitty.memory.retrieve("cycle_b") is None
    assert bitty.memory.retrieve("cycle_c") is None


def test_planner_returns_ordered_tasks(tmp_home):
    bitty = _orchestrator(tmp_home)

    async def run():
        await bitty.initialize()
        tasks = await bitty.planner.plan("Build a CLI")
        return tasks

    tasks = asyncio.run(run())
    assert len(tasks) >= 1
    assert all(isinstance(t, Task) for t in tasks)


def test_planner_prioritize_sorts_by_priority():
    config = Config.default()
    memory = MemorySystem("/tmp/__noexist__.pkl")
    import logging

    logger = logging.getLogger("test")
    agent = PlannerAgent(config, memory, logger)
    t1 = Task("1", "low", "", 5, [], "small", [])
    t2 = Task("2", "high", "", 1, [], "small", [])
    t3 = Task("3", "mid", "", 3, [], "small", [])

    async def run():
        return await agent.prioritize([t1, t2, t3])

    ordered = asyncio.run(run())
    assert [t.priority for t in ordered] == [1, 3, 5]


def test_tester_returns_test_result():
    config = Config.default()
    memory = MemorySystem("/tmp/__noexist__.pkl")
    import logging

    logger = logging.getLogger("test")
    agent = TesterAgent(config, memory, logger)

    async def run():
        return await agent.test("some implementation")

    result = asyncio.run(run())
    assert isinstance(result, TestResult)
    assert result.passed == result.total
