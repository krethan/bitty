"""Tests for the CLI entry point."""

import asyncio
import io
import os
import pickle
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout

import pytest


def _find_bitty_executable():
    """Locate the bitty console script, or skip if not installed."""
    path = shutil.which("bitty")
    if path is None:
        pytest.skip("bitty console script not on PATH (run `pip install -e python[dev]`)")
    return path


def test_cli_help_runs():
    """`bitty --help` should exit cleanly and print usage."""
    bitty = _find_bitty_executable()
    result = subprocess.run(
        [bitty, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--goal" in result.stdout
    assert "--continuous" in result.stdout


def test_cli_no_args_shows_help():
    """`bitty` with no args prints usage (the main() function calls parser.print_help())."""
    bitty = _find_bitty_executable()
    result = subprocess.run(
        [bitty],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # The current main() falls through to parser.print_help() rather than
    # parser.error(), so it exits 0 after printing usage.
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "usage:" in combined.lower() or "--goal" in combined


def test_cli_goal_runs_cycle(tmp_path, monkeypatch):
    """`bitty --goal X` actually runs the cycle and persists results."""
    bitty = _find_bitty_executable()
    monkeypatch.setenv("HOME", str(tmp_path))
    result = subprocess.run(
        [bitty, "--goal", "CLI integration test goal"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    memory_path = tmp_path / ".bitty" / "memory"
    assert memory_path.exists()


def test_cli_continuous_runs_multiple_goals(tmp_path, monkeypatch):
    """`bitty --goals X Y --continuous` runs both."""
    bitty = _find_bitty_executable()
    monkeypatch.setenv("HOME", str(tmp_path))
    result = subprocess.run(
        [bitty, "--goals", "first", "second", "--continuous"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    memory_path = tmp_path / ".bitty" / "memory"
    assert memory_path.exists()
    with open(memory_path, "rb") as f:
        data = pickle.load(f)
    assert "cycle_first" in data
    assert "cycle_second" in data


def test_cli_main_function_help(monkeypatch):
    """Test the cli()/main() entry point directly without subprocess."""
    from bitty.__init__ import main

    # Run with --help which makes argparse call parser.exit(0).
    monkeypatch.setattr(sys, "argv", ["bitty", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        # main() is a coroutine, so wrap in asyncio.run.
        asyncio.run(main())
    assert exc_info.value.code == 0
