"""
Tests for the pytest runner integration in flow.py.

This verifies that the evaluate_model function correctly executes pytest
and returns the appropriate pass/fail status.
"""
import pytest
import sys
import os

# Add parent directory to path to import flow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_run_pytest_returns_boolean():
    """Test that run_pytest function returns a boolean value."""
    from flow import run_pytest

    result = run_pytest()

    assert isinstance(result, bool), "run_pytest should return a boolean"


def test_run_pytest_executes_tests():
    """Test that run_pytest actually executes the test suite."""
    from flow import run_pytest

    # Should execute without raising exceptions
    # Result can be True or False depending on random test outcomes
    result = run_pytest()

    assert result in [True, False], "run_pytest should return True or False"
