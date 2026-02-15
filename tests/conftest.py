"""Pytest configuration and fixtures."""

import pytest
import numpy as np


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (run with '-m slow' or '--runslow')"
    )


def pytest_addoption(parser):
    """Add custom command line option for random seed."""
    parser.addoption(
        "--seed",
        action="store",
        default=None,
        help="Random seed for reproducible tests (default: random)"
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests by default unless --runslow or -m slow is used."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    # Check if user explicitly requested slow tests with -m
    markexpr = config.getoption("-m", "")
    if "slow" in markexpr:
        return

    skip_slow = pytest.mark.skip(reason="use --runslow or -m slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def test_seed(request):
    """Fixture to get seed from CLI or generate random one."""
    seed = request.config.getoption("--seed")
    if seed is None:
        seed = np.random.randint(0, 1000000)
        print(f"\nUsing random seed: {seed} (use --seed={seed} to reproduce)")
    else:
        seed = int(seed)
        print(f"\nUsing specified seed: {seed}")
    return seed
