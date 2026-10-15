"""Shared pytest fixtures for the dplPy test suite.

The data fixtures are *session-scoped* (each file is read once per test run)
and return ``.copy()`` so a test can never leak a mutation into another test.
``DATA`` resolves the test-data directory relative to this file, so the suite
runs from any working directory (not only the repository root).

Usage in a test module::

    def test_something(ca533_rwl, DATA):
        rwi = dpl.detrend(ca533_rwl, plot=False)
        other = dpl.readers(DATA / "rwl" / "co021.rwl")
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # headless: no display in tests/CI

import pytest

import dplpy as dpl

_DATA = Path(__file__).resolve().parent / "data"


def _read_quiet(path, **kwargs):
    """dpl.readers() with its DeprecationWarning/UserWarning chatter silenced
    (the readers tests cover those warnings explicitly)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dpl.readers(str(path), **kwargs)


@pytest.fixture(scope="session")
def DATA():
    """``Path`` to ``tests/data`` (independent of the current working directory)."""
    return _DATA


@pytest.fixture(scope="session")
def _ca533_rwl_cached():
    return _read_quiet(_DATA / "csv" / "ca533.csv")


@pytest.fixture(scope="session")
def _ca533_rwi_cached(_ca533_rwl_cached):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dpl.detrend(_ca533_rwl_cached, fit="Spline", plot=False)


@pytest.fixture(scope="session")
def _co021_rwl_cached():
    return _read_quiet(_DATA / "csv" / "co021.csv")


@pytest.fixture
def ca533_rwl(_ca533_rwl_cached):
    """Campito Mountain bristlecone pine ring widths (ITRDB ca533), 34 series,
    626-1983. A fresh copy per test."""
    return _ca533_rwl_cached.copy()


@pytest.fixture
def ca533_rwi(_ca533_rwi_cached):
    """ca533 detrended with the default smoothing spline (ratio indices). A fresh
    copy per test."""
    return _ca533_rwi_cached.copy()


@pytest.fixture
def co021_rwl(_co021_rwl_cached):
    """Mesa Verde Douglas-fir ring widths (ITRDB co021), 35 series, 1250-1963. A
    fresh copy per test."""
    return _co021_rwl_cached.copy()
