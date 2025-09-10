"""Core analysis functionality for dplpy."""

from .stats import stats
from .detrend import detrend
from .plot import plot
from .summary import summary
from .report import report

__all__ = ["stats", "detrend", "plot", "summary", "report"]