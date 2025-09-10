"""
dplpy: The Dendrochronology Program Library for Python

A scientific computing package for tree ring width time series analyses.
Provides functionality for reading dendrochronological data, statistical analysis,
detrending, chronology building, and cross-dating.
"""

__version__ = "0.1.7"
__author__ = "Tyson Lee Swetnam"
__email__ = "tswetnam@arizona.edu"

# Import main functionality for easy access
from .io.readers import readers
from .io.writers import write
from .core.stats import stats
from .core.detrend import detrend
from .analysis.chron import chron
from .analysis.xdate import xdate
from .core.plot import plot
from .core.summary import summary
from .core.report import report
from .analysis.autoreg import autoreg

# Define public API
__all__ = [
    "readers",
    "write", 
    "stats",
    "detrend",
    "chron",
    "xdate",
    "plot",
    "summary",
    "report",
    "autoreg",
]

# Version info
def version():
    """Return version information."""
    return __version__

def help():
    """Print help information."""
    from .cli.main import help_system
    help_system()