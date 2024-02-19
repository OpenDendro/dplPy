# -*- coding: utf-8 -*-

__author__ = "Tyson Lee Swetnam"
__email__ = "tswetnam@arizona.edu"
__version__ = "0.1.3"

__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2021  OpenDendro

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
__license__ = "GNU GPLv3"


import os
import sys

lpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(lpath)

del os, sys


_hard_dependencies = ("pandas", "numpy", "scipy", "csaps", "matplotlib", "statsmodels")
_missing_dependencies = []

for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        _missing_dependencies.append(f"{_dependency}: {_e}")

if _missing_dependencies:  # pragma: no cover
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _hard_dependencies, _dependency, _missing_dependencies


from readers import readers
from summary import summary
from stats import stats
from report import report
from plot import plot
from detrend import detrend
from autoreg import ar_func, autoreg
from chron import chron
from chron_stabilized import chron_stabilized
from xdate import xdate, xdate_plot
from series_corr import series_corr
from writers import writers

__all__ = [
    readers,
    summary,
    stats,
    report,
    plot,
    detrend,
    ar_func,
    autoreg,
    chron,
    chron_stabilized,
    xdate,
    xdate_plot,
    series_corr,
    writers
]