# -*- coding: utf-8 -*-

__author__ = "Kevin Anchukaitis"
__email__ = "kanchukaitis@arizona.edu"
__version__ = "0.3.0"

__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2026  OpenDendro

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


from .readers import readers
from .readers import metadata
from .readers_url import readers_url
from .summary import summary
from .stats import stats
from .samp_stats import samp_stats
from .fill_internal import fill_internal
from .report import report
from .plot import plot
from .detrend import detrend
from .autoreg import ar_func, autoreg
from .chron import chron
from .chron_stabilized import chron_stabilized
from .chron_ars import chron_ars
from .xdate import xdate, xdate_plot
from .series_corr import series_corr
from .interseries_corr import interseries_corr
from .read_ids import read_ids
from .rwi_stats import rwi_stats, rwi_stats_running
from .sensitivity import sens1, sens2
from .common_interval import common_interval
from .agedepspline import ads
from .pith import po_to_wc, wc_to_po
from .powt import powt
from .rcs import rcs
from .simplesignalfree import ssf
from .sss import sss
from .writers import writers
from .cli import help, readme

__all__ = [
    "readers",
    "metadata",
    "readers_url",
    "summary",
    "stats",
    "samp_stats",
    "report",
    "plot",
    "detrend",
    "fill_internal",
    "ar_func",
    "autoreg",
    "chron",
    "chron_stabilized",
    "chron_ars",
    "xdate",
    "xdate_plot",
    "series_corr",
    "interseries_corr",
    "read_ids",
    "rwi_stats",
    "rwi_stats_running",
    "sens1",
    "sens2",
    "common_interval",
    "ads",
    "po_to_wc",
    "wc_to_po",
    "powt",
    "rcs",
    "ssf",
    "sss",
    "writers",
    "help",
    "readme",
]
