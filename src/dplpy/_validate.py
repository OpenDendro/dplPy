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

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Title: _validate.py
# Project: OpenDendro dplPy
# Description: Small shared input-validation helpers, so the many public
#              functions that require a ring-width DataFrame -- or that accept a
#              DataFrame or a path to a file -- do it one consistent way with one
#              consistent error message, instead of each re-implementing the
#              check with slightly different wording.

import pandas as pd


def _require_dataframe(x):
    """Return ``x`` unchanged if it is a pandas DataFrame, else raise TypeError.

    The single home for the ``isinstance(x, pd.DataFrame)`` guard that opens most
    of dplPy's public functions.
    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError("Expected dataframe input, got " + str(type(x)) + " instead.")
    return x


def _coerce_to_frame(inp):
    """Return ``inp`` as a DataFrame.

    A DataFrame is returned as-is; a string is treated as a path (or URL) and
    read with ``readers()``; anything else raises TypeError. Used by the summary
    functions (summary, stats, report, plot) that accept either a frame or a file
    path.
    """
    if isinstance(inp, pd.DataFrame):
        return inp
    if isinstance(inp, str):
        from .readers import readers          # local import avoids an import cycle
        return readers(inp)
    raise TypeError("Input must be a pandas DataFrame or a path (str) to a "
                    "CSV/RWL file, not " + str(type(inp)) + ".")
