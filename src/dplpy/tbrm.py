__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2024  OpenDendro

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

# Date: 11/1/2022
# Author: Ifeoluwa Ale
# Title: tbrm.py
# Description: This file contains helper functions which find tukey's biweight robust mean of
#              an array like object.

import warnings

import numpy as np

def tbrm(data, c=9):
    data = np.asarray(data, dtype=float)
    e = 1 * pow(10, -6)  # regularization epsilon; matches dplR's tbrm C code (C*MAD + 1e-6)
    m = np.median(data)

    s = np.median(np.abs(data - m))

    u = (data - m) / ((c * s) + e)

    w = np.where(np.abs(u) <= 1, (1 - u ** 2) ** 2, 0.0)

    return np.sum(w*data)/np.sum(w)


def tbrm_rows(mat, c=9):
    """Vectorized, NaN-aware Tukey biweight robust mean applied to every ROW of a
    2-D array ``mat`` (nrows x k); returns a length-nrows vector.

    This is the single vectorized form of ``tbrm`` used by the hot per-year
    aggregation paths (the crossdating master, chron_ars, the ARSTAN infill).
    It reproduces ``tbrm(row_without_nan, c)`` row-for-row to machine precision
    (see test_tbrm) while avoiding a Python loop. A row that is entirely NaN
    yields NaN.
    """
    mat = np.asarray(mat, dtype=float)
    e = 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
        med = np.nanmedian(mat, axis=1)
        s = np.nanmedian(np.abs(mat - med[:, None]), axis=1)
    denom = c * s + e
    u = (mat - med[:, None]) / denom[:, None]
    w = np.where(np.abs(u) <= 1, (1 - u ** 2) ** 2, 0.0)
    w = np.where(np.isnan(mat), 0.0, w)              # NaN entries carry no weight
    num = np.nansum(w * mat, axis=1)
    den = np.sum(w, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)
