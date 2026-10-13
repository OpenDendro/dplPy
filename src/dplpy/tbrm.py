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


def _nanmedian_rows(mat):
    """Exact per-row median of a 2-D array, ignoring NaN. Faster than
    ``np.nanmedian(mat, axis=1)``, which routes a NaN-containing array through
    numpy's masked-array sort path. Here we sort once (numpy orders NaN last) and
    read the middle of each row's non-NaN count. A row that is entirely NaN
    yields NaN.
    """
    mat = np.asarray(mat, dtype=float)
    n = mat.shape[0]
    out = np.full(n, np.nan)
    if mat.shape[1] == 0:
        return out
    s = np.sort(mat, axis=1)                         # NaN sorts to the end
    cnt = np.count_nonzero(~np.isnan(mat), axis=1)   # real values per row
    good = cnt > 0
    rows = np.nonzero(good)[0]
    cg = cnt[good]
    lo = (cg - 1) // 2
    hi = cg // 2
    out[good] = 0.5 * (s[rows, lo] + s[rows, hi])    # middle value(s) of the real part
    return out


def tbrm_rows(mat, c=9):
    """Vectorized, NaN-aware Tukey biweight robust mean applied to every ROW of a
    2-D array ``mat`` (nrows x k); returns a length-nrows vector.

    This is the single vectorized form of ``tbrm`` used by the hot per-year
    aggregation paths (the crossdating master, chron_ars, the ARSTAN infill).
    It reproduces ``tbrm(row_without_nan, c)`` row-for-row (see test_tbrm) while
    avoiding a Python loop. A row that is entirely NaN yields NaN.
    """
    mat = np.asarray(mat, dtype=float)
    e = 1e-6
    med = _nanmedian_rows(mat)
    s = _nanmedian_rows(np.abs(mat - med[:, None]))
    denom = c * s + e
    u = (mat - med[:, None]) / denom[:, None]
    w = np.where(np.abs(u) <= 1, (1 - u ** 2) ** 2, 0.0)
    w = np.where(np.isnan(mat), 0.0, w)              # NaN entries carry no weight
    num = np.nansum(w * mat, axis=1)
    den = np.sum(w, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)
