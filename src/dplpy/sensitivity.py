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

# Title: sensitivity.py
# Project: OpenDendro dplPy
# Description: Mean sensitivity of a ring-width series -- sens1 and sens2 --
#              ported to match dplR's C implementations (dplR 1.7.9, src/sens.c,
#              written by Mikko Korpela) exactly.
#
#   sens1  -- the standard (Douglass) mean sensitivity, Eq. 1 of Biondi & Qeadan
#             (2008). For each adjacent pair the local relative change is
#             |x_i - x_{i-1}| / (x_i + x_{i-1}); these are summed (a 0/0 term from
#             a 0+0 pair is skipped, exactly as dplR's ISNAN guard does) and
#                 sens1 = 2 * sum(term) / (n - 1).
#   sens2  -- mean sensitivity for a series with a trend, Eq. 2 of the same. The
#             local two-year denominator is replaced by the series mean:
#                 sens2 = sum(|x_{i+1} - x_i|) / (sum(x) - sum(x)/n)
#                       = sum(|diff|) / ((n - 1) * mean(x)).
#
# Both drop NA first and return NaN when fewer than two values remain (dplR
# returns R_NaN in that case). dplR sums with exact/compensated arithmetic; we
# use math.fsum (correctly-rounded summation) so the results agree to machine
# precision. See dev/dplR_gap_analysis_2026-08.md.
#
# NOTE (fidelity): dplR deliberately leaves sens1/sens2 OUT of rwl.stats() -- the
# lines are present but commented in dplR's source, reflecting the Bunn et al.
# (2013) caution that mean sensitivity "should rarely, if ever, be used". dplPy's
# dpl.stats() matches that and does not report them; they live here as their own
# functions, called explicitly when wanted.
#
# example usage:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/ca533.csv")
# >>> dpl.sens1(data)            # -> Series of mean sensitivity, one per series
# >>> dpl.sens1(data["CAM011"])  # -> a single float for one series
# >>> dpl.sens2(data)

import math

import numpy as np
import pandas as pd

from ._validate import _coerce_to_frame


def sens1(data):
    """Mean sensitivity (dplR ``sens1``).

    The standard measure of year-to-year variability in dendrochronology
    (Douglass; Eq. 1 of Biondi and Qeadan 2008), typically computed on detrended
    series. For a single series of ``n`` (non-missing) values,

        sens1 = (2 / (n - 1)) * sum_i |x_i - x_{i-1}| / (x_i + x_{i-1})

    where a ``0/0`` term arising from an ``x_i + x_{i-1} == 0`` pair is skipped
    (matching dplR's C ``ISNAN`` guard) while ``n - 1`` is unchanged.

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series, 1-D array-like, or str
        A ring-width dataset (years x series), a single series, or a path/URL to
        a .csv/.rwl file. NA/NaN values are dropped before the calculation.

    Returns
    -------
    float or pandas.Series
        For a single series, the scalar mean sensitivity. For a DataFrame (or a
        file path, which is read into one), a Series of the statistic indexed by
        series name -- the equivalent of R's ``apply(rwl, 2, sens1)``. A series
        with fewer than two values yields ``NaN``.

    Examples
    --------
    >>> dpl.sens1(data)
    >>> dpl.sens1(data["CAM011"])

    References
    ----------
    .. [1] Biondi, F. and Qeadan, F. (2008) Inequality in paleorecords.
           Ecology 89, 1056-1067.
    .. [2] https:/opendendro.org/dplpy-man/#sens1
    """
    return _apply(data, _sens1_value, "sens1")


def sens2(data):
    """Mean sensitivity for a series with a trend (dplR ``sens2``).

    Eq. 2 of Biondi and Qeadan (2008): the local two-year denominator of
    :func:`sens1` is replaced by the series mean, so a growth trend does not
    inflate the statistic. For a single series of ``n`` (non-missing) values,

        sens2 = sum_i |x_{i+1} - x_i| / (sum(x) - sum(x)/n)
              = sum_i |x_{i+1} - x_i| / ((n - 1) * mean(x))

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series, 1-D array-like, or str
        A ring-width dataset (years x series), a single series, or a path/URL to
        a .csv/.rwl file. NA/NaN values are dropped before the calculation.

    Returns
    -------
    float or pandas.Series
        A scalar for a single series, or a Series indexed by series name for a
        DataFrame / file path. Fewer than two values yields ``NaN``.

    Examples
    --------
    >>> dpl.sens2(data)
    >>> dpl.sens2(data["CAM011"])

    References
    ----------
    .. [1] Biondi, F. and Qeadan, F. (2008) Inequality in paleorecords.
           Ecology 89, 1056-1067.
    .. [2] https:/opendendro.org/dplpy-man/#sens2
    """
    return _apply(data, _sens2_value, "sens2")


# --- per-series kernels (faithful ports of dplR 1.7.9 src/sens.c) -----------

def _as_float_1d(x):
    """Return ``x`` as a 1-D float ndarray with NA/NaN removed (as dplR does)."""
    arr = np.asarray(x, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def _sens1_value(x):
    """sens1 for one series given as 1-D array-like. Port of dplR C ``sens1``."""
    x = _as_float_1d(x)
    n = x.size
    if n < 2:
        return float("nan")
    prev = x[:-1]
    cur = x[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.abs(cur - prev) / (cur + prev)
    # dplR adds a term only when it is not NaN (a 0/0 from a 0+0 pair); a
    # nonzero/0 -> +/-Inf would be kept, so mask on NaN alone, not on all
    # non-finite values.
    s = math.fsum(terms[~np.isnan(terms)])       # compensated sum ~ dplR's exact sum
    return (s + s) / (n - 1)                      # dplR: (sum+sum)/(n-1)


def _sens2_value(x):
    """sens2 for one series given as 1-D array-like. Port of dplR C ``sens2``."""
    x = _as_float_1d(x)
    n = x.size
    if n < 2:
        return float("nan")
    sum1 = math.fsum(np.abs(np.diff(x)))         # sum of absolute first differences
    sum2 = math.fsum(x)                          # sum of the series
    return sum1 / (sum2 - sum2 / n)              # dplR: sum1/(sum2 - sum2/n)


# --- input dispatch ---------------------------------------------------------

def _apply(data, fn, name):
    """Scalar for a single series; a per-series Series for a DataFrame / path."""
    if isinstance(data, pd.DataFrame):
        frame = data
    elif isinstance(data, str):
        frame = _coerce_to_frame(data)
    elif isinstance(data, pd.Series):
        return fn(data.to_numpy())
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim <= 1:
            return fn(arr)                        # a single series -> scalar
        # a bare 2-D array (no column names): one value per column, by position
        return pd.Series([fn(arr[:, j]) for j in range(arr.shape[1])], name=name)
    return pd.Series({col: fn(frame[col].to_numpy()) for col in frame.columns},
                     name=name)
