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

# Title: glk.py
# Project: OpenDendro dplPy
# Description: Sign-agreement crossdating statistics between all pairs of series
#              in a ring-width dataset -- Gleichlaeufigkeit (glk) and synchronous
#              growth changes (sgc/ssgc). Ports of dplR 1.7.9 R/glk.R and R/sgc.R
#              (Visser's fast implementation) matched value-for-value.
#
# Shared core: take the sign of each series' year-to-year difference (+1 rise,
# -1 fall, 0 no change). For a pair of series the per-interval "growth-change"
# disagreement is GC = |sign_i - sign_j|, which is 0 when both move the same way,
# 2 when they move oppositely, and 1 when exactly one is flat. Over the N
# intervals the two series share (the "overlap"):
#
#   glk  = 1 - sum(GC) / (2*N)                    (Gleichlaeufigkeit, in [0, 1])
#   sgc  = count(GC == 0) / N                     (synchronous growth changes)
#   ssgc = count(GC == 1) / N                     (semi-synchronous growth changes)
#
# These satisfy glk == sgc + ssgc/2. Pairs whose overlap is below `overlap`
# intervals are returned as NaN. When `prob=True` a two-sided significance is
# added, p = 2*(1 - Phi((g - 0.5) / (1/(2*sqrt(N))))), exactly as dplR computes
# it (so p can exceed 1 when the statistic is below 0.5, as in dplR).
#
# example usage:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/ca533.csv")
# >>> res = dpl.glk(data)                 # dict of DataFrames: glk_mat, overlap, p_mat
# >>> res["glk_mat"]
# >>> dpl.sgc(data, overlap=30, prob=False)

import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr               # standard-normal CDF (== R's pnorm)

from ._validate import _coerce_to_frame


def glk(data, overlap=50, prob=True):
    """Gleichlaeufigkeit (sign-agreement) between all pairs of series.

    For every pair of series this is the proportion of shared intervals in which
    the two move in the same direction, counting a flat (no-change) interval as
    half agreement. A port of dplR's ``glk`` (Visser's implementation).

    Parameters
    ----------
    data : pandas.DataFrame or str
        A ring-width dataset (years x series), or a path/URL to a .csv/.rwl file.
    overlap : int, default 50
        Minimum number of overlapping growth-change intervals a pair must share;
        pairs below this are NaN. Must be a single integer >= 3. A warning is
        issued for values below 50 (matches likely become statistically
        insignificant).
    prob : bool, default True
        If True, also return a matrix of two-sided p-values.

    Returns
    -------
    dict of pandas.DataFrame
        ``glk_mat`` (the statistic, diagonal set to 1), ``overlap`` (the number
        of overlapping intervals per pair), and, when ``prob`` is True,
        ``p_mat`` (two-sided p-values). All are n x n, indexed by series name.

    See Also
    --------
    sgc : the synchronous / semi-synchronous decomposition (glk = sgc + ssgc/2).

    Examples
    --------
    >>> res = dpl.glk(data)
    >>> res["glk_mat"]

    References
    ----------
    .. [1] Schweingruber, F.H. (1988) Tree Rings. Kluwer.
    .. [2] Visser, R.M. (2021) On the similarity of tree-ring patterns:
           Journal of Archaeological Science 125.
    .. [3] https:/opendendro.org/dplpy-man/#glk
    """
    names, signs = _prep(data, overlap, prob)
    n = len(names)
    glk_mat = np.full((n, n), np.nan)
    overlap_mat = np.full((n, n), np.nan)

    for i in range(n):
        gc = np.abs(signs[:, [i]] - signs)           # (L-1, n), NaN where either NaN
        ncol = _overlap_counts(gc, overlap)          # (n,), NaN below threshold
        glk_mat[i, :] = 1.0 - np.nansum(gc, axis=0) / (2.0 * ncol)
        overlap_mat[i, :] = ncol

    np.fill_diagonal(glk_mat, 1.0)                    # dplR forces the diagonal to 1

    out = {"glk_mat": _frame(glk_mat, names),
           "overlap": _frame(overlap_mat, names)}
    if prob:
        out["p_mat"] = _frame(_p_values(glk_mat, overlap_mat), names)
    return out


def sgc(data, overlap=50, prob=True):
    """Synchronous and semi-synchronous growth changes between all pairs.

    A decomposition of :func:`glk`: ``sgc`` is the proportion of shared intervals
    in which two series move in the *same* direction, and ``ssgc`` the proportion
    in which exactly one is flat. A port of dplR's ``sgc``.

    Parameters
    ----------
    data : pandas.DataFrame or str
        A ring-width dataset (years x series), or a path/URL to a .csv/.rwl file.
    overlap : int, default 50
        Minimum overlapping intervals per pair (a single integer >= 3); pairs
        below this are NaN. A warning is issued below 50.
    prob : bool, default True
        If True, also return a matrix of two-sided p-values (computed from sgc).

    Returns
    -------
    dict of pandas.DataFrame
        ``sgc_mat``, ``ssgc_mat``, ``overlap``, and, when ``prob`` is True,
        ``p_mat`` -- all n x n, indexed by series name. (dplPy uses the
        ``*_mat`` keys in both branches; dplR's ``prob=False`` list names them
        ``sgc``/``ssgc`` -- the values are identical.)

    See Also
    --------
    glk : the combined statistic, glk = sgc + ssgc/2.

    Examples
    --------
    >>> res = dpl.sgc(data)
    >>> res["sgc_mat"], res["ssgc_mat"]

    References
    ----------
    .. [1] Visser, R.M. (2021) On the similarity of tree-ring patterns:
           Journal of Archaeological Science 125.
    .. [2] https:/opendendro.org/dplpy-man/#sgc
    """
    names, signs = _prep(data, overlap, prob)
    n = len(names)
    sgc_mat = np.full((n, n), np.nan)
    ssgc_mat = np.full((n, n), np.nan)
    overlap_mat = np.full((n, n), np.nan)

    for i in range(n):
        gc = np.abs(signs[:, [i]] - signs)           # (L-1, n)
        ncol = _overlap_counts(gc, overlap)          # (n,)
        # NaN == k is False, so these boolean sums naturally ignore missing
        # intervals -- no na.rm needed.
        sgc_mat[i, :] = np.sum(gc == 0.0, axis=0) / ncol
        ssgc_mat[i, :] = np.sum(gc == 1.0, axis=0) / ncol
        overlap_mat[i, :] = ncol

    out = {"sgc_mat": _frame(sgc_mat, names),
           "ssgc_mat": _frame(ssgc_mat, names),
           "overlap": _frame(overlap_mat, names)}
    if prob:
        out["p_mat"] = _frame(_p_values(sgc_mat, overlap_mat), names)
    return out


# --- shared internals -------------------------------------------------------

def _prep(data, overlap, prob):
    """Validate arguments and return (series names, sign-of-diff matrix)."""
    frame = _coerce_to_frame(data)
    _check_overlap(overlap)
    if not isinstance(prob, bool):
        raise ValueError("'prob' must be either True (the default) or False")
    names = list(frame.columns)
    X = frame.to_numpy(dtype=float)                  # (L years, n series)
    # sign of the year-to-year change; NaN propagates across gaps, exactly as
    # dplR's sign(apply(x, 2, diff)).
    signs = np.sign(np.diff(X, axis=0))              # (L-1, n)
    return names, signs


def _check_overlap(overlap):
    """dplR's rule: a single integer >= 3; warn (not error) below 50."""
    ok = (np.isscalar(overlap)
          and not isinstance(overlap, bool)
          and float(overlap) == int(overlap)
          and overlap >= 3)
    if not ok:
        raise ValueError("'overlap' should be a single integer >= 3")
    if overlap < 50:
        warnings.warn("The minimum number of overlap is lower than 50. This "
                      "might lead to statistically insignificant matches.")


def _overlap_counts(gc, overlap):
    """Per-column count of non-NaN intervals, NaN'd where below ``overlap``."""
    ncol = np.sum(~np.isnan(gc), axis=0).astype(float)
    ncol[ncol < overlap] = np.nan
    return ncol


def _p_values(stat_mat, overlap_mat):
    """Two-sided significance, exactly as dplR: 2*(1 - Phi((g-0.5)*2*sqrt(N)))."""
    with np.errstate(invalid="ignore", divide="ignore"):
        s = 1.0 / (2.0 * np.sqrt(overlap_mat))
        z = (stat_mat - 0.5) / s
    return 2.0 * (1.0 - ndtr(z))


def _frame(mat, names):
    return pd.DataFrame(mat, index=names, columns=names)
