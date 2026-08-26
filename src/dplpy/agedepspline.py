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

# Title: agedepspline.py
# Project: OpenDendro dplPy
# Description: Age-dependent smoothing spline (Melvin 2004; Melvin et al. 2007),
#              a port of dplR's ads(). The spline stiffness grows with cambial
#              age -- the i-th ring is fit with a (nyrs0 + i - 1)-year spline --
#              so young rings are smoothed gently and the curve stiffens with
#              age. dplR ships Ed Cook's original banded solver as Fortran; the
#              underlying problem is just a symmetric positive-definite banded
#              least-squares system, so here it is solved natively with SciPy
#              (reproducing dplR to ~1e-12).

import numpy as np
from scipy.linalg import solveh_banded

# Cook's spline stencil: the smoothing weight applied to the diagonal and first
# off-diagonal of the second-difference roughness penalty (see adsf.f95 / caps).
_W1 = 1.0 / 3.0
_W2 = 4.0 / 3.0


def _ads_curve(y, stiffness):
    """Fit the age-dependent smoothing spline to ``y``.

    ``stiffness`` is the per-ring spline wavelength (length n). The spline
    minimises a second-difference roughness penalty whose weight varies per
    point; that yields a symmetric positive-definite pentadiagonal system, which
    we solve with a banded Cholesky. Returns the spline curve, or None if the
    system is not positive definite.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return None
    m = n - 2                                   # number of interior equations

    # per-point smoothing weight p from the stiffness (50% frequency response)
    arg = 2.0 * np.pi / np.asarray(stiffness[:m], dtype=float)
    p = (6.0 * (np.cos(arg) - 1.0) ** 2) / (np.cos(arg) + 2.0)

    # Symmetric banded system M x = b (bandwidth 2), in SciPy's lower-diagonal
    # ordered form: row i has diagonal 6 + 4p/3, first sub-diagonal -4 + p/3,
    # second sub-diagonal 1. b is the second difference of y.
    ab = np.zeros((3, m))
    ab[0] = 6.0 + p * _W2                        # main diagonal
    ab[1, :-1] = (-4.0 + p * _W1)[1:]            # first sub-diagonal
    ab[2, :-2] = 1.0                             # second sub-diagonal
    b = y[:m] - 2.0 * y[1:m + 1] + y[2:m + 2]

    try:
        x = solveh_banded(ab, b, lower=True)
    except np.linalg.LinAlgError:
        return None

    # the spline curve is y minus the second difference of the solution
    return y - np.convolve(x, (1.0, -2.0, 1.0))


def ads(y, nyrs0=50, pos_slope=True):
    """Age-dependent smoothing spline of a single series (dplR's ads()).

    Extended Summary
    ----------------
    Fits a smoothing spline whose stiffness increases with cambial age: the i-th
    ring is fit with an (nyrs0 + i - 1)-year spline, so young rings are smoothed
    least and the curve grows stiffer with age (Melvin 2004; Melvin et al. 2007).
    The underlying cubic smoothing spline follows Cook & Peters (1981) with a 50%
    frequency cutoff. Reproduces dplR's ads() to ~1e-12.

    Parameters
    ----------
    y : array-like
        a single ring-width series (no missing values), oldest ring first.
    nyrs0 : int, default 50
        initial spline stiffness (wavelength, in years) at the first ring.
    pos_slope : bool, default True
        if False, once the spline stops decreasing its tail is held flat and the
        spline is refit -- preventing an artefactual upturn at the end of the
        series (dplR uses pos.slope=FALSE inside detrend()).

    Returns
    -------
    numpy.ndarray
        the fitted spline curve, same length as ``y``.

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwl = dpl.readers("../tests/data/csv/ca533.csv")
    >>> curve = dpl.ads(rwl["CAM011"].dropna().to_numpy(), nyrs0=50)

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/ads.html
    .. [2] Melvin, T. M. (2004) Historical Growth Rates and Changing Climatic
       Sensitivity of Boreal Conifers. PhD Thesis, Climatic Research Unit,
       University of East Anglia.
    .. [3] Melvin, T. M., Briffa, K. R., Nicolussi, K. & Grabner, M. (2007)
       Time-varying-response smoothing. Dendrochronologia, 25(2), 65-69.
    .. [4] Cook, E. R. & Peters, K. (1981) The Smoothing Spline: A New Approach
       to Standardizing Forest Interior Tree-Ring Width Series for Dendroclimatic
       Studies. Tree-Ring Bulletin, 41, 45-53.
    """
    y = np.asarray(y, dtype=float)
    nobs = len(y)
    if nobs < 3:
        raise ValueError("there must be at least 3 data points")
    if nobs > 10000:
        raise ValueError("y should not be longer than 1e4.")
    if not isinstance(nyrs0, (int, np.integer)) or nyrs0 <= 1:
        raise ValueError("'nyrs0' must be an integer greater than 1")

    stiffness = np.arange(nobs) + nyrs0             # ring i -> nyrs0 + (i - 1)
    curve = _ads_curve(y, stiffness)
    if curve is None:
        raise ValueError("age-dependent spline: banded system not positive definite")

    if not pos_slope:
        diffs = np.diff(curve, prepend=curve[0])    # dplR's c(0, diff(ySpl))
        cutoff = int(np.max(np.where(diffs <= 0)[0]))
        curve = curve.copy()
        curve[cutoff:] = curve[cutoff]              # hold the tail flat
        refit = _ads_curve(curve, stiffness)        # and refit once
        if refit is not None:
            curve = refit
    return curve
