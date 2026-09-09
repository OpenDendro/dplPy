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
# Title: smoothingspline.py
# Description: This contains the spline method which fits a series to
#              a spline curve.

from math import cos
from math import pi, floor
from csaps import csaps
import numpy as np

# Returns the spline parameter, given amplitude of the series and the period
def get_param(amp, period):
    freq = 1/period
    spline_param = 1/(((cos(2 * pi * freq) + 2) * (1 - amp)/(12 * amp * (cos(2 * pi * freq) - 1) ** 2))+ 1)
    return spline_param

def get_period(period, n):
    # Resolve the requested spline wavelength (in years) from `period` and the
    # series length `n`. Four conventions are supported:
    #   * None      -> the default "n-year spline" wavelength floor(0.67 * n),
    #                  matching dplR's detrend.series (nyrs = floor(nY2 * 0.67)).
    #                  (Flooring matters: an unfloored 0.67 * n left dplPy's
    #                  spline ~3e-5 off dplR's, the source of an early crossdating
    #                  gap.)
    #   * period < 0 -> a "percent spline" (ARSTAN convention): the absolute
    #                  value is read as a percentage of the series length, so
    #                  period=-10 gives a 10%-of-length wavelength (n * 10/100).
    #                  ARSTAN signals a percentage-of-length stiffness with a
    #                  negative spline selection: "idt < -9 -- cubic smoothing
    #                  spline (years cutoff = idt/100*n)" (ARSTAN source), i.e.
    #                  the same |value|/100 * n formula used here.
    #   * 0 < period <= 1 -> a fraction of the series length (n * period), e.g.
    #                  period=0.1 gives a 10%-of-length wavelength.
    #   * period > 1 -> a fixed wavelength in years, used as given.
    if period is None:
        return floor(n * 0.67)
    elif period < 0:
        return n * abs(period)/100
    elif period <= 1:
        return n * period
    else:
        return period

# The one csaps smoothing-spline call, shared by spline() and rcs's caps(): fit
# at x with an f-amplitude cutoff at the already-resolved wavelength `nyrs`, and
# evaluate back at x. Callers resolve `nyrs` themselves -- spline() via
# get_period, rcs's caps via integer truncation -- so this only centralizes the
# csaps/get_param wiring, leaving each caller's wavelength convention intact.
def _smooth_csaps(x, y, nyrs, f):
    return csaps(x, y, x, smooth=get_param(f, nyrs))

# Fits a curve to the series given as input and returns the y-values of the curve.
# `f` is the spline's frequency-response amplitude at the `period` wavelength
# (dplR's `f`, default 0.5: a 50% amplitude cutoff at that wavelength).
def spline(x, y, period=None, f=0.5):
    return _smooth_csaps(x, y, get_period(period, len(x)), f)


def variance_stabilize_spline(x, period=None, f=0.5, clip_negative=False, ok=None):
    """Ad-hoc spline variance stabilization -- the approach offered as a
    variance-stabilization option in ARSTAN and used in COFECHA's
    variance-stabilization step.

    Centres the series, fits a smoothing spline to the ABSOLUTE departures to
    capture their time-varying amplitude, divides the departures by that envelope
    (restoring sign), and rescales to the original mean and standard deviation.
    This flattens time-varying variance whatever its cause; it is strictly ad hoc
    and can remove real low-frequency variance (Osborn et al. 1997), so it is
    opt-in.

    Parameters
    ----------
    x : array-like
        the series to stabilize (no interior NaNs).
    period : int or float or None
        spline stiffness (wavelength) passed to :func:`spline`, resolved by
        :func:`get_period`. An int > 1 is a fixed wavelength in years (COFECHA
        uses 32); a float in (0, 1) is a fraction of the series length; ``None``
        uses the default 67% spline (``floor(0.67 * n)`` years), the same
        wavelength convention detrending uses. Note that ``period`` sets the
        *wavelength*; ``f`` (below) independently sets the frequency response at
        that wavelength -- the two are separate knobs.
    f : float, default 0.5
        spline frequency-response amplitude at ``period`` (a 50% amplitude cutoff
        at that wavelength).
    clip_negative : bool, default False
        set negatives to 0 after rescaling -- appropriate when the result is a
        chronology (chron.stabilized), not when it is a pre-AR filtered index
        (COFECHA), so it defaults off.
    ok : array-like of bool or None
        mask of positions to use for the mean/SD (e.g. sample depth > 0); the
        spline is still fit to all points. ``None`` uses every point.

    Returns
    -------
    numpy.ndarray of the variance-stabilized series (same length as ``x``).
    """
    tr = np.asarray(x, dtype=float).copy()
    n = len(tr)
    if ok is None:
        ok = np.ones(n, dtype=bool)
    else:
        ok = np.asarray(ok, dtype=bool)
    xbar1 = np.mean(tr[ok])
    sig1 = np.std(tr[ok], ddof=1)
    dep = tr - xbar1
    sign = np.where(dep < 0, -1.0, 1.0)
    absdep = np.abs(dep)
    xg = np.arange(1, n + 1)
    cv = np.asarray(spline(xg, absdep, period=period, f=f), dtype=float)
    cv = np.where(cv <= 0, np.nan, cv)              # guard: |departures| envelope > 0
    sb = (absdep / cv) * sign
    with np.errstate(invalid="ignore"):
        xbar = np.nanmean(sb[ok])
        sig = np.nanstd(sb[ok], ddof=1)
    if sig == 0 or np.isnan(sig):
        return tr
    sb = ((sb - xbar) / sig) * sig1 + xbar1
    if clip_negative:
        sb[sb < 0] = 0.0
    return sb