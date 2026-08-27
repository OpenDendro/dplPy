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

# Title: rcs.py
# Project: OpenDendro dplPy
# Description: Regional Curve Standardisation (single curve), a port of dplR's
#              rcs(). Series are aligned on a common cambial-age axis using their
#              pith offsets; the average growth at each cambial age forms the
#              "regional curve"; that curve is smoothed and each ring is divided
#              by (or has subtracted) the curve value at its cambial age. Unlike
#              curve-fitting standardisation this preserves low-frequency
#              (e.g. multi-centennial) signal. Reproduces dplR's rcs() to machine
#              precision.

import numpy as np
import pandas as pd
from ._validate import _require_dataframe
import matplotlib.pyplot as plt
from .agedepspline import ads, _ads_curve
from .smoothingspline import _smooth_csaps
from .xdate import _row_biweight


# --- CRUST-style regional-curve refinements (Melvin & Briffa 2014) ----------
# Validated against CRUST's compiled spline3 / splinec kernels to ~1e-11.

def _spline3(rws, cnt, ss=10, rise=False):
    """CRUST's age-dependent RCS smoother (stand.f90 spline3).

    Fits an age-dependent spline (per-point stiffness ``ss + cambial age``, i.e.
    our ads with nyrs0 = ss+1) only over the well-replicated part of the curve
    -- up to the last cambial age with sample depth >= 4 -- then holds the curve
    flat beyond. Unless ``rise`` (kept only for very old trees / basal area), the
    tail is flattened from the curve's minimum in its final third, preventing an
    artefactual old-age upturn.
    """
    n = len(rws)
    j = 0
    for jj in range(n, 0, -1):
        if cnt[jj - 1] >= 4:
            j = jj
            break
    if j < 4:                                   # too little replication to smooth
        j = n
    stiff = ss + np.arange(1, j + 1)            # ss+1, ss+2, ... = age-dependent
    rwp = np.full(n, np.nan)
    fit = _ads_curve(rws[:j], stiff)
    rwp[:j] = rws[:j] if fit is None else fit
    if rise:
        rwp[j:] = rwp[j - 1]                     # flat extend beyond the fit
    else:
        start = 2 * j // 3                       # final third (1-based start)
        q = int(np.argmin(rwp[start - 1:j])) + start   # 1-based index of the min
        rwp[q:] = rwp[q - 1]                     # flatten after it
    return rwp


def _crust_regional_curve(ca_m, ca_n, nrow, ss=10, rise=False, floor=0.02):
    """Build the CRUST-style regional curve: infill gaps, smooth with spline3,
    apply the minimum-value floor, and flat-extend to the full cambial-age grid."""
    last = int(np.max(np.where(ca_n > 0)[0]))    # last cambial age with data
    span = ca_m[:last + 1].copy()
    cnt = ca_n[:last + 1]
    # infill gaps: leading with the first value, interior by carry-forward
    valid = np.where(~np.isnan(span))[0]
    span[:valid[0]] = span[valid[0]]
    for m in range(valid[0] + 1, len(span)):
        if np.isnan(span[m]):
            span[m] = span[m - 1]
    rc_span = np.maximum(_spline3(span, cnt, ss=ss, rise=rise), floor)
    rc = np.full(nrow, np.nan)
    rc[:last + 1] = rc_span
    rc[last + 1:] = rc_span[-1]                   # flat extension
    return rc


def _caps(y, nyrs, f):
    """dplR's caps(): a cubic smoothing spline with an ``f`` (default 0.5)
    amplitude cutoff at wavelength ``nyrs``. dplR truncates nyrs to an integer
    (caps.R: as.integer(nyrs)), so we do too."""
    x = np.arange(1, len(y) + 1)
    return np.asarray(_smooth_csaps(x, y, int(nyrs), f))


def rcs(rwl: pd.DataFrame, po=None, nyrs=None, f=0.5, biweight=True,
        ratios=True, rc_out=False, make_plot=True, method="caps",
        min_n=None, pos_slope=True, preset=None):
    """Regional Curve Standardisation of a set of ring-width series.

    Extended Summary
    ----------------
    Detrends by a single "regional curve": every series is placed on a common
    cambial-age axis according to its pith offset, the average ring width at
    each cambial age is taken (the regional curve), that curve is smoothed, and
    each ring is divided by (ratios) or has subtracted (difference) the curve
    value at its cambial age. This preserves low-frequency growth signal that
    per-series detrending would remove. A port of dplR's rcs().

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series, years as the index and series as columns.
    po : pandas.DataFrame or None, default None
        pith offsets, with columns ``series`` and ``pith_offset`` (the cambial
        age of each series' first measured ring). None assumes a pith offset of
        1 for every series.
    nyrs : int or None, default None
        smoothing stiffness for the regional curve. None uses
        floor(0.1 * curve length) for method='caps' or 50 for method='ads'.
    f : float, default 0.5
        frequency-response amplitude for method='caps'.
    biweight : bool, default True
        build the regional curve with Tukey's biweight robust mean (C=9) rather
        than the arithmetic mean.
    ratios : bool, default True
        detrend by division (ratios); if False, by subtraction (difference).
    rc_out : bool, default False
        if True, return a dict with both the detrended ``rwi`` and the regional
        curve ``rc``; otherwise return just ``rwi``.
    make_plot : bool, default True
        draw the regional-curve figure (series and curve vs cambial age).
    method : {"caps", "ads"}, default "caps"
        smooth the regional curve with a fixed-stiffness spline (caps) or an
        age-dependent spline (ads).
    min_n : int or None, default None
        truncate the regional-curve tail where the by-age sample depth falls
        below this.
    pos_slope : bool, default True
        passed to ads when method='ads'.
    preset : {None, "crust"}, default None
        None reproduces dplR's rcs() exactly (the parameters above apply).
        "crust" builds the regional curve the CRUST way (Melvin & Briffa 2014):
        an age-dependent spline smoothed only where sample depth >= 4 with the
        "no rise in the final third" tail rule, gaps in the curve infilled, a
        0.02 mm minimum-value floor, and flat extension of the tail. Robust for
        the heterogeneous / relict data RCS is typically applied to. When set,
        ``nyrs`` (if given) overrides the age-spline offset (default 10) and
        ``method``/``f``/``pos_slope`` are ignored.

    Returns
    -------
    pandas.DataFrame or dict
        the detrended ``rwi`` (calendar-year indexed), or ``{"rwi", "rc"}`` when
        ``rc_out`` is True.

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/rcs.html
    .. [2] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 1,
       Signal-Free RCS. Dendrochronologia, 32, 7-20.
    .. [3] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further
       RCS options and recommendations. Dendrochronologia, 32, 343-356.
    """
    _require_dataframe(rwl)
    if method not in ("caps", "ads"):
        raise ValueError("method must be 'caps' or 'ads', got '" + str(method) + "'.")
    if preset not in (None, "crust"):
        raise ValueError("preset must be None or 'crust', got '" + str(preset) + "'.")

    cols = list(rwl.columns)
    ncol = len(cols)
    M = rwl.to_numpy(dtype=float)                    # (nyears, nseries)

    # pith offset: default 1 for every series
    if po is None:
        po_map = {c: 1 for c in cols}
    else:
        s = po["series"] if "series" in po.columns else po.iloc[:, 0]
        pv = po["pith_offset"] if "pith_offset" in po.columns else po.iloc[:, 1]
        po_map = dict(zip(np.asarray(s), np.asarray(pv)))

    counts = np.sum(~np.isnan(M), axis=0)            # non-NA rings per series
    max_len = int(counts.max())
    max_po = int(max(int(po_map[c]) for c in cols))

    # ring width by cambial age: each series' (gap-compacted) values placed
    # starting at row = its pith offset.
    rwca = np.full((max_len + max_po, ncol), np.nan)
    for i, c in enumerate(cols):
        vals = M[:, i][~np.isnan(M[:, i])]
        p = int(po_map[c])
        rwca[p - 1:p - 1 + len(vals), i] = vals

    # regional curve = average by cambial age
    if biweight:
        ca_m = _row_biweight(rwca)
    else:
        with np.errstate(invalid="ignore"):
            ca_m = np.nanmean(rwca, axis=1)
    ca_n = np.sum(~np.isnan(rwca), axis=1)

    if min_n is not None:
        last_valid = int(np.max(np.where(ca_n >= min_n)[0]))
        ca_m[np.arange(len(ca_m)) > last_valid] = np.nan

    if preset == "crust":
        # CRUST regional curve: infill, age-dependent spline smoothed only where
        # depth >= 4 (no-rise-final-third tail), 0.02 floor, flat extension.
        ss = 10 if nyrs is None else int(nyrs)
        rise = rwca.shape[0] > 1500              # "except trees > 1500 years old"
        rc = _crust_regional_curve(ca_m, ca_n, rwca.shape[0], ss=ss, rise=rise)
    else:
        # dplR: smooth the regional curve over its non-NA extent
        valid = ~np.isnan(ca_m)
        ym = ca_m[valid]
        if method == "caps":
            nyrs2 = int(np.floor(len(ym) * 0.1)) if nyrs is None else int(nyrs)
            smoothed = _caps(ym, nyrs2, f)
        else:
            nyrs2 = 50 if nyrs is None else int(nyrs)
            smoothed = ads(ym, nyrs0=nyrs2, pos_slope=pos_slope)
        rc = np.full(rwca.shape[0], np.nan)
        rc[valid] = smoothed

    # detrend in cambial-age space, then map back to calendar years
    rwica = rwca / rc[:, None] if ratios else rwca - rc[:, None]
    rwi = rwl.copy().astype(float)
    for i, c in enumerate(cols):
        mask = ~np.isnan(M[:, i])
        if not mask.any():
            continue
        first = int(np.argmax(mask))
        last = len(mask) - 1 - int(np.argmax(mask[::-1]))
        p = int(po_map[c])
        rwi.iloc[first:last + 1, i] = rwica[p - 1:p - 1 + int(counts[i]), i]

    if make_plot:
        _plot_rcs(rwca, ca_m, rc, ncol)

    if rc_out:
        return {"rwi": rwi, "rc": rc}
    return rwi


def _plot_rcs(rwca, ca_m, rc, ncol):
    """Regional-curve figure: each series and the regional curve vs cambial age
    (mirrors dplR's rcs plot), base-R style."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("white")
    x = np.arange(1, rwca.shape[0] + 1)
    alpha = min(0.15, 5.0 / ncol)
    for i in range(ncol):
        ax.plot(x, rwca[:, i], color="grey", lw=1, alpha=alpha, zorder=1)
    ax.plot(x, ca_m, color="0.3", lw=1, zorder=2, label="Mean")
    ax.plot(x, rc, color="steelblue", lw=2.5, zorder=3, label="Regional Curve")
    ax.plot([], [], color="grey", lw=1, label="Series")
    ax.set_xlabel("Cambial Age (Years)", fontsize=12)
    ax.set_ylabel("mm", fontsize=12)
    ax.legend(frameon=False, loc="upper right")
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.tight_layout()
    plt.show()
    return ax
