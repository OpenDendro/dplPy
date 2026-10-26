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

# Title: disturbance.py
# Project: OpenDendro dplPy
# Description: Canopy-disturbance release detection and detrending, a Python port
#              of Dan Druckenbrod's MATLAB (v267_chron / v267_rga), the method of
#              Druckenbrod, Cook, Pederson & Martin-Benito (2024, Dendrochronologia
#              85, 126195). Releases are detected by radial-growth averaging
#              (Lorimer & Frelich: running means, percent-growth-change, a
#              threshold, and the local maximum of each above-threshold segment).
#              A later addition, disturbance_detrend(), will power-transform each
#              series, remove each release with an age-dependent spline, age-detrend
#              the disturbance-free series, and return a climate index.
#
#              Parity: the release detector is validated against the MATLAB golden
#              master (v267_rga, sig=0) on Dan's mo024 post-oak test set, 94
#              releases across 52 of 61 series, release years identical. The MATLAB
#              source and a validation script live in dev/druckenbrod/. Where a
#              comment says "parity:" it reproduces the MATLAB exactly, not
#              necessarily what is statistically ideal.

import numpy as np
import pandas as pd

from ._validate import _warn


def _rga_one(raw, yrs, threshold, window_length):
    """Radial-growth averaging on ONE gap-free raw ring-width series.

    Returns ``(releases, running_mean, pct_growth_change)`` where ``releases`` is a
    list of ``(year, pgc)`` tuples, one per above-threshold segment, placed at the
    year of maximum percent growth change in that segment. ``running_mean`` and
    ``pct_growth_change`` are full-length arrays (NaN where undefined) for plotting.

    Mirrors v267_rga's RGA block exactly. ``raw`` is float ring widths with no NaN,
    ``yrs`` the aligned calendar years, both length n. Returns empty if n < 2*mav.
    """
    mav = int(window_length)
    n = len(raw)
    M = np.full(n, np.nan)
    pgc = np.full(n, np.nan)
    if n < 2 * mav:
        return [], M, pgc

    raw = np.asarray(raw, dtype=float)

    # Trailing running mean of mav years, placed at the last year of the window;
    # defined for index t in [mav-1, n-1]. (v267_rga series(:,3).)
    for t in range(mav - 1, n):
        M[t] = raw[t - mav + 1:t + 1].mean()

    # Percent growth change: preceding mav-year mean (ending at g) vs subsequent
    # mav-year mean (ending at g+mav), for g in [mav-1, n-mav-1]. (series(:,4).)
    col5 = np.zeros(n)                      # PGC where >= threshold, else 0 (series(:,5))
    for g in range(mav - 1, n - mav):
        pgc[g] = (M[g + mav] - M[g]) / M[g] * 100.0
        if pgc[g] >= threshold:
            col5[g] = pgc[g]

    # Local maximum of each contiguous above-threshold segment is the release.
    releases = []
    h = mav - 1
    while h < n:
        prev = col5[h - 1] if h - 1 >= 0 else 0.0
        if col5[h] >= threshold and prev < threshold:
            i = h
            while i < n and col5[i] >= threshold:
                i += 1
            seg_max = col5[h:i].max()
            # find(col5 == max): first global index equal to the segment max,
            # matching v267_rga (seg maxima are distinct floats in practice).
            idx = int(np.where(col5 == seg_max)[0][0])
            releases.append((int(yrs[idx]), float(pgc[idx])))
            h = i
        else:
            h += 1
    return releases, M, pgc


def releases(data, threshold=50, window_length=15, significance=False,
             method="rga", return_info=False):
    """Detect canopy-disturbance release events by radial-growth averaging.

    Finds releases in each series with the Lorimer & Frelich method as implemented
    in Druckenbrod et al. (2024): a trailing ``window_length``-year running mean of
    ring width, the percent growth change between the preceding and subsequent
    windows, and -- for each run of years whose percent growth change meets
    ``threshold`` -- the single year of maximum change, taken as the release.

    Detection runs on the RAW ring widths. The power transform in the Druckenbrod
    method is applied later, only for detrending (see the forthcoming
    ``disturbance_detrend``), so do not pre-transform the series before calling this.

    Parameters
    ----------
    data : pandas DataFrame or Series, or str
        ring widths from ``dpl.readers`` (years on the index, one column per series),
        a single series taken from one, or a path to an rwl/csv file (read with
        ``dpl.readers``). Series shorter than ``2 * window_length`` are skipped with
        a warning.
    threshold : int or float, default 50
        percent-growth-change threshold for a release, in percent. Druckenbrod et
        al. recommend 50 over the more permissive 25.
    window_length : int, default 15
        length in years of the running-mean window (Dan's ``ma``). The recommended
        value is 15; the method needs at least ``2 * window_length`` rings.
    significance : bool, default False
        whether to test each release for significance (Ed Cook's one-tailed
        two-sample t-test). Not implemented in this release and deferred pending
        confirmation of the test; ``True`` raises ``NotImplementedError``.
    method : {"rga"}, default "rga"
        the detection method. Only radial-growth averaging is available; the
        argument reserves room for other detectors.
    return_info : bool, default False
        if True, also return a dict keyed by series name, each value a DataFrame
        indexed by year with the ``running_mean`` and ``pct_growth_change`` curves,
        for plotting or inspection.

    Returns
    -------
    pandas.DataFrame
        one row per release with columns ``series``, ``year`` (year of maximum
        percent growth change) and ``pct_growth_change``, sorted by series then
        year. If ``return_info`` is True, returns ``(table, info)``.

    References
    ----------
    Druckenbrod, D.L., Cook, E.R., Pederson, N. & Martin-Benito, D. (2024).
    Dendrochronologia 85, 126195. Lorimer, C.G. & Frelich, L.E. (1989).
    Canadian Journal of Forest Research 19, 651-663.
    """
    if significance:
        raise NotImplementedError(
            "significance testing of releases is not implemented in this release "
            "(deferred pending confirmation of the test). Use significance=False.")
    if str(method).strip().lower() != "rga":
        raise ValueError("method must be 'rga' (the only detector available).")
    if window_length < 1:
        raise ValueError("window_length must be a positive integer.")

    if isinstance(data, str):
        from .readers import readers          # local import avoids an import cycle
        data = readers(data)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame or Series, or a path (str) "
                        "to an rwl/csv file, not " + str(type(data)) + ".")

    rows = []
    info = {}
    for name in data.columns:
        v = data[name].dropna()
        if v.empty:
            continue
        yrs = v.index.to_numpy()
        raw = v.to_numpy(dtype=float)
        if (~np.isnan(data[name].to_numpy())).sum() != len(raw):
            pass  # (dropna already contiguous for standard rwl input)
        rel, M, pgc = _rga_one(raw, yrs, threshold, window_length)
        if len(raw) < 2 * window_length:
            _warn(f"releases: series '{name}' has only {len(raw)} rings "
                  f"(< 2 * window_length = {2 * window_length}); skipped.")
        for yr, pg in rel:
            rows.append({"series": name, "year": yr, "pct_growth_change": pg})
        if return_info:
            info[name] = pd.DataFrame(
                {"running_mean": M, "pct_growth_change": pgc}, index=yrs)

    table = (pd.DataFrame(rows, columns=["series", "year", "pct_growth_change"])
             .sort_values(["series", "year"]).reset_index(drop=True))
    return (table, info) if return_info else table
