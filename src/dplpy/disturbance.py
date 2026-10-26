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
#              disturbance_detrend() then power-transforms each series, removes each
#              release with an age-dependent spline (dplPy's ads, the same routine
#              as Dan's ADSmelvin), age-detrends the disturbance-free series, and
#              returns a climate index that drops into chron() like detrend output.
#
#              Parity: both are validated against the MATLAB golden master
#              (v267_rga, sig=0) on Dan's mo024 post-oak test set. The detector
#              reproduces 94 releases across 52 of 61 series with identical release
#              years; the pipeline reproduces the per-series disturbance series, age
#              trend, index and residual to ~1e-11 (float64 round-off). The MATLAB
#              source and validation scripts live in dev/druckenbrod/. Where a
#              comment says "parity:" it reproduces the MATLAB exactly, not
#              necessarily what is statistically ideal.

import numpy as np
import pandas as pd

from ._validate import _warn
from .agedepspline import ads


def _rga_one(raw, yrs, threshold, window_length, min_mean=0.0):
    """Radial-growth averaging on ONE gap-free raw ring-width series.

    Returns ``(releases, running_mean, pct_growth_change)`` where ``releases`` is a
    list of ``(year, pgc)`` tuples, one per above-threshold segment, placed at the
    year of maximum percent growth change in that segment. ``running_mean`` and
    ``pct_growth_change`` are full-length arrays (NaN where undefined) for plotting.

    Mirrors v267_rga's RGA block. ``raw`` is float ring widths with no NaN, ``yrs``
    the aligned calendar years, both length n. Returns empty if n < 2*mav.

    ``min_mean`` guards the denominator: a release is only eligible where the
    preceding running mean is at least ``min_mean`` mm, which keeps a near-zero
    baseline (e.g. a window dominated by absent/recoded rings) from inflating the
    percent growth change into a spurious release. ``min_mean=0`` reproduces
    v267_rga exactly.
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
        with np.errstate(divide="ignore", invalid="ignore"):
            pgc[g] = (M[g + mav] - M[g]) / M[g] * 100.0
        if pgc[g] >= threshold and M[g] >= min_mean:   # min_mean=0 -> exact v267_rga
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


def _recode_zeros_or_warn(data, recode_zeros, where):
    """Handle literal zero (locally absent) rings before processing.

    If ``recode_zeros`` is True, return a COPY with zeros recoded to 0.001 (the
    same convention as ``dpl.ssf``; the caller's frame is never modified). If
    False and zeros are present, warn (naming the affected series) rather than
    letting them propagate silently, and return the frame unchanged. NaN is no
    data and is left alone; only literal zeros are touched.
    """
    zero_cols = [c for c in data.columns if (data[c] == 0).any()]
    if recode_zeros:
        if zero_cols:
            data = data.copy()
            data[data == 0] = 0.001
        return data
    if zero_cols:
        shown = ", ".join(zero_cols[:6]) + (" ..." if len(zero_cols) > 6 else "")
        _warn(f"{where}: {len(zero_cols)} series contain zero (absent) rings "
              f"({shown}); these are treated as zero growth and can yield NaN in "
              f"the log-transform branch. Pass recode_zeros=True to recode them to "
              f"0.001 before processing.")
    return data


def releases(data, threshold=50, window_length=15, significance=False,
             method="rga", recode_zeros=False, min_mean=0.01, return_info=False):
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
    recode_zeros : bool, default False
        recode literal zero (locally absent) rings to 0.001 before detection, the
        same convention as ``dpl.ssf`` (the input frame is never modified). When
        False, series containing zeros are detected as-is and a warning names them.
    min_mean : float, default 0.01
        minimum preceding running-mean ring width, in mm, for a year to be eligible
        as a release. Guards the percent-growth-change denominator so a near-zero
        baseline (a window dominated by absent or recoded rings) cannot inflate into
        a spurious release. The default touches only such pathological windows;
        set 0 to disable the guard (exact v267_rga behaviour).
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
    data = _recode_zeros_or_warn(data, recode_zeros, "releases")

    rows = []
    info = {}
    for name in data.columns:
        v = data[name].dropna()
        if v.empty:
            continue
        yrs = v.index.to_numpy()
        raw = v.to_numpy(dtype=float)
        rel, M, pgc = _rga_one(raw, yrs, threshold, window_length, min_mean)
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


def _power_transform(raw, power):
    """Druckenbrod's power transformation of one raw series.

    Returns ``(transformed, p)``. The optimal power is ``1 - slope`` from the
    regression of ``log|first difference|`` on ``log(subsequent ring width)`` over
    pairs where both are non-zero (natural logs, parity with v267_rga). ``p <= 0.05``
    takes natural logs, ``p > 1`` is clamped to 1, otherwise the series is raised to
    ``p``. ``power=False`` skips the transform (identity, ``p`` returned as None).
    """
    if not power:
        return np.asarray(raw, dtype=float), None
    raw = np.asarray(raw, dtype=float)
    sub = raw[1:]                        # fdiff(:,1) = raw(x+1)
    dif = np.abs(np.diff(raw))           # fdiff(:,2) = |raw(x+1) - raw(x)|
    mask = (sub != 0) & (dif != 0)
    x = np.log(sub[mask])
    y = np.log(dif[mask])
    X = np.column_stack([np.ones_like(x), x])
    slope = np.linalg.lstsq(X, y, rcond=None)[0][1]
    p = 1.0 - slope
    if p <= 0.05:
        return np.log(raw), p
    if p > 1:
        return raw ** 1.0, 1.0
    return raw ** p, p


def _disturb_one(raw, yrs, rel_idx, window_length, release_flex, age_flex, power):
    """Run the full Druckenbrod pipeline on ONE gap-free raw series.

    ``rel_idx`` are the release positions (0-based, ascending) from detection.
    Returns a dict of full-length arrays: ``DSt`` (disturbance standardized),
    ``At`` (age trend), ``St`` (climate index = DSt/At), ``StTR`` (residual index in
    transformed space), ``Dt`` (disturbance increment raw-DSt), and ``power``.
    Mirrors v267_rga after detection, line for line.
    """
    n = len(raw)
    mav = int(window_length)
    rawT, p = _power_transform(raw, power)

    # INCREMENTAL DETRENDING of each release, newest to oldest.
    DStT = rawT.copy()
    for q in range(len(rel_idx) - 1, -1, -1):
        ri = int(rel_idx[q])
        rline = np.asarray(ads(DStT[ri:n], nyrs0=release_flex, pos_slope=True), float)
        DSti = DStT.copy()
        DStT[ri:n] = DStT[ri:n] + rawT[ri] - rline          # detrend and re-anchor
        lo = ri + mav - 1                                    # transient test region
        if lo < n:
            cond = DSti[lo:n] < DStT[lo:n]
            if cond.any():                                   # transient release
                transient = int(np.argmax(cond)) + mav
                DStT = DSti.copy()                           # revert, detrend part way
                DStT[ri:ri + transient] = (DSti[ri:ri + transient]
                                           + rawT[ri] - rline[:transient])

    # Age detrend the disturbance-free series, then the residual index.
    AtT = np.asarray(ads(DStT, nyrs0=age_flex, pos_slope=True), float)
    StTR = DStT - AtT

    # Re-express in original units.
    if power and p <= 0.05:
        DSt = np.exp(DStT)
        At = np.exp(AtT)
    elif power:
        DStTf = DStT.copy()
        DStTf[DStTf < 0] = 0.001                             # parity: floor negatives
        DSt = DStTf ** (1.0 / p)
        At = AtT ** (1.0 / p)
    else:
        DSt = DStT
        At = AtT

    Dt = (raw - DSt) if len(rel_idx) > 0 else np.zeros(n)
    St = DSt / At
    return dict(DSt=DSt, At=At, St=St, StTR=StTR, Dt=Dt, power=p)


def disturbance_detrend(data, threshold=50, window_length=15, significance=False,
                        release_flex=10, age_flex=30, power=True,
                        recode_zeros=False, min_mean=0.01, return_info=False):
    """Standardize ring widths by removing canopy-disturbance releases and the age trend.

    The Druckenbrod et al. (2024) method. For each series: detect releases by radial
    growth averaging (see ``releases``), power-transform the series, remove each
    release in turn (newest first) with a fixed-length age-dependent spline while it
    sustains elevated growth, age-detrend the disturbance-free series with a second
    spline, and re-express in ring-width units. The climate index is the ratio of
    the disturbance-standardized series to the age trend.

    The returned index is per series and detrend-style, so it feeds ``dpl.chron``
    exactly like ``dpl.detrend`` output.

    Parameters
    ----------
    data : pandas DataFrame or Series, or str
        ring widths from ``dpl.readers``, a single series, or a path to an rwl/csv
        file. Series shorter than ``2 * window_length`` are returned all-NaN with a
        warning (parity with v267_rga, which cannot analyze them).
    threshold : int or float, default 50
        percent-growth-change threshold for a release (see ``releases``).
    window_length : int, default 15
        running-mean window in years for detection, and the offset used by the
        transient-release test.
    significance : bool, default False
        significance testing of releases. Not implemented; ``True`` raises
        ``NotImplementedError``.
    release_flex : int, default 10
        stiffness (``nyrs0``) in years of the age-dependent spline fit to each
        release event.
    age_flex : int, default 30
        stiffness (``nyrs0``) in years of the age-dependent spline fit to the
        disturbance-free series for the final age detrend.
    power : bool, default True
        apply Druckenbrod's power transformation before detrending and re-express
        afterwards. This is the method's own transform (optimal power ``1 - slope``,
        natural-log branch at power <= 0.05); it diverges from ``dpl.powt`` and is
        kept as the validated default. ``False`` operates in raw ring-width units
        with no transform.
    recode_zeros : bool, default False
        recode literal zero (locally absent) rings to 0.001 before processing, the
        ``dpl.ssf`` convention (the input frame is never modified). Absent rings
        otherwise flow through as zero growth and can yield NaN in the log-transform
        branch; when False and zeros are present, a warning names the series.
    min_mean : float, default 0.01
        minimum preceding running-mean ring width (mm) for a release to be eligible
        (see ``releases``). Guards against a near-zero baseline inflating the
        percent growth change; set 0 for exact v267_rga behaviour.
    return_info : bool, default False
        if True, return a dict instead of just the index (see Returns).

    Returns
    -------
    pandas.DataFrame or Series
        the age-and-disturbance-standardized climate index, same type as ``data``
        (DataFrame of years x series, or a single Series), NaN outside each series'
        span. If ``return_info`` is True, returns a dict with:
        ``"index"`` (the climate index), ``"disturbance_standardized"`` (DSt),
        ``"age_trend"`` (At), ``"residual_index"`` (StTR, the transformed-space
        residual DStT - AtT), ``"releases"`` (the detection table from ``releases``),
        and ``"averages"`` (a DataFrame of the straight across-series means: ring
        width, age trend, disturbance increment Dt, index, and running sample depth).

    References
    ----------
    Druckenbrod, D.L., Cook, E.R., Pederson, N. & Martin-Benito, D. (2024).
    Dendrochronologia 85, 126195.
    """
    if significance:
        raise NotImplementedError(
            "significance testing of releases is not implemented in this release "
            "(deferred pending confirmation of the test). Use significance=False.")
    if window_length < 1:
        raise ValueError("window_length must be a positive integer.")

    was_series = isinstance(data, pd.Series)
    if isinstance(data, str):
        from .readers import readers
        data = readers(data)
    if was_series:
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame or Series, or a path (str) "
                        "to an rwl/csv file, not " + str(type(data)) + ".")
    data = _recode_zeros_or_warn(data, recode_zeros, "disturbance_detrend")

    index = pd.DataFrame(index=data.index, dtype=float)
    dst = pd.DataFrame(index=data.index, dtype=float)
    at = pd.DataFrame(index=data.index, dtype=float)
    sttr = pd.DataFrame(index=data.index, dtype=float)
    dt = pd.DataFrame(index=data.index, dtype=float)
    rel_rows = []

    for name in data.columns:
        v = data[name].dropna()
        full = pd.Series(np.nan, index=data.index, dtype=float)
        index[name] = full
        dst[name] = full.copy(); at[name] = full.copy()
        sttr[name] = full.copy(); dt[name] = full.copy()
        if v.empty:
            continue
        yrs = v.index.to_numpy()
        raw = v.to_numpy(dtype=float)
        if len(raw) < 2 * window_length:
            _warn(f"disturbance_detrend: series '{name}' has only {len(raw)} rings "
                  f"(< 2 * window_length = {2 * window_length}); returned as NaN.")
            continue
        rel, _, _ = _rga_one(raw, yrs, threshold, window_length, min_mean)
        rel_idx = sorted(int(np.where(yrs == yr)[0][0]) for yr, _ in rel)
        res = _disturb_one(raw, yrs, rel_idx, window_length,
                           release_flex, age_flex, power)
        index.loc[yrs, name] = res["St"]
        dst.loc[yrs, name] = res["DSt"]
        at.loc[yrs, name] = res["At"]
        sttr.loc[yrs, name] = res["StTR"]
        dt.loc[yrs, name] = res["Dt"]
        for yr, pg in rel:
            rel_rows.append({"series": name, "year": yr, "pct_growth_change": pg})

    if not return_info:
        return index[data.columns[0]] if was_series else index

    raw_nozero = data.copy()
    depth = raw_nozero.notna().sum(axis=1)
    averages = pd.DataFrame({
        "ring_width": raw_nozero.mean(axis=1, skipna=True),
        "age_trend": at.mean(axis=1, skipna=True),
        "Dt": (raw_nozero - dst).mean(axis=1, skipna=True),
        "index": index.mean(axis=1, skipna=True),
        "sample_depth": depth,
    })
    releases_table = (pd.DataFrame(rel_rows,
                                   columns=["series", "year", "pct_growth_change"])
                      .sort_values(["series", "year"]).reset_index(drop=True))
    return {
        "index": index[data.columns[0]] if was_series else index,
        "disturbance_standardized": dst[data.columns[0]] if was_series else dst,
        "age_trend": at[data.columns[0]] if was_series else at,
        "residual_index": sttr[data.columns[0]] if was_series else sttr,
        "releases": releases_table,
        "averages": averages,
    }
