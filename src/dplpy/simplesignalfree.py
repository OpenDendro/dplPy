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

# Title: simplesignalfree.py
# Project: OpenDendro dplPy
# Description: Simple signal-free chronology (Melvin & Briffa 2008), a port of
#              dplR's ssf(). A chronology is built, each series is divided by it
#              to remove the common signal, the "signal-free" measurements are
#              re-detrended, and a new chronology is built; this repeats until
#              the high-frequency chronology stops changing (median absolute
#              difference below a threshold). Ports dplR's ssf() algorithm
#              (preset=None path); the CRUST-derived options depart from it as
#              documented on those parameters.

import numpy as np
import pandas as pd

from ._validate import (_require_dataframe, _warn, _alias_kwarg, _reject_kwargs,
                        _normalize_choice, _drop_allnan_columns)
from .stats import stats
from .chron import chron
from .tbrm import tbrm_rows
from .smoothingspline import spline, get_period, _caps
from .agedepspline import ads
from .detrend import _normalize_fit, _normalize_method, _FIT_CANON

# dplR's ssf() default spline is 0.6667 of the series length (detrend's is 0.67);
# both resolve through smoothingspline.get_period (audit E20).
_SSF_DEFAULT_FRACTION = 0.6667

# the basic method sets aside at most this fraction of the collection when the
def _bad_curve_cols(curves, columns):
    """Column labels whose fitted curve dips to <= 0 (ignoring all-NaN columns)."""
    bad = []
    with np.errstate(invalid="ignore"):
        for j in range(curves.shape[1]):
            col = curves[:, j]
            if np.any(np.isfinite(col)) and np.nanmin(col) <= 0:
                bad.append(columns[j])
    return bad


def _curve_cause(chron, years, difference):
    """The explanatory clause for a non-positive-curve message: the pointer-year
    value of the driving chronology, worded for the ratio or the difference
    method (difference does not divide, so it must not say "dividing by it")."""
    if chron is None or years is None:
        return ""
    with np.errstate(invalid="ignore"):
        if not np.any(np.isfinite(chron)):
            return ""
        ci = int(np.nanargmin(chron))
        val = float(np.nanmin(chron))
    yr = int(years[ci])
    if difference:
        return (" The signal-free chronology falls to " + format(val, ".4g")
                + " at year " + str(yr) + "; in difference mode the fitted curve "
                "can reach zero or below where the chronology exceeds growth in a "
                "year dominated by absent (zero-width) rings.")
    return (" This usually means the chronology is near zero in a year dominated "
            "by absent (zero-width) rings -- here it reaches " + format(val, ".4g")
            + " at year " + str(yr) + ", so dividing by it inflates the signal-free "
            "measurements and the refitted spline dips below zero.")


def _neg_curve_error(bad, chron, years, difference, crust):
    """The full error when a fitted signal-free curve goes <= 0: the offending
    series, the pointer-year cause, and the remedy. Like dplR, ssf stops here --
    a non-positive curve is driven by a near-zero chronology in a severe pointer
    year, which (as co021 shows) afflicts the whole collection rather than a few
    series, so dropping series would cascade. recode_zeros is suggested only for
    literal zero rings; preset='crust' (which floors the curve) is the route for
    data with many absent rings."""
    msg = ("[1] The signal-free detrending curve went <= 0 for series: "
           + ", ".join(str(b) for b in bad) + "."
           + _curve_cause(chron, years, difference))
    if crust:
        return msg + (" preset='crust' already floors the signal-free curve, so this "
                      "is an initial raw-data curve that cannot be fit for the series "
                      "above. See help(dplpy.ssf).")
    return msg + (" Use preset='crust', which floors the curve and keeps every series, "
                  "for data with many absent rings; recode_zeros helps only when the "
                  "series contain literal zero-width rings. See help(dplpy.ssf).")


def _sf_measurements(dat_vals, chron, difference, crust):
    """STEP 2: signal-free measurements = data / chronology (or minus, for
    difference). With crust=True, keep the raw measurement in any year where the
    chronology is near zero (< 0.01) -- CRUST's guard against the enormous
    signal-free values a near-zero chronology otherwise produces (the co021
    failure mode). dplR divides unconditionally."""
    if difference:
        return dat_vals - chron[:, None]
    sf = dat_vals / chron[:, None]
    if crust:
        guard = chron < 0.01
        sf[guard, :] = dat_vals[guard, :]
    return sf


def _sf_rescale(sf, dat_vals, crust, difference):
    """STEP 3: put each signal-free series back on its original mean. The basic
    (dplR) method shifts additively; CRUST rescales multiplicatively -- the
    model-consistent "mean offset" of Melvin & Briffa (2008, 2014), where tree
    growth is multiplicative and indices are fractional deviations."""
    mean_dat = np.nanmean(dat_vals, axis=0)
    with np.errstate(invalid="ignore"):
        mean_sf = np.nanmean(sf, axis=0)
    if crust and not difference:
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = np.where(mean_sf != 0, mean_dat / mean_sf, 1.0)
        return sf * factor
    return (sf - mean_sf) + mean_dat


def ssf(rwl,
        fit="Spline",
        method="ratio",
        spline_period=None,
        max_iterations=25,
        tol=5e-04,
        recode_zeros=False,
        return_info=False,
        verbose=True,
        preset=None,
        **kwargs):
    """Simple signal-free chronology (dplR's ssf()).

    Build a signal-free chronology (Melvin & Briffa 2008): form an initial
    chronology, divide every series by it to strip the common signal, rescale and
    re-detrend the resulting signal-free measurements, and build a new chronology,
    repeating until the high-frequency chronology stops changing (the sample-depth-
    weighted median absolute difference falls below ``tol``) or ``max_iterations``
    is reached. Reproduces dplR's ssf(); for data with many absent rings that the
    basic method cannot standardise, use ``preset="crust"`` (see the note below).

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series (raw), years as the index and series as columns. Series
        with no measured values are dropped with a warning.
    fit : {"Spline", "AgeDepSpline"}, default "Spline"
        detrending curve used at each iteration (the ``detrend`` vocabulary,
        case-insensitive): a cubic smoothing spline or an age-dependent spline
        (``dpl.ads``, fit with ``pos_slope=True`` as in dplR's ssf). Any other
        name raises. Before 0.9.0 this was called ``method=``; a curve name passed
        to ``method`` is still honoured with a DeprecationWarning.
    method : {"ratio", "difference"}, default "ratio"
        how indices are formed: ``"ratio"`` divides series by curve, ``"difference"``
        subtracts (the former ``difference=True`` flag, accepted with a
        DeprecationWarning).
    spline_period : int or float or None, default None
        stiffness of the curve. For ``fit="Spline"`` it is the spline wavelength
        (``get_period`` rule): None uses dplR ssf's 0.6667 of each series' length,
        a value in ``(0, 1]`` a fraction of length, a negative value a percent of
        length, and a value > 1 a fixed wavelength in years. For
        ``fit="AgeDepSpline"`` it is the initial stiffness ``nyrs0`` (None = 50)
        and must be an integer > 1 -- a fraction is only valid for ``Spline``.
        The former name ``nyrs=`` is accepted with a DeprecationWarning.
    max_iterations : int, default 25
        maximum signal-free iterations before giving up. Under ``preset="crust"``
        it is raised to at least 40 (CRUST's limit), as difficult sites can need
        more than 25 iterations to converge.
    tol : float, default 5e-4
        convergence threshold on the median absolute difference of the
        high-frequency chronology between successive iterations (dplR's
        ``mad.threshold``; the former name ``mad_threshold=`` is accepted with a
        DeprecationWarning).
    recode_zeros : bool, default False
        recode zero ring-widths to 0.001 before processing (avoids div-by-zero
        from literal zero rings). The input frame is not modified. Note this does
        not help a chronology that is low but positive in a severe pointer year
        (the usual cause of a non-positive signal-free curve); use ``preset="crust"``
        for that.
    return_info : bool, default False
        if True, return a dict of the full iteration history (chronologies,
        signal-free measurements and curves, MAD vector, ...) instead of just the
        final chronology.
    verbose : bool, default True
        print progress and the per-iteration convergence diagnostics.
    preset : {None, "crust"}, default None
        None reproduces the basic signal-free method (dplR's ssf; Melvin & Briffa
        2008). "crust" adds CRUST's robustness guards (Melvin & Briffa 2014):
        multiplicative rescaling of the signal-free measurements, the raw
        measurement kept in any year where the chronology is near zero (< 0.01),
        the fitted curves floored at 0.02 mm, and up to 40 iterations. These make
        ssf robust to sensitive sites with many absent rings (e.g. co021) that the
        basic method cannot standardise, and keep every series in the chronology.
        Case-insensitive.

    Returns
    -------
    pandas.DataFrame
        the signal-free chronology with columns ``sfc`` and ``samp_depth``,
        indexed by year. If ``return_info`` is True, a dict of intermediates is
        returned instead (see above).

    Notes
    -----
    The basic method stops with an error when a signal-free curve goes to zero or
    below -- typically in a year dominated by absent (zero-width) rings, which
    drives the chronology near zero so that dividing by it inflates the signal-free
    measurements and the refitted spline dips below zero. This afflicts the whole
    collection at that pointer year rather than a few individual series (on co021,
    for example, trying to drop the offending series would cascade to nearly all of
    them), so ssf does not drop series; it reports the cause and directs you to
    ``preset="crust"``, which floors the curve and standardises such data with
    every series retained. dplPy's error is more specific than dplR's but stops in
    the same place.

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/ssf.html
    .. [2] Melvin, T. M. & Briffa, K. R. (2008) A "signal-free" approach to
       dendroclimatic standardisation. Dendrochronologia, 26(2), 71-86.
    .. [3] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 1,
       Signal-Free RCS. Dendrochronologia, 32, 7-20.
    .. [4] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further
       RCS options and recommendations. Dendrochronologia, 32, 343-356.
    """
    # --- deprecated spellings (audit D3 / D5) -------------------------------
    spline_period = _alias_kwarg(kwargs, "nyrs", "spline_period", spline_period, "ssf")
    tol = _alias_kwarg(kwargs, "mad_threshold", "tol", tol, "ssf")
    if "difference" in kwargs:
        method = "difference" if _alias_kwarg(kwargs, "difference", "method",
                                              None, "ssf") else "ratio"
    _reject_kwargs(kwargs)
    if str(method).strip().lower() in _FIT_CANON:
        # the pre-0.9.0 meaning of method= was the curve name
        _warn("ssf(): method='" + str(method) + "' names a curve; the curve is now "
              "chosen with fit= and method= selects 'ratio' or 'difference'. "
              "Treating it as fit='" + str(method) + "'.", DeprecationWarning)
        fit, method = method, "ratio"
    fit = _normalize_fit(fit)
    if fit not in ("Spline", "AgeDepSpline"):
        raise ValueError("ssf: fit must be 'Spline' or 'AgeDepSpline', got '"
                         + str(fit) + "'.")
    method = _normalize_method(method)
    difference = (method == "difference")
    preset = _normalize_choice(preset, ("crust",), "preset", none_ok=True)
    crust = preset == "crust"
    nyrs = spline_period
    mad_threshold = float(tol)
    method2 = fit  # the curve name, as the iteration history reports it

    if max_iterations > 25:
        _warn("Having to set max_iterations > 25 may indicate non-ideal data for "
              "signal-free detrending.")
    if not (1e-04 < mad_threshold < 1e-03):
        _warn("The stopping criteria (tol) should probably be between 1e-4 and 1e-3 "
              "unless you have a good reason to think otherwise.")

    if crust:
        # CRUST allows up to 40 signal-free iterations; difficult sites (e.g.
        # co021) can need a few more than the basic default of 25 to converge.
        max_iterations = max(max_iterations, 40)

    # error messages for the input-validation and convergence guards
    maxIterMsg = "[2] Reached maximum iterations and stopping criteria are not satisfied. See help (?ssf)."
    crn0Msg = "[3] The initial chronology contains at least one row (year) with a zero, creating div0 problems. See help (?ssf)."
    input0Msg = "[4] Input data contain at least one row (year) with all zero values, creating div0 problems. See help (?ssf)."
    zeroColMsg = "[5] Input data contain at least one series with all zero values. See help (?ssf)."
    inputNAmsg = "[6] Input data contain at least one row (year) with all NA values, creating div0 problems. See help (?ssf)."

    # work on a copy so the caller's frame is never modified (audit A2)
    dat = _require_dataframe(rwl).copy()
    dat = _drop_allnan_columns(dat, "rwl")

    # recode zeros to 0.001 if asked.
    if recode_zeros:
        dat[dat == 0] = 0.001

    # Input validation (once, on the full collection).
    if dat.isna().all(axis=1).any():
        raise ValueError(inputNAmsg)   # unconnected floaters
    if dat.apply(lambda x: x.sum(skipna=True) == 0, axis=1).any():
        raise ValueError(input0Msg)    # an all-zero year -> chron can be zero -> div0
    if dat.apply(lambda x: x.sum(skipna=True) == 0, axis=0).any():
        raise ValueError(zeroColMsg)   # an all-zero series

    # A simplish detrending function modified from detrend.series(). detrend()
    # itself disallows negative values, so this is a trimmed-down version.
    def getCurve(y, method=method2, nyrs=None, posSlope=True):
        good_y = np.where(~np.isnan(y))[0]           # NAs are reinserted later
        if len(good_y) == 0:
            raise ValueError("All values are 'NA'")
        elif any(np.diff(good_y) != 1):
            raise ValueError("'NA's are not allowed in the middle of the series")
        y2 = y.iloc[good_y] if isinstance(y, (pd.Series, pd.DataFrame)) else y[good_y]
        y2[y2 == 0] = 0.001                          # recode zero rings to avoid div0

        if method == "Spline":
            # the spline wavelength resolves through get_period (E20) with dplR
            # ssf's 0.6667-of-length default; caps() truncates it to an integer.
            nyrs2 = get_period(nyrs, len(y2), default=_SSF_DEFAULT_FRACTION)
            Curve = _caps(np.asarray(y2, dtype=float), nyrs2)
        else:   # AgeDepSpline: spline_period is the initial stiffness nyrs0
            nyrs2 = 50 if nyrs is None else nyrs
            Curve = ads(y=y2, nyrs0=nyrs2, pos_slope=True)
        Curve2 = np.full_like(y, np.nan, dtype=float)
        Curve2[good_y] = Curve
        return Curve2

    def _ssf_once(dat):
        """One full signal-free run over the collection. Raises a clear ValueError
        when a fitted curve goes <= 0 (a near-zero pointer-year chronology), as
        well as for the validation/convergence guards."""
        na_mask = np.isnan(dat.values)
        nSeries = dat.shape[1]
        nYrs = dat.shape[0]
        medianAbsDiff = 1
        datSummary = stats(dat)
        medianSegLength = datSummary['year'].median()

        # storage: [nYrs, nSeries, max_iterations]
        sfRW_Array = np.full((nYrs, nSeries, max_iterations), np.nan)
        sfRWRescaled_Array = np.full((nYrs, nSeries, max_iterations), np.nan)
        sfRWRescaledCurves_Array = np.full((nYrs, nSeries, max_iterations), np.nan)
        sfRWI_Array = np.full((nYrs, nSeries, max_iterations), np.nan)
        sfCrn_Mat = np.full((nYrs, max_iterations), np.nan)
        hfCrn_Mat = np.full((nYrs, max_iterations), np.nan)
        MAD_Vec = np.zeros(max_iterations - 1)
        hfCrnResids_Mat = np.full((nYrs, max_iterations - 1), np.nan)

        def apply_getCurve(arr, method, nyrs=None):
            out = np.full_like(arr, np.nan, dtype=float)
            for i in range(arr.shape[1]):
                col = arr.iloc[:, i] if isinstance(arr, (pd.Series, pd.DataFrame)) else arr[:, i]
                out[:, i] = getCurve(col, method=method, nyrs=nyrs)
            return out

        def _rescale_step(sf):
            """STEP 3+4: rescale to the original mean, put sample-depth==1 rows
            back to the raw measurements, and restore ONLY the structural gaps
            (cells outside each series' span). A legitimate interior absent
            (zero-width) ring is kept -- getCurve recodes it to a small positive
            -- rather than being turned into an illegal interior NaN, which was
            the bug that broke the crust path on any absent ring."""
            tmp = _sf_rescale(sf, dat.values, crust, difference)
            tmp[datSampDepth == 1, :] = dat.values[datSampDepth == 1, :]
            tmp[na_mask] = np.nan
            stray = np.isnan(tmp) & ~na_mask
            tmp[stray] = dat.values[stray]
            return tmp

        # STEP 1 - initial chronology
        datCurves = apply_getCurve(dat, method=method2, nyrs=nyrs)
        bad = _bad_curve_cols(datCurves, dat.columns)
        if bad:                                      # raw series cannot be fit positive
            raise ValueError(_neg_curve_error(bad, None, None, difference, crust))

        datRWI = (dat.values - datCurves) if difference else (dat.values / datCurves)
        datRWI = pd.DataFrame(datRWI)
        datRWI.insert(0, "Years", dat.index)
        datRWI.set_index("Years", inplace=True)

        iter0Crn = chron(datRWI, biweight=True, plot=False)
        # Zeros in a biweight chron (very sensitive sites) cause div0 later; fall
        # back to the straight mean, then error if zeros persist (bunk data).
        if any((iter0Crn.iloc[:, 0]) == 0):
            iter0Crn = chron(datRWI, biweight=False, plot=False)
        if any((iter0Crn.iloc[:, 0]) == 0):
            raise ValueError(crn0Msg)

        datSampDepth = iter0Crn.iloc[:, 1]
        normalizedSampleDepth = np.sqrt(datSampDepth - 1) / np.sqrt(np.max(datSampDepth) - 1)
        iter0Crn_col0 = iter0Crn.iloc[:, 0].values

        # STEP 2 - signal-free measurements
        sfRW_Array[:, :, 0] = _sf_measurements(dat.values, iter0Crn_col0, difference, crust)
        # STEP 3+4 - rescale, sample-depth==1, restore structural gaps
        sfRWRescaled_Array[:, :, 0] = _rescale_step(sfRW_Array[:, :, 0])
        # STEP 5 - fit curves to the signal-free measurements
        sfRWRescaledCurves_Array[:, :, 0] = apply_getCurve(
            pd.DataFrame(sfRWRescaled_Array[:, :, 0]), method=method2, nyrs=nyrs)
        if crust:                                    # CRUST: floor the curve at 1 ring
            sfRWRescaledCurves_Array[:, :, 0] = np.maximum(sfRWRescaledCurves_Array[:, :, 0], 0.02)
        bad = _bad_curve_cols(sfRWRescaledCurves_Array[:, :, 0], dat.columns)
        if bad:
            raise ValueError(_neg_curve_error(bad, iter0Crn_col0, dat.index.to_numpy(),
                                              difference, crust))

        # STEP 6 - indices from the signal-free curves
        if difference:
            sfRWI_Array[:, :, 0] = dat.values - sfRWRescaledCurves_Array[:, :, 0]
        else:
            sfRWI_Array[:, :, 0] = dat.values / sfRWRescaledCurves_Array[:, :, 0]

        # STEP 7 - first signal-free chronology (biweight mean per year)
        sfCrn_Mat[:, 0] = tbrm_rows(sfRWI_Array[:, :, 0])
        if any(sfCrn_Mat[:, 0] == 0):
            sfCrn_Mat[:, 0] = chron(pd.DataFrame(sfRWI_Array[:, :, 0]), biweight=False, plot=False).iloc[:, 0]

        hfCrn_Mat[:, 0] = sfCrn_Mat[:, 0] - spline(
            y=sfCrn_Mat[:, 0], x=np.arange(1, len(sfCrn_Mat[:, 0]) + 1),
            period=int(np.floor(medianSegLength)))

        if verbose:
            print("Data read. First iteration done.")

        # STEP 8 - iterate (2)-(7) to the MAD threshold or max_iterations
        iterationNumber = 1
        while medianAbsDiff > mad_threshold:
            k = iterationNumber
            sfRW_Array[:, :, k] = _sf_measurements(dat.values, sfCrn_Mat[:, k - 1], difference, crust)
            sfRWRescaled_Array[:, :, k] = _rescale_step(sfRW_Array[:, :, k])

            sfRWRescaledCurves_Array[:, :, k] = apply_getCurve(
                pd.DataFrame(sfRWRescaled_Array[:, :, k]), method=method2, nyrs=nyrs)
            if crust:
                sfRWRescaledCurves_Array[:, :, k] = np.maximum(sfRWRescaledCurves_Array[:, :, k], 0.02)
            bad = _bad_curve_cols(sfRWRescaledCurves_Array[:, :, k], dat.columns)
            if bad:
                raise ValueError(_neg_curve_error(bad, sfCrn_Mat[:, k - 1],
                                                  dat.index.to_numpy(), difference, crust))

            if difference:
                sfRWI_Array[:, :, k] = dat.values - sfRWRescaledCurves_Array[:, :, k]
            else:
                sfRWI_Array[:, :, k] = dat.values / sfRWRescaledCurves_Array[:, :, k]

            sfCrn_Mat[:, k] = tbrm_rows(sfRWI_Array[:, :, k])
            if any(sfCrn_Mat[:, k] == 0):
                sfCrn_Mat[:, k] = chron(pd.DataFrame(sfRWI_Array[:, :, k]), biweight=False, plot=False).iloc[:, 0]

            hfCrn_Mat[:, k] = sfCrn_Mat[:, k] - spline(
                y=sfCrn_Mat[:, k], x=np.arange(1, len(sfCrn_Mat[:, k]) + 1),
                period=int(np.floor(medianSegLength)))
            hfCrnResids_Mat[:, k - 1] = hfCrn_Mat[:, k] - hfCrn_Mat[:, k - 1]
            medianAbsDiff = np.median(np.abs(
                hfCrn_Mat[:, k] * normalizedSampleDepth - hfCrn_Mat[:, k - 1] * normalizedSampleDepth))
            MAD_Vec[k - 1] = medianAbsDiff

            if verbose:
                print(f"Iteration: {k+1}  Median Abs Diff: {round(medianAbsDiff, 5)}  "
                      f"({round(mad_threshold / medianAbsDiff * 100, 5)}% of threshold)")

            if k == (max_iterations - 1) and medianAbsDiff > mad_threshold:
                raise ValueError(maxIterMsg)
            iterationNumber += 1

        # keep columns 0..k (slice :k+1); slicing :k would drop the final iteration
        return {
            "index": dat.index,
            "columns": dat.columns,
            "iter0_std": iter0Crn_col0,
            "samp_depth": datSampDepth,
            "k": k,
            "sfRW_Array": sfRW_Array[:, :, :k + 1],
            "sfRWRescaled_Array": sfRWRescaled_Array[:, :, :k + 1],
            "sfRWRescaledCurves_Array": sfRWRescaledCurves_Array[:, :, :k + 1],
            "sfRWI_Array": sfRWI_Array[:, :, :k + 1],
            "sfCrn_Mat": sfCrn_Mat[:, :k + 1],
            "hfCrn_Mat": hfCrn_Mat,
            "hfCrnResids_Mat": hfCrnResids_Mat[:, :k],
            "MAD_Vec": MAD_Vec[:k],
        }

    # A non-positive signal-free curve stops the run (as in dplR), with a message
    # that names the pointer-year cause and points to preset='crust' (see
    # _neg_curve_error and the Notes in the docstring).
    core = _ssf_once(dat)

    idx = core["index"]
    samp_depth = core["samp_depth"]
    k = core["k"]
    iter0Crn = pd.DataFrame({"std": core["iter0_std"], "samp_depth": samp_depth}, index=idx)
    # the final chronology is the last computed iteration (column k), matching
    # dplR's finalCrn <- sfCrn_Mat[,k].
    finalCrn = pd.DataFrame({"sfc": core["sfCrn_Mat"][:, k], "samp_depth": samp_depth}, index=idx)

    infoList = {"fit": method2, "method": method, "spline_period": nyrs,
                "max_iterations": max_iterations, "tol": mad_threshold}
    if method2 == "AgeDepSpline":
        infoList["pos_slope"] = True

    if verbose:
        print("Simple Signal Free Chronology Complete")
        print("ssf was called with these arguments")
        print(f"fit: {method2}")
        print(f"method: {method}")
        print(f"spline_period: {nyrs}")
        if method2 == "AgeDepSpline":
            print("pos_slope: True")
        print(f"max_iterations: {max_iterations}")
        print(f"tol: {mad_threshold}")

    if return_info:
        # The full iteration history is a mix of DataFrames, 3-D arrays and a
        # dict, so it is returned as a dict (not coerced to a DataFrame, which
        # was the cause of the previous return_info error).
        return {
            "infoList": infoList,
            "iter0Crn": iter0Crn,
            "ssfCrn": finalCrn,
            "sfRW_Array": core["sfRW_Array"],             # SF measurements
            "sfRWRescaled_Array": core["sfRWRescaled_Array"],       # rescaled SF measurements
            "sfRWRescaledCurves_Array": core["sfRWRescaledCurves_Array"],  # rescaled SF curves
            "sfRWI_Array": core["sfRWI_Array"],           # SF RWI
            "sfCrn_Mat": core["sfCrn_Mat"],               # SF chronologies by iteration
            "hfCrn_Mat": core["hfCrn_Mat"],               # high-frequency chronologies
            "hfCrnResids_Mat": core["hfCrnResids_Mat"],   # HF chronology residuals
            "MAD_Vec": core["MAD_Vec"],                   # median abs diff per iteration
        }
    return finalCrn
