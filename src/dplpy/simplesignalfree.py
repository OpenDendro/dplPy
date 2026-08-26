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
#              difference below a threshold). Reproduces dplR's ssf() to machine
#              precision.

import numpy as np
import pandas as pd

from .readers import readers
from .stats import stats
from .chron import chron
from .smoothingspline import spline
from .agedepspline import ads


def _neg_curve_message(curves, series_names, chron=None, years=None):
    """Build an informative error for the non-positive signal-free curve guard.

    Reports which series' fitted curve dipped to <= 0 and, when the driving
    chronology is supplied, the near-zero chronology value and year that is the
    usual root cause (a year dominated by absent, zero-width rings inflates the
    signal-free measurements once every series is divided by that near-zero
    chronology).
    """
    bad = [str(series_names[j]) for j in range(curves.shape[1])
           if np.nanmin(curves[:, j]) <= 0]
    msg = ("[1] The signal-free detrending curve went <= 0 for series: "
           + ", ".join(bad) + ".")
    if chron is not None and years is not None and np.isfinite(np.nanmin(chron)):
        ci = int(np.nanargmin(chron))
        msg += (" This usually means the chronology is near zero in a year "
                "dominated by absent (zero-width) rings -- here it reaches "
                + format(float(np.nanmin(chron)), ".4g")
                + " at year " + str(int(years[ci]))
                + ", so dividing by it inflates the signal-free measurements and "
                "the refitted spline dips below zero. Consider a larger "
                "recode_zeros value, or a more robust signal-free approach for "
                "data with many absent rings.")
    msg += " See help(dplpy.ssf)."
    return msg


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
        method="Spline",
        nyrs=None,
        difference=False,
        max_iterations=25,
        mad_threshold=5e-04,
        recode_zeros=False,
        return_info=False,
        verbose=True,
        preset=None):
    """Simple signal-free chronology (dplR's ssf()).

    Extended Summary
    ----------------
    Builds a signal-free chronology (Melvin & Briffa 2008): an initial
    chronology is formed, every series is divided by it to strip the common
    signal, the resulting "signal-free" measurements are rescaled and
    re-detrended, and a new chronology is built. This repeats until the
    high-frequency component of the chronology stops changing between iterations
    (the sample-depth-weighted median absolute difference falls below
    ``mad_threshold``) or ``max_iterations`` is reached. Reproduces dplR's ssf().

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series (raw), years as the index and series as columns.
    method : {"Spline", "AgeDepSpline"}, default "Spline"
        detrending curve used at each iteration: a cubic smoothing spline
        (2/3-length stiffness) or an age-dependent spline (dpl.ads()).
    nyrs : int or None, default None
        spline stiffness. None uses 2/3 of each series' length for "Spline" or
        50 for "AgeDepSpline"; a value in (0, 1) is a fraction of series length.
    difference : bool, default False
        detrend by subtraction (series - curve) rather than division.
    max_iterations : int, default 25
        maximum signal-free iterations before giving up.
    mad_threshold : float, default 5e-4
        convergence threshold on the median absolute difference of the
        high-frequency chronology between successive iterations.
    recode_zeros : bool, default False
        recode zero ring-widths to 0.001 before processing (avoids div-by-zero).
    return_info : bool, default False
        if True, return a dict of the full iteration history (chronologies,
        signal-free measurements and curves, MAD vector, ...) instead of just
        the final chronology.
    verbose : bool, default True
        print progress and the per-iteration convergence diagnostics.
    preset : {None, "crust"}, default None
        None reproduces the basic signal-free method (dplR's ssf; Melvin &
        Briffa 2008). "crust" turns on the guards/refinements from CRUST (Melvin
        & Briffa 2014): the signal-free measurements are rescaled multiplicatively
        (the model-consistent "mean offset") rather than additively; the raw
        measurement is kept in any year where the chronology is near zero
        (< 0.01); and the fitted detrending curves are floored at 0.02 mm. These
        make ssf robust to sensitive sites with many absent rings (e.g. co021),
        which the basic method cannot standardise.

    Returns
    -------
    pandas.DataFrame
        the signal-free chronology with columns ``sfc`` and ``samp.depth``,
        indexed by year. If ``return_info`` is True, a dict of intermediates is
        returned instead (see above).

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
    if preset not in (None, "crust"):
        raise ValueError("preset must be None or 'crust', got '" + str(preset) + "'.")
    crust = preset == "crust"

    if max_iterations > 25:
        print("Warning: Having to set max_iterations > 25 may indicate non-ideal data for signal-free detrending.")
    if  not(1e-04 < mad_threshold < 1e-03):
        print("Warning: The stopping criteria should probably be between 1e-5 and 1e-4 unless you have a good reason to think otherwise.")

    if crust:
        # CRUST allows up to 40 signal-free iterations; difficult sites (e.g.
        # co021) can need a few more than the basic default of 25 to converge.
        max_iterations = max(max_iterations, 40)
    

    # error msgs for later

    maxIterMsg = "[2] Reached maximum iterations and stopping criteria are not satisfied. See help (?ssf)."

    crn0Msg = "[3] The initial chronology contains at least one row (year) with a zero, creating div0 problems. See help (?ssf)."

    input0Msg = "[4] Input data contain at least one row (year) with all zero values, creating div0 problems. See help (?ssf)."

    zeroColMsg = "[5] Input data contain at least one series with all zero values. See help (?ssf)."

    inputNAmsg = "[6] Input data contain at least one row (year) with all NA values, creating div0 problems. See help (?ssf)."

    # make a copy of rwl just in case we change it.     
    dat = rwl

    # check class of rwl
    if not isinstance(dat, pd.DataFrame):
        print("Input data needs to be a rwl DataFrame. Attempting to coerce.")
        dat = readers(dat)  #will error if cannot coerce 

    
   # recode zeros to 0.001 if asked.
    if recode_zeros:
        dat[dat == 0] = 0.001

    # Look for any rows where all the values are NA -- unconnected floaters
    if dat.isna().all(axis=1).any():
        raise ValueError(inputNAmsg)

    # Can't have all zeros across the board for a year. This is
    # a conservative check but if there are zeros for a year, the chron can eval to zero
    # which causes headaches with div0.
    zeroRowCheck = dat.apply(lambda x: x.sum(skipna=True) == 0, axis=1)
    if zeroRowCheck.any():
        raise ValueError(input0Msg)

    # Heck, look for zeros in series too. Never know what kind of silliness users come up with.
    zeroColCheck = dat.apply(lambda x: x.sum(skipna=True) == 0, axis=0)
    if zeroColCheck.any():
        raise ValueError(zeroColMsg)

    # get some detrending options
    method2 = method if method in ["Spline", "AgeDepSpline"] else "AgeDepSpline"

    # useful vars
    nSeries = dat.shape[1]
    nYrs = dat.shape[0]
    medianAbsDiff = 1
    datSummary = stats(dat)
    medianSegLength = datSummary['year'].median()

    # Make some storage objects
    # These are arrays of [nYrs,nSeries,max_iterations]
    # Array to hold the SF measurements
    sfRW_Array = np.full((nYrs, nSeries, max_iterations), np.nan)
    # Array to hold the rescaled SF measurements

    sfRWRescaled_Array = np.full((nYrs, nSeries, max_iterations), np.nan)
    # Array to hold the rescaled SF curves

    sfRWRescaledCurves_Array = np.full((nYrs, nSeries, max_iterations), np.nan)
    # Array to hold the SF RWI

    sfRWI_Array = np.full((nYrs, nSeries, max_iterations), np.nan)
    # Array (2d though) to hold the SF Crn

    sfCrn_Mat = np.full((nYrs, max_iterations), np.nan)
    # Array (2d though) to hold the HF Crn

    hfCrn_Mat = np.full((nYrs, max_iterations), np.nan)
    # Vector for storing median absolute difference (mad)

    MAD_Vec = np.zeros(max_iterations - 1)
    # Array (2d though) to hold the differences between the kth
    # and the kth-1 high freq chronology residuals
    hfCrnResids_Mat = np.full((nYrs, max_iterations - 1), np.nan)

    # Let's do it. First, here is a simplish detrending function modified from
    # detrend.series(). The issue with using detrend() is that negative values are
    # not allowed for the detrend funcs. Maybe they should be (e.g., z-scored 
    # data) but they aren't as of right now. So here is a simplified detrend function.
    def getCurve(y, method=method2, 
                 nyrs=None, 
                 posSlope=True):

        ## Remove NA from the data (they will be reinserted later)
        good_y = np.where(~np.isnan(y))[0]
        
        if len(good_y) == 0:
            raise ValueError("All values are 'NA'")
        elif any(np.diff(good_y) != 1):
            raise ValueError("'NA's are not allowed in the middle of the series")
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y2 = y.iloc[good_y]
        else:
            y2 = y[good_y]

        nY2 = len(y2) #not used (in R either)

        ## Recode any zero values to 0.001 to avoid div0
        y2[y2 == 0] = 0.001

        if method == "Spline":
            ## Age dep smoothing spline with nyrs (50 default) as the init stiffness
            ## are NULL
            if nyrs is None:
                nyrs2 = len(y2) * 0.6667
            elif 0 < nyrs < 1:
                nyrs2 = len(y2) * nyrs
            else:
                nyrs2 = nyrs

            y_inds = np.arange(1, len(y2) + 1)

            # dplR's caps() truncates its nyrs to an integer (caps.R:
            # "stiffness = as.integer(nyrs)"), so nyrs = length(y2)*0.6667 is
            # effectively floored before the spline is fit. Match that here --
            # passing the raw float instead leaves the spline ~1e-4 off caps.
            Curve = spline(y=y2, x=y_inds, period=int(nyrs2))
            # Put NA back in
            Curve2 = np.full_like(y, np.nan)
            Curve2[good_y] = Curve

        elif "AgeDepSpline" in method2: 
            nyrs2 = 50 if nyrs is None else nyrs
            Curve = ads(y=y2, nyrs0=nyrs2, pos_slope=True)
            Curve2 = np.full_like(y, np.nan)
            Curve2[good_y] = Curve

        return Curve2
    
   
    def apply_getCurve(dat, method, nyrs=None):
        n_rows, n_cols = dat.shape
        datCurves = np.full_like(dat, np.nan)

        for i in range(n_cols):
            if isinstance(dat, (pd.Series, pd.DataFrame)):
                y = dat.iloc[:, i]
            else:
                y = dat[:, i]


            datCurves[:, i] = getCurve(y, method=method, nyrs=nyrs)

        return datCurves

    # STEP 1 - GET AN INITIAL CHRONOLOGY
    # fit curves
    datCurves = apply_getCurve(dat, 
                               method=method2, 
                               nyrs=nyrs)
    
    if np.any(datCurves[~np.isnan(datCurves)] <= 0):
        raise ValueError(_neg_curve_message(datCurves, dat.columns))

    # get RWI
    if difference:
        datRWI = dat.values - datCurves
    else:
        datRWI = dat.values / datCurves

    datRWI= pd.DataFrame(datRWI) 
    datRWI.insert(0, "Years", dat.index)
    datRWI.set_index("Years", inplace=True)

    # and initial chron at iter0
    iter0Crn = chron(datRWI, biweight=True,plot=False)
    # Check for zeros in the chronology. This can happen in VERY sensitive
    # chrons with years that mostly zeros if the chron is built with tukey's
    # biweight robust mean (e.g., co021). This causes problems with div0 later on
    # so if there are any zeros in the chron, switch straight mean which should
    # head off any zeros in the chron unless the data themseleves are bunk.
    # e.g., UT024.
    if any((iter0Crn.iloc[:,0]) == 0):
        iter0Crn = chron(datRWI, biweight=False,plot=False)

    # Additional check. If there are still zeros it should mean that the OG data were passed in with zeros.
    if any((iter0Crn.iloc[:, 0]) == 0):
        raise ValueError(crn0Msg)

    datSampDepth = iter0Crn.iloc[:,1] # for later
    normalizedSampleDepth = np.sqrt(datSampDepth - 1) / np.sqrt(np.max(datSampDepth)-1) # for later
    iter0Crn_col0 = iter0Crn.iloc[:, 0].values # just keep the chron
    
    # STEP 2 - Divide each series of measurements by the chronology
    # NB: This can produce some very very funky values when iter0Crn is near zero.
    # E.g., in co021 row 615 has a tbrm RWI of 0.0044 which makes for some huge SF
    sfRW_Array[:, :, 0] = _sf_measurements(dat.values, iter0Crn_col0, difference, crust)

    # STEP 3 - Rescale to the original mean (additive; multiplicative for crust)
    sfRWRescaled_Array[:, :, 0] = _sf_rescale(sfRW_Array[:, :, 0], dat.values, crust, difference)

    # STEP 4 - Replace signal-free measurements with original measurements when samp depth is 1
    sfRWRescaled_Array[datSampDepth == 1, :, 0] = dat.values[datSampDepth == 1, :] # can this break if there is no sampDepth==1?

    # STEP 5 - Fit curves to signal free measurements
    sfRWRescaledCurves_Array[:, :, 0] = np.apply_along_axis(getCurve, axis=0, arr=sfRWRescaled_Array[:, :, 0], method=method2, nyrs=nyrs)
    if crust:                                  # CRUST: floor the curve at 1 ring (0.02 mm)
        sfRWRescaledCurves_Array[:, :, 0] = np.maximum(sfRWRescaledCurves_Array[:, :, 0], 0.02)

    if np.any(sfRWRescaledCurves_Array[:, :, 0] <= 0):
        raise ValueError(_neg_curve_message(sfRWRescaledCurves_Array[:, :, 0],
                                            dat.columns, iter0Crn_col0, dat.index.to_numpy()))

    # STEP 6 - divide original measurements by curve obtained from signal free measurements fitting
    if difference:
        sfRWI_Array[:, :, 0] = dat.values - sfRWRescaledCurves_Array[:, :, 0]
    else:
        sfRWI_Array[:, :, 0] = dat.values / sfRWRescaledCurves_Array[:, :, 0]

    # STEP 7 - create 1st signal-free chronology
    sfCrn_Mat[:, 0] = chron(pd.DataFrame(sfRWI_Array[:, :, 0]), biweight=True, plot=False).iloc[:, 0]
    # Check for zeros in the chronology. This can happen in VERY sensitive
    # chrons with years that mostly zeros if the chron is built with tukey's
    # biweight robust mean (e.g., co021). This causes problems with div0 later on
    # so if there are any zeros in the chron, switch straight mean which should
    # head off any zeros in the chron unless the data themseleves are bunk
    if any(sfCrn_Mat[:, 0] == 0):
        sfCrn_Mat[:, 0] = chron(pd.DataFrame(sfRWI_Array[:, :, 0]), biweight=False,plot=False).iloc[:, 0]
    
    # And calc the high freq crn that will be used to determine MAD stopping crit
    hfCrn_Mat[:, 0] = sfCrn_Mat[:, 0] - spline(y=sfCrn_Mat[:, 0], x=np.arange(1, len(sfCrn_Mat[:, 0]) + 1), period=int(np.floor(medianSegLength)))
    
    # STEP 8 - Repeat (2) through (7) until the MAD threshold
    # is reached or we hit maxIter
    if verbose:
        print("Data read. First iteration done.")

    iterationNumber = 1 # Start on 2 b/c we did one above

    while medianAbsDiff > mad_threshold:
        k = iterationNumber

        # STEP 2 - Divide each series of measurements by the last SF chronology
        sfRW_Array[:, :, k] = _sf_measurements(dat.values, sfCrn_Mat[:, k - 1], difference, crust)

        # STEP 3 - Rescale to the original mean (additive; multiplicative for crust)
        tmp = _sf_rescale(sfRW_Array[:, :, k], dat.values, crust, difference)

        # can get a nan if unlucky. set to? zero?
        tmp[np.isnan(tmp)] = 0
        sfRWRescaled_Array[:, :, k] = tmp
        # STEP 4 - Replace signal-free measurements with original measurements when sample depth is one
        sfRWRescaled_Array[datSampDepth == 1, :, k] = dat.values[datSampDepth == 1, :]

        #add this line bc of python/R matrix difference
        sfRWRescaled_Array[:, :, k] = np.where(sfRWRescaled_Array[:, :, k] == 0, np.nan, sfRWRescaled_Array[:, :, k])

        # STEP 5 - fit curves to signal free measurements
        sfRWRescaledCurves_Array[:, :, k] = np.apply_along_axis(getCurve, axis=0, arr=sfRWRescaled_Array[:, :, k], method=method2, nyrs=nyrs)
        if crust:                              # CRUST: floor the curve at 1 ring (0.02 mm)
            sfRWRescaledCurves_Array[:, :, k] = np.maximum(sfRWRescaledCurves_Array[:, :, k], 0.02)

        if np.any(sfRWRescaledCurves_Array[:, :, k] <= 0):
            raise ValueError(_neg_curve_message(sfRWRescaledCurves_Array[:, :, k],
                                                dat.columns, sfCrn_Mat[:, k - 1],
                                                dat.index.to_numpy()))
    
        # STEP 6 - divide original measurements by curve obtained from signal free curves    
        if difference:
            sfRWI_Array[:, :, k] = dat.values - sfRWRescaledCurves_Array[:, :, k]
        else:
            sfRWI_Array[:, :, k] = dat.values / sfRWRescaledCurves_Array[:, :, k]
        
        # STEP 7 - create kth signal-free chronology
        sfCrn_Mat[:, k] = chron(pd.DataFrame(sfRWI_Array[:, :, k]), biweight=True, plot=False).iloc[:, 0]
        # Check for zeros in the chronology. This can happen in VERY sensitive
        # chrons with years that mostly zeros if the chron is built with tukey's
        # biweight robust mean (e.g., co021). This causes problems with div0 later on
        # so if there are any zeros in the chron, switch straight mean which should
        # head off any zeros in the chron unless the data themseleves are bunk   
        if any(sfCrn_Mat[:, k] == 0):
            sfCrn_Mat[:, k] = chron(pd.DataFrame(sfRWI_Array[:, :, k]), biweight=False, plot=False).iloc[:, 0]

        # Now look at diffs in fit using median abs diff in the high freq resids
        # This is the (high freq) resids from the current iter minus the resids from prior iter
        hfCrn_Mat[:, k] = sfCrn_Mat[:, k] - spline(y=sfCrn_Mat[:, k], x=np.arange(1, len(sfCrn_Mat[:, k]) + 1), period=int(np.floor(medianSegLength)))
        
        hfCrnResids_Mat[:, k - 1] = hfCrn_Mat[:, k] - hfCrn_Mat[:, k - 1]
        # calculate the median absolute differences weighted by the normalized sample depth
        medianAbsDiff = np.median(np.abs(hfCrn_Mat[:, k] * normalizedSampleDepth - hfCrn_Mat[:, k - 1] * normalizedSampleDepth))
        
        MAD_Vec[k - 1] = medianAbsDiff
        
        if verbose:
            print(f"Iteration: {k+1}  Median Abs Diff: {round(medianAbsDiff, 5)}  ({round(mad_threshold / medianAbsDiff * 100, 5)}% of threshold)")

        if k == (max_iterations-1) and medianAbsDiff > mad_threshold:
            raise ValueError(maxIterMsg)
        iterationNumber += 1
    # Remove empty NAs from output that aren't needed anymore. `k` is the
    # 0-based index of the LAST computed iteration, so keep columns 0..k
    # (slice :k+1) -- slicing :k would drop the final chronology.
    # Trim the SF measurements
    sfRW_Array = sfRW_Array[:, :, :k + 1]
    # Trim the rescaled SF measurements
    sfRWRescaled_Array = sfRWRescaled_Array[:, :, :k + 1]
    # Trim the rescaled SF curves
    sfRWRescaledCurves_Array = sfRWRescaledCurves_Array[:, :, :k + 1]
    # Trim the SF RWI
    sfRWI_Array = sfRWI_Array[:, :, :k + 1]
    # Trim the SF crn
    sfCrn_Mat = sfCrn_Mat[:, :k + 1]
    # Trim the differences
    MAD_Vec = MAD_Vec[:k]
    hfCrnResids_Mat = hfCrnResids_Mat[:, :k]

    ### return final crn and add in the OG crn too for completeness

    iter0Crn = pd.DataFrame({"std": iter0Crn_col0, "samp.depth": datSampDepth}, index=dat.index)
    # the final chronology is the last computed iteration (column k), matching
    # dplR's finalCrn <- sfCrn_Mat[,k].
    finalCrn = pd.DataFrame({"sfc": sfCrn_Mat[:, k], "samp.depth": datSampDepth}, index=dat.index)

    if method2 == "AgeDepSpline":
        infoList = {"method": method2, "nyrs": nyrs, "pos_slope": True,
                        "max_iterations": max_iterations, "mad_threshold": mad_threshold}
    else:
        infoList = {"method": method2, "nyrs": nyrs, "max_iterations": max_iterations, 
                        "mad_threshold": mad_threshold}
    
    if verbose:
        print("Simple Signal Free Chronology Complete")
        print("ssf was called with these arguments")
        print(f"Detrending method: {method2}")
        print(f"nyrs: {nyrs}")
        if method2 == "AgeDepSpline":
            print("pos_slope: True")
        print(f"max_iterations: {max_iterations}")
        print(f"mad_threshold: {mad_threshold}")
    
    if return_info:
        # The full iteration history is a mix of DataFrames, 3-D arrays and a
        # dict, so it is returned as a dict (not coerced to a DataFrame, which
        # was the cause of the previous return_info error).
        return {
            "infoList": infoList,
            "iter0Crn": iter0Crn,
            "ssfCrn": finalCrn,
            "sfRW_Array": sfRW_Array,                     # SF measurements
            "sfRWRescaled_Array": sfRWRescaled_Array,     # rescaled SF measurements
            "sfRWRescaledCurves_Array": sfRWRescaledCurves_Array,  # rescaled SF curves
            "sfRWI_Array": sfRWI_Array,                   # SF RWI
            "sfCrn_Mat": sfCrn_Mat,                       # SF chronologies by iteration
            "hfCrn_Mat": hfCrn_Mat,                       # high-frequency chronologies
            "hfCrnResids_Mat": hfCrnResids_Mat,           # HF chronology residuals
            "MAD_Vec": MAD_Vec,                           # median abs diff per iteration
        }
    else:
        return finalCrn
