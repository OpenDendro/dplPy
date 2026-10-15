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

# -*- coding: utf-8 -*-
#
# Title: chron_stabilized.py
# Project: OpenDendro dplPy
# Description: Variance-stabilized mean-value chronology built straight from
#              the RWI matrix -- a port of dplR's chron.stabilized() (running
#              rbar), plus the constant-rbar (Briffa/Osborn) and ARSTAN spline
#              variants. To stabilize a chronology you already built, see
#              stabilize_chron().

import numpy as np
import pandas as pd

from ._validate import (_require_dataframe, _alias_kwarg, _reject_kwargs,
                        _normalize_choice, _carry_attrs, _warn)
from .rbar import running_rbar_vector, pairwise_corr_mean
from .chron import chron
from .stabilize import _spline_component

_METHODS = ("running_rbar", "briffa", "mean_rbar", "spline")


def chron_stabilized(rwi_data: pd.DataFrame, window_length=50, min_seg_ratio=1/3,
                     biweight=True, running_rbar=False, method="running_rbar",
                     spline_period=None, **kwargs):
    """Build a variance-stabilized mean-value chronology.

    Scales a mean-value chronology by the square root of the effective independent
    sample size Neff = n(t) / (1 + (n(t)-1) rbar(t)), where n(t) is the sample depth
    and rbar(t) the mean pairwise interseries correlation, computed over windows of
    length ``window_length``. Port of dplR's chron.stabilized().

    Parameters
    ----------
    rwi_data : pd.DataFrame
        detrended ring-width indices.
    window_length : int, default 50
        window length over which rbar values are calculated. (``win_length=`` is
        accepted as a deprecated alias.)
    min_seg_ratio : float, default 1/3
        minimum ratio of non-NA values to window length for a series to count in an
        Neff calculation (1/3 matches dplR's chron.stabilized()).
    biweight : boolean, default True
        average with Tukey's biweight robust mean if True, else the arithmetic mean.
    running_rbar : boolean, default False
        also return the running rbar values in the output (only for
        method='running_rbar').
    method : str, default 'running_rbar'
        the variance-stabilization method (case-insensitive):

        - 'running_rbar' : running-window-rbar effective-signal adjustment (Frank et
          al. 2006 "RUNNINGr"); the default. Uses window_length and min_seg_ratio.
        - 'briffa' (alias 'mean_rbar') : the same adjustment with a single
          time-constant rbar over all pairwise overlaps longer than 20 years (Osborn
          et al. 1997; Frank et al. 2006 "MEANr"). Corrects for changing sample size
          only.
        - 'spline' : ARSTAN's ad-hoc spline stabilization, removing any time trend in
          the absolute departures with a smoothing spline. Can remove real
          low-frequency variance -- use with care. Stiffness set by spline_period.
    spline_period : int, float or None, default None
        stiffness for method='spline'. An int > 1 is a fixed spline wavelength in
        years; a float in (0, 1) is a fraction of the chronology length; None uses
        the 67% spline (``floor(0.67*n)`` years, as stabilize_chron and dplR's
        detrend default; earlier releases used 0.5n). Ignored by the rbar methods.
        (``spline_nyrs=`` is accepted as a deprecated alias.)

    Returns
    -------
    stabilized_chron : pandas dataframe
        a variance-stabilized mean-value chronology indexed by year, with columns
        ``vsc``, ``running_rbar`` (only when running_rbar=True), and ``samp_depth``.
        Years with no series present have ``samp_depth`` 0 and NaN ``vsc``.

    Notes
    -----
    This function scales the departures and adds the grand mean back, as dplR
    does; `stabilize_chron()` instead restandardizes to the input chronology's
    full-period mean and SD, so the two differ slightly in level and scale.

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.chron_stabilized(data, window_length=60, min_seg_ratio=0.4)

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/chron.stabilized.html
    .. [2] https://opendendro.github.io/dplPy/reference/chronology/#chron_stabilized
    """
    _require_dataframe(rwi_data)
    window_length = _alias_kwarg(kwargs, "win_length", "window_length",
                                 window_length, "chron_stabilized")
    spline_period = _alias_kwarg(kwargs, "spline_nyrs", "spline_period",
                                 spline_period, "chron_stabilized")
    _reject_kwargs(kwargs)
    method = _normalize_choice(method, _METHODS, "method")
    if method == "mean_rbar":
        method = "briffa"

    num_years = rwi_data.shape[0]

    if window_length > num_years:
        raise ValueError("Window length should not be greater than the number of rows in the dataset")

    if min_seg_ratio <= 0 or min_seg_ratio > 1:
        raise ValueError("min_seg_ratio cannot be <= 0 or > 1")

    # window_length only drives the running-window rbar; warn about it only there.
    # Matches dplR's chron.stabilized(): these are two independent checks (an
    # absolute floor and a relative ceiling), not one combined 30-50% band --
    # either, both, or neither can fire depending on window_length and num_years.
    if method == "running_rbar":
        if window_length <= 30:
            _warn("Window length less than 30 is not recommended. Consider a longer window.\n")
        if window_length / num_years > 0.5:
            _warn("Window length greater than 50% of the chronology length is not recommended. Consider a shorter window.\n")

    # Center on the mean of the per-year (cross-series) means, matching dplR's
    # mean(rowMeans(x)). This is NOT the same as the mean of the per-series
    # means whenever series have staggered start/end years (the normal case
    # for ring-width data) -- and since this constant is multiplied through by
    # the stabilization factor and added back at the end, getting it right
    # matters for the final values, not just as an intermediate detail.
    mean_val = rwi_data.mean(axis=1).mean()

    zero_mean_data = rwi_data - mean_val

    # Sample depth per year (dplR's nSamps <- rowSums(!is.na(x0))). chron() keeps
    # zero-depth years (std NaN, depth 0), so its output is always the same
    # length as this vector.
    n_samps = zero_mean_data.notna().sum(axis=1).to_numpy()

    reg_chron = chron(zero_mean_data, biweight=biweight, plot=False)
    mean_rwis = reg_chron["std"].to_numpy()

    rbar_array = None
    if method == "running_rbar":
        # RUNNINGr: moving-window rbar (Frank et al. 2006), a port of dplR's
        # chron.stabilized(). rbar varies through time.
        rbar_array = running_rbar_vector(zero_mean_data, window_length,
                                         min_seg_ratio, n_samps)
        n_eff = _n_eff(n_samps, rbar_array)
        # The overall rbar constant is deliberately unfiltered by overlap, unlike
        # the moving-window rbar above -- dplR computes this one as a plain
        # pairwise-complete correlation mean over the whole record, with no
        # minimum-overlap filtering between series pairs.
        rbar_const = pairwise_corr_mean(zero_mean_data, "pearson")
        vsc = mean_rwis * np.sqrt(n_eff * rbar_const) + mean_val

    elif method == "briffa":
        # MEANr / Osborn constant-rbar / ARSTAN Briffa: a single rbar over all
        # pairwise overlaps longer than 20 years; sample-size correction only.
        rbar_b = pairwise_corr_mean(zero_mean_data, "pearson", min_overlap=20,
                                    strict=True)
        n_eff = _n_eff(n_samps, rbar_b)
        vsc = mean_rwis * np.sqrt(n_eff * rbar_b) + mean_val

    else:  # method == "spline"
        # ARSTAN's ad-hoc spline variance stabilization: flatten the time trend
        # in the chronology's absolute departures with a smoothing spline.
        vsc = _spline_stabilize(mean_rwis + mean_val, n_samps, spline_period)

    out = {"vsc": vsc, "samp_depth": n_samps}
    if method == "running_rbar" and running_rbar:
        out = {"vsc": vsc, "running_rbar": rbar_array, "samp_depth": n_samps}
    stabilized_chron = pd.DataFrame(data=out, index=reg_chron.index)
    return _carry_attrs(rwi_data, stabilized_chron, kind="crn")


def _n_eff(n_samps, rbar):
    """Effective independent sample size n / (1 + (n-1) rbar), capped at n.
    Where the (running) rbar is NaN, dplR falls back to the sample depth itself
    (audit A13), consistent with stabilize._rbar_component."""
    n_samps = np.asarray(n_samps, dtype=float)
    denom = 1.0 + (n_samps - 1.0) * np.asarray(rbar, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isnan(denom), n_samps, np.minimum(n_samps / denom, n_samps))


def _spline_stabilize(chron_series, n_samps, spline_period=None, **kwargs):
    """ARSTAN's ad-hoc spline variance stabilization of a chronology, resolving
    the stiffness convention (int > 1: years; 0 < float < 1: fraction of length;
    None: the 67% spline), clipping negatives, and taking the mean/SD only over
    years with sample depth > 0. Thin wrapper over `stabilize._spline_component()`.
    """
    spline_period = _alias_kwarg(kwargs, "spline_nyrs", "spline_period", spline_period,
                                 "_spline_stabilize")
    _reject_kwargs(kwargs)
    n = len(chron_series)
    if spline_period is None:
        nyrs = max(int(np.floor(0.67 * n)), 2)
    elif isinstance(spline_period, float) and 0 < spline_period < 1:
        nyrs = max(int(round(spline_period * n)), 2)
    else:
        nyrs = max(int(spline_period), 2)
    return _spline_component(chron_series, nyrs, 0.5, clip_negative=True,
                             ok=(np.asarray(n_samps) > 0))
