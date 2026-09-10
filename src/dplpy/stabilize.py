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

# -*- coding: utf-8 -*-
#
# Title: stabilize.py
# Project: OpenDendro dplPy
# Description: Variance-stabilize an already-built chronology (standard,
#              residual, or ARSTAN), decoupled from chronology construction --
#              the ARSTAN workflow, where variance stabilization is a step
#              applied to a finished chronology rather than being welded to
#              building the standard mean-value chronology from the RWI matrix
#              (which is what chron_stabilized(), a port of dplR's
#              chron.stabilized(), does). See
#              dev/variance_stabilization_design_2026-09-10.md.

import numpy as np
import pandas as pd

from .rbar import running_rbar_vector, mean_series_intercorrelation, pairwise_corr_mean
from .smoothingspline import variance_stabilize_spline

_METHODS = ("rbar", "spline", "both")
_RBAR_MODES = ("running", "constant")

# keys that chron()/chron_ars() set internally when they call stabilize_chron
# on each column, so the user's stabilize_kwargs must not also set them.
_STABILIZE_MANAGED = frozenset(
    {"rwi", "column", "samp_depth", "running_rbar", "rbar", "return_info",
     "method", "chron"}
)


def _prepare_stabilize_kwargs(stabilize, stabilize_kwargs):
    """Validate the ``stabilize=`` convenience-flag inputs shared by chron() and
    chron_ars().

    Returns a dict of keyword arguments to forward to :func:`stabilize_chron`
    (empty when ``stabilize_kwargs`` is None), or ``None`` when ``stabilize`` is
    None (i.e. no stabilization requested). Raises if stabilize_kwargs is given
    without a method, or tries to set an internally-managed argument.
    """
    if stabilize is None:
        if stabilize_kwargs:
            raise ValueError(
                "stabilize_kwargs was given but stabilize is None; set stabilize "
                "to 'rbar', 'spline', or 'both'.")
        return None
    if stabilize not in _METHODS:
        raise ValueError("stabilize must be one of %s (or None); got %r"
                         % (_METHODS, stabilize))
    skw = dict(stabilize_kwargs or {})
    bad = _STABILIZE_MANAGED & set(skw)
    if bad:
        raise ValueError(
            "stabilize_kwargs may not set %s (managed internally by "
            "chron/chron_ars)." % ", ".join(sorted(bad)))
    return skw


def _restandardize(a, target_mean, target_sd):
    """Affine-rescale ``a`` so its (nan-ignoring) mean/SD match the targets. A
    single uniform transform, so it sets overall level and scale without
    reintroducing any time-varying structure."""
    a = np.asarray(a, dtype=float)
    m = np.nanmean(a)
    s = np.nanstd(a, ddof=1)
    if not np.isfinite(s) or s == 0:
        return a
    return (a - m) / s * target_sd + target_mean


def _rbar_component(x, depth, rbar_per_year):
    """Osborn (1997) effective-independent-sample-size (N_eff) scaling on the
    chronology's departures, restandardized to the input's full-period mean/SD.

    N_eff = n / (1 + (n - 1) * rbar); departures are scaled by sqrt(N_eff), which
    flattens the sample-size / rbar-driven variance trend. ``rbar_per_year`` may
    be a scalar (constant rbar) or a per-year array (running rbar). NaN-safe: NaN
    positions in ``x`` (e.g. the leading years of a residual chronology) stay NaN.
    """
    x = np.asarray(x, dtype=float)
    depth = np.asarray(depth, dtype=float)
    rbar_per_year = np.asarray(rbar_per_year, dtype=float)
    xbar = np.nanmean(x)
    xsd = np.nanstd(x, ddof=1)
    denom = 1.0 + (depth - 1.0) * rbar_per_year
    with np.errstate(invalid="ignore", divide="ignore"):
        n_eff = np.where(denom > 0, depth / denom, depth)
    n_eff = np.minimum(n_eff, depth)
    stab = (x - xbar) * np.sqrt(n_eff)
    return _restandardize(stab, xbar, xsd), n_eff


def _spline_component(x, period, f):
    """ARSTAN stabit: absolute-departure spline envelope, restandardized to the
    input's mean/SD. Fits only on the finite values (a residual chronology can
    have leading NaN), reinserting NaN where the input was NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    finite = np.isfinite(x)
    if finite.sum() < 4:                      # too few points to fit a spline
        out[finite] = x[finite]
        return out
    out[finite] = variance_stabilize_spline(x[finite], period=period, f=f,
                                             clip_negative=False, ok=None)
    return out


def stabilize_chron(chron, method, *, column="std", rwi=None, rbar=None,
                    running_rbar=None, samp_depth=None, rbar_mode="running",
                    win_length=50, min_seg_ratio=1 / 3, spline_period=None,
                    f=0.5, clip_negative=True, return_info=False):
    """Variance-stabilize an already-built chronology.

    Unlike :func:`chron_stabilized` (a port of dplR's chron.stabilized(), which
    takes the RWI matrix and builds its own standard mean-value chronology
    internally), this operates on a chronology you already built with
    :func:`chron` or :func:`chron_ars` -- standard, residual, or ARSTAN -- which
    is how ARSTAN itself is organized (variance stabilization is a post-
    construction step, menu option ``isb``).

    Parameters
    ----------
    chron : pandas.Series or pandas.DataFrame
        The chronology to stabilize. A Series is used directly; a DataFrame (e.g.
        the output of chron()/chron_ars()) uses ``column`` and, if present, its
        ``samp_depth`` column.
    method : {"rbar", "spline", "both"}
        Required, explicit (the families do different things):

        - ``"rbar"``   -- Osborn (1997) / Frank (2006) N_eff scaling; corrects
          variance driven by changing sample size (and, with running rbar,
          changing interseries correlation). Needs sample depth + rbar (see
          ``rwi`` / the precomputed inputs).
        - ``"spline"`` -- ARSTAN's ad-hoc absolute-departure spline (stabit):
          flattens all time-varying variance whatever the cause; needs only the
          chronology. Can remove real low-frequency variance (Osborn 1997).
        - ``"both"``   -- ARSTAN ``isb=2`` / Frank: ``"rbar"`` then a 67%
          ``"spline"``, in sequence.
    column : str, default "std"
        Which column to stabilize when ``chron`` is a DataFrame.
    rwi : pandas.DataFrame, optional
        The source ring-width-index matrix for the chronology (for the ``"rbar"``
        family). When given, rbar, the running rbar, and sample depth are derived
        from it. For a residual or ARSTAN chronology this should be the residual
        (prewhitened) matrix -- following ARSTAN, which uses the residual rbar for
        the ARSTAN chronology.
    rbar : float, optional
        Precomputed scalar rbar (for ``rbar_mode="constant"`` without ``rwi``).
    running_rbar : array-like, optional
        Precomputed per-year running rbar (for ``rbar_mode="running"`` without
        ``rwi``).
    samp_depth : array-like, optional
        Per-year sample depth. Taken from the DataFrame's ``samp_depth`` column or
        from ``rwi`` when not given.
    rbar_mode : {"running", "constant"}, default "running"
        Running rbar (Frank RUNNINGr; the widely used default) or a single
        time-constant rbar (Osborn/ARSTAN Briffa MEANr).
    win_length, min_seg_ratio : int, float
        Running-rbar window length and minimum overlap ratio (as in
        chron_stabilized).
    spline_period : int, float or None, default None
        Spline stiffness for the ``"spline"``/``"both"`` families. ``None`` = the
        67% spline (``floor(0.67*n)`` years, ARSTAN's default for the combined
        method); an int > 1 is a fixed wavelength in years; a float in (0, 1) is a
        fraction of length; a negative value is a percent spline.
    f : float, default 0.5
        Spline frequency response (50% amplitude cutoff).
    clip_negative : bool, default True
        Clamp the stabilized chronology at >= 0 (ARSTAN behavior).
    return_info : bool, default False
        Also return a dict of diagnostics (rbar used, N_eff, sample depth).

    Returns
    -------
    pandas.DataFrame
        Indexed by year, with the stabilized chronology in column ``vsc`` and, when
        known, ``samp_depth``. If ``return_info`` is True, returns
        ``(DataFrame, info_dict)``.

    Notes
    -----
    Both families restandardize to the input chronology's **full-period** mean and
    SD. This differs from ARSTAN's ``stabbm``, which rescales to a user-chosen
    reference period; the full-period convention keeps the stabilized chronology on
    the same overall scale as the input while removing the time-varying variance.

    For the **ARSTAN** chronology, ARSTAN stabilizes the residual chronology and
    *then* re-reddens, so faithful ARSTAN stabilization of the ARSTAN chronology is
    produced by ``chron_ars(stabilize=...)`` (which does it in that order), not by
    calling this function on an already re-reddened series.

    References
    ----------
    Osborn, Briffa & Jones (1997) Dendrochronologia 15, 89-99.
    Frank, Esper & Cook (2006) TRACE 4, 56-66.
    """
    if method not in _METHODS:
        raise ValueError("method must be one of %s; got %r" % (_METHODS, method))
    if rbar_mode not in _RBAR_MODES:
        raise ValueError("rbar_mode must be 'running' or 'constant'; got %r" % (rbar_mode,))

    # resolve the chronology series (+ sample depth) from Series or DataFrame
    if isinstance(chron, pd.DataFrame):
        if column not in chron.columns:
            raise ValueError("column %r not found in chron; available: %s"
                             % (column, list(chron.columns)))
        series = chron[column]
        if samp_depth is None and "samp_depth" in chron.columns:
            samp_depth = chron["samp_depth"].to_numpy()
    elif isinstance(chron, pd.Series):
        series = chron
    else:
        raise TypeError("chron must be a pandas Series or DataFrame; got %s"
                        % type(chron).__name__)

    index = series.index
    x = series.to_numpy(dtype=float)
    n = len(x)
    info = {"method": method}

    def _resolve_rbar():
        """Return (depth, rbar_per_year) for the rbar family, from rwi or the
        precomputed inputs."""
        depth = None if samp_depth is None else np.asarray(samp_depth, dtype=float)
        if rwi is not None:
            rwi_a = rwi.reindex(index) if isinstance(rwi, pd.DataFrame) else rwi
            if depth is None:
                depth = rwi_a.notnull().sum(axis=1).to_numpy().astype(float)
            if rbar_mode == "running":
                # correlations are location-invariant, so the running rbar can be
                # computed on the matrix as-is (no need to zero-mean first).
                rby = running_rbar_vector(rwi_a, win_length, min_seg_ratio, depth)
            else:
                rby = np.full(n, pairwise_corr_mean(rwi_a, "pearson",
                                                    min_overlap=20, strict=True))
            return depth, rby
        # no matrix: use precomputed inputs
        if depth is None:
            raise ValueError("the 'rbar' method needs sample depth: pass rwi=, "
                             "samp_depth=, or a chron DataFrame with 'samp_depth'.")
        if rbar_mode == "running":
            if running_rbar is None:
                raise ValueError("rbar_mode='running' without rwi requires running_rbar=.")
            rby = np.asarray(running_rbar, dtype=float)
        else:
            if rbar is None:
                raise ValueError("rbar_mode='constant' without rwi requires rbar=.")
            rby = np.full(n, float(rbar))
        return depth, rby

    if method in ("rbar", "both"):
        depth, rbar_year = _resolve_rbar()
        result, n_eff = _rbar_component(x, depth, rbar_year)
        info["rbar"] = rbar_year
        info["n_eff"] = n_eff
        info["samp_depth"] = depth
        if method == "both":
            result = _spline_component(result, spline_period, f)
    else:  # spline only
        result = _spline_component(x, spline_period, f)
        if samp_depth is not None:
            info["samp_depth"] = np.asarray(samp_depth, dtype=float)

    if clip_negative:
        with np.errstate(invalid="ignore"):
            result = np.where(result < 0, 0.0, result)

    out = {"vsc": result}
    if samp_depth is not None:
        out["samp_depth"] = np.asarray(samp_depth)
    stabilized = pd.DataFrame(out, index=index)

    if return_info:
        return stabilized, info
    return stabilized
