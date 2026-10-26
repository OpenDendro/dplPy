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

# Title: moving_corr.py
# Project: OpenDendro dplPy
# Description: Moving (windowed) correlation between a tree-ring chronology and a
#              monthly or seasonal climate variable, for evaluating the temporal
#              stability of the climate signal -- with deliberate statistical care
#              about what an unstable-looking trace means.
#
#              The running trace is the easy part. The judgement is the point: a
#              moving correlation computed on overlapping windows wobbles on its
#              own even when the true relationship is perfectly constant (the
#              Slutsky-Yule effect), so the apparent "instability" has to be
#              compared against a null that wobbles the same way. The significance
#              engine here is ported, nearly verbatim, from K. Anchukaitis's
#              `moving_corr_sig` package (validated against Gershunov et al. 2001,
#              Table 1): red-noise AR(1) surrogates matched to BOTH series' lag-1
#              persistence and their background correlation, and three *global*
#              Monte-Carlo tests (std / peak / range) that each return one verdict
#              about the whole trace rather than a significant/not-significant flag
#              per window. The plot shows the estimate with its uncertainty and a
#              single multiplicity-aware threshold line, following Gelman & Stern
#              (2006): the difference between "significant" and "not significant"
#              is not itself significant, so dplPy does not shade per-window cells.
#
#              This is the fourth function in dplPy's climate-response module (after
#              seascorr, pcreg and sea); it reuses seascorr's seasonalization so the
#              month/season selection matches dpl.seascorr.
#
#              References:
#                Gershunov, A., Schneider, N., Barnett, T. (2001). Low-frequency
#                  modulation of the ENSO-Indian monsoon rainfall relationship:
#                  signal or noise? Journal of Climate 14, 2486-2492.
#                Gelman, A., Stern, H. (2006). The difference between
#                  "significant" and "not significant" is not itself statistically
#                  significant. The American Statistician 60(4), 328-331.
#                Zang, C., Biondi, F. (2014). treeclim: an R package for the
#                  numerical calibration of proxy-climate relationships. Ecography
#                  38(4), 431-436.
#                Biondi, F., Waikul, K. (2004) / Meko sliding-correlation practice
#                  for the windowing and effective-N conventions.

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.signal import lfilter

# Rejection-sampling tolerance / retry budget for surrogate generation (as in
# moving_corr_sig; nothing has needed to override these).
_TOL = 0.01
_MAX_TRIES = 200

# default null-distribution quantiles reported per test
_STD_LEVELS = (0.01, 0.05, 0.10, 0.90, 0.95, 0.99)
_PEAK_LEVELS = (0.90, 0.95, 0.99)


# --------------------------------------------------------------------------- #
# Significance engine (ported from moving_corr_sig, with offset support)
# --------------------------------------------------------------------------- #

def _as_1d_float(a, name="array"):
    a = np.asarray(a, dtype=float).squeeze()
    if a.ndim != 1:
        raise ValueError("%s must be 1-dimensional, got shape %s" % (name, a.shape))
    return a


def _moving_correlation(x, y, window, method="pearson", offset=1):
    """Vectorized moving (sliding-window) correlation between two equal-length
    series, subsampled every ``offset`` windows.

    ``r[k]`` is the correlation of the window starting at sample ``k*offset``;
    with ``offset=1`` every window is returned (a fully overlapping, smooth
    trace), with ``offset=window`` the windows are adjacent and non-overlapping
    (independent), and anything between trades smoothness for independence. The
    same ``offset`` is applied to the observed trace and to every Monte-Carlo
    surrogate, so the significance null carries the identical overlap structure.
    """
    x = _as_1d_float(x, "x")
    y = _as_1d_float(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must be the same length (got %s vs %s)"
                         % (x.shape, y.shape))
    n = x.shape[0]
    if window < 2:
        raise ValueError("window must be >= 2")
    if window >= n:
        raise ValueError("window (%d) must be shorter than the series length (%d)"
                         % (window, n))

    Wx = sliding_window_view(x, window)        # (n-window+1, window)
    Wy = sliding_window_view(y, window)

    if method == "spearman":
        Wx = stats.rankdata(Wx, axis=1)
        Wy = stats.rankdata(Wy, axis=1)
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")

    dx = Wx - Wx.mean(axis=1, keepdims=True)
    dy = Wy - Wy.mean(axis=1, keepdims=True)
    cov = (dx * dy).sum(axis=1)
    sx = np.sqrt((dx ** 2).sum(axis=1))
    sy = np.sqrt((dy ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / (sx * sy)

    if offset != 1:
        r = r[::offset]
    return r


def _ar1_fit(x):
    """Lag-1 AR coefficient via statsmodels AutoReg (least squares), clipped to
    (-0.98, 0.98) for stable downstream simulation. Mirrors moving_corr_sig."""
    from statsmodels.tsa.ar_model import AutoReg

    x = _as_1d_float(x, "x")
    res = AutoReg(x, lags=1, trend="c").fit()
    # index positionally: with lags=1, trend='c' the params are
    # [intercept, ar_lag1]; the lag-1 coefficient is last.
    return float(np.clip(res.params[-1], -0.98, 0.98))


def _white_noise_pair(n, target_corr, rng, tol=_TOL, max_tries=_MAX_TRIES):
    """Pair of unit-variance Gaussian white-noise series whose realized Pearson
    correlation is within ``tol`` of ``target_corr`` (rejection-sampled)."""
    target_corr = float(np.clip(target_corr, -0.999, 0.999))
    cov = np.array([[1.0, target_corr], [target_corr, 1.0]])
    x = y = None
    for _ in range(max_tries):
        X = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
        x, y = X[:, 0], X[:, 1]
        if abs(np.corrcoef(x, y)[0, 1] - target_corr) <= tol:
            return x, y
    return x, y


def _red_noise_pair(n, target_corr, phi_x, phi_y, rng, tol=_TOL, max_tries=_MAX_TRIES,
                    burn=500):
    """Pair of AR(1) ("red noise") series with lag-1 autocorrelations phi_x, phi_y
    whose realized overall Pearson correlation is within ``tol`` of ``target_corr``.

    The processes are driven by contemporaneously-correlated innovations; the
    stationary correlation relates to the innovation correlation rho_e in closed
    form, which is inverted to target the desired corr(x, y) in expectation and
    then rejection-sampled for the finite-sample deviation. Mirrors
    moving_corr_sig."""
    target_corr = float(np.clip(target_corr, -0.999, 0.999))
    phi_x = float(np.clip(phi_x, -0.98, 0.98))
    phi_y = float(np.clip(phi_y, -0.98, 0.98))
    denom = np.sqrt((1 - phi_x ** 2) * (1 - phi_y ** 2))
    if denom <= 1e-8:
        raise ValueError("phi_x/phi_y too close to +/-1 for stable red-noise simulation")
    rho_e = float(np.clip(target_corr * (1 - phi_x * phi_y) / denom, -0.999, 0.999))
    cov_e = np.array([[1.0, rho_e], [rho_e, 1.0]])
    ntot = n + burn
    x = y = None
    for _ in range(max_tries):
        E = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_e, size=ntot)
        xf = lfilter([1.0], [1.0, -phi_x], E[:, 0])
        yf = lfilter([1.0], [1.0, -phi_y], E[:, 1])
        x, y = xf[burn:], yf[burn:]
        if abs(np.corrcoef(x, y)[0, 1] - target_corr) <= tol:
            return x, y
    return x, y


def _mc_ensemble(n, window, target_corr, iters, noise, method, rng, offset,
                 phi_x=None, phi_y=None):
    """Run ``iters`` surrogate draws at ``target_corr``, compute each surrogate's
    moving-correlation trace (at the given offset) and return arrays of the trace
    std / max / min / range across draws -- one shared ensemble for all tests."""
    if noise not in ("white", "red"):
        raise ValueError("noise must be 'white' or 'red'")
    if noise == "red" and (phi_x is None or phi_y is None):
        raise ValueError("phi_x and phi_y are required when noise='red'")
    stds = np.empty(iters)
    maxs = np.empty(iters)
    mins = np.empty(iters)
    for i in range(iters):
        if noise == "white":
            sx, sy = _white_noise_pair(n, target_corr, rng)
        else:
            sx, sy = _red_noise_pair(n, target_corr, phi_x, phi_y, rng)
        r = _moving_correlation(sx, sy, window, method=method, offset=offset)
        stds[i] = np.nanstd(r)
        maxs[i] = np.nanmax(r)
        mins[i] = np.nanmin(r)
    return {"std": stds, "max": maxs, "min": mins, "range": maxs - mins}


@dataclass
class _TestResult:
    """One global significance test on a moving-correlation trace (std / peak /
    range). The null distribution is Monte-Carlo and the p-value is the tail
    fraction of the null at or beyond the observed statistic."""
    name: str
    observed: float
    null_distribution: np.ndarray
    p_value: float
    p_value_lower: Optional[float] = None
    quantiles: dict = field(default_factory=dict)
    noise: str = "red"
    target_corr: float = 0.0
    iters: int = 0
    window: int = 0
    null_min: Optional[np.ndarray] = None   # peak test only: the trough (min) null

    def threshold(self, level):
        """Null-distribution quantile at an arbitrary confidence level, computed
        on demand. For the peak test this is the flat *upper* significance line to
        draw across the trace: because the null is the maximum over each surrogate
        trace, one level already accounts for having searched the whole record."""
        return float(np.quantile(self.null_distribution, level))

    def threshold_lower(self, level):
        """The flat *lower* (trough) significance line for the peak test: the
        (1-level) quantile of the null distribution of each surrogate trace's
        *minimum* correlation. The correct counterpart to :meth:`threshold` for
        reading significant negative excursions -- not simply the negative of the
        upper line, since the null is centred on the background correlation."""
        if self.null_min is None:
            raise ValueError("no trough null available (peak test only).")
        return float(np.quantile(self.null_min, 1.0 - level))

    def __repr__(self):
        s = "_TestResult(%r, observed=%.4f, p=%.4f" % (self.name, self.observed,
                                                        self.p_value)
        if self.p_value_lower is not None:
            s += ", p_lower=%.4f" % self.p_value_lower
        return s + ", noise=%r, c=%.3f, iters=%d)" % (self.noise, self.target_corr,
                                                      self.iters)


def _assemble_tests(r_obs, ens, ens_zero, tests, noise, target_corr, iters, window,
                    peak_condition):
    """Build the requested _TestResult objects from one (or two) shared ensembles.

    std / range / peak(observed) all draw from ``ens`` (surrogates at the observed
    background correlation); peak(zero) draws from ``ens_zero`` (unrelated
    surrogates). Sharing one ensemble makes the tests mutually consistent and
    avoids recomputing the Monte Carlo for each."""
    out = {}
    if "std" in tests:
        obs = float(np.nanstd(r_obs))
        null = ens["std"]
        out["std"] = _TestResult(
            "Gershunov std test", obs, null,
            p_value=float(np.mean(null >= obs)),
            p_value_lower=float(np.mean(null <= obs)),
            quantiles={_q(q): float(np.quantile(null, q)) for q in _STD_LEVELS},
            noise=noise, target_corr=target_corr, iters=iters, window=window)
    if "range" in tests:
        obs = float(np.nanmax(r_obs) - np.nanmin(r_obs))
        null = ens["range"]
        out["range"] = _TestResult(
            "High-low range test", obs, null,
            p_value=float(np.mean(null >= obs)),
            quantiles={_q(q): float(np.quantile(null, q)) for q in _PEAK_LEVELS},
            noise=noise, target_corr=target_corr, iters=iters, window=window)
    if "peak" in tests:
        obs = float(np.nanmax(r_obs))
        src = ens_zero if peak_condition == "zero" else ens
        null = src["max"]
        tc = 0.0 if peak_condition == "zero" else target_corr
        out["peak"] = _TestResult(
            "Peak correlation test (condition=%s)" % peak_condition, obs, null,
            p_value=float(np.mean(null >= obs)),
            quantiles={_q(q): float(np.quantile(null, q)) for q in _PEAK_LEVELS},
            noise=noise, target_corr=tc, iters=iters, window=window,
            null_min=src["min"])
    return out


def _q(level):
    return "%g" % round(level * 100, 4)


# --------------------------------------------------------------------------- #
# Per-window confidence band (the Gelman-Stern "estimate + uncertainty" display)
# --------------------------------------------------------------------------- #

def _fisher_ci(r, n_eff, level):
    """Per-window Fisher-z confidence interval for a correlation, at an
    autocorrelation-adjusted effective sample size ``n_eff`` (scalar or array).

    z = atanh(r), SE = 1/sqrt(n_eff - 3), interval tanh(z +/- z_crit*SE). Short
    windows give a large SE and therefore a wide band -- which is exactly the
    point: a single short-window correlation is not precisely estimated, and the
    band is meant to keep the reader from over-reading one window."""
    from scipy.stats import norm
    r = np.asarray(r, dtype=float)
    n_eff = np.asarray(n_eff, dtype=float)
    zc = float(norm.ppf(0.5 + level / 2.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = np.clip(r, -0.999999, 0.999999)
        z = np.arctanh(rc)
        se = 1.0 / np.sqrt(np.maximum(n_eff - 3.0, 1e-6))
        lo = np.tanh(z - zc * se)
        hi = np.tanh(z + zc * se)
    bad = ~np.isfinite(n_eff) | (n_eff <= 4.0)
    lo = np.where(bad, np.nan, lo)
    hi = np.where(bad, np.nan, hi)
    return lo, hi


def _block_bootstrap_ci(x, y, window, starts, level, n_boot, rng, block=None):
    """Per-window confidence interval by a circular moving-block bootstrap of the
    paired samples inside each window. Heavier and stochastic, but assumption-light
    about the within-window distribution; offered as an alternative to Fisher-z."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if block is None:
        block = max(2, int(round(window ** (1.0 / 3.0))))   # ~ n^(1/3) blocks
    nstart = len(starts)
    lo = np.full(nstart, np.nan)
    hi = np.full(nstart, np.nan)
    alpha = (1.0 - level) / 2.0
    nblocks = int(np.ceil(window / block))
    for j, s in enumerate(starts):
        xs = x[s:s + window]
        ys = y[s:s + window]
        rb = np.empty(n_boot)
        for b in range(n_boot):
            offs = rng.integers(0, window, size=nblocks)
            idx = (offs[:, None] + np.arange(block)[None, :]).ravel() % window
            idx = idx[:window]
            xb, yb = xs[idx], ys[idx]
            if xb.std() == 0 or yb.std() == 0:
                rb[b] = np.nan
            else:
                rb[b] = np.corrcoef(xb, yb)[0, 1]
        lo[j] = np.nanquantile(rb, alpha)
        hi[j] = np.nanquantile(rb, 1.0 - alpha)
    return lo, hi


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #

class MovingCorrResult:
    """Result of :func:`moving_corr`.

    Attributes
    ----------
    trace : pandas.DataFrame
        One row per window, indexed by the window's position year (per
        ``position``): columns ``start``, ``end``, ``r``, and -- when a band was
        computed -- ``ci_lower`` / ``ci_upper``.
    tests : dict
        The requested global tests, keyed ``"std"`` / ``"peak"`` / ``"range"``,
        each a result object with ``observed``, ``p_value`` (and ``p_value_lower``
        for the two-tailed std test), ``quantiles`` and ``threshold(level)``.
    overall_corr : float
        Full-period correlation between the aligned chronology and climate series.
    r1_chronology, r1_climate : float
        Lag-1 autocorrelation of each aligned series (drives the red-noise null
        and the effective-N band).
    period : tuple(int, int)
        First and last year of the aligned analysis window.
    label : str
        The season label (e.g. ``"Jun-Aug"``) of the climate variable.
    settings : dict
        The call parameters actually used.
    """

    def __init__(self, trace, tests, overall_corr, r1_chronology, r1_climate,
                 period, label, settings):
        self.trace = trace
        self.tests = tests
        self.overall_corr = overall_corr
        self.r1_chronology = r1_chronology
        self.r1_climate = r1_climate
        self.period = period
        self.label = label
        self.settings = settings
        self.fig = None
        self.axes = None

    def threshold(self, level=0.95):
        """The peak-test upper significance line at ``level`` (the flat,
        multiplicity-aware threshold to read the trace against). Requires the peak
        test to have been run."""
        if "peak" not in self.tests:
            raise ValueError("no peak test was run; include 'peak' in tests=")
        return self.tests["peak"].threshold(level)

    def threshold_lower(self, level=0.95):
        """The peak-test lower (trough) significance line at ``level``: the
        multiplicity-aware threshold for significant negative excursions. Requires
        the peak test to have been run."""
        if "peak" not in self.tests:
            raise ValueError("no peak test was run; include 'peak' in tests=")
        return self.tests["peak"].threshold_lower(level)

    def __repr__(self):
        a, b = self.period
        verdicts = ", ".join("%s p=%.3f" % (k, v.p_value)
                             for k, v in self.tests.items())
        return ("MovingCorrResult(%s, %d-%d, window=%d offset=%d, overall r=%.3f; %s)"
                % (self.label, a, b, self.settings["window"],
                   self.settings["offset"], self.overall_corr, verdicts))

    def summary(self):
        a, b = self.period
        lines = ["Moving correlation: chronology vs %s" % self.label,
                 "  period           : %d-%d (%d years)" % (a, b, b - a + 1),
                 "  window / offset  : %d / %d" % (self.settings["window"],
                                                   self.settings["offset"]),
                 "  overall corr     : %.3f" % self.overall_corr,
                 "  lag-1 (chron/clim): %.3f / %.3f" % (self.r1_chronology,
                                                        self.r1_climate),
                 "  null model       : %s" % self.settings["noise"]]
        for k, t in self.tests.items():
            tail = (", p_lower=%.3f" % t.p_value_lower
                    if t.p_value_lower is not None else "")
            lines.append("  %-6s test     : observed=%.3f, p=%.3f%s"
                         % (k, t.observed, t.p_value, tail))
        return "\n".join(lines)

    def plot(self, **kwargs):
        from .moving_corr_plot import moving_corr_plot
        return moving_corr_plot(self, **kwargs)


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #

def _positions(years, window, starts, position):
    """Map each window (given by its start index into the aligned years) to a
    representative calendar year for plotting/indexing."""
    years = np.asarray(years)
    if position == "start":
        return years[starts]
    if position == "end":
        return years[starts + window - 1]
    if position == "center":
        return years[starts + (window - 1) // 2]
    raise ValueError("position must be 'center', 'end' or 'start'")


def moving_corr(crn, climate, end_month=9, season_length=1, season_lengths=None,
                agg="mean", window=30, offset=1, position="center",
                method="pearson", noise="red", tests=("std", "peak", "range"),
                peak_condition="observed", ci="fisher", ci_level=0.95,
                n_boot=1000, iters=1000, lc_iters=500, seed=None, period=None,
                column=None, plot=False):
    """Moving (windowed) correlation of a tree-ring chronology with a seasonal
    climate variable, with global significance tests for the stability of the
    relationship.

    The chronology is correlated with one seasonal climate series in a sliding
    window stepped by ``offset`` years, and the resulting trace is judged by
    Monte-Carlo tests that account for the dependence overlapping windows induce
    (Slutsky-Yule) and for having searched the whole record (multiplicity). See
    the guide for how to read the output; in short, a wobbling trace is not by
    itself evidence that the climate signal is unstable.

    Parameters
    ----------
    crn : Series or DataFrame
        Year-indexed chronology (standard or residual). If a DataFrame with
        several columns, pass ``column=`` to choose the index series.
    climate : DataFrame or Series
        One monthly climate variable, in any shape ``seascorr`` accepts: wide
        (year index/column + 12 month columns), tidy (year, month, value), or a
        monthly datetime/period-indexed series.
    end_month : int
        Ending month of the season (1=Jan .. 12=Dec), as in ``seascorr``.
    season_length : int
        Season length in months ending at ``end_month`` (1 = that month alone,
        3 = the three months ending at it, etc.). Used for the single-trace mode.
    season_lengths : sequence of int or None
        If given (e.g. ``(1, 3, 6, 12)``), switches to **grid mode**: a moving
        correlation for every season of these lengths across the 14 ending months
        up to ``end_month`` (the seascorr grid), returning a
        :class:`MovingCorrGridResult` with the three grid views and field
        significance. ``season_length`` is ignored in this mode.
    agg : {"mean", "sum"}
        How to aggregate the months of the season: ``"mean"`` for temperature-like
        variables, ``"sum"`` for precipitation totals.
    window : int
        Window width in years. Must be >= 10 and shorter than the aligned period.
    offset : int
        Step between successive windows, in years. ``1`` gives a fully overlapping,
        smooth trace; ``window`` gives adjacent, non-overlapping (independent)
        windows; values between trade smoothness for independence. The null is
        computed at the same offset, so the test stays fair at any overlap.
    position : {"center", "end", "start"}
        Which year of each window to plot/index the correlation at.
    method : {"pearson", "spearman"}
        Correlation type (computed identically for the observed trace and the null).
    noise : {"red", "white"}
        Surrogate null model. ``"red"`` (default) matches both series' lag-1
        persistence -- the appropriate null for persistent climate/tree-ring
        series; ``"white"`` is the classical Gershunov null.
    tests : sequence of {"std", "peak", "range"}
        Which global tests to run. ``std`` (Gershunov) asks whether the trace is
        more/less variable than the null; ``peak`` whether the single strongest
        window is real; ``range`` whether the high-to-low swing is real.
    peak_condition : {"observed", "zero"}
        For the peak test: ``"observed"`` tests elevation above the background
        correlation; ``"zero"`` tests existence of any relationship.
    ci : {"fisher", "bootstrap", "none"}
        Per-window uncertainty band. ``"fisher"`` (default) is an analytic Fisher-z
        interval at an autocorrelation-adjusted effective sample size; ``"bootstrap"``
        is a moving-block bootstrap; ``"none"`` omits the band.
    ci_level : float
        Confidence level for the per-window band (e.g. 0.95).
    n_boot : int
        Bootstrap resamples per window when ``ci="bootstrap"``.
    iters : int
        Monte-Carlo surrogate draws for the significance tests.
    seed : int or None
        Seed for the random generator (reproducible significance and bootstrap).
    period : None or (first, last)
        Restrict the analysis to these years before aligning and windowing.
    column : str or None
        Chronology column to use when ``crn`` is a multi-column DataFrame.
    plot : bool
        Draw the trace-with-uncertainty figure.

    Returns
    -------
    MovingCorrResult
    """
    # seasonalization + input coercion reuse seascorr's machinery
    from .climate import (_to_month3, _seasonalize, _as_chronology,
                          _normalize_agg, _lag1, _eff_n)

    agg = _normalize_agg(agg, "agg")
    end_month = int(end_month)
    if end_month not in range(1, 13):
        raise ValueError("end_month must be 1..12, got %r." % end_month)
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'.")
    if noise not in ("red", "white"):
        raise ValueError("noise must be 'red' or 'white'.")
    if ci not in ("fisher", "bootstrap", "none"):
        raise ValueError("ci must be 'fisher', 'bootstrap' or 'none'.")

    x = _as_chronology(crn, column)
    c3 = _to_month3(climate, "climate")

    # grid mode: a full seascorr-style season grid when season_lengths is given
    if season_lengths is not None:
        return _moving_corr_grid(
            x, c3, end_month, season_lengths, agg, window, offset, position,
            method, noise, ci, ci_level, iters, lc_iters, seed, period, plot)

    # ---- single-trace path ----
    season_length = int(season_length)
    if season_length < 1:
        raise ValueError("season_length must be >= 1, got %r." % season_length)
    if peak_condition not in ("observed", "zero"):
        raise ValueError("peak_condition must be 'observed' or 'zero'.")
    tests = tuple(t.lower() for t in tests)
    for t in tests:
        if t not in ("std", "peak", "range"):
            raise ValueError("unknown test %r (use std/peak/range)." % t)

    # seasonalize the climate variable; pick the single season (length,
    # end_month) ending in the current growth year
    frame, meta = _seasonalize(c3, end_month, (season_length,), agg)
    # the newest ending month (p = N-1) is the current-year season ending at
    # end_month; its column key is (season_length, N-1)
    ncols = len(meta)
    target_key = None
    for key, m in zip(frame.columns, meta):
        if (not m["prior"]) and m["end_month"] == end_month and \
                m["season_length"] == season_length:
            target_key = key
            label = m["label"]
            break
    if target_key is None:                       # pragma: no cover - defensive
        raise ValueError("could not form season length=%d ending month=%d."
                         % (season_length, end_month))
    clim = frame[target_key].dropna()
    clim.index = clim.index.astype(int)

    # align chronology and climate on common years
    common = x.index.intersection(clim.index)
    if period is not None:
        p0, p1 = int(period[0]), int(period[1])
        common = common[(common >= p0) & (common <= p1)]
    common = np.sort(common.to_numpy())
    if common.size == 0:
        raise ValueError("chronology and climate share no overlapping years.")
    xa = x.reindex(common).to_numpy(dtype=float)
    ca = clim.reindex(common).to_numpy(dtype=float)
    good = np.isfinite(xa) & np.isfinite(ca)
    if not good.all():
        # keep the longest unbroken run so windows are contiguous in time
        xa, ca, common = _longest_run(xa, ca, common, good)
    n = xa.size

    window = int(window)
    offset = int(offset)
    if window < 10:
        raise ValueError("window must be >= 10 years, got %d." % window)
    if window >= n:
        raise ValueError("window (%d) must be shorter than the aligned period "
                         "(%d years)." % (window, n))
    if offset < 1:
        raise ValueError("offset must be >= 1, got %d." % offset)

    rng = np.random.default_rng(seed)

    # observed trace
    r_obs = _moving_correlation(xa, ca, window, method=method, offset=offset)
    starts = np.arange(0, n - window + 1, offset)[:r_obs.size]
    pos_years = _positions(common, window, starts, position)

    overall = float(np.corrcoef(xa, ca)[0, 1])
    r1x = _lag1(xa)
    r1c = _lag1(ca)

    # significance: one ensemble at the observed background correlation for
    # std/range/peak(observed); a second at c=0 only if peak(zero) is requested
    phi_x = _ar1_fit(xa) if noise == "red" else None
    phi_y = _ar1_fit(ca) if noise == "red" else None
    need_obs_ens = ("std" in tests or "range" in tests or
                    ("peak" in tests and peak_condition == "observed"))
    ens = (_mc_ensemble(n, window, overall, iters, noise, method, rng, offset,
                        phi_x, phi_y) if need_obs_ens else None)
    ens_zero = None
    if "peak" in tests and peak_condition == "zero":
        ens_zero = _mc_ensemble(n, window, 0.0, iters, noise, method, rng, offset,
                                phi_x, phi_y)
    test_results = _assemble_tests(r_obs, ens, ens_zero, tests, noise, overall,
                                   iters, window, peak_condition)

    # per-window uncertainty band
    cols = {"start": common[starts], "end": common[starts + window - 1], "r": r_obs}
    if ci == "fisher":
        f = (1.0 - r1x * r1c) / (1.0 + r1x * r1c) if np.isfinite(r1x * r1c) else 1.0
        f = min(max(f, 0.0), 1.0)
        n_eff = _eff_n(window, f)
        lo, hi = _fisher_ci(r_obs, n_eff, ci_level)
        cols["ci_lower"], cols["ci_upper"] = lo, hi
    elif ci == "bootstrap":
        lo, hi = _block_bootstrap_ci(xa, ca, window, starts, ci_level, n_boot, rng)
        cols["ci_lower"], cols["ci_upper"] = lo, hi

    trace = pd.DataFrame(cols, index=pd.Index(pos_years, name="year"))

    settings = {"end_month": end_month, "season_length": season_length, "agg": agg,
                "window": window, "offset": offset, "position": position,
                "method": method, "noise": noise, "tests": tests,
                "peak_condition": peak_condition, "ci": ci, "ci_level": ci_level,
                "iters": iters, "seed": seed}
    result = MovingCorrResult(trace, test_results, overall, r1x, r1c,
                              (int(common[0]), int(common[-1])), label, settings)

    if plot:
        result.plot()
    return result


def _longest_run(xa, ca, years, good):
    """Keep the longest unbroken run of finite paired values, so the windows are
    contiguous in time. Ties resolve to the most recent block (as in climate's
    _longest_unbroken)."""
    best_len, best_end, run = 0, -1, 0
    for i, g in enumerate(good):
        run = run + 1 if g else 0
        if run >= best_len:
            best_len, best_end = run, i
    if best_len == 0:
        raise ValueError("chronology and climate have no overlapping finite years.")
    sl = slice(best_end - best_len + 1, best_end + 1)
    import warnings
    if best_len < good.size:
        warnings.warn("moving_corr: %d of %d aligned years had gaps; using the "
                      "longest unbroken run (%d years)."
                      % (good.size - best_len, good.size, best_len),
                      UserWarning, stacklevel=3)
    return xa[sl], ca[sl], years[sl]


# --------------------------------------------------------------------------- #
# Stage 2: the season grid (moving seascorr) and its field significance
# --------------------------------------------------------------------------- #

def _bh_fdr(p, alpha=0.05):
    """Benjamini-Hochberg FDR: return (boolean significant mask, critical p, k)."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * np.arange(1, m + 1) / m
    passed = ranked <= thresh
    if passed.any():
        k = int(np.max(np.where(passed)[0]) + 1)
        crit = float(ranked[k - 1])
    else:
        k, crit = 0, 0.0
    return (p <= crit), crit, k


def _longest_consecutive(years):
    """Longest run of consecutive integer years (so windows are contiguous)."""
    years = np.asarray(years, dtype=int)
    if years.size == 0:
        return years
    best_s, best_len, s = 0, 1, 0
    for i in range(1, years.size):
        if years[i] != years[i - 1] + 1:
            if i - s > best_len:
                best_s, best_len = s, i - s
            s = i
    if years.size - s > best_len:
        best_s, best_len = s, years.size - s
    return years[best_s:best_s + best_len]


def _moving_corr_grid(x, c3, end_month, season_lengths, agg, window, offset,
                      position, method, noise, ci, ci_level, iters, lc_iters,
                      seed, period, plot):
    """Compute the full seascorr-style season grid of moving correlations plus
    per-cell stability p-values and two field-significance methods (FDR and a
    Livezey-Chen count test). Returns a MovingCorrGridResult."""
    from .climate import _seasonalize, _lag1

    season_lengths = tuple(int(m) for m in season_lengths)
    if not season_lengths:
        raise ValueError("season_lengths must be a non-empty sequence of ints.")
    window = int(window)
    offset = int(offset)
    if offset < 1:
        raise ValueError("offset must be >= 1, got %d." % offset)

    frame, meta = _seasonalize(c3, end_month, season_lengths, agg)
    cells = list(frame.columns)
    ncell = len(cells)
    labels = np.array([m["label"] for m in meta])
    slen = np.array([m["season_length"] for m in meta])
    emon = np.array([m["end_month"] for m in meta])
    prior = np.array([m["prior"] for m in meta])

    # common contiguous year span across all cells and the chronology
    common = x.index
    for key in cells:
        common = common.intersection(frame[key].dropna().index.astype(int))
    common = np.sort(common.to_numpy())
    if period is not None:
        p0, p1 = int(period[0]), int(period[1])
        common = common[(common >= p0) & (common <= p1)]
    common = _longest_consecutive(common)
    nt = common.size
    if window < 10:
        raise ValueError("window must be >= 10 years, got %d." % window)
    if window >= nt:
        raise ValueError("window (%d) must be shorter than the aligned grid "
                         "period (%d years)." % (window, nt))

    xa = x.reindex(common).to_numpy(dtype=float)
    clim_mat = np.vstack([frame[key].reindex(common).to_numpy(dtype=float)
                          for key in cells])
    starts = np.arange(0, nt - window + 1, offset)
    pos_years = common[starts + (window - 1) // 2]
    L = starts.size

    rng = np.random.default_rng(seed)
    phi_x = _ar1_fit(xa) if noise == "red" else None
    r1x = _lag1(xa)

    cube = np.empty((ncell, L))
    overall = np.empty(ncell)
    pstd = np.empty(ncell)
    thr95 = np.empty(ncell)
    r1c = np.empty(ncell)
    for j in range(ncell):
        ca = clim_mat[j]
        r = _moving_correlation(xa, ca, window, method=method, offset=offset)
        cube[j] = r[:L]
        oc = float(np.corrcoef(xa, ca)[0, 1])
        overall[j] = oc
        r1c[j] = _lag1(ca)
        phi_c = _ar1_fit(ca) if noise == "red" else None
        ens = _mc_ensemble(nt, window, oc, iters, noise, method, rng, offset,
                           phi_x, phi_c)["std"]
        pstd[j] = float(np.mean(ens >= float(np.nanstd(r))))
        thr95[j] = float(np.quantile(ens, 0.95))
    anomaly = cube - overall[:, None]

    fdr_sig, fdr_crit, fdr_k = _bh_fdr(pstd, 0.05)

    # Livezey-Chen field test: the count of locally-significant cells vs the
    # distribution of that count under chronologies unrelated to climate (the
    # real climate grid is reused, so the inter-season correlation is preserved)
    K_obs = int(np.sum(pstd < 0.05))
    K_null = np.empty(int(lc_iters), dtype=int)
    for it in range(int(lc_iters)):
        if noise == "red":
            e = rng.standard_normal(nt + 500)
            s_sur = lfilter([1.0], [1.0, -phi_x], e)[500:]
        else:
            s_sur = rng.standard_normal(nt)
        cnt = 0
        for j in range(ncell):
            rr = _moving_correlation(s_sur, clim_mat[j], window, method=method,
                                     offset=offset)
            if np.nanstd(rr[:L]) >= thr95[j]:
                cnt += 1
        K_null[it] = cnt
    lc_p = float(np.mean(K_null >= K_obs))

    grid = pd.DataFrame({"season_length": slen, "end_month": emon,
                         "prior": prior, "label": labels, "overall_r": overall,
                         "stability_p": pstd, "fdr_sig": fdr_sig})
    field = {"fdr_crit": fdr_crit, "fdr_k": fdr_k, "n_fdr": int(fdr_sig.sum()),
             "lc_p": lc_p, "lc_K_obs": K_obs, "lc_K_null": K_null}
    settings = {"end_month": int(end_month), "season_lengths": season_lengths,
                "agg": agg, "window": window, "offset": offset,
                "position": position, "method": method, "noise": noise,
                "ci": ci, "ci_level": ci_level, "iters": int(iters),
                "lc_iters": int(lc_iters), "seed": seed}
    result = MovingCorrGridResult(grid, cube, anomaly, pos_years, overall, pstd,
                                  thr95, r1c, r1x, fdr_sig, field,
                                  (int(common[0]), int(common[-1])), settings)
    if plot:
        result.plot()
    return result


class MovingCorrGridResult:
    """Result of :func:`moving_corr` in grid mode (a moving seascorr).

    Attributes
    ----------
    grid : pandas.DataFrame
        One row per season cell: ``season_length``, ``end_month``, ``prior``,
        ``label``, ``overall_r`` (full-period correlation), ``stability_p`` (the
        Gershunov std-test p for the trace), and ``fdr_sig`` (survives FDR).
    cube : ndarray, shape (n_cells, n_windows)
        The moving-correlation trace of every cell.
    anomaly : ndarray
        ``cube`` minus each cell's ``overall_r`` -- the departure from baseline.
    pos_years : ndarray
        Window-position years (per ``position``), shared by all cells.
    field : dict
        Field significance: ``lc_p`` (Livezey-Chen grid-level p), ``lc_K_obs``
        and ``lc_K_null`` (observed and null counts of locally-significant cells),
        ``n_fdr`` / ``fdr_crit`` / ``fdr_k`` (the FDR set).
    overall, stability_p, thr95, r1_climate : ndarray (per cell)
    r1_chronology : float
    period : tuple(int, int)
    settings : dict
    """

    def __init__(self, grid, cube, anomaly, pos_years, overall, stability_p,
                 thr95, r1_climate, r1_chronology, fdr_sig, field, period,
                 settings):
        self.grid = grid
        self.cube = cube
        self.anomaly = anomaly
        self.pos_years = pos_years
        self.overall = overall
        self.stability_p = stability_p
        self.thr95 = thr95
        self.r1_climate = r1_climate
        self.r1_chronology = r1_chronology
        self.fdr_sig = fdr_sig
        self.field = field
        self.period = period
        self.settings = settings
        self.fig = None
        self.axes = None

    def __repr__(self):
        a, b = self.period
        return ("MovingCorrGridResult(%d seasons, %d-%d, window=%d; FDR %d sig, "
                "LC p=%.3f)" % (len(self.grid), a, b, self.settings["window"],
                                self.field["n_fdr"], self.field["lc_p"]))

    def summary(self):
        a, b = self.period
        lines = ["Moving correlation grid (moving seascorr)",
                 "  period        : %d-%d (%d years)" % (a, b, b - a + 1),
                 "  seasons       : %d cells, lengths %s"
                 % (len(self.grid), tuple(self.settings["season_lengths"])),
                 "  window/offset : %d / %d" % (self.settings["window"],
                                                self.settings["offset"]),
                 "  null model    : %s" % self.settings["noise"],
                 "  FDR           : %d of %d seasons significant (crit p=%.4f)"
                 % (self.field["n_fdr"], len(self.grid), self.field["fdr_crit"]),
                 "  Livezey-Chen  : %d locally-significant cells vs ~%d expected, "
                 "field p=%.3f" % (self.field["lc_K_obs"],
                                   int(np.median(self.field["lc_K_null"])),
                                   self.field["lc_p"])]
        return "\n".join(lines)

    def plot(self, kind="small_multiples", **kwargs):
        from .moving_corr_grid_plot import moving_corr_grid_plot
        return moving_corr_grid_plot(self, kind=kind, **kwargs)
