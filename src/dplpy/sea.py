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

# Title: sea.py
# Project: OpenDendro dplPy
# Description: Superposed epoch analysis (SEA) with the Rao et al. (2019) double
#              bootstrap -- a Python port of the MATLAB sea_dbl (K. Anchukaitis and
#              M. Rao). A climate series is composited in windows around a list of
#              event years (e.g. volcanic eruptions), normalized to each window's
#              pre-event mean. An outer bootstrap over unique event subsets gives a
#              response-uncertainty envelope (Rao's contribution); an inner block or
#              random bootstrap gives the significance thresholds. Validated against
#              the MATLAB: the deterministic kernel (windows, normalization,
#              composite, block length, percentiles with method="hazen") and the
#              subset envelope match to machine precision on identical draws. This
#              is the third function in dplPy's climate-response module (after
#              seascorr and pcreg).
#
#              Reference: Rao, M.P., Cook, E.R., Cook, B.I., Anchukaitis, K.J.,
#              D'Arrigo, R.D., Krusic, P.J., LeGrande, A.N. (2019). A double
#              bootstrap approach to Superposed Epoch Analysis to evaluate response
#              uncertainty. Dendrochronologia 55, 119-124.

from math import comb

import numpy as np
import pandas as pd

from ._validate import _warn

_P_SVAL = [5, 50, 95]                       # envelope percentiles (double bootstrap)
_P_CI = [1, 5, 10, 90, 95, 99]              # null-distribution percentiles


def _as_series(series):
    """Coerce input to (values, years) float arrays. Accepts a pandas Series with a
    year index, or a (values, years) pair."""
    if isinstance(series, pd.Series):
        y = series.to_numpy(dtype=float)
        yr = series.index.to_numpy(dtype=float)
    else:
        y, yr = series
        y = np.asarray(y, dtype=float)
        yr = np.asarray(yr, dtype=float)
    if y.shape != yr.shape or y.ndim != 1:
        raise ValueError("series must be 1-D with matching values and years.")
    return y, yr


def _windows(y, yr, events, pre, post):
    """Raw event-window matrix (len(events) x W): y[center-pre : center+post]."""
    W = pre + post + 1
    X = np.full((len(events), W), np.nan)
    for i, ev in enumerate(events):
        hit = np.where(yr == ev)[0]
        c = int(hit[0])
        X[i] = y[c - pre:c + post + 1]
    return X


def _normalize(X, pre):
    """Subtract each row's pre-event mean (columns 0:pre; year 0 excluded)."""
    return X - np.nanmean(X[:, :pre], axis=1, keepdims=True)


def _composite(Xnorm):
    return np.nanmean(Xnorm, axis=0)


def _block_length(y):
    """Lag-1 autocorrelation (biased, as acf.m) and block length p = floor(2*tau),
    tau = -1/ln(r1). r1<=0 is floored to 1e-30; p<=0 is set to 1."""
    v = y[~np.isnan(y)]
    vb = v.mean()
    denom = np.sum((v - vb) ** 2)
    r1 = np.sum((v[1:] - vb) * (v[:-1] - vb)) / denom
    if r1 <= 0:
        r1 = 1.0e-30
    tau = -1.0 / np.log(r1)
    p = int(np.floor(2 * tau))
    if p <= 0:
        p = 1
    return r1, p


def _unique_subsets(events, nsample, n_subsets, rng):
    """n_subsets distinct sorted subsets of size nsample drawn without replacement
    within each subset (Rao's outer resampling). Correct escape on exhaustion."""
    seen = set()
    out = []
    max_tries = 100 * n_subsets + 1000
    tries = 0
    while len(out) < n_subsets:
        s = tuple(sorted(rng.choice(events, size=nsample, replace=False)))
        tries += 1
        if s not in seen:
            seen.add(s)
            out.append(np.array(s))
        if tries > max_tries:
            raise RuntimeError("could not draw enough unique event subsets; reduce "
                               "n_subsets or n_sample.")
    return out


def _subset_composites(y, yr, subsets, pre, post):
    """Normalized composite for each event subset (n_subsets x W)."""
    out = np.empty((len(subsets), pre + post + 1))
    for j, sub in enumerate(subsets):
        out[j] = _composite(_normalize(_windows(y, yr, sub, pre, post), pre))
    return out


def _block_shuffle_row(row, p, op, oddcase, odd_choice, block_perm, shift):
    """One row's block reshuffle, faithful to sea_dbl's block branch, with the
    random choices passed in: ``odd_choice`` indexes ``oddcase`` for the short
    (length-op) block, ``block_perm`` permutes the remaining p-length blocks, and
    ``shift`` is the final circular shift. ``op = len(row) % p``."""
    n = len(row)
    mcm = np.full(n, np.nan)
    if op != 0:
        oddpos = oddcase[odd_choice]
        mcm[oddpos:oddpos + op] = row[oddpos:oddpos + op]
        remaining = np.delete(row, np.arange(oddpos, oddpos + op))
    else:
        remaining = row.copy()
    remcase = np.where(np.isnan(mcm))[0]
    rempos = remcase[::p]                           # available block starts
    remposmc = rempos[block_perm]                   # shuffled block starts
    postrem = np.arange(0, remaining.size, p)
    for j in range(len(remposmc)):
        mcm[remposmc[j]:remposmc[j] + p] = remaining[postrem[j]:postrem[j] + p]
    return np.roll(mcm, shift)


def _block_null(window_sets, pre, p, n_boot, rng):
    """Block-bootstrap null: for each iteration, take one set of raw event windows
    (one subset in double mode, the single event set otherwise), reshuffle each row
    in autocorrelation-preserving blocks of length p with a random circular shift,
    re-normalize to the pre-event mean, and composite. Returns n_boot x W."""
    W = window_sets[0].shape[1]
    op = W % p
    oddcase = np.arange(0, W, p)
    nblocks_remaining = None
    null = np.empty((n_boot, W))
    for b in range(n_boot):
        X = window_sets[rng.integers(len(window_sets))]
        mcm = np.empty_like(X)
        for k in range(X.shape[0]):
            # number of remaining blocks (positions left after the odd block)
            rem_size = W - op if op != 0 else W
            nblocks = int(np.ceil(rem_size / p)) if op != 0 else W // p
            odd_choice = rng.integers(len(oddcase)) if op != 0 else 0
            block_perm = rng.permutation(nblocks)
            shift = int(np.floor(rng.random() * W))
            mcm[k] = _block_shuffle_row(X[k], p, op, oddcase, odd_choice,
                                        block_perm, shift)
        null[b] = _composite(_normalize(mcm, pre))
    return null


def _random_null(y, yr, pre, post, n_events, n_boot, rng):
    """Random-draw null: pseudo-event years drawn with replacement from the
    interior of the series, windowed, normalized and composited. Returns n_boot x W."""
    interior = yr[pre:len(yr) - post]
    null = np.empty((n_boot, pre + post + 1))
    for b in range(n_boot):
        evs = rng.choice(interior, size=n_events, replace=True)
        null[b] = _composite(_normalize(_windows(y, yr, evs, pre, post), pre))
    return null


class SeaResult:
    """Result of a superposed epoch analysis (see ``dpl.sea``).

    Attributes
    ----------
    lags : np.ndarray
        window-relative years, ``-pre .. post``.
    composite : np.ndarray
        the ordinary all-event superposed-epoch composite (always computed).
    envelope : np.ndarray or None
        the double-bootstrap response envelope, a 3 x W array of the 5th/50th/95th
        percentiles across event-subset composites, or None if not run.
    null : pandas.DataFrame
        the null-distribution thresholds, indexed by percentile (1/5/10/90/95/99),
        columns the lags.
    windows : np.ndarray
        the normalized event-window matrix (events x W).
    events : np.ndarray
        the event years actually used (after excising those too near the ends).
    r1, block_length : float, int
        lag-1 autocorrelation and block length (block null only).
    settings : dict
        the call parameters.
    """

    def __init__(self, lags, composite, envelope, null, windows, events,
                 r1, block_length, settings):
        self.lags = lags
        self.composite = composite
        self.envelope = envelope
        self.null = null
        self.windows = windows
        self.events = events
        self.r1 = r1
        self.block_length = block_length
        self.settings = settings
        self.fig = None
        self.axes = None

    def __repr__(self):
        kind = "double-bootstrap" if self.envelope is not None else "ordinary"
        return ("SeaResult(%s SEA, %d events, pre=%d post=%d, null=%s)"
                % (kind, len(self.events), self.settings["pre"],
                   self.settings["post"], self.settings["boot_method"]))

    def plot(self, **kwargs):
        from .sea_plot import sea_plot
        return sea_plot(self, **kwargs)


def sea(series, events, pre=5, post=5, double_bootstrap=True, n_sample=None,
        n_subsets=1000, n_boot=1000, boot_method="block", seed=None, plot=False):
    """Superposed epoch analysis with the Rao et al. (2019) double bootstrap.

    Composite a climate series in windows around a set of event years, normalized to
    each window's pre-event mean, to estimate the average response to the events. The
    double bootstrap adds a response-uncertainty envelope (how much the composite
    depends on which events are included) and a Monte-Carlo null for significance.

    Parameters
    ----------
    series : pandas.Series or (values, years)
        the climate series, with its years (a Series indexed by year, or a pair of
        1-D arrays).
    events : array-like
        the event (key) years. Events whose window would run past either end of the
        series are dropped with a warning.
    pre, post : int, default 5
        window half-widths: years before and after each event. The window is
        ``W = pre + post + 1`` wide and the pre-event mean over the ``pre`` years
        before year 0 is the normalization baseline.
    double_bootstrap : bool, default True
        if True, run Rao's outer bootstrap over unique event subsets to produce the
        response envelope; if False, an ordinary SEA (the composite and a null only).
    n_sample : int, default None
        events per subset (the outer bootstrap). None uses about half the events,
        which maximizes the number of distinct subsets. Ignored unless
        ``double_bootstrap``.
    n_subsets : int, default 1000
        number of unique event subsets. If more than the number of possible subsets
        ``C(n_events, n_sample)``, the double bootstrap is switched off with a
        warning (an ordinary SEA is returned).
    n_boot : int, default 1000
        number of Monte-Carlo iterations for the null distribution.
    boot_method : {"block", "random"}, default "block"
        the null. ``"block"`` reshuffles the real event windows in
        autocorrelation-preserving blocks (tests whether the response is locked to
        year 0); ``"random"`` draws pseudo-event years from the series (tests the
        amplitude against random years).
    seed : int or None, default None
        seed for reproducibility (the MATLAB sets none, so its output varies run to
        run; dplPy makes it reproducible).
    plot : bool, default False
        if True, draw the result (shaded envelope, median, null thresholds).

    Returns
    -------
    SeaResult
        see the class for attributes. ``composite`` is always the ordinary all-event
        mean; ``envelope`` is the 5/50/95 response envelope when the double bootstrap
        ran; ``null`` holds the 1/5/10/90/95/99 significance thresholds per lag.

    References
    ----------
    Rao, M.P. et al. (2019). Dendrochronologia 55, 119-124. Adams, J.B., Mann, M.E.,
    Ammann, C.M. (2003). Nature 426, 274-278 (block bootstrap).
    """
    boot_method = str(boot_method).strip().lower()
    if boot_method not in ("block", "random"):
        raise ValueError("boot_method must be 'block' or 'random'.")
    y, yr = _as_series(series)
    events = np.sort(np.unique(np.asarray(events, dtype=float)))
    rng = np.random.default_rng(seed)

    # excise events whose window exceeds either end of the series
    keep = (events >= yr[0] + pre) & (events <= yr[-1] - post)
    if not keep.all():
        _warn("sea: excising %d event(s) whose window exceeds the series: %s"
              % (int((~keep).sum()), events[~keep].astype(int).tolist()))
        events = events[keep]
    if len(events) == 0:
        raise ValueError("no events remain after excising those too near the ends.")

    # truncate the series to start at the first event's pre-window (as sea_dbl does)
    strt = int(np.where(yr == events.min() - pre)[0][0])
    y, yr = y[strt:], yr[strt:]
    n_events = len(events)

    if double_bootstrap:
        if n_sample is None:
            n_sample = max(1, round(n_events / 2))
        possible = comb(n_events, n_sample)
        if possible < n_subsets:
            _warn("sea: C(%d, %d) = %d < n_subsets = %d; running an ordinary SEA "
                  "(no response envelope)." % (n_events, n_sample, possible, n_subsets))
            double_bootstrap = False

    # ordinary composite (always), from the full event set
    windows0 = _windows(y, yr, events, pre, post)
    comp = _composite(_normalize(windows0, pre))

    # double-bootstrap response envelope
    envelope = None
    null_row_sets = [windows0]
    if double_bootstrap:
        subsets = _unique_subsets(events, n_sample, n_subsets, rng)
        subcomp = _subset_composites(y, yr, subsets, pre, post)
        envelope = np.percentile(subcomp, _P_SVAL, axis=0, method="hazen")
        null_row_sets = [_windows(y, yr, s, pre, post) for s in subsets]

    # null distribution
    r1 = block_length = None
    if boot_method == "block":
        r1, block_length = _block_length(y)
        null = _block_null(null_row_sets, pre, block_length, n_boot, rng)
    else:
        n_draw = n_sample if double_bootstrap else n_events
        null = _random_null(y, yr, pre, post, n_draw, n_boot, rng)
    null_ci = np.percentile(null, _P_CI, axis=0, method="hazen")

    lags = np.arange(-pre, post + 1)
    null_df = pd.DataFrame(null_ci, index=_P_CI, columns=lags)
    settings = dict(pre=pre, post=post, double_bootstrap=double_bootstrap,
                    n_sample=n_sample if double_bootstrap else None,
                    n_subsets=n_subsets if double_bootstrap else None,
                    n_boot=n_boot, boot_method=boot_method, seed=seed)
    res = SeaResult(lags, comp, envelope, null_df, _normalize(windows0, pre),
                    events, r1, block_length, settings)
    if plot:
        res.plot()
    return res
