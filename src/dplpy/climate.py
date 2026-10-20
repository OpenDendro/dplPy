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

# Title: seascorr.py
# Project: OpenDendro dplPy
# Description: Seasonal correlation analysis of a tree-ring chronology against
#              monthly climate -- a faithful Python port of SEASCORR (Meko,
#              Touchan & Anchukaitis 2011, Computers & Geosciences 37, 1234-1241),
#              cross-checked against the authors' MATLAB. The chronology is
#              correlated (simple) with a primary climate variable and
#              (partial, controlling for the primary) with a secondary variable
#              across 56 seasons (4 season-lengths x 14 ending months).
#              Significance comes from exact spectral simulation of the
#              chronology (Percival & Constantine 2006). This is the first
#              function in dplPy's climate-response module.

import numpy as np
import pandas as pd

from ._validate import _warn

_MIN_YEARS = 31          # pdgmsim tooshort+1: spectrum needs a reasonable length
_N_END_MONTHS = 14       # months in the ending-month window (endmo + 13 before)
_MIN_NSIM = 1000         # below this the tail probability points are too coarse

_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_ALIAS = {}
for _i, _m in enumerate(_MONTH_ABBR, start=1):
    _MONTH_ALIAS[_m.lower()] = _i
    _MONTH_ALIAS[str(_i)] = _i
    _MONTH_ALIAS[("%02d" % _i)] = _i
_MONTH_FULL = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
               "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
               "november": 11, "december": 12}


# --------------------------------------------------------------------------- #
# Exact spectral simulation (port of Meko's pdgmsim.m)
# --------------------------------------------------------------------------- #

def _pdgmsim(x, nsim, rng, mtaper=0.1):
    """Exact simulation of a Gaussian series from its periodogram.

    A numpy port of Meko's ``pdgmsim.m``, which implements the simplified
    circulant-embedding variant of Percival & Constantine (2006, Stat. Comput.
    16, 25-35): draw complex Gaussian weights scaled by the square-root of the
    periodogram of the tapered/padded series, take a DFT, and read the first
    ``M`` real and first ``M`` imaginary parts as two independent surrogates.

    Each surrogate preserves the spectral structure (hence the autocorrelation)
    of ``x`` and is rescaled to the exact mean and standard deviation of ``x``
    (Meko's kopt=1, the setting SEASCORR uses).

    Parameters
    ----------
    x : 1-D array
        The observed series (a chronology), length N, no NaN.
    nsim : int
        Number of surrogate series to return.
    rng : numpy.random.Generator
        Random source (see ``numpy.random.default_rng``).
    mtaper : float
        Total fraction of the series tapered by a split cosine bell (half at
        each end). Meko uses 0.10.

    Returns
    -------
    ndarray, shape (N, nsim)
        Columns are surrogate series.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    xmn = x.mean()
    xstd = x.std(ddof=1)                       # MATLAB std is ddof=1

    # subtract mean
    x0 = x - xmn

    # split cosine-bell taper, mtaper/2 of the data at each end
    ends = int((mtaper / 2.0) * n)            # MATLAB fix(): truncate toward zero
    dwin = np.ones(n)
    if ends > 0:
        t = np.arange(1, ends + 1)
        dwin[:ends] = 0.5 * (1.0 - np.cos(np.pi * (t - 0.5) / ends))
        t2 = np.arange(n - ends + 1, n + 1)
        dwin[n - ends:] = 0.5 * (1.0 - np.cos(np.pi * (n - t2 + 0.5) / ends))
    xt = x0 * dwin
    xt = xt - xt.mean()                        # recenter tapered series to zero mean

    # pad with zeros to the next power of two >= 2N
    padlen = 1 << int(np.ceil(np.log2(2 * n)))
    xpad = np.zeros(padlen)
    xpad[:n] = xt

    # rescale so variance of tapered+padded equals variance of the original
    std1 = x.std(ddof=1)
    std2 = xpad.std(ddof=1)
    mn2 = xpad.mean()
    xpad = (xpad - mn2) * (std1 / std2) + mn2

    # periodogram (spectral weights), length padlen = 2M
    z = np.fft.fft(xpad)
    S = (z * np.conj(z)).real / padlen
    M = padlen // 2

    # algorithm yields surrogates in pairs; generate ceil(nsim/2) pairs
    nsim1 = int(np.ceil(nsim / 2.0))
    Z = rng.standard_normal((2 * padlen, nsim1))
    term1 = Z[0::2, :] + 1j * Z[1::2, :]       # padlen complex rows per pair
    term2 = np.sqrt(S / padlen)
    mu = term1 * term2[:, None]
    V = np.fft.fft(mu, axis=0)

    D1 = V.real[:M, :]
    D2 = V.imag[:M, :]
    D = np.concatenate([D1, D2], axis=1)       # (M, 2*nsim1) candidate surrogates
    D = D[:n, :nsim]                           # truncate to length N and nsim cols

    # force each surrogate to the observed mean and sd (kopt=1)
    Dmean = D.mean(axis=0)
    Dstd = D.std(axis=0, ddof=1)
    D = (D - Dmean) / Dstd * xstd + xmn
    return D


# --------------------------------------------------------------------------- #
# Empirical non-exceedance probability (port of Meko's subfun13)
# --------------------------------------------------------------------------- #

def _weibull_p(obs, sims):
    """Weibull-plotting-position non-exceedance probability of ``obs`` in the
    empirical distribution ``sims`` (Meko Eq. 2 / subfun13).

    ``p = j / (n + 1)`` where ``j`` is the interpolated rank (1..n) of ``obs``
    among the sorted ``sims``. Values at or beyond the extremes are capped at
    rank 1 or n rather than extrapolated. ``obs`` may be a scalar or an array;
    returns the matching shape.
    """
    s = np.sort(np.asarray(sims, dtype=float))
    n = s.size
    ranks = np.arange(1, n + 1, dtype=float)
    obs = np.asarray(obs, dtype=float)
    scalar = obs.ndim == 0
    o = np.atleast_1d(obs)
    j = np.interp(o, s, ranks)                 # clamps to [1, n] outside the range
    p = j / (n + 1.0)
    return float(p[0]) if scalar else p


# --------------------------------------------------------------------------- #
# Climate input normalization: accept wide / tidy-long / datetime-monthly
# --------------------------------------------------------------------------- #

def _month_number(name):
    """Map a column label to a month number 1..12, or None if unrecognizable."""
    key = str(name).strip().lower()
    if key in _MONTH_ALIAS:
        return _MONTH_ALIAS[key]
    if key in _MONTH_FULL:
        return _MONTH_FULL[key]
    if key[:3] in _MONTH_ALIAS:
        return _MONTH_ALIAS[key[:3]]
    return None


def _to_month3(clim, name):
    """Normalize one monthly climate variable to a tidy (year, month, value)
    DataFrame, sorted by year then month.

    Accepts three shapes (see the design doc): a Series/1-column DataFrame on a
    monthly DatetimeIndex or PeriodIndex; a tidy frame with year/month/value
    columns; or a wide frame with a year index-or-column plus 12 month columns.
    """
    # --- Series: must be datetime/period indexed ---
    if isinstance(clim, pd.Series):
        idx = clim.index
        if isinstance(idx, pd.PeriodIndex):
            idx = idx.to_timestamp()
        if isinstance(idx, pd.DatetimeIndex):
            return _mk3(idx.year, idx.month, clim.to_numpy(dtype=float), name)
        raise ValueError(name + ": a Series must be indexed by a monthly "
                         "DatetimeIndex or PeriodIndex.")

    if not isinstance(clim, pd.DataFrame):
        raise TypeError(name + " must be a pandas DataFrame or Series, not "
                        + str(type(clim)) + ".")

    df = clim
    idx = df.index
    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp()

    # --- datetime-indexed frame: one value column ---
    if isinstance(idx, pd.DatetimeIndex):
        if df.shape[1] != 1:
            raise ValueError(name + ": a datetime-indexed frame must have exactly "
                             "one value column; got " + str(list(df.columns)) + ".")
        return _mk3(idx.year, idx.month, df.iloc[:, 0].to_numpy(dtype=float), name)

    lower = {str(c).lower(): c for c in df.columns}

    # --- tidy long: year, month, value ---
    if "year" in lower and "month" in lower and "value" in lower:
        return _mk3(df[lower["year"]].to_numpy(),
                    df[lower["month"]].to_numpy(),
                    df[lower["value"]].to_numpy(dtype=float), name)

    # --- wide: year (index or column) + 12 month columns ---
    if "year" in lower:
        years = df[lower["year"]].to_numpy()
        month_cols = [c for c in df.columns if c != lower["year"]]
    else:
        years = np.asarray(df.index)
        month_cols = list(df.columns)
    if len(month_cols) != 12:
        raise ValueError(name + ": wide format needs a year index/column plus 12 "
                         "month columns; found " + str(len(month_cols))
                         + " non-year columns " + str([str(c) for c in month_cols])
                         + ". Or pass tidy (year, month, value) or a "
                         "datetime-indexed monthly series.")
    nums = [_month_number(c) for c in month_cols]
    if all(v is not None for v in nums):
        order = [month_cols[k] for k in np.argsort(nums)]   # sort to Jan..Dec
    else:
        order = month_cols                                   # assume already Jan..Dec
    years = np.asarray(years)
    yy = np.repeat(years, 12)
    mm = np.tile(np.arange(1, 13), len(years))
    vals = df[order].to_numpy(dtype=float).ravel()           # row-major: year then month
    return _mk3(yy, mm, vals, name)


def _mk3(year, month, value, name):
    out = pd.DataFrame({"year": np.asarray(year).astype(int),
                        "month": np.asarray(month).astype(int),
                        "value": np.asarray(value, dtype=float)})
    if out["month"].min() < 1 or out["month"].max() > 12:
        raise ValueError(name + ": month values must be in 1..12.")
    out = out.sort_values(["year", "month"]).reset_index(drop=True)
    if out.duplicated(["year", "month"]).any():
        raise ValueError(name + ": duplicate (year, month) entries.")
    return out


# --------------------------------------------------------------------------- #
# Seasonalization (port of subfun01 / mafilt1)
# --------------------------------------------------------------------------- #

def _season_label(season_length, end_month, prior):
    """Human-readable season label, e.g. 'Jun' / 'Mar-May' / 'Oct*' (prior year
    ending months carry a trailing '*')."""
    star = "*" if prior else ""
    if season_length == 1:
        return _MONTH_ABBR[end_month - 1] + star
    start_month = (end_month - season_length) % 12 + 1
    return _MONTH_ABBR[start_month - 1] + "-" + _MONTH_ABBR[end_month - 1] + star


def _seasonalize(m3, end_month, season_lengths, agg):
    """Collapse a tidy monthly series into the seasonal matrix.

    Returns ``(frame, meta)`` where ``frame`` is a DataFrame indexed by growth
    year with one column per season (ordered season-length outer, ending month
    inner, oldest ending month first -- matching SEASCORR's 56-column layout),
    and ``meta`` is a list of dicts (season_length, end_month, prior, label) one
    per column. A season whose months are not all present is NaN; the caller
    truncates to the longest unbroken block.
    """
    # dense monthly series on an absolute month index (year*12 + month-1)
    abs_idx = m3["year"].to_numpy() * 12 + (m3["month"].to_numpy() - 1)
    vals = m3["value"].to_numpy(dtype=float)
    lo, hi = int(abs_idx.min()), int(abs_idx.max())
    dense = np.full(hi - lo + 1, np.nan)
    dense[abs_idx - lo] = vals
    s = pd.Series(dense)

    rolled = {}
    for m in season_lengths:
        if agg == "sum":
            rolled[m] = s.rolling(m, min_periods=m).sum().to_numpy()
        else:
            rolled[m] = s.rolling(m, min_periods=m).mean().to_numpy()

    # growth years: a growth year Y ends at (Y, end_month); it needs the 14
    # ending months e-13..e (e = abs index of that month) all within range.
    years = np.unique(m3["year"].to_numpy())
    cols = {}
    meta = []
    growth_years = None
    for m in season_lengths:
        rm = rolled[m]
        for p in range(_N_END_MONTHS):          # p=0 oldest ending month .. 13 newest (end_month)
            off = _N_END_MONTHS - 1 - p          # months before end_month
            colname = (m, p)
            vcol = []
            gy = []
            for Y in years:
                e = Y * 12 + (end_month - 1) - lo
                pos = e - off
                if 0 <= pos < rm.size:
                    vcol.append(rm[pos])
                    gy.append(Y)
            cols[colname] = pd.Series(vcol, index=gy)
            if growth_years is None:
                growth_years = gy
            # metadata: calendar month/year of this ending month
            end_abs = (end_month - 1) - off
            cal_month = end_abs % 12 + 1
            prior = end_abs < 0
            meta.append({"season_length": m, "end_month": cal_month,
                         "prior": prior, "label": _season_label(m, cal_month, prior)})

    frame = pd.DataFrame(cols).sort_index()
    frame.index.name = "year"
    return frame, meta


# --------------------------------------------------------------------------- #
# Correlation core
# --------------------------------------------------------------------------- #

def _zcols(M):
    """Standardize each column to zero mean, unit (population) sd."""
    M = np.asarray(M, dtype=float)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    return (M - mu) / sd


def _corr_matrix(A, B):
    """Pearson r between every column of A and every column of B.

    A is (n, a), B is (n, b); returns (a, b). r = Za^T Zb / n.
    """
    n = A.shape[0]
    return (_zcols(A).T @ _zcols(B)) / n


def _corr_paired(A, B):
    """Pearson r between matching columns of A and B (both (n, k)) -> (k,)."""
    n = A.shape[0]
    return np.sum(_zcols(A) * _zcols(B), axis=0) / n


def _partial(r_xU, r_xV, r_UV):
    """First-order partial correlation of x with V controlling for U
    (Meko Eq. 1), all inputs elementwise per season."""
    denom = np.sqrt((1.0 - r_xU ** 2) * (1.0 - r_UV ** 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (r_xV - r_xU * r_UV) / denom


def _lag1(x):
    """Lag-1 autocorrelation as Pearson r of the series against itself shifted by
    one (Meko's subfun18: corrcoef of x[:-1], x[1:])."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    a, b = x[:-1], x[1:]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else np.nan


def _eff_n(n, f):
    """Effective sample size n*f, rounded, capped at n (Meko's sseff)."""
    ne = int(round(n * f))
    return min(ne, int(n))


def _corrdiff(r1, r2, n1, n2):
    """Two-sided test that two independent correlations estimate the same
    population value (Meko's corrdiff; Fisher z, Snedecor & Cochran 1989).

    Returns (z1, z2, delta_z, p). NaN where a sample size is too small (n<=3).
    """
    from scipy.stats import norm
    if n1 <= 3 or n2 <= 3 or not (-1 < r1 < 1) or not (-1 < r2 < 1):
        z1 = 0.5 * (np.log1p(r1) - np.log1p(-r1)) if -1 < r1 < 1 else np.nan
        z2 = 0.5 * (np.log1p(r2) - np.log1p(-r2)) if -1 < r2 < 1 else np.nan
        return z1, z2, (z1 - z2), np.nan
    z1 = 0.5 * (np.log1p(r1) - np.log1p(-r1))
    z2 = 0.5 * (np.log1p(r2) - np.log1p(-r2))
    d = z1 - z2
    v = 1.0 / (n1 - 3) + 1.0 / (n2 - 3)
    tstat = d / np.sqrt(v)
    cdf = norm.cdf(tstat)
    p = (1.0 - cdf) * 2.0 if cdf > 0.5 else cdf * 2.0
    return z1, z2, d, p


def _longest_unbroken(frame):
    """First and last index labels of the longest run of rows with no NaN in any
    column (port of nonan1; ties resolved to the most recent block)."""
    good = ~frame.isna().any(axis=1).to_numpy()
    idx = frame.index.to_numpy()
    best_len, best_end = 0, -1
    run = 0
    for i, g in enumerate(good):
        run = run + 1 if g else 0
        if run >= best_len:                    # >= keeps the most recent on ties
            best_len, best_end = run, i
    if best_len == 0:
        return None, None
    return int(idx[best_end - best_len + 1]), int(idx[best_end])


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #

class SeascorrResult:
    """Result of :func:`seascorr`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per season: season_length, end_month, label, r_primary,
        p_primary, r_partial, p_partial, r_clim (primary-vs-secondary), and the
        two-sided significance flags sig_05 / sig_01 for each correlation.
    period : tuple(int, int)
        First and last year of the analysis window.
    nsim : int
        Number of exact simulations used.
    end_month, season_lengths : analysis settings.
    highest : pandas.DataFrame
        Strongest primary-variable season per season-length.
    acf : ndarray
        Lag 1-5 autocorrelation of the chronology over the window.
    stability : pandas.DataFrame or None
        Two-period difference-of-correlation test, when two windows were given.
    """

    def __init__(self, table, period, nsim, end_month, season_lengths,
                 highest, acf, stability=None):
        self.table = table
        self.period = period
        self.nsim = nsim
        self.end_month = end_month
        self.season_lengths = season_lengths
        self.highest = highest
        self.acf = acf
        self.stability = stability
        self.fig = None
        self.axes = None

    def __repr__(self):
        a, b = self.period
        return ("SeascorrResult(period=%d-%d, nsim=%d, end_month=%d, "
                "season_lengths=%s)" % (a, b, self.nsim, self.end_month,
                                        tuple(self.season_lengths)))


# --------------------------------------------------------------------------- #
# Chronology input
# --------------------------------------------------------------------------- #

_CRN_PREFER = ("std", "res", "spline", "arstan", "mean", "value", "crn", "chron")


def _as_chronology(crn, column):
    """Coerce the chronology input to a year-indexed float Series."""
    if isinstance(crn, pd.Series):
        s = crn
    elif isinstance(crn, pd.DataFrame):
        if column is not None:
            if column not in crn.columns:
                raise ValueError("column " + repr(column) + " not in chronology; "
                                 "columns are " + str(list(crn.columns)) + ".")
            s = crn[column]
        elif crn.shape[1] == 1:
            s = crn.iloc[:, 0]
        else:
            lower = {str(c).lower(): c for c in crn.columns}
            pick = next((lower[k] for k in _CRN_PREFER if k in lower), None)
            if pick is None:
                raise ValueError("chronology has multiple columns "
                                 + str(list(crn.columns)) + "; pass column= to "
                                 "choose the index series.")
            s = crn[pick]
    else:
        raise TypeError("crn must be a year-indexed pandas Series or DataFrame, "
                        "not " + str(type(crn)) + ".")
    s = pd.Series(np.asarray(s, dtype=float), index=np.asarray(s.index).astype(int))
    s.index.name = "year"
    return s


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #

def seascorr(crn, primary, secondary, end_month=9, season_lengths=(1, 3, 6, 12),
             primary_agg="sum", secondary_agg="mean", nsim=1000, period=None,
             seed=None, column=None, plot=True, primary_name="primary",
             secondary_name="secondary", primary_color=None, secondary_color=None):
    """Seasonal correlation of a tree-ring chronology with monthly climate.

    A faithful port of SEASCORR (Meko, Touchan & Anchukaitis 2011). The
    chronology is correlated (simple Pearson) with a primary climate variable
    and (partial, controlling for the primary) with a secondary variable, across
    season-lengths x 14 ending months. Significance is assessed by exact
    spectral simulation of the chronology.

    Parameters
    ----------
    crn : Series or DataFrame
        Year-indexed chronology (standard or residual). If a DataFrame, pass
        ``column=`` to choose the index series.
    primary, secondary : DataFrame or Series
        One monthly climate variable each, in any of three shapes: wide
        (year index/column + 12 month columns), tidy (year, month, value), or a
        monthly datetime/period-indexed series.
    end_month : int
        Last month of the growth year (1=Jan ... 12=Dec).
    season_lengths : sequence of int
        Season lengths in months; the first must be 1.
    primary_agg, secondary_agg : {"sum", "mean"}
        How each variable is aggregated over a season (sum for totals such as
        precipitation, mean for temperature).
    nsim : int
        Number of exact simulations; must be >= 1000.
    period : None, (first, last), or ((e0, e1), (l0, l1))
        Analysis window. A single pair sets the window; a pair of pairs sets an
        early and a late window and triggers the stability test.
    seed : int or None
        Seed for the surrogate random generator (reproducible significance).
    column : str or None
        Chronology column to use when ``crn`` is a multi-column DataFrame.
    plot : bool
        Draw the seasonal bar chart (Figure 3 of the paper).

    Returns
    -------
    SeascorrResult
    """
    primary_agg = _normalize_agg(primary_agg, "primary_agg")
    secondary_agg = _normalize_agg(secondary_agg, "secondary_agg")
    season_lengths = tuple(int(m) for m in season_lengths)
    if not season_lengths or season_lengths[0] != 1:
        raise ValueError("season_lengths must start with 1 (so single-month "
                         "correlations are available); got " + str(season_lengths) + ".")
    if int(end_month) not in range(1, 13):
        raise ValueError("end_month must be 1..12, got " + repr(end_month) + ".")
    end_month = int(end_month)
    if int(nsim) < _MIN_NSIM:
        raise ValueError("nsim must be at least %d (fewer simulations cannot "
                         "resolve the 0.01 significance level); got %r."
                         % (_MIN_NSIM, nsim))
    nsim = int(nsim)

    x = _as_chronology(crn, column)
    p3 = _to_month3(primary, "primary")
    t3 = _to_month3(secondary, "secondary")

    U_all, meta = _seasonalize(p3, end_month, season_lengths, primary_agg)
    V_all, _ = _seasonalize(t3, end_month, season_lengths, secondary_agg)

    # analysis window(s)
    single, early, late = _parse_period(period)

    full = _resolve_window(x, U_all, V_all, single)
    res = _run_one(x, U_all, V_all, meta, full, nsim, seed, season_lengths,
                   end_month)

    if early is not None:
        res.stability = _stability_test(x, U_all, res, early, late)

    if plot:
        from .seascorr_plot import _plot_seascorr      # local import (matplotlib)
        res.fig, res.axes = _plot_seascorr(
            res, primary_name=primary_name, secondary_name=secondary_name,
            primary_color=primary_color, secondary_color=secondary_color)

    return res


def _normalize_agg(agg, param):
    a = str(agg).strip().lower()
    if a not in ("sum", "mean"):
        raise ValueError(param + ' must be "sum" or "mean", got ' + repr(agg) + ".")
    return a


def _parse_period(period):
    """Return (single, early, late). ``single`` is a (first,last) window or None;
    ``early``/``late`` are set (and single None) when a pair of windows is given."""
    if period is None:
        return None, None, None
    seq = list(period)
    if len(seq) == 2 and np.isscalar(seq[0]) and np.isscalar(seq[1]):
        return (int(seq[0]), int(seq[1])), None, None
    if len(seq) == 2 and hasattr(seq[0], "__len__") and hasattr(seq[1], "__len__"):
        e = (int(seq[0][0]), int(seq[0][1]))
        l = (int(seq[1][0]), int(seq[1][1]))
        return None, e, l
    raise ValueError("period must be None, (first, last), or "
                     "((early0, early1), (late0, late1)); got " + repr(period) + ".")


def _resolve_window(x, U_all, V_all, single):
    """Pick the analysis window: the longest unbroken overlap of chronology and
    both climate matrices, optionally clipped to ``single``."""
    # longest unbroken block of each climate matrix
    ug0, ug1 = _longest_unbroken(U_all)
    vg0, vg1 = _longest_unbroken(V_all)
    if ug0 is None or vg0 is None:
        raise ValueError("climate data has no gap-free block of seasons; fill "
                         "missing months before calling seascorr.")
    xg = x.dropna()
    lo = max(ug0, vg0, int(xg.index.min()))
    hi = min(ug1, vg1, int(xg.index.max()))
    if single is not None:
        lo = max(lo, single[0])
        hi = min(hi, single[1])
    if hi - lo + 1 < _MIN_YEARS:
        raise ValueError("analysis window %d-%d has fewer than %d years; "
                         "seascorr needs a longer overlap (the spectral "
                         "simulation requires it)." % (lo, hi, _MIN_YEARS))
    # chronology must be gap-free across the window
    xs = x.loc[lo:hi]
    if xs.isna().any():
        bad = [int(y) for y in xs.index[xs.isna()]][:8]
        raise ValueError("chronology has missing values in the analysis window "
                         "(e.g. years " + str(bad) + "); fill them (see "
                         "fill_internal) or restrict period=.")
    return lo, hi


def _run_one(x, U_all, V_all, meta, window, nsim, seed, season_lengths, end_month):
    lo, hi = window
    years = np.arange(lo, hi + 1)
    xv = x.loc[lo:hi].to_numpy(dtype=float)
    U = U_all.loc[lo:hi].to_numpy(dtype=float)
    V = V_all.loc[lo:hi].to_numpy(dtype=float)

    # observed correlations
    r_xU = _corr_matrix(xv[:, None], U).ravel()        # x vs primary seasons
    r_xV = _corr_matrix(xv[:, None], V).ravel()        # x vs secondary seasons
    r_UV = _corr_paired(U, V)                          # primary vs secondary
    r1 = r_xU
    r2 = _partial(r_xU, r_xV, r_UV)

    # exact simulation significance
    rng = np.random.default_rng(seed)
    Y = _pdgmsim(xv, nsim, rng)                        # (n, nsim)
    # p1: simple correlation of each surrogate with each primary season
    Rsim_U = _corr_matrix(Y, U)                        # (nsim, 56)
    Rsim_V = _corr_matrix(Y, V)                        # (nsim, 56)
    p1 = np.array([_weibull_p(r1[k], Rsim_U[:, k]) for k in range(len(r1))])
    # p2: partial correlation of each surrogate with secondary, controlling
    #     primary -- closed form with the fixed primary/secondary correlation
    Rsim_part = _partial(Rsim_U, Rsim_V, r_UV[None, :])
    p2 = np.array([_weibull_p(r2[k], Rsim_part[:, k]) for k in range(len(r2))])

    table = pd.DataFrame({
        "season_length": [m["season_length"] for m in meta],
        "end_month": [m["end_month"] for m in meta],
        "label": [m["label"] for m in meta],
        "r_primary": r1,
        "p_primary": p1,
        "r_partial": r2,
        "p_partial": p2,
        "r_clim": r_UV,
    })
    table["sig05_primary"] = (table["p_primary"] < 0.025) | (table["p_primary"] > 0.975)
    table["sig01_primary"] = (table["p_primary"] < 0.005) | (table["p_primary"] > 0.995)
    table["sig05_partial"] = (table["p_partial"] < 0.025) | (table["p_partial"] > 0.975)
    table["sig01_partial"] = (table["p_partial"] < 0.005) | (table["p_partial"] > 0.995)

    # strongest primary season per season-length (season_idx = column position
    # in the 56-season layout, so the stability test can recover the series)
    ks = []
    for m in season_lengths:
        sub = table[table["season_length"] == m]
        ks.append(int(sub["r_primary"].abs().idxmax()))
    highest = table.loc[ks].reset_index(drop=True)
    highest.insert(0, "season_idx", ks)

    acf = _acf(xv, 5)
    return SeascorrResult(table, (lo, hi), nsim, end_month, season_lengths,
                          highest, acf)


def _acf(x, nlags):
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.sum(x * x)
    return np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


_MIN_SUB = 15      # minimum years in a sub-period (Meko's minlen_sub)


def _stability_test(x, U_all, full_res, early, late):
    """Test whether the primary-variable correlation is stable between an early
    and a late sub-period, for the strongest season of each season-length.

    Port of Meko's subfun17: lag-1 autocorrelations and the full-period
    correlation are computed over the full analysis window; the early and late
    correlations over their own windows; sample sizes are reduced to effective
    sizes (Dawdy & Matalas 1964) using the full-period autocorrelations; and the
    early-vs-late difference is tested with Fisher's z (Snedecor & Cochran 1989).
    p-values are floored at 0.001. Returns a tidy DataFrame, one row per season.
    """
    lo, hi = full_res.period
    e0, e1 = early
    l0, l1 = late
    # validate the two windows
    if not (lo <= e0 <= e1 <= hi) or not (lo <= l0 <= l1 <= hi):
        raise ValueError("early/late windows must lie within the analysis period "
                         "%d-%d; got early=%s late=%s." % (lo, hi, early, late))
    if e1 >= l0:
        raise ValueError("early and late sub-periods must not overlap and early "
                         "must come first; got early=%s late=%s." % (early, late))
    for w in (early, late):
        if w[1] - w[0] + 1 < _MIN_SUB:
            raise ValueError("each sub-period must be at least %d years; got %s."
                             % (_MIN_SUB, w))

    xv_full = x.loc[lo:hi].to_numpy(dtype=float)
    r1_x = _lag1(xv_full)

    rows = []
    for _, h in full_res.highest.iterrows():
        i = int(h["season_idx"])
        P = U_all.iloc[:, i]
        p_full = P.loc[lo:hi].to_numpy(dtype=float)
        r1_P = _lag1(p_full)

        # effective-sample-size factor from full-period lag-1 autocorrelations;
        # no adjustment if either series is not positively autocorrelated
        if r1_x <= 0 or r1_P <= 0 or np.isnan(r1_x) or np.isnan(r1_P):
            f = 1.0
        else:
            f = (1.0 - r1_x * r1_P) / (1.0 + r1_x * r1_P)

        r_full = _corr_paired(p_full[:, None], xv_full[:, None])[0]
        xe = x.loc[e0:e1].to_numpy(dtype=float)
        pe = P.loc[e0:e1].to_numpy(dtype=float)
        xl = x.loc[l0:l1].to_numpy(dtype=float)
        pl = P.loc[l0:l1].to_numpy(dtype=float)
        r_e = _corr_paired(pe[:, None], xe[:, None])[0]
        r_l = _corr_paired(pl[:, None], xl[:, None])[0]
        n_e, n_l = xe.size, xl.size
        ne, nl = _eff_n(n_e, f), _eff_n(n_l, f)

        _, _, dz, p = _corrdiff(r_e, r_l, ne, nl)
        if not np.isnan(p):
            p = max(p, 0.001)

        rows.append({
            "season_length": int(h["season_length"]),
            "label": h["label"],
            "r_full": r_full,
            "r_early": r_e,
            "r_late": r_l,
            "n_early": n_e,
            "n_late": n_l,
            "n_eff_early": ne,
            "n_eff_late": nl,
            "delta_z": dz,
            "p_value": p,
        })
    out = pd.DataFrame(rows)
    out.attrs["periods"] = {"full": (lo, hi), "early": early, "late": late}
    return out

