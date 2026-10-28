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

# Title: climate_response.py
# Project: OpenDendro dplPy
# Description: Bootstrapped correlation and response functions -- the monthly
#              climate-signal screen of Fritts (1971) and DENDROCLIM2002 (Biondi
#              & Waikul 2004), as popularized by the R package treeclim (Zang &
#              Biondi 2015). A chronology is related to each month of one or more
#              monthly climate variables over a prior-to-current-year window,
#              either by simple correlation (Pearson, per month) or by a response
#              function (regression on the principal components of the monthly
#              climate matrix, which removes the collinearity between adjacent
#              months). Significance comes from bootstrapping. This is the second
#              function in dplPy's climate module, after seascorr, and reuses its
#              machinery (monthly reshaping, exact spectral simulation, Weibull
#              significance) from climate.py.
#
# References:
#   Fritts HC, Blasing TJ, Hayden BP, Kutzbach JE (1971) Multivariate techniques
#     for specifying tree-growth and climate relationships ... J. Appl. Meteorol.
#     10:845-864.
#   Guiot J (1991) The bootstrapped response function. Tree-Ring Bull. 51:39-41.
#   Biondi F, Waikul K (2004) DENDROCLIM2002 ... Comput. Geosci. 30:303-311.
#   Zang C, Biondi F (2015) treeclim: an R package for the numerical calibration
#     of proxy-climate relationships. Ecography 38:431-436.

import numpy as np
import pandas as pd

from .climate import (_as_chronology, _to_month3, _zcols, _pdgmsim, _weibull_p,
                      _MONTH_ABBR, _MIN_YEARS)

# A response function needs at least this many predictors to be worth doing (the
# PCA/regression is pointless with one), matching treeclim's "needs at least 2".
_MIN_PREDICTORS = 2
_MIN_NSIM = 1000            # as in seascorr: fewer cannot resolve the 0.01 level
_MIN_STATIONARY = 18       # treeclim's floor for estimating a block length


# --------------------------------------------------------------------------- #
# Signed month-id window (negative = previous year, positive = current year)
# --------------------------------------------------------------------------- #
#
# Month ids follow the dendroclimatological convention also used by treeclim:
#   1..12    -> current growth year, January..December
#   -1..-12  -> previous year,       January..December   (-9 = previous September)
#   -13..-24 -> year before previous, January..December
# A window is the contiguous chronological span from ``start_month`` to
# ``end_month``. To make "contiguous" unambiguous we map each id to an absolute
# month counter (previous December = -1, current January = 0), walk the integer
# range, and map back. So start_month=-6, end_month=9 gives prev Jun .. curr Sep.

def _id_to_abs(mid):
    """Signed month id -> absolute month counter (current January = 0)."""
    if 1 <= mid <= 12:
        return mid - 1                      # current year: 0..11
    if -12 <= mid <= -1:
        return -13 - mid                    # previous year: -12..-1
    if -24 <= mid <= -13:
        return -37 - mid                    # year before previous: -24..-13
    raise ValueError("month id %d is outside -24..-1, 1..12 (0 is not a month; "
                     "previous-year months are negative, current-year positive)."
                     % mid)


def _abs_to_id(a):
    """Absolute month counter -> signed month id (inverse of _id_to_abs)."""
    if 0 <= a <= 11:
        return a + 1
    if -12 <= a <= -1:
        return -(a + 13)
    if -24 <= a <= -13:
        return -(a + 37)
    raise ValueError("absolute month %d is out of range." % a)


def _window_ids(start_month, end_month):
    """Ordered list of signed month ids for the chronological span
    start_month..end_month (inclusive)."""
    a0, a1 = _id_to_abs(int(start_month)), _id_to_abs(int(end_month))
    if a0 > a1:
        raise ValueError("start_month (%d) must come chronologically before "
                         "end_month (%d). Previous-year months are negative "
                         "(e.g. start_month=-6 is previous June), current-year "
                         "months positive (e.g. end_month=9 is current "
                         "September)." % (start_month, end_month))
    return [_abs_to_id(a) for a in range(a0, a1 + 1)]


def _id_year_offset(mid):
    """Calendar-year offset of a month id relative to the growth year."""
    if mid >= 1:
        return 0
    if mid >= -12:
        return -1
    return -2


def _id_calendar_month(mid):
    """Calendar month (1..12) of a signed month id."""
    if mid >= 1:
        return mid
    if mid >= -12:
        return -mid
    return -mid - 12


_YEAR_NAME = {0: "current", -1: "previous", -2: "before previous"}


def _id_label(mid):
    """Plain month label, lower-case for prior years and upper-case for the
    current year (the Fritts 1971 Fig. 4 convention)."""
    abbr = _MONTH_ABBR[_id_calendar_month(mid) - 1]
    off = _id_year_offset(mid)
    if off == 0:
        return abbr.upper()
    if off == -1:
        return abbr.lower()
    return abbr.lower() + "(-2)"


# --------------------------------------------------------------------------- #
# Climate input -> design matrix of (variable, month) predictors
# --------------------------------------------------------------------------- #

def _normalize_climate(climate, var_names):
    """Return an ordered dict {name: tidy (year, month, value) frame}.

    Accepts a single climate variable (DataFrame/Series, same shapes as
    seascorr), or a dict/mapping of several, or a sequence paired with
    ``var_names``.
    """
    out = {}
    if isinstance(climate, dict):
        items = list(climate.items())
    elif isinstance(climate, (pd.DataFrame, pd.Series)):
        name = "climate"
        if var_names is not None:
            nm = var_names if isinstance(var_names, str) else list(var_names)[0]
            name = nm
        items = [(name, climate)]
    elif isinstance(climate, (list, tuple)):
        if var_names is None or len(var_names) != len(climate):
            raise ValueError("when climate is a list/tuple of variables, pass "
                             "var_names with one name per variable.")
        items = list(zip(var_names, climate))
    else:
        raise TypeError("climate must be a DataFrame, Series, or a dict/list of "
                        "them, not " + str(type(climate)) + ".")
    if not items:
        raise ValueError("no climate variables supplied.")
    for name, var in items:
        out[str(name)] = _to_month3(var, str(name))
    return out


def _build_design(clim3, ids):
    """Assemble the design matrix of (variable, month) predictors.

    Returns ``(frame, meta)`` where ``frame`` is a DataFrame indexed by growth
    year with one column per (variable, month id) and ``meta`` is the matching
    list of dicts. A growth year is kept only if every predictor it needs is
    present (so the matrix is gap-free); the caller then aligns with the
    chronology.
    """
    cols = {}
    meta = []
    for name, m3 in clim3.items():
        # dense lookup: (calendar_year, month) -> value
        lut = {(int(y), int(mo)): v
               for y, mo, v in zip(m3["year"], m3["month"], m3["value"])}
        years = np.unique(m3["year"].to_numpy())
        for mid in ids:
            yoff = _id_year_offset(mid)
            cmon = _id_calendar_month(mid)
            key = (name, mid)
            series = {}
            for Y in years:
                val = lut.get((int(Y) + yoff, cmon))
                if val is not None:
                    series[int(Y)] = val
            cols[key] = pd.Series(series, dtype=float)
            meta.append({"variable": name, "month_id": int(mid),
                         "calendar_month": cmon,
                         "year": _YEAR_NAME[yoff],
                         "label": _id_label(mid)})
    frame = pd.DataFrame(cols).sort_index()
    frame.index.name = "year"
    return frame, meta


# --------------------------------------------------------------------------- #
# Coefficients: correlation function and response function
# --------------------------------------------------------------------------- #

def _zvec(g):
    """Z-score a 1-D array (population sd, matching _zcols)."""
    g = np.asarray(g, dtype=float)
    return (g - g.mean()) / g.std()


def _corr_coef(Uz, gz):
    """Correlation function: Pearson r of each z-scored predictor with the
    z-scored chronology (= OLS slope of z on z). Returns (k,)."""
    n = Uz.shape[0]
    return (Uz.T @ gz) / n


def _response_pca(Uz, min_pcs=1):
    """Eigen-reduce the predictor correlation matrix by the PVP criterion
    (Guiot 1991): keep leading eigenvectors while the cumulative product of
    (descending) eigenvalues exceeds 1. Returns ``(Ep, Z)``: retained
    eigenvectors (k, p) and the principal-component scores (n, p).

    Retaining every component instead would make the response coefficients the
    ordinary standardized multiple-regression solution (C^-1 r); the PVP cut is
    exactly what regularizes against the near-collinearity of adjacent months.
    When the criterion would keep nothing (a near-orthogonal predictor set, where
    all eigenvalues are ~1), the leading component is kept so the fit is defined.
    """
    n, k = Uz.shape
    C = (Uz.T @ Uz) / n                      # predictor correlation matrix
    w, V = np.linalg.eigh(C)                 # ascending eigenpairs
    w = w[::-1]
    V = V[:, ::-1]                           # descending
    keep = int(np.count_nonzero(np.cumprod(w) > 1.0))
    keep = max(keep, min_pcs)                # keep at least one component
    keep = min(keep, k, max(n - 1, 1))       # never more than the fit allows
    Ep = V[:, :keep]
    Z = Uz @ Ep
    return Ep, Z


def _response_fit(Ep, Z, gz):
    """Given a fixed PCA (Ep, Z), regress the z-scored chronology on the PC
    scores and back-transform to per-predictor response coefficients (k,)."""
    beta, _res, _rank, _sv = np.linalg.lstsq(Z, gz, rcond=None)
    return Ep @ beta


def _response_coef(Uz, gz):
    """Response function from scratch (PCA then fit); for resampling bootstraps
    where the predictor matrix changes each replicate."""
    Ep, Z = _response_pca(Uz)
    return _response_fit(Ep, Z, gz)


# --------------------------------------------------------------------------- #
# Bootstrap + significance
# --------------------------------------------------------------------------- #

def _coef_once(U, g, method):
    """Point coefficients for the observed data (U, g not yet standardized)."""
    Uz = _zcols(U)
    gz = _zvec(g)
    if method == "correlation":
        return _corr_coef(Uz, gz)
    return _response_coef(Uz, gz)


def _boot_exact(U, g, method, nsim, seed):
    """Exact (Fourier-surrogate) bootstrap: the climate matrix is held fixed and
    the chronology is simulated ``nsim`` times with the same spectrum (Meko's
    pdgmsim). Returns ``(obs, sims)`` where ``obs`` is (k,) and ``sims`` is
    (k, nsim) -- a null distribution for each coefficient."""
    Uz = _zcols(U)
    rng = np.random.default_rng(seed)
    Y = _pdgmsim(g, nsim, rng)               # (n, nsim) surrogate chronologies
    Yz = _zcols(Y)
    if method == "correlation":
        obs = _corr_coef(Uz, _zvec(g))
        sims = (Uz.T @ Yz) / Uz.shape[0]     # (k, nsim)
        return obs, sims
    # response: PCA fixed (U fixed), only the fit changes per surrogate
    Ep, Z = _response_pca(Uz)
    obs = _response_fit(Ep, Z, _zvec(g))
    beta, _r, _rk, _s = np.linalg.lstsq(Z, Yz, rcond=None)   # (p, nsim)
    sims = Ep @ beta                                         # (k, nsim)
    return obs, sims


# -- automatic block length for the stationary bootstrap ------------------- #
# Politis & White (2004), "Automatic block-length selection for the dependent
# bootstrap", as corrected by Patton, Politis & White (2009). This is a port of
# np::b.star (the stationary-bootstrap column), which treeclim uses to pick the
# block length; it is what makes the stationary bootstrap parameter-free.

def _flattop(s):
    """Politis flat-top lag window: 1 on |s|<1/2, tapering linearly to 0 at 1."""
    s = np.abs(s)
    return np.where(s < 0.5, 1.0, np.where(s <= 1.0, 2.0 * (1.0 - s), 0.0))


def _acvf(x, maxlag):
    """Sample autocovariances at lags 0..maxlag (divisor n, as R's acf)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    return np.array([(x[:n - k] * x[k:]).sum() / n for k in range(maxlag + 1)])


def _b_star_sb(x):
    """Optimal stationary-bootstrap block length for series ``x`` (Politis &
    White 2004 / Patton et al. 2009; np::b.star, SB column). Returns a rounded
    integer >= 1, as treeclim uses it."""
    x = np.asarray(x, dtype=float)
    n = x.size
    kn = max(5, int(np.ceil(np.log10(n))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    bmax = np.ceil(min(3.0 * np.sqrt(n), n / 3.0))
    c = 1.959963984540054                        # qnorm(0.975)

    acv = _acvf(x, mmax)
    rho = acv[1:] / acv[0]                        # autocorrelations, lags 1..mmax
    crit = c * np.sqrt(np.log10(n) / n)
    insignif = np.abs(rho) < crit

    # smallest lag m after which Kn consecutive autocorrelations are insignificant
    mhat = None
    for j in range(mmax - kn + 1):
        if insignif[j:j + kn].sum() == kn:
            mhat = j + 1                         # 1-indexed lag
            break
    if mhat is None:
        sig = np.nonzero(~insignif)[0]
        mhat = int(sig[-1] + 1) if sig.size else 1
    M = min(2 * mhat, mmax)

    acvM = _acvf(x, M)
    kk = np.arange(-M, M + 1)
    Rk = acvM[np.abs(kk)]
    w = _flattop(kk / M)
    Ghat = np.sum(w * np.abs(kk) * Rk)
    g0 = np.sum(w * Rk)
    DSB = 2.0 * g0 ** 2
    if DSB <= 0:                                  # degenerate (white noise)
        return 1
    bstar = (2.0 * Ghat ** 2 / DSB) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    bstar = min(bstar, bmax)
    return max(int(round(bstar)), 1)


def _stationary_indices(n, block, rng):
    """Stationary-bootstrap resampling indices (Politis & Romano 1994): blocks of
    geometric length with mean ``block``, wrapped circularly. Equivalent to
    boot::tsboot(sim="geom", l=block), which treeclim uses."""
    p = 1.0 / block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    starts = rng.random(n) < p
    steps = rng.integers(0, n, n)
    for t in range(1, n):
        idx[t] = steps[t] if starts[t] else (idx[t - 1] + 1) % n
    return idx


def _boot_resample(U, g, method, nboot, seed, block=None):
    """Resampling bootstrap: resample years with replacement (chronology and
    climate together), refit each replicate. ``block=None`` is the classic iid
    bootstrap; a positive ``block`` uses the stationary (geometric-block)
    bootstrap, which preserves the series' autocorrelation. Returns ``(obs,
    reps)`` where ``reps`` is (k, nboot) -- a distribution *around* each
    coefficient."""
    obs = _coef_once(U, g, method)
    n, k = U.shape
    rng = np.random.default_rng(seed)
    reps = np.empty((k, nboot))
    for j in range(nboot):
        if block is None:
            idx = rng.integers(0, n, n)
        else:
            idx = _stationary_indices(n, block, rng)
        reps[:, j] = _coef_once(U[idx], g[idx], method)
    return obs, reps


def _sig_exact(obs, sims, ci):
    """Two-sided significance from the null surrogate distribution via the
    Weibull non-exceedance probability (as in seascorr); no CI is defined for
    the exact method. Returns (coef, p, significant, ci_lower, ci_upper)."""
    k = obs.size
    p = np.array([_weibull_p(obs[i], sims[i, :]) for i in range(k)])
    sig = (p < ci / 2.0) | (p > 1.0 - ci / 2.0)
    nan = np.full(k, np.nan)
    return obs, p, sig, nan, nan


def _sig_percentile(obs, reps, ci):
    """Significance from the resampling bootstrap by the percentile method
    (Dixon 2001): the (1-ci) percentile interval; significant when it excludes
    zero. Returns (coef, p, significant, ci_lower, ci_upper)."""
    lo = np.percentile(reps, 100.0 * (ci / 2.0), axis=1)
    hi = np.percentile(reps, 100.0 * (1.0 - ci / 2.0), axis=1)
    sig = np.sign(lo) == np.sign(hi)         # interval does not bracket zero
    p = np.full(obs.size, np.nan)            # the percentile method yields no p
    return obs, p, sig, lo, hi


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #

class ClimateResponseResult:
    """Result of :func:`climate_response`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per (variable, month) predictor, in chronological month order:
        ``id, variable, month_id, year, month, coef, ci_lower, ci_upper,
        significant, p``. ``ci_lower``/``ci_upper`` are NaN for the exact
        bootstrap (which yields a significance test, not a confidence interval);
        ``p`` is NaN for the resampling bootstrap (whose test is the confidence
        interval). For ``boot="none"`` both are NaN.
    method : {"response", "correlation"}
    boot : {"exact", "std", "stationary", "none"}
    period : tuple(int, int)
        First and last growth year of the analysis window.
    settings : dict
        start_month, end_month, ci, nboot, variables, and block_length (the
        stationary-bootstrap block length, or None for other methods).
    """

    def __init__(self, table, method, boot, period, settings):
        self.table = table
        self.method = method
        self.boot = boot
        self.period = period
        self.settings = settings
        self.fig = None
        self.axes = None

    def __repr__(self):
        a, b = self.period
        nsig = int(self.table["significant"].sum())
        return ("ClimateResponseResult(method=%s, boot=%s, period=%d-%d, "
                "%d predictors, %d significant)"
                % (self.method, self.boot, a, b, len(self.table), nsig))

    def summary(self):
        """A compact text summary: the significant predictors, strongest first."""
        a, b = self.period
        sig = self.table[self.table["significant"]].copy()
        lines = ["%s function (%s bootstrap), %d-%d, %d of %d predictors "
                 "significant at the %.0f%% level:"
                 % (self.method, self.boot, a, b, len(sig), len(self.table),
                    100 * (1 - self.settings["ci"]))]
        if len(sig) == 0:
            lines.append("  (none)")
        else:
            sig = sig.reindex(sig["coef"].abs().sort_values(ascending=False).index)
            for _, r in sig.iterrows():
                lines.append("  %-4s %-14s coef=%+.3f"
                             % (r["month"], r["variable"], r["coef"]))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #

def climate_response(crn, climate, start_month=-6, end_month=9,
                     method="response", boot="exact", ci=0.05, nboot=1000,
                     period=None, seed=None, column=None, var_names=None,
                     plot=True):
    """Bootstrapped correlation or response function of a chronology vs monthly
    climate.

    Relates a tree-ring chronology to the individual months of one or more
    monthly climate variables over a window running from ``start_month`` to
    ``end_month`` (see the sign convention below). Two methods:

    * ``"correlation"`` -- the Pearson correlation of the chronology with each
      month separately (DENDROCLIM2002-style correlation function).
    * ``"response"`` -- a regression of the chronology on the principal
      components of the whole monthly-climate matrix, back-transformed to one
      coefficient per month (Fritts 1971; Guiot 1991). The PCA removes the
      collinearity between neighbouring months before attributing signal; PCs are
      retained by the PVP criterion (cumulative product of eigenvalues > 1).

    Significance is bootstrapped. ``boot="exact"`` (the default) holds the
    climate fixed and simulates the chronology 1000x with the same spectrum
    (Meko's exact method, as in seascorr), testing each coefficient against that
    null by a Weibull non-exceedance probability. ``boot="std"`` resamples years
    with replacement (the classic DENDROCLIM2002 bootstrap) and reports a
    percentile confidence interval, calling a coefficient significant when the
    interval excludes zero. ``boot="stationary"`` is the same but uses the
    stationary (geometric-block) bootstrap of Politis & Romano (1994), which
    preserves the series' autocorrelation; the block length is chosen
    automatically by the Politis & White (2004) rule (as in treeclim), and is
    reported in ``result.settings["block_length"]``. ``boot="none"`` returns
    point estimates only.

    Month sign convention (as used by seascorr's ``end_month`` and by treeclim):
    positive months are the current growth year (1=Jan .. 12=Dec) and negative
    months the previous year (-1=Jan .. -12=Dec), so ``start_month=-6,
    end_month=9`` is the window from previous June to current September, and
    ``start_month=-9`` would be previous September. Months -13..-24 reach into
    the year before previous.

    Parameters
    ----------
    crn : Series or DataFrame
        Year-indexed chronology. If a DataFrame, pass ``column=`` to choose the
        index series (same rules as seascorr).
    climate : DataFrame, Series, or dict/list of them
        One or more monthly climate variables, each in any shape ``seascorr``
        accepts (wide year x 12 months, tidy year/month/value, or a monthly
        datetime-indexed series). A dict keys the variables by name; a bare frame
        is named from ``var_names`` or "climate".
    start_month, end_month : int
        The month window (signed convention above). Defaults -6 (previous June)
        to 9 (current September).
    method : {"response", "correlation"}
    boot : {"exact", "std", "stationary", "none"}
    ci : float
        Test level; one of 0.1, 0.05, 0.01. The confidence interval (for the
        resampling bootstraps) is set accordingly.
    nboot : int
        Number of bootstrap replicates / simulations (>= 1000).
    period : None or (first, last)
        Restrict the analysis to these growth years; default is the longest
        common gap-free window.
    seed : int or None
        Seed for the bootstrap random generator (reproducible significance).
    column : str or None
        Chronology column when ``crn`` is a multi-column DataFrame.
    var_names : str, sequence, or None
        Names for the climate variable(s) when not supplied as a dict.
    plot : bool
        Draw the coefficient figure.

    Returns
    -------
    ClimateResponseResult

    Notes
    -----
    PC retention currently uses the PVP criterion only; Fritts' original
    percent-variance / stepwise selection is a planned option (dplPy does a lot
    that traces back to Fritts, and this is worth revisiting).
    """
    method = str(method).strip().lower()
    if method not in ("response", "correlation"):
        raise ValueError('method must be "response" or "correlation", got '
                         + repr(method) + ".")
    boot = str(boot).strip().lower()
    if boot not in ("exact", "std", "stationary", "none"):
        raise ValueError('boot must be "exact", "std", "stationary", or "none", '
                         "got " + repr(boot) + ".")
    if ci not in (0.1, 0.05, 0.01):
        raise ValueError("ci must be one of 0.1, 0.05, 0.01; got %r." % (ci,))
    nboot = int(nboot)
    if boot != "none" and nboot < _MIN_NSIM:
        raise ValueError("nboot must be at least %d; got %d." % (_MIN_NSIM, nboot))

    ids = _window_ids(start_month, end_month)

    x = _as_chronology(crn, column)
    clim3 = _normalize_climate(climate, var_names)
    design, meta = _build_design(clim3, ids)

    if design.shape[1] < _MIN_PREDICTORS:
        raise ValueError("the window yields only %d predictor(s); a response or "
                         "correlation function needs at least %d. Widen "
                         "start_month..end_month or add a climate variable."
                         % (design.shape[1], _MIN_PREDICTORS))

    # longest common gap-free window of chronology and every predictor
    xg = x.dropna()
    good = design.dropna()
    common = good.index.intersection(xg.index)
    if period is not None:
        lo, hi = int(period[0]), int(period[1])
        common = common[(common >= lo) & (common <= hi)]
    if common.empty:
        raise ValueError("chronology and climate share no complete overlapping "
                         "years for this window.")
    lo, hi = int(common.min()), int(common.max())
    years = np.arange(lo, hi + 1)
    # require the span to be contiguous and gap-free in both
    span = design.loc[lo:hi]
    xb = x.loc[lo:hi]
    if span.isna().any().any() or xb.isna().any():
        # fall back to the explicit common set if the span has interior gaps
        years = np.array(sorted(common))
    n = len(years)
    if boot == "exact" and n < _MIN_YEARS:
        raise ValueError("the analysis window %d-%d has %d years; the exact "
                         "bootstrap needs at least %d (the spectral simulation "
                         "requires it). Use boot=\"std\" or a longer overlap."
                         % (lo, hi, n, _MIN_YEARS))
    if boot == "stationary" and n < _MIN_STATIONARY:
        raise ValueError("the analysis window %d-%d has %d years; the stationary "
                         "bootstrap needs at least %d to estimate a block length. "
                         "Use boot=\"std\" or a longer overlap."
                         % (lo, hi, n, _MIN_STATIONARY))
    if n <= design.shape[1]:
        raise ValueError("the window has %d years but %d predictors; need more "
                         "years than predictors. Shorten the month window or "
                         "restrict to fewer variables." % (n, design.shape[1]))

    U = design.loc[years].to_numpy(dtype=float)
    g = x.loc[years].to_numpy(dtype=float)

    block_length = None
    if boot == "exact":
        obs, sims = _boot_exact(U, g, method, nboot, seed)
        coef, p, sig, cl, cu = _sig_exact(obs, sims, ci)
    elif boot == "std":
        obs, reps = _boot_resample(U, g, method, nboot, seed)
        coef, p, sig, cl, cu = _sig_percentile(obs, reps, ci)
    elif boot == "stationary":
        block_length = _b_star_sb(g)             # Politis-White automatic length
        obs, reps = _boot_resample(U, g, method, nboot, seed, block=block_length)
        coef, p, sig, cl, cu = _sig_percentile(obs, reps, ci)
    else:
        coef = _coef_once(U, g, method)
        k = coef.size
        p = np.full(k, np.nan)
        sig = np.full(k, False)
        cl = cu = np.full(k, np.nan)

    table = pd.DataFrame({
        "id": np.arange(1, len(meta) + 1),
        "variable": [m["variable"] for m in meta],
        "month_id": [m["month_id"] for m in meta],
        "year": [m["year"] for m in meta],
        "month": [m["label"] for m in meta],
        "coef": coef,
        "ci_lower": cl,
        "ci_upper": cu,
        "significant": sig,
        "p": p,
    })

    settings = {"start_month": int(start_month), "end_month": int(end_month),
                "ci": float(ci), "nboot": nboot,
                "variables": list(clim3.keys()), "block_length": block_length}
    res = ClimateResponseResult(table, method, boot, (int(years[0]),
                                int(years[-1])), settings)

    if plot:
        from .climate_response_plot import _plot_climate_response
        res.fig, res.axes = _plot_climate_response(res)

    return res
