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

# Title: pcreg.py
# Project: OpenDendro dplPy
# Description: Orthogonal spatial regression / principal-components regression
#              climate reconstruction -- a faithful Python port of Ed Cook's
#              OSR FORTRAN (pcregOSXerc2), the method described in Cook, Briffa
#              & Jones (1994, Int. J. Climatol. 14, 379-402). A set of tree-ring
#              predictors is screened against a climate target, reduced to
#              principal components, and the target is regressed on a subset of
#              those PCs; the fit is back-transformed to per-predictor weights,
#              a reconstruction is produced over the full predictor period, and
#              it is verified against withheld data. This is the second function
#              in dplPy's climate-response module (after seascorr).
#
#              Parity: validated number-for-number against the compiled FORTRAN
#              "oracle" on a single-target fixture (Khangai, 101.25E/48.75N;
#              calibration 1965-1989, verification 1959-1964). See
#              dev/pcreg_DECISIONS.md for every quirk that is preserved on
#              purpose. Where a comment says "parity:" it is reproducing the
#              FORTRAN exactly, not choosing what is statistically ideal.

import numpy as np
import pandas as pd
from scipy import stats

from ._validate import _warn

# --------------------------------------------------------------------------- #
# A note for readers coming from MATLAB / FORTRAN
# --------------------------------------------------------------------------- #
# Three Python/NumPy conventions trip up a direct reading of the FORTRAN:
#
#  1. ``ddof``. NumPy's ``std``/``var`` default to the *population* estimator
#     (divide by N) only if you pass ``ddof=0``; the *sample* estimator (divide
#     by N-1, MATLAB's default for ``std``) is ``ddof=1``. Cook's ``cross``
#     routine standardizes with the population SD (divide by N). We therefore
#     pass ``ddof=0`` wherever we reproduce that standardization. Getting this
#     wrong scales every standardized series by sqrt((N-1)/N) ~ 0.98, which
#     cancels in a correlation (so r still matches) but quietly throws off RE
#     and CE. This was the single subtlest parity bug in the port.
#
#  2. Zero-based indexing. FORTRAN arrays start at 1; NumPy at 0. "The first m
#     eigenvectors" is ``[:m]`` here, and "PC number j" in output is ``j+1``.
#
#  3. ``numpy.linalg.eigh`` returns eigenvalues in *ascending* order. Cook works
#     in descending order (largest eigenvalue = leading PC), so we reverse both
#     the eigenvalues and the eigenvector columns right after the call.

_MIN_CAL_YEARS = 10      # fewer than this and the regression is not meaningful
_TINY = 1e-12


# Significance throughout uses SciPy's exact distributions rather than Cook's
# Wilson-Hilferty ``prob`` approximation. Ed wrote that approximation (and the
# Numerical Recipes rank routines) because the language and era required it; the
# exact distributions are more accurate -- most visibly at low degrees of freedom
# (a short verification window) and right at a screening threshold, where a
# close-call predictor can fall on the other side of alpha than Cook's code put
# it. Exact reproduction of Ed's selection is not the goal; correctness is.


# --------------------------------------------------------------------------- #
# Step 1 -- screen predictors by correlation with the target
# --------------------------------------------------------------------------- #

def _screen(P, y, cal, alpha, tails, sign, min_coverage):
    """Keep predictors whose calibration correlation with the target is
    significant at ``alpha`` (port of the ``screen`` significance test).

    Parameters
    ----------
    P : DataFrame
        Year-indexed predictor matrix (one column per chronology).
    y : Series
        Year-indexed target.
    cal : ndarray of int
        Calibration years.
    alpha : float
        Significance level (0 < alpha <= 1). Set to 1.0 with ``tails=2`` to
        force every predictor into the PCA (Cook's "normal response function").
    tails : {1, 2}
        One- or two-tailed test.
    sign : {'+', '-', None}
        Required sign of the correlation for a one-tailed test.
    min_coverage : int
        Minimum number of overlapping finite year-pairs required to consider a
        predictor at all.

    Returns
    -------
    (kept, info, dropped) : (list, DataFrame, list)
        ``kept`` preserves column order; ``info`` carries r, t and p per kept
        predictor; ``dropped`` lists the columns excluded for having fewer than
        ``min_coverage`` overlapping years (so the caller can report them).
    """
    yc = y.reindex(cal).to_numpy()
    kept = []
    rs, ts, ps = [], [], []
    dropped = []                               # excluded for insufficient coverage
    for col in P.columns:
        xc = P[col].reindex(cal).to_numpy()
        m = np.isfinite(xc) & np.isfinite(yc)
        n = int(m.sum())
        if n < min_coverage:
            # The predictor does not span the full (lag-reached) calibration
            # window, so it never reaches the significance test. Unlike the
            # FORTRAN, dplPy does not pad the gap -- it reports and drops it.
            dropped.append(col)
            continue
        rdf = n - 2
        if rdf <= 0:
            continue
        r = np.corrcoef(xc[m], yc[m])[0, 1]
        if not np.isfinite(r):
            continue
        # Correlation t-statistic, tested against SciPy's exact t distribution.
        denom = (1.0 - r * r) / rdf
        if denom <= 0.0:
            t = np.inf * np.sign(r) if r != 0 else 0.0
            p = 0.0 if r != 0 else 1.0
        else:
            t = r / np.sqrt(denom)
            p = 2.0 * float(stats.t.sf(abs(t), rdf))   # two-tailed
        if tails == 1:
            p = p / 2.0                 # one-tailed: half the two-tailed p
            if sign == '+' and r <= 0:
                continue
            if sign == '-' and r >= 0:
                continue
        if p > alpha:
            continue
        kept.append(col)
        rs.append(r)
        ts.append(t)
        ps.append(p)
    info = pd.DataFrame({"r": rs, "t": ts, "p": ps}, index=pd.Index(kept, name="predictor"))
    return kept, info, dropped


# --------------------------------------------------------------------------- #
# Step 2 -- principal components of the screened predictors
# --------------------------------------------------------------------------- #

def _pca(Pstd_cal, criterion, n_pc, ev_cutoff, cum_pct):
    """Eigen-decompose the screened predictors and choose how many PCs to keep.

    Parameters
    ----------
    Pstd_cal : ndarray (n_cal, n_pred)
        Screened predictors, standardized by their calibration mean/SD, over the
        calibration years.
    criterion : {'first_m', 'kaiser', 'cum_pct', 'pvp', 'all'}
        Truncation rule (Cook's eigenvalue cutoff options 0, 1, 2, 3, 4).
    n_pc : int
        Number to keep for ``criterion='first_m'``.
    ev_cutoff : float
        Rescaled-eigenvalue cutoff for ``criterion='kaiser'`` (classic Kaiser
        rule is 1.0).
    cum_pct : float
        Cumulative-variance target in percent for ``criterion='cum_pct'``.

    Returns
    -------
    dict with keys
        eigvals : original eigenvalues (descending)
        eigvecs : eigenvectors as columns (descending), sign-aligned
        rlam    : rescaled eigenvalues (eigval / mean of the first npeig)
        neig    : number of PCs retained
    """
    # parity: correlation matrix of the standardized predictors over the
    # calibration window (not the full period). For standardized columns this is
    # the same as their covariance, but we form it explicitly to match Cook.
    R = np.corrcoef(Pstd_cal, rowvar=False)

    # eigh is for symmetric matrices and returns ASCENDING eigenvalues; reverse
    # to Cook's descending order (leading PC first).
    w, V = np.linalg.eigh(R)
    w = w[::-1]
    V = V[:, ::-1]

    n_pred = Pstd_cal.shape[1]
    n_cal = Pstd_cal.shape[0]
    npeig = min(n_pred, n_cal)                 # parity: usable eigenvalue count
    avgev = w[:npeig].mean()                   # parity: mean over the first npeig
    rlam = w[:npeig] / avgev                   # rescaled eigenvalues (Kaiser/PVP)

    if criterion == "first_m":
        neig = int(n_pc)
    elif criterion == "kaiser":
        # keep leading PCs whose rescaled eigenvalue exceeds the cutoff
        neig = int(np.sum(rlam > ev_cutoff))
    elif criterion == "cum_pct":
        frac = np.cumsum(w[:npeig]) / np.sum(w[:npeig]) * 100.0
        neig = int(np.searchsorted(frac, cum_pct) + 1)
    elif criterion == "pvp":
        # Guiot's product rule: stop at the first PC where the running product
        # of rescaled eigenvalues drops below 1.
        neig = npeig
        prod = 1.0
        for i in range(npeig):
            prod *= rlam[i]
            if prod < 1.0:
                neig = i            # 0-based count = index of the first <1
                break
    elif criterion == "all":
        neig = npeig
    else:
        raise ValueError("unknown pc_criterion %r" % (criterion,))

    neig = max(1, min(neig, npeig))

    # Eigenvector signs are arbitrary out of eigh. Fix a deterministic
    # convention (largest-magnitude loading positive) so scores/weights are
    # reproducible run to run; the sign choice does not affect the fit.
    for j in range(V.shape[1]):
        k = np.argmax(np.abs(V[:, j]))
        if V[k, j] < 0:
            V[:, j] = -V[:, j]

    return {"eigvals": w, "eigvecs": V, "rlam": rlam, "neig": neig}


# --------------------------------------------------------------------------- #
# Step 3 -- regress the target on the retained PCs
# --------------------------------------------------------------------------- #

def _pcr(scores_cal, y_cal, eigvecs, eigvals, neig, order_method, t_cutoff, ncal):
    """Select PCs and fit the regression in PC space, then back-transform to
    per-predictor weights.

    The scores are orthonormal after standardization, so the multiple
    regression reduces to a set of independent simple regressions: the weight on
    PC j is just its correlation with the standardized target, and the
    calibration R-squared is the sum of squared correlations over the entered
    PCs. Cook enters PCs in order of descending |correlation| and stops by a
    t-test (not by eigenvalue order).

    Parameters
    ----------
    scores_cal : ndarray (n_cal, neig)
        PC scores over calibration, already standardized (population SD).
    y_cal : ndarray (n_cal,)
        Standardized target over calibration (population SD).
    eigvecs : ndarray (n_pred, >=neig)
        Eigenvectors (columns), descending.
    eigvals : ndarray
        ORIGINAL eigenvalues (descending) -- used in the back-transform.
    neig : int
        Number of PCs available.
    order_method : {'tstat'}
        Entry/stop rule. Phase 1 implements the t-statistic rule.
    t_cutoff : float
        |t| threshold for the t-statistic rule.
    ncal : int
        Number of calibration years (for the t-stat dof).

    Returns
    -------
    dict with keys
        corr   : correlation of each PC with the target (length neig)
        sel    : indices (into 0..neig-1) of the PCs entered, in entry order
        order  : PCs ranked by descending |corr| (entry candidate order)
        calR2  : calibration R-squared
        betas  : back-transformed weight per screened predictor
    """
    corr = np.array([np.corrcoef(scores_cal[:, j], y_cal)[0, 1] for j in range(neig)])
    order = np.argsort(-np.abs(corr))          # descending |corr|; ties -> stable

    rdf = ncal - 2
    sel = []
    calR2 = 0.0
    if order_method == "tstat":
        cum = 0.0
        last = 0
        for step, j in enumerate(order, start=1):
            cj = corr[j]
            cum += cj * cj
            denom = (1.0 - cj * cj) / rdf
            t = cj / np.sqrt(denom) if denom > 0 else np.inf
            if abs(t) > t_cutoff:
                last = step                     # parity: last step still passing
                calR2 = cum
        sel = list(order[:last])
    else:
        raise ValueError("phase 1 implements order_method='tstat' only; "
                         "got %r" % (order_method,))

    # Back-transform PC weights to per-predictor weights.
    #   beta_i = sum_j  corr_j * load(i, j) / sqrt(lambda_j)
    # over the SELECTED PCs, using the ORIGINAL eigenvalue lambda_j (Cook
    # restores ev to raw eigenvalues before this step -- NOT the rescaled rlam).
    n_pred = eigvecs.shape[0]
    betas = np.zeros(n_pred)
    for j in sel:
        lam = eigvals[j]
        if lam <= _TINY:
            continue
        betas += corr[j] * eigvecs[:, j] / np.sqrt(lam)

    return {"corr": corr, "sel": sel, "order": order, "calR2": calR2, "betas": betas}


# --------------------------------------------------------------------------- #
# Sign test and product-means test (Cook constructions; SciPy distributions)
# --------------------------------------------------------------------------- #

def _sign_and_pm(a, e):
    """Sign test and product-means (cross-products means) test -- the two verify
    panels SciPy has no off-the-shelf equivalent for, so we keep Cook's
    construction and take the significance from an exact SciPy distribution.

    Both rest on the signed products ``prod2 = (a - mean(a)) * (e - mean(a))`` --
    whether actual and estimate fall on the same side of the actual mean.

    Returns a dict: sign_pos/sign_neg/sign_p and pm_diff/pm_t/pm_p (one-sided,
    positive skill expected).
    """
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    ma = a.mean()
    prod2 = (a - ma) * (e - ma)
    pos = prod2 > 0
    npe = int(np.sum(pos))
    nme = int(prod2.size - npe)                        # prod2 <= 0 (Cook's split)
    out = {"sign_pos": npe, "sign_neg": nme}

    # Sign test: exact one-sided binomial that the same-side count beats chance.
    if npe + nme > 0:
        out["sign_p"] = float(stats.binomtest(npe, npe + nme, 0.5,
                                              alternative="greater").pvalue)
    else:
        out["sign_p"] = np.nan

    # Product-means test: Cook's two-sample t between the mean positive product
    # and the mean magnitude of the negative products, one-sided via SciPy's t.
    cpmp = prod2[pos]
    cpmm = prod2[~pos]
    if npe >= 2 and nme >= 2:
        mp = cpmp.mean()
        mm = abs(cpmm.mean())
        vp = float(np.sum(cpmp ** 2) - mp * mp * npe) / (npe - 1)
        vm = float(np.sum(cpmm ** 2) - mm * mm * nme) / (nme - 1)
        se = np.sqrt(vp / npe + vm / nme)
        if se > 0:
            t = (mp - mm) / se
            out["pm_diff"] = mp - mm
            out["pm_t"] = t
            out["pm_p"] = float(stats.t.sf(t, npe + nme - 2)) if t > 0 else 1.0
        else:
            out["pm_diff"], out["pm_t"], out["pm_p"] = mp - mm, np.nan, np.nan
    else:
        out["pm_diff"] = out["pm_t"] = out["pm_p"] = np.nan
    return out

# --------------------------------------------------------------------------- #
# Step 4 -- verification statistics (port of `verify`, undifferenced pass)
# --------------------------------------------------------------------------- #

def _verify(actual_std, est_std, re_baseline):
    """Verification statistics in standardized units, as Cook's ``verify`` /
    ``rece`` compute them (lines 4848, 5418).

    Both inputs are standardized by the *calibration* mean and SD of the target.
    ``verify`` compares the standardized actual target against ``recon_std`` (the
    PC fit before variance rescaling), NOT the variance-recovered
    reconstruction.

    ``rece`` reduces (after the algebra cancels) to

        RE = 1 - SSE / sum((actual - re_baseline)^2)
        CE = 1 - SSE / sum((actual - mean(actual))^2)

    where ``re_baseline`` is the *calibration* climatology Cook passes in as
    ``xm``. For the undifferenced pass that is the calibration mean of the
    standardized target (= 0); for the 1st-differenced pass it is the
    calibration mean of the *differenced* target, which is generally non-zero.
    CE always uses the verification period's own mean.

    Parameters
    ----------
    actual_std : ndarray
        Standardized actual target over the verification years.
    est_std : ndarray
        Standardized reconstruction (recon_std) over the same years.
    re_baseline : float
        Calibration climatology baseline for RE (Cook's ``xm``).

    Returns
    -------
    dict : r, RE, CE, and the RE/CE sums of squares.
    """
    a = np.asarray(actual_std, dtype=float)
    e = np.asarray(est_std, dtype=float)
    sse = float(np.sum((a - e) ** 2))

    # RE baseline = calibration climatology (0 undifferenced; diff-mean differenced).
    ss_cal = float(np.sum((a - re_baseline) ** 2))
    RE = 1.0 - sse / ss_cal if ss_cal > _TINY else np.nan

    # CE baseline = verification-period mean of the (standardized) actual.
    ss_ver = float(np.sum((a - a.mean()) ** 2))
    CE = 1.0 - sse / ss_ver if ss_ver > _TINY else np.nan

    out = {"RE": RE, "CE": CE, "sse": sse, "ss_cal": ss_cal, "ss_ver": ss_ver}

    # Correlations: SciPy's exact distributions, one-sided (``greater``) because a
    # valid reconstruction is expected to correlate positively -- Cook's
    # methodology, computed exactly rather than through his Wilson-Hilferty /
    # normal approximations.
    ok = a.size >= 3 and np.std(a) > _TINY and np.std(e) > _TINY
    if ok:
        pr = stats.pearsonr(a, e, alternative="greater")
        sp = stats.spearmanr(a, e, alternative="greater")
        kt = stats.kendalltau(a, e, alternative="greater")
        out["r"] = float(pr.statistic)
        out["r_p"] = float(pr.pvalue)
        out["spearman"] = float(sp.statistic)
        out["spearman_p"] = float(sp.pvalue)
        out["kendall"] = float(kt.statistic)
        out["kendall_p"] = float(kt.pvalue)
    else:
        out.update({"r": np.nan, "r_p": np.nan, "spearman": np.nan,
                    "spearman_p": np.nan, "kendall": np.nan, "kendall_p": np.nan})

    # Equality of means: Welch's two-sample t (two-sided), from SciPy.
    if a.size >= 2 and e.size >= 2:
        tt = stats.ttest_ind(a, e, equal_var=False)
        out["eqmeans_diff"] = float(a.mean() - e.mean())
        out["eqmeans_t"] = float(tt.statistic)
        out["eqmeans_p"] = float(tt.pvalue)
    else:
        out.update({"eqmeans_diff": np.nan, "eqmeans_t": np.nan, "eqmeans_p": np.nan})

    out.update(_sign_and_pm(a, e))
    return out


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

class PcregResult:
    """Result of :func:`pcreg`.

    Attributes
    ----------
    reconstruction : pandas.Series
        Year-indexed reconstruction of the target over the full predictor
        period, in the target's own units (variance rescaled to the calibration
        mean and SD).
    calibration, verification : tuple(int, int) or None
        The calibration and verification windows (first, last).
    predictors_screened : list
        Column labels retained by the screening step, in column order.
    screen_info : pandas.DataFrame
        Per-predictor r, t and p from screening.
    n_pc : int
        Number of principal components retained.
    pcs_selected : list of int
        1-based PC numbers entered in the regression, in entry order.
    cal_r2 : float
        Calibration R-squared.
    stats : dict
        Verification statistics on the undifferenced data, the same set on the
        1st-differenced data under ``_diff`` keys, and -- when ``cv`` was
        requested -- the cross-validated skill under ``cv_`` keys. The panel
        mirrors Cook's ``verify``: ``RE``, ``CE``; Pearson ``r`` (``r_p``);
        ``spearman`` and ``kendall`` rank correlations (``_p``); the
        ``sign_pos``/``sign_neg`` sign test (``sign_p``); the product-means test
        ``pm_diff`` (``pm_t``, ``pm_p``); and the equality-of-means test
        ``eqmeans_diff`` (``eqmeans_t``, ``eqmeans_p``). Significance comes from
        SciPy's exact distributions; the correlation p-values are one-sided
        (positive skill expected) and the sign test is an exact binomial. The
        robust/biweight correlation is not yet ported.
    cv_reconstruction : pandas.Series or None
        The pooled held-out cross-validation prediction per calibration year
        (target units), or ``None`` when ``cv`` was not requested.
    dropped_coverage : list
        Predictor columns excluded before screening for insufficient calibration
        coverage (with lead/lag, because the lagged window runs past their data).
        Reported as a warning at fit time; kept here for inspection.
    weights : pandas.Series
        Back-transformed weight per screened predictor.
    eigenvalues : numpy.ndarray
        Original eigenvalues of the screened-predictor correlation matrix
        (descending).
    settings : dict
        The resolved analysis settings.
    """

    def __init__(self, reconstruction, calibration, verification,
                 predictors_screened, screen_info, n_pc, pcs_selected, cal_r2,
                 stats, weights, eigenvalues, settings):
        self.reconstruction = reconstruction
        self.calibration = calibration
        self.verification = verification
        self.predictors_screened = predictors_screened
        self.screen_info = screen_info
        self.n_pc = n_pc
        self.pcs_selected = pcs_selected
        self.cal_r2 = cal_r2
        self.stats = stats
        self.weights = weights
        self.eigenvalues = eigenvalues
        self.settings = settings
        self.cv_reconstruction = None
        self.dropped_coverage = []
        self.fig = None
        self.axes = None

    def __repr__(self):
        cal = self.calibration
        rv = self.stats.get("r")
        re = self.stats.get("RE")
        return ("PcregResult(cal=%s, n_pred=%d, n_pc=%d, order=%d, "
                "cal_R2=%.3f, ver_r=%s, ver_RE=%s)"
                % (("%d-%d" % cal) if cal else None,
                   len(self.predictors_screened), self.n_pc,
                   len(self.pcs_selected), self.cal_r2,
                   ("%.3f" % rv) if rv is not None and np.isfinite(rv) else "n/a",
                   ("%.3f" % re) if re is not None and np.isfinite(re) else "n/a"))


# --------------------------------------------------------------------------- #
# Input coercion
# --------------------------------------------------------------------------- #

def _as_year_frame(predictors, drop_substring="samp_depth"):
    """Coerce the predictor input to a year-indexed float DataFrame.

    Accepts a DataFrame (e.g. the output of :func:`read_crn`) or a dict/sequence
    of Series. Columns whose name contains ``drop_substring`` (the sample-depth
    companions that ``read_crn`` emits) are dropped automatically.
    """
    if isinstance(predictors, pd.Series):
        P = predictors.to_frame()
    elif isinstance(predictors, pd.DataFrame):
        P = predictors
    else:
        raise TypeError("predictors must be a year-indexed pandas DataFrame or "
                        "Series, not " + str(type(predictors)) + ".")
    if drop_substring:
        keep_cols = [c for c in P.columns if drop_substring not in str(c)]
        P = P[keep_cols]
    P = P.apply(pd.to_numeric, errors="coerce")
    P.index = np.asarray(P.index).astype(int)
    P.index.name = "year"
    return P


def _as_year_series(target, name="target"):
    """Coerce the target input to a year-indexed float Series."""
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("target DataFrame must have exactly one column; "
                             "got " + str(list(target.columns)) + ".")
        s = target.iloc[:, 0]
    elif isinstance(target, pd.Series):
        s = target
    else:
        raise TypeError("target must be a year-indexed pandas Series or 1-column "
                        "DataFrame, not " + str(type(target)) + ".")
    out_name = s.name if getattr(s, "name", None) is not None else name
    s = pd.Series(np.asarray(s, dtype=float), index=np.asarray(s.index).astype(int))
    s.index.name = "year"
    s.name = out_name
    return s


def _parse_window(window, what):
    if window is None:
        return None
    a, b = window
    a, b = int(a), int(b)
    if a > b:
        a, b = b, a
    return (a, b)


# --------------------------------------------------------------------------- #
# Lead/lag masks (port of `lagit`, line 3418)
# --------------------------------------------------------------------------- #

def _parse_lag(lag):
    """Normalize the ``lag`` argument to a sorted list of integer offsets.

    An offset ``d`` means a copy of each predictor whose value at reference year
    ``Y`` is the predictor at ``Y + d`` (so ``d < 0`` pulls earlier ring years --
    a lag; ``d > 0`` a lead). Accepts:

    * ``None``            -> no lagging (returns ``None``);
    * an 11-char string   -> Cook's mask over positions ``54321012345`` /
      ``-----0+++++``; a non-``0``/space at position ``jj`` (1..11) selects
      offset ``jj - 6`` (so ``"00000110000"`` -> offsets ``0`` and ``+1``);
    * a sequence of ints  -> the offsets directly (e.g. ``(-1, 0, 1)``).
    """
    if lag is None:
        return None
    if isinstance(lag, str):
        m = lag.strip()
        if len(m) != 11:
            raise ValueError("a string lag mask must be exactly 11 characters "
                             "(positions -5..+5); got %d." % (len(m),))
        offsets = [jj - 5 for jj, ch in enumerate(m) if ch not in ("0", " ")]
        # jj here is 0-based over the 11 slots, so slot index jj maps to offset
        # jj-5 (slot 0 -> -5, slot 5 -> 0, slot 10 -> +5).
    else:
        try:
            offsets = [int(d) for d in lag]
        except TypeError:
            raise TypeError("lag must be None, an 11-char mask string, or a "
                            "sequence of integer offsets; got %r." % (lag,))
    offsets = sorted(set(offsets))
    if not offsets:
        raise ValueError("the lag mask selected no offsets.")
    if min(offsets) < -5 or max(offsets) > 5:
        raise ValueError("lag offsets must be within -5..+5; got %s." % (offsets,))
    return offsets


def _apply_lag(P, offsets):
    """Expand each predictor into one column per lag offset.

    The copy for offset ``d`` is ``P[col].shift(-d)`` -- its value at year ``Y``
    is the predictor at ``Y + d`` -- named ``"<col>@t{+d}"`` (e.g. ``COL@t+0``,
    ``COL@t-1``). Mirrors Cook's ``lagit``: lagging runs before screening, so the
    screen/PCA/regression see the expanded pool, and the usable period shrinks at
    the ends (the shifted columns are NaN there, which the full-coverage screen
    requirement handles).
    """
    if not offsets:
        return P
    cols = {}
    for col in P.columns:
        s = P[col]
        for d in offsets:
            cols["%s@t%+d" % (col, d)] = s.shift(-d)
    out = pd.DataFrame(cols, index=P.index)
    out.index.name = P.index.name
    return out


# --------------------------------------------------------------------------- #
# Core fit (screen -> PCA -> PC regression -> reconstruction), reused by CV
# --------------------------------------------------------------------------- #

def _core_fit(P, y, cal_years, s, recon_years=None, strict=True):
    """Fit the whole pipeline on ``cal_years`` and reconstruct at ``recon_years``.

    Returns a dict with the fitted pieces and the reconstruction (standardized
    ``recon_std`` and variance-rescaled ``recon_scaled``), or ``None`` when
    ``strict`` is False and the fit degenerates (empty screen / no PC entered) --
    which lets a cross-validation fold skip gracefully.
    """
    ncal = len(cal_years)
    kept, screen_info, dropped_cov = _screen(P, y, cal_years, s["screen_alpha"],
                                             s["screen_tails"], s["screen_sign"],
                                             s["min_coverage"])
    if len(kept) == 0:
        if strict:
            raise ValueError("no predictors survived screening at alpha=%g; relax "
                             "screen_alpha or check the calibration overlap."
                             % (s["screen_alpha"],))
        return None
    if strict and len(kept) < 2:
        _warn("only one predictor survived screening; PCA/PCR is degenerate.")

    Pk = P[kept]
    mu = Pk.reindex(cal_years).mean()
    sd = Pk.reindex(cal_years).std(ddof=0).replace(0.0, np.nan)
    Pstd = (Pk - mu) / sd
    Pstd_cal = Pstd.reindex(cal_years).to_numpy()

    pca = _pca(Pstd_cal, s["pc_criterion"], s["n_pc"], s["ev_cutoff"], s["cum_pct"])
    V = pca["eigvecs"]
    eigvals = pca["eigvals"]
    neig = pca["neig"]

    scores_cal_raw = Pstd_cal @ V[:, :neig]
    smu = scores_cal_raw.mean(axis=0)
    ssd = scores_cal_raw.std(axis=0, ddof=0)
    ssd = np.where(ssd == 0.0, np.nan, ssd)
    scores_cal = (scores_cal_raw - smu) / ssd

    pmu = float(y.reindex(cal_years).mean())
    psd = float(y.reindex(cal_years).std(ddof=0))
    y_cal = ((y.reindex(cal_years) - pmu) / psd).to_numpy()

    pcr = _pcr(scores_cal, y_cal, V, eigvals, neig, s["order_method"], s["t_cutoff"], ncal)
    sel = pcr["sel"]
    corr = pcr["corr"]
    if len(sel) == 0 and not strict:
        return None

    if recon_years is None:
        recon_years = Pstd.dropna().index.to_numpy()
    else:
        recon_years = np.asarray(recon_years)
    # The variance rescale needs the reconstruction's mean/SD over the CALIBRATION
    # (fitting) years, so always reconstruct those too -- not only ``recon_years``
    # (which, for a cross-validation fold, is the held-out year and excludes the
    # training window). Reconstruct the union, then slice.
    need = np.union1d(recon_years, cal_years)
    Sneed = (Pstd.reindex(need).to_numpy() @ V[:, :neig] - smu) / ssd
    rstd_need = pd.Series(Sneed[:, sel] @ corr[sel] if len(sel) else np.zeros(len(need)),
                          index=need)
    recon_std = rstd_need.reindex(recon_years)

    rc = rstd_need.reindex(cal_years).to_numpy()
    rc_mu = rc.mean()
    rc_sd = rc.std(ddof=0)
    recon_scaled = ((recon_std - rc_mu) / rc_sd * psd + pmu) if rc_sd > _TINY \
        else pd.Series(np.full(len(recon_years), pmu), index=recon_years)

    return {"kept": kept, "screen_info": screen_info, "dropped_coverage": dropped_cov,
            "eigvecs": V, "eigvals": eigvals, "neig": neig, "sel": sel, "corr": corr,
            "betas": pcr["betas"], "calR2": pcr["calR2"], "pmu": pmu, "psd": psd,
            "recon_std": recon_std, "recon_scaled": recon_scaled}


def _cross_validate(P, y, cal_years, s, cv):
    """Leave-k-out cross-validation that refits the whole pipeline each fold.

    The calibration period is tiled into contiguous blocks of ``cv`` years
    (``cv=1`` -> true leave-one-out). Each block is held out, the model is refit
    on the remaining calibration years, and the held-out year(s) are predicted;
    the pooled held-out predictions give the cross-validated skill.

    Returns ``(cv_pred Series, stats dict)`` in the target's own units. Over the
    calibration window the RE and CE climatology baselines coincide (both are the
    calibration mean), so ``cv_RE`` and ``cv_CE`` are equal there; ``cv_r`` is the
    independent discriminator.
    """
    cal_years = np.asarray(cal_years)
    ncal = len(cal_years)
    cv = int(cv)
    if cv < 1:
        raise ValueError("cv must be >= 1 (1 = leave-one-out).")
    if cv >= ncal:
        raise ValueError("cv (%d) must be smaller than the calibration length "
                         "(%d)." % (cv, ncal))

    preds = pd.Series(np.nan, index=cal_years, dtype=float)
    n_skipped = 0
    for start in range(0, ncal, cv):
        held = cal_years[start:start + cv]
        train = cal_years[~np.isin(cal_years, held)]
        sf = dict(s)
        # a fold screens over the shorter training window, so relax the coverage
        # requirement to the training length (or the user's, whichever is smaller)
        sf["min_coverage"] = min(s["min_coverage"], len(train))
        fit = _core_fit(P, y, train, sf, recon_years=held, strict=False)
        if fit is None:
            n_skipped += len(held)
            continue
        preds.loc[held] = fit["recon_scaled"].reindex(held).to_numpy()

    if n_skipped:
        _warn("%d cross-validation year(s) could not be predicted (degenerate "
              "fold) and were dropped." % (n_skipped,))

    actual = y.reindex(cal_years).to_numpy()
    pred = preds.to_numpy()
    good = np.isfinite(actual) & np.isfinite(pred)
    a = actual[good]
    p = pred[good]
    stats = {}
    if a.size >= 2:
        clim = float(a.mean())                         # calibration-mean climatology
        sse = float(np.sum((a - p) ** 2))
        ss = float(np.sum((a - clim) ** 2))
        stats["cv_RE"] = 1.0 - sse / ss if ss > _TINY else np.nan
        stats["cv_CE"] = stats["cv_RE"]                # same baseline over calibration
        stats["cv_r"] = float(np.corrcoef(a, p)[0, 1]) if (np.std(a) > _TINY and
                                                           np.std(p) > _TINY) else np.nan
        stats["cv_n"] = int(a.size)
        stats["cv_k"] = cv
    return preds, stats


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #

def pcreg(predictors, target, calibration, verification=None,
          screen_alpha=0.10, screen_tails=2, screen_sign=None,
          min_coverage=None,
          pc_criterion="pvp", n_pc=12, ev_cutoff=1.0, cum_pct=90.0,
          order_method="tstat", t_cutoff=1.0,
          prewhiten=False, ar_max=10, ar_method="burg",
          lag=None, cv=None,
          first_differences=True, plot=False):
    """Orthogonal spatial regression / PC-regression climate reconstruction.

    A faithful port of Ed Cook's OSR FORTRAN (Cook, Briffa & Jones 1994). The
    predictors are screened against the target over the calibration window,
    reduced to principal components, and the target is regressed on the PCs that
    pass a t-test; the fit is back-transformed to per-predictor weights, a
    reconstruction is produced over the full predictor period, and (if a
    verification window is given) it is verified against the withheld data.

    Parameters
    ----------
    predictors : DataFrame or Series
        Year-indexed tree-ring predictors (e.g. the output of
        :func:`read_crn`). Sample-depth companion columns are dropped
        automatically.
    target : Series or 1-column DataFrame
        Year-indexed climate target to reconstruct. Pre-seasonalize it yourself
        (e.g. a March-May PDSI mean) before calling.
    calibration : (first, last)
        Calibration window, inclusive.
    verification : (first, last) or None
        Verification window, inclusive. When ``None``, only the calibration fit
        and the reconstruction are returned (no verification statistics).
    screen_alpha : float
        Screening significance level. Use ``1.0`` with ``screen_tails=2`` to
        force every predictor into the PCA ("normal response function").
    screen_tails : {1, 2}
        One- or two-tailed screening test.
    screen_sign : {'+', '-', None}
        Required correlation sign for a one-tailed test.
    min_coverage : int or None
        Minimum overlapping years to admit a predictor. Defaults to the full
        calibration length (every predictor must cover the whole window), which
        is what the fixture uses.
    pc_criterion : {'pvp', 'kaiser', 'first_m', 'cum_pct', 'all'}
        PC truncation rule (Cook's eigenvalue-cutoff options 3, 1, 0, 2, 4).
    n_pc : int
        Number of PCs for ``pc_criterion='first_m'``.
    ev_cutoff : float
        Rescaled-eigenvalue cutoff for ``pc_criterion='kaiser'``.
    cum_pct : float
        Cumulative-variance percent for ``pc_criterion='cum_pct'``.
    order_method : {'tstat'}
        PC entry/stop rule. Phase 1 implements the t-statistic rule.
    t_cutoff : float
        |t| threshold for ``order_method='tstat'``.
    prewhiten : bool
        Prewhiten predictors and target with a best-fit AR model before
        screening (reuses :func:`dplpy.autoreg.ar_func`). The AR order is chosen
        by AIC up to ``ar_max``.
    ar_max : int
        Maximum AR order when ``prewhiten=True``.
    ar_method : {'burg', 'yw', 'ols'}
        AR estimator. ``'burg'`` (maximum entropy) matches Cook's ``memcof``.
    lag : None, 11-char mask string, or sequence of int
        Lead/lag the predictors before screening (Cook's ``lagit``). Each
        predictor is expanded into one column per offset: offset ``d`` uses the
        predictor at year ``Y + d`` to predict year ``Y`` (``d < 0`` = a lag on
        earlier ring years, ``d > 0`` = a lead). Give the offsets directly
        (e.g. ``(0, 1)`` or ``(-1, 0, 1)``) or as an 11-char mask over positions
        ``-5..+5`` (e.g. ``"00000110000"`` = offsets 0 and +1). The same mask is
        applied to every predictor.
    cv : None or int
        Cross-validate by refitting the whole pipeline on held-out folds of the
        calibration period. ``cv=1`` is leave-one-out; ``cv=k`` leaves out
        contiguous ``k``-year blocks. Results land in ``stats`` under the ``cv_``
        keys (``cv_r``, ``cv_RE``, ``cv_CE``; ``cv_RE`` and ``cv_CE`` coincide
        over the calibration window). ``None`` skips cross-validation.
    first_differences : bool
        Also report verification statistics on the 1st-differenced series (the
        FORTRAN's second verification pass), under ``_diff`` keys in ``stats``.
    plot : bool
        Draw the reconstruction/verification figure.

    Returns
    -------
    PcregResult

    Notes
    -----
    A single target, PC-regression core with optional AR prewhitening, lead/lag
    predictor masks, 1st-difference verification, and leave-k-out
    cross-validation. Deferred to a later phase (and documented in
    ``dev/pcreg_DECISIONS.md``): canonical regression, robust / biweight
    correlations, and the Southern-Hemisphere auto-shift.
    """
    if screen_tails not in (1, 2):
        raise ValueError("screen_tails must be 1 or 2, got %r" % (screen_tails,))
    if screen_tails == 1 and screen_sign not in ('+', '-'):
        raise ValueError("a one-tailed screen needs screen_sign='+' or '-'.")
    if not (0.0 < screen_alpha <= 1.0):
        raise ValueError("screen_alpha must be in (0, 1], got %r" % (screen_alpha,))

    P = _as_year_frame(predictors)
    y = _as_year_series(target)

    cal = _parse_window(calibration, "calibration")
    ver = _parse_window(verification, "verification")
    if cal is None:
        raise ValueError("calibration window is required.")
    cal_years = np.arange(cal[0], cal[1] + 1)
    if len(cal_years) < _MIN_CAL_YEARS:
        raise ValueError("calibration window is only %d years; need at least %d."
                         % (len(cal_years), _MIN_CAL_YEARS))
    if min_coverage is None:
        min_coverage = len(cal_years)

    # Optional AR prewhitening, reusing dplPy's existing AR machinery.
    if prewhiten:
        from .autoreg import ar_func
        # ar_func whitens each column over its own finite span; align back to
        # year index afterwards. Burg == Cook's memcof (maximum entropy).
        P = ar_func(P, ar_max=ar_max, aic=True, method=ar_method)
        yw = ar_func(y.to_frame(), ar_max=ar_max, aic=True, method=ar_method)
        y = yw.iloc[:, 0] if isinstance(yw, pd.DataFrame) else yw
        P = _as_year_frame(P)
        y = _as_year_series(y)

    # Optional lead/lag expansion (before screening, after prewhitening), Cook's
    # `lagit`. Each predictor becomes one column per offset.
    offsets = _parse_lag(lag)
    if offsets is not None:
        P = _apply_lag(P, offsets)

    settings = {
        "screen_alpha": screen_alpha, "screen_tails": screen_tails,
        "screen_sign": screen_sign, "min_coverage": min_coverage,
        "pc_criterion": pc_criterion, "n_pc": n_pc, "ev_cutoff": ev_cutoff,
        "cum_pct": cum_pct, "order_method": order_method, "t_cutoff": t_cutoff,
        "prewhiten": prewhiten, "ar_max": ar_max, "ar_method": ar_method,
        "lag": offsets, "cv": cv,
    }

    # ---- Core fit: screen -> PCA -> PC regression -> reconstruction ------ #
    fit = _core_fit(P, y, cal_years, settings, recon_years=None, strict=True)
    kept = fit["kept"]
    sel = fit["sel"]

    # Surface any predictors dropped before screening for insufficient coverage
    # (the mirror of the FORTRAN's hidden edge-padding -- here it is reported, not
    # patched over). Reported once for the main fit; cross-validation folds stay
    # quiet.
    dropped_cov = fit["dropped_coverage"]
    if dropped_cov:
        preview = ", ".join(str(c) for c in dropped_cov[:8])
        more = "" if len(dropped_cov) <= 8 else (", ... (+%d more)" % (len(dropped_cov) - 8))
        lagnote = " (the lag/lead window reaches beyond their data)" if offsets else ""
        _warn("%d predictor(s) were excluded before screening for insufficient "
              "calibration coverage (< %d overlapping years)%s: %s%s."
              % (len(dropped_cov), settings["min_coverage"], lagnote, preview, more))

    if len(sel) == 0:
        _warn("no PC passed the t-stat cutoff; nothing entered the regression.")
    recon_std = fit["recon_std"]
    recon_scaled = fit["recon_scaled"]
    recon_scaled.name = "reconstruction"
    pmu, psd = fit["pmu"], fit["psd"]

    # ---- Verification --------------------------------------------------- #
    stats = {}
    if ver is not None:
        ver_years = np.arange(ver[0], ver[1] + 1)
        # Work in the same standardized units `verify` uses: standardized actual
        # target, and recon_std (the PC fit before variance rescaling).
        y_std = (y - pmu) / psd

        # RE climatology baselines (Cook's xm / xbar), computed over CALIBRATION:
        #   undifferenced -> calibration mean of the standardized target (= 0),
        #   differenced   -> calibration mean of the differenced target.
        y_std_cal = y_std.reindex(cal_years).to_numpy()
        base_undiff = float(np.nanmean(y_std_cal))
        base_diff = float(np.nanmean(np.diff(y_std_cal)))

        a = y_std.reindex(ver_years).to_numpy()
        e = recon_std.reindex(ver_years).to_numpy()
        good = np.isfinite(a) & np.isfinite(e)
        vstats = _verify(a[good], e[good], base_undiff)
        stats.update(vstats)

        if first_differences:
            # parity: difference both series (consecutive years), then recompute
            # with the calibration differenced-mean baseline (Cook's xbar1).
            da = np.diff(a)
            de = np.diff(e)
            g2 = np.isfinite(da) & np.isfinite(de)
            if g2.sum() >= 2:
                d = _verify(da[g2], de[g2], base_diff)
                stats.update({k + "_diff": v for k, v in d.items()})

    # ---- Cross-validation (leave-k-out, full refit) --------------------- #
    cv_reconstruction = None
    if cv is not None:
        cv_reconstruction, cv_stats = _cross_validate(P, y, cal_years, settings, cv)
        stats.update(cv_stats)

    result = PcregResult(
        reconstruction=recon_scaled,
        calibration=cal,
        verification=ver,
        predictors_screened=kept,
        screen_info=fit["screen_info"],
        n_pc=fit["neig"],
        pcs_selected=[int(j) + 1 for j in sel],       # 1-based for reporting
        cal_r2=fit["calR2"],
        stats=stats,
        weights=pd.Series(fit["betas"], index=pd.Index(kept, name="predictor")),
        eigenvalues=fit["eigvals"],
        settings=settings,
    )
    result.cv_reconstruction = cv_reconstruction
    result.dropped_coverage = dropped_cov

    if plot:
        try:
            from .pcreg_plot import _plot_pcreg
            result.fig, result.axes = _plot_pcreg(result, y)
        except Exception as exc:                       # plotting must never break the fit
            _warn("pcreg plot failed: %s" % (exc,))

    return result
