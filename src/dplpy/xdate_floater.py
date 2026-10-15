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

# Title: xdate_floater.py
# Description: Crossdate a *floating* (undated) ring-width series against a dated
#   reference collection, to estimate the calendar placement of the undated wood
#   (e.g. a beam from a historic building or a remnant/sub-fossil log). Based on
#   and extends dplR's xdate.floater(): it combines that sliding-placement
#   approach with the t statistic and the additional crossdating statistics of
#   Wilson (2026) that dendro dating practice relies on.
#
#   Method: build a master chronology from the dated `rwl` (divide each series by
#   its mean, apply a high-pass transform, then a Tukey-biweight robust row mean);
#   transform the floating series the same way; slide it across the master over
#   every offset with at least `min_overlap` overlapping rings; and at each offset
#   record the correlation r and the crossdating statistics below. The offset with
#   the highest t-value is the estimated dating.
#
#   Two deliberate departures from dplR's xdate.floater():
#     * Best placement is selected by the highest t statistic, not the highest
#       correlation r (dplR uses which.max(r)). t and r rank identically when the
#       overlap n is fixed, but differ at overhang offsets where n varies; t is
#       used here because it accounts for the degrees of freedom (Wilson 2026).
#     * dplPy does not apply dplR's future-year trim (dplR drops offsets whose
#       last year exceeds the current calendar year); every offset with adequate
#       overlap is reported.
#
#   At each offset we report:
#     * r         -- the sliding correlation (Spearman by default);
#     * t         -- the Baillie & Pilcher (1973) t statistic, t = r*sqrt(df-2) /
#                    sqrt(1-r^2), the standard crossdating measure (a value >= ~3.5
#                    is the classic acceptance threshold; see Fowler & Bridge 2017);
#     * eff_df    -- the effective degrees of freedom, reduced from the overlap n
#                    for the lag-1 autocorrelation of the two series (autocorrelated
#                    series carry less independent information; Wigley et al. 1987;
#                    Wilson 2026);
#     * p_bonf    -- the one-tailed p-value of t, Bonferroni-corrected for the many
#                    offsets tested (conservative, but honest about the multiple
#                    comparisons a sliding search makes);
#   and for the best match, an isolation factor (IF) -- how many times larger the
#   runner-up's p-value is than the best offset's; a large IF means the date stands
#   out cleanly from the alternatives (Wigley et al. 1987; Wilson 2026).
#
# References:
#   Baillie, M.G.L., Pilcher, J.R., 1973. A simple cross-dating program for
#     tree-ring research. Tree-Ring Bulletin 33, 7-14.
#   Fowler, A.M., Bridge, M.C., 2017. Empirically-determined statistical
#     significance of the Baillie and Pilcher (1973) t statistic for British Isles
#     oak. Dendrochronologia 42, 51-55.
#   Wigley, T.M.L., Jones, P.D., Briffa, K.R., 1987. Cross-dating methods in
#     dendrochronology. J. Archaeol. Sci. 14 (1), 51-64.
#   Wilson, R., 2026. Multi-parameter crossdating for sub-fossil and historical
#     samples. Dendrochronologia 96, 126485.
#
# example usage:
#   >>> import dplpy as dpl
#   >>> ref = dpl.readers("reference.rwl")        # a dated reference collection
#   >>> res = dpl.xdate_floater(ref, floating_series, series_name="beam1")
#   >>> res["best"]     # {min_year, max_year, r, t, eff_df, p_bonf, isolation_factor, n}
#   >>> res["floater_cor_stats"]                  # every offset, best (highest t) first

import numpy as np
import pandas as pd
import scipy.stats
import datetime

from ._validate import (_require_dataframe, _normalize_corr, _plot_kwarg, _warn,
                        _normalize_choice, _require_year_index)
from .stats import get_ar1
from .xdate import (normalize_for_crossdating, dense_year_grid,
                    _row_biweight, _row_mean, _fast_corr)

_TRANSFORMS = ("pw", "fd", "none")


def _bp_t(r, df):
    """Baillie & Pilcher (1973) t from a correlation ``r`` over ``df`` degrees of
    freedom: ``t = r * sqrt(df - 2) / sqrt(1 - r^2)``. Vectorized; ``r`` is clipped
    just inside +/-1 so a perfect correlation gives a large finite t. The one
    Baillie-Pilcher t for xdate_floater and xdate_undated."""
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    df = np.asarray(df, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return r * np.sqrt(df - 2.0) / np.sqrt(1.0 - r * r)


def _trim_nan(values):
    """Drop leading and trailing NaN of a 1-D array; interior NaN are KEPT in
    place so array index stays ring position (audit A3). Returns (trimmed,
    n_leading_dropped)."""
    v = np.asarray(values, dtype=float)
    ok = np.flatnonzero(~np.isnan(v))
    if ok.size == 0:
        return v[:0], 0
    return v[ok[0]:ok[-1] + 1], int(ok[0])


def _as_series_values(series):
    """Coerce the floating series to a 1-D float array of ring positions, oldest
    first: leading/trailing NaN are stripped, interior NaN (unmeasured rings) are
    kept in place so that array index == ring position (audit A3)."""
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("`series` as a DataFrame must have exactly one "
                             "column (the floating series).")
        series = series.iloc[:, 0]
    vals = np.asarray(pd.Series(series).to_numpy(), dtype=float)
    return _trim_nan(vals)[0]


def _series_label(series):
    """A display name carried by the floating series itself: a named pandas Series,
    or a one-column DataFrame's column header. Returns None when it carries no
    usable name (a bare list/array, or a blank/None name), so the caller keeps the
    default label."""
    name = None
    if isinstance(series, pd.DataFrame) and series.shape[1] == 1:
        name = series.columns[0]
    elif isinstance(series, pd.Series):
        name = series.name
    if name is None:
        return None
    name = str(name).strip()
    return name or None


def _build_master(data, transform, biweight, ar_max=10, first_aic_min=True):
    """Master chronology from the dated collection: mean-normalize each series,
    apply the high-pass transform, then a robust row mean. Returns (values, years)
    on the DENSE one-year grid from the master's first to last measured year --
    years with no measurement in any series stay NaN in place (never dropped, so
    index offset == calendar offset; audit A3). ``ar_max``/``first_aic_min``
    control AR order selection when ``transform=="pw"`` (see
    `xdate.normalize_for_crossdating()`)."""
    ready = normalize_for_crossdating(data, prewhiten=(transform == "pw"),
                                      ar_max=ar_max, first_aic_min=first_aic_min)
    ready, years, _, _ = dense_year_grid(ready)
    M = ready.to_numpy(dtype=float)
    if transform == "fd":                       # first difference; label by later year
        M = np.diff(M, axis=0)
        years = years[1:]
    good = np.array([np.sum(~np.isnan(M[:, i])) > 3 for i in range(M.shape[1])])
    if not good.any():
        raise ValueError("no reference series had enough data to build a master.")
    row_master = _row_biweight if biweight else _row_mean
    master = row_master(M[:, good])
    master, lead = _trim_nan(master)
    return master, years[lead:lead + master.shape[0]].astype(int)


def _transform_series(y_raw, transform, ar_max=10, first_aic_min=True):
    """Mean-normalize then transform the floating series with the same
    `xdate.normalize_for_crossdating()` transform the master gets. Returns an
    array the SAME LENGTH as ``y_raw`` (index == ring position): the first
    ``order`` values of a prewhitened series, the first value of a differenced
    one, and any interior unmeasured ring are NaN in place."""
    y_raw = np.asarray(y_raw, dtype=float)
    frame = pd.DataFrame({"floater": y_raw}, index=pd.RangeIndex(len(y_raw)))
    if transform == "pw":
        ready = normalize_for_crossdating(frame, prewhiten=True, ar_max=ar_max,
                                          first_aic_min=first_aic_min)
    else:
        ready = normalize_for_crossdating(frame, prewhiten=False)
    y = ready["floater"].to_numpy(dtype=float)
    if transform == "fd":
        y = np.concatenate([[np.nan], np.diff(y)])      # label by the later ring
    return y


def _abs_ac1(values):
    """|lag-1 autocorrelation| of a 1-D array (0 if undefined)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3 or np.all(v == v[0]):
        return 0.0
    return abs(float(get_ar1(pd.Series(v))))


# --- segment-consensus dating (COFECHA UDATE-style) --------------------------
_SEG_LEN = 50        # segment window scanned across the master
_SEG_LAG = 25        # step between segments (50% overlap)
_SEG_MIN_FRAC = 0.8  # a segment placement needs this fraction of complete pairs


def _slide_corr(x, seg, method, min_pairs):
    """Correlation of ``seg`` with every length-``len(seg)`` window of ``x``
    (window start j = 0..len(x)-len(seg)), over pairwise-complete pairs, NaN
    where fewer than ``min_pairs`` pairs are complete. Complete windows are
    ranked/centred once and correlated in one matrix product (the vectorization
    `xdate._lag_table()` uses); windows with a NaN fall back to `_fast_corr()`."""
    L = seg.shape[0]
    nw = x.shape[0] - L + 1
    out = np.full(max(nw, 0), np.nan)
    if nw <= 0:
        return out
    W = np.lib.stride_tricks.sliding_window_view(x, L)
    seg_ok = ~np.isnan(seg)
    full = ~np.isnan(W).any(axis=1)
    if seg_ok.all() and full.any() and method != "kendall":
        if method == "spearman":
            sv = scipy.stats.rankdata(seg)
            Mv = scipy.stats.rankdata(W[full], axis=1)
        else:
            sv, Mv = seg, W[full]
        a = sv - sv.mean()
        B = Mv - Mv.mean(axis=1, keepdims=True)
        den = np.sqrt(np.dot(a, a) * (B * B).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[full] = np.where(den > 0, (B @ a) / den, np.nan)
        rest = np.where(~full)[0]
    else:
        rest = np.arange(nw)
    for j in rest:
        if int((seg_ok & ~np.isnan(W[j])).sum()) >= min_pairs:
            out[j] = _fast_corr(W[j], seg, method)
    return out


def _segment_consensus(x, yrs, y, method, master_year, topk=5):
    """Date each overlapping segment of the (transformed) floater independently
    against the master, in the spirit of COFECHA's UDATE. Each segment votes for a
    whole-series placement (the implied calendar year of the floater's youngest
    ring); a single dominant year means the floater is internally consistent, while
    a coherent one-year step between older and younger segments is the signature of
    a missing or false ring. Returns (segments_df, consensus_series, candidates_df).
    ``x`` (the master) and ``y`` (the floater) may carry NaN in place; each
    placement is scored on its pairwise-complete pairs and needs at least
    ``_SEG_MIN_FRAC`` of the segment complete.

    segments_df has one row per segment (oldest rings first): ``ring_lo``/``ring_hi``
    (1-based ring positions within the floater), ``impl_end`` (implied youngest-ring
    calendar year at the segment's best match) and ``best_r``. ``consensus`` is a vote
    count of those single-best implied end years (the value shown for each segment on
    the staircase plot), most-supported first. ``candidates_df`` lists, for each
    segment, its ``topk`` best placements (``rank`` 1..topk, ``impl_end``, ``r``) --
    so a near-tie between two dates is visible instead of hidden behind the winner."""
    ny = len(y)
    seg_len = min(_SEG_LEN, ny)
    if ny <= seg_len:
        starts = [0]
    else:
        starts = list(range(0, ny - seg_len + 1, _SEG_LAG))
        if starts[-1] != ny - seg_len:
            starts.append(ny - seg_len)      # end anchor: always test the youngest rings
    rows, cand_rows = [], []
    for a in starts:
        L = min(seg_len, ny - a)
        seg = y[a:a + L]
        rs = _slide_corr(x, seg, method, max(3, int(np.ceil(_SEG_MIN_FRAC * L))))
        finite = np.where(np.isfinite(rs))[0]
        if finite.size == 0:
            continue
        order = finite[np.argsort(-rs[finite])]          # best-first, NaNs excluded
        bj = int(order[0])
        rows.append({"ring_lo": a + 1, "ring_hi": a + L,
                     "impl_end": master_year((bj - a) + ny - 1),
                     "best_r": round(float(rs[bj]), 3)})
        for rank, j in enumerate(order[:topk], start=1):
            j = int(j)
            cand_rows.append({"ring_lo": a + 1, "ring_hi": a + L, "rank": rank,
                              "impl_end": master_year((j - a) + ny - 1),
                              "r": round(float(rs[j]), 3)})
    seg_df = pd.DataFrame(rows, columns=["ring_lo", "ring_hi", "impl_end", "best_r"])
    candidates = pd.DataFrame(cand_rows,
                              columns=["ring_lo", "ring_hi", "rank", "impl_end", "r"])
    # the consensus is a vote count of each segment's single best placement -- the
    # same implied end years the diagnostic uses and the staircase plot shows, so
    # the reported dominant year always corresponds to a point on the plot.
    consensus = (seg_df["impl_end"].value_counts() if not seg_df.empty
                 else pd.Series(dtype=int))
    return seg_df, consensus, candidates


def _diagnose_internal_error(seg_df):
    """Detect a coherent one-year step between older and younger segments -- the
    fingerprint of an internal dating error in the floater (a missing or false
    ring), which a whole-series slide cannot reveal. Returns a dict
    ``{kind, reliable_end, other_end, near_ring, n_older, n_younger}`` or None.

    A missing ring makes the OLDER segments date one year EARLY (their portion is
    a ring short); a false ring makes them date one year LATE. Junk straddling
    segments (low best_r) are dropped so the two real blocks stand out."""
    if len(seg_df) < 4:
        return None
    s = seg_df.sort_values("ring_lo").reset_index(drop=True)
    thr = max(0.35, float(s["best_r"].median()) * 0.5)
    conf = s[s["best_r"] >= thr].reset_index(drop=True)
    if len(conf) < 4:
        return None
    vc = conf["impl_end"].value_counts()
    if len(vc) < 2 or int(vc.iloc[1]) < 2:
        return None
    e1, e2 = int(vc.index[0]), int(vc.index[1])
    if abs(e1 - e2) != 1:
        return None
    g1 = conf[conf["impl_end"] == e1]
    g2 = conf[conf["impl_end"] == e2]
    # which block is older (smaller ring positions)?
    if g1["ring_lo"].median() < g2["ring_lo"].median():
        older_end, younger_end, older, younger = e1, e2, g1, g2
    else:
        older_end, younger_end, older, younger = e2, e1, g2, g1
    if not (older["ring_hi"].max() <= younger["ring_lo"].max()):
        return None                          # blocks not spatially separated
    kind = "missing" if older_end < younger_end else "false"
    near = int(round((older["ring_hi"].max() + younger["ring_lo"].min()) / 2))
    return {"kind": kind, "reliable_end": younger_end, "other_end": older_end,
            "near_ring": near, "n_older": int(len(older)), "n_younger": int(len(younger))}


def _consensus_summary(seg_df, best):
    """One honest sentence describing what the segments agreed on -- phrased to match
    the staircase plot (each segment's single-best implied end year) and to avoid
    over-claiming when there are too few segments or the whole-series match is weak."""
    ends = [int(e) for e in seg_df["impl_end"]]
    n = len(ends)
    if n == 0:
        return "no segments could be scored."
    vc = seg_df["impl_end"].value_counts()
    top_end, top_n = int(vc.index[0]), int(vc.iloc[0])
    if top_n == n:
        head = "all %d segments imply end year %d" % (n, top_end)
    elif top_n > n - top_n:
        others = ", ".join(str(e) for e in ends if e != top_end)
        head = ("%d of %d segments imply end year %d (the rest: %s)"
                % (top_n, n, top_end, others))
    else:
        head = ("the %d segments do not agree (implied end years %s)"
                % (n, ", ".join(str(e) for e in ends)))
    if n < 4:
        tail = "; too few segments to test for an internal dating error"
    else:
        tail = "; no coherent one-year split, so no internal dating error is flagged"
    if best["t"] < 3.5 or best["p_bonf"] > 0.05:
        tail += (". Note the whole-series match itself is weak (t = %.2f, p = %.2g), "
                 "so the segment dates are unreliable" % (best["t"], best["p_bonf"]))
    return head + tail + "."


def xdate_floater(data: pd.DataFrame, series, series_name=None,
                  min_overlap=50, transform="pw", prewhiten=None, biweight=True,
                  corr="spearman", ar_max=10, first_aic_min=True,
                  plot=False, return_rwl=False,
                  segmented=False, segment_topk=5, verbose=True, **kwargs):
    """Estimate the calendar dating of a floating (undated) ring-width series.

    Slide the floating series against a dated master and score each offset by
    correlation, reporting the best-fit placement with crossdating statistics
    (t-value, effective df, Bonferroni-corrected p, isolation factor).

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The **dated** reference (year-indexed). A DataFrame is a collection of
        ring-width series, averaged into the master; a Series (or one-column
        DataFrame) is used directly as the master -- e.g. ``crn["std"]`` from
        ``dpl.read_crn``. For an already-residual/prewhitened chronology, pass
        ``transform="none"`` to avoid prewhitening twice. Years with no
        measurement in any series are kept as NaN on the calendar grid (they are
        never dropped), and every offset is scored on its complete pairs.
    series : sequence or pandas.Series/DataFrame
        The floating (undated) ring-width series -- ring values oldest to youngest;
        any calendar index is ignored. An interior NaN is an unmeasured ring and
        keeps its position (it is not compressed out).
    series_name : str or None, default None
        Label for the floating series. None takes it from the series itself (a named
        Series or one-column DataFrame header), falling back to "Unknown".
    min_overlap : int, default 50
        Minimum overlapping (pairwise-complete) rings required to score an offset.
    transform : {"pw", "fd", "none"}, default "pw"
        High-pass transform applied (after mean-normalization) before correlating,
        case-insensitive: ``"pw"`` AR prewhitening (`xdate.normalize_for_crossdating()`,
        the package-standard ``autoreg._prewhiten``), ``"fd"`` first differencing,
        or ``"none"``. First differencing sharpens the high-frequency signal but
        sharply reduces the effective degrees of freedom.
    prewhiten : bool, optional
        Deprecated alias: ``True`` -> ``transform="pw"``, ``False`` ->
        ``transform="none"``. Overrides ``transform`` if given.
    biweight : bool, default True
        Build the master with a Tukey biweight robust mean (else arithmetic mean).
    corr : {"spearman", "pearson", "kendall"}, default "spearman"
        Correlation used to score each offset (one-sided).
    ar_max : int or None, default 10
        AR order ceiling for ``transform="pw"``. The default follows COFECHA/ARSTAN
        (cap at 10); ``None`` uses dplR's ``floor(10*log10(n))``.
    first_aic_min : bool, default True
        Select the AR order at the FIRST local AIC minimum (COFECHA/ARSTAN rule)
        rather than the global AIC minimum (dplR's ``ar()``; set False).
    plot : bool, default False
        Plot the sliding t-value against the series' end year, marking the best fit.
    return_rwl : bool, default False
        Also return the floating series placed at its best-fit years (``placed``)
        and combined with the reference (``combined``).
    segmented : bool, default False
        Also date each overlapping segment of the floater independently against the
        master and check that they agree (COFECHA UDATE-style). Adds
        ``result["segments"]``, ``result["consensus"]``,
        ``result["segment_candidates"]`` and ``result["internal_error"]`` (a dict
        describing a detected missing/false ring, or None). A coherent one-year step
        between older and younger segments flags an internal dating error the
        whole-series fit cannot see; at least four segments are needed for that
        verdict. The single-best-t placement remains the primary result.
    segment_topk : int, default 5
        With ``segmented``, how many best placements to record per segment in
        ``result["segment_candidates"]``.
    verbose : bool, default True
        Print a short summary of the best-fit dating.

    Returns
    -------
    dict
        ``series_name``; ``floater_cor_stats`` (a DataFrame with one row per offset,
        highest t first, columns ``min_year, max_year, r, t, eff_df, p_bonf, n``);
        ``best`` (the top-scoring placement, incl. ``isolation_factor`` and
        ``one_over_p``); and, when ``return_rwl=True``, ``placed`` and ``combined``.

    Notes
    -----
    Ports dplR's ``xdate.floater`` and adds crossdating statistics (t-value,
    autocorrelation-adjusted effective degrees of freedom, a Bonferroni-corrected
    p-value, and an isolation factor; Baillie & Pilcher 1973, Wigley et al. 1987,
    Wilson 2026). The correlation scoring reuses dplPy's crossdating internals, so
    results are consistent with ``dpl.xdate``.

    References
    ----------
    .. [1] https://opendendro.github.io/dplPy/reference/crossdating/#xdate_floater
    """
    plot = _plot_kwarg(plot, kwargs)

    # The reference may be a whole collection (a DataFrame, master built by
    # averaging its series) or a single pre-built chronology. Accept that single
    # chronology as a pandas Series too -- e.g. crn["res"] -- coercing it to a
    # one-column frame so the master is just that series (transformed).
    if isinstance(data, pd.Series):
        data = data.to_frame()
    _require_dataframe(data)
    data = _require_year_index(data, "data")
    method = _normalize_corr(corr)
    if prewhiten is not None:                   # back-compat alias
        _warn("xdate_floater(): prewhiten= is deprecated; use transform='pw' or "
              "transform='none' instead.", DeprecationWarning)
        transform = "pw" if prewhiten else "none"
    transform = _normalize_choice(transform, _TRANSFORMS, "transform")

    # No explicit label: take one from the series itself (a named pandas Series, or
    # a one-column DataFrame's header), falling back to "Unknown" for a bare
    # list/array. An explicit series_name (e.g. "Alaska subfossil") always wins.
    if series_name is None:
        series_name = _series_label(series) or "Unknown"

    y_raw = _as_series_values(series)
    n_series = len(y_raw)                        # original (full timber) ring count
    if n_series == 0:
        raise ValueError("`series` contains no finite ring-width values.")

    # Master and floater both stay on their dense position grids with NaN in
    # place (audit A3); only leading/trailing NaN are trimmed, so the youngest
    # ring is the last position of ``y`` and index offsets are calendar offsets.
    x, yrs = _build_master(data, transform, biweight,
                           ar_max=ar_max, first_aic_min=first_aic_min)
    nx = len(x)
    y, _ = _trim_nan(_transform_series(y_raw, transform,
                                       ar_max=ar_max, first_aic_min=first_aic_min))
    ny = len(y)
    n_valid = int((~np.isnan(y)).sum())
    nx_valid = int((~np.isnan(x)).sum())

    if min_overlap > n_valid:
        raise ValueError("min_overlap (%d) must be <= the number of measured rings "
                         "after transform (%d)." % (min_overlap, n_valid))
    if min_overlap > nx_valid:
        raise ValueError("min_overlap (%d) exceeds the reference master length "
                         "(%d measured years)." % (min_overlap, nx_valid))

    # lag-1 autocorrelation of each (transformed) series -> effective-df reduction
    ac1_ref, ac1_und = _abs_ac1(x), _abs_ac1(y)
    ac_factor = (1.0 - ac1_ref * ac1_und) / (1.0 + ac1_ref * ac1_und)

    if verbose:
        print("Reference master years: %d to %d (%d)"
              % (int(yrs.min()), int(yrs.max()), nx))
        print("Floating series length: %d (transform '%s' -> %d)"
              % (n_series, transform, n_valid))
        print("Minimum overlap for search:", min_overlap)

    # Slide the floating series across the master over every offset with at least
    # `min_overlap` overlapping rings. `d` is the master index (0-based) that the
    # floater's OLDEST ring aligns to. It runs from the most-old placement (the
    # floater's youngest ring `min_overlap` rings into the master's old end) to the
    # most-young placement (the floater's oldest ring `min_overlap` rings from the
    # master's young end). Expressing the crawl this way handles every regime with
    # the same clamped slices -- floater inside the master, floater overhanging
    # either end, and (when the floater is LONGER than the master) the master lying
    # entirely inside the floater -- with no negative-index slicing. For ny <= nx
    # it reproduces dplR's three-regime crawl offset-for-offset.
    def _master_year(j):
        # calendar year at master index j, extrapolating on the master's 1-yr grid
        # when j falls beyond either end (an overhang placement).
        if j < 0:
            return int(yrs[0]) + j
        if j >= nx:
            return int(yrs[nx - 1]) + (j - (nx - 1))
        return int(yrs[j])

    # Each offset is scored on the pairwise-complete pairs of the overlap (NaN in
    # either the master or the floater simply drop out); ``n`` is that count.
    rows = []
    for d in range(nx - min_overlap, min_overlap - ny - 1, -1):
        lo, hi = max(0, d), min(nx - 1, d + ny - 1)
        if hi - lo + 1 < min_overlap:           # guard; the extremes hit it exactly
            continue
        xseg, yseg = x[lo:hi + 1], y[lo - d:hi - d + 1]
        n = int((~np.isnan(xseg) & ~np.isnan(yseg)).sum())
        if n < min_overlap:
            continue
        r = _fast_corr(xseg, yseg, method)
        max_year = _master_year(d + ny - 1)     # calendar year of the floater's youngest ring
        rows.append({"min_year": max_year - n_series + 1, "max_year": max_year,
                     "r": r, "n": n})

    stats = pd.DataFrame(rows, columns=["min_year", "max_year", "r", "n"])
    if stats.empty:
        raise ValueError("no offset overlaps the reference by at least min_overlap=%d "
                         "measured rings." % min_overlap)

    # crossdating statistics (Baillie & Pilcher t; effective df adjusted for
    # autocorrelation; one-tailed p, Bonferroni-corrected for the offsets tested)
    r = stats["r"].to_numpy(dtype=float)
    eff_df = stats["n"].to_numpy(dtype=float) * ac_factor
    with np.errstate(invalid="ignore", divide="ignore"):
        t = _bp_t(r, eff_df)
        p = scipy.stats.t.sf(t, df=eff_df)      # one-tailed p-value of t
    n_offsets = int(np.isfinite(t).sum())
    p_bonf = np.minimum(p * n_offsets, 1.0)
    stats["t"] = t
    stats["eff_df"] = eff_df
    stats["p_bonf"] = p_bonf
    stats = stats[["min_year", "max_year", "r", "t", "eff_df", "p_bonf", "n"]]
    stats = stats.sort_values("t", ascending=False,
                              na_position="last").reset_index(drop=True)

    # isolation factor: how many times larger the runner-up p is than the best p
    # (invariant to the Bonferroni constant, so computed from the raw t-p values)
    finite_p = np.sort(p[np.isfinite(p)])
    if finite_p.size >= 2 and finite_p[0] > 0:
        distinct = finite_p[finite_p > finite_p[0]]
        isolation = float(round(distinct[0] / finite_p[0])) if distinct.size else np.inf
    else:
        isolation = np.nan

    top = stats.iloc[0]
    best = {"min_year": int(top["min_year"]), "max_year": int(top["max_year"]),
            "r": float(top["r"]), "t": float(top["t"]),
            "eff_df": float(top["eff_df"]), "p_bonf": float(top["p_bonf"]),
            "one_over_p": (float("inf") if top["p_bonf"] == 0
                           else 1.0 / float(top["p_bonf"])),
            "isolation_factor": isolation, "n": int(top["n"])}

    # dplPy (unlike dplR) also scores placements that run past the reference into
    # future calendar years. That enriches the null distribution and is a useful
    # trap -- but if the WINNER lands there, the date is impossible and the match
    # must be treated as spurious. Flag it and scream (regardless of `verbose`).
    this_year = datetime.date.today().year
    best["date_warning"] = "future" if best["max_year"] > this_year else None
    if best["date_warning"] == "future":
        line = "=" * 78
        print("\n" + line)
        print("  !!!  WARNING: THE BEST-FIT DATE IS IN THE FUTURE  !!!")
        print("  '%s' dates its youngest ring to %d CE, past the present year (%d)."
              % (series_name, best["max_year"], this_year))
        print("  That is an IMPOSSIBLE calendar date -- treat this match as spurious.")
        print("  Check the series orientation (oldest -> youngest) and the reference.")
        print(line + "\n")

    if verbose:
        if_disp = (">1000" if np.isfinite(isolation) and isolation > 1000
                   else str(isolation))
        oop = best["one_over_p"]
        oop_disp = ">1 million" if oop > 1e6 else "%.3g" % oop
        print("\nBest match for '%s': %d to %d  (t = %.2f, r = %.3f, "
              "eff_df = %.0f, p_bonf = %.2e, 1/p = %s, IF = %s, overlap n = %d)"
              % (series_name, best["min_year"], best["max_year"], best["t"],
                 best["r"], best["eff_df"], best["p_bonf"], oop_disp,
                 if_disp, best["n"]))

    result = {"series_name": series_name, "floater_cor_stats": stats, "best": best}

    # Segment-consensus dating (COFECHA UDATE-style): date each overlapping segment
    # of the floater independently and look for agreement. This catches an internal
    # dating error (a missing or false ring) that a whole-series slide hides -- the
    # error splits the floater so older and younger segments date one year apart.
    if segmented:
        seg_df, consensus, candidates = _segment_consensus(
            x, yrs, y, method, _master_year, topk=segment_topk)
        diag = _diagnose_internal_error(seg_df)
        result["segments"] = seg_df
        result["consensus"] = consensus
        result["segment_candidates"] = candidates
        result["internal_error"] = diag
        if diag is not None:
            article = "a missing" if diag["kind"] == "missing" else "a false"
            line = "=" * 78
            print("\n" + line)
            print("  !!!  POSSIBLE INTERNAL DATING ERROR in '%s'  !!!" % series_name)
            print("  Segments do not agree on one date: younger rings date to %d CE,"
                  % diag["reliable_end"])
            print("  older rings to %d CE (one year %s) -- the signature of %s ring"
                  % (diag["other_end"],
                     "earlier" if diag["kind"] == "missing" else "later", article))
            print("  near ring ~%d of the series. The younger-segment date (%d) is the"
                  % (diag["near_ring"], diag["reliable_end"]))
            print("  reliable outer date; re-examine the measurements around that ring.")
            print(line + "\n")
        elif verbose:
            print("Segment consensus: " + _consensus_summary(seg_df, best))

    if return_rwl or plot:
        placed = pd.DataFrame(
            {series_name: y_raw},
            index=pd.Index(range(best["min_year"], best["min_year"] + n_series),
                           name=data.index.name or "Year"))
        result["placed"] = placed
        # a floater whose name already exists in the reference collection (e.g. a
        # leave-one-out test that did not drop the column) would otherwise crash the
        # join on the overlapping name; suffix the placed copy in that case.
        result["combined"] = data.join(placed, how="outer", rsuffix="_floater")

    if plot:
        _plot_floater(result, biweight=biweight)
        if segmented:
            _plot_segment_consensus(result)

    return result


def _plot_segment_consensus(result, show=True):
    """Staircase diagnostic: each segment's implied youngest-ring year against its
    position in the floater. A flat line means the floater dates internally
    consistently; a one-year step flags a missing/false ring at that ring."""
    import matplotlib.pyplot as plt
    from ._plot_style import style_axes, finalize_font, ACCENT, ACCENT_WARM

    seg = result.get("segments")
    if seg is None or seg.empty:
        return None
    diag = result.get("internal_error")
    mid = (seg["ring_lo"] + seg["ring_hi"]) / 2.0
    r = seg["best_r"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    # connect the votes, then mark each segment by how well it correlates
    ax.plot(mid, seg["impl_end"], "-", lw=0.8, color="0.6", zorder=1)
    sc = ax.scatter(mid, seg["impl_end"], c=r, cmap="viridis", vmin=0, vmax=1,
                    s=42, zorder=3, edgecolor="white", linewidth=0.4)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("segment correlation", fontsize=8)
    if diag is not None:
        ax.axhline(diag["reliable_end"], ls="--", lw=0.8, color=ACCENT,
                   label="reliable end %d" % diag["reliable_end"])
        ax.axvline(diag["near_ring"], ls=":", lw=1.0, color=ACCENT_WARM)
        ax.text(diag["near_ring"], ax.get_ylim()[1], " ~ring %d" % diag["near_ring"],
                va="top", ha="left", color=ACCENT_WARM, fontsize=8)
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_xlabel("ring number from pith  (1 = innermost, outward to bark; segment center)")
    ax.set_ylabel("implied youngest-ring year")
    title = "Segment consensus: " + str(result["series_name"])
    if diag is not None:
        title += "   —   possible %s ring near ring %d" % (diag["kind"],
                                                                diag["near_ring"])
    ax.text(0.0, 1.04, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, color=("#b3202c" if diag is not None else "0.15"),
            fontweight="bold")
    style_axes(ax, xgrid=True, ygrid=True)
    finalize_font(fig)
    if show:
        plt.show()
    return fig


def _plot_floater(result, biweight=True, show=True):
    """A four-panel dating figure after Wilson (2026): the sliding t-value and the
    sliding Bonferroni-adjusted p-value as stacked line plots against calendar
    year, and the distribution of all t-values (with the best marked) to the right.
    Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    from ._plot_style import style_axes, finalize_font, ACCENT, ACCENT_WARM

    stats = result["floater_cor_stats"]
    best = result["best"]
    name = result["series_name"]
    order = stats.sort_values("max_year")
    yr = order["max_year"].to_numpy(dtype=float)
    tval = order["t"].to_numpy(dtype=float)
    pval = order["p_bonf"].to_numpy(dtype=float)
    all_t = stats["t"].to_numpy(dtype=float)
    all_t = all_t[np.isfinite(all_t)]
    thr = 4.0 * np.nanstd(all_t)

    # House palette: data in accent blue, best-match highlight in brick (so it
    # contrasts with the data), significance/threshold lines as neutral guides.
    data_c, best_c, ref_c, thr_c = ACCENT, ACCENT_WARM, "0.55", "0.6"
    # display caps, matching Wilson's convention
    IF = best["isolation_factor"]
    if_disp = ">1000" if (np.isfinite(IF) and IF > 1000) else (
        "%d" % IF if np.isfinite(IF) else "NA")
    oop = best["one_over_p"]
    oop_disp = ">1 million" if oop > 1e6 else "%.3g" % oop
    p_disp = "<0.0001" if best["p_bonf"] < 1e-4 else "%.4f" % best["p_bonf"]

    def _titles(ax, main, sub):
        ax.text(0.0, 1.16, main, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11)
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, ha="left", va="bottom",
                color="0.4", fontsize=9)

    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1], height_ratios=[1, 1],
                          hspace=0.62, wspace=0.22, top=0.9, bottom=0.09,
                          left=0.06, right=0.98)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[1, 0], sharex=ax_t)
    ax_h = fig.add_subplot(gs[0, 1])
    ax_o = fig.add_subplot(gs[1, 1])

    # --- (1) sliding t-values ---
    ax_t.axhline(0, ls="--", lw=0.5, color="0.7")
    ax_t.axhline(thr, ls="--", lw=0.5, color=thr_c)
    ax_t.text(yr.min(), thr, " 4 STDEVs", va="bottom", ha="left",
              fontsize=8, color=thr_c)
    ax_t.plot(yr, tval, lw=0.4, color=data_c)
    ax_t.plot([best["max_year"]], [best["t"]], "o", color=best_c, ms=7, zorder=5)
    ax_t.set_ylabel("T-value")
    style_axes(ax_t, xgrid=True, ygrid=True)
    _titles(ax_t, "Sliding T values",
            "Strongest candidate outer-year date = %d CE    T = %.2f"
            % (best["max_year"], best["t"]))

    # --- (2) sliding Bonferroni-adjusted p-values (reversed log) ---
    pv = np.clip(pval, 1e-300, 1.0)
    ax_p.plot(yr, pv, lw=0.4, color=data_c)
    ax_p.set_yscale("log")
    # Inverted axis (small p at top). Always extend it past the most significant
    # reference threshold so the 0.05 / 0.0001 lines and their labels stay INSIDE
    # the panel -- otherwise, when the best p is poor (e.g. an undated floater with
    # min p ~0.7), those labels land far above the axis and (with bbox_inches=
    # "tight") balloon the figure with white space.
    top = min(pv.min(), 1e-4) * 0.2
    ax_p.set_ylim(1.0, max(top, 1e-300))
    ax_p.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    for ref, lab in ((0.05, "p = 0.05"), (1e-4, "p = 0.0001")):
        ax_p.axhline(ref, ls="--", lw=0.7, color=ref_c)
        ax_p.text(yr.min(), ref, " " + lab, va="bottom", ha="left",
                  fontsize=7, color=ref_c, clip_on=True)
    ax_p.set_ylabel("Adjusted p values")
    ax_p.set_xlabel("Calendar years CE")
    style_axes(ax_p, xgrid=True, ygrid=False)
    _titles(ax_p, "Sliding Bonferroni-adjusted p values",
            "p = %s    1/p = %s    IF = %s" % (p_disp, oop_disp, if_disp))

    # --- (3) distribution of all t-values, best marked ---
    ax_h.hist(all_t, bins=40, density=True, color=data_c, edgecolor="white",
              linewidth=0.3)
    ax_h.axvline(best["t"], ls="--", lw=1.2, color=best_c)
    ax_h.annotate("", xy=(best["t"], ax_h.get_ylim()[1] * 0.55),
                  xytext=(best["t"] - 0.18 * (all_t.max() - all_t.min()),
                          ax_h.get_ylim()[1] * 0.55),
                  arrowprops=dict(arrowstyle="->", color=best_c, lw=1.2))
    ax_h.set_xlabel("T-value")
    ax_h.set_ylabel("Density")
    style_axes(ax_h, xgrid=False, ygrid=True)
    _titles(ax_h, "T-value density distribution",
            "%d CE    T = %.2f" % (best["max_year"], best["t"]))

    # --- (4) reference chronology vs the best-placed floating series ---
    # Overlay the display chronology (a mean-normalized robust mean of the
    # reference series) and the floater placed at its best-fit years, over the
    # youngest ~50 yr of the overlap plus a short reference tail past the
    # floater's end. Both are z-scored over the overlap so their common
    # high-frequency signal is directly comparable regardless of amplitude.
    ref = result["combined"].drop(columns=[name])
    mvals, myears = _build_master(ref, "none", biweight)
    master_s = pd.Series(mvals, index=np.asarray(myears, dtype=int))
    placed_s = result["placed"][name].dropna()
    me, ms = int(best["max_year"]), int(best["min_year"])
    win_lo = max(ms, me - 49)                      # ~50 yr of overlap
    win_hi = min(int(master_s.index.max()), me + 5)   # + up to 5 yr of reference
    ov_m = master_s.loc[win_lo:me]
    ov_f = placed_s.loc[win_lo:me]
    if len(ov_m) >= 2 and len(ov_f) >= 2 and ov_m.std() > 0 and ov_f.std() > 0:
        m_disp = (master_s.loc[win_lo:win_hi] - ov_m.mean()) / ov_m.std()
        f_disp = (ov_f - ov_f.mean()) / ov_f.std()
        ax_o.axhline(0, ls="--", lw=0.5, color="0.7")
        ax_o.plot(m_disp.index, m_disp.values, lw=1.0, color=data_c,
                  label="Reference chronology")
        ax_o.plot(f_disp.index, f_disp.values, lw=1.0, color=best_c, label=name)
        if win_hi > me:                            # mark where the floater ends
            ax_o.axvline(me, ls=":", lw=0.8, color="0.6")
        ax_o.legend(loc="best", fontsize=8, frameon=False)
    else:
        ax_o.text(0.5, 0.5, "insufficient overlap to plot", transform=ax_o.transAxes,
                  ha="center", va="center", color="0.5", fontsize=9)
    ax_o.set_xlabel("Calendar years CE")
    ax_o.set_ylabel("z-score (over overlap)")
    style_axes(ax_o, xgrid=True, ygrid=True)
    _titles(ax_o, "Reference vs. best-placed floater",
            "%d–%d overlap" % (win_lo, me))

    if best.get("date_warning") == "future":
        fig.suptitle("%s   —   BEST DATE %d CE IS IN THE FUTURE (LIKELY SPURIOUS)"
                     % (name, best["max_year"]), color="#b3202c",
                     fontweight="bold", fontsize=13, y=1.0)
    else:
        fig.suptitle(name, color="0.15", fontweight="bold", fontsize=13, y=1.0)
    finalize_font(fig)
    if show:
        plt.show()
    return fig
