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

# Title: xdate_undated.py
# Description: Crossdate a set of *undated* ("floating") ring-width series against
#   ONE ANOTHER -- no dated reference exists -- and assemble them into a single
#   relatively-dated floating chronology. This is the workflow for sub-fossil or
#   historical material where a whole stand overlaps in time but does not crossdate
#   into any absolutely-dated regional chronology (e.g. river-drowned trees): the
#   samples are positioned relative to each other and the earliest inferred ring is
#   given an arbitrary Year 1.
#
#   It is the dplPy analogue of COFECHA's UFLOAT routine (Holmes 1983), which is
#   what COFECHA runs when only an undated-series file is supplied and the dated
#   prompt is left blank. UFLOAT matches every floating series against every other
#   at all positions and prints the best matches; xdate_undated does that pairwise
#   step and then goes further, resolving those pairwise offsets into one consistent
#   relative timeline by growing a master chronology.
#
#   Method:
#     1. Transform every series the same high-pass way used elsewhere in dplPy
#        (divide by its mean, then prewhiten / difference), keeping the array at raw
#        ring length so array index == ring position across series.
#     2. Pairwise pass: match every pair at all offsets with at least `min_overlap`
#        overlapping rings; record each pair's best lag, correlation and t. This is
#        the UFLOAT table (`result["pairwise"]`).
#     3. Grow-the-master: seed with the strongest pair, then add each remaining
#        series at the position implied by its strongest already-placed partner,
#        refined a few rings against the running mean. The mean acts as the consensus
#        and absorbs conflicting pairwise offsets. A series is added only if it meets
#        `crit_t`; the join order and the t at which each joined are recorded.
#     4. Leave-one-out verification: slide each placed series against the mean of the
#        OTHERS across all positions, recording where it lands, its t there, and how
#        isolated that position is from the next-best (a decisiveness check). Low
#        isolation flags a placement to review by eye, exactly as a dendrochronologist
#        would confirm a marginal match.
#     5. Anchor the earliest ring at `anchor_year` (default 1) and return the placed
#        raw series and their mean as the floating chronology.
#
#   The single-best-t placement is not blindly trusted: `result["verify"]` gives the
#   isolation and a `review` flag so weak/ambiguous placements are visible rather than
#   silently wrong.
#
# References:
#   Holmes, R.L., 1983. Computer-assisted quality control in tree-ring dating and
#     measurement. Tree-Ring Bulletin 43, 69-78.  (COFECHA / UFLOAT)
#   Baillie, M.G.L., Pilcher, J.R., 1973. A simple cross-dating program for
#     tree-ring research. Tree-Ring Bulletin 33, 7-14.  (the t statistic)
#
# example usage:
#   >>> import dplpy as dpl
#   >>> floats = dpl.readers("drowned_stand.rwl")   # undated collection
#   >>> res = dpl.xdate_undated(floats, make_plot=True)
#   >>> res["offsets"]       # each series' relative start/end year (earliest = 1)
#   >>> res["chronology"]    # the mean floating chronology
#   >>> res["verify"]        # isolation and review flags per series

import numpy as np
import pandas as pd

from ._validate import _normalize_corr
from .xdate import _corr_pval
from .xdate_floater import _transform_series

_TRANSFORMS = ("pw", "fd", "none")


def _t_from_r(r, n):
    """Baillie-Pilcher t from a correlation r over n overlapping rings."""
    rr = min(max(float(r), -0.999999), 0.999999)
    return rr * np.sqrt((n - 2) / (1 - rr * rr))


def _tr_aligned(raw, transform):
    """Transform a series but keep it at RAW ring length, so array index == ring
    position. Prewhitening/differencing drop the oldest rings (residuals are
    end-aligned), so pad the front with NaN to keep every series on the same footing.
    Without this, series with different AR orders would be mis-registered by a few
    rings."""
    w = _transform_series(raw, transform)          # front-trimmed, end-aligned
    out = np.full(len(raw), np.nan)
    if len(w):
        out[len(raw) - len(w):] = w
    return out


def _best_offset(a, b, method, min_overlap):
    """Best lag d (start of b relative to start of a) matching aligned arrays a, b
    (which may carry leading NaN). Returns (d, n_overlap, r, t) or None."""
    na, nb = len(a), len(b)
    best = None
    for d in range(-(nb - min_overlap), (na - min_overlap) + 1):
        lo = max(0, d)
        hi = min(na, d + nb)
        aw = a[lo:hi]
        bw = b[lo - d:hi - d]
        ok = ~np.isnan(aw) & ~np.isnan(bw)
        n = int(ok.sum())
        if n < min_overlap:
            continue
        r, _ = _corr_pval(aw[ok], bw[ok], method)
        if not np.isfinite(r):
            continue
        t = _t_from_r(r, n)
        if best is None or t > best[3]:
            best = (d, n, round(float(r), 3), round(float(t), 2))
    return best


def _coerce_collection(data):
    """Return an ordered dict {name: 1-D float array of non-NaN ring widths}."""
    series = {}
    if isinstance(data, pd.DataFrame):
        for c in data.columns:
            series[str(c)] = data[c].dropna().to_numpy(dtype=float)
    elif isinstance(data, dict):
        for k, v in data.items():
            arr = np.asarray(v, dtype=float)
            series[str(k)] = arr[~np.isnan(arr)]
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data, start=1):
            arr = np.asarray(v, dtype=float)
            series["S%d" % i] = arr[~np.isnan(arr)]
    else:
        raise TypeError("`data` must be a DataFrame, dict, or list of series.")
    if len(series) < 2:
        raise ValueError("Need at least two undated series to crossdate.")
    return series


def xdate_undated(data, transform="pw", corr="spearman", min_overlap=50,
                  crit_t=4.0, anchor_year=1, min_isolation=1.0,
                  make_plot=False, verbose=True):
    """Crossdate a set of undated ("floating") ring-width series against one another
    and assemble a single relatively-dated floating chronology (COFECHA UFLOAT-style).

    Use this when no dated reference exists but the samples overlap in time -- e.g. a
    stand of sub-fossil or river-drowned trees. The series are positioned relative to
    each other and the earliest inferred ring is set to ``anchor_year`` (default 1).

    Parameters
    ----------
    data : pandas.DataFrame, dict, or list
        The undated collection. A DataFrame is read as one column per series (the
        index is ignored -- these are undated); a dict maps names to ring-width
        arrays; a bare list is auto-named ``S1``, ``S2``, ...
    transform : {"pw", "fd", "none"}, default "pw"
        High-pass transform applied before matching: ``pw`` prewhitens (AR), ``fd``
        first-differences, ``none`` uses mean-normalized widths.
    corr : {"spearman", "pearson", "kendall"}, default "spearman"
        Correlation used to score every offset.
    min_overlap : int, default 50
        Minimum overlapping rings required to score a match (COFECHA UFLOAT uses 40).
    crit_t : float, default 4.0
        A series joins the master only if its best t against the running mean reaches
        this. Higher is safer against spurious placement; COFECHA flags t > 3.5.
    anchor_year : int, default 1
        Relative calendar year assigned to the earliest ring of the assembled
        chronology.
    min_isolation : float, default 1.0
        A placement whose leave-one-out isolation (the t gap to the next-best,
        well-separated position) falls below this is flagged for review.
    make_plot : bool, default False
        Draw a placement chart (each series positioned on the relative axis, coloured
        by isolation) with the mean floating chronology beneath it.
    verbose : bool, default True
        Print a short summary.

    Returns
    -------
    dict with:
        ``pairwise`` : DataFrame of every pair's best match (``series_i``,
            ``series_j``, ``lag``, ``n``, ``r``, ``t``) -- the UFLOAT table.
        ``offsets`` : DataFrame (indexed by series) of ``start_year``, ``end_year``,
            ``n`` on the relative axis (earliest ring = ``anchor_year``).
        ``placed`` : DataFrame of the raw ring widths positioned on the relative axis.
        ``chronology`` : the mean floating chronology (Series on the relative axis).
        ``order`` : DataFrame of the join order with the ``join_t`` at which each
            series was added.
        ``verify`` : DataFrame of the leave-one-out check per placed series
            (``loo_best_pos``, ``loo_t``, ``isolation``, ``agrees``, ``review``).
        ``unplaced`` : list of series that never reached ``crit_t``.
    """
    method = _normalize_corr(corr)
    if transform not in _TRANSFORMS:
        raise ValueError("transform must be one of %s" % (_TRANSFORMS,))
    raw = _coerce_collection(data)
    names = list(raw)
    tr = {nm: _tr_aligned(raw[nm], transform) for nm in names}
    R = 6   # local refine half-window (rings) around a pairwise-predicted position

    # 1) all-pairs best offsets (full scan, once)
    pair_rows = []
    lag_of, t_of = {}, {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            res = _best_offset(tr[ni], tr[nj], method, min_overlap)
            if res is None:
                continue
            d, n, r, t = res
            pair_rows.append({"series_i": ni, "series_j": nj, "lag": d,
                              "n": n, "r": r, "t": t})
            lag_of[(ni, nj)] = d; lag_of[(nj, ni)] = -d
            t_of[(ni, nj)] = t; t_of[(nj, ni)] = t
    if not pair_rows:
        raise ValueError("No pair of series overlaps by at least min_overlap=%d "
                         "rings at any offset; cannot crossdate." % min_overlap)
    pairwise = pd.DataFrame(pair_rows).sort_values("t", ascending=False) \
        .reset_index(drop=True)

    def running_mean(placed_pos):
        lo = min(placed_pos[nm] for nm in placed_pos)
        hi = max(placed_pos[nm] + len(tr[nm]) for nm in placed_pos)
        acc = np.zeros(hi - lo); cnt = np.zeros(hi - lo)
        for nm in placed_pos:
            s = placed_pos[nm] - lo; seg = tr[nm]
            acc[s:s + len(seg)] += np.nan_to_num(seg)
            cnt[s:s + len(seg)] += ~np.isnan(seg)
        with np.errstate(invalid="ignore"):
            M = np.where(cnt > 0, acc / np.where(cnt == 0, 1, cnt), np.nan)
        return lo, M

    def score_at(nm, p, lo, M):
        """(n, r, t) for series nm placed with start at absolute position p vs mean M
        spanning [lo, lo+len(M)), or None if the overlap is too small."""
        b = tr[nm]; nb = len(b); nM = len(M)
        d = p - lo
        clo = max(0, d); chi = min(nM, d + nb)
        if chi - clo < min_overlap:
            return None
        mw = M[clo:chi]; bw = b[clo - d:chi - d]
        ok = ~np.isnan(mw) & ~np.isnan(bw)
        n = int(ok.sum())
        if n < min_overlap:
            return None
        r, _ = _corr_pval(mw[ok], bw[ok], method)
        if not np.isfinite(r):
            return None
        return n, round(float(r), 3), round(float(_t_from_r(r, n)), 2)

    # 2) seed with the strongest pair
    top = pairwise.iloc[0]
    pos = {top["series_i"]: 0, top["series_j"]: int(top["lag"])}
    order = [(top["series_i"], np.nan, np.nan), (top["series_j"], top["t"], top["n"])]

    # 3) grow the master: place each unplaced series from its placed partners, then
    #    refine locally against the mean. Add the highest-t candidate each round.
    unplaced = [nm for nm in names if nm not in pos]
    while unplaced:
        lo, M = running_mean(pos)
        cand = None                       # (name, pos, n, r, t)
        for nm in unplaced:
            preds = {}
            for p in pos:
                if (p, nm) in lag_of:
                    p0 = pos[p] + lag_of[(p, nm)]
                    preds[p0] = max(preds.get(p0, 0.0), t_of[(p, nm)])
            if not preds:
                continue
            best = None
            for p0 in preds:
                for delta in range(-R, R + 1):
                    sc = score_at(nm, p0 + delta, lo, M)
                    if sc is None:
                        continue
                    n, r, t = sc
                    if best is None or t > best[3]:
                        best = (p0 + delta, n, r, t)
            if best is not None and (cand is None or best[3] > cand[4]):
                cand = (nm, best[0], best[1], best[2], best[3])
        if cand is None or cand[4] < crit_t:
            break
        pos[cand[0]] = cand[1]
        order.append((cand[0], cand[4], cand[2]))
        unplaced.remove(cand[0])

    # 3b) leave-one-out verification + isolation
    SEP = 20
    verify_rows = []
    for nm in pos:
        others = [o for o in pos if o != nm]
        if not others:
            continue
        olo, M = running_mean({o: pos[o] for o in others})
        b = tr[nm]; nb = len(b); nM = len(M)
        scan = []
        for d in range(-(nb - min_overlap), (nM - min_overlap) + 1):
            sc = score_at(nm, olo + d, olo, M)
            if sc is not None:
                scan.append((olo + d, sc[2]))
        if not scan:
            verify_rows.append({"series": nm, "loo_best_pos": np.nan, "loo_t": np.nan,
                                "isolation": np.nan, "agrees": False, "review": True})
            continue
        scan.sort(key=lambda z: -z[1])
        best_pos, best_t = scan[0]
        second_t = next((t for p, t in scan[1:] if abs(p - best_pos) >= SEP), 0.0)
        iso = round(float(best_t - second_t), 2)
        agrees = bool(best_pos == pos[nm])
        verify_rows.append({"series": nm, "loo_best_pos": int(best_pos),
                            "loo_t": round(float(best_t), 2), "isolation": iso,
                            "agrees": agrees,
                            "review": bool((iso < min_isolation) or (not agrees)
                                           or (best_t < crit_t))})

    # 4) anchor earliest ring at anchor_year
    shift = anchor_year - min(pos[nm] for nm in pos)
    start_year = {nm: pos[nm] + shift for nm in pos}

    # 5) placed raw matrix + mean floating chronology
    lo = min(start_year.values())
    hi = max(start_year[nm] + len(raw[nm]) for nm in start_year)
    idx = pd.Index(range(lo, hi), name="rel_year")
    placed = pd.DataFrame(index=idx)
    for nm in start_year:
        col = pd.Series(raw[nm],
                        index=range(start_year[nm], start_year[nm] + len(raw[nm])))
        placed[nm] = col.reindex(idx)
    chronology = placed.mean(axis=1)
    chronology.name = "floating_chronology"

    offsets = pd.DataFrame({
        "start_year": pd.Series(start_year),
        "end_year": pd.Series({nm: start_year[nm] + len(raw[nm]) - 1
                               for nm in start_year}),
        "n": pd.Series({nm: len(raw[nm]) for nm in start_year}),
    }).sort_values("start_year")
    order_df = pd.DataFrame(order, columns=["series", "join_t", "join_n"])
    verify = pd.DataFrame(
        verify_rows,
        columns=["series", "loo_best_pos", "loo_t", "isolation", "agrees", "review"])
    if not verify.empty:
        # report the verification position on the same anchored axis
        verify["loo_best_pos"] = verify["loo_best_pos"] + shift

    result = {"pairwise": pairwise, "offsets": offsets, "placed": placed,
              "chronology": chronology, "order": order_df, "verify": verify,
              "unplaced": unplaced}

    if verbose:
        n_review = int(verify["review"].sum()) if not verify.empty else 0
        span_lo, span_hi = int(chronology.index.min()), int(chronology.index.max())
        line = "-" * 60
        print(line)
        print(" Undated-series crossdating  (grow-the-master, %s, t≥%.1f)"
              % (transform, crit_t))
        print(line)
        print("  Series provided        : %d" % len(names))
        print("  Placed                 : %d" % len(start_year))
        if unplaced:
            print("  Unplaced               : %d  (%s)"
                  % (len(unplaced), ", ".join(unplaced)))
        else:
            print("  Unplaced               : 0")
        print("  Floating chronology    : year %d to %d  (%d yrs)"
              % (span_lo, span_hi, span_hi - span_lo + 1))
        print("  Placements to review   : %d  (isolation < %.1f or ambiguous)"
              % (n_review, min_isolation))
        if n_review:
            flagged = list(verify.loc[verify["review"], "series"])
            print("    -> %s" % ", ".join(flagged))
        print(line)

    if make_plot:
        _plot_undated(result, anchor_year=anchor_year)

    return result


def _plot_undated(result, anchor_year=1):
    """Placement chart: each placed series as a horizontal bar on the relative axis,
    coloured by leave-one-out isolation (grey = flagged for review), with the mean
    floating chronology beneath."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from ._plot_style import style_axes, finalize_font, ACCENT

    offsets = result["offsets"]
    verify = result["verify"].set_index("series")
    chron = result["chronology"]

    iso = verify["isolation"].reindex(offsets.index)
    review = verify["review"].reindex(offsets.index).fillna(True)
    imax = float(np.nanmax(iso.values)) if np.isfinite(iso.values).any() else 1.0
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=max(imax, 1.0))

    fig = plt.figure(figsize=(10, max(3.5, 0.32 * len(offsets) + 2.2)))
    gs = GridSpec(2, 1, height_ratios=[len(offsets) + 1, 4], hspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0])
    axc = fig.add_subplot(gs[1], sharex=ax)

    any_review = False
    for row, nm in enumerate(offsets.index):
        s, e = offsets.loc[nm, "start_year"], offsets.loc[nm, "end_year"]
        if bool(review.loc[nm]) or not np.isfinite(iso.loc[nm]):
            color = "0.6"; any_review = True
        else:
            color = cmap(norm(iso.loc[nm]))
        ax.plot([s, e], [row, row], "-", lw=4, color=color, solid_capstyle="butt")
    ax.set_yticks(range(len(offsets)))
    ax.set_yticklabels(offsets.index, fontsize=7)
    ax.set_ylim(-1, len(offsets))
    ax.invert_yaxis()
    ax.set_ylabel("series (grey = review)")
    ax.text(0.0, 1.02, "Undated series positioned relative to one another",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
            fontweight="bold", color="0.15")
    style_axes(ax, xgrid=True, ygrid=False)
    plt.setp(ax.get_xticklabels(), visible=False)

    axc.plot(chron.index, chron.values, "-", lw=0.9, color=ACCENT)
    axc.set_xlabel("relative year  (earliest ring = %d)" % anchor_year)
    axc.set_ylabel("mean width")
    axc.text(0.0, 1.02, "Mean floating chronology", transform=axc.transAxes,
             ha="left", va="bottom", fontsize=9, color="0.3")
    style_axes(axc, xgrid=True, ygrid=True)

    # colorbar for the isolation scale (shared across both panels so they stay aligned)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=[ax, axc], pad=0.02, fraction=0.045)
    cb.set_label("placement isolation\n(t gap to next-best position)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if any_review:
        ax.legend([Line2D([0], [0], color="0.6", lw=4)],
                  ["flagged for review"], loc="lower right", fontsize=7,
                  frameon=False)

    finalize_font(fig)
    plt.show()
    return fig
