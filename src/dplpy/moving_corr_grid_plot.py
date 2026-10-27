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

# Title: moving_corr_grid_plot.py
# Project: OpenDendro dplPy
# Description: The three grid views for dpl.moving_corr in grid mode (a moving
#              seascorr): "small_multiples" (the default -- the honest stage-1
#              trace-with-band tiled across chosen seasons), "dominant" (which
#              season carries the signal over time), and "surface" (the stability
#              anomaly heatmap). None uses a per-cell significance mask; field
#              significance is carried as a grid-level verdict (Livezey-Chen) and a
#              per-season annotation (FDR), following Gelman & Stern (2006).
#
#              The surface uses matplotlib's RdBu_r (ColorBrewer, colourblind-safe:
#              red = positive correlation, white = 0, blue = negative) on the raw
#              moving correlation; mode="anomaly" instead maps the departure from
#              each season's full-period correlation. The dominant-season view uses
#              Crameri's "lajolla" (perceptually uniform, colourblind-safe), embedded
#              as compact control points so dplPy needs no extra dependency.
#              Crameri, F. (2018), Scientific colour maps, Zenodo,
#              doi:10.5281/zenodo.1243862; Crameri, Shephard & Heron (2020),
#              Nature Communications 11, 5444.

import numpy as np

from ._plot_style import style_axes, finalize_font, ACCENT
from .detrend import _finish_figure

_MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_SHADE = "#bcd0e6"
_FDR = "#c1442e"

# Crameri "lajolla" (sequential, for the dominant-season strength), 32 control
# points. The surface uses matplotlib's RdBu_r (ColorBrewer, colourblind-safe).
_LAJOLLA = [
    (0.0988,0.0997,0.0001),(0.1258,0.1093,0.0156),(0.1532,0.1190,0.0318),
    (0.1833,0.1294,0.0502),(0.2220,0.1425,0.0706),(0.2615,0.1559,0.0894),
    (0.3060,0.1709,0.1120),(0.3557,0.1878,0.1382),(0.4173,0.2084,0.1707),
    (0.4763,0.2275,0.2006),(0.5385,0.2460,0.2289),(0.6032,0.2628,0.2529),
    (0.6767,0.2797,0.2730),(0.7390,0.2961,0.2854),(0.7930,0.3195,0.2943),
    (0.8337,0.3525,0.3010),(0.8622,0.3973,0.3065),(0.8765,0.4382,0.3102),
    (0.8862,0.4776,0.3133),(0.8943,0.5155,0.3162),(0.9028,0.5570,0.3193),
    (0.9103,0.5934,0.3220),(0.9178,0.6298,0.3247),(0.9254,0.6668,0.3278),
    (0.9344,0.7100,0.3334),(0.9433,0.7512,0.3458),(0.9537,0.7966,0.3764),
    (0.9656,0.8447,0.4355),(0.9786,0.8962,0.5309),(0.9880,0.9354,0.6240),
    (0.9951,0.9687,0.7129),(1.0000,0.9978,0.7942),
]


def _cmap(name, points):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, points, N=256)


def moving_corr_grid_plot(result, kind="small_multiples", show=True, **kwargs):
    """Dispatch a MovingCorrGridResult to one of the three grid views."""
    if kind == "small_multiples":
        return _small_multiples(result, show=show, **kwargs)
    if kind == "dominant":
        return _dominant(result, show=show, **kwargs)
    if kind == "surface":
        return _surface(result, show=show, **kwargs)
    raise ValueError("kind must be 'small_multiples', 'dominant' or 'surface'.")


# --------------------------------------------------------------------------- #
# small multiples (default)
# --------------------------------------------------------------------------- #

def _resolve_seasons(result, seasons):
    """Return a list of row indices into result.grid for the requested seasons,
    or the strongest-per-length fallback (with a warning) when none are given."""
    g = result.grid
    if seasons is None:
        import warnings
        picks = []
        for m in sorted(set(g["season_length"])):
            sub = g[(g["season_length"] == m) & (~g["prior"])]
            if len(sub):
                picks.append(int(sub["overall_r"].abs().idxmax()))
        warnings.warn("moving_corr: no seasons= given for the small-multiples "
                      "view; showing the strongest ending month per season length. "
                      "Pass seasons=[(length, end_month), ...] to choose.",
                      UserWarning, stacklevel=3)
        return picks
    picks = []
    for item in seasons:
        m, mo = int(item[0]), int(item[1])
        prior = bool(item[2]) if len(item) > 2 else False
        sub = g[(g["season_length"] == m) & (g["end_month"] == mo)
                & (g["prior"] == prior)]
        if not len(sub):
            raise ValueError("season (length=%d, end_month=%d) is not in the grid."
                             % (m, mo))
        picks.append(int(sub.index[0]))
    return picks


def _small_multiples(result, seasons=None, ci="fisher", ci_level=None, show=True):
    import matplotlib.pyplot as plt
    from .moving_corr import _fisher_ci
    from .climate import _eff_n

    picks = _resolve_seasons(result, seasons)
    n = len(picks)
    ncols = 2 if n == 4 else min(3, n)
    nrows = int(np.ceil(n / ncols))
    level = ci_level if ci_level is not None else result.settings["ci_level"]
    window = result.settings["window"]
    t = result.pos_years.astype(float)
    r1x = result.r1_chronology

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.5 * nrows + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, j in zip(axes.ravel(), picks):
        ax.set_visible(True)
        r = result.cube[j]
        if ci == "fisher":
            rc = r1x * result.r1_climate[j]
            f = (1.0 - rc) / (1.0 + rc) if np.isfinite(rc) else 1.0
            neff = _eff_n(window, min(max(f, 0.0), 1.0))
            lo, hi = _fisher_ci(r, neff, level)
            ax.fill_between(t, lo, hi, color=_SHADE)
        ax.axhline(0, color="black", lw=0.7)
        ax.axhline(result.overall[j], color="0.55", lw=0.9, ls=":")
        ax.plot(t, r, color=ACCENT, lw=2.0)
        lab = str(result.grid["label"].iloc[j])
        ax.set_title(lab + ("  *" if result.fdr_sig[j] else ""),
                     fontsize=9.5, loc="left")
        ax.set_ylim(-0.85, 1.0)
        style_axes(ax)
    for c in range(ncols):
        col = [axes[r, c] for r in range(nrows) if axes[r, c].get_visible()]
        if col:
            col[-1].set_xlabel("Year")
            col[-1].tick_params(labelbottom=True)
    for r in range(nrows):
        if axes[r, 0].get_visible():
            axes[r, 0].set_ylabel("Correlation")

    fig.suptitle("Moving correlation by season  -  estimate and per-window "
                 "uncertainty  (window %d, step %d)"
                 % (result.settings["window"], result.settings["offset"]),
                 fontsize=11.5, x=0.06, ha="left")
    fig.text(0.06, 0.005, "dotted line = full-period r.   * = FDR-significant "
             "season (%d of %d).   grid Livezey-Chen field test p = %.3f."
             % (result.field["n_fdr"], len(result.grid), result.field["lc_p"]),
             fontsize=8, color="0.3")
    finalize_font(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    result.fig, result.axes = fig, axes
    if show:
        _finish_figure(fig)
    return fig


# --------------------------------------------------------------------------- #
# dominant-season trace
# --------------------------------------------------------------------------- #

def _dominant(result, season_length=None, show=True):
    import matplotlib.pyplot as plt
    g = result.grid
    lengths = sorted(set(g["season_length"]))
    if season_length is None:
        season_length = 3 if 3 in lengths else lengths[0]
    idx = g.index[g["season_length"] == season_length].to_numpy()
    if idx.size == 0:
        raise ValueError("season_length %d is not in the grid." % season_length)
    emo = g["end_month"].to_numpy()[idx].astype(float)
    pri = g["prior"].to_numpy()[idx]
    ypos = np.where(pri, emo - 12, emo)
    sub = result.cube[idx]
    best = np.argmax(sub, axis=0)
    bestr = sub[best, np.arange(sub.shape[1])]
    besty = ypos[best]
    t = result.pos_years.astype(float)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k in range(sub.shape[1]):
        near = np.where(sub[:, k] >= 0.9 * bestr[k])[0]
        if near.size:
            ax.plot([t[k], t[k]], [ypos[near].min(), ypos[near].max()],
                    color="0.8", lw=2, zorder=1, solid_capstyle="round")
    sc = ax.scatter(t, besty, c=bestr, cmap=_cmap("lajolla", _LAJOLLA).reversed(),
                    vmin=0, vmax=1, s=28, zorder=3, edgecolor="white", linewidth=0.3)
    lo, hi = int(np.floor(ypos.min())), int(np.ceil(ypos.max()))
    ticks = list(range(max(lo, -2), min(hi, 12) + 1))
    ax.set_yticks(ticks)
    ax.set_yticklabels([_MONTH[v - 1] if v >= 1 else _MONTH[v + 11] + "*"
                        for v in ticks], fontsize=7)
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xlabel("Year (window %s)" % result.settings["position"])
    ax.set_ylabel("Ending month of strongest %d-month season" % season_length)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("correlation of the winning season  (darker = stronger)", fontsize=8)
    ax.set_title("Which season carries the signal, over time  (%d-month seasons, "
                 "window %d, step %d)" % (season_length, result.settings["window"],
                 result.settings["offset"]), fontsize=11, loc="left")
    ax.text(0.0, -0.17, "grey bar = other seasons within 90%% of the best "
            "(narrow = sharp choice, wide = ambiguous).   grid field test p = %.3f."
            % result.field["lc_p"], transform=ax.transAxes, fontsize=8,
            color="0.3", va="top")
    style_axes(ax)
    finalize_font(fig)
    fig.tight_layout()
    result.fig, result.axes = fig, ax
    if show:
        _finish_figure(fig)
    return fig


# --------------------------------------------------------------------------- #
# stability surface
# --------------------------------------------------------------------------- #

def _surface(result, show=True, mode="correlation"):
    import matplotlib.pyplot as plt
    if mode not in ("correlation", "anomaly"):
        raise ValueError("mode must be 'correlation' or 'anomaly'.")
    g = result.grid
    lengths = sorted(set(g["season_length"]))
    t = result.pos_years.astype(float)
    field = result.cube if mode == "correlation" else result.anomaly
    vmax = float(np.nanpercentile(np.abs(field), 98))
    div = plt.cm.RdBu_r                       # red = positive r, white = 0, blue = negative

    fig, axes = plt.subplots(len(lengths), 1, figsize=(9, 2.3 * len(lengths) + 1),
                             sharex=True, squeeze=False,
                             gridspec_kw=dict(hspace=0.22))
    axes = axes.ravel()
    im = None
    for ax, m in zip(axes, lengths):
        idx = g.index[g["season_length"] == m].to_numpy()
        A = field[idx]
        im = ax.pcolormesh(t, np.arange(len(idx)), A, cmap=div,
                           vmin=-vmax, vmax=vmax, shading="nearest")
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels([str(g["label"].iloc[j]) for j in idx], fontsize=6)
        ax.set_ylabel("%d-mo" % m, fontsize=9)
        ax.invert_yaxis()
        for tl, j in zip(ax.get_yticklabels(), idx):
            if result.fdr_sig[j]:
                tl.set_color(_FDR)
                tl.set_fontweight("bold")
        ax.tick_params(length=0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("Year (window %s)" % result.settings["position"])
    cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    win = "  (window %d, step %d)" % (result.settings["window"],
                                      result.settings["offset"])
    if mode == "correlation":
        cb.set_label("moving correlation", fontsize=8)
        title = "Moving-correlation surface  -  chronology vs season" + win
    else:
        cb.set_label("moving r  minus  full-period r   (departure from baseline)",
                     fontsize=8)
        title = ("Signal stability surface  -  departure from baseline" + win)
    fig.suptitle(title, fontsize=11, x=0.09, ha="left")
    fig.text(0.09, 0.012, "season labels in red = field-significant by FDR (%d of "
             "%d).   Livezey-Chen field test: %d locally-significant cells vs ~%d "
             "expected, p = %.3f." % (result.field["n_fdr"], len(result.grid),
             result.field["lc_K_obs"], int(np.median(result.field["lc_K_null"])),
             result.field["lc_p"]), fontsize=8, color="0.3")
    finalize_font(fig)
    result.fig, result.axes = fig, axes
    if show:
        _finish_figure(fig)
    return fig
