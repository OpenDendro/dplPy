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

# Title: seascorr_plot.py
# Description: The one figure seascorr draws -- Figure 3 of Meko, Touchan &
#   Anchukaitis (2011): two stacked bar charts, the simple correlations of the
#   chronology with the primary climate variable (top) and the partial
#   correlations with the secondary variable (bottom), across the four
#   season-lengths x 14 ending months, color-coded by significance. Styled
#   per-Axes via _plot_style, like the rest of dplPy's plots.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ._plot_style import style_axes, finalize_font

_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# face colors for the three significance states, by variable family. The
# convention (Meko et al. 2011) ties color to the variable, not the panel:
# temperature is red, moisture (precip / PDSI / ...) is blue, whichever is
# primary. Auto-detected from the variable name, overridable per panel.
_BLUE = {"ns": "white", "s05": "#9ecae1", "s01": "#08519c"}
_RED = {"ns": "white", "s05": "#fcae91", "s01": "#a50f15"}
_BW = {"ns": "white", "s05": "#bdbdbd", "s01": "#525252"}
_FAMILIES = {"blue": _BLUE, "red": _RED}
_EDGE = "#555555"
_GROUP_GAP = 2.0

_RED_WORDS = ("temp", "tmean", "tmax", "tmin", "tavg", "tmp", "degc")
_BLUE_WORDS = ("precip", "prcp", "ppt", "rain", "moist", "pdsi", "scpdsi",
               "spei", "spi", "drought", "swe", "runoff", "streamflow",
               "discharge", "flow")


def _classify(name):
    """Guess a color family from a variable name: 'red' for temperature,
    'blue' for moisture, or None if it matches neither."""
    s = str(name).lower()
    if any(w in s for w in _RED_WORDS):
        return "red"
    if any(w in s for w in _BLUE_WORDS):
        return "blue"
    return None


def _resolve_cmap(explicit, name, default, color):
    """Pick the color family for a panel: the explicit override if given, else
    the name-based guess, else the positional default. B/W when color=False."""
    if not color:
        return _BW
    if explicit is not None:
        key = str(explicit).strip().lower()
        if key not in _FAMILIES:
            raise ValueError("color must be 'red' or 'blue', got " + repr(explicit) + ".")
        return _FAMILIES[key]
    guess = _classify(name)
    return _FAMILIES[guess] if guess is not None else _FAMILIES[default]


def _nice_ylim(vals):
    """Y-limits locked to [-0.3, 0.3], expanding (rounded out to 0.1, with a
    little headroom) only when correlations exceed that range."""
    v = np.asarray(vals, dtype=float)
    dmin = min(float(np.nanmin(v)), 0.0)
    dmax = max(float(np.nanmax(v)), 0.0)
    lo = min(-0.3, dmin)
    hi = max(0.3, dmax)
    if lo < -0.3:
        lo = np.floor((lo - 0.02) / 0.1) * 0.1
    if hi > 0.3:
        hi = np.ceil((hi + 0.02) / 0.1) * 0.1
    return lo, hi


def _bar_faces(sig05, sig01, cmap):
    return [cmap["s01"] if s1 else (cmap["s05"] if s5 else cmap["ns"])
            for s5, s1 in zip(sig05, sig01)]


def _layout(table, season_lengths, n_end=14):
    """x position of every bar, and the center x of each season-length group."""
    positions = np.zeros(len(table))
    centers = []
    pos = 0.0
    i = 0
    for _m in season_lengths:
        start = pos
        for _p in range(n_end):
            positions[i] = pos
            pos += 1.0
            i += 1
        centers.append((start + pos - 1.0) / 2.0)
        pos += _GROUP_GAP
    return positions, centers


def _end_labels(table, n_end=14):
    """Ending-month tick labels for one group (same for every group): month
    abbreviation, prior-year months flagged with '*'."""
    g0 = table.iloc[:n_end]
    out = []
    for _, row in g0.iterrows():
        star = "*" if str(row["label"]).endswith("*") else ""
        out.append(_MONTH_ABBR[int(row["end_month"]) - 1] + star)
    return out


def _plot_seascorr(res, show=True, color=True, primary_name="primary",
                   secondary_name="secondary", primary_color=None,
                   secondary_color=None):
    """Draw Figure 3: the seasonal correlation / partial-correlation bar chart.

    Bar color follows the variable: temperature red, moisture blue, auto-detected
    from the names and overridable with ``primary_color`` / ``secondary_color``
    ('red' or 'blue'). Returns ``(fig, (ax_primary, ax_secondary))``.
    """
    table = res.table
    season_lengths = list(res.season_lengths)
    n_end = 14
    prim = _resolve_cmap(primary_color, primary_name, "blue", color)
    sec = _resolve_cmap(secondary_color, secondary_name, "red", color)

    positions, centers = _layout(table, season_lengths, n_end)
    width = 0.85

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)

    ax1.bar(positions, table["r_primary"].to_numpy(),
            color=_bar_faces(table["sig05_primary"], table["sig01_primary"], prim),
            edgecolor=_EDGE, linewidth=0.5, width=width)
    ax2.bar(positions, table["r_partial"].to_numpy(),
            color=_bar_faces(table["sig05_partial"], table["sig01_partial"], sec),
            edgecolor=_EDGE, linewidth=0.5, width=width)

    # group separators + group labels
    for ax in (ax1, ax2):
        ax.axhline(0.0, color="black", linewidth=0.8)
        for gi in range(1, len(season_lengths)):
            sep = centers[gi] - (n_end - 1) / 2.0 - _GROUP_GAP / 2.0
            ax.axvline(sep, color="0.8", linewidth=0.8, linestyle="--")
        style_axes(ax, xgrid=False, ygrid=True)

    ax1.set_ylim(*_nice_ylim(table["r_primary"].to_numpy()))
    ax2.set_ylim(*_nice_ylim(table["r_partial"].to_numpy()))

    # season-length group labels above the top panel
    for m, cx in zip(season_lengths, centers):
        lab = ("%d month" % m) if m == 1 else ("%d months" % m)
        ax1.text(cx, 1.02, lab, transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=9, color="0.25")

    # x ticks: ending months, repeated per group
    ticks = positions
    labels = _end_labels(table, n_end) * len(season_lengths)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)

    ax1.set_ylabel("r  (%s)" % primary_name)
    ax2.set_ylabel("partial r  (%s)" % secondary_name)
    ax2.set_xlabel("ending month  (* = previous year)")

    a, b = res.period
    fig.suptitle("Seasonal correlations, %d–%d  (nsim=%d)" % (a, b, res.nsim),
                 fontsize=11)

    # significance legends, colored to each panel, placed top-left and stacked
    # (alpha=0.01 above alpha=0.05) to match the MATLAB SEASCORR figure
    ax1.legend(handles=[Patch(facecolor=prim["s01"], edgecolor=_EDGE, label="α = 0.01"),
                        Patch(facecolor=prim["s05"], edgecolor=_EDGE, label="α = 0.05")],
               loc="upper left", fontsize=8, frameon=False, ncol=1,
               handlelength=1.2, handletextpad=0.5, borderaxespad=0.4)
    ax2.legend(handles=[Patch(facecolor=sec["s01"], edgecolor=_EDGE, label="α = 0.01"),
                        Patch(facecolor=sec["s05"], edgecolor=_EDGE, label="α = 0.05")],
               loc="upper left", fontsize=8, frameon=False, ncol=1,
               handlelength=1.2, handletextpad=0.5, borderaxespad=0.4)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    finalize_font(fig)

    if show:
        try:
            plt.show()
        except Exception:               # pragma: no cover - headless backends
            pass
    return fig, (ax1, ax2)
