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

# Title: climate_response_plot.py
# Description: The figure climate_response draws -- one panel per climate
#   variable, a bar per month of the response or correlation coefficient across
#   the prior-to-current-year window (previous-year months lower-case, current
#   upper-case, in the manner of Fritts 1971 Fig. 4), significant months filled
#   and the rest left open, with a confidence-interval whisker when the
#   resampling bootstrap provides one. Styled per-Axes via _plot_style, like the
#   rest of dplPy's plots, and deliberately plainer than treeclim's output.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ._plot_style import style_axes, finalize_font
from .seascorr_plot import _classify, _FAMILIES, _BW, _EDGE


def _family(name, color):
    """Significance colors for a variable: temperature red, moisture blue,
    auto-detected from the name; grey when color is off or the name is neutral."""
    if not color:
        return _BW
    guess = _classify(name)
    return _FAMILIES[guess] if guess is not None else _BW


def _nice_ylim(vals):
    """Symmetric-ish y-limits with a floor of +-0.3, expanding to the data."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    dmin = min(float(v.min()) if v.size else 0.0, 0.0)
    dmax = max(float(v.max()) if v.size else 0.0, 0.0)
    lo = min(-0.3, dmin)
    hi = max(0.3, dmax)
    if lo < -0.3:
        lo = np.floor((lo - 0.03) / 0.1) * 0.1
    if hi > 0.3:
        hi = np.ceil((hi + 0.03) / 0.1) * 0.1
    return lo, hi


def _plot_climate_response(res, show=True, color=True):
    """Draw the climate_response figure: one panel per climate variable.

    Returns ``(fig, axes)`` where ``axes`` is a 1-D array of Axes, one per
    variable.
    """
    table = res.table
    variables = res.settings["variables"]
    has_ci = table["ci_lower"].notna().any()

    nvar = len(variables)
    fig, axes = plt.subplots(nvar, 1, figsize=(10, 2.4 * nvar + 0.8),
                             squeeze=False, sharex=False)
    axes = axes[:, 0]

    ymins, ymaxs = [], []
    for ax, var in zip(axes, variables):
        sub = table[table["variable"] == var]
        x = np.arange(len(sub))
        coef = sub["coef"].to_numpy(dtype=float)
        sig = sub["significant"].to_numpy(dtype=bool)
        cmap = _family(var, color)
        faces = [cmap["s01"] if s else cmap["ns"] for s in sig]

        ax.bar(x, coef, width=0.8, color=faces, edgecolor=_EDGE, linewidth=0.6,
               zorder=2)

        # confidence-interval whiskers (resampling bootstrap only)
        if has_ci:
            lo = sub["ci_lower"].to_numpy(dtype=float)
            hi = sub["ci_upper"].to_numpy(dtype=float)
            ok = np.isfinite(lo) & np.isfinite(hi)
            ax.vlines(x[ok], lo[ok], hi[ok], color="0.25", linewidth=0.9,
                      zorder=3)
            ymins.append(np.nanmin(lo))
            ymaxs.append(np.nanmax(hi))
        ymins.append(float(np.nanmin(coef)) if coef.size else 0.0)
        ymaxs.append(float(np.nanmax(coef)) if coef.size else 0.0)

        # previous|current year divider
        year = sub["year"].to_numpy()
        switch = np.where(year[:-1] != year[1:])[0]
        for s in switch:
            ax.axvline(s + 0.5, color="0.8", linewidth=0.9, linestyle="--",
                       zorder=1)

        ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["month"].to_list(), rotation=90, fontsize=7)
        ax.set_xlim(-0.7, len(sub) - 0.3)
        ax.set_ylabel(("response" if res.method == "response" else "r")
                      + "\n%s" % var)
        style_axes(ax, xgrid=False, ygrid=True)

    ylim = _nice_ylim(np.array(ymins + ymaxs))
    for ax in axes:
        ax.set_ylim(*ylim)

    axes[-1].set_xlabel("month  (lower-case = previous year, "
                        "UPPER-CASE = current year)")

    a, b = res.period
    lvl = 100 * (1 - res.settings["ci"])
    boot_txt = {"exact": "exact bootstrap", "std": "resampling bootstrap",
                "stationary": "stationary bootstrap",
                "none": "no bootstrap"}[res.boot]
    fig.suptitle("%s function  (%s), %d–%d"
                 % (res.method.capitalize(), boot_txt, a, b), fontsize=11)

    # one legend on the top panel: filled = significant at the level
    if res.boot != "none":
        cmap0 = _family(variables[0], color)
        axes[0].legend(
            handles=[Patch(facecolor=cmap0["s01"], edgecolor=_EDGE,
                           label="significant (%.0f%%)" % lvl),
                     Patch(facecolor="white", edgecolor=_EDGE,
                           label="not significant")],
            loc="upper left", fontsize=8, frameon=False,
            handlelength=1.2, handletextpad=0.5, borderaxespad=0.4)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    finalize_font(fig)

    if show:
        try:
            plt.show()
        except Exception:                   # pragma: no cover - headless backends
            pass
    return fig, axes
