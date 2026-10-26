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

# Title: moving_corr_plot.py
# Project: OpenDendro dplPy
# Description: The moving-correlation figure for dpl.moving_corr, in the
#              Gelman & Stern (2006) spirit: the correlation trace with its
#              per-window uncertainty band and a single multiplicity-aware
#              significance line, plus the whole-trace test verdicts in the
#              caption. Deliberately NOT a grid of per-window significance
#              flags: the difference between a window that clears the line and
#              one that does not is not itself something to read as a change in
#              the signal.

import numpy as np

from ._plot_style import style_axes, finalize_font, ACCENT, ACCENT_WARM, NEUTRAL
from .detrend import _finish_figure

_SHADE = "#bcd0e6"          # light tint of ACCENT for the per-window band


def moving_corr_plot(result, show=True, level=0.95, annotate_tests=True):
    """Draw a MovingCorrResult: the correlation trace, its per-window confidence
    band, the overall correlation, and (if a peak test was run) the flat
    significance threshold at ``level``.

    Parameters
    ----------
    result : MovingCorrResult
    show : bool
        Display the figure (otherwise just build and return it).
    level : float
        Confidence level for the peak-test threshold line, e.g. 0.95.
    annotate_tests : bool
        Write the global test verdicts (std / peak / range p-values) into the
        caption. These are statements about the whole trace, not per-window.
    """
    import matplotlib.pyplot as plt

    tr = result.trace
    t = tr.index.to_numpy(dtype=float)
    r = tr["r"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    # per-window uncertainty band (estimate +/- uncertainty), same colour family
    if "ci_lower" in tr.columns:
        ax.fill_between(t, tr["ci_lower"].to_numpy(), tr["ci_upper"].to_numpy(),
                        color=_SHADE, label="%d%% per-window interval"
                        % round(result.settings["ci_level"] * 100))

    # zero and overall-correlation references
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(result.overall_corr, color="0.55", lw=0.9, ls=":",
               label="overall r = %.2f" % result.overall_corr)

    # the trace
    ax.plot(t, r, color=ACCENT, lw=2.2, label="moving correlation", zorder=3)

    # flat, multiplicity-aware significance lines from the peak test: an upper
    # (peak) line and a lower (trough) line, plus the windows that cross either
    if "peak" in result.tests:
        pt = result.tests["peak"]
        up = pt.threshold(level)
        lines = [("%d%% peak/trough threshold" % round(level * 100), up)]
        lo = pt.threshold_lower(level) if pt.null_min is not None else None
        if lo is not None:
            lines.append((None, lo))
        for lab, y in lines:
            ax.axhline(y, color=NEUTRAL, lw=1.0, ls="--",
                       label=lab, zorder=2)
            ax.annotate("%d%%" % round(level * 100), (t[-1], y), fontsize=7,
                        color=NEUTRAL, xytext=(3, 0), textcoords="offset points",
                        va="center")
        exceed = r > up
        if lo is not None:
            exceed = exceed | (r < lo)
        if exceed.any():
            ax.scatter(t[exceed], r[exceed], color=ACCENT_WARM, s=18, zorder=4,
                       label="crosses threshold")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Year (%s of window)" % result.settings["position"])
    ax.set_ylabel("Correlation")

    s = result.settings
    ax.set_title("Moving correlation  -  chronology vs %s  (window %d, step %d)"
                 % (result.label, s["window"], s["offset"]),
                 fontsize=11, loc="left")

    # global verdicts as a caption line, not per-window stars (Gelman-Stern)
    if annotate_tests and result.tests:
        bits = []
        names = {"std": "variability", "peak": "peak", "range": "range"}
        for k in ("std", "peak", "range"):
            if k in result.tests:
                bits.append("%s p=%.3f" % (names[k], result.tests[k].p_value))
        ax.text(0.0, -0.22, "whole-trace tests (%s null):  " % s["noise"]
                + "    ".join(bits), transform=ax.transAxes, fontsize=8,
                color="0.3", va="top")

    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    style_axes(ax)
    finalize_font(fig)
    fig.tight_layout()
    result.fig, result.axes = fig, ax
    if show:
        _finish_figure(fig)
    return fig
