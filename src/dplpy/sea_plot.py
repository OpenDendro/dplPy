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

# Title: sea_plot.py
# Project: OpenDendro dplPy
# Description: The superposed-epoch figure for dpl.sea (Rao et al. 2019, Fig. 7
#              style): the response composite with its double-bootstrap uncertainty
#              envelope, and the Monte-Carlo null thresholds, against window-relative
#              year.

import numpy as np

from ._plot_style import style_axes, finalize_font, ACCENT, NEUTRAL
from .detrend import _finish_figure

_SHADE = "#bcd0e6"          # light tint of ACCENT for the response envelope


def sea_plot(result, show=True):
    """Draw a SeaResult: the composite (or double-bootstrap envelope and median)
    and the null-distribution thresholds against window-relative year."""
    import matplotlib.pyplot as plt

    lags = np.asarray(result.lags, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))

    # response: envelope + median (double bootstrap) or the plain composite, the
    # mean line and its uncertainty band sharing the ACCENT colour family
    if result.envelope is not None:
        lo, med, hi = result.envelope
        ax.fill_between(lags, lo, hi, color=_SHADE, label="5-95% response envelope")
        ax.plot(lags, med, color=ACCENT, lw=2.4, label="median response")
    else:
        ax.plot(lags, result.composite, color=ACCENT, lw=2.4, label="composite")

    # null thresholds: 5/95% dashed, 1/99% dotted (the significance bounds)
    null = result.null
    for q, style in ((5, "--"), (95, "--"), (1, ":"), (99, ":")):
        if q in null.index:
            ax.plot(lags, null.loc[q].to_numpy(), color=NEUTRAL, lw=0.9, ls=style)
    # label the outer bounds at the right edge
    for q in (95, 99):
        if q in null.index:
            ax.annotate(f"{q}%", (lags[-1], null.loc[q].iloc[-1]), fontsize=7,
                        color=NEUTRAL, xytext=(3, 0), textcoords="offset points",
                        va="center")

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="0.55", lw=0.9)
    ax.set_xlim(lags[0], lags[-1])
    ax.set_xlabel("Years relative to event year")
    ax.set_ylabel("Normalized anomaly")
    kind = "double-bootstrap SEA" if result.envelope is not None else "SEA"
    ax.set_title("%s  -  %d events, %s null"
                 % (kind, len(result.events), result.settings["boot_method"]),
                 fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    style_axes(ax)
    finalize_font(fig)
    fig.tight_layout()
    result.fig, result.axes = fig, ax
    if show:
        _finish_figure(fig)
    return fig
