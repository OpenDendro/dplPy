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

# Date: 9/8/2022 (redesigned 2026-09)
# Author: Ifeoluwa Ale
# Title: plot.py
# Description: Generates plots of tree-ring width data from a dataframe. Supports
#   line ('line'), spaghetti ('spag') and segment ('seg') plots.
#
#   The plots use a clean, light, publication-friendly look (see _plot_style),
#   applied per-Axes so nothing leaks into the user's global matplotlib state.
#   Each function accepts an existing ``ax`` and returns the Axes it drew on, so
#   figures can be saved, composed into subplots, or restyled after the fact.
#
# example usages:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.plot(data)                                  # segment plot (default)
# >>> dpl.plot(data, type="spag")                     # spaghetti plot
# >>> fig, ax = dpl.plot(data, type="seg", show=False)  # keep the figure to save
# >>> fig.savefig("segments.png", dpi=300)
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ._validate import _coerce_to_frame
from .stats import stats
from ._plot_style import (style_axes, clamp, ACCENT)


def plot(inp: pd.DataFrame | str, type="seg", ax=None, show=True, **kwargs):
    """Plot a ring-width dataset as a line, spaghetti, or segment plot.

    Parameters
    ----------
    inp : str | DataFrame
        a file path to a .csv or .rwl file, or a dataframe from dpl.readers().
    type : str, default 'seg'
        'seg' (coverage segments, one bar per series), 'spag' (spaghetti: each
        series offset vertically, black by default -- pass color= a colormap
        name to shade by first year), or 'line' (all series overplotted
        against year).
    ax : matplotlib Axes, optional
        draw into this Axes instead of creating a new figure. When omitted, a
        figure is created and auto-sized to the data.
    show : bool, default True
        call ``plt.show()`` after drawing. Set False to keep the figure for
        saving or further editing (the returned figure is still fully drawn).
    **kwargs
        passed through to the underlying plotter (e.g. ``zfac`` and ``color``
        for the spaghetti plot -- ``color`` takes a colormap name like
        'viridis'/'turbo' or a single colour like 'black').

    Returns
    -------
    (fig, ax) : tuple of matplotlib Figure and Axes
        the figure and axes drawn on, so callers can save or restyle them.
        (Earlier versions returned None; the figure is now returned for reuse.)

    Examples
    --------
    >>> dpl.plot(data)
    >>> dpl.plot(data, type="spag")
    >>> fig, ax = dpl.plot(data, type="seg", show=False); fig.savefig("seg.png")

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#plot
    """
    series_data = _coerce_to_frame(inp)

    if type == "line":
        ax = line_plot(series_data, ax=ax, **kwargs)
    elif type == "spag":
        ax = spag_plot(series_data, ax=ax, **kwargs)
    elif type == "seg":
        ax = seg_plot(series_data, ax=ax, **kwargs)
    else:
        raise ValueError("Unsupported plot type: %r (use 'line', 'spag' or "
                         "'seg')." % (type,))

    fig = ax.figure
    if show:
        plt.show()
    return fig, ax


def _series_by_start(data):
    """Series names ordered by their first year of data (oldest first)."""
    data_stats = stats(data)
    return data_stats.sort_values(by="first")["series"].tolist()


def _new_axes(ax, width, height):
    """Return (ax, created?) -- make a right-sized figure if no ax was given."""
    if ax is not None:
        return ax, False
    _, ax = plt.subplots(figsize=(width, height))
    return ax, True


def _resolve_series_colors(color, n):
    """Turn a ``color`` argument into a list of n per-series colours.

    A registered colormap name (e.g. 'viridis', 'turbo') is sampled across the
    n series (shading them by first year); any single matplotlib colour spec
    (e.g. 'black', 'k', '#3b6ea5', an RGB tuple) is repeated for every series.
    Anything that is neither raises a clear ValueError.
    """
    n = max(n, 1)
    # a colormap name -> gradient across series
    if isinstance(color, str) and color in plt.colormaps():
        return list(plt.get_cmap(color)(np.linspace(0, 1, n)))
    # otherwise it must be a single valid colour -> monochrome stack
    try:
        single = mcolors.to_rgba(color)
    except (ValueError, TypeError):
        raise ValueError(
            "color=%r is neither a matplotlib colormap name (e.g. 'viridis', "
            "'turbo') nor a single colour (e.g. 'black', 'k', '#3b6ea5')."
            % (color,))
    return [single] * n


def line_plot(data, ax=None):
    """Overplot every series against year (thin lines, shared axes)."""
    years = data.index.to_numpy()
    span = float(years[-1] - years[0]) if years.size > 1 else 1.0
    width = clamp(7 + span / 300.0, 7, 16)
    ax, _ = _new_axes(ax, width, 4.0)

    ax.plot(years, data.to_numpy(), linewidth=0.7, alpha=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Ring width")
    ax.set_title("%d series" % data.shape[1], fontsize=10)
    style_axes(ax, xgrid=True, ygrid=True)
    ax.figure.tight_layout()
    return ax


def spag_plot(data, ax=None, zfac=1.0, color="black"):
    """Spaghetti plot: each series centred on its own mean and offset onto its
    own horizontal lane, ordered by first year.

    Relative amplitudes are preserved across series (a single shared scale), so
    a genuinely more variable series still looks more variable. ``zfac`` tunes
    that common amplitude (larger = taller wiggles); it mirrors dplR's ``zfac``.

    Parameters
    ----------
    color : str, default 'black'
        how to colour the series. Either any single matplotlib colour (e.g.
        'black', 'k', '#3b6ea5'), drawing every series in that one colour (the
        default); or the name of a matplotlib colormap (e.g. 'viridis',
        'turbo', 'plasma'), in which case series are shaded by first year
        (oldest -> newest), turning the stack into a recruitment timeline.
    """
    order = _series_by_start(data)
    years = data.index.to_numpy()
    n = len(order)

    # A single shared vertical scale keeps relative amplitudes comparable while
    # spacing lanes one unit apart. Base it on the typical (5th-95th pct) span of
    # a series so a normal series fills ~0.7 of its lane and neighbours don't
    # collide; guard the degenerate all-flat case.
    spans = [np.nanpercentile(data[s], 95) - np.nanpercentile(data[s], 5)
             for s in order]
    typical = np.nanmedian(spans)
    scale = (typical / (0.7 * zfac)) if (typical and np.isfinite(typical)) else 1.0

    span = float(years[-1] - years[0]) if years.size > 1 else 1.0
    width = clamp(7 + span / 200.0, 8, 18)
    height = clamp(0.32 * n, 3, 24)
    ax, _ = _new_axes(ax, width, height)

    colors = _resolve_series_colors(color, n)
    for i, s in enumerate(order):
        y = (data[s] - data[s].mean()) / scale + i
        ax.plot(years, y.to_numpy(), linewidth=0.8, alpha=0.9, color=colors[i])

    ax.set_yticks(range(n))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_ylim(-1, n)
    ax.set_xlabel("Year")
    ax.set_title("%d series (relative amplitude preserved)" % n, fontsize=10)
    style_axes(ax, xgrid=True, hide_spines=("top", "right", "left"),
               hide_yticks=True)
    ax.figure.tight_layout()
    return ax


def seg_plot(data, ax=None, color=ACCENT):
    """Segment (coverage) plot: one horizontal bar per series spanning its first
    to last year, stacked and ordered by first year. Interior gaps break the bar,
    and end-caps mark the first/last measured ring."""
    order = _series_by_start(data)
    years = data.index.to_numpy()
    n = len(order)

    span = float(years[-1] - years[0]) if years.size > 1 else 1.0
    width = clamp(6 + span / 300.0, 7, 16)
    height = clamp(0.28 * n, 2.5, 22)
    ax, _ = _new_axes(ax, width, height)

    for i, s in enumerate(order):
        present = years[data[s].notna().to_numpy()]
        if present.size == 0:
            continue
        lo, hi = present.min(), present.max()
        # draw the covered years so interior NaN gaps show as breaks in the bar
        yy = np.where(data[s].notna().to_numpy(), float(i), np.nan)
        ax.plot(years, yy, linewidth=2.4, alpha=0.9, color=color,
                solid_capstyle="butt")
        ax.plot([lo, hi], [i, i], marker="|", linestyle="none", color=color,
                markersize=6, markeredgewidth=1.4)

    ax.set_yticks(range(n))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_ylim(-1, n)
    ax.set_xlabel("Year")
    if n:
        ax.set_title("%d series · %d–%d"
                     % (n, int(years.min()), int(years.max())), fontsize=10)
    style_axes(ax, xgrid=True, hide_spines=("top", "right", "left"),
               hide_yticks=True)
    ax.figure.tight_layout()
    return ax
