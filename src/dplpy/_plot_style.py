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

# Title: _plot_style.py
# Description: A tiny, shared styling layer so every dplPy plot reads as one
#   family -- a clean, light, publication-friendly look. Deliberately does NOT
#   call matplotlib.pyplot.style.use(): that mutates matplotlib's GLOBAL state
#   for the rest of the session (the old plots did this with
#   'seaborn-v0_8-darkgrid', which then bled into every later figure, dplPy's
#   or the user's). Instead we style each Axes we own, one at a time, and leave
#   the user's rcParams untouched.

# Muted, colourblind-friendly accents used across dplPy plots.
ACCENT = "#3b6ea5"        # blue  -- primary series / detrended index
ACCENT_WARM = "#c1442e"   # brick -- fitted growth curves
NEUTRAL = "0.35"          # grey  -- raw ring width under a fitted curve
GRID = "0.9"              # hairline gridlines
SPAG_CMAP = "viridis"     # sequential map for the spaghetti plot (by first year)


def style_axes(ax, xgrid=True, ygrid=False, hide_spines=("top", "right"),
               hide_yticks=False):
    """Apply the shared dplPy look to a single Axes (no global side effects).

    Parameters
    ----------
    ax : matplotlib Axes
        the axes to style.
    xgrid, ygrid : bool
        draw a hairline grid along the x- and/or y-axis.
    hide_spines : iterable of str
        which spines to remove (default: top and right).
    hide_yticks : bool
        drop the y-axis tick marks (used by the seg/spag plots, where the
        y-axis carries series names, not a measured quantity).
    """
    ax.set_axisbelow(True)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    for spine in hide_spines:
        ax.spines[spine].set_visible(False)
    if hide_yticks:
        ax.tick_params(axis="y", length=0)
    return ax


def clamp(value, lo, hi):
    """Constrain a value to [lo, hi] -- used to keep auto-sized figures sane."""
    return max(lo, min(hi, value))
