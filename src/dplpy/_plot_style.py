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

# Font stack for every dplPy plot: Helvetica first, then Arial, then their
# metric-compatible clones (so Linux / CI still render in a Helvetica-like face),
# ending in DejaVu Sans + the generic 'sans-serif'. Because DejaVu Sans is always
# installed with matplotlib, the fallback always resolves -- so there are no
# "findfont: Font family not found" warnings when Helvetica is absent.
FONT_STACK = ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans",
              "DejaVu Sans", "sans-serif"]


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


def finalize_font(fig, stack=FONT_STACK):
    """Set the dplPy font on every text element of a finished figure.

    This is applied per-figure, on the text objects themselves, rather than by
    changing matplotlib's global rcParams -- so it never alters the font of the
    user's own (non-dplPy) plots. matplotlib resolves an unset font lazily from
    rcParams at draw time, so we first draw once to materialise the automatic
    tick labels, then stamp the family explicitly; that also makes the choice
    survive a later ``fig.savefig()`` on a figure returned with ``show=False``.
    Call it once, last, just before showing/returning the figure.
    """
    import logging
    import matplotlib.font_manager as fm

    # Resolve the preference list to ONE concrete installed font, once, with the
    # font_manager logger hushed so the 'Font family not found' notices for the
    # (expected) missing entries on Helvetica-less machines don't spam the log.
    # Stamping the resolved name (rather than the list) also means later redraws
    # don't re-trigger that lookup.
    flog = logging.getLogger("matplotlib.font_manager")
    prev_level = flog.level
    try:
        flog.setLevel(logging.ERROR)
        resolved = fm.FontProperties(
            fname=fm.findfont(fm.FontProperties(family=stack))).get_name()
    except Exception:                # pragma: no cover - fall back to the list
        resolved = stack
    finally:
        flog.setLevel(prev_level)

    try:
        fig.canvas.draw()            # materialise auto tick labels
    except Exception:                # pragma: no cover - headless edge cases
        pass
    for ax in fig.get_axes():        # includes twin / secondary axes
        items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        items += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        legend = ax.get_legend()
        if legend is not None:
            items += list(legend.get_texts())
        items += list(ax.texts)
        for t in items:
            t.set_fontfamily(resolved)
    for t in fig.texts:              # figure-level suptitle / fig.text()
        t.set_fontfamily(resolved)
    return fig
