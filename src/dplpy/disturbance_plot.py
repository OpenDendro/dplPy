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

# Title: disturbance_plot.py
# Project: OpenDendro dplPy
# Description: Figures for the canopy-disturbance method (see disturbance.py):
#              a summary chronology figure (release histogram with sample depth,
#              mean widths with the disturbance removed and the age trend, and the
#              mean standardized index) and a per-series three-panel figure
#              (radial-growth-averaging detection, disturbance removal, and the
#              index). disturbance_detrend(plot=True) draws the summary for a
#              DataFrame or the per-series figure for a single Series -- never one
#              figure per series -- and disturbance_plot() draws named series on
#              demand.

import numpy as np
import pandas as pd

from ._plot_style import style_axes, finalize_font, ACCENT, ACCENT_WARM, NEUTRAL
from .detrend import _finish_figure

_SHADE = "#bcd0e6"          # light tint of ACCENT for the "disturbance removed" band


def _series_fig(name, yrs, raw, DSt, At, St, pgc, threshold, relyrs, show):
    """Per-series three-panel figure: detection, disturbance removal, index."""
    import matplotlib.pyplot as plt

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True)

    def guides(ax):
        for ry in relyrs:
            ax.axvline(ry, color=ACCENT_WARM, lw=0.7, alpha=0.4)

    # (1) radial-growth averaging: ring width + percent growth change
    a1.plot(yrs, raw, color="black", lw=0.9)
    a1.set_ylabel("Ring width (mm)")
    a1b = a1.twinx()
    a1b.plot(yrs, pgc, color=NEUTRAL, lw=0.8, ls="--")
    a1b.axhline(threshold, color=NEUTRAL, lw=0.7, ls=":")
    a1b.set_ylabel("% growth change")
    guides(a1)
    for ry in relyrs:
        yv = pgc[np.where(yrs == ry)[0][0]] if ry in yrs else np.nanmax(pgc)
        a1b.plot(ry, yv, marker="v", color=ACCENT_WARM, ms=7, clip_on=False)
    a1.set_title("%s  -  radial-growth averaging and disturbance detrending" % name,
                 fontsize=11, loc="left")
    style_axes(a1)

    # (2) disturbance removal: shaded removed growth (top edge is raw) + age trend
    a2.fill_between(yrs, DSt, raw, where=(raw >= DSt), color=_SHADE,
                    label="disturbance removed")
    a2.plot(yrs, DSt, color="black", lw=1.0, label="disturbance-standardized")
    a2.plot(yrs, At, color=ACCENT, lw=1.3, label="age trend")
    a2.set_ylabel("Ring width (mm)")
    a2.legend(loc="upper right", fontsize=7, frameon=False)
    guides(a2)
    style_axes(a2)

    # (3) standardized index
    a3.axhline(1.0, color=NEUTRAL, lw=0.7, ls=":")
    a3.plot(yrs, St, color="black", lw=0.9)
    a3.set_ylabel("Standardized index")
    a3.set_xlabel("Year")
    guides(a3)
    style_axes(a3)

    finalize_font(fig)
    fig.tight_layout()
    if show:
        _finish_figure(fig)
    return fig


def _summary_fig(years, mean_raw, mean_DSt, mean_At, mean_St, depth, rel_years,
                 title, show):
    """Summary chronology figure: release histogram + depth, mean widths with the
    disturbance removed and the age trend, and the mean standardized index."""
    import matplotlib.pyplot as plt

    years = np.asarray(years, dtype=float)
    fig, (b1, b2, b3) = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True)

    # (A) release-event histogram (decadal) + sample depth
    lo = int(np.floor(years.min() / 10) * 10)
    hi = int(np.ceil(years.max() / 10) * 10)
    bins = np.arange(lo, hi + 10, 10)
    b1.hist(np.asarray(rel_years, dtype=float), bins=bins, color=NEUTRAL,
            edgecolor="white", lw=0.3)
    b1.set_ylabel("Release events")
    b1b = b1.twinx()
    b1b.plot(years, depth, color=ACCENT, lw=1.0)
    b1b.set_ylabel("Sample depth", color=ACCENT)
    b1b.tick_params(axis="y", colors=ACCENT)
    b1.set_title(title, fontsize=11, loc="left")
    style_axes(b1)

    # (B) mean widths: shaded mean disturbance removed + mean age trend
    b2.fill_between(years, mean_DSt, mean_raw, where=(mean_raw >= mean_DSt),
                    color=_SHADE, label="mean disturbance removed")
    b2.plot(years, mean_DSt, color="black", lw=1.0,
            label="mean disturbance-standardized")
    b2.plot(years, mean_At, color=ACCENT, lw=1.3, label="mean age trend")
    b2.set_ylabel("Ring width (mm)")
    b2.legend(loc="upper right", fontsize=7, frameon=False)
    style_axes(b2)

    # (C) mean standardized index
    b3.axhline(1.0, color=NEUTRAL, lw=0.7, ls=":")
    b3.plot(years, mean_St, color="black", lw=0.9)
    b3.set_ylabel("Mean standardized index")
    b3.set_xlabel("Year")
    style_axes(b3)

    finalize_font(fig)
    fig.tight_layout()
    if show:
        _finish_figure(fig)
    return fig


def _summary_from_frames(data, dst, at, index, averages, rel_years, show):
    """Build the summary figure from already-computed pipeline frames."""
    years = data.index.to_numpy()
    title = "Canopy disturbance chronology (%d series)" % data.shape[1]
    return _summary_fig(years,
                        data.mean(axis=1, skipna=True).to_numpy(),
                        dst.mean(axis=1, skipna=True).to_numpy(),
                        averages["age_trend"].to_numpy(),
                        averages["index"].to_numpy(),
                        averages["sample_depth"].to_numpy(),
                        rel_years, title, show)


def disturbance_plot(data, series=None, threshold=50, window_length=15,
                     significance=False, release_flex=10, age_flex=30,
                     transform="druckenbrod", recode_zeros=False, min_mean=0.01,
                     show=True):
    """Plot the canopy-disturbance result for a chronology or named series.

    With ``series=None`` and a DataFrame, draws the summary chronology figure; with
    ``series=None`` and a single Series, draws that series' three-panel figure.
    Pass ``series="DEM01A"`` or ``series=["DEM01A", "DEM07B"]`` to draw specific
    series' three-panel figures (detection, disturbance removal, index) -- the way
    to inspect individual cores without a figure per series. The detection and
    detrending parameters match ``disturbance_detrend``.

    Returns the matplotlib Figure (or a list of them when several series are drawn).
    """
    from .disturbance import disturbance_detrend, releases, _rga_one

    if isinstance(data, str):
        from .readers import readers
        data = readers(data)
    kw = dict(threshold=threshold, window_length=window_length,
              significance=significance, release_flex=release_flex,
              age_flex=age_flex, transform=transform, recode_zeros=recode_zeros,
              min_mean=min_mean)

    # Resolve which series (if any) to draw in per-series detail.
    if series is None and isinstance(data, pd.DataFrame):
        info = disturbance_detrend(data, return_info=True, **kw)
        rel_years = info["releases"]["year"].to_numpy()
        return _summary_from_frames(data, info["disturbance_standardized"],
                                    info["age_trend"], info["index"],
                                    info["averages"], rel_years, show)

    if series is None:                                   # a single Series
        names = [data.name]
        frame = data.to_frame()
    else:
        names = [series] if isinstance(series, str) else list(series)
        frame = (data.to_frame() if isinstance(data, pd.Series)
                 else data[names])

    info = disturbance_detrend(frame, return_info=True, **kw)
    _, rel_info = releases(frame, return_info=True, threshold=threshold,
                           window_length=window_length, recode_zeros=recode_zeros,
                           min_mean=min_mean)
    reltab = info["releases"]
    figs = []
    for name in names:
        raw = frame[name].dropna()
        yrs = raw.index.to_numpy()
        relyrs = sorted(reltab[reltab["series"] == name]["year"].tolist())
        figs.append(_series_fig(
            name, yrs, raw.to_numpy(),
            info["disturbance_standardized"][name].reindex(raw.index).to_numpy(),
            info["age_trend"][name].reindex(raw.index).to_numpy(),
            info["index"][name].reindex(raw.index).to_numpy(),
            rel_info[name]["pct_growth_change"].reindex(raw.index).to_numpy(),
            threshold, relyrs, show))
    return figs[0] if len(figs) == 1 else figs
