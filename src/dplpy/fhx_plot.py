"""Fire-history plots: the fire-chart (demograph) and interval distribution.

A matplotlib port of burnr's ``fhx_demograph`` and ``_plot_intervals_dist``
(Malevich, Guiterman & Margolis; Malevich et al. 2018, Dendrochronologia,
doi:10.1016/j.dendro.2018.02.005), following dplPy's plotting convention: each
function accepts an existing ``ax`` and ``show`` flag and returns ``(fig, ax)``.

:func:`fhx_demograph` draws one horizontal line per series over its span (a
dashed non-recording baseline with solid recording periods on top), marks fire
scars, injuries, and pith/bark years, and can add a composite rug along the
bottom. Faceting (burnr's ``facet_group``/``facet_id``) is not yet implemented;
colour grouping is.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fhx import _require_fhx, fhx_series_names, fhx_get_events
from .fhx_stats import _recording_segments, fhx_composite

# event markers, mirroring burnr's shapes (scar bar, open down-triangle, dot)
_MARKERS = {"Scar": "|", "Injury": "v", "Pith/Bark": "o"}
_DEFAULT_EVENT_SIZE = {"Scar": 4.0, "Injury": 2.0, "Pith/Bark": 1.5}
_PITHBARK = ("pith_year", "bark_year")


def _finalize(fig, show):
    if show:
        plt.show()
        if not matplotlib.is_interactive():
            plt.close(fig)
    return fig


def _series_groups(color_group, names):
    """Normalize ``color_group`` (a dict or Series mapping series name -> group,
    or None) to a ``{series: group}`` dict and an ordered list of group labels."""
    if color_group is None:
        return None, []
    if isinstance(color_group, pd.Series):
        color_group = color_group.to_dict()
    mapping = {str(k): v for k, v in dict(color_group).items()}
    labels = []
    for n in names:
        g = mapping.get(n)
        if g is not None and g not in labels:
            labels.append(g)
    return mapping, labels


def fhx_demograph(x, color_group=None, composite_rug=False, filter_prop=0.25,
                   filter_min_rec=2, filter_min_events=1, injury_event=False,
                   plot_legend=False, event_size=None, ylabels=True,
                   yearlims=None, ax=None, show=True):
    """Fire chart (demograph) of an ``fhx`` object.

    Parameters
    ----------
    x : fhx
        the fire-history object.
    color_group : dict, pandas.Series, or None
        optional mapping of series name -> group label (e.g. species); series
        lines are coloured by group. ``None`` draws every series in one colour.
    composite_rug : bool
        if True, add a rug of composite fire years along the bottom (using the
        ``filter_*`` and ``injury_event`` arguments, as in :func:`dplpy.composite`).
    filter_prop, filter_min_rec, filter_min_events, injury_event
        composite parameters, used only when ``composite_rug`` is True.
    plot_legend : bool
        if True, draw a legend (event markers, and group colours if given).
    event_size : dict or None
        marker sizes for ``"Scar"``, ``"Injury"``, ``"Pith/Bark"``; defaults to
        burnr's ``{"Scar": 4, "Injury": 2, "Pith/Bark": 1.5}``.
    ylabels : bool
        if True, label the y-axis with series names; if False, hide them.
    yearlims : (int, int) or None
        optional x-axis (year) limits.
    ax : matplotlib Axes, optional
        axes to draw on; a new figure/axes is created if omitted.
    show : bool
        call ``plt.show()`` after drawing (closing the figure under a
        non-interactive backend). The returned figure can still be saved.

    Returns
    -------
    (fig, ax) : tuple of matplotlib Figure and Axes
    """
    _require_fhx(x)
    names = fhx_series_names(x)
    n = len(names)
    ypos = {s: i for i, s in enumerate(names)}       # first series at the bottom
    sizes = dict(_DEFAULT_EVENT_SIZE)
    if event_size:
        sizes.update(event_size)

    mapping, labels = _series_groups(color_group, names)
    if labels:
        cmap = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")
        group_color = {g: cmap(i % cmap.N) for i, g in enumerate(labels)}
    else:
        group_color = {}

    def _color(series):
        if mapping is None:
            return "black"
        return group_color.get(mapping.get(series), "black")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(2.5, 0.25 * n + 1)))
    else:
        fig = ax.figure

    # non-recording baseline (full span, dashed) per series
    for s in names:
        df = x[x["series"].astype(str) == s]
        y = ypos[s]
        ax.plot([int(df["year"].min()), int(df["year"].max())], [y, y],
                linestyle="--", linewidth=0.8, color=_color(s), zorder=1)

    # recording periods (solid) on top
    for s, first, last in _recording_segments(x, injury_event=injury_event):
        y = ypos[s]
        ax.plot([first, last], [y, y], linestyle="-", linewidth=1.6,
                color=_color(s), zorder=2)

    # event markers (scars, injuries, pith/bark); recorder years are not marked
    rt = x["rec_type"].astype(str)
    cats = {
        "Scar": rt.str.contains("_fs"),
        "Injury": rt.str.contains("_fi"),
        "Pith/Bark": rt.isin(_PITHBARK),
    }
    for label, msk in cats.items():
        sub = x[msk.to_numpy()]
        if len(sub) == 0:
            continue
        xs = sub["year"].to_numpy()
        ys = [ypos[str(s)] for s in sub["series"]]
        ms = sizes[label] * 2.2 + 3
        if label == "Injury":                        # open down-triangle
            ax.scatter(xs, ys, marker=_MARKERS[label], s=(ms) ** 2,
                       facecolors="none", edgecolors="black", linewidths=0.9,
                       zorder=3)
        else:
            ax.scatter(xs, ys, marker=_MARKERS[label], s=(ms) ** 2,
                       color="black", linewidths=1.0, zorder=3)

    ymin = -0.5
    if composite_rug:
        comp = fhx_composite(x, filter_prop=filter_prop, filter_min_rec=filter_min_rec,
                         filter_min_events=filter_min_events,
                         injury_event=injury_event)
        cnames = fhx_series_names(comp)
        comp_years = (fhx_get_events(comp)[cnames[0]] if cnames else np.array([]))
        divider = -1.0
        ymin = -2.0
        ax.axhline(divider, color="grey", linewidth=0.8, zorder=1)
        for yr in np.asarray(comp_years, dtype=int):
            ax.plot([yr, yr], [ymin + 0.25, divider], color="black",
                    linewidth=0.8, zorder=2)

    ax.set_ylim(ymin, n - 0.5)
    if ylabels:
        ax.set_yticks(range(n))
        ax.set_yticklabels(names)
    else:
        ax.set_yticks([])
    if yearlims is not None:
        ax.set_xlim(yearlims)
    ax.set_xlabel("Year")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if plot_legend:
        handles = []
        for label in ("Scar", "Injury", "Pith/Bark"):
            if label == "Injury":
                handles.append(plt.Line2D([], [], linestyle="none",
                               marker=_MARKERS[label], markerfacecolor="none",
                               markeredgecolor="black", label=label))
            else:
                handles.append(plt.Line2D([], [], linestyle="none",
                               marker=_MARKERS[label], color="black", label=label))
        for g in labels:
            handles.append(plt.Line2D([], [], color=group_color[g], label=str(g)))
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=min(5, len(handles)),
                  frameon=False)

    return _finalize(fig, show), ax


def _plot_intervals_dist(x, binwidth=None, ax=None, show=True):
    """Histogram (with a rug) of the observed fire intervals.

    Parameters
    ----------
    x : FhxIntervals or array-like
        a result from :func:`dplpy.intervals`, or the intervals themselves.
    binwidth : int or None
        histogram bin width in years; if ``None``, matplotlib's default binning
        is used.
    ax : matplotlib Axes, optional
    show : bool

    Returns
    -------
    (fig, ax) : tuple of matplotlib Figure and Axes
    """
    iv = np.asarray(getattr(x, "intervals", x), dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if binwidth is not None and len(iv) > 0:
        lo = np.floor(iv.min() / binwidth) * binwidth
        hi = np.ceil(iv.max() / binwidth) * binwidth + binwidth
        bins = np.arange(lo, hi, binwidth)
    else:
        bins = "auto"
    ax.hist(iv, bins=bins, color="0.6", edgecolor="black")
    # rug of individual intervals
    ax.plot(iv, np.full(len(iv), 0.0), marker="|", linestyle="none",
            color="black", markersize=10)
    ax.set_xlabel("Interval (years)")
    ax.set_ylabel("Count")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _finalize(fig, show), ax
