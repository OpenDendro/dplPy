import matplotlib
matplotlib.use("Agg")                                     # headless: no display
import matplotlib.pyplot as plt

import pytest
import dplpy as dpl


def _drawn(fig_ax):
    """True when the returned figure actually has plotted content."""
    fig, ax = fig_ax
    return isinstance(fig, plt.Figure) and (len(ax.lines) > 0 or len(ax.collections) > 0)


def test_plot_types_draw_and_close_under_agg(ca533_rwl):
    # Under a non-interactive backend plt.show() is a no-op, so plot() closes
    # the figure after "showing" it rather than leaking one per call; the
    # returned (fig, ax) is still fully drawn.
    for t in ("line", "spag", "seg"):
        plt.close("all")
        out = dpl.plot(ca533_rwl, type=t)
        assert _drawn(out), t
        assert plt.get_fignums() == [], t                 # nothing left open
    plt.close("all")


def test_plot_show_false_keeps_figure_open(ca533_rwl):
    plt.close("all")
    fig, ax = dpl.plot(ca533_rwl, type="seg", show=False)
    assert plt.get_fignums() == [fig.number]              # kept for saving
    plt.close("all")


def test_plot_from_filepath_spag_and_seg(DATA):
    # ST5 regression: dpl.plot(<path>, type="spag"/"seg") used to pass the raw
    # filepath string to spag_plot/seg_plot and crash; it must now coerce first.
    for t in ("spag", "seg"):
        plt.close("all")
        out = dpl.plot(str(DATA / "csv" / "ca533.csv"), type=t, show=False)
        assert _drawn(out), t
    plt.close("all")


def test_plot_bad_type_raises(ca533_rwl):
    with pytest.raises(ValueError):
        dpl.plot(ca533_rwl, type="bogus")


def test_plot_uses_dplpy_font_without_global_change(ca533_rwl):
    # every plot's text (title + tick labels) is stamped with dplPy's resolved
    # font, applied per-figure so global rcParams stay untouched.
    import matplotlib.font_manager as fm
    from dplpy._plot_style import FONT_STACK
    expected = fm.FontProperties(
        fname=fm.findfont(fm.FontProperties(family=FONT_STACK))).get_name()
    before = list(plt.rcParams["font.family"])
    plt.close("all")
    fig, ax = dpl.plot(ca533_rwl, type="seg", show=False)
    assert ax.title.get_fontfamily() == [expected]
    assert ax.get_xticklabels()[0].get_fontfamily() == [expected]
    assert plt.rcParams["font.family"] == before      # no global mutation
    plt.close("all")


def test_plot_returns_fig_and_ax(ca533_rwl):
    # redesign: plot() now returns (fig, ax) so callers can save/restyle
    plt.close("all")
    for t in ("line", "spag", "seg"):
        out = dpl.plot(ca533_rwl, type=t, show=False)
        assert isinstance(out, tuple) and len(out) == 2
        fig, ax = out
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_draws_into_supplied_ax(ca533_rwl):
    # passing ax= must draw into that axes, not create a new figure
    plt.close("all")
    fig, ax = plt.subplots()
    before = len(plt.get_fignums())
    _, used = dpl.plot(ca533_rwl, type="seg", ax=ax, show=False)
    assert used is ax
    assert len(plt.get_fignums()) == before      # no extra figure created
    plt.close("all")


def test_spag_color_accepts_colormap_and_single_color(ca533_rwl):
    # spaghetti coloring is selectable: a colormap name shades by first year,
    # a single color draws every series that one color
    from dplpy.plot import _resolve_series_colors
    grad = _resolve_series_colors("turbo", 5)
    assert len(grad) == 5 and len({tuple(c) for c in grad}) == 5   # 5 distinct
    mono = _resolve_series_colors("black", 5)
    assert len(mono) == 5 and len({tuple(c) for c in mono}) == 1   # all identical
    # both drive dpl.plot without error
    plt.close("all")
    for c in ("viridis", "turbo", "black", "#3b6ea5"):
        dpl.plot(ca533_rwl, type="spag", color=c, show=False)
    plt.close("all")


def test_spag_bad_color_raises(ca533_rwl):
    with pytest.raises(ValueError):
        dpl.plot(ca533_rwl, type="spag", color="definitely_not_a_color", show=False)


def test_plot_does_not_pollute_global_style(ca533_rwl):
    # regression: the old code called plt.style.use('seaborn-v0_8-darkgrid'),
    # which permanently mutated global rcParams (background, grid, fonts...) for
    # every later figure. The redesign styles each Axes it owns instead.
    plt.close("all")
    watched = ("axes.facecolor", "axes.edgecolor", "figure.facecolor",
               "axes.grid", "font.family")
    before = {k: plt.rcParams[k] for k in watched}
    for t in ("line", "spag", "seg"):
        dpl.plot(ca533_rwl, type=t, show=False)
    after = {k: plt.rcParams[k] for k in watched}
    assert before == after, "dpl.plot mutated global matplotlib rcParams"
    plt.close("all")


def test_plot_does_not_mutate_input(ca533_rwl):
    before = ca533_rwl.copy()
    plt.close("all")
    for t in ("line", "spag", "seg"):
        dpl.plot(ca533_rwl, type=t, show=False)
    plt.close("all")
    assert ca533_rwl.equals(before)
