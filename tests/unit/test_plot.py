import matplotlib
matplotlib.use("Agg")                                     # headless: no display
import matplotlib.pyplot as plt

import pytest
import dplpy as dpl


def _ca533():
    return dpl.readers("./tests/data/csv/ca533.csv")


def test_plot_types_create_figures():
    data = _ca533()
    for t in ("line", "spag", "seg"):
        plt.close("all")
        dpl.plot(data, type=t)
        assert len(plt.get_fignums()) >= 1, t             # a figure was drawn
    plt.close("all")


def test_plot_from_filepath_spag_and_seg():
    # ST5 regression: dpl.plot(<path>, type="spag"/"seg") used to pass the raw
    # filepath string to spag_plot/seg_plot and crash; it must now coerce first.
    for t in ("spag", "seg"):
        plt.close("all")
        dpl.plot("./tests/data/csv/ca533.csv", type=t)
        assert len(plt.get_fignums()) >= 1, t
    plt.close("all")


def test_plot_bad_type_raises():
    with pytest.raises(ValueError):
        dpl.plot(_ca533(), type="bogus")


def test_plot_uses_dplpy_font_without_global_change():
    # every plot's text (title + tick labels) is stamped with dplPy's resolved
    # font, applied per-figure so global rcParams stay untouched.
    import matplotlib.font_manager as fm
    from dplpy._plot_style import FONT_STACK
    expected = fm.FontProperties(
        fname=fm.findfont(fm.FontProperties(family=FONT_STACK))).get_name()
    before = list(plt.rcParams["font.family"])
    plt.close("all")
    fig, ax = dpl.plot(_ca533(), type="seg", show=False)
    assert ax.title.get_fontfamily() == [expected]
    assert ax.get_xticklabels()[0].get_fontfamily() == [expected]
    assert plt.rcParams["font.family"] == before      # no global mutation
    plt.close("all")


def test_plot_returns_fig_and_ax():
    # redesign: plot() now returns (fig, ax) so callers can save/restyle
    plt.close("all")
    for t in ("line", "spag", "seg"):
        out = dpl.plot(_ca533(), type=t, show=False)
        assert isinstance(out, tuple) and len(out) == 2
        fig, ax = out
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_draws_into_supplied_ax():
    # passing ax= must draw into that axes, not create a new figure
    plt.close("all")
    fig, ax = plt.subplots()
    before = len(plt.get_fignums())
    _, used = dpl.plot(_ca533(), type="seg", ax=ax, show=False)
    assert used is ax
    assert len(plt.get_fignums()) == before      # no extra figure created
    plt.close("all")


def test_spag_color_accepts_colormap_and_single_color():
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
        dpl.plot(_ca533(), type="spag", color=c, show=False)
    plt.close("all")


def test_spag_bad_color_raises():
    with pytest.raises(ValueError):
        dpl.plot(_ca533(), type="spag", color="definitely_not_a_color", show=False)


def test_plot_does_not_pollute_global_style():
    # regression: the old code called plt.style.use('seaborn-v0_8-darkgrid'),
    # which permanently mutated global rcParams (background, grid, fonts...) for
    # every later figure. The redesign styles each Axes it owns instead.
    plt.close("all")
    watched = ("axes.facecolor", "axes.edgecolor", "figure.facecolor",
               "axes.grid", "font.family")
    before = {k: plt.rcParams[k] for k in watched}
    for t in ("line", "spag", "seg"):
        dpl.plot(_ca533(), type=t, show=False)
    after = {k: plt.rcParams[k] for k in watched}
    assert before == after, "dpl.plot mutated global matplotlib rcParams"
    plt.close("all")
