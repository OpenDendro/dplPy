"""Tests for the fire-history plots (dplpy.fhx_plot) and the recording-segment
helper behind them.

The recording-segment golden values match burnr's internal get_rec_tbl on lgr2
and pgm (confirmed identical as a set during development, including the trimmed
segment ends and the multi-segment split of LGR52). The plot tests are smoke
tests: they assert the figure is built with the expected pieces, since the
numeric content is validated through the segment helper and the stats layer.
"""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import dplpy as dpl
from dplpy import fhx_stats as fs

LGR2 = "tests/data/fhx/lgr2.fhx"
PGM = "tests/data/fhx/pgm.fhx"


@pytest.fixture
def lgr2():
    return dpl.read_fhx(LGR2)


@pytest.fixture
def pgm():
    return dpl.read_fhx(PGM)


# --- recording segments (burnr get_rec_tbl golden) -------------------------

def test_recording_segments_match_burnr(lgr2):
    segs = fs._recording_segments(lgr2)
    assert len(segs) == 27                        # burnr get_rec_tbl(lgr2)
    sset = set(segs)
    # trimmed ends (boundary years excluded) and the LGR52 split into two
    assert ("LGR54", 1424, 1482) in sset
    assert ("LGR47", 1581, 1769) in sset
    assert ("LGR52", 1773, 1805) in sset
    assert ("LGR52", 1808, 1837) in sset
    assert sum(1 for s, *_ in segs if s == "LGR52") == 2


def test_recording_segments_counts(lgr2, pgm):
    assert len(fs._recording_segments(lgr2, injury_event=True)) == 26
    assert len(fs._recording_segments(pgm)) == 45


# --- plot_demograph smoke tests --------------------------------------------

def test_plot_demograph_returns_fig_ax(lgr2):
    fig, ax = dpl.plot_demograph(lgr2, show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    n = len(dpl.series_names(lgr2))
    # one dashed baseline per series + one solid line per recording segment
    assert len(ax.lines) >= n + len(fs._recording_segments(lgr2))
    # three event-marker scatter collections (scars, injuries, pith/bark present)
    assert len(ax.collections) == 3
    # y labels are the series names
    assert [t.get_text() for t in ax.get_yticklabels()] == dpl.series_names(lgr2)
    plt.close(fig)


def test_plot_demograph_scar_marker_count(lgr2):
    fig, ax = dpl.plot_demograph(lgr2, show=False)
    # the scar scatter has one point per fire scar (count_scar == 9 for lgr2)
    scar_coll = ax.collections[0]
    assert len(scar_coll.get_offsets()) == dpl.count_scar(lgr2)
    plt.close(fig)


def test_plot_demograph_composite_rug_and_legend(pgm):
    fig, ax = dpl.plot_demograph(pgm, composite_rug=True, plot_legend=True,
                                 ylabels=False, show=False)
    assert ax.get_legend() is not None
    assert ax.get_ylim()[0] <= -1.0            # room made for the rug
    assert ax.get_yticks().size == 0           # y labels suppressed
    plt.close(fig)


def test_plot_demograph_color_group_accepts_series(lgr2):
    names = dpl.series_names(lgr2)
    groups = pd.Series({s: ("odd" if i % 2 else "even")
                        for i, s in enumerate(names)})
    fig, ax = dpl.plot_demograph(lgr2, color_group=groups, plot_legend=True,
                                 show=False)
    # two distinct line colors used for the two groups
    colors = {tuple(l.get_color()) if not isinstance(l.get_color(), str)
              else l.get_color() for l in ax.lines}
    assert len(colors) >= 2
    plt.close(fig)


def test_plot_demograph_does_not_mutate(lgr2):
    before = lgr2.copy(deep=True)
    attrs = copy.deepcopy(dict(lgr2.attrs))
    fig, _ = dpl.plot_demograph(lgr2, composite_rug=True, show=False)
    plt.close(fig)
    pd.testing.assert_frame_equal(lgr2, before, check_exact=True)
    assert dict(lgr2.attrs) == attrs


# --- plot_intervals_dist ---------------------------------------------------

def test_plot_intervals_dist(pgm):
    iv = dpl.intervals(dpl.composite(pgm))
    fig, ax = dpl.plot_intervals_dist(iv, binwidth=5, show=False)
    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) > 0                 # histogram bars
    assert ax.get_xlabel() == "Interval (years)"
    plt.close(fig)
    # also accepts the raw interval array
    fig2, ax2 = dpl.plot_intervals_dist(iv.intervals, show=False)
    assert len(ax2.patches) > 0
    plt.close(fig2)
