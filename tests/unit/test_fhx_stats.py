"""Tests for the fire-history statistics, compositing, and interval analysis
layers (dplpy.fhx_stats and dplpy.fhx_intervals).

Golden values are burnr's own: the series-statistics, sample-depth, and
percent-scarred figures match the assertions in burnr's test-stats.R on lgr2,
and the pgm composite years plus interval statistics match burnr's
test-intervals.R and a direct burnr run (composite(pgm) -> intervals()). The full
fhx_stats table, fhx_sample_depth, percent_scarred, and composite year lists were
confirmed identical to burnr during development.
"""
import copy

import numpy as np
import pandas as pd
import pytest

import importlib

import dplpy as dpl


def _series(x):
    return list(x["series"].unique())


def _get_series(x, s):
    return x[x["series"].astype(str) == s]
fs = importlib.import_module("dplpy.fhx_stats")  # module, not the fhx_stats() fn

LGR2 = "tests/data/fhx/lgr2.fhx"
PGM = "tests/data/fhx/pgm.fhx"


@pytest.fixture
def lgr2():
    return dpl.fhx.read(LGR2)


@pytest.fixture
def pgm():
    return dpl.fhx.read(PGM)


# --- per-series scalar pieces (burnr test-stats.R golden) ------------------

def test_scalar_pieces_single_and_multi(lgr2):
    single = _get_series(lgr2, "LGR53")
    assert fs._first_year(single) == 1772
    assert fs._first_year(lgr2) == 1366
    assert fs._last_year(single) == 1843
    assert fs._last_year(lgr2) == 2012
    assert fs._count_year_span(single) == 72
    assert fs._count_year_span(lgr2) == 647
    assert fs._outer_type(single) == "outer_year"
    assert fs._outer_type(lgr2) == "bark_year"
    assert fs._inner_type(single) == "pith_year"
    assert fs._inner_type(lgr2) == "inner_year"
    assert fs._count_scar(single) == 1
    assert fs._count_scar(lgr2) == 9
    assert fs._count_injury(single) == 0
    assert fs._count_injury(lgr2) == 6
    assert fs._count_recording(single) == 38
    assert fs._count_recording(lgr2) == 514
    assert fs._count_recording(single, injury_event=True) == 38
    assert fs._count_recording(lgr2, injury_event=True) == 514


def test_series_mean_interval(lgr2):
    assert np.isnan(fs._series_mean_interval(_get_series(lgr2, "LGR53")))
    assert fs._series_mean_interval(_get_series(lgr2, "LGR46")) == 16.5
    with pytest.warns(dpl.DplpyWarning):          # warns on multiple series
        assert fs._series_mean_interval(lgr2) == 47.9


def test_series_stats_table(lgr2):
    ss = dpl.fhx.stats(lgr2)
    assert list(ss.columns) == [
        "series", "first", "last", "years", "inner_type", "outer_type",
        "number_scars", "number_injuries", "recording_years", "mean_interval"]
    assert len(ss) == 26
    # first-appearance order, matching burnr's factor-level order
    assert list(ss["series"][:4]) == ["LGR54", "LGR44", "LGR47", "LGR48"]
    lgr46 = ss[ss["series"] == "LGR46"].iloc[0]
    assert (int(lgr46["first"]), int(lgr46["last"])) == (1705, 1849)
    assert int(lgr46["number_scars"]) == 3
    assert int(lgr46["recording_years"]) == 77
    assert lgr46["mean_interval"] == 16.5


def test_summary_fhx(lgr2):
    assert dpl.fhx.summary(lgr2) == {
        "number_series": 26, "first_year": 1366, "last_year": 2012,
        "number_scars": 9, "number_injuries": 6}


# --- sample depth ----------------------------------------------------------

def test_sample_depth(lgr2):
    sd = dpl.fhx.sample_depth(lgr2).set_index("year")["samp_depth"]
    assert [int(sd.loc[y]) for y in (1366, 1436, 2011, 2012)] == [1, 2, 13, 2]
    single = dpl.fhx.sample_depth(_get_series(lgr2, "LGR53")).set_index("year")
    assert int(single.loc[1800, "samp_depth"]) == 1


# --- percent scarred -------------------------------------------------------

def test_percent_scarred(lgr2):
    ps = dpl.fhx.percent_scarred(lgr2).set_index("year")["percent_scarred"]
    assert ps.loc[1773] == 50
    assert ps.loc[1774] == 0
    psi = dpl.fhx.percent_scarred(lgr2, injury_event=True).set_index(
        "year")["percent_scarred"]
    assert psi.loc[1806] == 88
    # columns and recording denominator
    cols = list(dpl.fhx.percent_scarred(lgr2).columns)
    assert cols == ["year", "num_rec", "num_scars", "percent_scarred"]


# --- composite -------------------------------------------------------------

def _comp_years(comp):
    name = _series(comp)[0]
    return sorted(int(v) for v in dpl.fhx.events(comp)[name])


def test_composite_lgr2(lgr2):
    assert _comp_years(dpl.fhx.composite(lgr2)) == [1423, 1448, 1773, 1786, 1806]


def test_composite_pgm(pgm):
    assert _comp_years(dpl.fhx.composite(pgm)) == [
        1636, 1648, 1664, 1674, 1685, 1714, 1724, 1737, 1746, 1748, 1752,
        1782, 1789, 1795, 1798, 1806, 1818, 1824, 1842, 1863, 1871]
    # non-default thresholds with injuries
    assert _comp_years(dpl.fhx.composite(pgm, filter_prop=0.1, filter_min_rec=3,
                                     injury_event=True)) == [
        1664, 1737, 1746, 1748, 1752, 1782, 1789, 1795, 1798, 1801, 1806,
        1818, 1824, 1829, 1842, 1863, 1871, 1993]


def test_composite_is_one_series_fhx(pgm):
    comp = dpl.fhx.composite(pgm)
    assert dpl.fhx.is_fhx(comp)
    assert len(_series(comp)) == 1
    assert _series(comp)[0] == "COMP"
    # composite names the scar years unknown_fs, the ends inner/outer
    assert fs._inner_type(comp) == "inner_year"
    assert fs._outer_type(comp) == "outer_year"


def test_composite_empty_when_nothing_qualifies(lgr2):
    comp = dpl.fhx.composite(lgr2, filter_min_rec=9999)
    assert dpl.fhx.is_fhx(comp)
    assert len(comp) == 0


# --- intervals (burnr test-intervals.R golden on pgm) ----------------------

def test_intervals_observed_stats(pgm):
    iv = dpl.fhx.intervals(dpl.fhx.composite(pgm))
    assert iv.intervals.size == 20
    assert iv.mean() == 11.75
    assert iv.median() == 10
    assert iv.min() == 2
    assert iv.max() == 30
    assert iv.comp_name == "COMP"
    assert iv.event_range == (1636, 1871)


def test_intervals_weibull_fit_and_quantiles(pgm):
    iv = dpl.fhx.intervals(dpl.fhx.composite(pgm))
    assert iv.densfun == "weibull"
    assert iv.fit["shape"] == pytest.approx(1.656521, abs=1e-4)
    assert iv.fit["scale"] == pytest.approx(13.21254, abs=1e-3)
    q = np.atleast_1d(iv.quantile())
    assert q == pytest.approx([3.918526, 10.590013, 20.555245], abs=1e-3)
    assert iv.quantile(0.25) == pytest.approx(6.227954, abs=1e-3)
    assert iv.weibull_median == pytest.approx(10.59001, abs=1e-3)
    # goodness of fit: KS D matches R exactly; Shapiro W matches R
    assert iv.kstest["statistic"] == pytest.approx(0.1277607, abs=1e-5)
    assert iv.shapirotest["statistic"] == pytest.approx(0.8848471, abs=1e-5)


def test_intervals_lognormal_fit(pgm):
    iv = dpl.fhx.intervals(dpl.fhx.composite(pgm), densfun="lognormal")
    assert iv.fit["meanlog"] == pytest.approx(2.25387, abs=1e-4)
    assert iv.fit["sdlog"] == pytest.approx(0.6779964, abs=1e-5)
    q = np.atleast_1d(iv.quantile())
    assert q == pytest.approx([4.366392, 9.524522, 20.776083], abs=1e-3)


def test_intervals_summary_and_repr(pgm):
    iv = dpl.fhx.intervals(dpl.fhx.composite(pgm))
    s = iv.summary()
    assert s["total_intervals"] == 20
    assert s["mean_interval"] == 11.8          # rounded to 1 dp
    assert s["minimum_interval"] == 2
    assert set(s["percentiles"]) == {"lower_12.5", "median_50.0", "upper_87.5"}
    assert isinstance(repr(iv), str) and "Fire interval analysis" in repr(iv)


def test_intervals_guards(pgm):
    comp = dpl.fhx.composite(pgm)
    with pytest.raises(ValueError):
        dpl.fhx.intervals(comp, densfun="bogus")
    with pytest.raises(ValueError):            # multiple series
        dpl.fhx.intervals(dpl.fhx.frame([1900, 1901], ["A", "B"],
                              ["unknown_fs", "unknown_fs"]))
    with pytest.raises(ValueError):            # too few events
        dpl.fhx.intervals(dpl.fhx.frame([1900, 1905], ["C", "C"],
                              ["unknown_fs", "unknown_fs"]))


# --- non-mutation ----------------------------------------------------------

def test_fhx_stats_do_not_mutate_input(pgm):
    before = pgm.copy(deep=True)
    attrs = copy.deepcopy(dict(pgm.attrs))
    dpl.fhx.stats(pgm)
    dpl.fhx.sample_depth(pgm)
    dpl.fhx.percent_scarred(pgm, injury_event=True)
    fs._yearly_recording(pgm)
    dpl.fhx.intervals(dpl.fhx.composite(pgm))
    pd.testing.assert_frame_equal(pgm, before, check_exact=True)
    assert dict(pgm.attrs) == attrs
