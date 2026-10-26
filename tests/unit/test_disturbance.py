import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _read_quiet(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.readers(path)


# Release years from the MATLAB golden master (v267_rga, sig=0) on Dan
# Druckenbrod's mo024 post-oak test set: 94 releases across 52 of 61 series.
# See dev/druckenbrod/ for the MATLAB source and validate_releases.py.
_MO024_RELEASES = {
    'DEM01A': [1915], 'DEM01B': [1914], 'DEM01C': [1914],
    'DEM03A': [1809, 1813, 1964], 'DEM07A': [1779, 1803], 'DEM07B': [1781, 1801],
    'DEM07C': [1780, 1803], 'DEM11A': [1809, 1907], 'DEM11B': [1804, 1905],
    'DEM11C': [1804, 1906], 'DEM14B': [1902, 1907], 'DEM14E': [1809],
    'DEM15A': [1839, 1901], 'DEM17B': [1804, 1901], 'DEM17C': [1746, 1750, 1901],
    'DEM18B': [1769, 1803, 1807], 'DEM19A': [1839], 'DEM19B': [1901],
    'DEM21A': [1865, 1872, 1874], 'DEM21B': [1906], 'DEM26A': [1756, 1803],
    'DEM26B': [1907, 1909], 'DEM37A': [1760, 1801, 1905, 1938], 'DEM38A': [1895],
    'DEM38B': [1807, 1901], 'DEM40B': [1806, 1836], 'DEM41A': [1768, 1809],
    'DEM41B': [1767, 1809], 'DEM43A': [1715], 'DEM43B': [1715, 1918, 1946],
    'DEM44A': [1936], 'DEM44B': [1934], 'DEM46A': [1938], 'DEM46B': [1934, 1939],
    'DEM47A': [1835, 1901], 'DEM47B': [1901], 'DEM49A': [1810, 1839, 1901],
    'DEM49B': [1814, 1895, 1901], 'DEM50A': [1925, 1939, 1944], 'DEM50B': [1934],
    'DEM51A': [1944], 'DEM52A': [1804, 1907, 1938], 'DEM52B': [1902, 1941],
    'DEM53A': [1941], 'DEM53B': [1943], 'DEM54A': [1906, 1941], 'DEM54B': [1901],
    'DEM55A': [1902, 1954], 'DEM55B': [1902], 'DEM57A': [1882, 1901],
    'DEM57B': [1901], 'DEM59B': [1907],
}


def test_releases_synthetic_known_event():
    # A clean step from width 1 to width 3 at index 20. With a 5-year window the
    # percent growth change peaks uniquely at the first year whose preceding
    # 5-year mean is still 1.0 and whose subsequent 5-year mean is already 3.0
    # (index 19 -> year 1919), at exactly 200%.
    raw = np.r_[np.ones(20), 3.0 * np.ones(20)]
    s = pd.Series(raw, index=np.arange(1900, 1940), name="SYN")
    r = dpl.releases(s, threshold=50, window_length=5)
    assert r["year"].tolist() == [1919]
    assert abs(r["pct_growth_change"].iloc[0] - 200.0) < 1e-9


def test_releases_constant_series_has_none():
    s = pd.Series(np.ones(60), index=np.arange(1900, 1960), name="FLAT")
    r = dpl.releases(s, threshold=50, window_length=15)
    assert len(r) == 0
    assert list(r.columns) == ["series", "year", "pct_growth_change"]


def test_releases_too_short_series_warns_and_skips():
    s = pd.Series(np.linspace(1, 2, 10), index=np.arange(2000, 2010), name="SH")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = dpl.releases(s, window_length=15)
    assert len(r) == 0
    assert any("skipped" in str(x.message) for x in w)


def test_releases_dataframe_sorted_series_then_year():
    raw = np.r_[np.ones(20), 3.0 * np.ones(20)]
    df = pd.DataFrame({"B": raw, "A": raw}, index=np.arange(1900, 1940))
    r = dpl.releases(df, threshold=50, window_length=5)
    assert r["series"].tolist() == ["A", "B"]


def test_releases_return_info_curves():
    raw = np.r_[np.ones(20), 3.0 * np.ones(20)]
    s = pd.Series(raw, index=np.arange(1900, 1940), name="SYN")
    table, info = dpl.releases(s, threshold=50, window_length=5, return_info=True)
    assert set(info) == {"SYN"}
    cols = list(info["SYN"].columns)
    assert cols == ["running_mean", "pct_growth_change"]
    assert info["SYN"].index.min() == 1900 and info["SYN"].index.max() == 1939


def test_releases_significance_not_implemented():
    s = pd.Series(np.ones(60), index=np.arange(1900, 1960), name="FLAT")
    with pytest.raises(NotImplementedError):
        dpl.releases(s, significance=True)


def test_releases_bad_method_raises():
    s = pd.Series(np.ones(60), index=np.arange(1900, 1960), name="FLAT")
    with pytest.raises(ValueError):
        dpl.releases(s, method="trader")


def test_releases_accepts_series_and_frame_equivalently():
    raw = np.r_[np.ones(20), 3.0 * np.ones(20)]
    s = pd.Series(raw, index=np.arange(1900, 1940), name="SYN")
    r_series = dpl.releases(s, threshold=50, window_length=5)
    r_frame = dpl.releases(s.to_frame(), threshold=50, window_length=5)
    pd.testing.assert_frame_equal(r_series, r_frame)


def test_releases_mo024_matches_matlab_golden_master():
    # Full regression against Druckenbrod's MATLAB on his mo024 test set.
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    r = dpl.releases(data, threshold=50, window_length=15)
    assert len(r) == 94
    assert r["series"].nunique() == 52
    got = {nm: sorted(g["year"].tolist())
           for nm, g in r.groupby("series")}
    assert got == _MO024_RELEASES


# ----- disturbance_detrend -----

# Reference values from the MATLAB golden master (v267_chron(3,50,15,0,.), mo024).
# St = DSt/At climate index (sheet 4), DSt (sheet 2), At (sheet 3), StTR residual.
def test_disturbance_detrend_mo024_matches_golden_master():
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    info = dpl.disturbance_detrend(data, threshold=50, window_length=15,
                                   return_info=True)
    St = info["index"]; DSt = info["disturbance_standardized"]
    At = info["age_trend"]; StTR = info["residual_index"]
    atol = 1e-6
    assert abs(St.loc[1915, "DEM01A"] - 1.1219009924) < atol
    assert abs(St.loc[1982, "DEM01A"] - 1.1659081864) < atol
    assert abs(St.loc[1803, "DEM07A"] - 0.9508481645) < atol
    assert abs(St.loc[1900, "DEM04A"] - 0.7862283674) < atol   # a no-release series
    assert abs(DSt.loc[1946, "DEM43B"] - 0.2717307208) < atol
    assert abs(At.loc[1946, "DEM43B"] - 0.3155444786) < atol
    assert abs(StTR.loc[1914, "DEM01C"] - (-0.0723035449)) < atol
    # per-series index means
    assert abs(St["DEM01A"].mean() - 1.0329213232) < atol
    assert abs(St["DEM43B"].mean() - 1.0331350531) < atol


def test_disturbance_detrend_return_info_keys():
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    info = dpl.disturbance_detrend(data, return_info=True)
    assert set(info) == {"index", "disturbance_standardized", "age_trend",
                         "residual_index", "releases", "averages"}
    assert list(info["averages"].columns) == [
        "ring_width", "age_trend", "Dt", "index", "sample_depth"]
    assert len(info["releases"]) == 94


def test_disturbance_detrend_index_feeds_chron():
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    idx = dpl.disturbance_detrend(data)
    assert isinstance(idx, pd.DataFrame)
    assert idx.shape[1] == data.shape[1]
    ch = dpl.chron(idx, plot=False)          # must accept it like detrend output
    assert len(ch) == len(idx)


def test_disturbance_detrend_no_release_series_has_zero_increment():
    # DEM04A has no releases, so the disturbance increment Dt is zero throughout
    # and the index reduces to an age-only detrend.
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    s = data["DEM04A"]
    info = dpl.disturbance_detrend(s, return_info=True)
    dst = info["disturbance_standardized"].dropna()
    raw = s.dropna()
    assert np.allclose(dst.to_numpy(), raw.to_numpy(), atol=1e-6)  # DSt == raw


def test_disturbance_detrend_transform_variants_differ():
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    dan = dpl.disturbance_detrend(data, transform="druckenbrod")
    none = dpl.disturbance_detrend(data, transform=None)
    cook = dpl.disturbance_detrend(data, transform="powt")
    assert not np.allclose(dan.to_numpy(), none.to_numpy(), equal_nan=True)
    assert not np.allclose(dan.to_numpy(), cook.to_numpy(), equal_nan=True)


def test_disturbance_detrend_bad_transform_raises():
    s = pd.Series(np.ones(60), index=np.arange(1900, 1960), name="FLAT")
    with pytest.raises(ValueError):
        dpl.disturbance_detrend(s, transform="banana")


def test_disturbance_detrend_series_in_series_out():
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    out = dpl.disturbance_detrend(data["DEM01A"])
    assert isinstance(out, pd.Series)


def test_disturbance_detrend_significance_not_implemented():
    s = pd.Series(np.ones(60), index=np.arange(1900, 1960), name="FLAT")
    with pytest.raises(NotImplementedError):
        dpl.disturbance_detrend(s, significance=True)


def test_disturbance_detrend_too_short_all_nan_warns():
    s = pd.Series(np.linspace(1, 2, 20), index=np.arange(2000, 2020), name="SH")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = dpl.disturbance_detrend(s, window_length=15)
    assert out.isna().all()
    assert any("NaN" in str(x.message) for x in w)


# ----- Q8: absent rings (recode_zeros) and the micro-ring denominator guard -----

def test_min_mean_guard_suppresses_near_zero_baseline():
    # A 100% "release" between two essentially-absent levels (0.001 -> 0.002) is a
    # denominator artifact. The default guard removes it; min_mean=0 reproduces the
    # unguarded v267_rga behaviour and detects it.
    raw = np.r_[np.full(20, 0.001), np.full(20, 0.002)]
    s = pd.Series(raw, index=np.arange(1900, 1940), name="ART")
    assert len(dpl.releases(s, window_length=5, threshold=50, min_mean=0.0)) == 1
    assert len(dpl.releases(s, window_length=5, threshold=50, min_mean=0.01)) == 0


def test_min_mean_default_preserves_real_release():
    # A genuine release off a healthy baseline is untouched by the default guard.
    raw = np.r_[np.ones(20), 3.0 * np.ones(20)]
    s = pd.Series(raw, index=np.arange(1900, 1940), name="SYN")
    assert dpl.releases(s, window_length=5)["year"].tolist() == [1919]


def test_releases_zeros_warn_when_not_recoded():
    raw = np.r_[np.ones(19), [0.0], np.ones(20)]       # one absent ring
    s = pd.Series(raw, index=np.arange(1900, 1940), name="ABS")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dpl.releases(s, window_length=5)
    assert any("absent" in str(x.message) for x in w)


def test_recode_zeros_does_not_mutate_input():
    raw = np.r_[np.ones(19), [0.0], np.ones(20)]
    s = pd.Series(raw, index=np.arange(1900, 1940), name="ABS")
    before = s.copy(deep=True)
    dpl.releases(s, window_length=5, recode_zeros=True)
    dpl.disturbance_detrend(s, window_length=5, recode_zeros=True)
    pd.testing.assert_series_equal(s, before)           # zero still present, unchanged


def test_disturbance_detrend_mo024_unchanged_by_default_guard():
    # The default min_mean=0.01 must not perturb the validated golden-master result.
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    st_default = dpl.disturbance_detrend(data)
    st_noguard = dpl.disturbance_detrend(data, min_mean=0.0)
    pd.testing.assert_frame_equal(st_default, st_noguard)


# ----- plotting (phase 3) -----

def test_disturbance_plot_summary_returns_figure():
    import matplotlib.pyplot as plt
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    fig = dpl.disturbance_plot(data, show=False)
    assert fig.__class__.__name__ == "Figure"
    assert len(fig.axes) == 4            # 3 panels + 1 twin axis (depth)
    plt.close("all")


def test_disturbance_plot_named_series_returns_figure():
    import matplotlib.pyplot as plt
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    fig = dpl.disturbance_plot(data, series="DEM01A", show=False)
    assert fig.__class__.__name__ == "Figure"
    assert len(fig.axes) == 4            # 3 panels + 1 twin axis (pgc)
    plt.close("all")


def test_disturbance_plot_series_list_returns_list():
    import matplotlib.pyplot as plt
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    figs = dpl.disturbance_plot(data, series=["DEM01A", "DEM07B"], show=False)
    assert isinstance(figs, list) and len(figs) == 2
    plt.close("all")


def test_disturbance_detrend_plot_preserves_numerics():
    import matplotlib.pyplot as plt
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    base = dpl.disturbance_detrend(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        withplot = dpl.disturbance_detrend(data, plot=True)
    pd.testing.assert_frame_equal(base, withplot)
    plt.close("all")


def test_disturbance_detrend_plot_series_input():
    import matplotlib.pyplot as plt
    data = _read_quiet("tests/data/rwl/mo024.rwl")
    out = dpl.disturbance_detrend(data["DEM01A"], plot=True)
    assert isinstance(out, pd.Series)
    plt.close("all")
