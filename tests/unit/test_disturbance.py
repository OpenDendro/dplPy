import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


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
