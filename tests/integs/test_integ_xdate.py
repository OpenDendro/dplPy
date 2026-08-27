import io
import contextlib

import pytest
import dplpy as dpl

# xdate() is exercised over several bin/slide/corr settings on real, messy data.
# Runtime scales with series count and each test calls xdate a few times, so we
# trim to a real subset of the series -- every code path is still hit. Assertions
# check the result structure and pin a stable whole-series correlation so a silent
# numeric change is caught, not just a crash.

_KEYS = {"avg_seg_corr", "bins", "flags", "overall", "p_val", "rwi", "seg_corr"}


def _xdate(df, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return dpl.xdate(df, **kw)


def _assert_shape(res, n_series):
    assert isinstance(res, dict) and _KEYS.issubset(res.keys())
    ov = res["overall"]
    assert list(ov.columns) == ["rho", "p_val"]
    assert len(ov) == n_series
    assert ((ov["rho"] >= -1) & (ov["rho"] <= 1)).all()   # valid correlations


def test_xdate_diff_bins():
    ca533 = dpl.readers("./tests/data/csv/ca533.csv").iloc[:, :15]
    for bf in (0, 10, 100):
        res = _xdate(ca533, bin_floor=bf)
        _assert_shape(res, 15)
    # overall rho is a whole-series correlation, independent of bin_floor:
    # pin CAM011 (well-dated) against the current dplR-validated value.
    assert res["overall"].loc["CAM011", "rho"] == pytest.approx(0.5146, abs=1e-3)


def test_xdate_diff_slide_periods():
    ca533 = dpl.readers("./tests/data/csv/ca533.csv").iloc[:, :15]
    for sp in (30, 50, 80):
        _assert_shape(_xdate(ca533, slide_period=sp), 15)


def test_xdate_diff_corrs():
    ca667 = dpl.readers("./tests/data/rwl/ca667.rwl", header=True).iloc[:, :30]
    for corr in ("Spearman", "Pearson"):
        _assert_shape(_xdate(ca667, corr=corr), 30)


def test_xdate_not_prewhitened():
    ca667 = dpl.readers("./tests/data/rwl/ca667.rwl", header=True).iloc[:, :30]
    _assert_shape(_xdate(ca667, prewhiten=False), 30)
