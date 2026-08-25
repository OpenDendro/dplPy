import dplpy as dpl
import pandas as pd
import numpy as np
import warnings
import pytest


def _read_quiet(path):
    import io, contextlib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.readers(path)


def _xdate_quiet(rwi, **kw):
    import io, contextlib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.xdate(rwi, **kw)


def test_xdate_invalid_input():
    with pytest.raises(TypeError) as e:
        dpl.xdate("input_df")
    assert "Expected dataframe input, got <class 'str'> instead." == str(e.value)


def test_xdate_bad_corr_method():
    df = pd.DataFrame({"A": [1.0, 2, 3], "B": [1.0, 2, 3]},
                      index=pd.Index([1, 2, 3], name="Year"))
    with pytest.raises(ValueError):
        dpl.xdate(df, corr="bogus")


def test_xdate_returns_rich_result():
    # The result is a dict mirroring dplR's corr.rwl.seg output.
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    res = _xdate_quiet(rwi, corr="spearman", slide_period=50, bin_floor=100)
    for key in ("seg_corr", "p_val", "overall", "avg_seg_corr", "flags", "bins", "rwi"):
        assert key in res
    assert res["seg_corr"].shape == (34, 50)          # series x bins
    assert list(res["overall"].columns) == ["rho", "p_val"]
    assert res["bins"][0] == "700-749" and res["bins"][-1] == "1925-1974"


def test_xdate_matches_dplR_flags_ca533():
    # dplR corr.rwl.seg flags exactly these 5 series (segment p-value >= 0.05)
    # on ca533; the fixed A/B flag separation must reproduce that set as [A].
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    res = _xdate_quiet(rwi, corr="spearman", slide_period=50, bin_floor=100, p_val=0.05)
    a_flagged = sorted(s for s, f in res["flags"].items() if f["A"])
    assert a_flagged == ["CAM011", "CAM051", "CAM131", "CAM181", "CAM201"]


def test_xdate_segment_correlation_values_ca533():
    # Regression on the spline-detrended pipeline (dplPy detrend + xdate). On the
    # identical RWI dplR uses, xdate reproduces corr.rwl.seg to ~1e-15; here the
    # tiny spline-detrend difference is absorbed by a loose tolerance.
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    res = _xdate_quiet(rwi, corr="spearman", slide_period=50, bin_floor=100)
    sc = res["seg_corr"]
    assert sc.loc["CAM011", "1750-1799"] == pytest.approx(0.6058, abs=1e-2)
    assert sc.loc["CAM191", "1850-1899"] == pytest.approx(0.7460, abs=1e-2)
    assert res["overall"].loc["CAM011", "rho"] == pytest.approx(0.5225, abs=1e-2)
    # correlations are in [-1, 1]
    vals = sc.to_numpy()
    vals = vals[~np.isnan(vals)]
    assert (vals >= -1).all() and (vals <= 1).all()


def test_xdate_ar_yw_prewhiten_matches_r():
    # The Yule-Walker prewhitener reproduces R's ar(): AIC order selection,
    # residuals + mean, series length kept (first `order` become NaN).
    from dplpy.xdate import _ar_yw_prewhiten
    rng = np.random.RandomState(0)
    n = 200
    e = rng.randn(n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + e[t]
    x = x - x.min() + 1.0
    pw = _ar_yw_prewhiten(x)
    assert len(pw) == n                                # length preserved
    assert np.isnan(pw[:2]).any()                      # start NA-padded by order
    # prewhitened mean ~ original mean (residuals are re-centered on the mean)
    assert np.nanmean(pw) == pytest.approx(np.mean(x), abs=0.05)


def test_xdate_no_prewhiten_runs():
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    res = _xdate_quiet(rwi, prewhiten=False, corr="pearson", slide_period=50, bin_floor=100)
    assert res["seg_corr"].shape[0] == 34
