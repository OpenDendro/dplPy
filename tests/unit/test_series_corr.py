import dplpy as dpl
import pandas as pd
import pytest

def test_series_corr_wrong_data_type():
    with pytest.raises(TypeError) as errorMsg:
        dpl.series_corr("input_df", "series_name")
    expected_errorMsg = "Expected dataframe input, got <class 'str'> instead."
    assert expected_errorMsg == str(errorMsg.value)


def test_series_corr_wrong_series_name_type():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))
    with pytest.raises(TypeError) as errorMsg:
        dpl.series_corr(input_df, 3)
    expected_errorMsg = "Expected string input as series name, got <class 'int'> instead."
    assert expected_errorMsg == str(errorMsg.value)


def test_series_corr_series_name_not_in_df():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))
    with pytest.raises(ValueError) as errorMsg:
        dpl.series_corr(input_df, "SeriesC")
    expected_errorMsg = "Series named SeriesC not found in provided dataframe."
    assert expected_errorMsg == str(errorMsg.value)

# TODO: Add tests that validates plotted data


def test_series_corr_returns_results_and_matches_xdate():
    # series_corr's leave-one-out overall correlation for a series should equal
    # xdate's overall for that same series (both exclude the series from the
    # master). Also checks the rich return without plotting.
    import io
    import contextlib
    import warnings
    import dplpy as dpl
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            data = dpl.readers("tests/data/csv/ca533.csv")
            rwi = dpl.detrend(data, fit="spline", plot=False)
            sc = dpl.series_corr(rwi, "CAM011", plot=False)
            xd = dpl.xdate(rwi, show_flags=False)
    assert set(sc.keys()) == {"moving_corr", "seg_corr", "overall", "lag_table",
                              "ccf", "ccf_bins", "bins", "rwi"}
    # series_corr and xdate compute the same leave-one-out overall correlation;
    # xdate rounds its reported rho to 3 decimals (series_corr keeps full precision),
    # so they agree once series_corr's value is rounded the same way. D9: overall
    # is a one-row DataFrame shaped like xdate's.
    assert list(sc["overall"].columns) == ["rho", "p_val"]
    assert list(sc["overall"].index) == ["CAM011"]
    assert round(sc["overall"].loc["CAM011", "rho"], 3) == xd["overall"].loc["CAM011", "rho"]
    # dplR-style ccf: 21 lags (-10..10, the package-wide lag default), and a
    # well-dated series peaks at lag 0
    assert list(sc["ccf"].index) == ["lag." + str(k) for k in range(-10, 11)]
    assert list(sc["lag_table"].index) == list(sc["ccf"].index)
    for b in sc["ccf_bins"]:
        col = sc["ccf"][b]
        if col.notna().all():
            assert col.idxmax() == "lag.0"
    assert sc["rwi"].attrs.get("kind") == "rwi"


def test_series_corr_which_selects_figures():
    # `which` chooses which figure(s) render; make_plot is the master switch.
    import io
    import contextlib
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            data = dpl.readers("tests/data/csv/ca533.csv")
            rwi = dpl.detrend(data, fit="spline", plot=False)
            counts = {}
            for w in ("both", "moving", "ccf"):
                plt.close("all")
                dpl.series_corr(rwi, "CAM011", plot=True, which=w)
                counts[w] = len(plt.get_fignums())
            plt.close("all")
            dpl.series_corr(rwi, "CAM011", plot=False)
            counts["none"] = len(plt.get_fignums())
    assert counts == {"both": 2, "moving": 1, "ccf": 1, "none": 0}
    with pytest.raises(ValueError):
        dpl.series_corr(rwi, "CAM011", which="bogus")


def test_series_corr_ccf_matches_dplR_ccf_series_rwl():
    # dplPy's Pearson ccf reproduces R's ccf() (stats::ccf) exactly. Here we
    # reproduce R's ccf definition inline and confirm dplPy's ccf table matches
    # it for a well-dated series (series_x=True => ccf(x=series, y=master)).
    import io
    import contextlib
    import warnings
    import numpy as np
    from dplpy.xdate import normalize_for_crossdating, _row_biweight
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            data = dpl.readers("tests/data/csv/ca533.csv")
            rwi = dpl.detrend(data, fit="spline", plot=False)
            sc = dpl.series_corr(rwi, "CAM011", window_length=50, bin_floor=100,
                                 plot=False, series_x=True, lag=5)
            ready = normalize_for_crossdating(rwi, True)
    fy, ly = int(ready.first_valid_index()), int(ready.last_valid_index())
    years = np.arange(fy, ly + 1); ready = ready.reindex(years)
    names = list(ready.columns); M = ready.to_numpy(float)
    good = np.array([np.sum(~np.isnan(M[:, i])) > 3 for i in range(M.shape[1])])
    i = names.index("CAM011"); keep = good.copy(); keep[i] = False
    master = _row_biweight(M[:, keep]); series = M[:, i]

    def rccf(x, y, lm=5):
        x = x - x.mean(); y = y - y.mean(); n = len(x)
        den = np.sqrt(np.mean(x ** 2) * np.mean(y ** 2)); o = []
        for k in range(-lm, lm + 1):
            o.append((np.sum(x[k:] * y[:n - k]) if k >= 0 else np.sum(x[:n + k] * y[-k:])) / n / den)
        return np.array(o)

    for b in sc["ccf_bins"]:
        lo, hi = int(b.split("-")[0]), int(b.split("-")[1])
        m = (years >= lo) & (years <= hi)
        if m.sum() != 50 or np.isnan(series[m]).any() or np.isnan(master[m]).any():
            continue
        ref = rccf(series[m], master[m], 5)     # series.x=TRUE convention
        assert np.allclose(sc["ccf"][b].to_numpy(), ref, atol=1e-9)


# --- v0.9.0 audit follow-ups ---------------------------------------------------

def _missing_ring_frame(n_series=8, n=200, seed=0, drop_at=100, victim="S0"):
    import numpy as np
    rng = np.random.default_rng(seed)
    years = np.arange(1800, 1800 + n)
    common = rng.normal(0, 1, n)
    cols = {}
    for k in range(n_series):
        cols["S%d" % k] = np.clip(1.0 + 0.5 * common + rng.normal(0, 0.2, n), 0.05, None)
    raw = pd.DataFrame(cols, index=pd.Index(years, name="Year"))
    v = raw[victim].to_numpy().copy()
    raw[victim] = np.concatenate([[np.nan], np.delete(v, drop_at)])   # last year stays dated
    return raw


def _quiet(fn, *a, **k):
    import io
    import contextlib
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)


def test_series_corr_lag_sign_matches_xdate_and_cofecha():
    # A9: a MISSING ring gives best lag -1 in BOTH the lag_table (xdate._lag_table,
    # master slides) and the default ccf (series_x=False, dplR's stock convention),
    # exactly as xdate and check report it. Before 0.9.0 series_corr peaked at +1.
    import numpy as np
    raw = _missing_ring_frame()
    sc = _quiet(dpl.series_corr, raw, "S0", prewhiten=False, bin_floor=0, plot=False)
    xd = _quiet(dpl.xdate, raw, prewhiten=False, bin_floor=0, show_flags=False)
    early = [b for b in sc["bins"] if int(b.split("-")[1]) < 1900
             and sc["lag_table"][b].notna().any()]
    assert early
    for b in early:
        assert sc["lag_table"][b].idxmax() == "lag.-1"
    # the dplR-style ccf (its own bins, floored from the series' first year) agrees
    early_ccf = [b for b in sc["ccf_bins"] if int(b.split("-")[1]) < 1900
                 and sc["ccf"][b].notna().any()]
    assert early_ccf
    for b in early_ccf:
        assert sc["ccf"][b].idxmax() == "lag.-1"
    assert all(e["best_lag"] == -1 for e in xd["flags"]["S0"]["B"])
    # the lag_table IS xdate's lag scan: identical numbers for the same segment
    seg = xd["flags"]["S0"]["B"][0]
    assert np.allclose(sc["lag_table"][seg["segment"]].to_numpy(), seg["lags"], equal_nan=True)
    # series_x=True flips the ccf sign convention (and only the ccf)
    flipped = _quiet(dpl.series_corr, raw, "S0", prewhiten=False, bin_floor=0,
                     plot=False, series_x=True)
    assert flipped["ccf"][early_ccf[0]].idxmax() == "lag.1"
    assert flipped["lag_table"][early[0]].idxmax() == "lag.-1"


def test_series_corr_aliases_defaults_and_options():
    # D3/D4/D11: seg_length= is a deprecated alias of window_length; lag defaults
    # to 10; which= is case-insensitive; ar_max is accepted.
    import warnings
    import io
    import contextlib
    raw = _missing_ring_frame()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            old = dpl.series_corr(raw, "S1", seg_length=40, bin_floor=0, plot=False)
    assert any(isinstance(x.message, DeprecationWarning) for x in w)
    new = _quiet(dpl.series_corr, raw, "S1", window_length=40, bin_floor=0, plot=False)
    pd.testing.assert_series_equal(old["seg_corr"], new["seg_corr"])
    assert len(new["lag_table"]) == 21
    _quiet(dpl.series_corr, raw, "S1", which="CCF", plot=False)      # case-insensitive
    with pytest.raises(ValueError):
        _quiet(dpl.series_corr, raw, "S1", which="bogus", plot=False)
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl").iloc[:, :8]
    a = _quiet(dpl.series_corr, rwl, "CAM011", plot=False, ar_max=1)
    b = _quiet(dpl.series_corr, rwl, "CAM011", plot=False, ar_max=10)
    assert not a["moving_corr"].equals(b["moving_corr"])


def test_series_corr_unsorted_index_matches_sorted():
    # A4
    raw = _missing_ring_frame()
    ref = _quiet(dpl.series_corr, raw, "S1", bin_floor=0, plot=False)
    got = _quiet(dpl.series_corr, raw.sample(frac=1.0, random_state=0), "S1",
                 bin_floor=0, plot=False)
    pd.testing.assert_series_equal(ref["seg_corr"], got["seg_corr"])
    pd.testing.assert_frame_equal(ref["lag_table"], got["lag_table"])


def test_series_corr_no_bins_warns():
    # B12
    import warnings
    import io
    import contextlib
    raw = _missing_ring_frame()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            res = dpl.series_corr(raw, "S1", window_length=500, plot=False)
    assert res["bins"] == []
    assert any("nothing can be tested" in str(x.message) for x in w)
