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
    import io, contextlib, warnings
    import dplpy as dpl
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            data = dpl.readers("tests/data/csv/ca533.csv")
            rwi = dpl.detrend(data, fit="spline", plot=False)
            sc = dpl.series_corr(rwi, "CAM011", make_plot=False)
            xd = dpl.xdate(rwi, show_flags=False)
    assert set(sc.keys()) == {"moving_corr", "seg_corr", "overall", "lag_table",
                              "ccf", "ccf_bins", "bins"}
    assert sc["overall"][0] == pytest.approx(xd["overall"].loc["CAM011", "rho"], abs=1e-9)
    # dplR-style ccf: 11 lags (-5..5), and a well-dated series peaks at lag 0
    assert list(sc["ccf"].index) == ["lag." + str(k) for k in range(-5, 6)]
    for b in sc["ccf_bins"]:
        col = sc["ccf"][b]
        if col.notna().all():
            assert col.idxmax() == "lag.0"


def test_series_corr_which_selects_figures():
    # `which` chooses which figure(s) render; make_plot is the master switch.
    import io, contextlib, warnings
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
                dpl.series_corr(rwi, "CAM011", make_plot=True, which=w)
                counts[w] = len(plt.get_fignums())
            plt.close("all")
            dpl.series_corr(rwi, "CAM011", make_plot=False)
            counts["none"] = len(plt.get_fignums())
    assert counts == {"both": 2, "moving": 1, "ccf": 1, "none": 0}
    with pytest.raises(ValueError):
        dpl.series_corr(rwi, "CAM011", which="bogus")


def test_series_corr_ccf_matches_dplR_ccf_series_rwl():
    # dplPy's Pearson ccf reproduces R's ccf() (stats::ccf) exactly. Here we
    # reproduce R's ccf definition inline and confirm dplPy's ccf table matches
    # it for a well-dated series (series_x=True => ccf(x=series, y=master)).
    import io, contextlib, warnings
    import numpy as np
    from dplpy.xdate import normalize_for_crossdating, _row_biweight
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            data = dpl.readers("tests/data/csv/ca533.csv")
            rwi = dpl.detrend(data, fit="spline", plot=False)
            sc = dpl.series_corr(rwi, "CAM011", seg_length=50, bin_floor=100,
                                 make_plot=False, series_x=True)
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
