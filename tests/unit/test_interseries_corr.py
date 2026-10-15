import pandas as pd
import dplpy as dpl
import pytest


def test_interseries_cor_wrong_data_type():
    with pytest.raises(TypeError) as errorMsg:
        dpl.interseries_corr("input_df")
    expected_errorMsg = "Expected dataframe input, got <class 'str'> instead."
    assert expected_errorMsg == str(errorMsg.value)


def test_interseries_cor_wrong_corr_type():
    input_df = pd.DataFrame(data={"SeriesA": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0],
                                    "SeriesB": [8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0],
                                    "SeriesC": [20.0, 19.0, 22.0, 21.0, 23.0, 25.0, 24.0, 26.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))
    with pytest.raises(ValueError) as errorMsg:
        dpl.interseries_corr(input_df, corr="Bogus")
    expected_errorMsg = "corr must be one of Spearman / Pearson / Kendall, got 'Bogus'."
    assert expected_errorMsg == str(errorMsg.value)
    # kendall is accepted (the one _corr_pval helper supports it), case-insensitively
    res = dpl.interseries_corr(input_df, corr="KENDALL", prewhiten=False)
    assert res["table"]["rho"].between(-1, 1).all()


'''
    Values chosen so the result can be hand-verified: with prewhiten=False and
    biweight=False, each series is only horizontal-detrended (divided by its
    own mean) before being correlated (Spearman) against the plain arithmetic
    mean of the other two series -- no AR model or robust mean involved.
    Independently recomputing this in numpy/pandas/scipy outside of dplPy
    reproduces these exact numbers.
'''
def test_interseries_cor_values_no_prewhiten():
    input_df = pd.DataFrame(data={"SeriesA": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0],
                                    "SeriesB": [8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0],
                                    "SeriesC": [20.0, 19.0, 22.0, 21.0, 23.0, 25.0, 24.0, 26.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))

    res = dpl.interseries_corr(input_df, prewhiten=False, biweight=False)
    mean_corr, result_df = res["mean"], res["table"]

    expected_df = pd.DataFrame(
        data={"rho": [0.857, 0.934, 0.929],
              "p_val": [0.0032650086273576452, 0.0003395528726155486, 0.00043148409144998836]},
        index=pd.Index(data=["SeriesA", "SeriesB", "SeriesC"], name="series"),
    )

    pd.testing.assert_frame_equal(expected_df, result_df)
    # the mean is the dict's primary key, rounded to 3 decimals (D9)
    assert mean_corr == round((0.857 + 0.934 + 0.929) / 3, 3)
    assert res["n_series"] == 3 and res["too_short"] == []


def test_interseries_cor_values_pearson_no_prewhiten():
    input_df = pd.DataFrame(data={"SeriesA": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0],
                                    "SeriesB": [8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0],
                                    "SeriesC": [20.0, 19.0, 22.0, 21.0, 23.0, 25.0, 24.0, 26.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))

    res = dpl.interseries_corr(input_df, prewhiten=False, biweight=False, corr="Pearson")
    mean_corr, result_df = res["mean"], res["table"]

    assert list(result_df.index) == ["SeriesA", "SeriesB", "SeriesC"]
    assert (result_df["rho"] > 0.8).all()
    assert (result_df["p_val"] < 0.05).all()
    assert 0.8 < mean_corr <= 1.0


'''
    Structural sanity check on the default settings (prewhiten=True,
    biweight=True, corr="Spearman"): exact values depend on the AR model fit
    to each series, so this checks shape and value ranges rather than exact
    numbers.
'''
def test_interseries_cor_default_settings_sane():
    input_df = pd.DataFrame(
        data={
            "SeriesA": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0, 20.0, 18.0],
            "SeriesB": [8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0, 13.0, 15.0, 16.0, 14.0],
            "SeriesC": [20.0, 19.0, 22.0, 21.0, 23.0, 25.0, 24.0, 26.0, 25.0, 27.0, 28.0, 26.0],
        },
        index=pd.Index(data=list(range(1, 13)), name="Year"),
    )

    res = dpl.interseries_corr(input_df)
    mean_corr, result_df = res["mean"], res["table"]

    assert list(result_df.index) == ["SeriesA", "SeriesB", "SeriesC"]
    assert list(result_df.columns) == ["rho", "p_val"]
    assert result_df["rho"].between(-1, 1).all()
    assert result_df["p_val"].between(0, 1).all()
    assert -1 <= mean_corr <= 1
    assert mean_corr == round(mean_corr, 3)              # reported to 3 decimals


def test_short_and_empty_series_get_nan_not_crash():
    # A series that overlaps the master by fewer than 3 years (or not at all) used
    # to crash interseries_corr (pearsonr on n<2) or return a spurious +/-1. It must
    # now get an NaN correlation and a naming warning, and must not drag the mean.
    import io
    import contextlib
    import warnings
    import numpy as np
    import pandas as pd
    import dplpy as dpl

    def _quiet(fn, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)

    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    sub = rwl[["CAM011", "CAM021", "CAM031"]].copy()
    yrs = sub.dropna(how="all").index
    short = pd.Series(np.nan, index=sub.index)
    short.loc[yrs[:2]] = [0.5, 0.6]                 # only 2 overlapping years
    sub["SHORT1"] = short
    sub["EMPTY1"] = np.nan                           # no values at all

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = _quiet(dpl.interseries_corr, sub)
    mean_corr, tbl = res["mean"], res["table"]

    assert np.isnan(tbl.loc["SHORT1", "rho"])
    assert np.isnan(tbl.loc["EMPTY1", "rho"])
    assert not np.isnan(tbl.loc["CAM011", "rho"])   # real series unaffected
    assert np.isfinite(mean_corr)                                # mean skips the NaNs
    assert any("fewer than 3 years" in str(x.message) for x in w)
    assert sorted(res["too_short"]) == ["EMPTY1", "SHORT1"]


def test_interseries_corr_rows_in_input_order_and_match_xdate():
    # D9: rows follow the input column order (not alphabetical) and the per-series
    # values are xdate's overall rho (same transform, same leave-one-out master).
    import io
    import contextlib
    import warnings
    import numpy as np

    def _quiet(fn, *a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                return fn(*a, **k)

    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    sub = rwl[["CAM031", "CAM011", "CAM021"]]
    res = _quiet(dpl.interseries_corr, sub)
    assert list(res["table"].index) == ["CAM031", "CAM011", "CAM021"]
    xd = _quiet(dpl.xdate, sub, show_flags=False)
    assert np.allclose(res["table"]["rho"].to_numpy(), xd["overall"]["rho"].to_numpy())
    assert res["mean"] == round(float(res["table"]["rho"].mean()), 3)
    # A4: a shuffled index gives the same answer
    shuf = _quiet(dpl.interseries_corr, sub.sample(frac=1.0, random_state=4))
    pd.testing.assert_frame_equal(res["table"], shuf["table"])
    # A10: ar_max reaches the prewhitening
    r1 = _quiet(dpl.interseries_corr, sub, ar_max=1)
    assert not r1["table"]["rho"].equals(res["table"]["rho"])
