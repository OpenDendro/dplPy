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
        dpl.interseries_corr(input_df, corr="Kendall")
    expected_errorMsg = "corr must be one of Spearman / Pearson, got 'Kendall'."
    assert expected_errorMsg == str(errorMsg.value)


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

    mean_corr, result_df = dpl.interseries_corr(input_df, prewhiten=False, biweight=False)

    expected_df = pd.DataFrame(
        data={"interseries_corr": [0.857, 0.934, 0.929],
              "p_val": [0.0032650086273576452, 0.0003395528726155486, 0.00043148409144998836]},
        index=pd.Index(data=["SeriesA", "SeriesB", "SeriesC"], name="series"),
    )

    pd.testing.assert_frame_equal(expected_df, result_df)
    # the mean is returned first, rounded to 3 decimals
    assert mean_corr == round((0.857 + 0.934 + 0.929) / 3, 3)


def test_interseries_cor_values_pearson_no_prewhiten():
    input_df = pd.DataFrame(data={"SeriesA": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0],
                                    "SeriesB": [8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0],
                                    "SeriesC": [20.0, 19.0, 22.0, 21.0, 23.0, 25.0, 24.0, 26.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))

    mean_corr, result_df = dpl.interseries_corr(input_df, prewhiten=False, biweight=False, corr="Pearson")

    assert list(result_df.index) == ["SeriesA", "SeriesB", "SeriesC"]
    assert (result_df["interseries_corr"] > 0.8).all()
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

    mean_corr, result_df = dpl.interseries_corr(input_df)

    assert list(result_df.index) == ["SeriesA", "SeriesB", "SeriesC"]
    assert list(result_df.columns) == ["interseries_corr", "p_val"]
    assert result_df["interseries_corr"].between(-1, 1).all()
    assert result_df["p_val"].between(0, 1).all()
    assert -1 <= mean_corr <= 1
    assert mean_corr == round(mean_corr, 3)              # reported to 3 decimals


def test_short_and_empty_series_get_nan_not_crash():
    # A series that overlaps the master by fewer than 3 years (or not at all) used
    # to crash interseries_corr (pearsonr on n<2) or return a spurious +/-1. It must
    # now get an NaN correlation and a naming warning, and must not drag the mean.
    import io, contextlib, warnings
    import numpy as np, pandas as pd
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
        mean_corr, tbl = _quiet(dpl.interseries_corr, sub)

    assert np.isnan(tbl.loc["SHORT1", "interseries_corr"])
    assert np.isnan(tbl.loc["EMPTY1", "interseries_corr"])
    assert not np.isnan(tbl.loc["CAM011", "interseries_corr"])   # real series unaffected
    assert np.isfinite(mean_corr)                                # mean skips the NaNs
    assert any("fewer than 3 years" in str(x.message) for x in w)
