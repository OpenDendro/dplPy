import pandas as pd
import dplpy as dpl
import pytest


def test_interseries_cor_wrong_data_type():
    with pytest.raises(TypeError) as errorMsg:
        dpl.interseries_cor("input_df")
    expected_errorMsg = "Expected dataframe input, got <class 'str'> instead."
    assert expected_errorMsg == str(errorMsg.value)


def test_interseries_cor_wrong_corr_type():
    input_df = pd.DataFrame(data={"SeriesA": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0],
                                    "SeriesB": [8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0],
                                    "SeriesC": [20.0, 19.0, 22.0, 21.0, 23.0, 25.0, 24.0, 26.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))
    with pytest.raises(ValueError) as errorMsg:
        dpl.interseries_cor(input_df, corr="Kendall")
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

    result_df = dpl.interseries_cor(input_df, prewhiten=False, biweight=False)

    expected_df = pd.DataFrame(
        data={"interseries_cor": [0.857, 0.934, 0.929],
              "p_val": [0.0032650086273576452, 0.0003395528726155486, 0.00043148409144998836]},
        index=pd.Index(data=["SeriesA", "SeriesB", "SeriesC"], name="series"),
    )

    pd.testing.assert_frame_equal(expected_df, result_df)


def test_interseries_cor_values_pearson_no_prewhiten():
    input_df = pd.DataFrame(data={"SeriesA": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0],
                                    "SeriesB": [8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0],
                                    "SeriesC": [20.0, 19.0, 22.0, 21.0, 23.0, 25.0, 24.0, 26.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))

    result_df = dpl.interseries_cor(input_df, prewhiten=False, biweight=False, corr="Pearson")

    assert list(result_df.index) == ["SeriesA", "SeriesB", "SeriesC"]
    assert (result_df["interseries_cor"] > 0.8).all()
    assert (result_df["p_val"] < 0.05).all()


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

    result_df = dpl.interseries_cor(input_df)

    assert list(result_df.index) == ["SeriesA", "SeriesB", "SeriesC"]
    assert list(result_df.columns) == ["interseries_cor", "p_val"]
    assert result_df["interseries_cor"].between(-1, 1).all()
    assert result_df["p_val"].between(0, 1).all()
