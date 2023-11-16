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
