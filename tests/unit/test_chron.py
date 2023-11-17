import pandas as pd
import dplpy as dpl
import pytest
from unittest.mock import patch, Mock

def mock_tbrm_out(inp):
    return sum(inp)

def mock_ar_func_out(inp_series):
    inp_series += 0.01
    return inp_series

def test_chron_simple_means():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))

    expected_df = pd.DataFrame(data={"Mean RWI": [0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55],
                                    "Sample depth": [2, 2, 2, 2, 2, 2, 2, 2]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    result_df = dpl.chron(input_df, biweight=False, prewhiten=False, plot=False)
    
    pd.testing.assert_frame_equal(expected_df, result_df)
    

def test_wrong_input():
    with pytest.raises(TypeError) as errorMsg:
        dpl.chron("string")
    
    assert "Expected pandas dataframe as input, got <class 'str'> instead" == str(errorMsg.value)

@patch('chron.tbrm')
@patch('chron.ar_func')
def test_chron_biweight_means(mock_ar_func: Mock, mock_tbrm: Mock):
    mock_tbrm.side_effect = mock_tbrm_out
    mock_ar_func.side_effect = mock_ar_func_out

    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    expected_df = pd.DataFrame(data={"Mean RWI": [0.3, 0.7, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1],
                                     "Sample depth": [2, 2, 2, 2, 2, 2, 2, 2]},
                                     index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    result_df = dpl.chron(input_df, biweight=True, prewhiten=False, plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)
    mock_tbrm.assert_called()
    mock_ar_func.assert_not_called()
    

@patch('chron.tbrm')
@patch('chron.ar_func')
def test_chron_prewhiten(mock_ar_func: Mock, mock_tbrm: Mock):
    mock_tbrm.side_effect = mock_tbrm_out
    mock_ar_func.side_effect = mock_ar_func_out

    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    expected_df = pd.DataFrame(data={"Mean RWI": [0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55],
                                     "Mean Res": [0.16, 0.36, 0.56, 0.76, 0.96, 1.16, 1.36, 1.56],
                                     "Sample depth": [2, 2, 2, 2, 2, 2, 2, 2]},
                                     index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    result_df = dpl.chron(input_df, biweight=False, prewhiten=True, plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)
    mock_tbrm.assert_not_called()
    mock_ar_func.assert_called()

# TODO: Add unit test for plot
def test_chron_plot():
    pass