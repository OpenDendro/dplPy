import dplpy as dpl
import pandas as pd
import pytest
from detrend import residual, difference
from unittest.mock import patch, Mock

def mock_spline_method(x, inp_arr, period):
    return inp_arr

def mock_negex_method(x, inp_arr):
    return inp_arr * 0.5

def mock_hugershoff_method(x, inp_arr):
    return inp_arr * 0.25

def mock_linear_method(x, inp_arr):
    return inp_arr * 4

def mock_horizontal_method(x, inp_arr):
    return inp_arr * 2

def test_detrend_with_invalid_input():
    with pytest.raises(TypeError) as errorMsg:
        dpl.detrend("input_df", fit="spline", plot=False)
    invalid_input_msg = "argument should be either pandas dataframe or pandas series."
    assert invalid_input_msg == str(errorMsg.value)

@patch('curvefit.horizontal')
@patch('curvefit.linear')
@patch('curvefit.hugershoff')
@patch('curvefit.negex')
@patch('detrend.spline')
def test_detrend_with_spline(mock_spline: Mock, mock_negex: Mock, mock_hugershoff: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
    mock_negex.side_effect = mock_negex_method
    mock_hugershoff.side_effect = mock_hugershoff_method
    mock_linear.side_effect = mock_linear_method
    mock_horizontal.side_effect = mock_horizontal_method
    
    expected_df = pd.DataFrame(data={"SeriesA": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                    "SeriesB": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    result_df = dpl.detrend(input_df, fit="spline", plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)

    mock_spline.assert_called()
    mock_negex.assert_not_called()
    mock_hugershoff.assert_not_called()
    mock_linear.assert_not_called()
    mock_horizontal.assert_not_called()

@patch('curvefit.horizontal')
@patch('curvefit.linear')
@patch('curvefit.hugershoff')
@patch('curvefit.negex')
@patch('detrend.spline')
def test_detrend_with_modnegex(mock_spline: Mock, mock_negex: Mock, mock_hugershoff: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
    mock_negex.side_effect = mock_negex_method
    mock_hugershoff.side_effect = mock_hugershoff_method
    mock_linear.side_effect = mock_linear_method
    mock_horizontal.side_effect = mock_horizontal_method

    expected_df = pd.DataFrame(data={"SeriesA": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                                    "SeriesB": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    result_df = dpl.detrend(input_df, fit="ModNegEx", plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)

    mock_spline.assert_not_called()
    mock_negex.assert_called()
    mock_hugershoff.assert_not_called()
    mock_linear.assert_not_called()
    mock_horizontal.assert_not_called()
    

@patch('curvefit.horizontal')
@patch('curvefit.linear')
@patch('curvefit.hugershoff')
@patch('curvefit.negex')
@patch('detrend.spline')
def test_detrend_with_hugershoff(mock_spline: Mock, mock_negex: Mock, mock_hugershoff: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
    mock_negex.side_effect = mock_negex_method
    mock_hugershoff.side_effect = mock_hugershoff_method
    mock_linear.side_effect = mock_linear_method
    mock_horizontal.side_effect = mock_horizontal_method

    expected_df = pd.DataFrame(data={"SeriesA": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                                    "SeriesB": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    result_df = dpl.detrend(input_df, fit="Hugershoff", plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)

    mock_spline.assert_not_called()
    mock_negex.assert_not_called()
    mock_hugershoff.assert_called()
    mock_linear.assert_not_called()
    mock_horizontal.assert_not_called()


@patch('curvefit.horizontal')
@patch('curvefit.linear')
@patch('curvefit.hugershoff')
@patch('curvefit.negex')
@patch('detrend.spline')
def test_detrend_with_linear(mock_spline: Mock, mock_negex: Mock, mock_hugershoff: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
    mock_negex.side_effect = mock_negex_method
    mock_hugershoff.side_effect = mock_hugershoff_method
    mock_linear.side_effect = mock_linear_method
    mock_horizontal.side_effect = mock_horizontal_method
    
    expected_df = pd.DataFrame(data={"SeriesA": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                                    "SeriesB": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    result_df = dpl.detrend(input_df, fit="linear", plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)

    mock_spline.assert_not_called()
    mock_negex.assert_not_called()
    mock_hugershoff.assert_not_called()
    mock_linear.assert_called()
    mock_horizontal.assert_not_called()

@patch('curvefit.horizontal')
@patch('curvefit.linear')
@patch('curvefit.hugershoff')
@patch('curvefit.negex')
@patch('detrend.spline')
def test_detrend_with_horizontal(mock_spline: Mock, mock_negex: Mock, mock_hugershoff: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
    mock_negex.side_effect = mock_negex_method
    mock_hugershoff.side_effect = mock_hugershoff_method
    mock_linear.side_effect = mock_linear_method
    mock_horizontal.side_effect = mock_horizontal_method
    
    expected_df = pd.DataFrame(data={"SeriesA": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                    "SeriesB": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    result_df = dpl.detrend(input_df, fit="horizontal", plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)

    mock_spline.assert_not_called()
    mock_negex.assert_not_called()
    mock_hugershoff.assert_not_called()
    mock_linear.assert_not_called()
    mock_horizontal.assert_called()


@patch('detrend.difference')
@patch('detrend.residual')
@patch('detrend.spline')
def test_detrend_residual(mock_spline: Mock, mock_res: Mock, mock_diff: Mock):
    mock_spline.side_effect = mock_spline_method
    mock_res.side_effect = residual
    mock_diff.side_effect = difference

    expected_df = pd.DataFrame(data={"SeriesA": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                    "SeriesB": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    result_df = dpl.detrend(input_df, method='residual', plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)
    mock_res.assert_called()
    mock_diff.assert_not_called()


@patch('detrend.difference')
@patch('detrend.residual')
@patch('detrend.spline')
def test_detrend_difference(mock_spline: Mock, mock_res: Mock, mock_diff: Mock):
    mock_spline.side_effect = mock_spline_method
    mock_res.side_effect = residual
    mock_diff.side_effect = difference
    
    expected_df = pd.DataFrame(data={"SeriesA": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    "SeriesB": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    result_df = dpl.detrend(input_df, method='difference', plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)
    mock_res.assert_not_called()
    mock_diff.assert_called()


# add assertion to make sure none of the curvefit methods are called
def test_detrend_invalid_fit():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    with pytest.raises(ValueError) as errorMsg:
        dpl.detrend(input_df, fit="vertical", plot=False)
    invalid_fit_msg = "unsupported keyword for curve-fit type. See documentation for more info."
    assert invalid_fit_msg == str(errorMsg.value)


def test_detrend_invalid_method():
    pass