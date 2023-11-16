import dplpy as dpl
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, Mock

def mock_ar_sel_order_method(inp_ser, max_lag, ic='aic', old_names=False):
    inp_ser_name = inp_ser.name
    param_name = inp_ser_name + ".L1"
    res = pd.Series(data=[0.5, 0.5], index=pd.Index(data=['const', param_name]))
    mock_results = Mock()
    mock_model = Mock()
    mock_fit = Mock()
    mock_fit.params = res
    mock_model.fit = Mock(return_value=mock_fit)
    mock_results.model = mock_model
    return mock_results
    

@patch('autoreg.ar_select_order')
def test_ar_func_invalid_dtype(mock_ar_sel_order: Mock):
    with pytest.raises(TypeError) as errorMsg:
        dpl.ar_func("input_df")
    expected_errMsg = "Data argument should be either pandas dataframe or pandas series."
    assert expected_errMsg == str(errorMsg.value)
    mock_ar_sel_order.assert_not_called()


@patch('autoreg.ar_select_order')
def test_ar_func_on_series(mock_ar_sel_order: Mock):
    mock_ar_sel_order.side_effect = mock_ar_sel_order_method

    data = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    actual_ser_output = dpl.ar_func(data['SeriesA'])

    expected_ser_output = pd.Series(name="SeriesA", 
                                    data=[0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15],
                                    index=pd.Index(name="Year", data=[2, 3, 4, 5, 6, 7, 8]))
    pd.testing.assert_series_equal(expected_ser_output, actual_ser_output)
    mock_ar_sel_order.assert_called_once()
    

@patch('autoreg.ar_select_order')
def test_ar_func_on_df(mock_ar_sel_order: Mock):
    mock_ar_sel_order.side_effect = mock_ar_sel_order_method

    data = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    actual_ser_output = dpl.ar_func(data)

    expected_ser_output = pd.DataFrame(data={"SeriesA":[np.nan, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15],
                                             "SeriesB":[np.nan, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
                                    index=pd.Index(name="Year", data=[1, 2, 3, 4, 5, 6, 7, 8]))
    pd.testing.assert_frame_equal(expected_ser_output, actual_ser_output)
    mock_ar_sel_order.assert_called()


@patch('autoreg.ar_select_order')
def test_autoreg_invalid_input(mock_ar_sel_order: Mock):
    mock_ar_sel_order.side_effect = mock_ar_sel_order_method
    with pytest.raises(TypeError) as errorMsg:
        dpl.autoreg("input_df")
    expected_errMsg = "Data argument should be pandas series. Received <class 'str'> instead."
    assert expected_errMsg == str(errorMsg.value)
    mock_ar_sel_order.assert_not_called()


@patch('autoreg.ar_select_order')
def test_autoreg_valid_input(mock_ar_sel_order: Mock):
    mock_ar_sel_order.side_effect = mock_ar_sel_order_method
    data = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    actual_res = dpl.autoreg(data['SeriesA'])
    expected_res = pd.Series(data=[0.5, 0.5], 
                             index=pd.Index(data=['const', 'SeriesA.L1']))
    pd.testing.assert_series_equal(expected_res, actual_res)
    mock_ar_sel_order.assert_called_once()