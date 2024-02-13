import dplpy as dpl
import pandas as pd
import pytest
from unittest.mock import patch, Mock

def test_xdate_invalid_input():
    
    with pytest.raises(TypeError) as errorMsg:
        dpl.xdate("input_df")
    
    expected_errMsg = "Expected dataframe input, got <class 'str'> instead."

    assert expected_errMsg == str(errorMsg.value)


def mock_detrend_func(data, fit="spline", plot=False):
    if fit == "horizontal":
        return data * 0.5
    else:
        return data * 2
    
def mock_ar_func_func(data, max_lag):
    return 2 * data

def mock_chron_func(data, plot=True):
    if plot:
        return None
    samp_dep = data.count(axis='columns')
    mean_rwis = data.mean(axis=1)

    out_df = pd.DataFrame(index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                            name="Year"))
    out_df["Sample Depth"] = samp_dep
    out_df["Mean RWI"] = mean_rwis

    return out_df

def mock_ppf_func(alpha=0.01, n=50, type=''):
    return alpha * 10

def mock_corr_func(data, type):
    if type == "Spearman":
        return data.mean().mean()
    elif type == "Pearson":
        return data.mean().mean() - 0.1


# mock detrend, mock_chron, mock ar_func_series, 
@patch('xdate.correlate')
@patch('scipy.stats.t.ppf')
@patch('xdate.chron')
@patch('xdate.ar_func_series')
@patch('xdate.detrend')
def test_xdate_spearman_corr(mock_detrend: Mock, mock_ar_func: Mock, mock_chron: Mock,
                             mock_ppf: Mock, mock_corr: Mock):
    mock_detrend.side_effect = mock_detrend_func
    mock_ar_func.side_effect = mock_ar_func_func
    mock_chron.side_effect = mock_chron_func
    mock_ppf.side_effect = mock_ppf_func
    mock_corr.side_effect = mock_corr_func

    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], 
                                  "SeriesC": [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]},
                            index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], name="Year"))
    
    expected = pd.DataFrame(data={"SeriesA": [0.275, 0.475, 0.675, 0.875, 1.075, 1.275], 
                                  "SeriesB": [0.300, 0.500, 0.700, 0.900, 1.100, 1.300],
                                  "SeriesC": [0.325, 0.525, 0.725, 0.925, 1.125, 1.325]},
                            index=pd.Index(data=['1-2', '2-3', '3-4', '4-5', '5-6', '6-7']))
    
    res = dpl.xdate(input_df, corr="Spearman", slide_period=2, bin_floor=0)
    
    pd.testing.assert_frame_equal(expected, res)

    mock_detrend.assert_called()
    mock_ar_func.assert_called()
    mock_chron.assert_called()
    mock_ppf.assert_called()
    mock_corr.assert_called()


@patch('xdate.correlate')
@patch('scipy.stats.t.ppf')
@patch('xdate.chron')
@patch('xdate.ar_func_series')
@patch('xdate.detrend')
def test_xdate_pearson_corr(mock_detrend: Mock, mock_ar_func: Mock, mock_chron: Mock,
                             mock_ppf: Mock, mock_corr: Mock):
    mock_detrend.side_effect = mock_detrend_func
    mock_ar_func.side_effect = mock_ar_func_func
    mock_chron.side_effect = mock_chron_func
    mock_ppf.side_effect = mock_ppf_func
    mock_corr.side_effect = mock_corr_func

    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], 
                                  "SeriesC": [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]},
                            index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], name="Year"))
    
    expected = pd.DataFrame(data={"SeriesA": [0.175, 0.375, 0.575, 0.775, 0.975, 1.175], 
                                  "SeriesB": [0.200, 0.400, 0.600, 0.800, 1.000, 1.200],
                                  "SeriesC": [0.225, 0.425, 0.625, 0.825, 1.025, 1.225]},
                            index=pd.Index(data=['1-2', '2-3', '3-4', '4-5', '5-6', '6-7']))
    
    res = dpl.xdate(input_df, corr="Pearson", slide_period=2, bin_floor=0)
    
    pd.testing.assert_frame_equal(expected, res)

    mock_detrend.assert_called()
    mock_ar_func.assert_called()
    mock_chron.assert_called()
    mock_ppf.assert_called()
    mock_corr.assert_called()


@patch('xdate.correlate')
@patch('scipy.stats.t.ppf')
@patch('xdate.chron')
@patch('xdate.ar_func_series')
@patch('xdate.detrend')
def test_xdate_no_prewhiten(mock_detrend: Mock, mock_ar_func: Mock, mock_chron: Mock,
                             mock_ppf: Mock, mock_corr: Mock):
    mock_detrend.side_effect = mock_detrend_func
    mock_ar_func.side_effect = mock_ar_func_func
    mock_chron.side_effect = mock_chron_func
    mock_ppf.side_effect = mock_ppf_func
    mock_corr.side_effect = mock_corr_func

    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], 
                                  "SeriesC": [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]},
                            index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], name="Year"))
    
    expected = pd.DataFrame(data={"SeriesA": [0.1375, 0.2375, 0.3375, 0.4375, 0.5375, 0.6375], 
                                  "SeriesB": [0.1500, 0.2500, 0.3500, 0.4500, 0.5500, 0.6500],
                                  "SeriesC": [0.1625, 0.2625, 0.3625, 0.4625, 0.5625, 0.6625]},
                            index=pd.Index(data=['1-2', '2-3', '3-4', '4-5', '5-6', '6-7']))
    
    res = dpl.xdate(input_df, prewhiten=False, corr="Spearman", slide_period=2, bin_floor=0)
    
    pd.testing.assert_frame_equal(expected, res)

    mock_detrend.assert_called()
    mock_ar_func.assert_not_called()
    mock_chron.assert_called()
    mock_ppf.assert_called()
    mock_corr.assert_called()
