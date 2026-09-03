import dplpy as dpl
import pytest
import pandas as pd
from unittest.mock import patch, Mock

# Data being read:
# SeriesA  1   10 30  50  70  90  110 130 150 999
# SeriesB  1   20 40  60  80  100 120 140 160 999

def mock_readers_output(file_name):
    if file_name == "valid_file":
        return pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))


# Need to mock output of readers

import importlib
_m_stats = importlib.import_module("dplpy.stats")
_m_readers = importlib.import_module("dplpy.readers")

@patch.object(_m_readers, 'readers')
def test_stats_with_inp_string(mock_readers: Mock):
    mock_readers.side_effect = mock_readers_output
    
    expected_df = pd.DataFrame(data={"series": ["SeriesA", "SeriesB"],
                                     "first": [1, 1],
                                     "last": [8, 8],
                                     "year": [8, 8],
                                     "mean": [0.8, 0.9],
                                     "median": [0.8, 0.9],
                                     "stdev": [0.49, 0.49],
                                     "skew": [0.0, 0.0],
                                     "kurtosis": [-1.651, -1.651],
                                     "gini": [0.328, 0.292],
                                     "ar1": [0.625, 0.625]
                                     },
                                index=[1, 2])
    results = dpl.stats("valid_file")

    mock_readers.assert_called_once_with("valid_file")
    pd.testing.assert_frame_equal(results, expected_df)


@patch.object(_m_readers, 'readers')
def test_stats_with_inp_df(mock_readers: Mock):
    mock_readers.side_effect = mock_readers_output
    
    expected_df = pd.DataFrame(data={"series": ["SeriesA", "SeriesB"],
                                     "first": [1, 1],
                                     "last": [8, 8],
                                     "year": [8, 8],
                                     "mean": [0.8, 0.9],
                                     "median": [0.8, 0.9],
                                     "stdev": [0.49, 0.49],
                                     "skew": [0.0, 0.0],
                                     "kurtosis": [-1.651, -1.651],
                                     "gini": [0.328, 0.292],
                                     "ar1": [0.625, 0.625]
                                     },
                                index=[1, 2])
    
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))

    results = dpl.stats(input_df)
    mock_readers.assert_not_called()
    pd.testing.assert_frame_equal(results, expected_df)


def test_stats_median_3dp_and_kurtosis_present():
    # median must round to 3 dp (a 2-dp round would turn 0.285 into 0.28/0.29),
    # and kurtosis is now reported (matching dplR rwl.stats), between skew and gini.
    import warnings
    df = pd.DataFrame({"S": [0.10, 0.20, 0.28, 0.29, 0.40, 0.50]},
                      index=pd.Index([1, 2, 3, 4, 5, 6], name="Year"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = dpl.stats(df)
    assert list(res.columns).index("kurtosis") == list(res.columns).index("skew") + 1
    assert list(res.columns).index("kurtosis") == list(res.columns).index("gini") - 1
    assert res["median"].iloc[0] == pytest.approx(0.285)   # 3-dp, not 0.28/0.29


def test_get_kurtosis_matches_dplR_formula():
    # excess kurtosis: n*sum(y2^4)/(sum(y2^2)^2)*(1-1/n)^2 - 3   (dplR rwl.stats kurt)
    import numpy as np
    from dplpy.stats import get_kurtosis
    x = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    y = x.to_numpy(); n = len(y); y2 = y - y.mean()
    expected = n * np.sum(y2**4) / (np.sum(y2**2)**2) * (1 - 1/n)**2 - 3
    assert get_kurtosis(x) == pytest.approx(expected)


def test_get_ar1_is_lag1_autocorrelation():
    # dplR rwl.stats' ar1 is the acf at lag 1 (biased, mean-centred), NOT an OLS
    # AR(1) slope:  r1 = sum(y2_t*y2_{t+1}) / sum(y2_t^2),  y2 = y - mean(y)
    import numpy as np
    from dplpy.stats import get_ar1
    x = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    y = x.to_numpy(); y2 = y - y.mean()
    expected = np.sum(y2[1:] * y2[:-1]) / np.sum(y2**2)
    assert get_ar1(x) == pytest.approx(expected)
    assert round(get_ar1(x), 3) == 0.625
    # matches statsmodels' acf(adjusted=False) at lag 1
    from statsmodels.tsa.stattools import acf
    assert get_ar1(x) == pytest.approx(acf(y, nlags=1, adjusted=False, fft=False)[1])
