import dplpy as dpl
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, Mock

def mock_ar_sel_order_method(inp_ser, max_lag, ic='aic'):
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
    

import importlib
_m_autoreg = importlib.import_module("dplpy.autoreg")

@patch.object(_m_autoreg, 'ar_select_order')
def test_ar_func_invalid_dtype(mock_ar_sel_order: Mock):
    with pytest.raises(TypeError) as errorMsg:
        dpl.ar_func("input_df")
    expected_errMsg = "Data argument should be either pandas dataframe or pandas series."
    assert expected_errMsg == str(errorMsg.value)
    mock_ar_sel_order.assert_not_called()


@patch.object(_m_autoreg, 'ar_select_order')
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
    

@patch.object(_m_autoreg, 'ar_select_order')
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


@patch.object(_m_autoreg, 'ar_select_order')
def test_autoreg_invalid_input(mock_ar_sel_order: Mock):
    mock_ar_sel_order.side_effect = mock_ar_sel_order_method
    with pytest.raises(TypeError) as errorMsg:
        dpl.autoreg("input_df")
    expected_errMsg = "Data argument should be pandas series. Received <class 'str'> instead."
    assert expected_errMsg == str(errorMsg.value)
    mock_ar_sel_order.assert_not_called()


@patch.object(_m_autoreg, 'ar_select_order')
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

# --- Yule-Walker AR path (dplR ar() default estimator) -----------------------
def test_autoreg_yw_matches_statsmodels_yule_walker():
    # autoreg(method="yw") must reproduce statsmodels' yule_walker coefficients
    # (which are verified elsewhere to match R's ar(method="yule-walker") to
    # ~1e-15). The returned layout is [intercept, phi_1..phi_p] with the AR
    # coefficients following the intercept, so compare params[1:] to yule_walker.
    import numpy as np
    from statsmodels.regression.linear_model import yule_walker
    rng = np.random.default_rng(0)
    # a series with real autocorrelation so a nonzero order is selected
    e = rng.standard_normal(400)
    x = np.zeros(400)
    for t in range(2, 400):
        x[t] = 0.6 * x[t - 1] - 0.2 * x[t - 2] + e[t]
    s = pd.Series(x + 5.0)

    params = dpl.autoreg(s, max_lag=10, aic=False, method="yw")   # fixed AR(10)
    phi = np.asarray(params[1:], dtype=float)
    assert len(phi) == 10
    xd = (s - s.mean()).to_numpy()
    rho, _ = yule_walker(xd, order=10, method="mle", demean=False)
    assert np.max(np.abs(phi - rho)) < 1e-12
    # intercept encodes the mean: intercept == mean * (1 - sum(phi))
    assert abs(params[0] - s.mean() * (1 - phi.sum())) < 1e-9


def test_autoreg_yw_differs_from_ols_and_bad_method_raises():
    import numpy as np
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(300)) * 0.0 + rng.standard_normal(300)  # AR-ish noise
    s = pd.Series(x)
    yw = np.asarray(dpl.autoreg(s, max_lag=5, aic=False, method="yw")[1:], dtype=float)
    ols = np.asarray(dpl.autoreg(s, max_lag=5, aic=False, method="ols")[1:], dtype=float)
    # same length, but the two estimators are not identical
    assert len(yw) == len(ols) == 5
    assert np.max(np.abs(yw - ols)) > 1e-6
    with pytest.raises(ValueError):
        dpl.autoreg(s, method="bogus")


# --- first-local-min AR order selection (ARSTAN's firstAICmin rule) ----------
def test_first_aic_min_rule():
    from dplpy.autoreg import _first_aic_min
    # dips to a local min at order 2, rises, then a lower GLOBAL min at order 5:
    # the first-local rule must take order 2 (parsimony), not the global 5.
    assert _first_aic_min([10.0, 6.0, 4.0, 5.0, 4.5, 3.0]) == 2
    # AIC already rising from order 0 -> order 0
    assert _first_aic_min([1.0, 2.0, 3.0]) == 0
    # monotone decreasing (never turns up) -> fall back to the highest order
    assert _first_aic_min([5.0, 4.0, 3.0, 2.0]) == 3


def test_autoreg_first_aic_min_never_exceeds_global():
    # on a crafted series whose global AIC min is a late, shallow one, the
    # first-local rule selects an order no higher than the global rule.
    import numpy as np
    rng = np.random.default_rng(7)
    e = rng.standard_normal(600)
    x = np.zeros(600)
    for t in range(3, 600):
        x[t] = 0.5 * x[t - 1] + 0.2 * x[t - 3] + e[t]
    s = pd.Series(x)
    p_global = len(dpl.autoreg(s, max_lag=15, aic=True, method="yw")) - 1
    p_first = len(dpl.autoreg(s, max_lag=15, aic=True, method="yw",
                              first_aic_min=True)) - 1
    assert p_first <= p_global
