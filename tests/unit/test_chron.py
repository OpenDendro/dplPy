import pandas as pd
import dplpy as dpl
import pytest
from unittest.mock import patch, Mock

def mock_tbrm_out(inp):
    return sum(inp)

def mock_ar_func_out(inp_series, max_lag=10, aic=True, method="yw", first_aic_min=False):
    inp_series += 0.01
    return inp_series

import importlib
_m_chron = importlib.import_module("dplpy.chron")

def test_chron_simple_means():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))

    expected_df = pd.DataFrame(data={"std": [0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55],
                                    "samp_depth": [2, 2, 2, 2, 2, 2, 2, 2]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    result_df = dpl.chron(input_df, biweight=False, prewhiten=False, plot=False)
    
    pd.testing.assert_frame_equal(expected_df, result_df)
    

def test_wrong_input():
    with pytest.raises(TypeError) as errorMsg:
        dpl.chron("string")
    
    assert "Expected dataframe input, got <class 'str'> instead." == str(errorMsg.value)

@patch.object(_m_chron, 'tbrm')
@patch.object(_m_chron, 'ar_func')
def test_chron_biweight_means(mock_ar_func: Mock, mock_tbrm: Mock):
    mock_tbrm.side_effect = mock_tbrm_out
    mock_ar_func.side_effect = mock_ar_func_out

    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    expected_df = pd.DataFrame(data={"std": [0.3, 0.7, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1],
                                     "samp_depth": [2, 2, 2, 2, 2, 2, 2, 2]},
                                     index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    result_df = dpl.chron(input_df, biweight=True, prewhiten=False, plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)
    mock_tbrm.assert_called()
    mock_ar_func.assert_not_called()
    

@patch.object(_m_chron, 'tbrm')
@patch.object(_m_chron, 'ar_func')
def test_chron_prewhiten(mock_ar_func: Mock, mock_tbrm: Mock):
    mock_tbrm.side_effect = mock_tbrm_out
    mock_ar_func.side_effect = mock_ar_func_out

    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    
    expected_df = pd.DataFrame(data={"std": [0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55],
                                     "res": [0.16, 0.36, 0.56, 0.76, 0.96, 1.16, 1.36, 1.56],
                                     "samp_depth": [2, 2, 2, 2, 2, 2, 2, 2]},
                                     index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    result_df = dpl.chron(input_df, biweight=False, prewhiten=True, plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)
    mock_tbrm.assert_not_called()
    mock_ar_func.assert_called()

def test_chron_plot():
    # headless smoke test: chron(plot=True) draws a figure without error
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                            index=pd.Index(data=[1, 2, 3, 4], name="Year"))
    plt.close("all")
    dpl.chron(input_df, prewhiten=False, plot=True)
    assert len(plt.get_fignums()) >= 1
    plt.close("all")


# --- stabilize= convenience flag -------------------------------------------
# see dev/variance_stabilization_design_2026-09-10.md
import warnings

import numpy as np


def _staggered_rwi(n_years=200, n_series=12, seed=0):
    """years x series RWI-like frame with staggered starts (depth rises)."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, n_years)
    years = np.arange(1, n_years + 1)
    cols = {}
    for s in range(n_series):
        start = s * (n_years // (2 * n_series))
        v = np.full(n_years, np.nan)
        noise = rng.normal(0, 1, n_years)
        v[start:] = 1.0 + 0.15 * (common[start:] + noise[start:])
        cols["s%02d" % s] = v
    return pd.DataFrame(cols, index=years)


def test_chron_stabilize_none_is_default_unchanged():
    rwi = _staggered_rwi()
    base = dpl.chron(rwi, prewhiten=True, plot=False)
    assert "std_vsc" not in base.columns and "res_vsc" not in base.columns
    assert list(base.columns) == ["std", "res", "samp_depth"]


def test_chron_stabilize_adds_vsc_columns():
    rwi = _staggered_rwi()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dpl.chron(rwi, prewhiten=True, plot=False, stabilize="both")
    # vsc columns sit beside their base chronology; base columns unchanged
    assert list(out.columns) == ["std", "std_vsc", "res", "res_vsc", "samp_depth"]
    base = dpl.chron(rwi, prewhiten=True, plot=False)
    assert np.allclose(out["std"], base["std"], equal_nan=True)
    assert np.allclose(out["res"], base["res"], equal_nan=True)
    assert np.isfinite(out["std_vsc"]).any() and np.isfinite(out["res_vsc"]).any()


def test_chron_stabilize_no_res_vsc_without_prewhiten():
    rwi = _staggered_rwi()
    out = dpl.chron(rwi, prewhiten=False, plot=False, stabilize="rbar")
    assert list(out.columns) == ["std", "std_vsc", "samp_depth"]


def test_chron_std_vsc_matches_direct_stabilize_chron():
    # the flag must reproduce a direct stabilize_chron call on the std column,
    # using the standard (detrended) matrix's rbar
    rwi = _staggered_rwi()
    out = dpl.chron(rwi, plot=False, stabilize="both")
    ref = dpl.stabilize_chron(dpl.chron(rwi, plot=False)["std"], "both", rwi=rwi)["vsc"]
    assert np.allclose(out["std_vsc"], ref, equal_nan=True)


def test_chron_stabilize_kwargs_forwarded():
    # a passthrough kwarg (constant rbar) changes the result vs the running default
    rwi = _staggered_rwi()
    running = dpl.chron(rwi, plot=False, stabilize="rbar")["std_vsc"].to_numpy()
    const = dpl.chron(rwi, plot=False, stabilize="rbar",
                      stabilize_kwargs={"rbar_mode": "constant"})["std_vsc"].to_numpy()
    ok = np.isfinite(running) & np.isfinite(const)
    assert not np.allclose(running[ok], const[ok])


def test_chron_stabilize_kwargs_rejects_managed_keys():
    rwi = _staggered_rwi()
    with pytest.raises(ValueError):
        dpl.chron(rwi, plot=False, stabilize="rbar", stabilize_kwargs={"rwi": rwi})


def test_chron_stabilize_kwargs_without_method_raises():
    rwi = _staggered_rwi()
    with pytest.raises(ValueError):
        dpl.chron(rwi, plot=False, stabilize_kwargs={"rbar_mode": "constant"})


def test_chron_invalid_stabilize_method():
    rwi = _staggered_rwi()
    with pytest.raises(ValueError):
        dpl.chron(rwi, plot=False, stabilize="nope")