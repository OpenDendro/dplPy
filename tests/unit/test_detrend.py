import warnings

import dplpy as dpl
import pandas as pd
import pytest
from unittest.mock import patch, Mock

def mock_spline_method(x, inp_arr, period, f=0.5):
    return inp_arr

def mock_negex_method(x, inp_arr):
    return inp_arr * 0.5

def mock_hugershoff_method(x, inp_arr):
    return inp_arr * 0.25

def mock_linear_method(x, inp_arr):
    return inp_arr * 4

def mock_horizontal_method(x, inp_arr):
    return inp_arr * 2

import importlib
_m_detrend = importlib.import_module("dplpy.detrend")


def _quiet_read(path):
    import io, contextlib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            return dpl.readers(path)


def _detrend_quiet(data, **kw):
    import io, contextlib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            return dpl.detrend(data, plot=False, **kw)

def test_detrend_with_invalid_input():
    with pytest.raises(TypeError) as errorMsg:
        dpl.detrend("input_df", fit="spline", plot=False)
    invalid_input_msg = "argument should be either pandas dataframe or pandas series."
    assert invalid_input_msg == str(errorMsg.value)

@patch('dplpy.curvefit.horizontal')
@patch('dplpy.curvefit.linear')
@patch('dplpy.curvefit.hugershoff')
@patch('dplpy.curvefit.negex')
@patch.object(_m_detrend, 'spline')
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

@patch('dplpy.curvefit.horizontal')
@patch('dplpy.curvefit.linear')
@patch('dplpy.curvefit.hugershoff')
@patch('dplpy.curvefit.mod_neg_exp')
@patch.object(_m_detrend, 'spline')
def test_detrend_with_modnegex(mock_spline: Mock, mock_mod_neg_exp: Mock, mock_hugershoff: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
    # mod_neg_exp is called as (x, y, pos_slope, name); return y*0.5 -> ratio 2.0
    mock_mod_neg_exp.side_effect = lambda x, y, *a, **k: y * 0.5
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
    mock_mod_neg_exp.assert_called()
    mock_hugershoff.assert_not_called()
    mock_linear.assert_not_called()
    mock_horizontal.assert_not_called()
    

@patch('dplpy.curvefit.horizontal')
@patch('dplpy.curvefit.linear')
@patch('dplpy.curvefit.mod_hugershoff')
@patch('dplpy.curvefit.mod_neg_exp')
@patch.object(_m_detrend, 'spline')
def test_detrend_with_hugershoff(mock_spline: Mock, mock_mod_neg_exp: Mock, mock_mod_hugershoff: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
    mock_mod_neg_exp.side_effect = lambda x, y, *a, **k: y * 0.5
    # mod_hugershoff is called as (x, y, pos_slope, name); return y*0.25 -> ratio 4.0
    mock_mod_hugershoff.side_effect = lambda x, y, *a, **k: y * 0.25
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
    mock_mod_neg_exp.assert_not_called()
    mock_mod_hugershoff.assert_called()
    mock_linear.assert_not_called()
    mock_horizontal.assert_not_called()


@patch('dplpy.curvefit.horizontal')
@patch('dplpy.curvefit.linear')
@patch('dplpy.curvefit.hugershoff')
@patch('dplpy.curvefit.negex')
@patch.object(_m_detrend, 'spline')
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

@patch('dplpy.curvefit.horizontal')
@patch('dplpy.curvefit.linear')
@patch('dplpy.curvefit.hugershoff')
@patch('dplpy.curvefit.negex')
@patch.object(_m_detrend, 'spline')
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


@patch.object(_m_detrend, 'spline')
def test_detrend_ratio(mock_spline: Mock):
    # method='ratio' (the default) divides the data by the fitted curve
    mock_spline.side_effect = mock_spline_method

    expected_df = pd.DataFrame(data={"SeriesA": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                    "SeriesB": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))

    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))
    result_df = dpl.detrend(input_df, method='ratio', plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)
    # 'division' is an accepted synonym, and 'ratio' is the default
    div = dpl.detrend(input_df, method='division', plot=False)
    default = dpl.detrend(input_df, plot=False)
    pd.testing.assert_frame_equal(expected_df, div)
    pd.testing.assert_frame_equal(expected_df, default)


@patch.object(_m_detrend, 'spline')
def test_detrend_residual_deprecated_alias(mock_spline: Mock):
    # 'residual' is a deprecated alias for 'ratio' (division): same numbers, but
    # it must warn, since in dplPy 'residual' now names only the AR chronology.
    mock_spline.side_effect = mock_spline_method
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]},
                            index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], name="Year"))
    with pytest.warns(FutureWarning):
        aliased = dpl.detrend(input_df, method='residual', plot=False)
    ratio = dpl.detrend(input_df, method='ratio', plot=False)
    pd.testing.assert_frame_equal(ratio, aliased)


@patch.object(_m_detrend, 'spline')
def test_detrend_difference(mock_spline: Mock):
    mock_spline.side_effect = mock_spline_method
    
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


# add assertion to make sure none of the curvefit methods are called
def test_detrend_invalid_fit():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))
    with pytest.raises(ValueError) as errorMsg:
        dpl.detrend(input_df, fit="vertical", plot=False)
    msg = str(errorMsg.value)
    assert "unsupported curve-fit type" in msg
    assert "Spline" in msg and "ModNegExp" in msg          # lists the options


def test_detrend_invalid_method():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]},
                            index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], name="Year"))
    with pytest.raises(ValueError) as errorMsg:
        dpl.detrend(input_df, method="bogus", plot=False)
    assert "unsupported detrending method" in str(errorMsg.value)


def test_normalize_fit_canonical_and_aliases():
    from dplpy.detrend import _normalize_fit
    # dplR canonical names round-trip
    assert _normalize_fit("Spline") == "Spline"
    assert _normalize_fit("ModNegExp") == "ModNegExp"
    assert _normalize_fit("ModHugershoff") == "ModHugershoff"
    assert _normalize_fit("Mean") == "Mean"
    assert _normalize_fit("AgeDepSpline") == "AgeDepSpline"
    assert _normalize_fit("Linear") == "Linear"
    # case-insensitive
    assert _normalize_fit("spline") == "Spline"
    assert _normalize_fit("MODNEGEXP") == "ModNegExp"
    # legacy dplPy spellings map to the canonical dplR names
    assert _normalize_fit("ModNegEx") == "ModNegExp"
    assert _normalize_fit("Hugershoff") == "ModHugershoff"
    assert _normalize_fit("horizontal") == "Mean"


@patch.object(_m_detrend, 'spline')
def test_detrend_fit_names_case_insensitive(mock_spline: Mock):
    # 'spline', 'Spline', 'SpLiNe' all reach the spline fitter and give the same result
    mock_spline.side_effect = mock_spline_method
    df = pd.DataFrame({"A": [0.1, 0.3, 0.5, 0.7]},
                      index=pd.Index([1, 2, 3, 4], name="Year"))
    a = dpl.detrend(df, fit="spline", plot=False)
    b = dpl.detrend(df, fit="Spline", plot=False)
    c = dpl.detrend(df, fit="SpLiNe", plot=False)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


def test_mod_neg_exp_fits_decaying_series_no_fallback():
    # a clean decaying series is fit by the negative exponential itself -- no
    # fallback warning, and the curve is the fitted exponential (not the mean).
    import numpy as np
    from dplpy import curvefit as cf
    x = np.arange(1, 31)
    y = 5.0 * np.exp(-0.1 * np.arange(30)) + 1.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yi = cf.mod_neg_exp(x, y, pos_slope=False, name="decay")
    fallback = [x for x in w if issubclass(x.category, UserWarning)
                and ("mean" in str(x.message) or "linear" in str(x.message))]
    assert not fallback                                        # negexp was used
    assert np.all(yi > 0) and yi[0] > yi[-1]
    assert not np.allclose(yi, yi.mean())                      # not the flat fallback


def test_mod_neg_exp_fallback_chain_when_fit_fails():
    # Force ONLY the neg-exp fit to fail (as dplR's nls can) -- the linear
    # fallback uses curve_fit too, so leave that working -- then check the cascade:
    #  rising + pos_slope=False -> series mean; rising + pos_slope=True -> line;
    #  decreasing + pos_slope=False -> line (slope <= 0 is accepted).
    import numpy as np
    import scipy.optimize as so
    from dplpy import curvefit as cf
    from dplpy.curvefit import negex_function
    real_curve_fit = so.curve_fit

    def only_negex_fails(f, *a, **k):
        if f is negex_function:
            raise RuntimeError("no fit")
        return real_curve_fit(f, *a, **k)

    x = np.arange(1, 31)
    rising = 0.5 * np.arange(1, 31) + 2.0
    falling = -0.3 * np.arange(1, 31) + 20.0
    with patch("dplpy.curvefit.curve_fit", side_effect=only_negex_fails):
        with pytest.warns(UserWarning, match="series mean"):
            m = cf.mod_neg_exp(x, rising, pos_slope=False, name="r")
        assert np.allclose(m, np.mean(rising))                    # -> mean

        with pytest.warns(UserWarning, match="linear fit"):
            up = cf.mod_neg_exp(x, rising, pos_slope=True, name="r")
        assert up[-1] > up[0] and np.all(up > 0)                  # -> rising line

        with pytest.warns(UserWarning, match="linear fit"):
            dn = cf.mod_neg_exp(x, falling, pos_slope=False, name="f")
        assert dn[-1] < dn[0] and np.all(dn > 0)                  # -> falling line


def test_detrend_spline_f_exposed():
    # D5: f defaults to 0.5 (unchanged) and actually tunes the spline when set.
    import io, contextlib
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    s = data["CAM011"]
    default = _detrend_quiet(s)
    same = _detrend_quiet(s, f=0.5)
    diff = _detrend_quiet(s, f=0.1)
    assert np.allclose(default.dropna(), same.dropna())
    assert not np.allclose(default.dropna(), diff.dropna())


def test_detrend_verbose_prints_per_series():
    # D7: verbose prints one line per series naming the curve and method.
    import io, contextlib
    data = _quiet_read("tests/data/csv/ca533.csv")[["CAM011", "CAM021"]]
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buf):
            dpl.detrend(data, plot=False, verbose=True)
    out = buf.getvalue()
    assert "CAM011" in out and "CAM021" in out
    assert "fit=Spline" in out and "method=ratio" in out


def test_mod_hugershoff_fallback_chain_when_fit_fails():
    # Same fallback structure as ModNegExp: force ONLY the Hugershoff nls to fail
    # (its linear fallback uses curve_fit too), then check the cascade.
    import numpy as np
    import scipy.optimize as so
    from dplpy import curvefit as cf
    from dplpy.curvefit import hugershoff_function
    real_curve_fit = so.curve_fit

    def only_hug_fails(f, *a, **k):
        if f is hugershoff_function:
            raise RuntimeError("no fit")
        return real_curve_fit(f, *a, **k)

    x = np.arange(1, 31)
    rising = 0.5 * np.arange(1, 31) + 2.0
    falling = -0.3 * np.arange(1, 31) + 20.0
    with patch("dplpy.curvefit.curve_fit", side_effect=only_hug_fails):
        with pytest.warns(UserWarning, match="series mean"):
            m = cf.mod_hugershoff(x, rising, pos_slope=False, name="r")
        assert np.allclose(m, np.mean(rising))
        with pytest.warns(UserWarning, match="linear fit"):
            up = cf.mod_hugershoff(x, rising, pos_slope=True, name="r")
        assert up[-1] > up[0] and np.all(up > 0)
        with pytest.warns(UserWarning, match="linear fit"):
            dn = cf.mod_hugershoff(x, falling, pos_slope=False, name="f")
        assert dn[-1] < dn[0] and np.all(dn > 0)


def test_detrend_hugershoff_real_series_positive():
    # end-to-end: ModHugershoff on a real series returns a finite, all-positive RWI
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    rwi = _detrend_quiet(data["CAM011"], fit="ModHugershoff")
    v = rwi.dropna().to_numpy()
    assert np.all(np.isfinite(v)) and np.all(v > 0)


def test_detrend_method_given_curve_name_is_guarded():
    # D12: passing a curve name to method= must raise a helpful error pointing to fit=
    df = pd.DataFrame({"A": [0.1, 0.3, 0.5, 0.7]},
                      index=pd.Index([1, 2, 3, 4], name="Year"))
    with pytest.raises(ValueError) as e:
        dpl.detrend(df, method="Spline", plot=False)
    msg = str(e.value)
    assert "looks like a curve type" in msg
    assert "fit=" in msg