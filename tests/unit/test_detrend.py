import warnings

import dplpy as dpl
import pandas as pd
import pytest
from unittest.mock import patch, Mock

def mock_spline_method(x, inp_arr, period, f=0.5):
    return inp_arr

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
@patch.object(_m_detrend, 'spline')
def test_detrend_with_spline(mock_spline: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
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
    mock_linear.assert_not_called()
    mock_horizontal.assert_not_called()

@patch('dplpy.curvefit.horizontal')
@patch('dplpy.curvefit.linear')
@patch('dplpy.curvefit.mod_neg_exp')
@patch.object(_m_detrend, 'spline')
def test_detrend_with_modnegex(mock_spline: Mock, mock_mod_neg_exp: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
    # mod_neg_exp is called as (x, y, pos_slope, name); return y*0.5 -> ratio 2.0
    mock_mod_neg_exp.side_effect = lambda x, y, *a, **k: y * 0.5
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
    result_df = dpl.detrend(input_df, fit="ModHugershoff", plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)

    mock_spline.assert_not_called()
    mock_mod_neg_exp.assert_not_called()
    mock_mod_hugershoff.assert_called()
    mock_linear.assert_not_called()
    mock_horizontal.assert_not_called()


@patch('dplpy.curvefit.horizontal')
@patch('dplpy.curvefit.linear')
@patch.object(_m_detrend, 'spline')
def test_detrend_with_linear(mock_spline: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
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
    mock_linear.assert_called()
    mock_horizontal.assert_not_called()

@patch('dplpy.curvefit.horizontal')
@patch('dplpy.curvefit.linear')
@patch.object(_m_detrend, 'spline')
def test_detrend_with_horizontal(mock_spline: Mock, mock_linear: Mock, mock_horizontal: Mock):
    mock_spline.side_effect = mock_spline_method
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
    assert _normalize_fit("LinearAny") == "LinearAny"
    assert _normalize_fit("LinearNegative") == "LinearNegative"
    # case-insensitive
    assert _normalize_fit("spline") == "Spline"
    assert _normalize_fit("MODNEGEXP") == "ModNegExp"
    # legacy dplPy spellings map to the canonical dplR names
    assert _normalize_fit("ModNegEx") == "ModNegExp"
    assert _normalize_fit("horizontal") == "Mean"
    # 'Hugershoff' is Cook's ARSTAN closed form; 'ModHugershoff' is dplR's nls
    assert _normalize_fit("Hugershoff") == "Hugershoff"
    assert _normalize_fit("ModHugershoff") == "ModHugershoff"
    # ARSTAN additions: 'NegExp' (deterministic neg-exp, ARSTAN's `curve`) sits
    # beside dplR's nls 'ModNegExp'; 'GeneralExp' is ARSTAN's gexp (Hugershoff b=1)
    assert _normalize_fit("NegExp") == "NegExp"
    assert _normalize_fit("negexp") == "NegExp"
    assert _normalize_fit("negex") == "NegExp"
    assert _normalize_fit("GeneralExp") == "GeneralExp"
    assert _normalize_fit("generalexp") == "GeneralExp"
    assert _normalize_fit("genexp") == "GeneralExp"
    # 'linear' is a back-compatible alias of ARSTAN option 4 ('LinearAny', any
    # slope); 'LinearNegative' is ARSTAN option 5 (slope constrained non-positive)
    assert _normalize_fit("linear") == "LinearAny"
    assert _normalize_fit("linearany") == "LinearAny"
    assert _normalize_fit("linearnegative") == "LinearNegative"
    assert _normalize_fit("linearneg") == "LinearNegative"


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


def test_detrend_return_info_series_structure():
    # D6: return_info gives a dict mirroring dplR's return.info, and rwi is
    # identical to the plain detrend output.
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    info = _detrend_quiet(data["CAM011"], return_info=True)
    assert set(info) == {"rwi", "curves", "model_info", "data_info", "dirty_dog"}
    plain = _detrend_quiet(data["CAM011"])
    assert np.allclose(info["rwi"].dropna(), plain.dropna())
    assert info["model_info"]["method"] == "Spline"
    assert info["model_info"]["nyrs"] > 0 and info["model_info"]["f"] == 0.5
    # CAM011 has two zero rings (1753, 1782) reported from the raw input
    assert info["data_info"]["n_zeros"] == 2
    assert 1753 in info["data_info"]["zero_years"]
    assert info["dirty_dog"] is False


def test_detrend_return_info_modnegexp_coefs():
    data = _quiet_read("tests/data/csv/ca533.csv")
    info = _detrend_quiet(data["CAM011"], fit="ModNegExp", return_info=True)
    mi = info["model_info"]
    assert mi["method"] == "NegativeExponential"
    assert set(mi["coefs"]) == {"a", "b", "k"}


def test_detrend_return_info_dataframe():
    data = _quiet_read("tests/data/csv/ca533.csv")[["CAM011", "CAM021"]]
    info = _detrend_quiet(data, return_info=True)
    assert info["rwi"].shape == data.shape
    assert info["curves"].shape == data.shape
    assert set(info["model_info"]) == {"CAM011", "CAM021"}
    assert set(info["data_info"]) == {"CAM011", "CAM021"}
    assert isinstance(info["dirty_dog"], bool)


def test_detrend_return_info_dirty_dog_on_mean_fallback():
    # when a requested curve falls back to the series mean, dirty_dog is True
    import numpy as np
    df = pd.DataFrame({"A": [0.5, 0.6, 0.7, 0.8, 0.9]},
                      index=pd.Index([1, 2, 3, 4, 5], name="Year"))
    fell_to_mean = (np.full(5, 0.7), {"method": "Mean", "mean": 0.7})
    with patch("dplpy.curvefit.mod_neg_exp", return_value=fell_to_mean):
        info = _detrend_quiet(df, fit="ModNegExp", return_info=True)
    assert info["model_info"]["A"]["method"] == "Mean"
    assert info["dirty_dog"] is True


def test_detrend_fit_list_on_series_returns_method_columns():
    # D9: a list of fits on a Series -> DataFrame with one column per method,
    # each equal to the corresponding single-method detrend.
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    out = _detrend_quiet(data["CAM011"], fit=["Spline", "ModNegExp", "Mean"])
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["Spline", "ModNegExp", "Mean"]
    for m in ["Spline", "ModNegExp", "Mean"]:
        one = _detrend_quiet(data["CAM011"], fit=m)
        assert np.allclose(out[m].dropna(), one.dropna())


def test_detrend_fit_list_on_dataframe_returns_dict():
    data = _quiet_read("tests/data/csv/ca533.csv")[["CAM011", "CAM021"]]
    out = _detrend_quiet(data, fit=["Spline", "Mean"])
    assert isinstance(out, dict)
    assert set(out) == {"CAM011", "CAM021"}
    assert list(out["CAM011"].columns) == ["Spline", "Mean"]


def test_detrend_fit_list_single_and_dedup():
    # a one-element list collapses to the scalar (Series) behavior; duplicates and
    # case variants are de-duplicated (order preserved, canonical names).
    data = _quiet_read("tests/data/csv/ca533.csv")
    assert isinstance(_detrend_quiet(data["CAM011"], fit=["Spline"]), pd.Series)
    dd = _detrend_quiet(data["CAM011"], fit=["spline", "Spline", "MEAN"])
    assert list(dd.columns) == ["Spline", "Mean"]


def test_detrend_fit_list_with_return_info_raises():
    data = _quiet_read("tests/data/csv/ca533.csv")
    with pytest.raises(ValueError):
        _detrend_quiet(data["CAM011"], fit=["Spline", "Mean"], return_info=True)


# --- validation against dplR 1.7.9 (detrend.series on ca533$CAM011) -----------
# Reference RWI (ratios, default) generated from dplR 1.7.9; the ModNegExp/Hugershoff
# logic is identical in the 1.8.0 source. CAM011 fits every curve (no fallback).
# Closed-form fits (Spline, Mean) match to machine precision. The nls fits
# (ModNegExp, ModHugershoff) match to ~1e-3: R's nls only converges to tol~1e-5,
# so iterative fits cannot agree more tightly than their own convergence.
# NOTE: on the finicky 4-param Hugershoff, scipy is MORE robust than R's nls and
# fits ~6 ca533 series (e.g. CAM151) where dplR falls back to a line/mean -- an
# accepted, documented difference (dplPy matches dplR wherever R's nls converges).
_DPLR_CAM011 = {
    "Spline":        [1.1963223, 1.02753817, 1.19355965, 0.81416, 0.80551108],
    "ModNegExp":     [1.07078863, 0.92366339, 1.0774591, 0.73805365, 0.73324942],
    "ModHugershoff": [1.3462947, 1.09135316, 1.22727013, 0.81924458, 0.79799274],
    "Mean":          [2.36586295, 2.02463271, 2.34311426, 1.59240775, 1.56965907],
}


@pytest.mark.parametrize("fit,atol", [
    ("Spline", 1e-6), ("Mean", 1e-8),
    ("ModNegExp", 1e-3), ("ModHugershoff", 1e-3),
])
def test_detrend_matches_dplR_ca533_cam011(fit, atol):
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    rwi = _detrend_quiet(data["CAM011"], fit=fit).dropna().to_numpy()[:5]
    assert np.allclose(rwi, _DPLR_CAM011[fit], atol=atol)


def test_modhugershoff_warns_on_ill_conditioned_fit():
    # scipy is more robust than dplR's nls and will fit some series dplR rejects.
    # When the Hugershoff fit is badly ill-conditioned we warn so the divergence
    # is never silent; a clean fit (CAM011) does not warn. model_info carries cond.
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    # CAM021 is a severely ill-conditioned Hugershoff fit (cond ~1e23)
    with pytest.warns(UserWarning, match="poorly constrained"):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            info = dpl.detrend(data["CAM021"], fit="ModHugershoff", plot=False,
                               return_info=True)
    assert info["model_info"]["cond"] > 1e12
    # CAM011 is a clean, well-conditioned Hugershoff fit -> no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        info2 = dpl.detrend(data["CAM011"], fit="ModHugershoff", plot=False,
                            return_info=True)
    assert not [x for x in w if "poorly constrained" in str(x.message)]
    assert info2["model_info"]["cond"] < 1e12


def test_detrend_hugershoff_arstan_deterministic_and_positive():
    # fit='Hugershoff' is Ed Cook's ARSTAN closed-form: always a positive curve,
    # deterministic, no fallback and no warning even on the ill-conditioned series
    # where the nls 'ModHugershoff' warns/diverges.
    import io, contextlib
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    # CAM021 is the series where the nls ModHugershoff warns; ARSTAN must not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            a = dpl.detrend(data["CAM021"], fit="Hugershoff", plot=False)
    assert not [x for x in w if "poorly constrained" in str(x.message)]
    b = _detrend_quiet(data["CAM021"], fit="Hugershoff")
    v = a.dropna().to_numpy()
    assert np.all(np.isfinite(v)) and np.all(v > 0)                # positive index
    assert np.array_equal(v, b.dropna().to_numpy())               # deterministic


def test_detrend_hugershoff_arstan_differs_from_nls_and_reports_coefs():
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    info = _detrend_quiet(data["CAM011"], fit="Hugershoff", return_info=True)
    assert info["model_info"]["method"] == "Hugershoff"
    assert set(info["model_info"]["coefs"]) == {"a", "m", "k"}
    nls = _detrend_quiet(data["CAM011"], fit="ModHugershoff")
    # the two Hugershoff estimators give genuinely different curves
    assert not np.allclose(info["rwi"].dropna(), nls.dropna(), atol=1e-2)


def test_detrend_method_given_curve_name_is_guarded():
    # D12: passing a curve name to method= must raise a helpful error pointing to fit=
    df = pd.DataFrame({"A": [0.1, 0.3, 0.5, 0.7]},
                      index=pd.Index([1, 2, 3, 4], name="Year"))
    with pytest.raises(ValueError) as e:
        dpl.detrend(df, method="Spline", plot=False)
    msg = str(e.value)
    assert "looks like a curve type" in msg
    assert "fit=" in msg


# --- ARSTAN curve-fit additions: NegExp, GeneralExp, LinearAny, LinearNegative ---

def test_negexp_deterministic_fits_decaying_series_no_fallback():
    # ARSTAN's deterministic `curve` fits a clean decay itself (no line/mean
    # fallback): a positive, monotonically declining neg-exp, not the flat mean.
    import numpy as np
    from dplpy import curvefit as cf
    x = np.arange(1, 41)
    y = 5.0 * np.exp(-0.08 * np.arange(40)) + 1.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yi, info = cf.neg_exp_arstan(x, y, name="decay", info=True)
    fallback = [m for m in w if issubclass(m.category, UserWarning)]
    assert not fallback                                        # curve was used
    assert info["method"] == "NegativeExponential"
    assert set(info["coefs"]) == {"a", "b", "k"}
    assert info["coefs"]["b"] < 0 and info["coefs"]["a"] > 0   # decaying, positive
    assert np.all(yi > 0) and yi[0] > yi[-1]
    assert not np.allclose(yi, yi.mean())                      # not the flat fallback


def test_negexp_matches_modnegexp_when_nls_converges():
    # NegExp (deterministic, ARSTAN `curve`) and ModNegExp (nls, dplR) minimise the
    # SAME data-space RSS for the SAME model y=a*exp(b*t)+k -- so when the nls
    # converges cleanly the two curves coincide (unlike the Hugershoff pair, which
    # optimise different error spaces). NegExp's distinction is robustness, not shape.
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    det = _detrend_quiet(data["CAM011"], fit="NegExp")
    nls = _detrend_quiet(data["CAM011"], fit="ModNegExp")
    d = det.dropna().to_numpy(); m = nls.dropna().to_numpy()
    assert np.all(np.isfinite(d)) and np.all(d > 0)
    assert np.corrcoef(d, m)[0, 1] > 0.999                     # essentially the same
    assert np.allclose(d, m, atol=1e-2)                        # coincide at convergence


def test_negexp_reports_coefs_via_return_info():
    data = _quiet_read("tests/data/csv/ca533.csv")
    info = _detrend_quiet(data["CAM011"], fit="NegExp", return_info=True)
    assert info["model_info"]["method"] in ("NegativeExponential", "Line", "Mean")


def test_generalexp_produces_valid_positive_curve_and_coefs():
    # ARSTAN gexp: f = a*t*exp(b*t). On a series with a juvenile rise it gives a
    # positive, finite curve and reports its two coefficients.
    import numpy as np
    from dplpy import curvefit as cf
    t = np.arange(1, 61)
    y = 2.0 * t * np.exp(-0.05 * t) + 0.5      # rise-then-decline
    yi, info = cf.general_exp(np.arange(1, 61), y, name="rise", info=True)
    assert info["method"] == "GeneralExponential"
    assert set(info["coefs"]) == {"a", "b"}
    assert np.all(yi > 0) and np.all(np.isfinite(yi))
    assert yi.argmax() > 0                                     # peaks after t=1 (a rise)


def test_generalexp_detrend_runs_on_real_series():
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    a = _detrend_quiet(data["CAM011"], fit="GeneralExp")
    v = a.dropna().to_numpy()
    assert np.all(np.isfinite(v)) and np.all(v > 0)


def test_linearany_is_the_linear_alias():
    # 'linear' is a back-compatible alias of 'LinearAny' -> identical result.
    import numpy as np
    data = _quiet_read("tests/data/csv/ca533.csv")
    a = _detrend_quiet(data["CAM011"], fit="LinearAny")
    b = _detrend_quiet(data["CAM011"], fit="linear")
    assert np.array_equal(a.dropna().to_numpy(), b.dropna().to_numpy())


def test_linearnegative_matches_linearany_on_declining_series():
    # On a declining series the best-fit slope is <= 0, so LinearNegative accepts
    # the same line LinearAny returns (no mean fallback).
    import numpy as np
    from dplpy import curvefit as cf
    x = np.arange(1, 31, dtype=float)
    y = 10.0 - 0.2 * x + 0.01 * np.sin(x)      # gently declining, all positive
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        neg = cf.linear_neg(x, y, name="decl")
    any_ = cf.linear(x, y)
    assert not [m for m in w if issubclass(m.category, UserWarning)]   # no fallback
    assert np.allclose(neg, any_)


def test_linearnegative_falls_back_to_mean_on_rising_series():
    # A positively-sloped series is rejected by LinearNegative -> series mean.
    import numpy as np
    from dplpy import curvefit as cf
    x = np.arange(1, 31, dtype=float)
    y = 1.0 + 0.2 * x                          # rising
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        neg = cf.linear_neg(x, y, name="rise")
    assert [m for m in w if issubclass(m.category, UserWarning)]       # warned
    assert np.allclose(neg, np.mean(y))                               # flat mean