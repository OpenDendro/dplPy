import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _read_quiet(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.readers(path)


# dplR powt(method='cook') per-series powers for the first few ca533 series
# (dplR 1.7.9). Cook is deterministic OLS, so dplPy reproduces these exactly.
_DPLR_COOK_CA533 = {
    "CAM011": 0.71236143893634,
    "CAM021": 0.770498860140902,
    "CAM031": 0.675286368446009,
    "CAM032": 0.837457599335083,
    "CAM041": 0.589542905376192,
}


def test_powt_cook_matches_dplR_powers():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    pt, power = dpl.powt(rwl, method="cook", return_power=True)
    for series, expected in _DPLR_COOK_CA533.items():
        assert power[series] == pytest.approx(expected, abs=1e-6)
    # transform is x**p at the series' own power
    x = rwl["CAM011"].dropna().to_numpy()
    assert np.allclose(pt["CAM011"].dropna().to_numpy(),
                       x ** power["CAM011"], atol=1e-10)


def test_powt_cook_default_method():
    # cook is the default (unlike dplR, whose default is 'universal')
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    _, p_default = dpl.powt(rwl, return_power=True)
    _, p_cook = dpl.powt(rwl, method="cook", return_power=True)
    assert (p_default == p_cook).all()


def test_powt_cook_vector():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    s = rwl["CAM011"]
    out, p = dpl.powt(s, method="cook", return_power=True)
    assert isinstance(out, pd.Series)
    assert p == pytest.approx(0.7123613, abs=1e-6)


def test_powt_rescale_restores_moments():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    rescaled = dpl.powt(rwl, method="cook", rescale=True)
    col = "CAM011"
    orig = rwl[col].to_numpy()
    got = rescaled[col].to_numpy()
    assert np.nanmean(got) == pytest.approx(np.nanmean(orig), abs=1e-8)
    assert np.nanstd(got, ddof=1) == pytest.approx(np.nanstd(orig, ddof=1), abs=1e-8)


def test_powt_universal_matches_dplR():
    pytest.importorskip("statsmodels")
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = dpl.powt(rwl, method="universal", return_power=True)
    # dplR lme4 estimate is 0.568198; the statsmodels optimizer matches to ~1e-5
    assert p == pytest.approx(0.568198, abs=1e-4)


def test_powt_universal_vector_rejected():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    with pytest.raises(ValueError):
        dpl.powt(rwl["CAM011"], method="universal")


def test_powt_negative_values_rejected():
    df = pd.DataFrame({"A": [1.0, 2.0, -1.0]}, index=pd.Index([1, 2, 3], name="Year"))
    with pytest.raises(ValueError):
        dpl.powt(df)


def test_powt_bad_method():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    with pytest.raises(ValueError):
        dpl.powt(rwl, method="bogus")


def test_powt_bad_input():
    with pytest.raises(TypeError):
        dpl.powt("not a frame")
