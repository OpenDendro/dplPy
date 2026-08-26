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


# dplR ads(CAM011, nyrs0=50) values (dplR 1.7.9). The native SciPy banded solve
# reproduces Ed Cook's Fortran to ~1e-12 (LAPACK factorises in a different but
# equivalent order), well within the 1e-9 tolerance below.
_DPLR_ADS_POST = [0.907646803828732, 0.89980232316213, 0.891989632853055,
                  0.884239486481173, 0.876591128952404]
_DPLR_ADS_POSF = [0.883537310392112, 0.879297712650668, 0.875064761706647]


def test_ads_matches_dplR_pos_slope_true():
    data = _read_quiet("tests/data/csv/ca533.csv")
    y = data["CAM011"].dropna().to_numpy()
    sp = dpl.ads(y, nyrs0=50, pos_slope=True)
    assert np.allclose(sp[:5], _DPLR_ADS_POST, atol=1e-9)
    assert len(sp) == len(y)


def test_ads_matches_dplR_pos_slope_false():
    data = _read_quiet("tests/data/csv/ca533.csv")
    y = data["CAM011"].dropna().to_numpy()
    sp = dpl.ads(y, nyrs0=50, pos_slope=False)
    assert np.allclose(sp[:3], _DPLR_ADS_POSF, atol=1e-9)


def test_ads_stiffness_depends_on_nyrs0():
    data = _read_quiet("tests/data/csv/ca533.csv")
    y = data["CAM011"].dropna().to_numpy()
    # a stiffer initial spline (larger nyrs0) is smoother -> smaller local wiggle
    s20 = dpl.ads(y, nyrs0=20)
    s100 = dpl.ads(y, nyrs0=100)
    assert np.std(np.diff(s100)) < np.std(np.diff(s20))


def test_ads_bad_args():
    with pytest.raises(ValueError):
        dpl.ads(np.array([1.0, 2.0]))          # < 3 points
    with pytest.raises(ValueError):
        dpl.ads(np.arange(1.0, 20.0), nyrs0=1)  # nyrs0 must be > 1


def test_detrend_agedepspline_runs_and_positive():
    data = _read_quiet("tests/data/csv/ca533.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rwi = dpl.detrend(data, fit="AgeDepSpline", plot=False)
    assert rwi.shape[1] == data.shape[1]
    # residual RWI should be centred near 1 and finite
    vals = rwi.to_numpy()
    vals = vals[~np.isnan(vals)]
    assert np.all(np.isfinite(vals))
    assert 0.8 < np.nanmean(vals) < 1.2


def test_detrend_agedepspline_matches_dplR_ca533():
    # Full pipeline: dplPy detrend(fit='AgeDepSpline') vs dplR
    # detrend(method='AgeDepSpline'). CAM011 first values from dplR.
    data = _read_quiet("tests/data/csv/ca533.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rwi = dpl.detrend(data, fit="AgeDepSpline", plot=False)
    cam011 = rwi["CAM011"].dropna().to_numpy()[:3]
    assert np.allclose(cam011, [1.17709, 1.01217, 1.17706], atol=1e-4)
