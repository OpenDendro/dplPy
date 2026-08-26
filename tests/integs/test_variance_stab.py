import warnings
import io, contextlib

import numpy as np
import pytest
import dplpy as dpl


def _rwi():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.detrend(dpl.readers("./tests/data/csv/ca533.csv"), plot=False)


def _stab(rwi, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.chron_stabilized(rwi, **kw)


def test_chron_stab_method_briffa_mean_rbar():
    # ARSTAN Briffa / Frank MEANr: constant rbar. Runs, right columns, and differs
    # from the running-window default. 'mean_rbar' is an alias for 'briffa'.
    rwi = _rwi()
    run = _stab(rwi, method="running_rbar")
    bri = _stab(rwi, method="briffa")
    assert list(bri.columns) == ["vsc", "samp_depth"]
    assert np.all(np.isfinite(bri["vsc"].dropna().to_numpy()))
    assert not np.allclose(run["vsc"].dropna(), bri["vsc"].dropna())
    alias = _stab(rwi, method="mean_rbar")
    assert np.allclose(alias["vsc"].dropna(), bri["vsc"].dropna())


def test_chron_stab_method_spline():
    # ARSTAN ad-hoc spline: runs, all-positive, stiffness (spline_nyrs) changes it.
    rwi = _rwi()
    spl = _stab(rwi, method="spline")
    assert list(spl.columns) == ["vsc", "samp_depth"]
    v = spl["vsc"].dropna().to_numpy()
    assert np.all(np.isfinite(v)) and np.all(v >= 0)
    fixed = _stab(rwi, method="spline", spline_nyrs=100)      # fixed 100-yr wavelength
    frac = _stab(rwi, method="spline", spline_nyrs=0.3)       # 30% of length
    assert not np.allclose(fixed["vsc"].dropna(), frac["vsc"].dropna())


def test_chron_stab_bad_method():
    rwi = _rwi()
    with pytest.raises(ValueError):
        _stab(rwi, method="bogus")


def test_chron_stab_no_biweight():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    res = dpl.chron_stabilized(data, biweight=False)
    # TODO: assert contents of res


def test_chron_stab_with_biweight():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    res = dpl.chron_stabilized(data)
    # TODO: assert contents of res

def test_chron_stab_with_running_rbar():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    res = dpl.chron_stabilized(data, running_rbar=True)
    # TODO: assert contents of res
