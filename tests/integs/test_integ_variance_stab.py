import warnings
import io
import contextlib

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
    # ARSTAN ad-hoc spline: runs, all-positive, stiffness (spline_period) changes it.
    rwi = _rwi()
    spl = _stab(rwi, method="spline")
    assert list(spl.columns) == ["vsc", "samp_depth"]
    v = spl["vsc"].dropna().to_numpy()
    assert np.all(np.isfinite(v)) and np.all(v >= 0)
    fixed = _stab(rwi, method="spline", spline_period=100)    # fixed 100-yr wavelength
    frac = _stab(rwi, method="spline", spline_period=0.3)     # 30% of length
    assert not np.allclose(fixed["vsc"].dropna(), frac["vsc"].dropna())


def test_chron_stab_bad_method():
    rwi = _rwi()
    with pytest.raises(ValueError):
        _stab(rwi, method="bogus")


def test_chron_stab_no_biweight():
    data = dpl.readers("./tests/data/csv/ca533.csv")
    res = dpl.chron_stabilized(data, biweight=False)
    assert list(res.columns) == ["vsc", "samp_depth"]
    assert len(res) == data.shape[0]
    assert np.all(np.isfinite(res["vsc"].to_numpy()))


def test_chron_stab_with_biweight():
    data = dpl.readers("./tests/data/csv/ca533.csv")
    res = dpl.chron_stabilized(data)
    assert list(res.columns) == ["vsc", "samp_depth"]
    assert np.all(np.isfinite(res["vsc"].to_numpy()))
    assert (res["samp_depth"] >= 0).all()


def test_chron_stab_with_running_rbar():
    data = dpl.readers("./tests/data/csv/ca533.csv")
    res = dpl.chron_stabilized(data, running_rbar=True)
    # running_rbar=True adds the per-window rbar column (D10: snake_case name)
    assert list(res.columns) == ["vsc", "running_rbar", "samp_depth"]
    assert np.all(np.isfinite(res["vsc"].to_numpy()))


def test_chron_stab_running_rbar_pinned_ca533():
    # regression pin of the dplR-style running-rbar chronology on detrended ca533
    # (values from the per-window DataFrame.corr implementation of v0.8)
    rwi = _rwi()
    res = _stab(rwi, running_rbar=True)
    ref = {700: (0.8106334565, 0.2137712393), 1200: (1.1720279498, 0.4199472710),
           1900: (1.0479641948, 0.3625816560), 1983: (1.3061500012, 0.4334263925)}
    for yr, (v, r) in ref.items():
        assert res.loc[yr, "vsc"] == pytest.approx(v, abs=1e-8)
        assert res.loc[yr, "running_rbar"] == pytest.approx(r, abs=1e-8)
    bri = _stab(rwi, method="briffa")
    assert bri.loc[1200, "vsc"] == pytest.approx(1.1690089975, abs=1e-8)
    crn = dpl.chron(rwi, plot=False)
    assert dpl.stabilize_chron(crn, "rbar", rwi=rwi).loc[1200, "vsc"] == pytest.approx(
        1.1885092136, abs=1e-8)
    assert dpl.stabilize_chron(crn, "both", rwi=rwi).loc[1200, "vsc"] == pytest.approx(
        1.1994755948, abs=1e-6)
