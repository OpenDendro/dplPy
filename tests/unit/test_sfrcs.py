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


def _sfrcs_quiet(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.sfrcs(*args, verbose=False, **kw)


def _gp():
    rwl = _read_quiet("tests/data/csv/gp_rwl.csv")
    po = pd.read_csv("tests/data/csv/gp_po.csv")
    return rwl, po


# NOTE: there is no dplR reference for signal-free RCS (dplR has no such
# function). These are dplPy's own converged values on the gp data -- a
# REGRESSION anchor (does the port keep reproducing itself), NOT a match against
# an external gold standard. The regional-curve smoother underneath is separately
# validated against CRUST's compiled spline3 kernel in test_rcs.py.
_SFC_GP = [1.00975254, 0.91797774, 0.74175436, 1.03246588, 1.22140989, 1.06727725]
_RC_GP = [1.05662908, 1.29672372, 1.53181446, 1.75210504, 1.95325716, 2.12763972]
_RWI_01A_GP = [0.439466, 0.489377, 0.480054]


def test_sfrcs_runs_and_converges():
    rwl, po = _gp()
    info = _sfrcs_quiet(rwl, po=po, make_plot=False, return_info=True)
    assert info["converged"] is True
    assert info["n_iter"] <= 40
    # convergence measure decreases monotonically and ends below tol
    conv = info["conv"]
    assert conv[-1] < 1e-3
    assert np.all(np.diff(conv) <= 1e-9)


def test_sfrcs_regression_values():
    rwl, po = _gp()
    info = _sfrcs_quiet(rwl, po=po, make_plot=False, return_info=True)
    assert np.allclose(info["sfc"]["sfc"].to_numpy()[:6], _SFC_GP, atol=1e-6)
    assert np.allclose(info["rc"][:6], _RC_GP, atol=1e-6)
    a = info["rwi"]["01A"].dropna().to_numpy()[:3]
    assert np.allclose(a, _RWI_01A_GP, atol=1e-5)


def test_sfrcs_return_shape_and_finite():
    rwl, po = _gp()
    rwi = _sfrcs_quiet(rwl, po=po, make_plot=False)
    assert isinstance(rwi, pd.DataFrame)
    assert rwi.shape == rwl.shape
    present = ~np.isnan(rwl.to_numpy())
    assert np.all(np.isfinite(rwi.to_numpy()[present]))


def test_sfrcs_differs_from_plain_rcs():
    # the signal-free iteration must actually change the result vs one-pass RCS
    rwl, po = _gp()
    plain = dpl.rcs(rwl, po, preset="crust", make_plot=False).to_numpy()
    sf = _sfrcs_quiet(rwl, po=po, make_plot=False).to_numpy()
    m = ~np.isnan(rwl.to_numpy())
    assert not np.allclose(plain[m], sf[m])         # signal-free adjusts it
    assert np.corrcoef(plain[m], sf[m])[0, 1] > 0.9  # but stays highly correlated


def test_sfrcs_iteration_moves_then_settles():
    # one iteration (no signal removed) differs from the converged chronology;
    # running to convergence changes the low frequencies by a real amount
    rwl, po = _gp()
    one = _sfrcs_quiet(rwl, po=po, make_plot=False, max_iterations=1,
                       return_info=True)
    conv = _sfrcs_quiet(rwl, po=po, make_plot=False, return_info=True)
    d = np.nanmax(np.abs(one["sfc"]["sfc"].to_numpy()
                         - conv["sfc"]["sfc"].to_numpy()))
    assert d > 1e-3
    assert one["n_iter"] == 1 and conv["n_iter"] > 1


def test_sfrcs_guards_stabilise_hard_data():
    # co021 (near-zero chronology years) defeats the basic ssf; the SF-RCS
    # division guard + 0.02 curve floor keep it finite and convergent even with
    # default pith offsets (all = 1).
    co = _read_quiet("tests/data/csv/co021.csv")
    info = _sfrcs_quiet(co, make_plot=False, return_info=True)
    sfc = info["sfc"]["sfc"].to_numpy()
    assert np.all(np.isfinite(sfc))
    assert np.nanmin(info["rc"]) >= 0.02 - 1e-12    # curve never dives to zero


def test_sfrcs_residuals_path():
    # CRUST residual mode rescales residuals to the ratios' moments, so they sit
    # on the ratio scale (centred near 1) and differ from the ratio indices.
    rwl, po = _gp()
    resid = _sfrcs_quiet(rwl, po=po, make_plot=False, ratios=False)
    ratio = _sfrcs_quiet(rwl, po=po, make_plot=False, ratios=True)
    present = ~np.isnan(rwl.to_numpy())
    assert np.all(np.isfinite(resid.to_numpy()[present]))
    assert 0.8 < np.nanmean(resid.to_numpy()) < 1.2          # on the ratio scale
    assert not np.allclose(resid.to_numpy()[present],
                           ratio.to_numpy()[present])         # but not identical


def test_sfrcs_return_info_keys():
    rwl, po = _gp()
    info = _sfrcs_quiet(rwl, po=po, make_plot=False, return_info=True)
    for key in ("rwi", "sfc", "samp_depth", "conv", "n_iter", "converged", "rc"):
        assert key in info
    assert list(info["sfc"].columns) == ["sfc", "samp.depth"]
    # samp.depth is the count of trees per calendar year
    assert info["sfc"]["samp.depth"].max() <= rwl.shape[1]


def test_sfrcs_default_po_is_one():
    rwl, _ = _gp()
    rwi = _sfrcs_quiet(rwl, make_plot=False)           # no pith offsets
    assert rwi.shape == rwl.shape


def test_sfrcs_bad_args():
    rwl, _ = _gp()
    with pytest.raises(TypeError):
        _sfrcs_quiet("not a frame", make_plot=False)
    with pytest.raises(ValueError):
        _sfrcs_quiet(rwl, make_plot=False, max_iterations=0)
