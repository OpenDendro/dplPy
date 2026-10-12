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
    # gp_po.csv is dplR-convention (cambial age of the first ring, >=1); dplPy 0.8
    # uses a TRUE offset (0 = pith present), so subtract one.
    po = po.copy()
    col = "pith_offset" if "pith_offset" in po.columns else po.columns[1]
    po[col] = po[col] - 1
    return rwl, po


# NOTE: there is no dplR reference for signal-free RCS (dplR has no such
# function). These are dplPy's own converged values on the gp data at the default
# settings (biweight_crn=False, matching CRUST's KRB=1) -- a REGRESSION anchor
# (does the port keep reproducing itself), NOT a match against an external gold
# standard. The regional-curve smoother underneath is separately validated
# against CRUST's compiled spline3 kernel in test_rcs.py; and the full iteration
# matches a headless build of CRUST to ~1e-3 on finnmrg/norwmrg (see
# dev/sfrcs_testing_2026-09-10.md).
_SFC_GP = [1.0088354, 0.91715642, 0.74109939, 1.03156439, 1.20396242, 1.04915458]
_RC_GP = [1.05785466, 1.298724, 1.53274415, 1.74856698, 1.94272965, 2.11017885]
_RWI_01A_GP = [0.436544, 0.486024, 0.476674]


def test_sfrcs_runs_and_converges():
    rwl, po = _gp()
    info = _sfrcs_quiet(rwl, po=po, plot=False, return_info=True)
    assert info["converged"] is True
    assert info["n_iter"] <= 40
    conv = info["conv"]
    assert conv[-1] < 1e-3
    assert np.all(np.diff(conv) <= 1e-9)


def test_sfrcs_regression_values():
    rwl, po = _gp()
    info = _sfrcs_quiet(rwl, po=po, plot=False, return_info=True)
    assert np.allclose(info["sfc"]["sfc"].to_numpy()[:6], _SFC_GP, atol=1e-6)
    assert np.allclose(info["rc"][:6], _RC_GP, atol=1e-6)
    a = info["rwi"]["01A"].dropna().to_numpy()[:3]
    assert np.allclose(a, _RWI_01A_GP, atol=1e-5)


def test_sfrcs_return_shape_and_finite():
    rwl, po = _gp()
    rwi = _sfrcs_quiet(rwl, po=po, plot=False)
    assert isinstance(rwi, pd.DataFrame)
    assert rwi.shape == rwl.shape
    present = ~np.isnan(rwl.to_numpy())
    assert np.all(np.isfinite(rwi.to_numpy()[present]))


def test_sfrcs_differs_from_plain_rcs():
    rwl, po = _gp()
    plain = dpl.rcs(rwl, po, preset="crust", plot=False).to_numpy()
    sf = _sfrcs_quiet(rwl, po=po, plot=False).to_numpy()
    m = ~np.isnan(rwl.to_numpy())
    assert not np.allclose(plain[m], sf[m])
    assert np.corrcoef(plain[m], sf[m])[0, 1] > 0.9


def test_sfrcs_iteration_moves_then_settles():
    rwl, po = _gp()
    one = _sfrcs_quiet(rwl, po=po, plot=False, max_iterations=1, return_info=True)
    conv = _sfrcs_quiet(rwl, po=po, plot=False, return_info=True)
    d = np.nanmax(np.abs(one["sfc"]["sfc"].to_numpy()
                         - conv["sfc"]["sfc"].to_numpy()))
    assert d > 1e-3
    assert one["n_iter"] == 1 and conv["n_iter"] > 1


def test_sfrcs_guards_stabilise_hard_data():
    co = _read_quiet("tests/data/csv/co021.csv")
    info = _sfrcs_quiet(co, plot=False, return_info=True)
    sfc = info["sfc"]["sfc"].to_numpy()
    assert np.all(np.isfinite(sfc))
    assert np.nanmin(info["rc"]) >= 0.02 - 1e-12


def test_sfrcs_difference_path():
    # CRUST residual mode rescales residuals to the ratios' moments, so they sit
    # on the ratio scale (centred near 1) and differ from the ratio indices.
    rwl, po = _gp()
    resid = _sfrcs_quiet(rwl, po=po, plot=False, method="difference")
    ratio = _sfrcs_quiet(rwl, po=po, plot=False, method="ratio")
    present = ~np.isnan(rwl.to_numpy())
    assert np.all(np.isfinite(resid.to_numpy()[present]))
    assert 0.8 < np.nanmean(resid.to_numpy()) < 1.2
    assert not np.allclose(resid.to_numpy()[present], ratio.to_numpy()[present])


def test_sfrcs_return_info_keys():
    rwl, po = _gp()
    info = _sfrcs_quiet(rwl, po=po, plot=False, return_info=True)
    for key in ("rwi", "sfc", "samp_depth", "conv", "n_iter", "converged", "rc"):
        assert key in info
    assert list(info["sfc"].columns) == ["sfc", "samp.depth"]
    assert info["sfc"]["samp.depth"].max() <= rwl.shape[1]


def test_sfrcs_default_po_is_zero():
    rwl, _ = _gp()
    rwi = _sfrcs_quiet(rwl, plot=False)
    assert rwi.shape == rwl.shape


def test_sfrcs_make_plot_deprecated():
    rwl, po = _gp()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dpl.sfrcs(rwl, po=po, make_plot=False, verbose=False, max_iterations=3)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_sfrcs_bad_args():
    rwl, _ = _gp()
    with pytest.raises(TypeError):
        _sfrcs_quiet("not a frame", plot=False)
    with pytest.raises(ValueError):
        _sfrcs_quiet(rwl, plot=False, max_iterations=0)
