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
# rc is the regional curve that PRODUCED rwi (the last iteration's curve, audit
# E4); before 0.9.0 it was rebuilt from the converged chronology, one signal-free
# step further on (~3e-4 different).
_RC_GP = [1.0575637, 1.2983658, 1.53232346, 1.74809542, 1.94221369, 2.10962028]
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
    # D6: rescale_difference (default False, as in rcs) controls the scale of
    # difference indices. False: residuals in measurement units centred on ~0,
    # signal-free iteration run additively. True: CRUST residual mode, rescaled
    # onto the ratio scale (centred near 1); either way they differ from ratios.
    rwl, po = _gp()
    plain = _sfrcs_quiet(rwl, po=po, plot=False, method="difference")
    resid = _sfrcs_quiet(rwl, po=po, plot=False, method="difference",
                         rescale_difference=True)
    ratio = _sfrcs_quiet(rwl, po=po, plot=False, method="ratio")
    present = ~np.isnan(rwl.to_numpy())
    assert np.all(np.isfinite(plain.to_numpy()[present]))
    assert abs(np.nanmean(plain.to_numpy())) < 0.2
    assert np.all(np.isfinite(resid.to_numpy()[present]))
    assert 0.8 < np.nanmean(resid.to_numpy()) < 1.2
    assert not np.allclose(resid.to_numpy()[present], ratio.to_numpy()[present])
    assert not np.allclose(plain.to_numpy()[present], resid.to_numpy()[present])


def test_sfrcs_difference_one_iteration_is_rcs_centred():
    # with one iteration the additive engine is rcs(method="difference") minus
    # its grand mean
    rwl, po = _gp()
    a = _sfrcs_quiet(rwl, po=po, plot=False, method="difference", max_iterations=1)
    b = dpl.rcs(rwl, po=po, preset="crust", biweight=False, method="difference",
                plot=False)
    bn = b.to_numpy()
    bn = bn - np.nanmean(bn)
    assert np.allclose(a.to_numpy(), bn, equal_nan=True, atol=1e-12)


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


# --- v0.9.0 audit follow-ups ------------------------------------------------

def _ca533():
    return _read_quiet("tests/data/csv/ca533.csv")


def test_sfrcs_interior_gap_advances_cambial_age():
    # A1: before 0.9.0 the years after a gap were written shifted by the gap
    # length and the series' last years became NaN. Now the gap stays NaN and
    # every later year keeps its own value.
    rwl = _ca533()
    g = rwl.copy()
    g.loc[[1630, 1631, 1632], "CAM011"] = np.nan
    full = _sfrcs_quiet(rwl, plot=False)["CAM011"]
    gap = _sfrcs_quiet(g, plot=False)["CAM011"]
    assert gap.loc[[1630, 1631, 1632]].isna().all()
    assert gap.notna().sum() == full.notna().sum() - 3
    post = full.index > 1632
    same = np.nanmax(np.abs(full[post].to_numpy() - gap[post].to_numpy()))
    shifted = np.nanmax(np.abs(full[post].to_numpy()[3:] - gap[post].to_numpy()[:-3]))
    assert same < 5e-3 and shifted > 0.5


def test_sfrcs_one_iteration_equals_rcs_for_every_fit():
    # D7: the curve step is rcs's _fit_regional_curve, so one signal-free
    # iteration is rcs up to the mean-1 normalisation, for any fit.
    rwl, po = _gp()
    for fit in ("AgeDepSpline", "Spline", "ModNegExp", "Hugershoff", "LinearAny",
                "Mean", None):
        a = _sfrcs_quiet(rwl, po=po, plot=False, fit=fit, preset=None,
                         max_iterations=1).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = dpl.rcs(rwl, po, fit=fit, biweight=False, plot=False).to_numpy()
        assert np.allclose(a, b / np.nanmean(b), equal_nan=True, atol=1e-9), fit
    # and the default (preset="crust") is rcs(preset="crust", biweight=False)
    a = _sfrcs_quiet(rwl, po=po, plot=False, max_iterations=1).to_numpy()
    b = dpl.rcs(rwl, po, preset="crust", biweight=False, plot=False).to_numpy()
    assert np.allclose(a, b / np.nanmean(b), equal_nan=True, atol=1e-9)


def test_sfrcs_biweight_aliases():
    rwl, po = _gp()
    ref = _sfrcs_quiet(rwl, po=po, plot=False, biweight=True, max_iterations=3)
    with pytest.warns(DeprecationWarning, match="biweight_curve"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a = dpl.sfrcs(rwl, po=po, plot=False, biweight_curve=True,
                          biweight_crn=True, verbose=False, max_iterations=3)
    assert np.allclose(a.to_numpy(), ref.to_numpy(), equal_nan=True)
    # disagreeing pair: biweight_crn wins, with a warning saying so
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            b = dpl.sfrcs(rwl, po=po, plot=False, biweight_curve=False,
                          biweight_crn=True, verbose=False, max_iterations=3)
    assert any("disagree" in str(x.message) for x in w)
    assert np.allclose(b.to_numpy(), ref.to_numpy(), equal_nan=True)


def test_sfrcs_period_alias_and_bad_kwarg():
    rwl, po = _gp()
    with pytest.warns(DeprecationWarning, match="period="):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dpl.sfrcs(rwl, po=po, plot=False, fit="Spline", preset=None,
                      period=0.1, verbose=False, max_iterations=2)
    with pytest.raises(TypeError):
        _sfrcs_quiet(rwl, po=po, plot=False, bogus=1)


def test_sfrcs_return_info_rc_is_the_curve_that_made_rwi():
    # E4: rwi == unalign(rwca / rc) / grand mean, with the returned rc
    from dplpy.rcs import _align, _unalign, _resolve_po
    rwl, po = _gp()
    info = _sfrcs_quiet(rwl, po=po, plot=False, return_info=True)
    M = rwl.to_numpy(dtype=float)
    po_arr = _resolve_po(rwl, po)
    rwica = _align(M, po_arr) / info["rc"][:, None]
    rwica = rwica / np.nanmean(rwica)
    assert np.allclose(_unalign(rwica, M, po_arr), info["rwi"].to_numpy(),
                       equal_nan=True, atol=1e-12)


def test_sfrcs_output_attrs():
    rwl, po = _gp()
    rwl.attrs.update({"dplpy_precision": 0.01, "kind": "rwl"})
    rwi = _sfrcs_quiet(rwl, po=po, plot=False, max_iterations=2)
    assert rwi.attrs["kind"] == "rwi" and "dplpy_precision" not in rwi.attrs
