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


def _gp():
    rwl = _read_quiet("tests/data/csv/gp_rwl.csv")
    po = pd.read_csv("tests/data/csv/gp_po.csv")
    # gp_po.csv stores dplR-convention pith offsets (the cambial age of the first
    # measured ring, minimum 1). dplPy 0.8 uses a TRUE offset (0 = pith present),
    # so subtract one to reproduce dplR's reference values.
    po = po.copy()
    col = "pith_offset" if "pith_offset" in po.columns else po.columns[1]
    po[col] = po[col] - 1
    return rwl, po


# dplR rcs(gp.rwl, gp.po) reference values (dplR 1.7.9), reproduced with the
# fixed-stiffness spline at 10% of the curve length -- fit="Spline", period=0.1
# (dplPy 0.8's spelling of pre-0.8 method="caps" with the default stiffness).
_DPLR_RC_CAPS = [1.77953972318469, 1.81512159923917, 1.85060629609674,
                 1.88585048834372, 1.92069114443547]      # regional curve, first 5
_DPLR_RWI_01A = [0.478881, 0.532360, 0.521238]            # series 01A, first 3 (yrs 1640-42)


def test_rcs_matches_dplR_spline():
    # Parity anchor: fit="Spline", period=0.1 reproduces dplR's caps() rcs exactly.
    rwl, po = _gp()
    res = dpl.rcs(rwl, po, fit="Spline", period=0.1, plot=False, rc_out=True)
    assert np.allclose(res["rc"][:5], _DPLR_RC_CAPS, atol=1e-8)
    a = res["rwi"]["01A"].dropna().to_numpy()[:3]
    assert np.allclose(a, _DPLR_RWI_01A, atol=1e-5)


def test_rcs_period_percent_equivalence():
    # period=-10 (ARSTAN percent-of-length) == period=0.1 (fraction): both 10%.
    rwl, po = _gp()
    a = dpl.rcs(rwl, po, fit="Spline", period=0.1, plot=False).to_numpy()
    b = dpl.rcs(rwl, po, fit="Spline", period=-10, plot=False).to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_rcs_ratio_vs_difference():
    rwl, po = _gp()
    ratio = dpl.rcs(rwl, po, method="ratio", plot=False)
    diff = dpl.rcs(rwl, po, method="difference", plot=False)
    # ratios are centred near 1, differences near 0
    assert 0.8 < np.nanmean(ratio.to_numpy()) < 1.2
    assert abs(np.nanmean(diff.to_numpy())) < 0.5


def test_rcs_difference_rescale_to_ratio_scale():
    # rescale_difference puts each series' residual index on its ratio scale (~1)
    rwl, po = _gp()
    plain = dpl.rcs(rwl, po, method="difference", plot=False)
    rescaled = dpl.rcs(rwl, po, method="difference", rescale_difference=True,
                       plot=False)
    assert abs(np.nanmean(plain.to_numpy())) < 0.5           # centred ~0
    assert 0.8 < np.nanmean(rescaled.to_numpy()) < 1.2       # lifted to ~1


def test_rcs_default_po_is_zero():
    # default pith offset is now 0 (pith present); None == an all-zero offset table
    rwl, _ = _gp()
    res = dpl.rcs(rwl, plot=False)
    assert res.shape == rwl.shape
    assert np.isfinite(res.to_numpy()[~np.isnan(res.to_numpy())]).all()
    zeros = pd.DataFrame({"series": list(rwl.columns),
                          "pith_offset": [0] * rwl.shape[1]})
    res0 = dpl.rcs(rwl, zeros, plot=False)
    a, b = res.to_numpy(), res0.to_numpy()
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])


def test_rcs_pith_offset_shifts_alignment():
    # a uniform offset of k starts the regional curve k cambial ages deeper
    rwl, _ = _gp()
    cols = list(rwl.columns)
    off = pd.DataFrame({"series": cols, "pith_offset": [4] * len(cols)})
    rc = dpl.rcs(rwl, off, fit=None, plot=False, rc_out=True)["rc"]
    assert np.isnan(rc[:4]).all() and np.isfinite(rc[4])


def test_rcs_default_fit_is_agedepspline():
    # the default curve is now the age-dependent spline (ARSTAN-faithful), which
    # differs from the fixed spline
    rwl, po = _gp()
    default = dpl.rcs(rwl, po, plot=False, rc_out=True)["rc"]
    spline = dpl.rcs(rwl, po, fit="Spline", period=0.1, plot=False,
                     rc_out=True)["rc"]
    assert not np.allclose(default, spline, equal_nan=True)


def test_rcs_agedepspline_runs():
    rwl, po = _gp()
    res = dpl.rcs(rwl, po, fit="AgeDepSpline", plot=False, rc_out=True)
    assert res["rwi"].shape == rwl.shape
    assert np.all(res["rc"][~np.isnan(res["rc"])] > 0)


def test_rcs_fit_none_is_unsmoothed():
    # fit=None uses the age-aligned mean itself as the regional curve
    rwl, po = _gp()
    none = dpl.rcs(rwl, po, fit=None, plot=False, rc_out=True)["rc"]
    spline = dpl.rcs(rwl, po, fit="Spline", period=0.1, plot=False,
                     rc_out=True)["rc"]
    assert np.any(np.isfinite(none))
    assert not np.allclose(none, spline, equal_nan=True)     # raw != smoothed


def test_rcs_parametric_rc_runs():
    # a curvefit parametric curve (ModNegExp) can smooth the regional curve
    rwl, po = _gp()
    res = dpl.rcs(rwl, po, fit="ModNegExp", plot=False, rc_out=True)
    assert res["rwi"].shape == rwl.shape
    assert np.all(res["rc"][~np.isnan(res["rc"])] > 0)


def test_rcs_rc_out_toggle():
    rwl, po = _gp()
    plain = dpl.rcs(rwl, po, plot=False)
    full = dpl.rcs(rwl, po, plot=False, rc_out=True)
    assert isinstance(plain, pd.DataFrame)
    assert isinstance(full, dict) and set(full.keys()) == {"rwi", "rc"}
    assert np.allclose(plain.to_numpy()[~np.isnan(plain.to_numpy())],
                       full["rwi"].to_numpy()[~np.isnan(full["rwi"].to_numpy())])


def test_rcs_make_plot_deprecated():
    rwl, po = _gp()
    with pytest.warns(DeprecationWarning):
        dpl.rcs(rwl, po, make_plot=False)


def test_rcs_bad_args():
    rwl, _ = _gp()
    with pytest.raises(ValueError):            # unknown curve
        dpl.rcs(rwl, fit="bogus", plot=False)
    with pytest.raises(ValueError):            # unknown arithmetic
        dpl.rcs(rwl, method="bogus", plot=False)
    with pytest.raises(ValueError):            # curve name passed to method=
        dpl.rcs(rwl, method="Spline", plot=False)
    with pytest.raises(ValueError):
        dpl.rcs(rwl, preset="bogus", plot=False)
    with pytest.raises(TypeError):
        dpl.rcs("not a frame", plot=False)


# CRUST's compiled spline3 kernel, run on the gp regional curve (ss=10),
# reproduced by dplPy to ~3.6e-11. These first regional-curve values bake that
# direct-against-CRUST validation into CI without needing the CRUST binary.
_CRUST_RC_GP = [1.06287953, 1.26619931, 1.45329834, 1.61293684,
                1.75231530, 1.87680382, 1.99123400, 2.09755297]


def test_rcs_crust_preset_matches_crust_kernel():
    rwl, po = _gp()
    res = dpl.rcs(rwl, po, preset="crust", plot=False, rc_out=True)
    assert np.allclose(res["rc"][:8], _CRUST_RC_GP, atol=1e-6)
    # crust preset differs from the default (age-dependent-spline) path
    p1 = dpl.rcs(rwl, po, plot=False, rc_out=True)
    assert not np.allclose(res["rc"][:8], p1["rc"][:8])


def test_rcs_crust_curve_floor_and_flat_tail():
    from dplpy.rcs import _crust_regional_curve
    # a curve that dips below the 0.02 floor and has a sparse tail
    ca_m = np.concatenate([np.linspace(1.0, 0.005, 60), np.full(20, 0.005)])
    ca_n = np.concatenate([np.full(60, 10), np.arange(20, 0, -1)])  # tail depth < 4
    rc = _crust_regional_curve(ca_m, ca_n, len(ca_m), ss=10, rise=False)
    assert np.nanmin(rc) >= 0.02 - 1e-12                 # floored
    assert np.allclose(rc[-5:], rc[-5])                  # flat tail/extension


def test_rcs_crust_gap_infill():
    from dplpy.rcs import _crust_regional_curve
    # interior gap (NaN) in the regional curve must be carried forward, not left NA
    ca_m = np.array([2.0, 1.8, np.nan, np.nan, 1.5, 1.4] + [1.3] * 20)
    ca_n = np.array([10, 10, 0, 0, 10, 10] + [10] * 20)
    rc = _crust_regional_curve(ca_m, ca_n, len(ca_m), ss=10, rise=False)
    assert np.all(np.isfinite(rc))                       # no NaNs survive infill
