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
    return rwl, po


# dplR rcs(gp.rwl, gp.po) reference values (dplR 1.7.9).
_DPLR_RC_CAPS = [1.77953972318469, 1.81512159923917, 1.85060629609674,
                 1.88585048834372, 1.92069114443547]      # regional curve, first 5
_DPLR_RWI_01A = [0.478881, 0.532360, 0.521238]            # series 01A, first 3 (yrs 1640-42)


def test_rcs_matches_dplR_caps():
    rwl, po = _gp()
    res = dpl.rcs(rwl, po, method="caps", make_plot=False, rc_out=True)
    assert np.allclose(res["rc"][:5], _DPLR_RC_CAPS, atol=1e-8)
    a = res["rwi"]["01A"].dropna().to_numpy()[:3]
    assert np.allclose(a, _DPLR_RWI_01A, atol=1e-5)


def test_rcs_ratios_vs_residuals():
    rwl, po = _gp()
    ratio = dpl.rcs(rwl, po, ratios=True, make_plot=False)
    resid = dpl.rcs(rwl, po, ratios=False, make_plot=False)
    # ratios are centred near 1, residuals near 0
    assert 0.8 < np.nanmean(ratio.to_numpy()) < 1.2
    assert abs(np.nanmean(resid.to_numpy())) < 0.5


def test_rcs_default_po_is_one():
    # with no pith offsets every series starts at cambial age 1
    rwl, _ = _gp()
    res = dpl.rcs(rwl, make_plot=False)
    assert res.shape == rwl.shape
    assert np.isfinite(res.to_numpy()[~np.isnan(res.to_numpy())]).all()


def test_rcs_ads_method_runs():
    rwl, po = _gp()
    res = dpl.rcs(rwl, po, method="ads", make_plot=False, rc_out=True)
    assert res["rwi"].shape == rwl.shape
    assert np.all(res["rc"][~np.isnan(res["rc"])] > 0)


def test_rcs_rc_out_toggle():
    rwl, po = _gp()
    plain = dpl.rcs(rwl, po, make_plot=False)
    full = dpl.rcs(rwl, po, make_plot=False, rc_out=True)
    assert isinstance(plain, pd.DataFrame)
    assert isinstance(full, dict) and set(full.keys()) == {"rwi", "rc"}
    assert np.allclose(plain.to_numpy()[~np.isnan(plain.to_numpy())],
                       full["rwi"].to_numpy()[~np.isnan(full["rwi"].to_numpy())])


def test_rcs_bad_args():
    rwl, _ = _gp()
    with pytest.raises(ValueError):
        dpl.rcs(rwl, method="bogus", make_plot=False)
    with pytest.raises(ValueError):
        dpl.rcs(rwl, preset="bogus", make_plot=False)
    with pytest.raises(TypeError):
        dpl.rcs("not a frame", make_plot=False)


# CRUST's compiled spline3 kernel, run on the gp regional curve (ss=10),
# reproduced by dplPy to ~3.6e-11. These first regional-curve values bake that
# direct-against-CRUST validation into CI without needing the CRUST binary.
_CRUST_RC_GP = [1.06287953, 1.26619931, 1.45329834, 1.61293684,
                1.75231530, 1.87680382, 1.99123400, 2.09755297]


def test_rcs_crust_preset_matches_crust_kernel():
    rwl, po = _gp()
    res = dpl.rcs(rwl, po, preset="crust", make_plot=False, rc_out=True)
    assert np.allclose(res["rc"][:8], _CRUST_RC_GP, atol=1e-6)
    # crust preset leaves the dplR path untouched: default still equals dplR
    p1 = dpl.rcs(rwl, po, make_plot=False, rc_out=True)
    assert not np.allclose(res["rc"][:8], p1["rc"][:8])   # the two paths differ


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
