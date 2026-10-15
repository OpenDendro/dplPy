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
# fixed-stiffness spline at 10% of the curve length -- fit="Spline", spline_period=0.1
# (dplPy 0.8's spelling of pre-0.8 method="caps" with the default stiffness).
_DPLR_RC_CAPS = [1.77953972318469, 1.81512159923917, 1.85060629609674,
                 1.88585048834372, 1.92069114443547]      # regional curve, first 5
_DPLR_RWI_01A = [0.478881, 0.532360, 0.521238]            # series 01A, first 3 (yrs 1640-42)


def test_rcs_matches_dplR_spline():
    # Parity anchor: fit="Spline", spline_period=0.1 reproduces dplR's caps() rcs exactly.
    rwl, po = _gp()
    res = dpl.rcs(rwl, po, fit="Spline", spline_period=0.1, plot=False, rc_out=True)
    assert np.allclose(res["rc"][:5], _DPLR_RC_CAPS, atol=1e-8)
    a = res["rwi"]["01A"].dropna().to_numpy()[:3]
    assert np.allclose(a, _DPLR_RWI_01A, atol=1e-5)


def test_rcs_period_percent_equivalence():
    # spline_period=-10 (ARSTAN percent-of-length) == spline_period=0.1 (fraction): both 10%.
    rwl, po = _gp()
    a = dpl.rcs(rwl, po, fit="Spline", spline_period=0.1, plot=False).to_numpy()
    b = dpl.rcs(rwl, po, fit="Spline", spline_period=-10, plot=False).to_numpy()
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
    spline = dpl.rcs(rwl, po, fit="Spline", spline_period=0.1, plot=False,
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
    spline = dpl.rcs(rwl, po, fit="Spline", spline_period=0.1, plot=False,
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


# --- v0.9.0 audit follow-ups ------------------------------------------------

def _ca533():
    return _read_quiet("tests/data/csv/ca533.csv")


def _gapped(rwl):
    g = rwl.copy()
    g.loc[[1630, 1631, 1632], "CAM011"] = np.nan
    return g


def test_rcs_interior_gap_advances_cambial_age():
    # A1: an interior NaN is an unmeasured ring -- it stays NaN, every later
    # year keeps its true cambial age, and the result is (near) the gap-free
    # value in the SAME year, not shifted by the gap length.
    rwl = _ca533()
    full = dpl.rcs(rwl, plot=False)["CAM011"]
    gap = dpl.rcs(_gapped(rwl), plot=False)["CAM011"]
    assert gap.loc[[1630, 1631, 1632]].isna().all()
    post = full.index > 1632
    same = np.nanmax(np.abs(full[post].to_numpy() - gap[post].to_numpy()))
    shifted = np.nanmax(np.abs(full[post].to_numpy()[3:] - gap[post].to_numpy()[:-3]))
    assert same < 5e-3                       # only the regional curve moved a hair
    assert shifted > 0.5                     # a shifted mapping would be way off
    # pre-gap years are untouched up to the (slightly) changed regional curve
    pre = full.index < 1630
    assert np.nanmax(np.abs(full[pre].to_numpy() - gap[pre].to_numpy())) < 5e-3


def test_align_unalign_roundtrip_with_gap():
    from dplpy.rcs import _align, _unalign
    M = np.array([[1.0, np.nan], [2.0, 5.0], [np.nan, 6.0], [4.0, 7.0]])
    po = np.array([0, 3])
    rwca = _align(M, po)
    # series 0: ages 1,2,(gap),4 ; series 1: starts at offset 3 -> rows 3,4,5
    assert rwca.shape == (6, 2)
    assert np.allclose(rwca[:, 0], [1, 2, np.nan, 4, np.nan, np.nan], equal_nan=True)
    assert np.allclose(rwca[:, 1], [np.nan, np.nan, np.nan, 5, 6, 7], equal_nan=True)
    back = _unalign(rwca, M, po)
    assert np.allclose(back, M, equal_nan=True)


def test_rcs_true_age_axis_with_zero_coverage_ages():
    # A15: two series, po=0 (ages 1-30) and po=40 (ages 41-70); ages 31-40 have
    # no coverage. A linear fit on the TRUE age axis recovers an exactly linear
    # regional mean; on a compacted 1..60 axis it would not.
    age0 = np.arange(1, 31)
    age1 = np.arange(41, 71)
    f = lambda a: 2.0 - 0.01 * a
    yrs = np.arange(1900, 1930)
    rwl = pd.DataFrame({"A": f(age0), "B": f(age1)}, index=pd.Index(yrs, name="Year"))
    po = pd.DataFrame({"series": ["A", "B"], "pith_offset": [0, 40]})
    with pytest.warns(UserWarning, match="no series covers"):
        res = dpl.rcs(rwl, po, fit="LinearAny", biweight=False, plot=False, rc_out=True)
    rc = res["rc"]
    ages = np.arange(1, len(rc) + 1)
    valid = ~np.isnan(rc)
    assert np.isnan(rc[30:40]).all()                   # no data -> no curve there
    assert np.allclose(rc[valid], f(ages[valid]), atol=1e-9)
    # the splines carry the mean across the gap and run without error
    for fit in ("Spline", "AgeDepSpline", "ModNegExp"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = dpl.rcs(rwl, po, fit=fit, biweight=False, plot=False, rc_out=True)
        assert np.isfinite(r["rc"][valid]).all()


def test_resolve_po_policies():
    from dplpy.rcs import _resolve_po
    rwl, _ = _gp()
    cols = list(rwl.columns)
    # None -> zeros
    assert np.all(_resolve_po(rwl, None) == 0)
    # dict / Series / DataFrame agree
    d = {c: i % 3 for i, c in enumerate(cols)}
    a = _resolve_po(rwl, d)
    b = _resolve_po(rwl, pd.Series(d))
    c = _resolve_po(rwl, pd.DataFrame({"series": cols, "pith_offset": list(d.values())}))
    e = _resolve_po(rwl, pd.DataFrame({"id": cols, "offset": list(d.values())}))  # positional
    assert np.array_equal(a, b) and np.array_equal(a, c) and np.array_equal(a, e)
    # missing series -> 0 with a warning naming them
    with pytest.warns(UserWarning, match="no pith offset"):
        r = _resolve_po(rwl, {cols[0]: 5})
    assert r[0] == 5 and np.all(r[1:] == 0)
    # NA / <NA> (as wc_to_po gives) -> 0 with a warning
    tab = pd.DataFrame({"series": cols,
                        "pith_offset": pd.array([2] + [pd.NA] * (len(cols) - 1), dtype="Int64")})
    with pytest.warns(UserWarning, match="NA"):
        r = _resolve_po(rwl, tab)
    assert r[0] == 2 and np.all(r[1:] == 0)
    # negative or fractional -> ValueError
    with pytest.raises(ValueError, match="negative"):
        _resolve_po(rwl, {c: -1 for c in cols})
    with pytest.raises(ValueError, match="whole number"):
        _resolve_po(rwl, {c: 1.5 for c in cols})


def test_rcs_min_n_too_large_raises():
    rwl, po = _gp()
    with pytest.raises(ValueError, match="min_n"):
        dpl.rcs(rwl, po, min_n=10_000, plot=False)


def test_rcs_short_regional_curve_falls_back_to_mean():
    # B9: a 3-age regional curve cannot carry an age-dependent spline -> mean
    yrs = pd.Index([2000, 2001, 2002], name="Year")
    rwl = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.5, 2.5, 3.5]}, index=yrs)
    with pytest.warns(UserWarning, match="dirty dog"):
        res = dpl.rcs(rwl, fit="AgeDepSpline", biweight=False, plot=False, rc_out=True)
    assert np.allclose(res["rc"], [2.25, 2.25, 2.25])


def test_rcs_period_alias_and_case_insensitive_preset():
    rwl, po = _gp()
    with pytest.warns(DeprecationWarning, match="period="):
        a = dpl.rcs(rwl, po, fit="Spline", period=0.1, plot=False)
    b = dpl.rcs(rwl, po, fit="Spline", spline_period=0.1, plot=False)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    c = dpl.rcs(rwl, po, preset="CRUST", plot=False)
    d = dpl.rcs(rwl, po, preset="crust", plot=False)
    assert np.allclose(c.to_numpy(), d.to_numpy(), equal_nan=True)
    with pytest.raises(TypeError):
        dpl.rcs(rwl, po, bogus=1, plot=False)


def test_rcs_output_attrs_are_indices():
    # G10: provenance carried, measurement precision dropped, kind retagged
    rwl, po = _gp()
    rwl.attrs.update({"dplpy_precision": 0.01, "kind": "rwl", "site": "gp"})
    rwi = dpl.rcs(rwl, po, plot=False)
    assert rwi.attrs["kind"] == "rwi"
    assert rwi.attrs["site"] == "gp"
    assert "dplpy_precision" not in rwi.attrs


def test_rcs_plot_closes_figure_when_not_interactive():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from unittest.mock import patch
    rwl, po = _gp()
    plt.close("all")
    with patch("matplotlib.pyplot.show") as show:
        dpl.rcs(rwl, po, plot=True)
    show.assert_called_once()
    assert len(plt.get_fignums()) == 0
