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
    # gp_po.csv is dplR-convention; dplPy uses a true offset (0 = pith present)
    po = po.copy()
    col = "pith_offset" if "pith_offset" in po.columns else po.columns[1]
    po[col] = po[col] - 1
    return rwl, po


def _mrcs(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dpl.mrcs(*a, **k)


def test_mrcs_one_curve_equals_rcs():
    # the key invariant: a single curve is exactly single-curve RCS
    rwl, po = _gp()
    a = _mrcs(rwl, po=po, n_curves=1, plot=False)
    b = dpl.rcs(rwl, po=po, plot=False)
    assert a.shape == b.shape
    an, bn = a.to_numpy(), b.to_numpy()
    assert np.allclose(an, bn, equal_nan=True)


def test_mrcs_runs_and_shape():
    rwl, po = _gp()
    rwi = _mrcs(rwl, po=po, n_curves=2, plot=False)
    assert isinstance(rwi, pd.DataFrame)
    assert rwi.shape == rwl.shape
    present = ~np.isnan(rwl.to_numpy())
    assert np.all(np.isfinite(rwi.to_numpy()[present]))


def test_mrcs_rc_out_structure():
    rwl, po = _gp()
    out = _mrcs(rwl, po=po, n_curves=2, plot=False, rc_out=True)
    assert set(out.keys()) == {"rwi", "rc", "alloc"}
    assert len(out["rc"]) == 2
    assert set(out["alloc"].values()) == {0, 1}          # every tree allocated
    assert len(out["alloc"]) == rwl.shape[1]
    # the two growth-class curves are not identical
    rc0, rc1 = out["rc"][0], out["rc"][1]
    n = min(len(rc0), len(rc1))
    assert not np.allclose(rc0[:n], rc1[:n], equal_nan=True)


def test_mrcs_reset_means_restores_single_means():
    # with reset_means, each tree's mean returns to its one-curve RCS mean
    rwl, po = _gp()
    single = dpl.rcs(rwl, po=po, plot=False)
    reset = _mrcs(rwl, po=po, n_curves=2, reset_means=True, plot=False)
    for c in rwl.columns:
        ms = np.nanmean(single[c].to_numpy())
        mr = np.nanmean(reset[c].to_numpy())
        if np.isfinite(ms) and np.isfinite(mr):
            assert abs(ms - mr) < 1e-9
    # but the multi-curve correction still changes the year-by-year indices
    assert not np.allclose(reset.to_numpy(), single.to_numpy(), equal_nan=True)


def test_mrcs_reset_vs_noreset_differ():
    rwl, po = _gp()
    a = _mrcs(rwl, po=po, n_curves=2, reset_means=True, plot=False)
    b = _mrcs(rwl, po=po, n_curves=2, reset_means=False, plot=False)
    assert not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("sort", ["growth", "age", "diameter", "none"])
def test_mrcs_sort_options_run(sort):
    rwl, po = _gp()
    rwi = _mrcs(rwl, po=po, n_curves=2, sort=sort, plot=False)
    assert rwi.shape == rwl.shape


def test_mrcs_difference_skips_reset_with_warning():
    rwl, po = _gp()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dpl.mrcs(rwl, po=po, n_curves=2, method="difference",
                      reset_means=True, plot=False)
    assert any("reset_means is only applied" in str(x.message) for x in w)


def test_mrcs_sparse_curve_warns():
    rwl, po = _gp()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # gp has ~34 series; 2 curves -> ~17 each, below the 40-tree rule
            dpl.mrcs(rwl, po=po, n_curves=2, plot=False)
    assert any("< 40" in str(x.message) for x in w)


def test_mrcs_bad_args():
    rwl, po = _gp()
    with pytest.raises(ValueError):
        _mrcs(rwl, po=po, n_curves=0, plot=False)
    with pytest.raises(ValueError):
        _mrcs(rwl, po=po, sort="bogus", plot=False)
    with pytest.raises(TypeError):
        _mrcs("not a frame", plot=False)


# --- v0.9.0 audit follow-ups ------------------------------------------------

def test_mrcs_interior_gap_runs_and_keeps_years():
    # A1: an interior NaN used to crash mrcs ("setting an array element with a
    # sequence"); now it stays NaN and later years keep their own values.
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    g = rwl.copy()
    g.loc[[1630, 1631, 1632], "CAM011"] = np.nan
    full = _mrcs(rwl, n_curves=2, plot=False)["CAM011"]
    gap = _mrcs(g, n_curves=2, plot=False)["CAM011"]
    assert gap.loc[[1630, 1631, 1632]].isna().all()
    post = full.index > 1632
    same = np.nanmax(np.abs(full[post].to_numpy() - gap[post].to_numpy()))
    shifted = np.nanmax(np.abs(full[post].to_numpy()[3:] - gap[post].to_numpy()[:-3]))
    assert same < 5e-3 and shifted > 0.5


def test_mrcs_period_alias_sort_case_and_attrs():
    rwl, po = _gp()
    rwl.attrs.update({"dplpy_precision": 0.01, "kind": "rwl"})
    with pytest.warns(DeprecationWarning, match="period="):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            a = dpl.mrcs(rwl, po=po, n_curves=2, fit="Spline", period=0.1, plot=False)
    b = _mrcs(rwl, po=po, n_curves=2, fit="Spline", spline_period=0.1, plot=False)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    c = _mrcs(rwl, po=po, n_curves=2, sort="GROWTH", plot=False)      # D11
    d = _mrcs(rwl, po=po, n_curves=2, sort="growth", plot=False)
    assert np.allclose(c.to_numpy(), d.to_numpy(), equal_nan=True)
    assert c.attrs["kind"] == "rwi" and "dplpy_precision" not in c.attrs


def test_mrcs_po_as_dict_missing_series_warns():
    rwl, _ = _gp()
    with pytest.warns(UserWarning, match="no pith offset"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.filterwarnings("always", message=".*no pith offset.*")
            rwi = dpl.mrcs(rwl, po={rwl.columns[0]: 3}, n_curves=2, plot=False)
    assert rwi.shape == rwl.shape
