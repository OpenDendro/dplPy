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
    po = po.copy()
    col = "pith_offset" if "pith_offset" in po.columns else po.columns[1]
    po[col] = po[col] - 1
    return rwl, po


def _q(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.sfmrcs(*a, verbose=False, **k)


def test_sfmrcs_one_curve_equals_sfrcs():
    # the key invariant: one curve is exactly single-curve signal-free RCS
    rwl, po = _gp()
    a = _q(rwl, po=po, n_curves=1, plot=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            b = dpl.sfrcs(rwl, po=po, verbose=False, plot=False)
    assert a.shape == b.shape
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_sfmrcs_runs_and_converges():
    rwl, po = _gp()
    info = _q(rwl, po=po, n_curves=2, plot=False, return_info=True)
    assert info["converged"] is True
    assert info["n_iter"] <= 40
    assert info["conv"][-1] < 1e-3


def test_sfmrcs_return_info_structure():
    rwl, po = _gp()
    info = _q(rwl, po=po, n_curves=2, plot=False, return_info=True)
    for key in ("rwi", "sfc", "samp_depth", "conv", "n_iter", "converged",
                "rc", "alloc"):
        assert key in info
    assert len(info["rc"]) == 2
    assert set(info["alloc"].values()) == {0, 1}
    assert info["rwi"].shape == rwl.shape
    present = ~np.isnan(rwl.to_numpy())
    assert np.all(np.isfinite(info["rwi"].to_numpy()[present]))
    rc0, rc1 = info["rc"][0], info["rc"][1]
    n = min(len(rc0), len(rc1))
    assert not np.allclose(rc0[:n], rc1[:n], equal_nan=True)


def test_sfmrcs_reset_vs_noreset_differ():
    rwl, po = _gp()
    a = _q(rwl, po=po, n_curves=2, reset_means=True, plot=False)
    b = _q(rwl, po=po, n_curves=2, reset_means=False, plot=False)
    assert not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_sfmrcs_differs_from_single():
    rwl, po = _gp()
    multi = _q(rwl, po=po, n_curves=2, plot=False).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            single = dpl.sfrcs(rwl, po=po, verbose=False, plot=False).to_numpy()
    m = ~np.isnan(rwl.to_numpy())
    assert not np.allclose(multi[m], single[m])


@pytest.mark.parametrize("sort", ["growth", "age", "diameter", "none"])
def test_sfmrcs_sort_options_run(sort):
    rwl, po = _gp()
    rwi = _q(rwl, po=po, n_curves=2, sort=sort, plot=False, max_iterations=10)
    assert rwi.shape == rwl.shape


def test_sfmrcs_difference_skips_reset_with_warning():
    rwl, po = _gp()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dpl.sfmrcs(rwl, po=po, n_curves=2, method="difference",
                       reset_means=True, verbose=False, plot=False,
                       max_iterations=10)
    assert any("reset_means is only applied" in str(x.message) for x in w)


def test_sfmrcs_bad_args():
    rwl, po = _gp()
    with pytest.raises(ValueError):
        _q(rwl, po=po, n_curves=0, plot=False)
    with pytest.raises(ValueError):
        _q(rwl, po=po, sort="bogus", plot=False)
    with pytest.raises(ValueError):
        _q(rwl, po=po, max_iterations=0, plot=False)
    with pytest.raises(TypeError):
        _q("not a frame", plot=False)
