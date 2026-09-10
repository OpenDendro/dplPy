"""Tests for stabilize_chron() -- ARSTAN-style variance stabilization of an
already-built chronology. See dev/variance_stabilization_design_2026-09-10.md."""
import numpy as np
import pandas as pd
import pytest

import dplpy as dpl
from dplpy.smoothingspline import variance_stabilize_spline


def _synthetic_rwi(n_years=200, n_series=12, seed=0):
    """A years x series RWI-like frame with staggered starts (so sample depth and
    running rbar vary through time), values oscillating about ~1.0."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, n_years)              # shared signal
    years = np.arange(1, n_years + 1)
    cols = {}
    for s in range(n_series):
        start = (s * (n_years // (2 * n_series)))    # staggered onset -> rising depth
        v = np.full(n_years, np.nan)
        noise = rng.normal(0, 1, n_years)
        v[start:] = 1.0 + 0.15 * (common[start:] + noise[start:])
        cols["s%02d" % s] = v
    return pd.DataFrame(cols, index=years)


def _running_sd(a, w=40):
    return pd.Series(np.asarray(a, dtype=float)).dropna().rolling(w).std().dropna()


@pytest.fixture(scope="module")
def rwi():
    return _synthetic_rwi()


@pytest.fixture(scope="module")
def crn(rwi):
    return dpl.chron(rwi, plot=False)               # std + samp_depth


# --- method="spline" reuses the verified variance_stabilize_spline core -------
def test_spline_matches_variance_stabilize_spline(crn):
    out = dpl.stabilize_chron(crn, "spline")
    ref = variance_stabilize_spline(crn["std"].to_numpy(), period=None, f=0.5,
                                    clip_negative=True, ok=None)
    assert np.allclose(out["vsc"].to_numpy(), ref, equal_nan=True)


# --- method="rbar" (running) flattens the sample-size variance trend ----------
def test_rbar_running_reduces_variance_trend(crn, rwi):
    out = dpl.stabilize_chron(crn, "rbar", rwi=rwi)
    raw = _running_sd(crn["std"])
    vsc = _running_sd(out["vsc"])
    # the early (low-replication) variance inflation is reduced
    assert vsc.max() / vsc.min() < raw.max() / raw.min()


def test_rbar_restandardizes_to_full_period_mean(crn, rwi):
    out = dpl.stabilize_chron(crn, "rbar", rwi=rwi, clip_negative=False)
    assert np.nanmean(out["vsc"].to_numpy()) == pytest.approx(
        np.nanmean(crn["std"].to_numpy()), abs=1e-9)


# --- both input modes for the rbar family (design decision C) -----------------
def test_rbar_constant_precomputed(crn):
    out = dpl.stabilize_chron(crn, "rbar", rbar=0.4,
                              samp_depth=crn["samp_depth"].to_numpy(),
                              rbar_mode="constant")
    assert np.isfinite(out["vsc"].to_numpy()).all()


def test_rbar_running_without_rwi_needs_running_rbar(crn):
    with pytest.raises(ValueError):
        dpl.stabilize_chron(crn, "rbar", samp_depth=crn["samp_depth"].to_numpy(),
                            rbar_mode="running")


def test_rbar_needs_depth(crn):
    bare = crn["std"]                                # Series, no samp_depth
    with pytest.raises(ValueError):
        dpl.stabilize_chron(bare, "rbar", rbar=0.4, rbar_mode="constant")


# --- method="both" == rbar then spline (ARSTAN isb=2 / Frank) -----------------
def test_both_is_rbar_then_spline(crn, rwi):
    both = dpl.stabilize_chron(crn, "both", rwi=rwi, clip_negative=False)
    r = dpl.stabilize_chron(crn, "rbar", rwi=rwi, clip_negative=False)
    composed = dpl.stabilize_chron(r["vsc"], "spline", clip_negative=False)
    assert np.allclose(both["vsc"].to_numpy(), composed["vsc"].to_numpy(),
                       equal_nan=True)


# --- Series and DataFrame inputs agree (design decision A3) --------------------
def test_series_and_dataframe_inputs_agree(crn):
    from_df = dpl.stabilize_chron(crn, "spline")
    from_series = dpl.stabilize_chron(crn["std"], "spline")
    assert np.allclose(from_df["vsc"].to_numpy(),
                       from_series["vsc"].to_numpy(), equal_nan=True)


# --- leading NaN (as in a residual chronology) is handled --------------------
def test_leading_nan_preserved(crn, rwi):
    # a residual/ARSTAN chronology can carry leading NaN (the first AR-order
    # years); inject some so the test does not depend on the AR order selected.
    holed = crn.copy()
    holed.iloc[:3, holed.columns.get_loc("std")] = np.nan
    for method, kw in [("spline", {}), ("rbar", {"rwi": rwi}), ("both", {"rwi": rwi})]:
        out = dpl.stabilize_chron(holed, method, **kw)
        v = out["vsc"].to_numpy()
        assert np.isnan(v[:3]).all(), method            # NaN preserved at the front
        assert np.isfinite(v[3:]).all(), method         # stabilized where data exist


# --- diagnostics + validation -------------------------------------------------
def test_return_info(crn, rwi):
    out, info = dpl.stabilize_chron(crn, "rbar", rwi=rwi, return_info=True)
    assert {"method", "rbar", "n_eff", "samp_depth"} <= set(info)
    assert len(info["n_eff"]) == len(crn)


def test_invalid_method(crn):
    with pytest.raises(ValueError):
        dpl.stabilize_chron(crn, "nope")


def test_invalid_rbar_mode(crn, rwi):
    with pytest.raises(ValueError):
        dpl.stabilize_chron(crn, "rbar", rwi=rwi, rbar_mode="sideways")
