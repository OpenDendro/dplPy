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
    # D9: return_info=True returns ONE dict with the DataFrame under the stable
    # primary key "vsc" plus the diagnostics (as the other return_info= functions)
    info = dpl.stabilize_chron(crn, "rbar", rwi=rwi, return_info=True)
    assert isinstance(info, dict)
    assert {"vsc", "method", "rbar", "n_eff", "samp_depth"} <= set(info)
    assert isinstance(info["vsc"], pd.DataFrame) and "vsc" in info["vsc"].columns
    assert len(info["n_eff"]) == len(crn)
    plain = dpl.stabilize_chron(crn, "rbar", rwi=rwi)
    pd.testing.assert_frame_equal(info["vsc"], plain)


def test_window_length_alias_and_too_long_window_raises(crn, rwi):
    # D3: win_length= -> window_length= (deprecated alias); B10: a running rbar
    # needs a window shorter than the chronology, else ValueError (not a silent
    # rbar=0 degrade)
    base = dpl.stabilize_chron(crn, "rbar", rwi=rwi, window_length=40)
    with pytest.warns(DeprecationWarning):
        alias = dpl.stabilize_chron(crn, "rbar", rwi=rwi, win_length=40)
    pd.testing.assert_frame_equal(base, alias)
    with pytest.raises(ValueError, match="window_length"):
        dpl.stabilize_chron(crn, "rbar", rwi=rwi, window_length=len(crn))
    short = crn.iloc[:40]
    with pytest.raises(ValueError, match="window_length"):
        dpl.stabilize_chron(short, "rbar", rwi=rwi.iloc[:40])          # default 50 > 40
    ok = dpl.stabilize_chron(short, "rbar", rwi=rwi.iloc[:40], rbar_mode="constant")
    assert np.isfinite(ok["vsc"].to_numpy()).any()
    with pytest.raises(TypeError):
        dpl.stabilize_chron(crn, "rbar", rwi=rwi, bogus=1)


def test_method_and_rbar_mode_case_insensitive(crn, rwi):
    a = dpl.stabilize_chron(crn, "rbar", rwi=rwi, rbar_mode="constant")
    b = dpl.stabilize_chron(crn, "Rbar", rwi=rwi, rbar_mode="CONSTANT")
    pd.testing.assert_frame_equal(a, b)


def test_stabilize_chron_carries_attrs_as_crn(crn, rwi):
    c = crn.copy()
    c.attrs["site"] = "x"
    out = dpl.stabilize_chron(c, "spline")
    assert out.attrs["kind"] == "crn" and out.attrs["site"] == "x"


def test_spline_component_handles_nan_ok_and_clip():
    from dplpy.stabilize import _spline_component
    rng = np.random.default_rng(3)
    x = 1 + 0.3 * rng.standard_normal(120) * np.linspace(2, 0.5, 120)
    x[:5] = np.nan
    out = _spline_component(x, 40, 0.5)
    assert np.isnan(out[:5]).all() and np.isfinite(out[5:]).all()
    ok = np.ones(120, dtype=bool)
    ok[5:10] = False
    clipped = _spline_component(x, 40, 0.5, clip_negative=True, ok=ok)
    assert (clipped[5:] >= 0).all()
    ref = variance_stabilize_spline(x[5:], period=40, f=0.5, clip_negative=True,
                                    ok=ok[5:])
    assert np.allclose(clipped[5:], ref)


def test_invalid_method(crn):
    with pytest.raises(ValueError):
        dpl.stabilize_chron(crn, "nope")


def test_invalid_rbar_mode(crn, rwi):
    with pytest.raises(ValueError):
        dpl.stabilize_chron(crn, "rbar", rwi=rwi, rbar_mode="sideways")
