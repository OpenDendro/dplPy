import warnings

import numpy as np
import pandas as pd
import dplpy as dpl
import pytest

import importlib
_m_chron = importlib.import_module("dplpy.chron")

def test_chron_simple_means():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))

    expected_df = pd.DataFrame(data={"std": [0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55],
                                    "samp_depth": [2, 2, 2, 2, 2, 2, 2, 2]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))

    result_df = dpl.chron(input_df, biweight=False, prewhiten=False, plot=False)

    pd.testing.assert_frame_equal(expected_df, result_df)


def test_wrong_input():
    with pytest.raises(TypeError) as errorMsg:
        dpl.chron("string")

    assert "Expected dataframe input, got <class 'str'> instead." == str(errorMsg.value)


def test_chron_biweight_means():
    # the std column is the row-wise Tukey biweight robust mean (tbrm_rows); a
    # year with a single series equals that value, and an outlier is downweighted
    from dplpy.tbrm import tbrm, tbrm_rows
    rng = np.random.default_rng(0)
    arr = 1.0 + 0.1 * rng.standard_normal((30, 6))
    arr[5, 0] = 9.0                                   # outlier
    arr[:10, 5] = np.nan                              # staggered start
    df = pd.DataFrame(arr, index=pd.Index(range(1901, 1931), name="Year"),
                      columns=list("ABCDEF"))
    out = dpl.chron(df, biweight=True, plot=False)
    assert np.allclose(out["std"].to_numpy(), tbrm_rows(arr))
    assert np.allclose(out["std"].to_numpy(),
                       [tbrm(r[~np.isnan(r)]) for r in arr])
    assert abs(out["std"].iloc[5] - 1.0) < 0.2         # outlier downweighted
    assert (out["samp_depth"].to_numpy() == df.notna().sum(axis=1).to_numpy()).all()


def test_chron_prewhiten_routes_through_prewhiten(monkeypatch):
    # chron(prewhiten=True) whitens every series with the package-standard
    # autoreg._prewhiten (audit D1) and averages the result into 'res'.
    calls = []

    def fake_prewhiten(x, estimator="yw", ar_max=10, first_aic_min=True, order=None,
                       backcast=True, return_params=False):
        calls.append((estimator, ar_max, first_aic_min, order, backcast))
        return np.asarray(x, dtype=float) + 0.01

    monkeypatch.setattr(_m_chron, "_prewhiten", fake_prewhiten)
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                            index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], name="Year"))
    expected_df = pd.DataFrame(data={"std": [0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55],
                                     "res": [0.16, 0.36, 0.56, 0.76, 0.96, 1.16, 1.36, 1.56],
                                     "samp_depth": [2, 2, 2, 2, 2, 2, 2, 2]},
                               index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], name="Year"))
    result_df = dpl.chron(input_df, biweight=False, prewhiten=True, plot=False)
    pd.testing.assert_frame_equal(expected_df, result_df)
    # package defaults: YW, ar_max=10, first local AIC min, AIC selection,
    # dplR-style NaN warm-up (backcast=False)
    assert calls == [("yw", 10, True, None, False)] * 2
    calls.clear()
    with pytest.warns(DeprecationWarning):
        dpl.chron(input_df, biweight=False, prewhiten=True, plot=False, max_lag=4,
                  ar_method="Burg", first_aic_min=False, backcast=True, aic=False)
    assert calls == [("burg", 4, False, 4, True)] * 2


def test_chron_keeps_zero_depth_years_as_nan():
    # A14: a year with no series present stays in the output with std NaN and
    # samp_depth 0 (as dplR's chron()), so the index always equals the input's
    idx = pd.Index(range(1900, 1910), name="Year")
    df = pd.DataFrame({"A": np.linspace(0.9, 1.1, 10), "B": np.linspace(1.1, 0.9, 10)},
                      index=idx)
    df.loc[1904:1905, :] = np.nan
    out = dpl.chron(df, plot=False, prewhiten=True)
    assert list(out.index) == list(idx)
    assert out.loc[1904, "samp_depth"] == 0 and np.isnan(out.loc[1904, "std"])
    assert np.isnan(out.loc[1905, "res"])
    assert (out.drop([1904, 1905])["samp_depth"] == 2).all()
    assert np.isfinite(out.drop([1904, 1905])["std"]).all()
    # chron_stabilized builds on chron and no longer crashes on this input
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = dpl.chron_stabilized(df, window_length=6)
    assert len(cs) == 10 and cs.loc[1904, "samp_depth"] == 0


def test_chron_default_ci_replicates_is_100():
    import inspect
    assert inspect.signature(dpl.chron).parameters["R"].default == 100


def test_chron_carries_attrs_as_crn():
    rwi = _staggered_rwi()
    rwi.attrs["site"] = "x"
    rwi.attrs["dplpy_precision"] = 0.01
    out = dpl.chron(rwi, plot=False)
    assert out.attrs["kind"] == "crn" and out.attrs["site"] == "x"
    assert "dplpy_precision" not in out.attrs


def test_chron_bad_ar_method_names_parameter():
    rwi = _staggered_rwi()
    with pytest.raises(ValueError, match="ar_method"):
        dpl.chron(rwi, plot=False, prewhiten=True, ar_method="nope")


def test_chron_plot():
    # headless smoke test: chron(plot=True) draws a figure without error
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                            index=pd.Index(data=[1, 2, 3, 4], name="Year"))
    plt.close("all")
    dpl.chron(input_df, prewhiten=False, plot=True)
    assert len(plt.get_fignums()) >= 1
    plt.close("all")


# --- stabilize= convenience flag -------------------------------------------
# see dev/variance_stabilization_design_2026-09-10.md
def _staggered_rwi(n_years=200, n_series=12, seed=0):
    """years x series RWI-like frame with staggered starts (depth rises)."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, n_years)
    years = np.arange(1, n_years + 1)
    cols = {}
    for s in range(n_series):
        start = s * (n_years // (2 * n_series))
        v = np.full(n_years, np.nan)
        noise = rng.normal(0, 1, n_years)
        v[start:] = 1.0 + 0.15 * (common[start:] + noise[start:])
        cols["s%02d" % s] = v
    return pd.DataFrame(cols, index=years)


def test_chron_stabilize_none_is_default_unchanged():
    rwi = _staggered_rwi()
    base = dpl.chron(rwi, prewhiten=True, plot=False)
    assert "std_vsc" not in base.columns and "res_vsc" not in base.columns
    assert list(base.columns) == ["std", "res", "samp_depth"]


def test_chron_stabilize_adds_vsc_columns():
    rwi = _staggered_rwi()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dpl.chron(rwi, prewhiten=True, plot=False, stabilize="both")
    # vsc columns sit beside their base chronology; base columns unchanged
    assert list(out.columns) == ["std", "std_vsc", "res", "res_vsc", "samp_depth"]
    base = dpl.chron(rwi, prewhiten=True, plot=False)
    assert np.allclose(out["std"], base["std"], equal_nan=True)
    assert np.allclose(out["res"], base["res"], equal_nan=True)
    assert np.isfinite(out["std_vsc"]).any() and np.isfinite(out["res_vsc"]).any()


def test_chron_stabilize_no_res_vsc_without_prewhiten():
    rwi = _staggered_rwi()
    out = dpl.chron(rwi, prewhiten=False, plot=False, stabilize="rbar")
    assert list(out.columns) == ["std", "std_vsc", "samp_depth"]


def test_chron_stabilize_with_ci_and_prewhiten_column_order():
    # B2: stabilize= and ci=True used to raise KeyError together; one final
    # column order keeps every CI / vsc column beside its base chronology
    rwi = _staggered_rwi()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dpl.chron(rwi, plot=False, stabilize="rbar", ci=True, R=50, seed=0)
        both = dpl.chron(rwi, plot=False, prewhiten=True, stabilize="rbar", ci=True,
                         R=50, seed=0)
    assert list(out.columns) == ["std", "lower_ci", "upper_ci", "std_vsc", "samp_depth"]
    assert list(both.columns) == ["std", "lower_ci", "upper_ci", "std_vsc", "res",
                                  "res_vsc", "samp_depth"]
    w = both[both["samp_depth"] >= 2]
    assert (w["lower_ci"] <= w["std"] + 1e-9).all()
    assert np.isfinite(both["std_vsc"]).any() and np.isfinite(both["res_vsc"]).any()


def test_chron_std_vsc_matches_direct_stabilize_chron():
    # the flag must reproduce a direct stabilize_chron call on the std column,
    # using the standard (detrended) matrix's rbar
    rwi = _staggered_rwi()
    out = dpl.chron(rwi, plot=False, stabilize="both")
    ref = dpl.stabilize_chron(dpl.chron(rwi, plot=False)["std"], "both", rwi=rwi)["vsc"]
    assert np.allclose(out["std_vsc"], ref, equal_nan=True)


def test_chron_stabilize_kwargs_forwarded():
    # a passthrough kwarg (constant rbar) changes the result vs the running default
    rwi = _staggered_rwi()
    running = dpl.chron(rwi, plot=False, stabilize="rbar")["std_vsc"].to_numpy()
    const = dpl.chron(rwi, plot=False, stabilize="rbar",
                      stabilize_kwargs={"rbar_mode": "constant"})["std_vsc"].to_numpy()
    ok = np.isfinite(running) & np.isfinite(const)
    assert not np.allclose(running[ok], const[ok])


def test_chron_stabilize_kwargs_rejects_managed_keys():
    rwi = _staggered_rwi()
    with pytest.raises(ValueError):
        dpl.chron(rwi, plot=False, stabilize="rbar", stabilize_kwargs={"rwi": rwi})


def test_chron_stabilize_kwargs_without_method_raises():
    rwi = _staggered_rwi()
    with pytest.raises(ValueError):
        dpl.chron(rwi, plot=False, stabilize_kwargs={"rbar_mode": "constant"})


def test_chron_invalid_stabilize_method():
    rwi = _staggered_rwi()
    with pytest.raises(ValueError):
        dpl.chron(rwi, plot=False, stabilize="nope")

# --- chron(ci=True): bootstrap confidence intervals on the standard chronology ---

import os
import io
import contextlib


def _ca533_rwi():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "data", "rwl", "ca533.rwl")
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buf):
            return dpl.detrend(dpl.readers(path), fit="Spline", plot=False)


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buf):
            return fn(*a, **k)


def test_ci_default_off_no_columns():
    rwi = _ca533_rwi()
    out = _quiet(dpl.chron, rwi, plot=False)
    assert "lower_ci" not in out.columns and "upper_ci" not in out.columns


def test_ci_columns_and_order():
    rwi = _ca533_rwi()
    out = _quiet(dpl.chron, rwi, plot=False, ci=True, R=100, seed=0)
    assert list(out.columns) == ["std", "lower_ci", "upper_ci", "samp_depth"]
    # samp_depth equals the per-year count of measured series
    assert (out["samp_depth"].to_numpy() == rwi.notna().sum(axis=1).to_numpy()).all()


def test_ci_std_matches_plain_chron():
    rwi = _ca533_rwi()
    plain = _quiet(dpl.chron, rwi, plot=False)
    out = _quiet(dpl.chron, rwi, plot=False, ci=True, R=200, seed=0)
    common = plain.index.intersection(out.index)
    assert np.allclose(plain.loc[common, "std"].to_numpy(),
                       out.loc[common, "std"].to_numpy(), equal_nan=True)


def test_ci_brackets_std():
    out = _quiet(dpl.chron, _ca533_rwi(), plot=False, ci=True, R=300, seed=1)
    w = out[out["samp_depth"] >= 2]
    assert (w["lower_ci"] <= w["std"] + 1e-9).all()
    assert (w["upper_ci"] >= w["std"] - 1e-9).all()


def test_ci_reproducible_with_seed():
    rwi = _ca533_rwi()
    a = _quiet(dpl.chron, rwi, plot=False, ci=True, R=200, seed=42)
    b = _quiet(dpl.chron, rwi, plot=False, ci=True, R=200, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_ci_higher_conf_is_wider():
    rwi = _ca533_rwi()
    c90 = _quiet(dpl.chron, rwi, plot=False, ci=True, conf=0.90, R=400, seed=7)
    c99 = _quiet(dpl.chron, rwi, plot=False, ci=True, conf=0.99, R=400, seed=7)
    w = c90["samp_depth"] >= 5
    width90 = (c90["upper_ci"] - c90["lower_ci"])[w]
    width99 = (c99["upper_ci"] - c99["lower_ci"])[w]
    assert (width99 >= width90 - 1e-9).all()


def test_ci_thin_year_nan_and_warns():
    idx = pd.Index(range(1990, 2001), name="Year")
    df = pd.DataFrame({"A": np.linspace(0.9, 1.1, 11),
                       "B": np.linspace(1.1, 0.9, 11)}, index=idx)
    df.loc[2000, "B"] = np.nan                       # only one series at 2000
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.warns(UserWarning):
            out = dpl.chron(df, plot=False, ci=True, R=100, seed=0)
    assert out.loc[2000, "samp_depth"] == 1
    assert np.isnan(out.loc[2000, "lower_ci"])
    assert np.isnan(out.loc[2000, "upper_ci"])
    assert not np.isnan(out.loc[2000, "std"])         # chronology value still present


def test_ci_arithmetic_mean_option():
    rwi = _ca533_rwi()
    out = _quiet(dpl.chron, rwi, plot=False, ci=True, biweight=False, R=200, seed=0)
    w = out[out["samp_depth"] >= 2]
    assert (w["lower_ci"] <= w["std"] + 1e-9).all()
    assert (w["upper_ci"] >= w["std"] - 1e-9).all()
