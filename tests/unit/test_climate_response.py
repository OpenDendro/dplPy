"""Tests for dpl.climate_response (bootstrapped correlation / response functions).

Covers the signed month-window mapping (against treeclim's correct_continuous
cases), the correlation function (vs numpy), the response function (vs the
standardized-OLS identity and an independent recomputation), the three bootstrap
paths and their significance conventions, multi/single-variable input, period
restriction, guard rails, reproducibility, the figure, and non-mutation.
"""
import numpy as np
import pandas as pd
import pytest

import importlib

import dplpy as dpl
# the package attribute ``dplpy.climate_response`` is the exported function, so
# reach the module (for its helpers) through importlib, as for moving_corr
CR = importlib.import_module("dplpy.climate_response")
from dplpy.climate import _zcols


# --------------------------------------------------------------- fixtures ---

def _make(seed=0, planted="prev_jul_temp"):
    """Deterministic chronology + two monthly climate variables (wide format).

    By default the chronology responds to previous-year July temperature.
    """
    rng = np.random.default_rng(seed)
    yrs = np.arange(1931, 2021)
    n = len(yrs)
    temp = pd.DataFrame(rng.normal(13, 4, (n, 12)), index=yrs,
                        columns=list(range(1, 13)))
    prec = pd.DataFrame(rng.normal(65, 20, (n, 12)), index=yrs,
                        columns=list(range(1, 13)))
    def z(s):
        return (s - np.nanmean(s)) / np.nanstd(s)
    if planted == "prev_jul_temp":
        drive = pd.Series(temp[7]).shift(1).to_numpy()
    else:
        drive = temp[7].to_numpy()
    drive = np.nan_to_num(drive, nan=np.nanmean(drive))
    crn = pd.Series(0.7 * z(drive) + rng.normal(0, 0.5, n), index=yrs)
    crn.index.name = "year"
    return crn, {"temp": temp, "prec": prec}


# ------------------------------------------------------------ import paths ---

def test_import_paths():
    assert dpl.climate_response is CR.climate_response
    assert dpl.ClimateResponseResult is CR.ClimateResponseResult


# ------------------------------------------------- month-window mapping ---

@pytest.mark.parametrize("start,end,expect", [
    (-6, 9, [-6, -7, -8, -9, -10, -11, -12, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (-1, 10, [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12] + list(range(1, 11))),
    (-12, 2, [-12, 1, 2]),
    (-21, 2, [-21, -22, -23, -24] + list(range(-1, -13, -1)) + [1, 2]),
    (-9, 8, [-9, -10, -11, -12, 1, 2, 3, 4, 5, 6, 7, 8]),   # Kevin's example
    (1, 12, list(range(1, 13))),
    (-1, -12, list(range(-1, -13, -1))),
])
def test_window_ids_match_treeclim_correct_continuous(start, end, expect):
    assert CR._window_ids(start, end) == expect


def test_id_abs_roundtrip_and_meaning():
    for mid in list(range(-24, 0)) + list(range(1, 13)):
        assert CR._abs_to_id(CR._id_to_abs(mid)) == mid
    # -9 is previous September; 8 is current August
    assert CR._id_year_offset(-9) == -1 and CR._id_calendar_month(-9) == 9
    assert CR._id_year_offset(8) == 0 and CR._id_calendar_month(8) == 8
    assert CR._id_label(-9) == "sep" and CR._id_label(8) == "AUG"


def test_start_after_end_raises():
    with pytest.raises(ValueError, match="chronologically before"):
        CR._window_ids(9, -6)
    with pytest.raises(ValueError, match="not a month"):
        CR._id_to_abs(0)


# ------------------------------------------------- correlation function ---

def test_correlation_equals_pearson():
    crn, clim = _make()
    res = dpl.climate_response(crn, clim, method="correlation", boot="none",
                               plot=False)
    design, _ = CR._build_design(CR._normalize_climate(clim, None),
                                 CR._window_ids(-6, 9))
    common = design.dropna().index.intersection(crn.dropna().index)
    lo, hi = int(common.min()), int(common.max())
    U = design.loc[lo:hi].to_numpy(float)
    g = crn.loc[lo:hi].to_numpy(float)
    ref = np.array([np.corrcoef(U[:, j], g)[0, 1] for j in range(U.shape[1])])
    assert np.allclose(res.table["coef"].to_numpy(), ref, atol=1e-12)


# --------------------------------------------------- response function ---

def test_response_full_retention_is_standardized_ols():
    rng = np.random.default_rng(1)
    n, k = 80, 9
    F = rng.normal(size=(n, 3))
    U = F @ rng.normal(size=(3, k)) + rng.normal(0, 0.5, (n, k))
    g = U @ rng.normal(size=k) + rng.normal(0, 1, n)
    Uz, gz = _zcols(U), CR._zvec(g)
    Ep, Z = CR._response_pca(Uz, min_pcs=k)          # force every PC
    b_full = CR._response_fit(Ep, Z, gz)
    b_ols = np.linalg.solve(np.corrcoef(Uz, rowvar=False), CR._corr_coef(Uz, gz))
    assert np.allclose(b_full, b_ols, atol=1e-8)


def test_pvp_truncates_for_collinear_predictors():
    rng = np.random.default_rng(2)
    n, k = 90, 12
    F = rng.normal(size=(n, 3))                       # only 3 real dimensions
    U = F @ rng.normal(size=(3, k)) + rng.normal(0, 0.3, (n, k))
    Ep, _ = CR._response_pca(_zcols(U))
    assert 1 <= Ep.shape[1] < k                       # kept fewer than all


def test_orthogonal_predictors_keep_at_least_one_pc():
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(60, 6)))     # exactly orthogonal
    Ep, _ = CR._response_pca(_zcols(Q))
    assert Ep.shape[1] >= 1                            # PVP floor, no crash


# ------------------------------------------- bootstrap paths + columns ---

@pytest.mark.parametrize("method", ["correlation", "response"])
@pytest.mark.parametrize("boot", ["exact", "std", "stationary", "none"])
def test_all_method_boot_combinations(method, boot):
    crn, clim = _make()
    res = dpl.climate_response(crn, clim, method=method, boot=boot, seed=1,
                               plot=False)
    assert len(res.table) == len(CR._window_ids(-6, 9)) * 2
    if boot == "exact":
        assert res.table["p"].between(0, 1).all()
        assert res.table["ci_lower"].isna().all()     # no CI for exact
    elif boot in ("std", "stationary"):
        assert res.table["ci_lower"].notna().all()
        assert res.table["p"].isna().all()            # no p for percentile method
    else:
        assert not res.table["significant"].any()     # none -> point estimate only


# ----------------------------------------------- stationary bootstrap ---

def test_block_length_grows_with_autocorrelation():
    def ar1(phi, n, rng):
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal()
        return x
    med = {}
    for phi in (0.0, 0.5, 0.9):
        bs = [CR._b_star_sb(ar1(phi, 400, np.random.default_rng(k)))
              for k in range(15)]
        med[phi] = np.median(bs)
    assert med[0.0] <= med[0.5] <= med[0.9]
    assert med[0.0] <= 2 and med[0.9] >= 10


def test_stationary_resampler_preserves_autocorrelation():
    rng = np.random.default_rng(1)
    x = np.zeros(400)
    for t in range(1, 400):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    def r1(s):
        return np.corrcoef(s[:-1], s[1:])[0, 1]
    b = CR._b_star_sb(x)
    rng2 = np.random.default_rng(2)
    stat = np.mean([r1(x[CR._stationary_indices(len(x), b, rng2)])
                    for _ in range(200)])
    iid = np.mean([r1(x[rng2.integers(0, len(x), len(x))]) for _ in range(200)])
    assert stat > iid + 0.1          # blocks keep persistence; iid destroys it


def test_stationary_records_block_length_and_guards_short_windows():
    crn, clim = _make()
    res = dpl.climate_response(crn, clim, method="correlation", boot="stationary",
                               seed=1, plot=False)
    assert isinstance(res.settings["block_length"], int)
    assert res.settings["block_length"] >= 1
    with pytest.raises(ValueError, match="stationary bootstrap needs"):
        dpl.climate_response(crn, clim, boot="stationary", period=(1995, 2010),
                             plot=False)


def test_exact_recovers_planted_signal():
    crn, clim = _make(planted="prev_jul_temp")
    res = dpl.climate_response(crn, clim, method="correlation", boot="exact",
                               seed=1, plot=False)
    tt = res.table[res.table["variable"] == "temp"]
    top = tt.reindex(tt["coef"].abs().sort_values(ascending=False).index).iloc[0]
    assert top["month"] == "jul" and top["year"] == "previous"
    assert bool(top["significant"])


def test_std_significance_is_ci_excludes_zero():
    crn, clim = _make()
    res = dpl.climate_response(crn, clim, method="correlation", boot="std",
                               seed=5, plot=False)
    lo = res.table["ci_lower"].to_numpy()
    hi = res.table["ci_upper"].to_numpy()
    expect = np.sign(lo) == np.sign(hi)
    assert np.array_equal(res.table["significant"].to_numpy(), expect)


def test_seed_reproducible():
    crn, clim = _make()
    a = dpl.climate_response(crn, clim, boot="exact", seed=7, plot=False)
    b = dpl.climate_response(crn, clim, boot="exact", seed=7, plot=False)
    assert np.array_equal(a.table["significant"], b.table["significant"])
    assert np.allclose(a.table["p"], b.table["p"])


# ------------------------------------------------------- input variety ---

def test_single_variable_series_and_window():
    crn, clim = _make()
    res = dpl.climate_response(crn, clim["temp"], start_month=-9, end_month=8,
                               var_names="temp", method="correlation",
                               boot="none", plot=False)
    assert res.settings["variables"] == ["temp"]
    assert len(res.table) == 12
    assert set(res.table["variable"]) == {"temp"}


def test_period_restriction():
    crn, clim = _make()
    res = dpl.climate_response(crn, clim, boot="none", period=(1960, 2000),
                               plot=False)
    assert res.period[0] >= 1960 and res.period[1] <= 2000


# ------------------------------------------------------------ guard rails ---

def test_guard_rails():
    crn, clim = _make()
    with pytest.raises(ValueError, match="method"):
        dpl.climate_response(crn, clim, method="bogus", plot=False)
    with pytest.raises(ValueError, match="boot"):
        dpl.climate_response(crn, clim, boot="bogus", plot=False)
    with pytest.raises(ValueError, match="ci must be"):
        dpl.climate_response(crn, clim, ci=0.2, plot=False)
    with pytest.raises(ValueError, match="at least"):
        dpl.climate_response(crn, clim, boot="exact", nboot=100, plot=False)
    # one-month window -> too few predictors (single variable)
    with pytest.raises(ValueError, match="at least"):
        dpl.climate_response(crn, clim["temp"], start_month=5, end_month=5,
                             var_names="temp", plot=False)


def test_exact_needs_enough_years():
    crn, clim = _make()
    with pytest.raises(ValueError, match="exact bootstrap needs"):
        dpl.climate_response(crn, clim, boot="exact", period=(1990, 2010),
                             plot=False)


# ------------------------------------------------------------------ plot ---

def test_plot_runs_headless():
    import matplotlib
    matplotlib.use("Agg")
    crn, clim = _make()
    res = dpl.climate_response(crn, clim, method="response", boot="std", seed=2,
                               plot=True)
    assert res.fig is not None
    assert len(res.axes) == 2                          # one panel per variable
    import matplotlib.pyplot as plt
    plt.close("all")


# -------------------------------------------------------- non-mutation ---

def test_inputs_not_mutated():
    crn, clim = _make()
    crn0 = crn.copy()
    t0 = clim["temp"].copy()
    p0 = clim["prec"].copy()
    dpl.climate_response(crn, clim, method="response", boot="exact", seed=1,
                         plot=False)
    pd.testing.assert_series_equal(crn, crn0)
    pd.testing.assert_frame_equal(clim["temp"], t0)
    pd.testing.assert_frame_equal(clim["prec"], p0)
