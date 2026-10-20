"""Tests for dpl.seascorr (the climate-response module).

Covers the seasonalizer, the exact-simulation surrogate generator, the climate
input normalizer, simple and partial correlations, empirical p-values, the
two-period stability test, guard rails, the bar-chart color logic, a frozen
regression snapshot, and non-mutation of inputs.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl
import dplpy.climate as C
from dplpy import seascorr_plot as SP


# --------------------------------------------------------------- fixtures ---

def _make(seed=0):
    """A fully deterministic synthetic chronology + monthly climate pair."""
    rng = np.random.default_rng(seed)
    yrs = np.arange(1950, 2011)
    myrs = np.arange(1948, 2011)

    def monthly(base, sd):
        rows = [(y, m, rng.normal(base, sd)) for y in myrs for m in range(1, 13)]
        return pd.DataFrame(rows, columns=["year", "month", "value"])

    P = monthly(40.0, 12.0)
    T = monthly(14.0, 4.0)
    x = pd.Series(1.0 + rng.normal(0, 1, len(yrs)), index=yrs)
    x.index.name = "year"
    return x, P, T


# ------------------------------------------------------------ import paths ---

def test_import_paths():
    assert dpl.seascorr is dpl.climate.seascorr
    assert dpl.SeascorrResult is dpl.climate.SeascorrResult


# --------------------------------------------------------- input normalize ---

def _wide_from_tidy(tidy):
    w = tidy.pivot(index="year", columns="month", values="value")
    w.columns = list(range(1, 13))
    w.index.name = "year"
    return w


def test_to_month3_three_shapes_identical():
    _, P, _ = _make()
    tidy = C._to_month3(P, "P")
    wide = C._to_month3(_wide_from_tidy(P), "P")
    di = pd.date_range("1948-01-01", periods=len(P), freq="MS")
    dser = pd.Series(P.sort_values(["year", "month"])["value"].to_numpy(), index=di)
    dt = C._to_month3(dser, "P")
    assert tidy.equals(wide)
    assert tidy.equals(dt)


def test_to_month3_scrambled_month_columns():
    _, P, _ = _make()
    wide = _wide_from_tidy(P)
    scrambled = wide[[12, 1, 6, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
    assert C._to_month3(scrambled, "P").equals(C._to_month3(wide, "P"))


# ---------------------------------------------------------- seasonalizer ---

def test_seasonalize_hand_values():
    # value == absolute month index -> easy to verify sums/means and labels
    rows = [(y, m, y * 12 + (m - 1)) for y in range(1950, 1956) for m in range(1, 13)]
    m3 = pd.DataFrame(rows, columns=["year", "month", "value"])
    frame, meta = C._seasonalize(m3, end_month=9, season_lengths=(1, 3, 6, 12), agg="sum")
    labels = [m["label"] for m in meta[:14]]
    assert labels == ["Aug*", "Sep*", "Oct*", "Nov*", "Dec*", "Jan", "Feb",
                      "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    # 3-month sum ending Sep 1954 = v[Jul]+v[Aug]+v[Sep]
    vsep = 1954 * 12 + 8
    assert frame[(3, 13)].loc[1954] == vsep + (vsep - 1) + (vsep - 2)
    assert frame[(1, 13)].loc[1954] == vsep


def test_seasonalize_sum_vs_mean():
    rows = [(y, m, 1.0) for y in range(1950, 1970) for m in range(1, 13)]
    m3 = pd.DataFrame(rows, columns=["year", "month", "value"])
    fmean, _ = C._seasonalize(m3, 9, (1, 3, 6, 12), "mean")
    fsum, _ = C._seasonalize(m3, 9, (1, 3, 6, 12), "sum")
    assert np.allclose(fmean.dropna().to_numpy(), 1.0)
    # 12-month sum of ones == 12
    assert np.allclose(fsum[[(12, p) for p in range(14)]].dropna().to_numpy(), 12.0)


# ------------------------------------------------------------- pdgmsim ---

def test_pdgmsim_preserves_mean_sd_and_spectrum():
    rng = np.random.default_rng(1)
    n, phi = 120, 0.6
    e = rng.standard_normal(n + 200)
    ar = np.zeros(n + 200)
    for t in range(1, n + 200):
        ar[t] = phi * ar[t - 1] + e[t]
    x = ar[200:]
    sims = C._pdgmsim(x, 2000, np.random.default_rng(2))
    assert sims.shape == (n, 2000)
    # kopt=1 forces exact mean/sd
    assert np.max(np.abs(sims.mean(0) - x.mean())) < 1e-10
    assert np.max(np.abs(sims.std(0, ddof=1) - x.std(ddof=1))) < 1e-10

    def acf1(v):
        v = v - v.mean()
        return np.sum(v[1:] * v[:-1]) / np.sum(v * v)
    obs = acf1(x)
    sur = np.mean([acf1(sims[:, j]) for j in range(sims.shape[1])])
    # surrogates recover clearly-positive autocorrelation (not white noise)
    assert sur > 0.2 and abs(sur - obs) < 0.12

    # spectrum preservation: mean surrogate periodogram tracks the observed one
    def pdgm(v):
        v = v - v.mean()
        z = np.fft.rfft(v)
        return (z * np.conj(z)).real
    p_obs = pdgm(x)
    p_sur = np.mean([pdgm(sims[:, j]) for j in range(sims.shape[1])], axis=0)
    assert np.corrcoef(p_obs[1:], p_sur[1:])[0, 1] > 0.8


def test_pdgmsim_reproducible_with_seed():
    x = np.random.default_rng(3).standard_normal(80)
    a = C._pdgmsim(x, 100, np.random.default_rng(7))
    b = C._pdgmsim(x, 100, np.random.default_rng(7))
    assert np.array_equal(a, b)


# ------------------------------------------------------------ weibull p ---

def test_weibull_p():
    sims = np.linspace(-1, 1, 1000)
    assert C._weibull_p(0.0, sims) == pytest.approx(0.5, abs=0.01)
    # at/above the max saturates near n/(n+1)
    assert C._weibull_p(5.0, sims) == pytest.approx(1000 / 1001.0, abs=1e-6)
    assert C._weibull_p(-5.0, sims) == pytest.approx(1 / 1001.0, abs=1e-6)


# -------------------------------------------- partial correlation identity ---

def test_partial_closedform_equals_residual():
    x, P, T = _make()
    U, meta = C._seasonalize(C._to_month3(P, "P"), 8, (1, 3, 6, 12), "sum")
    V, _ = C._seasonalize(C._to_month3(T, "T"), 8, (1, 3, 6, 12), "mean")
    lo, hi = 1952, 2010
    xv = x.loc[lo:hi].to_numpy()
    Um = U.loc[lo:hi].to_numpy()
    Vm = V.loc[lo:hi].to_numpy()
    # closed form (as seascorr computes it)
    r_xU = C._corr_matrix(xv[:, None], Um).ravel()
    r_xV = C._corr_matrix(xv[:, None], Vm).ravel()
    r_UV = C._corr_paired(Um, Vm)
    closed = C._partial(r_xU, r_xV, r_UV)
    # explicit regression residuals
    resid = []
    for k in range(Um.shape[1]):
        A = np.column_stack([np.ones_like(Um[:, k]), Um[:, k]])
        ex = xv - A @ np.linalg.lstsq(A, xv, rcond=None)[0]
        ev = Vm[:, k] - A @ np.linalg.lstsq(A, Vm[:, k], rcond=None)[0]
        resid.append(np.corrcoef(ex, ev)[0, 1])
    assert np.allclose(closed, resid, atol=1e-10)


# -------------------------------------------------- end-to-end behaviour ---

def test_recovers_planted_signal():
    rng = np.random.default_rng(42)
    yrs = np.arange(1900, 1990)
    myrs = np.arange(1898, 1990)
    rows = [(y, m, rng.normal(50, 10)) for y in myrs for m in range(1, 13)]
    P = pd.DataFrame(rows, columns=["year", "month", "value"])
    T = pd.DataFrame([(y, m, rng.normal(15, 5)) for y in myrs for m in range(1, 13)],
                     columns=["year", "month", "value"])
    U, meta = C._seasonalize(C._to_month3(P, "P"), 9, (1, 3, 6, 12), "sum")
    jun3 = [i for i, m in enumerate(meta)
            if m["season_length"] == 3 and m["end_month"] == 6 and not m["prior"]][0]
    sig = U[list(U.columns)[jun3]].reindex(yrs)
    sig = (sig - sig.mean()) / sig.std()
    x = pd.Series(1.0 + 0.8 * sig.to_numpy() + 0.5 * rng.normal(0, 1, len(yrs)), index=yrs)
    x.index.name = "year"
    res = dpl.seascorr(x, P, T, end_month=9, nsim=1000, seed=1, plot=False)
    top = res.table.loc[res.table["r_primary"].abs().idxmax()]
    assert top["season_length"] == 3 and top["end_month"] == 6
    assert top["r_primary"] > 0.5 and top["sig01_primary"]


def test_regression_snapshot():
    x, P, T = _make()
    res = dpl.seascorr(x, primary=P, secondary=T, end_month=8,
                       season_lengths=(1, 3, 6, 12), primary_agg="sum",
                       secondary_agg="mean", nsim=1000, period=(1952, 2010),
                       seed=0, plot=False)
    t = res.table
    assert res.period == (1952, 2010) and len(t) == 56
    # deterministic correlations (frozen)
    assert t["r_primary"].sum() == pytest.approx(-4.788408, abs=1e-5)
    assert t["r_partial"].sum() == pytest.approx(-6.415086, abs=1e-5)
    assert t["r_primary"].to_numpy()[:3] == pytest.approx(
        [-0.109956, 0.04223, -0.066924], abs=1e-5)
    # deterministic p-values (seeded surrogates)
    assert t["p_primary"].to_numpy()[:2] == pytest.approx([0.198855, 0.624971], abs=1e-5)
    assert np.all((t["p_primary"] >= 0) & (t["p_primary"] <= 1))


# ------------------------------------------------------------ guard rails ---

def test_guard_nsim():
    x, P, T = _make()
    with pytest.raises(ValueError, match="nsim"):
        dpl.seascorr(x, P, T, nsim=500, plot=False)


def test_guard_short_window():
    x, P, T = _make()
    with pytest.raises(ValueError, match="fewer than"):
        dpl.seascorr(x, P, T, nsim=1000, period=(1990, 2005), plot=False)


def test_guard_interior_nan_chronology():
    x, P, T = _make()
    x.loc[1970] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        dpl.seascorr(x, P, T, nsim=1000, plot=False)


def test_guard_season_lengths_start_one():
    x, P, T = _make()
    with pytest.raises(ValueError, match="start with 1"):
        dpl.seascorr(x, P, T, season_lengths=(3, 6, 12), nsim=1000, plot=False)


# -------------------------------------------------------- stability test ---

def test_stability_single_period_none():
    x, P, T = _make()
    res = dpl.seascorr(x, P, T, nsim=1000, period=(1955, 2008), seed=0, plot=False)
    assert res.stability is None


def test_stability_stable_vs_unstable():
    rng = np.random.default_rng(42)
    yrs = np.arange(1900, 1990)
    myrs = np.arange(1898, 1990)
    P = pd.DataFrame([(y, m, rng.normal(50, 10)) for y in myrs for m in range(1, 13)],
                     columns=["year", "month", "value"])
    T = pd.DataFrame([(y, m, rng.normal(15, 5)) for y in myrs for m in range(1, 13)],
                     columns=["year", "month", "value"])
    U, meta = C._seasonalize(C._to_month3(P, "P"), 9, (1, 3, 6, 12), "sum")
    jun3 = [i for i, m in enumerate(meta)
            if m["season_length"] == 3 and m["end_month"] == 6 and not m["prior"]][0]
    sig = U[list(U.columns)[jun3]].reindex(yrs)
    sig = (sig - sig.mean()) / sig.std()
    noise = 0.6 * rng.normal(0, 1, len(yrs))

    stable = pd.Series(1.0 + 0.8 * sig.to_numpy() + noise, index=yrs)
    stable.index.name = "year"
    rs = dpl.seascorr(stable, P, T, end_month=9, nsim=1000, seed=7, plot=False,
                      period=((1900, 1944), (1945, 1989)))
    # strongest 3-month season: stable relationship -> no significant difference
    row = rs.stability[rs.stability["season_length"] == 3].iloc[0]
    assert row["p_value"] > 0.05

    w = np.where(yrs < 1945, 0.0, 1.0)
    unstable = pd.Series(1.0 + 0.9 * w * sig.to_numpy() + noise, index=yrs)
    unstable.index.name = "year"
    ru = dpl.seascorr(unstable, P, T, end_month=9, nsim=1000, seed=7, plot=False,
                      period=((1900, 1944), (1945, 1989)))
    row = ru.stability[ru.stability["season_length"] == 3].iloc[0]
    assert row["p_value"] <= 0.05
    assert "n_eff_early" in ru.stability.columns


def test_stability_rejects_overlap():
    x, P, T = _make()
    with pytest.raises(ValueError, match="overlap"):
        dpl.seascorr(x, P, T, nsim=1000, period=((1952, 1985), (1980, 2010)),
                     plot=False)


# ------------------------------------------------------------- plot color ---

def test_color_classify():
    assert SP._classify("Temperature") == "red"
    assert SP._classify("tmean") == "red"
    assert SP._classify("Precipitation") == "blue"
    assert SP._classify("scPDSI") == "blue"
    assert SP._classify("PC1 score") is None


def test_color_resolve_override_and_default():
    assert SP._resolve_cmap("red", "anything", "blue", True) is SP._RED
    assert SP._resolve_cmap(None, "precip total", "red", True) is SP._BLUE
    assert SP._resolve_cmap(None, "mystery", "blue", True) is SP._BLUE   # default
    assert SP._resolve_cmap(None, "temp", "red", False) is SP._BW         # B/W
    with pytest.raises(ValueError):
        SP._resolve_cmap("green", "x", "blue", True)


def test_ylim_locks_then_expands():
    assert SP._nice_ylim(np.array([0.1, -0.2, 0.05])) == (-0.3, 0.3)
    lo, hi = SP._nice_ylim(np.array([0.615, -0.05]))
    assert lo == -0.3 and hi >= 0.65


# ----------------------------------------------------------- non-mutation ---

def test_inputs_not_mutated():
    x, P, T = _make()
    xs, Ps, Ts = x.copy(deep=True), P.copy(deep=True), T.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dpl.seascorr(x, P, T, nsim=1000, period=(1955, 2008), seed=0, plot=False)
    assert x.equals(xs) and P.equals(Ps) and T.equals(Ts)
