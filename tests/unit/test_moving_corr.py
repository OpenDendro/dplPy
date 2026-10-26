"""Unit tests for dpl.moving_corr (moving correlation with significance).

Self-contained: the synthetic chronology + monthly climate are regenerated
deterministically here, so the test needs no data files. The deterministic
quantities (the moving-correlation trace, the overall correlation, the lag-1
autocorrelations, the season selection) are checked exactly against embedded
golden-master values and, for one window, against a plain pandas computation.
The Monte-Carlo significance is stochastic, so it is checked behaviourally with
fixed seeds and generous margins: a built-in regime change is flagged by all
three tests, and a genuinely stable (constant) relationship is flagged by none.

The significance engine is a port of K. Anchukaitis's moving_corr_sig (validated
against Gershunov et al. 2001, Table 1); that cross-language validation lives in
dev/moving_correlation/.
"""
import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


# --------------------------------------------------------------------------- #
# Deterministic synthetic data
# --------------------------------------------------------------------------- #

def _synthetic(regime=True):
    """A chronology and a monthly climate frame with a known Jun-Aug signal.

    ``regime=True``  : the chronology tracks summer climate strongly in the first
                       half of the record and weakly in the second (a real change
                       in the climate signal).
    ``regime=False`` : a constant, moderate relationship throughout (stable).
    """
    rng = np.random.default_rng(20260601 if regime else 20260602)
    years = np.arange(1901, 2001)
    n = years.size
    jja = rng.standard_normal(n)                      # the summer climate index
    rows = []
    for i, Y in enumerate(years):
        for m in range(1, 13):
            v = rng.standard_normal()
            if m in (6, 7, 8):                        # inject the summer signal
                v += jja[i]
            rows.append((Y, m, v))
    clim = pd.DataFrame(rows, columns=["year", "month", "value"])
    if regime:
        s = np.concatenate([np.full(50, 0.85), np.full(50, 0.10)])
    else:
        s = np.full(n, 0.45)
    chron = pd.Series(s * jja + np.sqrt(1 - s ** 2) * rng.standard_normal(n),
                      index=years)
    return chron, clim


# embedded golden masters for the regime synthetic, JJA (len 3, end month 8),
# window 25, offset 1 (deterministic: pure numpy/pandas, version-stable)
_GM_LABEL = "Jun-Aug"
_GM_PERIOD = (1901, 2000)
_GM_NWIN = 76
_GM_OVERALL = 0.386518
_GM_R1X = -0.134476
_GM_R1C = 0.056289
_GM_TRACE_FIRST3 = [0.784447, 0.776775, 0.75143]
_GM_TRACE_LAST3 = [0.148743, 0.085379, 0.160786]
_GM_TRACE_MAX = 0.816664
_GM_TRACE_MIN = -0.040146
_GM_POS_FIRST, _GM_POS_LAST = 1913, 1988


# --------------------------------------------------------------------------- #
# Deterministic kernel
# --------------------------------------------------------------------------- #

def test_season_selection_and_period():
    chron, clim = _synthetic()
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          offset=1, iters=50, seed=1, tests=(), ci="none")
    assert res.label == _GM_LABEL
    assert res.period == _GM_PERIOD
    assert len(res.trace) == _GM_NWIN
    assert res.trace.index[0] == _GM_POS_FIRST
    assert res.trace.index[-1] == _GM_POS_LAST


def test_trace_matches_golden_and_pandas():
    chron, clim = _synthetic()
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          offset=1, iters=50, seed=1, tests=(), ci="none")
    r = res.trace["r"].to_numpy()
    assert res.overall_corr == pytest.approx(_GM_OVERALL, abs=1e-5)
    assert res.r1_chronology == pytest.approx(_GM_R1X, abs=1e-5)
    assert res.r1_climate == pytest.approx(_GM_R1C, abs=1e-5)
    np.testing.assert_allclose(r[:3], _GM_TRACE_FIRST3, atol=1e-5)
    np.testing.assert_allclose(r[-3:], _GM_TRACE_LAST3, atol=1e-5)
    assert float(np.nanmax(r)) == pytest.approx(_GM_TRACE_MAX, abs=1e-5)
    assert float(np.nanmin(r)) == pytest.approx(_GM_TRACE_MIN, abs=1e-5)

    # independent single-window check against a plain pandas correlation
    xa = chron.to_numpy()
    jja_mean = (clim[clim.month.isin([6, 7, 8])]
                .groupby("year")["value"].mean()
                .reindex(np.arange(1901, 2001)).to_numpy())
    w0 = np.corrcoef(xa[0:25], jja_mean[0:25])[0, 1]
    assert r[0] == pytest.approx(w0, abs=1e-9)


def test_offset_controls_window_count():
    chron, clim = _synthetic()
    base = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                           offset=1, iters=50, seed=1, tests=(), ci="none")
    full_r = base.trace["r"].to_numpy()
    for off in (2, 5, 25):
        res = dpl.moving_corr(chron, clim, end_month=8, season_length=3,
                              window=25, offset=off, iters=50, seed=1,
                              tests=(), ci="none")
        # the offset trace is exactly the full trace subsampled every `off`
        np.testing.assert_allclose(res.trace["r"].to_numpy(), full_r[::off],
                                   atol=1e-12)


def test_position_conventions():
    chron, clim = _synthetic()
    first = {}
    for pos in ("center", "end", "start"):
        res = dpl.moving_corr(chron, clim, end_month=8, season_length=3,
                              window=25, offset=1, position=pos, iters=50,
                              seed=1, tests=(), ci="none")
        row = res.trace.iloc[0]
        first[pos] = (int(row["start"]), int(row["end"]), int(res.trace.index[0]))
    assert first["start"][2] == first["start"][0]          # plotted at start year
    assert first["end"][2] == first["end"][1]              # plotted at end year
    assert first["center"][2] == first["start"][0] + 12    # (25-1)//2 = 12


# --------------------------------------------------------------------------- #
# Confidence band
# --------------------------------------------------------------------------- #

def test_fisher_ci_present_and_ordered():
    chron, clim = _synthetic()
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          ci="fisher", ci_level=0.95, iters=50, seed=1, tests=())
    tr = res.trace
    assert {"ci_lower", "ci_upper"}.issubset(tr.columns)
    assert (tr["ci_lower"] <= tr["r"] + 1e-9).all()
    assert (tr["ci_upper"] >= tr["r"] - 1e-9).all()


def test_bootstrap_ci_present_and_ordered():
    chron, clim = _synthetic()
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          offset=5, ci="bootstrap", n_boot=200, iters=50, seed=1,
                          tests=())
    tr = res.trace
    assert {"ci_lower", "ci_upper"}.issubset(tr.columns)
    assert (tr["ci_lower"] <= tr["r"] + 1e-9).all()
    assert (tr["ci_upper"] >= tr["r"] - 1e-9).all()


def test_ci_none_has_no_band():
    chron, clim = _synthetic()
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          ci="none", iters=50, seed=1, tests=())
    assert "ci_lower" not in res.trace.columns


# --------------------------------------------------------------------------- #
# Significance (behavioural, fixed seeds, generous margins)
# --------------------------------------------------------------------------- #

def test_regime_change_is_flagged():
    chron, clim = _synthetic(regime=True)
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          offset=1, noise="red", iters=500, seed=7)
    assert res.tests["std"].p_value < 0.05
    assert res.tests["peak"].p_value < 0.05
    assert res.tests["range"].p_value < 0.05


def test_stable_relationship_not_flagged():
    chron, clim = _synthetic(regime=False)
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          offset=1, noise="red", iters=500, seed=7)
    # a genuinely constant relationship wobbles only as much as red noise does
    assert res.tests["std"].p_value > 0.10
    assert res.tests["peak"].p_value > 0.10


def test_thresholds_and_exceedance_logic():
    chron, clim = _synthetic(regime=True)
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          noise="red", iters=500, seed=7)
    up = res.threshold(0.95)
    lo = res.threshold_lower(0.95)
    assert -1 < lo < up < 1
    # the strong early peaks exceed the upper line
    assert (res.trace["r"] > up).any()


def test_reproducible_with_seed():
    chron, clim = _synthetic()
    a = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                        iters=200, seed=42, ci="none")
    b = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                        iters=200, seed=42, ci="none")
    assert a.tests["std"].p_value == b.tests["std"].p_value
    assert a.tests["peak"].p_value == b.tests["peak"].p_value
    np.testing.assert_array_equal(a.trace["r"].to_numpy(), b.trace["r"].to_numpy())


def test_white_vs_red_both_run():
    chron, clim = _synthetic()
    for noise in ("white", "red"):
        res = dpl.moving_corr(chron, clim, end_month=8, season_length=3,
                              window=25, noise=noise, iters=100, seed=1)
        assert 0.0 <= res.tests["std"].p_value <= 1.0


def test_peak_condition_zero():
    chron, clim = _synthetic()
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          peak_condition="zero", iters=200, seed=1)
    assert res.tests["peak"].target_corr == 0.0


# --------------------------------------------------------------------------- #
# Input handling / validation
# --------------------------------------------------------------------------- #

def test_does_not_mutate_inputs():
    chron, clim = _synthetic()
    chron_copy, clim_copy = chron.copy(), clim.copy()
    dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                    iters=20, seed=1)
    assert chron.equals(chron_copy)
    assert clim.equals(clim_copy)


def test_tests_subset_selectable():
    chron, clim = _synthetic()
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          tests=("std",), iters=100, seed=1)
    assert set(res.tests) == {"std"}


@pytest.mark.parametrize("kwargs,msg", [
    (dict(window=5), "window must be >= 10"),
    (dict(window=500), "shorter than the aligned period"),
    (dict(offset=0), "offset must be >= 1"),
    (dict(end_month=13), "end_month must be 1..12"),
    (dict(method="kendall"), "method must be"),
    (dict(noise="pink"), "noise must be"),
    (dict(ci="loess"), "ci must be"),
    (dict(peak_condition="maybe"), "peak_condition must be"),
    (dict(tests=("bogus",)), "unknown test"),
])
def test_input_validation(kwargs, msg):
    chron, clim = _synthetic()
    base = dict(end_month=8, season_length=3, window=25, iters=20, seed=1)
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        dpl.moving_corr(chron, clim, **base)


def test_period_restriction():
    chron, clim = _synthetic()
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          period=(1921, 1980), iters=50, seed=1, tests=(), ci="none")
    assert res.period[0] >= 1921 and res.period[1] <= 1980


def test_result_repr_and_summary():
    chron, clim = _synthetic()
    res = dpl.moving_corr(chron, clim, end_month=8, season_length=3, window=25,
                          iters=100, seed=1)
    assert "MovingCorrResult" in repr(res)
    assert "Moving correlation" in res.summary()
