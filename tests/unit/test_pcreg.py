"""Tests for dpl.pcreg (OSR / principal-components regression reconstruction).

Covers the ported probability function, the validation statistics (checked
number-for-number against Ed Cook's FORTRAN "oracle" on the Khangai fixture),
the PCA truncation rules, the full pipeline on deterministic synthetic data (a
frozen snapshot), the optional AR-prewhitening path, guard rails, and
non-mutation of inputs.

The real validation fixture (a 310-chronology network + gridded PDSI) is far too
large to vendor into the test suite, so the exact-parity assertions here use the
small frozen arrays that Cook's FORTRAN printed for that run. The end-to-end
parity against the full network lives in dev/pcreg_DECISIONS.md.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import importlib

import dplpy as dpl

# The public function dplpy.pcreg shadows the submodule of the same name in the
# package namespace, so reach the module (and its private helpers) explicitly.
P = importlib.import_module("dplpy.pcreg")


# ------------------------------------------------------------ import paths ---

def test_import_paths():
    assert callable(dpl.pcreg)
    assert dpl.pcreg is P.pcreg
    assert dpl.PcregResult is P.PcregResult


# --------------------------------------------- _validation_stats statistics ---
# The 1959-1964 validation window Cook's FORTRAN printed for the Khangai
# single-target run (standardized actual target vs recon_std), rounded to the
# 3 d.p. of the FORTRAN log.

_KHANGAI_ACTUAL = np.array([0.118, 0.399, -0.271, -0.284, 0.068, 2.400])
_KHANGAI_EST = np.array([0.675, 1.014, 0.250, -0.243, 0.434, 0.657])


def test_validate_matches_cook_oracle():
    v = P._validation_stats(_KHANGAI_ACTUAL, _KHANGAI_EST, re_baseline=0.0)
    assert v["r"] == pytest.approx(0.442, abs=5e-4)
    assert v["RE"] == pytest.approx(0.321, abs=5e-4)
    assert v["CE"] == pytest.approx(0.191, abs=5e-4)


def test_validate_re_vs_ce_baselines():
    # RE uses the supplied climatology baseline; CE uses the actual's own mean.
    # A nonzero baseline must change RE but never CE.
    v0 = P._validation_stats(_KHANGAI_ACTUAL, _KHANGAI_EST, re_baseline=0.0)
    v1 = P._validation_stats(_KHANGAI_ACTUAL, _KHANGAI_EST, re_baseline=0.5)
    assert v0["RE"] != v1["RE"]
    assert v0["CE"] == pytest.approx(v1["CE"])


def test_validate_perfect_fit():
    v = P._validation_stats(_KHANGAI_ACTUAL, _KHANGAI_ACTUAL.copy(), re_baseline=0.0)
    assert v["r"] == pytest.approx(1.0)
    assert v["RE"] == pytest.approx(1.0)
    assert v["CE"] == pytest.approx(1.0)


def test_validate_full_panel():
    # The statistic VALUES reproduce Cook's oracle panel (r, rank correlations,
    # sign counts, product-means and equality-of-means differences); the
    # significance comes from SciPy's exact distributions, not Ed's
    # Wilson-Hilferty / normal approximations, so the p-values are the exact ones
    # we ship (one-sided for the correlations, as Cook's method intends).
    v = P._validation_stats(_KHANGAI_ACTUAL, _KHANGAI_EST, re_baseline=0.0)
    # statistic values -- match the oracle
    assert v["r"] == pytest.approx(0.442, abs=1e-3)
    assert v["spearman"] == pytest.approx(0.829, abs=1e-3)
    assert v["kendall"] == pytest.approx(0.733, abs=1e-3)
    assert v["sign_pos"] == 3 and v["sign_neg"] == 3
    assert v["pm_diff"] == pytest.approx(0.321, abs=1e-3)
    assert v["pm_t"] == pytest.approx(2.536, abs=2e-3)
    assert v["eqmeans_diff"] == pytest.approx(-0.060, abs=1e-3)
    assert v["eqmeans_t"] == pytest.approx(-0.133, abs=1e-3)
    # significance -- SciPy exact (one-sided correlations; exact binomial sign test)
    assert v["r_p"] == pytest.approx(0.19028, abs=1e-3)
    assert v["spearman_p"] == pytest.approx(0.02078, abs=1e-3)
    assert v["kendall_p"] == pytest.approx(0.02778, abs=1e-3)
    assert v["sign_p"] == pytest.approx(0.65625, abs=1e-3)
    assert v["pm_p"] == pytest.approx(0.03212, abs=1e-3)
    assert v["eqmeans_p"] == pytest.approx(0.89835, abs=1e-3)   # exact Welch
    # robust = biweight midcorrelation (the modern standard, not Cook's 0.905 robcor)
    assert v["robust"] == pytest.approx(0.8656, abs=1e-3)
    assert v["robust_p"] == pytest.approx(0.01294, abs=1e-3)


def test_bicor_matches_astropy_formula():
    # The de novo biweight midcorrelation equals astropy's, derived here directly
    # (midcov/sqrt(midvar*midvar) reduces to sum(dx*dy)/sqrt(sum dx^2 sum dy^2)).
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=40)
        y = 0.6 * x + 0.5 * rng.normal(size=40)
        M = np.median
        c = 9.0

        def dev(v):
            mad = M(np.abs(v - M(v)))
            u = (v - M(v)) / (c * mad)
            w = (np.abs(u) < 1).astype(float)
            return (v - M(v)) * w * (1 - u ** 2) ** 2
        dx, dy = dev(x), dev(y)
        ref = np.sum(dx * dy) / np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
        assert P._bicor(x, y) == pytest.approx(ref)
    # zero-spread guard
    assert np.isnan(P._bicor(np.ones(6), np.arange(6.0)))


def test_validate_rank_stats_are_one_sided():
    # A negatively-correlated estimate carries no positive skill, so the
    # one-sided (``greater``) correlation p-values sit near 1.0.
    rng = np.random.default_rng(5)
    a = rng.normal(size=12)
    e = -a + 0.1 * rng.normal(size=12)                 # strong negative, not exact
    v = P._validation_stats(a, e, re_baseline=0.0)
    assert v["r"] < 0
    assert v["r_p"] > 0.99
    assert v["spearman_p"] > 0.99
    assert v["kendall_p"] > 0.99


def test_validate_uses_scipy_exact():
    # The shipped p-values are SciPy's, not a hand-rolled approximation.
    from scipy import stats
    a, e = _KHANGAI_ACTUAL, _KHANGAI_EST
    v = P._validation_stats(a, e, re_baseline=0.0)
    assert v["r_p"] == pytest.approx(stats.pearsonr(a, e, alternative="greater").pvalue)
    assert v["kendall_p"] == pytest.approx(stats.kendalltau(a, e, alternative="greater").pvalue)
    assert v["eqmeans_p"] == pytest.approx(stats.ttest_ind(a, e, equal_var=False).pvalue)


def test_validate_panel_flows_into_result():
    predictors, target = _synthetic()
    res = dpl.pcreg(predictors, target, calibration=(1950, 1999),
                    validation=(1900, 1949), plot=False)
    for k in ("spearman", "kendall", "robust", "sign_pos", "pm_diff",
              "eqmeans_diff", "spearman_diff", "kendall_diff", "robust_diff"):
        assert k in res.stats


# ------------------------------------------------------------ _pca rules -----

def _spd_block(rng, n_obs, n_var):
    """A small observation matrix with real correlation structure."""
    f = rng.normal(size=(n_obs, 3))                    # 3 latent factors
    load = rng.normal(size=(3, n_var))
    return f @ load + 0.3 * rng.normal(size=(n_obs, n_var))


def test_pca_descending_and_counts():
    rng = np.random.default_rng(1)
    A = _spd_block(rng, 40, 8)
    A = (A - A.mean(0)) / A.std(0, ddof=0)
    out = P._pca(A, "all", 0, 1.0, 90.0)
    w = out["eigvals"]
    assert np.all(np.diff(w) <= 1e-9)                  # strictly descending
    assert out["neig"] == min(A.shape)                 # 'all' keeps every PC
    # first_m keeps exactly m
    assert P._pca(A, "first_m", 3, 1.0, 90.0)["neig"] == 3
    # kaiser keeps the PCs with rescaled eigenvalue > cutoff
    k = P._pca(A, "kaiser", 0, 1.0, 90.0)
    assert 1 <= k["neig"] <= 8


def test_pca_eigvec_sign_deterministic():
    rng = np.random.default_rng(2)
    A = _spd_block(rng, 30, 5)
    A = (A - A.mean(0)) / A.std(0, ddof=0)
    V1 = P._pca(A, "all", 0, 1.0, 90.0)["eigvecs"]
    V2 = P._pca(A, "all", 0, 1.0, 90.0)["eigvecs"]
    np.testing.assert_allclose(V1, V2)                 # reproducible sign choice
    for j in range(V1.shape[1]):
        k = np.argmax(np.abs(V1[:, j]))
        assert V1[k, j] >= 0                           # largest loading positive


# ------------------------------------------- full pipeline (synthetic) -------

def _synthetic(seed=7, n_pred=20, first=1900, last=1999):
    """Deterministic predictors + target with a real shared signal."""
    rng = np.random.default_rng(seed)
    years = np.arange(first, last + 1)
    signal = rng.normal(size=years.size)
    # half the predictors carry the signal, half are noise
    cols = {}
    for i in range(n_pred):
        if i < n_pred // 2:
            x = 0.7 * signal + 0.7 * rng.normal(size=years.size)
        else:
            x = rng.normal(size=years.size)
        cols["P%02d" % i] = x
    predictors = pd.DataFrame(cols, index=years)
    predictors.index.name = "year"
    target = pd.Series(1.5 * signal + 0.5 * rng.normal(size=years.size),
                       index=years, name="clim")
    return predictors, target


def test_pipeline_runs_and_is_frozen():
    predictors, target = _synthetic()
    res = dpl.pcreg(predictors, target, calibration=(1950, 1999),
                    validation=(1900, 1949), pc_criterion="pvp", plot=False)
    assert isinstance(res, dpl.PcregResult)
    # signal-bearing predictors should dominate those retained
    signal_kept = [c for c in res.predictors_screened if int(c[1:]) < 10]
    assert len(signal_kept) >= len(res.predictors_screened) / 2
    # reconstruction spans the full predictor period and is finite
    assert res.reconstruction.index.min() == 1900
    assert res.reconstruction.index.max() == 1999
    assert np.isfinite(res.reconstruction.to_numpy()).all()
    # a real climate signal -> positive calibration skill and RE
    assert res.cal_r2 > 0.3
    assert res.stats["RE"] > 0.0
    # weights are one per screened predictor
    assert list(res.weights.index) == list(res.predictors_screened)


def test_pipeline_reconstruction_rescaled_to_target_units():
    # The variance-rescaled reconstruction should sit on the target's own scale
    # (calibration mean/SD), not the standardized z-scale.
    predictors, target = _synthetic()
    res = dpl.pcreg(predictors, target, calibration=(1950, 1999), plot=False)
    cal = np.arange(1950, 2000)
    rec_cal = res.reconstruction.reindex(cal)
    assert rec_cal.mean() == pytest.approx(target.reindex(cal).mean(), abs=1e-6)
    assert rec_cal.std(ddof=0) == pytest.approx(target.reindex(cal).std(ddof=0), abs=1e-6)


def test_prewhiten_path_runs():
    predictors, target = _synthetic()
    res = dpl.pcreg(predictors, target, calibration=(1950, 1999),
                    validation=(1900, 1949), prewhiten=True,
                    ar_method="burg", plot=False)
    assert isinstance(res, dpl.PcregResult)
    assert res.reconstruction.notna().any()


def test_force_all_screen_keeps_everything():
    predictors, target = _synthetic()
    res = dpl.pcreg(predictors, target, calibration=(1950, 1999),
                    screen_alpha=1.0, screen_tails=2, plot=False)
    assert len(res.predictors_screened) == predictors.shape[1]


# --------------------------------------------------------------- guards ------

def test_guard_bad_tails():
    predictors, target = _synthetic()
    with pytest.raises(ValueError):
        dpl.pcreg(predictors, target, calibration=(1950, 1999), screen_tails=3, plot=False)


def test_guard_one_tailed_needs_sign():
    predictors, target = _synthetic()
    with pytest.raises(ValueError):
        dpl.pcreg(predictors, target, calibration=(1950, 1999),
                  screen_tails=1, screen_sign=None, plot=False)


def test_guard_alpha_range():
    predictors, target = _synthetic()
    with pytest.raises(ValueError):
        dpl.pcreg(predictors, target, calibration=(1950, 1999), screen_alpha=0.0, plot=False)


def test_guard_calibration_too_short():
    predictors, target = _synthetic()
    with pytest.raises(ValueError):
        dpl.pcreg(predictors, target, calibration=(1950, 1955), plot=False)


def test_guard_empty_screen_raises():
    # a target uncorrelated with every predictor -> nothing survives a strict screen
    rng = np.random.default_rng(99)
    years = np.arange(1900, 2000)
    predictors = pd.DataFrame(
        {"P%02d" % i: rng.normal(size=years.size) for i in range(10)}, index=years)
    predictors.index.name = "year"
    target = pd.Series(rng.normal(size=years.size), index=years)
    with pytest.raises(ValueError):
        dpl.pcreg(predictors, target, calibration=(1950, 1999),
                  screen_alpha=0.001, plot=False)


# --------------------------------------------------------- lead/lag masks ---

def test_parse_lag_mask_and_offsets():
    # Cook's 11-char mask: position jj (0-based) -> offset jj-5.
    assert P._parse_lag("00000110000") == [0, 1]       # t0 and t+1
    assert P._parse_lag("00000100000") == [0]          # contemporaneous only
    assert P._parse_lag("10000000001") == [-5, 5]      # extreme lag and lead
    assert P._parse_lag(None) is None
    assert P._parse_lag((1, 0, -1)) == [-1, 0, 1]      # sequence, sorted/deduped
    assert P._parse_lag([0, 0, 2]) == [0, 2]


def test_parse_lag_guards():
    with pytest.raises(ValueError):
        P._parse_lag("0110")                           # wrong length
    with pytest.raises(ValueError):
        P._parse_lag((0, 6))                           # offset out of -5..+5
    with pytest.raises(ValueError):
        P._parse_lag("00000000000")                    # selects nothing


def test_apply_lag_alignment():
    # The @t+d copy at year Y must hold the predictor at Y+d.
    years = np.arange(2000, 2011)
    P1 = pd.DataFrame({"A": np.arange(11.0)}, index=years)   # A(Y) = Y-2000
    out = P._apply_lag(P1, [-1, 0, 1])
    assert set(out.columns) == {"A@t-1", "A@t+0", "A@t+1"}
    # at 2005: t0 -> A(2005)=5, t-1 -> A(2004)=4, t+1 -> A(2006)=6
    assert out.loc[2005, "A@t+0"] == 5
    assert out.loc[2005, "A@t-1"] == 4
    assert out.loc[2005, "A@t+1"] == 6
    # ends go NaN where the shift runs off the data
    assert np.isnan(out.loc[2010, "A@t+1"])
    assert np.isnan(out.loc[2000, "A@t-1"])


def test_lag_expands_predictor_pool():
    # Predictors run 1900-2010 so the +1 lead has coverage inside a 1950-1990 cal.
    predictors, target = _synthetic(n_pred=12, first=1900, last=2010)
    base = dpl.pcreg(predictors, target, calibration=(1950, 1990),
                     screen_alpha=1.0, screen_tails=2, plot=False)
    lagged = dpl.pcreg(predictors, target, calibration=(1950, 1990),
                       screen_alpha=1.0, screen_tails=2, lag=(0, 1), plot=False)
    # two offsets -> twice as many candidate predictors enter
    assert len(lagged.predictors_screened) == 2 * len(base.predictors_screened)
    assert all("@t" in c for c in lagged.predictors_screened)
    assert lagged.settings["lag"] == [0, 1]


# ----------------------------------------------------- cross-validation ----

def test_cv_loo_runs_and_reports():
    predictors, target = _synthetic()
    res = dpl.pcreg(predictors, target, calibration=(1950, 1999),
                    validation=(1900, 1949), cv=1, plot=False)
    assert res.cv_reconstruction is not None
    assert res.cv_reconstruction.index.min() == 1950
    assert res.cv_reconstruction.index.max() == 1999
    for k in ("cv_r", "cv_RE", "cv_CE", "cv_n", "cv_k"):
        assert k in res.stats
    assert res.stats["cv_k"] == 1
    assert res.stats["cv_n"] == 50


def test_cv_detects_real_skill_and_overfit():
    # Genuine shared signal -> positive, high LOO skill, a touch below cal R^2.
    predictors, target = _synthetic()
    good = dpl.pcreg(predictors, target, calibration=(1950, 1999), cv=1, plot=False)
    assert good.stats["cv_RE"] > 0.5
    assert good.stats["cv_r"] > 0.7
    assert good.stats["cv_RE"] <= good.cal_r2 + 1e-9

    # Pure noise predictors -> LOO must expose the lack of skill (RE < 0).
    rng = np.random.default_rng(3)
    years = np.arange(1900, 1980)
    noise = pd.DataFrame({"N%02d" % i: rng.normal(size=years.size) for i in range(25)},
                         index=years)
    noise.index.name = "year"
    tgt = pd.Series(rng.normal(size=years.size), index=years)
    ov = dpl.pcreg(noise, tgt, calibration=(1940, 1979), screen_alpha=1.0, cv=1, plot=False)
    assert ov.stats["cv_RE"] < 0.0


def test_cv_leave_k_out_matches_manual():
    # The driver's held-out predictions must equal an independent refit.
    predictors, target = _synthetic()
    res = dpl.pcreg(predictors, target, calibration=(1950, 1999), cv=1, plot=False)
    Pf = P._as_year_frame(predictors)
    yf = P._as_year_series(target)
    cal = np.arange(1950, 2000)
    for hold in (1950, 1973, 1999):
        train = cal[cal != hold]
        sf = dict(res.settings)
        sf["min_coverage"] = len(train)
        fit = P._core_fit(Pf, yf, train, sf, recon_years=[hold], strict=False)
        manual = fit["recon_scaled"].reindex([hold]).iloc[0]
        assert res.cv_reconstruction.loc[hold] == pytest.approx(manual, abs=1e-9)


def test_cv_guards():
    predictors, target = _synthetic()
    with pytest.raises(ValueError):
        dpl.pcreg(predictors, target, calibration=(1950, 1999), cv=0, plot=False)
    with pytest.raises(ValueError):
        # cv >= calibration length is impossible
        dpl.pcreg(predictors, target, calibration=(1950, 1959), cv=50, plot=False)


# ------------------------------------------- coverage-drop reporting ---------

def test_coverage_drops_are_reported():
    # Predictors run only to 2000; a +1 lead over a 1960-2000 cal pushes every
    # lagged column off the end, so each @t+1 is dropped and must be reported.
    predictors, target = _synthetic(first=1900, last=2000)
    with pytest.warns(UserWarning, match="insufficient calibration coverage"):
        res = dpl.pcreg(predictors, target, calibration=(1960, 2000),
                        screen_alpha=1.0, lag=(0, 1), plot=False)
    assert len(res.dropped_coverage) > 0
    assert all(c.endswith("@t+1") for c in res.dropped_coverage)   # the lead cols
    assert all("@t+1" not in k for k in res.predictors_screened)   # none survived


def test_no_coverage_drops_no_warning():
    # Full coverage, no lag -> nothing dropped, no coverage warning.
    predictors, target = _synthetic()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # any coverage warning fails
        res = dpl.pcreg(predictors, target, calibration=(1950, 1999), plot=False)
    assert res.dropped_coverage == []


def test_cv_does_not_repeat_coverage_warning():
    # The message is for the main fit; folds must not each re-emit it.
    predictors, target = _synthetic(first=1900, last=2000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dpl.pcreg(predictors, target, calibration=(1960, 2000),
                  screen_alpha=1.0, lag=(0, 1), cv=1, plot=False)
    cov = [x for x in w if "insufficient calibration coverage" in str(x.message)]
    assert len(cov) == 1


# ---------------------------------------------------------- non-mutation -----

def test_inputs_not_mutated():
    # Predictors run past the calibration so the +1 lead keeps coverage (no
    # incidental coverage warning); this test is about input mutation only.
    predictors, target = _synthetic(first=1900, last=2010)
    p0 = predictors.copy()
    t0 = target.copy()
    dpl.pcreg(predictors, target, calibration=(1950, 1990),
              validation=(1910, 1949), lag=(0, 1), cv=1, plot=False)
    pd.testing.assert_frame_equal(predictors, p0)
    pd.testing.assert_series_equal(target, t0)
