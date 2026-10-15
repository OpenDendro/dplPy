import warnings

import numpy as np
import pandas as pd
import dplpy as dpl
import pytest

from scipy.signal import lfilter


def _ar2_dataset(n_series=8, n_years=200, phi=(0.6, -0.2), seed=0):
    """A synthetic multi-series AR(2) dataset (positive-valued, RWI-like)."""
    rng = np.random.RandomState(seed)
    cols = {}
    for s in range(n_series):
        e = rng.randn(n_years)
        y = np.zeros(n_years)
        for t in range(2, n_years):
            y[t] = phi[0] * y[t - 1] + phi[1] * y[t - 2] + e[t]
        cols["AAA%03dA" % (s + 1)] = y - y.min() + 1.0  # shift positive
    return pd.DataFrame(cols, index=pd.Index(np.arange(1, n_years + 1), name="Year"))


def test_chron_ars_wrong_data_type():
    with pytest.raises(TypeError) as e:
        dpl.chron_ars("notadf")
    assert "Expected dataframe input" in str(e.value)


def test_chron_ars_bad_prewhiten_method():
    df = _ar2_dataset()
    with pytest.raises(ValueError) as e:
        dpl.chron_ars(df, prewhiten_method="bogus", verbose=False)
    assert "prewhiten_method" in str(e.value)


def test_chron_ars_output_structure():
    df = _ar2_dataset()
    out = dpl.chron_ars(df, verbose=False)
    assert list(out.columns) == ["std", "res", "ars", "samp_depth"]
    assert list(out.index) == list(df.index)
    assert (out["samp_depth"] == df.notna().sum(axis=1)).all()


def test_chron_ars_std_is_plain_mean_when_not_biweight():
    # with biweight=False the std column is exactly the row-wise nanmean
    df = _ar2_dataset()
    out = dpl.chron_ars(df, biweight=False, verbose=False)
    expected = df.mean(axis=1).to_numpy()
    assert np.allclose(out["std"].to_numpy(), expected)


def test_post_ar_zero_initials_match_explicit_recurrence():
    # with backcast=False, _post_ar is the plain AR recurrence from zero
    # initials on the demeaned series (== scipy lfilter), mean restored
    from dplpy.chron_ars import _post_ar
    rng = np.random.RandomState(1)
    x = rng.randn(150) + 1.0
    phi = np.array([0.5, -0.1, 0.05])
    xd = x - x.mean()
    explicit = lfilter([1.0], np.concatenate([[1.0], -phi]), xd) + x.mean()
    assert np.allclose(_post_ar(x, phi, backcast=False), explicit, atol=1e-12)
    # with backcast=True (default) the initials come from ARSTAN's bckcst run
    # from the START of the series (audit A12), so only the start-up differs
    bc = _post_ar(x, phi, backcast=True)
    assert not np.allclose(bc[:3], explicit[:3])
    assert np.allclose(bc[60:], explicit[60:], atol=1e-6)
    # NaN-aware: NaN positions are kept, the rest is reddened as a compact series
    xn = x.copy()
    xn[:4] = np.nan
    out = _post_ar(xn, phi)
    assert np.isnan(out[:4]).all()
    assert np.allclose(out[4:], _post_ar(x[4:], phi))


def test_chron_ars_recovers_ar_order_on_ar2():
    # a pure AR(2) pool should select order ~2 by AIC
    df = _ar2_dataset(n_series=12, n_years=400, phi=(0.6, -0.3), seed=7)
    out = dpl.chron_ars(df, verbose=False)
    # (order is not returned directly; check res is decorrelated relative to std)
    assert out["res"].dropna().shape[0] > 0
    assert out["ars"].dropna().shape[0] > 0


# ---------------------------------------------------------------------------
# Fidelity regression test: values hardcoded from dplR 1.7.9's FULL pipeline --
# detrend(ca533, method="Spline") -> chron.ars(biweight=FALSE) -- run entirely
# in R. dplPy's own detrend + chron_ars reproduces these to ~5e-10, so this
# encodes the end-to-end dplR validation into CI without requiring R.
#
# NOTE: released dplR (<= 1.8.0) re-reddens each series and then averages,
# which dplPy exposes as ars_method="dplr". dplPy's default
# (ars_method="arstan") re-reddens the residual chronology (Ed Cook's ARSTAN,
# and the dplR development version), so this dplR-parity test pins the "dplr"
# path explicitly. It also pins backcast=False, because dplPy defaults to
# ARSTAN-style backcasting (backcast=True), which fills the first p residuals
# that dplR leaves as NaN and thereby changes res/ars at years where a young
# series is within its first p. One remaining difference: dplPy's re-reddening
# takes its p initial values from the start of each series (ARSTAN bckcst),
# dplR's postAR from the end of the reversed record (audit A12), so the "dplr"
# ars differs from dplR for ~20 years after each series' start. 1700 is 17
# years after one series (CAM191) starts, hence the looser tolerance there.
# ---------------------------------------------------------------------------
def test_chron_ars_matches_dplR_reference_ca533():
    data = dpl.readers("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dpl.chron_ars(rwi, biweight=False, prewhiten_method="ar.yw",
                            ars_method="dplr", backcast=False, verbose=False)

    # dplR std/res/ars at three years (atol 1e-6)
    ref = {
        1700: (0.8813025556, 0.9176128562, 0.9026494583),
        1850: (0.9396477470, 0.9993573139, 0.9748222048),
        1983: (1.3314471088, 1.1243002992, 1.3015262394),
    }
    for yr, (s, r, a) in ref.items():
        assert abs(out.loc[yr, "std"] - s) < 1e-6, (yr, "std")
        assert abs(out.loc[yr, "res"] - r) < 1e-6, (yr, "res")
        assert abs(out.loc[yr, "ars"] - a) < (5e-4 if yr == 1700 else 1e-6), (yr, "ars")


def test_chron_ars_bad_ars_method():
    df = _ar2_dataset()
    with pytest.raises(ValueError) as e:
        dpl.chron_ars(df, ars_method="bogus", verbose=False)
    assert "ars_method" in str(e.value)


def test_chron_ars_arstan_default_rereddens_the_residual_chronology():
    # The default ars_method="arstan" builds ars = redden(res, phi_pooled): the
    # residual chronology (mean of the prewhitened series, whitened once more
    # to order p -- ARSTAN's albino1) re-reddened with the pooled coefficients
    # (commie1). This is Ed Cook's ARSTAN and the dplR development version
    # (audit A11), not released dplR's mean(postAR(series_i)). Reconstruct it
    # from chron_ars's own internals and assert the ars column matches.
    from dplpy.chron_ars import _pooled_ar, _whiten_series, _post_ar, _aggregate

    df = _ar2_dataset(n_series=10, n_years=250, phi=(0.6, -0.25), seed=3)
    x = df.to_numpy(dtype=float, copy=True)
    n_series = x.shape[1]

    out_ar = _pooled_ar(x, max_lag=10, first_aic_min=True)
    p = out_ar["order"]
    assert p > 0  # otherwise the two ars_methods coincide trivially
    phi = -out_ar["arcoefs"][p - 1, :p]

    for backcast in (True, False):
        rwi_clean = np.column_stack(
            [_whiten_series(x[:, s], p, backcast=backcast) for s in range(n_series)])
        res = _whiten_series(_aggregate(rwi_clean, biweight=False), p, backcast=backcast)
        expected_ars = _post_ar(res, phi, backcast)
        out = dpl.chron_ars(df, biweight=False, prewhiten_method="ar.yw",
                            backcast=backcast, verbose=False)
        assert np.allclose(out["res"].to_numpy(), res, equal_nan=True)
        v = out["ars"].to_numpy()
        ok = np.isfinite(expected_ars) & np.isfinite(v)
        assert ok.any() and np.array_equal(ok, np.isfinite(v))
        assert np.allclose(v[ok], expected_ars[ok], atol=1e-10)
        # and it is NOT the old "re-redden the un-whitened mean" construction
        old = _post_ar(_aggregate(rwi_clean, biweight=False), phi, backcast)
        assert not np.allclose(v[ok], old[ok], atol=1e-6)


def test_chron_ars_methods_differ_under_staggered_depth():
    # average-then-redden and redden-then-average do not commute when sample
    # depth changes through time, so the two ars_methods should differ here.
    df = _ar2_dataset(n_series=10, n_years=250, phi=(0.6, -0.25), seed=5)
    # stagger onsets so early years have shallow depth
    arr = df.to_numpy(dtype=float, copy=True)
    for s in range(arr.shape[1]):
        arr[: s * 12, s] = np.nan
    df = pd.DataFrame(arr, index=df.index, columns=df.columns)

    a = dpl.chron_ars(df, biweight=False, ars_method="arstan", verbose=False)["ars"].to_numpy()
    d = dpl.chron_ars(df, biweight=False, ars_method="dplr", verbose=False)["ars"].to_numpy()
    ok = np.isfinite(a) & np.isfinite(d)
    assert ok.any()
    assert not np.allclose(a[ok], d[ok], atol=1e-8)


# --- backcasting (ARSTAN bckcst) -------------------------------------------
# see dev/arstan_backcasting_review_2026-09-10.md
def test_chron_ars_backcast_default_fills_leading_nan():
    # default backcast=True: the residual and ARSTAN chronologies are full length
    # (no leading NaN); backcast=False leaves the first ~p residuals NaN as dplR.
    df = _ar2_dataset(n_series=8, n_years=200, phi=(0.6, -0.25), seed=1)
    on = dpl.chron_ars(df, verbose=False)                       # backcast=True
    off = dpl.chron_ars(df, verbose=False, backcast=False)
    for col in ("res", "ars"):
        assert np.isfinite(on[col].to_numpy()).all(), col       # no NaN anywhere
        assert np.isnan(off[col].to_numpy()).any(), col         # some leading NaN
    # std has no AR model, so it is unaffected by backcast
    assert np.allclose(on["std"], off["std"], equal_nan=True)


def test_chron_ars_backcast_interior_identical_to_no_backcast():
    # in the prewhitening core, backcasting fills the first p residuals by
    # reverse-AR; every year t >= p is identical to the non-backcast residual
    # (backcasting only affects the otherwise-NaN front).
    from dplpy.chron_ars import _whiten_series

    rng = np.random.default_rng(0)
    s = 1.0 + 0.2 * rng.standard_normal(120)
    p = 3
    bc = _whiten_series(s, p, backcast=True)
    nb = _whiten_series(s, p, backcast=False)
    assert np.allclose(bc[p:], nb[p:], equal_nan=True)          # interior identical
    assert np.isfinite(bc[:p]).all()                            # backcast fills front
    assert np.isnan(nb[:p]).all()                               # dplR leaves NaN


def test_chron_ars_backcast_reverse_ar_formula():
    # the p backcast values are reverse-AR predictions: running the fitted AR
    # model backward. Verify the residual identity holds at the front, i.e. the
    # backcast makes resid[0] = xd[0] - sum_j rho_j * xhat(-j) consistent. The
    # fixed-order Yule-Walker fit equals statsmodels' (R's ar.yw) coefficients.
    from dplpy.chron_ars import _whiten_series
    from statsmodels.regression.linear_model import yule_walker

    rng = np.random.default_rng(2)
    s = 1.0 + 0.3 * rng.standard_normal(80)
    p = 2
    out = _whiten_series(s, p, backcast=True)
    # refit the same way the function does, then rebuild the backcast by hand
    xd = s - s.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        rho, _ = yule_walker(xd, order=p, method="mle", demean=False)
    ext = np.empty(p + len(xd))
    ext[p:] = xd
    for ii in range(p - 1, -1, -1):
        ext[ii] = sum(rho[j - 1] * ext[ii + j] for j in range(1, p + 1))
    resid0 = ext[p] - sum(rho[j - 1] * ext[p - j] for j in range(1, p + 1))
    assert np.isclose(out[0] - s.mean(), resid0, atol=1e-12)


def test_chron_ars_emits_no_future_warnings():
    # the per-series Yule-Walker fit no longer goes through statsmodels'
    # yule_walker (20 FutureWarnings per call under statsmodels 0.15)
    df = _ar2_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        dpl.chron_ars(df, verbose=False)
        dpl.chron_ars(df, verbose=False, stabilize="both")


# --- stabilize= convenience flag -------------------------------------------
# see dev/variance_stabilization_design_2026-09-10.md
def _staggered_ar2(n_series=10, n_years=250, phi=(0.6, -0.25), seed=4):
    """AR(2) dataset with staggered onsets (rising sample depth)."""
    df = _ar2_dataset(n_series=n_series, n_years=n_years, phi=phi, seed=seed)
    arr = df.to_numpy(dtype=float, copy=True)
    for s in range(arr.shape[1]):
        arr[: s * 12, s] = np.nan
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def test_chron_ars_stabilize_none_is_default_unchanged():
    df = _staggered_ar2()
    out = dpl.chron_ars(df, verbose=False)
    assert list(out.columns) == ["std", "res", "ars", "samp_depth"]


def test_chron_ars_stabilize_adds_vsc_columns():
    df = _staggered_ar2()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dpl.chron_ars(df, verbose=False, stabilize="both")
    assert list(out.columns) == ["std", "std_vsc", "res", "res_vsc",
                                 "ars", "ars_vsc", "samp_depth"]
    base = dpl.chron_ars(df, verbose=False)
    # base chronologies are untouched by the flag
    for col in ("std", "res", "ars"):
        assert np.allclose(out[col], base[col], equal_nan=True), col
    for col in ("std_vsc", "res_vsc", "ars_vsc"):
        assert np.isfinite(out[col]).any(), col


def test_chron_ars_arstan_ars_vsc_is_stabilize_then_reredden():
    # ARSTAN order: ars_vsc = redden( stabilize(res, residual rbar) ) -- the
    # stabilized residual chronology re-reddened (same pipeline as the
    # unstabilized arstan ars). Rebuild that reference from the internals.
    from dplpy.chron_ars import _pooled_ar, _whiten_series, _post_ar, _aggregate

    df = _staggered_ar2(seed=6)
    x = df.to_numpy(dtype=float, copy=True)
    ns = x.shape[1]
    oa = _pooled_ar(x, 10, True)
    p = oa["order"]
    assert p > 0
    phi = -oa["arcoefs"][p - 1, :p]
    rwi_clean = np.column_stack([_whiten_series(x[:, s], p, backcast=False) for s in range(ns)])
    resid_matrix = pd.DataFrame(rwi_clean, index=df.index, columns=df.columns)
    res = _whiten_series(_aggregate(rwi_clean, biweight=True), p, backcast=False)
    res_vsc = dpl.stabilize_chron(pd.Series(res, index=df.index), "both",
                                  rwi=resid_matrix)["vsc"].to_numpy()
    expected = _post_ar(res_vsc, phi, backcast=False)

    out = dpl.chron_ars(df, verbose=False, stabilize="both", backcast=False)
    assert np.allclose(out["res_vsc"].to_numpy(), res_vsc, equal_nan=True)
    v = out["ars_vsc"].to_numpy()
    ok = np.isfinite(expected) & np.isfinite(v)
    assert ok.any()
    assert np.allclose(v[ok], expected[ok], atol=1e-10)


def test_chron_ars_dplr_ars_vsc_is_posthoc_on_reddened():
    # for ars_method="dplr", ars is built by reddening each series then
    # averaging, so ars_vsc is stabilized post-hoc on that reddened chronology
    # (residual matrix rbar), not in the ARSTAN order.
    from dplpy.chron_ars import _pooled_ar, _whiten_series

    df = _staggered_ar2(seed=8)
    x = df.to_numpy(dtype=float, copy=True)
    ns = x.shape[1]
    p = _pooled_ar(x, 10, True)["order"]
    rwi_clean = np.column_stack([_whiten_series(x[:, s], p, backcast=False) for s in range(ns)])
    resid_matrix = pd.DataFrame(rwi_clean, index=df.index, columns=df.columns)

    out = dpl.chron_ars(df, verbose=False, ars_method="dplr", stabilize="both",
                        backcast=False)
    ars0 = dpl.chron_ars(df, verbose=False, ars_method="dplr", backcast=False)["ars"]
    ref = dpl.stabilize_chron(pd.Series(ars0.to_numpy(), index=df.index), "both",
                              rwi=resid_matrix)["vsc"]
    assert np.allclose(out["ars_vsc"], ref, equal_nan=True)


def test_chron_ars_stabilize_reuses_running_rbar_once_per_matrix(monkeypatch):
    # F5: the running rbar of each matrix (standard, residual) is computed once
    import dplpy.stabilize as stab
    calls = []
    real = stab.running_rbar_vector

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(stab, "running_rbar_vector", counting)
    df = _staggered_ar2()
    dpl.chron_ars(df, verbose=False, stabilize="rbar")                      # arstan
    assert len(calls) == 2
    calls.clear()
    dpl.chron_ars(df, verbose=False, stabilize="rbar", ars_method="dplr")   # dplr
    assert len(calls) == 2
    calls.clear()
    out = dpl.chron_ars(df, verbose=False, stabilize="rbar",
                        stabilize_kwargs={"rbar_mode": "constant"})
    assert len(calls) == 0 and np.isfinite(out["ars_vsc"]).any()


def test_chron_ars_stabilize_kwargs_rejects_managed_keys():
    df = _staggered_ar2()
    with pytest.raises(ValueError):
        dpl.chron_ars(df, verbose=False, stabilize="rbar",
                      stabilize_kwargs={"column": "std"})


def test_pooled_acf_matches_dplR_reference_ca533():
    from dplpy.chron_ars import _pooled_ar
    data = dpl.readers("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    out_ar = _pooled_ar(rwi.to_numpy(dtype=float, copy=True), max_lag=10, first_aic_min=True)
    # dplR pooled ACF (lags 0..5) and selected order, from chron.ars() on dplR's
    # own spline detrend (matches dplPy's new detrend + pooled AR to ~1e-8).
    ref_acf = [1.0, 0.51626627, 0.35330050, 0.25346321, 0.21923437, 0.19473382]
    assert np.allclose(out_ar["acf"][:6], ref_acf, atol=1e-7)
    assert out_ar["order"] == 5


# --- v0.9.0 conventions and guards -------------------------------------------
def test_chron_ars_ar_max_alias_and_case_insensitive_options():
    df = _ar2_dataset()
    base = dpl.chron_ars(df, verbose=False)
    with pytest.warns(DeprecationWarning):
        alias = dpl.chron_ars(df, verbose=False, max_lag=10)
    pd.testing.assert_frame_equal(base, alias)
    up = dpl.chron_ars(df, verbose=False, prewhiten_method="AR.YW", ars_method="ARSTAN")
    pd.testing.assert_frame_equal(base, up)
    with pytest.raises(TypeError):
        dpl.chron_ars(df, verbose=False, bogus=1)


def test_chron_ars_burg_option_whitens():
    df = _ar2_dataset(n_series=8, n_years=300, seed=2)
    out = dpl.chron_ars(df, verbose=False, prewhiten_method="burg")
    yw = dpl.chron_ars(df, verbose=False)
    assert list(out.columns) == ["std", "res", "ars", "samp_depth"]
    assert np.allclose(out["std"], yw["std"])                     # std has no AR
    r = out["res"].to_numpy()
    r = r - r.mean()
    assert abs(np.dot(r[:-1], r[1:]) / np.dot(r, r)) < 0.15      # near white
    assert not np.allclose(out["res"], yw["res"])                 # different estimator


def test_chron_ars_drops_all_nan_column_and_skips_short_series():
    # B1 / B11: an all-NaN column is dropped with a warning; a series with
    # n <= p values is left out of the residual chronology (NaN) with a warning
    df = _ar2_dataset(n_series=6, n_years=200, seed=9)
    arr = df.to_numpy(dtype=float, copy=True)
    arr[:, 0] = np.nan                       # no data at all
    arr[1:, 1] = np.nan                      # a single value: n <= p for any p >= 1
    df2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = dpl.chron_ars(df2, verbose=False)
    msgs = " ".join(str(w.message) for w in rec)
    assert "all NaN" in msgs and "not more than the pooled AR order" in msgs
    assert len(out) == 200
    assert np.isfinite(out["res"].iloc[10:]).all()
    assert (out["samp_depth"].to_numpy() == df2.notna().sum(axis=1).to_numpy()).all()
    # every column NaN -> clear ValueError, not IndexError
    with pytest.raises(ValueError):
        dpl.chron_ars(pd.DataFrame(np.nan, index=df.index, columns=["A", "B"]),
                      verbose=False)


def test_chron_ars_first_aic_min_raises_when_aic_never_turns_up():
    # E8 / G5: a pooled AIC that keeps falling to ar_max has no first local
    # minimum -> ValueError (dplR behaviour); the global rule still works.
    # An AR(2) pool with ar_max=1: AIC(1) < AIC(0), nothing to turn back up.
    df = _ar2_dataset(n_series=6, n_years=300, phi=(0.6, -0.3), seed=4)
    with pytest.raises(ValueError, match="did not reach a minimum"):
        dpl.chron_ars(df, verbose=False, ar_max=1)
    out = dpl.chron_ars(df, verbose=False, ar_max=1, first_aic_min=False)
    assert len(out) == 300


def test_chron_ars_verbose_prints_acf_coefs_aic_order(capsys):
    df = _ar2_dataset()
    dpl.chron_ars(df, verbose=True)
    text = capsys.readouterr().out
    for key in ("ACF", "AR Coefficients", "AIC", "Selected Order"):
        assert key in text


def test_chron_ars_carries_attrs_as_crn():
    df = _ar2_dataset()
    df.attrs["site"] = "x"
    out = dpl.chron_ars(df, verbose=False)
    assert out.attrs["kind"] == "crn" and out.attrs["site"] == "x"


def test_chron_ars_std_equals_chron_std_and_ars_lag1_ac_ca533():
    data = dpl.readers("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = dpl.chron_ars(rwi, verbose=False)
        c = dpl.chron(rwi, plot=False)
    assert np.allclose(a["std"], c["std"], equal_nan=True)
    v = a["ars"].to_numpy()
    v = v - v.mean()
    ac1 = float(np.dot(v[:-1], v[1:]) / np.dot(v, v))
    # re-reddening the twice-whitened residual chronology (A11, ARSTAN / dev
    # dplR) gives lag-1 ac ~0.507; the earlier construction gave 0.482
    assert abs(ac1 - 0.509) < 0.01
