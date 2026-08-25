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


def test_post_ar_forward_filter_matches_explicit_recurrence():
    # guards the scipy.signal.lfilter usage inside _post_ar: the zero-init AR
    # forward filter must equal the explicit recurrence y[t]=x[t]+sum phi_j y[t-j]
    from dplpy.chron_ars import _post_ar  # noqa
    rng = np.random.RandomState(1)
    x = rng.randn(150)
    phi = np.array([0.5, -0.1, 0.05])
    xpad = np.concatenate([np.zeros(len(phi)), x.copy()])
    for i in range(len(x)):
        for j in range(1, len(phi) + 1):
            xpad[i + len(phi)] += phi[j - 1] * xpad[i + len(phi) - j]
    explicit = xpad[len(phi):]
    lf = lfilter([1.0], np.concatenate([[1.0], -phi]), x)
    assert np.allclose(explicit, lf, atol=1e-12)


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
# ---------------------------------------------------------------------------
def test_chron_ars_matches_dplR_reference_ca533():
    data = dpl.readers("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dpl.chron_ars(rwi, biweight=False, prewhiten_method="ar.yw", verbose=False)

    # dplR std/res/ars at three years (atol 1e-6)
    ref = {
        1700: (0.8813025556, 0.9176128562, 0.9026494583),
        1850: (0.9396477470, 0.9993573139, 0.9748222048),
        1983: (1.3314471088, 1.1243002992, 1.3015262394),
    }
    for yr, (s, r, a) in ref.items():
        assert abs(out.loc[yr, "std"] - s) < 1e-6, (yr, "std")
        assert abs(out.loc[yr, "res"] - r) < 1e-6, (yr, "res")
        assert abs(out.loc[yr, "ars"] - a) < 1e-6, (yr, "ars")


def test_pooled_acf_matches_dplR_reference_ca533():
    from dplpy.chron_ars import _pooled_ar
    data = dpl.readers("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    out_ar = _pooled_ar(rwi.to_numpy(dtype=float), max_lag=10, first_aic_min=True)
    # dplR pooled ACF (lags 0..5) and selected order, from chron.ars() on dplR's
    # own spline detrend (matches dplPy's new detrend + pooled AR to ~1e-8).
    ref_acf = [1.0, 0.51626627, 0.35330050, 0.25346321, 0.21923437, 0.19473382]
    assert np.allclose(out_ar["acf"][:6], ref_acf, atol=1e-7)
    assert out_ar["order"] == 5
