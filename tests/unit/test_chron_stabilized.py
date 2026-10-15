import re
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl
from dplpy.rbar import running_rbar_vector, pairwise_corr_mean


def _two_linear_series(n=100):
    """Two perfectly correlated linear series: rbar is exactly 1 in every window."""
    idx = pd.Index(np.arange(1, n + 1), name="Year")
    return pd.DataFrame({"SeriesA": 0.01 + 0.02 * np.arange(n),
                         "SeriesB": 0.02 + 0.02 * np.arange(n)}, index=idx)


def _staggered_rwi(n_years=200, n_series=12, seed=0):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, n_years)
    years = pd.Index(np.arange(1, n_years + 1), name="Year")
    cols = {}
    for s in range(n_series):
        start = s * (n_years // (2 * n_series))
        v = np.full(n_years, np.nan)
        noise = rng.normal(0, 1, n_years)
        v[start:] = 1.0 + 0.15 * (common[start:] + noise[start:])
        cols["s%02d" % s] = v
    return pd.DataFrame(cols, index=years)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def test_chron_stabilized_valid_input():
    # two perfectly correlated series: running rbar == 1 and rbar constant == 1,
    # so Neff == 1 every year and the stabilized chronology is the plain mean
    input_df = _two_linear_series()
    result = dpl.chron_stabilized(input_df, running_rbar=True)
    plain = dpl.chron(input_df, plot=False)
    assert list(result.columns) == ["vsc", "running_rbar", "samp_depth"]
    assert list(result.index) == list(input_df.index)
    assert np.allclose(result["running_rbar"].to_numpy(), 1.0)
    assert np.allclose(result["vsc"].to_numpy(), plain["std"].to_numpy())
    assert (result["samp_depth"] == 2).all()
    assert result.attrs["kind"] == "crn"


def test_chron_stabilized_wrong_input_type():
    with pytest.raises(TypeError) as errorMsg:
        dpl.chron_stabilized("data")
    assert "Expected dataframe input, got <class 'str'> instead." == str(errorMsg.value)


def test_chron_stabilized_invalid_winlen():
    input_df = _two_linear_series()
    with pytest.raises(ValueError) as errorMsg:
        dpl.chron_stabilized(input_df, window_length=101)
    expected_msg = "Window length should not be greater than the number of rows in the dataset"
    assert expected_msg == str(errorMsg.value)
    # win_length= is a deprecated alias of window_length=
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError):
            dpl.chron_stabilized(input_df, win_length=101)
    with pytest.raises(TypeError):
        dpl.chron_stabilized(input_df, bogus=1)


def test_chron_stabilized_invalid_min_seg_ratio():
    with pytest.raises(ValueError) as errorMsg:
        dpl.chron_stabilized(_two_linear_series(), min_seg_ratio=1.1)
    assert "min_seg_ratio cannot be <= 0 or > 1" == str(errorMsg.value)


def test_chron_stabilized_weird_winlen():
    expected_msg = "Window length greater than 50% of the chronology length is not recommended. Consider a shorter window.\n"
    with pytest.warns(UserWarning, match=re.escape(expected_msg)):
        dpl.chron_stabilized(_two_linear_series(), window_length=100)


def test_chron_stabilized_short_winlen():
    # window_length=25 is <=30 (triggers the absolute-floor warning) but is only
    # 25% of the 100-row chronology (well under the 50% relative-ceiling check),
    # so this should be the ONLY warning raised -- verifying the two checks
    # fire independently, matching dplR's chron.stabilized().
    expected_msg = "Window length less than 30 is not recommended. Consider a longer window.\n"
    with pytest.warns(UserWarning, match=re.escape(expected_msg)) as warning_record:
        dpl.chron_stabilized(_two_linear_series(), window_length=25)
    assert len(warning_record) == 1


def test_chron_stabilized_running_uses_shared_running_rbar_vector():
    # E14: the running rbar is rbar.running_rbar_vector on the zero-mean matrix
    # (one implementation), and the dplR formula is applied around it
    rwi = _staggered_rwi()
    out = _quiet(dpl.chron_stabilized, rwi, running_rbar=True)
    mean_val = rwi.mean(axis=1).mean()
    zm = rwi - mean_val
    n = zm.notna().sum(axis=1).to_numpy()
    rbar = running_rbar_vector(zm, 50, 1 / 3, n)
    assert np.allclose(out["running_rbar"].to_numpy(), rbar, equal_nan=True)
    n_eff = np.minimum(n / (1 + (n - 1) * rbar), n)
    std = dpl.chron(zm, plot=False)["std"].to_numpy()
    expected = std * np.sqrt(n_eff * pairwise_corr_mean(zm, "pearson")) + mean_val
    assert np.allclose(out["vsc"].to_numpy(), expected)


def test_chron_stabilized_interior_nan_rbar_falls_back_to_depth():
    # A13: a window where no pair meets the overlap requirement gives a NaN
    # running rbar; dplR then uses Neff = n (not NaN), so vsc stays finite
    rng = np.random.default_rng(1)
    n = 160
    idx = pd.Index(np.arange(1, n + 1), name="Year")
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    a[:70] = 1 + 0.1 * rng.standard_normal(70)        # A: years 1-70
    b[90:] = 1 + 0.1 * rng.standard_normal(70)        # B: years 91-160, no overlap
    c = 1 + 0.1 * rng.standard_normal(n)              # C: all years
    df = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)
    out = _quiet(dpl.chron_stabilized, df, window_length=40, running_rbar=True)
    rb = out["running_rbar"].to_numpy()
    assert np.isnan(rb).any()                          # some windows have no valid pair
    depth = out["samp_depth"].to_numpy()
    assert np.isfinite(out["vsc"].to_numpy()[depth > 0]).all()
    # where rbar is NaN, the scaling used sqrt(n * rbar_const): rebuild it
    mean_val = df.mean(axis=1).mean()
    zm = df - mean_val
    std = dpl.chron(zm, plot=False)["std"].to_numpy()
    rc = pairwise_corr_mean(zm, "pearson")
    hole = np.isnan(rb) & (depth > 0)
    assert np.allclose(out["vsc"].to_numpy()[hole],
                       std[hole] * np.sqrt(depth[hole] * rc) + mean_val)


def test_chron_stabilized_zero_depth_year_kept():
    # A14: a year with no series present is kept (samp_depth 0, vsc NaN)
    df = _two_linear_series(60)
    df.loc[30, :] = np.nan
    out = _quiet(dpl.chron_stabilized, df, window_length=20)
    assert len(out) == 60
    assert out.loc[30, "samp_depth"] == 0 and np.isnan(out.loc[30, "vsc"])
    assert np.isfinite(out["vsc"].drop(30)).all()


def test_chron_stabilized_method_case_insensitive_and_alias():
    rwi = _staggered_rwi()
    a = _quiet(dpl.chron_stabilized, rwi, method="briffa")
    b = _quiet(dpl.chron_stabilized, rwi, method="Briffa")
    c = _quiet(dpl.chron_stabilized, rwi, method="MEAN_RBAR")
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    with pytest.raises(ValueError, match="method"):
        dpl.chron_stabilized(rwi, method="bogus")


def test_chron_stabilized_spline_period_default_and_alias():
    # D3: spline stiffness is spline_period; None = the 67% spline (floor(0.67 n)),
    # the same default as stabilize_chron (earlier releases used 0.5 n);
    # spline_nyrs= is a deprecated alias
    rwi = _staggered_rwi()
    n = len(rwi)
    default = _quiet(dpl.chron_stabilized, rwi, method="spline")
    explicit = _quiet(dpl.chron_stabilized, rwi, method="spline",
                      spline_period=int(np.floor(0.67 * n)))
    half = _quiet(dpl.chron_stabilized, rwi, method="spline", spline_period=0.5)
    pd.testing.assert_frame_equal(default, explicit)
    assert not np.allclose(default["vsc"], half["vsc"])
    with pytest.warns(DeprecationWarning):
        alias = dpl.chron_stabilized(rwi, method="spline", spline_nyrs=0.5)
    pd.testing.assert_frame_equal(alias, half)
    assert (default["vsc"].dropna() >= 0).all()           # clipped at 0


def test_chron_stabilized_spline_uses_shared_spline_component():
    # E15: the spline path is stabilize._spline_component on the chronology
    from dplpy.stabilize import _spline_component
    rwi = _staggered_rwi()
    out = _quiet(dpl.chron_stabilized, rwi, method="spline", spline_period=60)
    std = dpl.chron(rwi, plot=False)["std"].to_numpy()
    n = rwi.notna().sum(axis=1).to_numpy()
    ref = _spline_component(std, 60, 0.5, clip_negative=True, ok=n > 0)
    assert np.allclose(out["vsc"].to_numpy(), ref, equal_nan=True)


def test_rbar_uses_flat_mean_not_column_means():
    # dplR's chron.stabilized averages the windowed correlation matrix with a FLAT
    # mean -- mean(corMat, na.rm=TRUE), every retained pair weighted equally. A
    # mean-of-column-means drifts from dplR (~1e-4 on some windows) once the overlap
    # mask drops pairs unevenly across series. Build such a case (C and D never
    # overlap, so the mask is uneven) and confirm pairwise_corr_mean returns the
    # flat mean, not the column-means mean.
    rng = np.random.default_rng(0)
    n = 40
    base = rng.standard_normal(n)
    df = pd.DataFrame({
        "A": base + 0.10 * rng.standard_normal(n),
        "B": base + 0.50 * rng.standard_normal(n),
        "C": base + 0.90 * rng.standard_normal(n),
        "D": -base + 0.30 * rng.standard_normal(n),
    })
    df.loc[:19, "C"] = np.nan      # C present years 20..39
    df.loc[20:, "D"] = np.nan      # D present years 0..19 -> C and D never overlap
    corr = df.corr()
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, np.nan)
    pres = df.notnull().astype(int)
    overlap = (pres.T @ pres).to_numpy()
    masked = np.where(overlap >= 10, arr, np.nan)
    flat = np.nanmean(masked)
    col_means = np.nanmean(np.nanmean(masked, axis=0))

    impl = pairwise_corr_mean(df, "pearson", min_overlap=10, strict=False)
    assert abs(impl - flat) < 1e-12            # matches dplR's flat mean
    assert abs(flat - col_means) > 1e-3        # and the case is meaningful (they differ)


def test_running_rbar_vector_matches_per_window_corr():
    # F5: the sliding-sum running rbar equals the per-window DataFrame.corr route
    # (to 1e-10), including constant runs, gaps and a zero-depth block
    rng = np.random.default_rng(0)
    n, s = 300, 7
    X = 1 + 0.2 * rng.standard_normal((n, s)) + 0.3 * rng.standard_normal((n, 1))
    for j in range(s):
        X[:j * 20, j] = np.nan
    X[150:165, :] = np.nan
    X[200:250, 3] = 1.0
    X[rng.random((n, s)) < 0.05] = np.nan
    df = pd.DataFrame(X)
    for w, ratio in ((50, 1 / 3), (41, 0.5), (30, 0.9)):
        ref = np.full(n, np.nan)
        target = w // 2 if w % 2 == 0 else (w - 1) // 2
        for i in range(n - w + 1):
            seg = df.iloc[i:i + w]
            ref[i + target] = pairwise_corr_mean(seg, "pearson", min_overlap=w * ratio)
        valid = np.flatnonzero(~np.isnan(ref))
        ref[:valid[0]] = ref[valid[0]]
        ref[valid[-1]:] = ref[valid[-1]]
        ref[df.notna().sum(axis=1).to_numpy() == 0] = np.nan
        got = running_rbar_vector(df, w, ratio)
        assert np.array_equal(np.isnan(got), np.isnan(ref))
        ok = ~np.isnan(ref)
        assert np.max(np.abs(got[ok] - ref[ok])) < 1e-10
