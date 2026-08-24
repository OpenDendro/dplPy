import numpy as np
import pandas as pd
import dplpy as dpl
import pytest


def _synthetic(seed=1, n=60):
    rng = np.random.RandomState(seed)
    base = np.cumsum(rng.randn(n)) * 0.1 + 5
    return pd.DataFrame(
        {
            "AAA001A": base + rng.randn(n) * 0.3,
            "AAA001B": base + rng.randn(n) * 0.3,
            "AAA002A": base + rng.randn(n) * 0.3,
            "BBB001A": base * 0.5 + rng.randn(n) * 0.5 + 2,
        },
        index=pd.Index(np.arange(1900, 1900 + n), name="Year"),
    )


def test_sss_wrong_data_type():
    with pytest.raises(TypeError) as e:
        dpl.sss("notadf")
    assert "Expected dataframe input" in str(e.value)


def test_sss_returns_year_indexed_series():
    df = _synthetic()
    s = dpl.sss(df, corr="Pearson")
    assert isinstance(s, pd.Series)
    assert s.name == "sss"
    assert list(s.index) == list(df.index)


def test_sss_matches_wigley_formula_no_ids():
    df = _synthetic()
    s = dpl.sss(df, corr="Pearson")
    # independent: fixed rbar_eff and N from whole-record rwi_stats, per-year cores
    st = dpl.rwi_stats(df, corr="Pearson", round_decimals=None)
    rbar = st.iloc[0]["rbar_eff"]
    big_n = st.iloc[0]["n_trees"]
    n_t = df.notna().sum(axis=1).to_numpy(dtype=float)
    expected = n_t * (1 + (big_n - 1) * rbar) / (big_n * (1 + (n_t - 1) * rbar))
    assert np.allclose(s.to_numpy(), expected)


def test_sss_matches_wigley_formula_with_ids():
    df = _synthetic()
    ids = {"AAA001A": "AAA001", "AAA001B": "AAA001",
           "AAA002A": "AAA002", "BBB001A": "BBB001"}
    s = dpl.sss(df, ids=ids, corr="Pearson")
    st = dpl.rwi_stats(df, ids=ids, corr="Pearson", round_decimals=None)
    rbar = st.iloc[0]["rbar_eff"]
    big_n = st.iloc[0]["n_trees"]           # trees, not cores
    assert big_n == 3
    # per-year TREE depth (all 4 cores span all years here -> 3 trees every year)
    n_t = np.full(len(df), 3.0)
    expected = n_t * (1 + (big_n - 1) * rbar) / (big_n * (1 + (n_t - 1) * rbar))
    assert np.allclose(s.to_numpy(), expected)
    # full depth every year -> SSS == 1 throughout
    assert np.allclose(s.to_numpy(), 1.0)


def test_sss_rises_with_sample_depth():
    # a staggered dataset: SSS should be lower early (few series) than late
    df = _synthetic(n=80)
    # knock out the first 40 years of two series to reduce early depth
    df.iloc[:40, [1, 2]] = np.nan
    s = dpl.sss(df, corr="Pearson")
    assert s.iloc[0] < s.iloc[-1]
