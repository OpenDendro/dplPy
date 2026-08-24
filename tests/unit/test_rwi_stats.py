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


def _independent_stats(df, ids, method="pearson", min_overlap=30):
    """Recompute rbar/eps/snr the long way, outside dplPy, for cross-checking."""
    cols = list(df.columns)
    trees = np.array([ids[c] for c in cols], dtype=object)
    cm = df.corr(method, min_periods=min_overlap).to_numpy(copy=True)
    np.fill_diagonal(cm, np.nan)
    iu = np.triu_indices(len(cols), 1)
    pr = cm[iu]
    valid = ~np.isnan(pr)
    same = trees[iu[0]] == trees[iu[1]]
    bt = valid & ~same
    wt = valid & same
    rbar_bt = pr[bt].mean() if bt.any() else np.nan
    rbar_wt = pr[wt].mean() if wt.any() else np.nan
    rbar_tot = pr[valid].mean() if valid.any() else np.nan
    good = set(trees[iu[0]][bt]) | set(trees[iu[1]][bt])
    n_good = len(good)
    if wt.sum() == 0:
        rbar_eff = rbar_bt
    else:
        from collections import Counter
        present = [cols[k] for k in range(len(cols)) if df[cols[k]].notna().any()]
        cpt = Counter(ids[c] for c in present)
        nc = np.array([max(cpt[t], 1) for t in good], dtype=float)
        rproc = np.mean(1.0 / nc)
        rbar_eff = rbar_bt / (rbar_wt + (1 - rbar_wt) * rproc)
    eps = n_good * rbar_eff / ((n_good - 1) * rbar_eff + 1)
    snr = n_good * rbar_eff / (1 - rbar_eff)
    return dict(rbar_tot=rbar_tot, rbar_wt=rbar_wt, rbar_bt=rbar_bt,
                rbar_eff=rbar_eff, eps=eps, snr=snr, n=n_good)


def test_rwi_stats_wrong_data_type():
    with pytest.raises(TypeError) as e:
        dpl.rwi_stats("notadf")
    assert "Expected dataframe input" in str(e.value)


def test_rwi_stats_wrong_corr():
    with pytest.raises(ValueError) as e:
        dpl.rwi_stats(_synthetic(), corr="Kendall")
    assert "Spearman" in str(e.value)


def test_rwi_stats_wrong_period():
    with pytest.raises(ValueError) as e:
        dpl.rwi_stats(_synthetic(), period="foo")
    assert "period" in str(e.value)


def test_rwi_stats_needs_two_trees():
    df = _synthetic()[["AAA001A"]]
    with pytest.raises(ValueError) as e:
        dpl.rwi_stats(df)
    assert "2 trees" in str(e.value)


def test_rwi_stats_no_ids_matches_independent():
    df = _synthetic()
    # no ids -> every series its own tree; rbar_eff == rbar_bt == rbar_tot
    res = dpl.rwi_stats(df, corr="Pearson", round_decimals=None)
    ids = {c: c for c in df.columns}
    exp = _independent_stats(df, ids, "pearson")
    row = res.iloc[0]
    assert row["n_wt"] == 0
    assert row["n_bt"] == 6
    assert np.isclose(row["rbar_tot"], exp["rbar_tot"])
    assert np.isclose(row["rbar_eff"], exp["rbar_tot"])
    assert np.isclose(row["eps"], exp["eps"])
    assert np.isclose(row["snr"], exp["snr"])
    assert row["c_eff"] == 1.0


def test_rwi_stats_with_ids_within_between_split():
    df = _synthetic()
    ids = {"AAA001A": "AAA001", "AAA001B": "AAA001",
           "AAA002A": "AAA002", "BBB001A": "BBB001"}
    res = dpl.rwi_stats(df, ids=ids, corr="Pearson", round_decimals=None)
    exp = _independent_stats(df, ids, "pearson")
    row = res.iloc[0]
    assert row["n_trees"] == 3
    assert row["n_wt"] == 1     # only AAA001A-AAA001B share a tree
    assert row["n_bt"] == 5
    assert row["n"] == 3
    assert np.isclose(row["rbar_wt"], exp["rbar_wt"])
    assert np.isclose(row["rbar_bt"], exp["rbar_bt"])
    assert np.isclose(row["rbar_tot"], exp["rbar_tot"])
    assert np.isclose(row["c_eff"], 1.2)   # cores/tree = {2,1,1} -> 1/mean(1/nc)
    assert np.isclose(row["rbar_eff"], exp["rbar_eff"])
    assert np.isclose(row["eps"], exp["eps"])
    assert np.isclose(row["snr"], exp["snr"])


def test_rwi_stats_ids_accepts_dataframe():
    df = _synthetic()
    ids_dict = {"AAA001A": "AAA001", "AAA001B": "AAA001",
                "AAA002A": "AAA002", "BBB001A": "BBB001"}
    ids_df = pd.DataFrame({"tree": pd.Series(ids_dict)})
    r1 = dpl.rwi_stats(df, ids=ids_dict, corr="Pearson", round_decimals=None)
    r2 = dpl.rwi_stats(df, ids=ids_df, corr="Pearson", round_decimals=None)
    pd.testing.assert_frame_equal(r1, r2)


def test_rwi_stats_ids_missing_series_raises():
    df = _synthetic()
    with pytest.raises(ValueError) as e:
        dpl.rwi_stats(df, ids={"AAA001A": "AAA001"})  # incomplete
    assert "missing tree assignments" in str(e.value)


def test_rwi_stats_running_structure():
    df = _synthetic(n=120)
    run = dpl.rwi_stats_running(df, window_length=40, window_overlap=20)
    # windows start at 0, stride 20, last full window at 80 -> starts 0,20,40,60,80
    assert list(run["start_year"]) == [1900, 1920, 1940, 1960, 1980]
    assert list(run["end_year"]) == [1939, 1959, 1979, 1999, 2019]
    assert list(run["mid_year"]) == [1919, 1939, 1959, 1979, 1999]
    expected_cols = ["start_year", "mid_year", "end_year", "n_cores", "n_trees",
                     "n", "n_tot", "n_wt", "n_bt", "rbar_tot", "rbar_wt",
                     "rbar_bt", "c_eff", "rbar_eff", "eps", "snr"]
    assert list(run.columns) == expected_cols
    # all computed EPS values within (0, 1]
    eps = run["eps"].dropna()
    assert ((eps > 0) & (eps <= 1)).all()


def test_rwi_stats_is_running_false_single_row():
    df = _synthetic()
    a = dpl.rwi_stats(df, corr="Pearson")
    b = dpl.rwi_stats_running(df, corr="Pearson", running_window=False)
    assert a.shape[0] == 1
    pd.testing.assert_frame_equal(a, b)
    # non-running output has no year columns
    assert "start_year" not in a.columns
