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
        # ARSTAN-style effective replication: cores present, not counting a
        # core whose within-tree correlations all fail the overlap requirement
        present = [cols[k] for k in range(len(cols)) if df[cols[k]].notna().any()]
        has_wt = set(np.array(cols)[iu[0]][wt]) | set(np.array(cols)[iu[1]][wt])
        cpt = Counter(ids[c] for c in present if c in has_wt)
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
    # effective cores/tree = {2,1,1} -> 1/mean(1/nc); the AAA001 pair overlaps
    # all 60 years (>= min_corr_overlap), so both of its cores count
    assert np.isclose(row["c_eff"], 1.2)
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


def test_rwi_stats_c_eff_ignores_core_whose_within_tree_pairs_fail_overlap():
    # A8: ARSTAN-style effective replication. Tree AAA001's two cores overlap for
    # the whole record (valid within-tree pair -> 2 cores). Tree CCC001's two
    # cores overlap for only 10 years (< min_corr_overlap=30): neither core has a
    # valid within-tree correlation, so the tree counts as ONE core (as dplR,
    # whose n_wt-based count is 1 when no pair is valid), not two.
    rng = np.random.RandomState(3)
    n = 80
    base = np.cumsum(rng.randn(n)) * 0.1 + 5
    cols = {
        "AAA001A": base + rng.randn(n) * 0.3,
        "AAA001B": base + rng.randn(n) * 0.3,
        "BBB001A": base + rng.randn(n) * 0.3,
        "CCC001A": base + rng.randn(n) * 0.3,
        "CCC001B": base + rng.randn(n) * 0.3,
    }
    df = pd.DataFrame(cols, index=pd.Index(np.arange(1900, 1900 + n), name="Year"))
    df.loc[1945:, "CCC001A"] = np.nan                  # 1900-1944
    df.loc[:1934, "CCC001B"] = np.nan                  # 1935-1979 -> 10-yr overlap
    ids = {"AAA001A": "AAA001", "AAA001B": "AAA001", "BBB001A": "BBB001",
           "CCC001A": "CCC001", "CCC001B": "CCC001"}
    res = dpl.rwi_stats(df, ids=ids, corr="pearson", round_decimals=None)
    row = res.iloc[0]
    assert row["n_cores"] == 5 and row["n_trees"] == 3
    assert row["n_wt"] == 1                             # only the AAA001 pair is valid
    # effective cores/tree {2, 1, 1} -> c_eff = 1 / mean(1/2, 1, 1) = 1.2
    # (counting CCC001's two present cores would give {2, 1, 2} -> 1.5)
    assert np.isclose(row["c_eff"], 1.2)
    exp = _independent_stats(df, ids, "pearson")
    assert np.isclose(row["rbar_eff"], exp["rbar_eff"])
    assert np.isclose(row["eps"], exp["eps"])
    # the same rule inside a running window where the CCC001 pair fails
    run = dpl.rwi_stats_running(df, ids=ids, corr="pearson", window_length=40,
                                window_overlap=20, round_decimals=None)
    first = run.iloc[0]                                 # 1900-1939: CCC overlap 5 yrs
    assert first["n_wt"] == 1 and np.isclose(first["c_eff"], 1.2)


def test_rwi_stats_option_strings_case_insensitive_and_defaults():
    import inspect
    df = _synthetic()
    sig = inspect.signature(dpl.rwi_stats_running)
    assert sig.parameters["corr"].default == "spearman"
    assert sig.parameters["window_length"].default == 50
    assert inspect.signature(dpl.rwi_stats).parameters["corr"].default == "spearman"
    a = dpl.rwi_stats(df, corr="pearson", period="max")
    b = dpl.rwi_stats(df, corr="PEARSON", period="Max")
    pd.testing.assert_frame_equal(a, b)
    assert "corr" not in str(a.columns)                 # (sanity: table, not attrs)


def test_rwi_stats_prewhiten_uses_package_prewhitening():
    # D1: prewhiten=True whitens each series with autoreg._prewhiten (YW,
    # ar_max=10, first local AIC min, backcast), not the old OLS ar_func path
    from dplpy.autoreg import _prewhiten
    df = _synthetic(n=120)
    res = dpl.rwi_stats(df, corr="pearson", prewhiten=True, round_decimals=None)
    white = pd.DataFrame({c: _prewhiten(df[c].to_numpy(dtype=float)) for c in df.columns},
                         index=df.index)
    ref = dpl.rwi_stats(white, corr="pearson", round_decimals=None)
    pd.testing.assert_frame_equal(res, ref)
    plain = dpl.rwi_stats(df, corr="pearson", round_decimals=None)
    assert not np.isclose(res.iloc[0]["rbar_tot"], plain.iloc[0]["rbar_tot"])
    # the AR controls are exposed: global-minimum rule changes the result here
    # only if a different order is picked, so just check they are accepted
    dpl.rwi_stats(df, prewhiten=True, ar_max=5, ar_method="Burg", first_aic_min=False)
