import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


# Two trees, two cores each. T2a is missing 1998; T1a is missing 2000.
def _rwl():
    return pd.DataFrame(
        {"T1a": [1.0, 2.0, np.nan],
         "T1b": [3.0, 4.0, 6.0],
         "T2a": [np.nan, 20.0, 30.0],
         "T2b": [10.0, 40.0, 60.0]},
        index=pd.Index([1998, 1999, 2000], name="Year"),
    )


def _ids():
    return pd.DataFrame(
        {"tree": ["A", "A", "B", "B"], "core": [1, 2, 1, 2]},
        index=pd.Index(["T1a", "T1b", "T2a", "T2b"], name="series"),
    )


def test_tree_mean_default_na_rm_false():
    # na_rm=False: a year is NaN unless every core of the tree is present
    out = dpl.tree_mean(_rwl(), _ids())
    assert list(out.columns) == ["A", "B"]
    assert list(out.index) == [1998, 1999, 2000]
    # tree A: 1998 mean(1,3)=2 ; 1999 mean(2,4)=3 ; 2000 mean(nan,6)=nan
    assert out["A"].tolist()[:2] == [2.0, 3.0]
    assert np.isnan(out["A"].iloc[2])
    # tree B: 1998 mean(nan,10)=nan ; 1999 mean(20,40)=30 ; 2000 mean(30,60)=45
    assert np.isnan(out["B"].iloc[0])
    assert out["B"].tolist()[1:] == [30.0, 45.0]


def test_tree_mean_na_rm_true():
    # na_rm=True: average whatever cores are present
    out = dpl.tree_mean(_rwl(), _ids(), na_rm=True)
    assert out["A"].tolist() == [2.0, 3.0, 6.0]        # 2000 -> just T1b
    assert out["B"].tolist() == [10.0, 30.0, 45.0]     # 1998 -> just T2b


def test_matches_by_name_not_position():
    # ids rows shuffled: alignment is by series name, so the result is unchanged
    ids = _ids().iloc[[2, 0, 3, 1]]
    out = dpl.tree_mean(_rwl(), ids, na_rm=True)
    assert out["A"].tolist() == [2.0, 3.0, 6.0]
    assert out["B"].tolist() == [10.0, 30.0, 45.0]


def test_unique_tree_order_is_first_appearance():
    # columns ordered so tree B's first core appears before A's
    rwl = _rwl()[["T2a", "T1a", "T2b", "T1b"]]
    out = dpl.tree_mean(rwl, _ids(), na_rm=True)
    assert list(out.columns) == ["B", "A"]


def test_ids_as_dict():
    d = {"T1a": "A", "T1b": "A", "T2a": "B", "T2b": "B"}
    out = dpl.tree_mean(_rwl(), d, na_rm=True)
    assert out["A"].tolist() == [2.0, 3.0, 6.0]


def test_single_core_tree_is_unchanged():
    # a tree with one core just carries that core through (NaNs preserved)
    rwl = _rwl()[["T1a", "T2b"]]
    ids = {"T1a": "A", "T2b": "B"}
    out = dpl.tree_mean(rwl, ids)
    assert np.isnan(out["A"].iloc[2]) and out["A"].tolist()[:2] == [1.0, 2.0]
    assert out["B"].tolist() == [10.0, 40.0, 60.0]


def test_missing_tree_column_raises():
    bad = pd.DataFrame({"core": [1, 2, 1, 2]},
                       index=["T1a", "T1b", "T2a", "T2b"])
    with pytest.raises(ValueError):
        dpl.tree_mean(_rwl(), bad)


def test_missing_tree_id_raises():
    ids = _ids().copy()
    ids.loc["T2a", "tree"] = np.nan
    with pytest.raises(ValueError, match="missing tree"):
        dpl.tree_mean(_rwl(), ids)


def test_non_dataframe_rwl_raises():
    with pytest.raises(TypeError):
        dpl.tree_mean([[1, 2], [3, 4]], _ids())


# --- integration: the intended workflow on real data --------------------------
def test_feeds_chron_on_ca533():
    import warnings
    data = dpl.readers("./tests/data/csv/ca533.csv")
    # stc=(3,2,1) groups the two cores of a tree (e.g. CAM031, CAM032 -> CAM03)
    ids = dpl.read_ids(data, stc=(3, 2, 1))
    trees = dpl.tree_mean(data, ids, na_rm=True)
    # one column per unique tree, fewer than the 34 cores
    assert trees.shape[1] == ids["tree"].nunique() == 21
    assert trees.shape[1] < data.shape[1]
    assert list(trees.index) == list(data.index)
    # CAM03 is the per-year nanmean of its two cores
    expect = data[["CAM031", "CAM032"]].mean(axis=1, skipna=True)
    assert np.allclose(trees["CAM03"].to_numpy(), expect.to_numpy(),
                       rtol=1e-12, atol=1e-12, equal_nan=True)
    # a tree-level chronology builds without error and spans the same years
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crn = dpl.chron(trees, biweight=False, prewhiten=False, plot=False)
    assert list(crn.columns) == ["std", "samp_depth"]
    assert int(crn["samp_depth"].max()) <= trees.shape[1]
