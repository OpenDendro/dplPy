import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _df(cols, years):
    return pd.DataFrame(cols, index=pd.Index(years, name="Year"))


def test_two_frame_union_and_columns():
    a = _df({"A": [1.0, 2.0, 3.0]}, [2001, 2002, 2003])
    b = _df({"B": [4.0, 5.0]}, [2002, 2003])
    out = dpl.combine_rwl(a, b)
    assert list(out.columns) == ["A", "B"]
    assert list(out.index) == [2001, 2002, 2003]
    assert np.isnan(out.loc[2001, "B"])                 # B has no 2001
    assert out.loc[2002, "A"] == 2.0 and out.loc[2003, "B"] == 5.0


def test_list_form_matches_pairwise():
    a = _df({"A": [1.0, 1.0]}, [1, 2])
    b = _df({"B": [2.0, 2.0]}, [2, 3])
    c = _df({"C": [3.0]}, [3])
    out = dpl.combine_rwl([a, b, c])
    assert list(out.columns) == ["A", "B", "C"]
    assert list(out.index) == [1, 2, 3]


def test_disjoint_ranges_fill_contiguous():
    a = _df({"A": [1.0, 1.0]}, [1500, 1501])
    b = _df({"B": [2.0, 2.0]}, [1600, 1601])
    out = dpl.combine_rwl(a, b)
    assert list(out.index) == list(range(1500, 1602))    # gap years padded
    assert out["A"].loc[1550:1599].isna().all()
    assert np.isnan(out.loc[1550, "A"]) and np.isnan(out.loc[1550, "B"])


def test_duplicate_ids_kept_with_warning():
    a = _df({"A": [1.0, 1.0]}, [1, 2])
    b = _df({"A": [9.0, 9.0]}, [1, 2])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = dpl.combine_rwl(a, b)
        assert any("duplicate" in str(x.message).lower() for x in w)
    assert list(out.columns) == ["A", "A"]               # both kept, as in dplR
    assert out.shape == (2, 2)


def test_single_frame_reindexed_contiguous():
    a = _df({"A": [1.0, np.nan, 3.0]}, [2000, 2001, 2002])
    out = dpl.combine_rwl([a])
    assert list(out.index) == [2000, 2001, 2002]
    assert list(out.columns) == ["A"]


def test_index_name_preserved():
    a = _df({"A": [1.0]}, [1])
    b = _df({"B": [2.0]}, [1])
    assert dpl.combine_rwl(a, b).index.name == "Year"


def test_errors():
    a = _df({"A": [1.0]}, [1])
    with pytest.raises(TypeError):
        dpl.combine_rwl(a, [a])                           # y must be a DataFrame
    with pytest.raises(TypeError):
        dpl.combine_rwl([a, "not a frame"])
    with pytest.raises(ValueError):
        dpl.combine_rwl([])


def test_exported():
    assert hasattr(dpl, "combine_rwl")
