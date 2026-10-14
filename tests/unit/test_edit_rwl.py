# Tests for dpl.edit_rwl(): insert / delete / change one ring in a series.
# Expected outputs are derived directly from dplR's insert.ring / delete.ring.

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _series():
    # rw = 10..50, years 2001..2005
    return pd.Series([10.0, 20, 30, 40, 50],
                     index=pd.Index(range(2001, 2006), name="Year"), name="S1")


def _check(out, years, values):
    assert list(out.index) == list(years)
    got = out.to_numpy(dtype=float)
    exp = np.array(values, dtype=float)
    assert np.array_equal(np.isnan(got), np.isnan(exp))
    m = ~np.isnan(exp)
    assert np.allclose(got[m], exp[m])


# ---- insert (parity with dplR insert.ring) --------------------------------

def test_insert_fix_last_fix_length():
    out = dpl.edit_rwl(_series(), year=2003, action="insert", value=99)
    _check(out, range(2001, 2006), [20, 30, 99, 40, 50])      # oldest drops off


def test_insert_not_fix_last_fix_length():
    out = dpl.edit_rwl(_series(), year=2003, action="insert", value=99, fix_last=False)
    _check(out, range(2001, 2006), [10, 20, 30, 99, 40])      # youngest drops off


def test_insert_fix_last_not_fix_length():
    out = dpl.edit_rwl(_series(), year=2003, action="insert", value=99, fix_length=False)
    _check(out, range(2000, 2006), [10, 20, 30, 99, 40, 50])  # extends inner end


def test_insert_not_fix_last_not_fix_length():
    out = dpl.edit_rwl(_series(), year=2003, action="insert", value=99,
                       fix_last=False, fix_length=False)
    _check(out, range(2001, 2007), [10, 20, 30, 99, 40, 50])  # extends recent end


def test_insert_default_value_is_mean():
    out = dpl.edit_rwl(_series(), year=2003, action="insert", fix_length=False)
    assert out.loc[2003] == pytest.approx(30.0)              # mean of 10..50


def test_insert_prepend():
    # prepend a ring at the year before the series; fix_last keeps the recent end
    # dated so the new ring extends the inner end at 2000
    out = dpl.edit_rwl(_series(), year=2000, action="insert", value=99,
                       fix_last=True, fix_length=False)
    _check(out, range(2000, 2006), [99, 10, 20, 30, 40, 50])


# ---- delete (parity with dplR delete.ring) --------------------------------

def test_delete_fix_last_fix_length():
    out = dpl.edit_rwl(_series(), year=2003, action="delete")
    _check(out, range(2001, 2006), [np.nan, 10, 20, 40, 50])  # NA pads inner end


def test_delete_not_fix_last_fix_length():
    out = dpl.edit_rwl(_series(), year=2003, action="delete", fix_last=False)
    _check(out, range(2001, 2006), [10, 20, 40, 50, np.nan])  # NA pads recent end


def test_delete_fix_last_not_fix_length():
    out = dpl.edit_rwl(_series(), year=2003, action="delete", fix_length=False)
    _check(out, range(2002, 2006), [10, 20, 40, 50])


def test_delete_not_fix_last_not_fix_length():
    out = dpl.edit_rwl(_series(), year=2003, action="delete",
                       fix_last=False, fix_length=False)
    _check(out, range(2001, 2005), [10, 20, 40, 50])


# ---- change ---------------------------------------------------------------

def test_change_value():
    out = dpl.edit_rwl(_series(), year=2003, action="change", value=99)
    _check(out, range(2001, 2006), [10, 20, 99, 40, 50])


# ---- DataFrame integration -------------------------------------------------

def _frame():
    return pd.DataFrame(
        {"A": [10.0, 20, 30, 40, 50], "B": [1.0, 2, 3, 4, 5]},
        index=pd.Index(range(2001, 2006), name="Year"))


def test_dataframe_insert_extends_index_and_leaves_others():
    out = dpl.edit_rwl(_frame(), series="A", year=2003, action="insert", value=99,
                       fix_length=False)
    assert list(out.index) == list(range(2000, 2006))        # index extended by a year
    assert np.isnan(out.loc[2000, "B"])                      # other series NA there
    assert list(out["B"].dropna()) == [1, 2, 3, 4, 5]        # other series unchanged
    _check(out["A"].dropna(), range(2000, 2006), [10, 20, 30, 99, 40, 50])


def test_dataframe_change_is_local():
    out = dpl.edit_rwl(_frame(), series="A", year=2003, action="change", value=0.0)
    assert out.loc[2003, "A"] == 0.0
    assert list(out["B"]) == [1, 2, 3, 4, 5]
    assert list(out.index) == list(range(2001, 2006))


def test_returns_copy_not_in_place():
    df = _frame()
    dpl.edit_rwl(df, series="A", year=2003, action="change", value=0.0)
    assert df.loc[2003, "A"] == 30.0                         # original untouched


# ---- validation ------------------------------------------------------------

def test_bad_action():
    with pytest.raises(ValueError):
        dpl.edit_rwl(_series(), year=2003, action="frobnicate")


def test_change_requires_value():
    with pytest.raises(ValueError):
        dpl.edit_rwl(_series(), year=2003, action="change")


def test_year_outside_span():
    with pytest.raises(ValueError):
        dpl.edit_rwl(_series(), year=1800, action="delete")


def test_insert_year_too_far():
    with pytest.raises(ValueError):
        dpl.edit_rwl(_series(), year=1990, action="insert", value=1.0)


def test_interior_gap_rejected():
    s = _series().copy()
    s.loc[2003] = np.nan
    with pytest.raises(ValueError):
        dpl.edit_rwl(s, year=2004, action="change", value=1.0)


def test_dataframe_requires_series():
    with pytest.raises(ValueError):
        dpl.edit_rwl(_frame(), year=2003, action="change", value=1.0)


def test_series_not_found():
    with pytest.raises(ValueError):
        dpl.edit_rwl(_frame(), series="Z", year=2003, action="change", value=1.0)


def test_bad_type():
    with pytest.raises(TypeError):
        dpl.edit_rwl("not data", year=2003, action="change", value=1.0)


def test_negative_insert_value():
    with pytest.raises(ValueError):
        dpl.edit_rwl(_series(), year=2003, action="insert", value=-1.0)
